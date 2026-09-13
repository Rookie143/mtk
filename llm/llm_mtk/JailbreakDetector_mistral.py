import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import GenerationConfig

from fix.IsolationForest import PyTorchIsolationForest


class JailbreakDetector:
    def __init__(self, model, tokenizer, background_layered_activations, all_labels, your_flag,
                 n_estimators, random_state, max_samples, k_nb, target_layers=None, metric="l2"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = model.device
        self.your_flag = your_flag
        self.metric = metric
        self.max_samples = max_samples
        self.k_nb = k_nb
        self.background_activations_by_layer = background_layered_activations
        self.background_labels = all_labels
        self.num_layers = background_layered_activations.shape[1]
        valid_metrics = ["l1", "l2", "linf", "cos"]
        if self.metric not in valid_metrics:
            raise ValueError(f"Unsupported metric: {self.metric}. Please use {valid_metrics}")
        if target_layers is None:
            self.target_layers = list(range(1, self.model.config.num_hidden_layers + 1))
        else:
            self.target_layers = target_layers

        training_path = f"./{self.your_flag}/training_sequences.pt"
        if os.path.exists(training_path):
            training_sequences = torch.load(training_path, map_location=self.device)
        else:
            training_sequences = self._get_training_sequences()

        benign_indices = torch.where(self.background_labels == 0)[0]
        benign_training_sequences = training_sequences[benign_indices]
        self.mean = benign_training_sequences.mean(dim=0, keepdim=True)
        self.std = benign_training_sequences.std(dim=0, keepdim=True) + 1e-8
        normalized_train = (benign_training_sequences - self.mean) / self.std

        self.if_model = PyTorchIsolationForest(
            n_estimators=n_estimators,
            max_samples=max_samples,
            random_state=random_state,
        )
        self.if_model.fit(normalized_train)

    def predict(self, prompt_text: str = None, input_ids: torch.Tensor = None,
                return_score=True, attack_key=None, return_ranks=False):
        del attack_key
        if input_ids is None and prompt_text is not None:
            messages = [{"role": "user", "content": prompt_text}]
            input_ids = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt"
            ).to(self.device)
        elif input_ids is not None:
            input_ids = input_ids.to(self.device)
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)
        else:
            raise ValueError("Either prompt_text or input_ids must be provided!")

        new_activations = self.get_last_token_hidden_states(input_ids)
        ranks = self._calculate_single_rank_k_nb(
            new_activations,
            self.background_activations_by_layer,
            0,
            self.background_labels,
            k=self.k_nb,
            device=self.device,
        )
        scaled_sequence = (ranks - self.mean) / self.std
        anomaly_score = self.if_model.decision_function(scaled_sequence)[0].item()

        if anomaly_score < 0:
            result = ["Jailbreak Prompt", 1]
        else:
            result = ["Benign prompt", 0]
        if return_score:
            result.append(anomaly_score)
        if return_ranks:
            result.append(ranks.cpu().numpy())
        return tuple(result) if len(result) > 1 else result[0]

    def _restructure_activations(self, activations_list):
        if not activations_list:
            return []
        num_layers = len(activations_list[0])
        activations_by_layer = [[] for _ in range(num_layers)]
        for sample_activations in activations_list:
            for layer in range(num_layers):
                activations_by_layer[layer].append(sample_activations[layer])
        return [torch.stack(layer_acts, dim=0) for layer_acts in activations_by_layer]

    def _get_training_sequences(self):
        num_samples = len(self.background_labels)
        all_sequences = torch.empty(
            (num_samples, len(self.target_layers)), device=self.device
        )
        for i in tqdm(range(num_samples), desc="Generating training sequences"):
            mask = torch.ones(num_samples, dtype=torch.bool, device=self.device)
            mask[i] = False
            all_sequences[i] = self._calculate_single_rank_k_nb(
                self.background_activations_by_layer[i],
                self.background_activations_by_layer[mask],
                0,
                self.background_labels[mask],
                k=self.k_nb,
                device=self.device,
            )
        path = f"./{self.your_flag}/training_sequences.pt"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(all_sequences, path)
        return all_sequences

    def _calculate_single_rank_k_nb(self, test_vector, background_vectors,
                                    target_label, background_labels_arr, k, device):
        test_vector = test_vector.unsqueeze(0)
        if self.metric == "l2":
            layer_distances = (background_vectors - test_vector).norm(p=2, dim=-1)
        elif self.metric == "l1":
            layer_distances = (background_vectors - test_vector).abs().sum(dim=-1)
        elif self.metric == "linf":
            layer_distances = (background_vectors - test_vector).abs().max(dim=-1).values
        elif self.metric == "cos":
            layer_distances = 1 - F.cosine_similarity(background_vectors, test_vector, dim=-1)
        else:
            raise ValueError(f"Unknown metric: {self.metric}")

        layer_distances = layer_distances.permute(1, 0)
        sorted_indices = torch.argsort(layer_distances, dim=1)
        num_layers = layer_distances.shape[0]
        expanded_labels = background_labels_arr.view(1, -1).expand(num_layers, -1)
        sorted_labels = torch.gather(expanded_labels, 1, sorted_indices)
        ranks = torch.empty(num_layers, device=device)
        for layer in range(num_layers):
            positions = torch.where(sorted_labels[layer] == target_label)[0] + 1
            ranks[layer] = positions[:k].float().mean()
        return ranks

    def get_output(self, input_ids, max_new_tokens=100):
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long().to(self.device)
        generation_config = GenerationConfig(
            pad_token_id=self.tokenizer.pad_token_id, do_sample=False
        )
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict_in_generate=True,
                generation_config=generation_config,
                max_new_tokens=max_new_tokens,
            ).sequences
        new_tokens = output_ids[0, input_ids.shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    def get_last_token_hidden_states(self, input_ids):
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        input_ids = input_ids.to(self.device)
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )

        valid_indices = torch.nonzero(attention_mask[0], as_tuple=True)[0]
        target_idx = int(valid_indices[-1]) if len(valid_indices) else input_ids.shape[1] - 1
        token_string = self.tokenizer.decode([int(input_ids[0, target_idx])])
        while target_idx > 0 and ("INST" in token_string or "]" in token_string):
            target_idx -= 1
            token_string = self.tokenizer.decode([int(input_ids[0, target_idx])])

        selected_layer_states = [
            outputs.hidden_states[layer_idx][0, target_idx, :].clone()
            for layer_idx in self.target_layers
        ]
        return torch.stack(selected_layer_states, dim=0)
