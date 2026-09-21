from IsolationForest import PyTorchIsolationForest
import os
import torch
from tqdm import tqdm
from transformers import GenerationConfig


class JailbreakDetector:
    def __init__(self, model, tokenizer, background_layered_activations, all_labels, your_flag,
                 n_estimators=100, random_state=42, max_samples=512, k_nb=10):
        self.model = model
        self.tokenizer = tokenizer
        self.device = model.device
        self.your_flag = your_flag
        self.k_nb = k_nb
        self.background_activations_by_layer = background_layered_activations
        self.background_labels = all_labels
        self.num_layers = self.background_activations_by_layer.shape[1]
        if os.path.exists(f"./{self.your_flag}/training_sequences.pt"):
            training_sequences = torch.load(f"./{self.your_flag}/training_sequences.pt", map_location=self.device)
        else:
            training_sequences = self._get_training_sequences()
        y_train = self.background_labels
        benign_indices = torch.where(y_train == 0)[0]
        benign_training_sequences = training_sequences[benign_indices]
        self.mean = benign_training_sequences.mean(dim=0, keepdim=True)
        self.std = benign_training_sequences.std(dim=0, keepdim=True) + 1e-8
        X_train = (benign_training_sequences - self.mean) / self.std
        self.if_model = PyTorchIsolationForest(n_estimators=n_estimators, max_samples=max_samples,
                                               random_state=random_state)
        self.if_model.fit(X_train)

    def predict(self, prompt_text: str = None, input_ids: torch.Tensor = None, return_score=True, attack_key=None,
                return_ranks=False):
        if input_ids is None and prompt_text is not None:
            messages = [{"role": "user", "content": prompt_text}]
            input_ids = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=False
            ).to(self.device)
        elif input_ids is not None:
            input_ids = input_ids.to(self.device)
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)
            if prompt_text is None:
                prompt_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        else:
            raise ValueError("Either prompt_text or input_ids must be provided!")

        new_activations = self.get_last_token_hidden_states(input_ids)

        ranks = self._calculate_single_rank_k_nb(
            new_activations,
            self.background_activations_by_layer,
            0,
            self.background_labels,
            k=self.k_nb,
            device=self.device
        )

        scaled_sequence = (ranks - self.mean) / self.std

        anomaly_score = self.if_model.decision_function(scaled_sequence)[0].item()
        if anomaly_score < 0:
            label_str = "Jailbreak Prompt"
            pred_label = 1
        else:
            label_str = "Benign prompt"
            pred_label = 0

        result = [label_str, pred_label]

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
            for i in range(num_layers):
                activations_by_layer[i].append(sample_activations[i])

        return [torch.stack(layer_acts, dim=0) for layer_acts in activations_by_layer]

    def _get_training_sequences(self):
        num_samples = len(self.background_labels)
        num_layers = self.num_layers

        device = self.device

        all_sequences = torch.empty((num_samples, num_layers), device=device)

        background_activations_gpu = self.background_activations_by_layer
        background_labels_gpu = self.background_labels

        for i in tqdm(range(num_samples), desc="Generating training sequences"):
            current_vector = background_activations_gpu[i]
            mask = torch.ones(num_samples, dtype=torch.bool, device=device)
            mask[i] = False
            other_vectors = background_activations_gpu[mask]
            other_labels = background_labels_gpu[mask]

            ranks = self._calculate_single_rank_k_nb(
                current_vector,
                other_vectors,
                0,
                other_labels,
                k=self.k_nb,
                device=device
            )

            all_sequences[i] = ranks
        save_dir = os.path.dirname(f"./{self.your_flag}/training_sequences.pt")
        os.makedirs(save_dir, exist_ok=True)
        torch.save(all_sequences, f"./{self.your_flag}/training_sequences.pt")
        return all_sequences

    def _calculate_single_rank_k_nb(self, test_vector, background_vectors, target_label, background_labels_arr, k,
                                    device):
        test_vector = test_vector.unsqueeze(0)

        layer_distances = (
            (background_vectors - test_vector)
            .norm(p=2, dim=2)
            .permute(1, 0)
        )
        sorted_indices = torch.argsort(layer_distances, dim=1)
        sorted_background_labels = background_labels_arr[sorted_indices]
        match_indices_in_sorted_tensor = torch.empty((self.num_layers), device=device)
        for i, s in enumerate(sorted_background_labels):
            match_indices_in_sorted_tensor[i] = (torch.where(s == target_label)[0] + 1)[:k].float().mean()
        return match_indices_in_sorted_tensor

    def get_output(self, input_ids, max_new_tokens=100):
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long().to(self.device)
        generation_config = GenerationConfig(
            pad_token_id=self.tokenizer.pad_token_id,
            do_sample=False
        )
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict_in_generate=True,
                generation_config=generation_config,
                max_new_tokens=max_new_tokens
            ).sequences
        new_tokens = output_ids[0, input_ids.shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    def get_last_token_hidden_states(self, input_ids):
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        input_ids = input_ids.to(self.device)
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long().to(self.device)

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True
            )
        hidden_states = outputs.hidden_states[1:]
        last_token_states = torch.stack([h[:, -1, :] for h in hidden_states], dim=1).squeeze(0)

        return last_token_states