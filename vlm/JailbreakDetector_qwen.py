from pathlib import Path
import torch
from tqdm import tqdm

from IsolationForest import PyTorchIsolationForest

class JailbreakDetector:

    def __init__(
        self,
        model,
        processor,
        background_layered_activations,
        all_labels,
        flag,
        n_estimators=100,
        random_state=42,
        k_nb=5,
        max_samples=512,
        output_dir=None,
    ):
        self.model = model
        self.processor = processor
        self.device = self._infer_device()
        self.flag = flag
        self.k_nb = k_nb
        self.background_activations_by_layer = background_layered_activations.to(self.device)
        self.background_labels = all_labels.to(self.device)
        self.num_layers = self.background_activations_by_layer.shape[1]
        self.output_dir = Path(output_dir) if output_dir else Path("experimental_results") / flag
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.training_sequences_path = self.output_dir / "training_sequences.pt"
        if self.training_sequences_path.exists():
            training_sequences = torch.load(self.training_sequences_path, map_location=self.device)
        else:
            training_sequences = self._get_training_sequences()
        y_train = self.background_labels
        benign_indices = torch.where(y_train == 0)[0]
        benign_training_sequences = training_sequences[benign_indices]

        self.mean = benign_training_sequences.mean(dim=0, keepdim=True)
        self.std = benign_training_sequences.std(dim=0, keepdim=True) + 1e-8
        X_train = (benign_training_sequences - self.mean) / self.std
        self.if_model = PyTorchIsolationForest(
            n_estimators=n_estimators,
            max_samples=max_samples,
            random_state=random_state,
        )
        self.if_model.fit(X_train)

    def _infer_device(self):
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def make_context(
        self,
        tokenizer,
        query,
        history = None,
        system = "",
        max_window_size = 6144,
        chat_format = "chatml",
    ):
        if history is None:
            history = []

        if chat_format == "chatml":
            im_start, im_end = "<|im_start|>", "<|im_end|>"
            im_start_tokens = [tokenizer.im_start_id]
            im_end_tokens = [tokenizer.im_end_id]
            nl_tokens = tokenizer.encode("\n")

            def _tokenize_str(role, content):
                return f"{role}\n{content}", tokenizer.encode(
                    role, allowed_special=set(tokenizer.IMAGE_ST)
                ) + nl_tokens + tokenizer.encode(content, allowed_special=set(tokenizer.IMAGE_ST))

            system_text, system_tokens_part = _tokenize_str("system", system)
            system_tokens = im_start_tokens + system_tokens_part + im_end_tokens

            raw_text = ""
            context_tokens = []

            for turn_query, turn_response in reversed(history):
                query_text, query_tokens_part = _tokenize_str("user", turn_query)
                query_tokens = im_start_tokens + query_tokens_part + im_end_tokens
                if turn_response is not None:
                    response_text, response_tokens_part = _tokenize_str(
                        "assistant", turn_response
                    )
                    response_tokens = im_start_tokens + response_tokens_part + im_end_tokens

                    next_context_tokens = nl_tokens + query_tokens + nl_tokens + response_tokens
                    prev_chat = (
                        f"\n{im_start}{query_text}{im_end}\n{im_start}{response_text}{im_end}"
                    )
                else:
                    next_context_tokens = nl_tokens + query_tokens + nl_tokens
                    prev_chat = f"\n{im_start}{query_text}{im_end}\n"

                current_context_size = (
                    len(system_tokens) + len(next_context_tokens) + len(context_tokens)
                )
                if current_context_size < max_window_size:
                    context_tokens = next_context_tokens + context_tokens
                    raw_text = prev_chat + raw_text
                else:
                    break

            context_tokens = system_tokens + context_tokens
            raw_text = f"{im_start}{system_text}{im_end}" + raw_text
            context_tokens += (
                nl_tokens
                + im_start_tokens
                + _tokenize_str("user", query)[1]
                + im_end_tokens
                + nl_tokens
                + im_start_tokens
                + tokenizer.encode("assistant")
                + nl_tokens
            )
            raw_text += f"\n{im_start}user\n{query}{im_end}\n{im_start}assistant\n"

        elif chat_format == "raw":
            raw_text = query
            context_tokens = tokenizer.encode(raw_text)
        else:
            raise NotImplementedError(f"Unknown chat format {chat_format!r}")
        return raw_text, context_tokens

    def predict(self, sentence):

        query = self.processor.from_list_format([
                {'image': sentence[1]},
                {'text': sentence[0]},
            ])
        raw_text, context_tokens = self.make_context(tokenizer=self.processor, query=query)
        input_ids = torch.tensor([context_tokens]).to(self.device)
        with torch.no_grad():
            outputs = self.model(input_ids, output_hidden_states=True)
            hidden_states = outputs.hidden_states[1:]

            activations = [layer_hidden_state[0, -1, :].clone() for layer_hidden_state in hidden_states]
        new_activations = torch.stack(activations, dim=0).squeeze(0)
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
        pred_label = 0 if anomaly_score >= 0 else 1

        return pred_label, anomaly_score

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
        num_layers = self.background_activations_by_layer.shape[1]

        device = self.device

        all_sequences = torch.empty((num_samples, num_layers), device=device)

        background_activations_gpu = self.background_activations_by_layer.to(device)
        background_labels_gpu = self.background_labels.to(device)

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

        torch.save(all_sequences, self.training_sequences_path)
        return all_sequences

    def _calculate_single_rank_k_nb(self, test_vector, background_vectors, target_label, background_labels_arr, k, device):
        test_vector = test_vector.unsqueeze(0)

        layer_distances = (
            (background_vectors - test_vector)
            .norm(p=2, dim=2)
            .permute(1, 0)
        )
        sorted_indices = torch.argsort(layer_distances, dim=1)
        sorted_background_labels = background_labels_arr[sorted_indices]
        match_indices_in_sorted_tensor = torch.empty((sorted_background_labels.shape[0]), device=device)

        for i, s in enumerate(sorted_background_labels):
            matches = (torch.where(s == target_label)[0] + 1)[:k].float()
            match_indices_in_sorted_tensor[i] = matches.mean() if len(matches) else float(s.numel() + 1)

        return match_indices_in_sorted_tensor
