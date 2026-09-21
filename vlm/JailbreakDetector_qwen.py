from IsolationForest import PyTorchIsolationForest
import os
import torch
from tqdm import tqdm

class JailbreakDetector:
    """
    Jailbreak prompt detector based on K-NB Rank + IsolationForest
    Core logic: Calculate similarity ranking with benign samples, use IsolationForest for anomaly detection
    """

    def __init__(self, model, processor, background_layered_activations, all_labels, flag, n_estimators=100, random_state=42, k_nb=5):
        self.model = model
        self.processor = processor
        self.device = model.device
        self.flag = flag
        self.k_nb = k_nb 
        # Restructure hidden layer features: organize by layer (original: organized by sample)
        self.background_activations_by_layer = background_layered_activations
        self.background_labels = all_labels  # Background sample labels (1=benign, 0=malicious)
        self.num_layers = len(self.background_activations_by_layer) 
        # Generate training features
        if os.path.exists(f"./experimental_results/{flag}/training_sequences.pt"):
            training_sequences = torch.load(f"./experimental_results/{flag}/training_sequences.pt")
        else:
            training_sequences = self._get_training_sequences()
        y_train = self.background_labels  # Training labels (one-to-one with features)
        # Filter benign samples (IsolationForest trains only on benign samples to learn "normal" distribution)
        benign_indices = torch.where(y_train == 0)[0]
        benign_training_sequences = training_sequences[benign_indices]

        # PyTorch standardization (avoid NumPy conversion)
        self.mean = benign_training_sequences.mean(dim=0, keepdim=True)
        self.std = benign_training_sequences.std(dim=0, keepdim=True) + 1e-8  # Prevent division by zero
        X_train = (benign_training_sequences - self.mean) / self.std  # Standardization on GPU
        # PyTorch IsolationForest
        self.if_model = PyTorchIsolationForest(n_estimators=100, max_samples=256, random_state=42)
        self.if_model.fit(X_train)  # Train with Tensor directly

    def make_context(
        self,
        tokenizer,
        query,
        history = None,
        system = "",
        max_window_size = 6144,
        chat_format = "chatml",
    ):
        """
        Create context tokens for chat model input
        Args:
            tokenizer: Model tokenizer
            query: Current user query
            history: Chat history (list of (query, response) pairs)
            system: System prompt
            max_window_size: Max context window size
            chat_format: Chat format (chatml/raw)
        Returns:
            raw_text: Formatted text context
            context_tokens: Tokenized context
        """
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
        """
        Predict if a prompt is jailbreak
        Args:
            sentence: Input data (text + image url tuple)
        Returns:
            numeric label (0/1), anomaly score
            0 = malicious (jailbreak), 1 = benign
        """
        
        query = self.processor.from_list_format([
                {'image': sentence[1]},
                {'text': sentence[0]},
            ])
        raw_text, context_tokens = self.make_context(tokenizer=self.processor, query=query)
        input_ids = torch.tensor([context_tokens]).to('cuda')
        with torch.no_grad():
            outputs = self.model(input_ids, output_hidden_states=True)
            hidden_states = outputs.hidden_states[1:]

            # Critical fix: remove reference to hidden_states
            activations = [layer_hidden_state[0, -1, :].clone() for layer_hidden_state in hidden_states]
        new_activations = torch.stack(activations, dim=0).squeeze(0)
        # Calculate K-NB rank sequence for test sample (layer-wise iteration)
        ranks = self._calculate_single_rank_k_nb(
            new_activations,
            self.background_activations_by_layer,
            0,
            self.background_labels,
            k=self.k_nb,
            device=self.device
        )
        # Feature standardization (consistent with training)
        scaled_sequence = (ranks - self.mean) / self.std
        # Calculate anomaly score (PyTorch IsolationForest)
        anomaly_score = self.if_model.decision_function(scaled_sequence)[0].item()
        if anomaly_score > 0:  
            pred_label = 1 
        else:  
            pred_label = 0 

        return pred_label, anomaly_score

    def _restructure_activations(self, activations_list):
        """
        Restructure tensor list from sample-layer to layer-sample
        Args:
            activations_list: List of per-sample layer features (GPU tensor list)
                              e.g., [sample1_feats, sample2_feats, ...]
                              sample1_feats = [layer1_tensor, layer2_tensor, ...] (shape: [feat_dim])
        Returns:
            activations_by_layer: List of per-layer sample features (GPU tensor)
                                  e.g., [layer1_tensor, layer2_tensor, ...] (shape: [num_samples, feat_dim])
        """
        if not activations_list:
            return []
        
        # Get total layers (from first sample's feature list length)
        num_layers = len(activations_list[0])
        # Initialize empty list for layer-wise storage
        activations_by_layer = [[] for _ in range(num_layers)]
        
        # Traverse each sample's features and add to corresponding layer list
        for sample_activations in activations_list:
            for i in range(num_layers):
                activations_by_layer[i].append(sample_activations[i])
        
        # Concatenate each layer list to 2D GPU tensor (replace np.array conversion)
        return [torch.stack(layer_acts, dim=0) for layer_acts in activations_by_layer]

    def _get_training_sequences(self):
        """Calculate K-NB rank sequence as features for each sample in mixed background pool (GPU accelerated)"""
        num_samples = len(self.background_labels)
        num_layers = 32
        
        # Check GPU availability
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Pre-create Tensor to store all sequences (avoid list append then conversion)
        all_sequences = torch.empty((num_samples, num_layers), device=device)
        
        # Keep background activations and labels on GPU
        background_activations_gpu = self.background_activations_by_layer
        background_labels_gpu = self.background_labels
        
        for i in tqdm(range(num_samples), desc="Generating training sequences"):
            # Get current sample vector (on GPU)
            current_vector = background_activations_gpu[i]
            # Create mask to get other samples
            mask = torch.ones(num_samples, dtype=torch.bool, device=device)
            mask[i] = False
            other_vectors = background_activations_gpu[mask]
            other_labels = background_labels_gpu[mask]
            
            # Calculate ranking
            ranks = self._calculate_single_rank_k_nb(
                current_vector,
                other_vectors,
                0,
                other_labels,
                k=self.k_nb,
                device=device
            )
            
            # Store directly in Tensor
            all_sequences[i] = ranks
        
        # Save multi-dimensional Tensor
        torch.save(all_sequences, f"./experimental_results/{self.flag}/training_sequences.pt")
        return all_sequences

    def _calculate_single_rank_k_nb(self, test_vector, background_vectors, target_label, background_labels_arr, k, device):
        """
        Calculate K-NB ranking of single sample in a layer (GPU accelerated)
        Args:
            test_vector: Test sample feature vector
            background_vectors: Background sample feature vectors
            target_label: Target label for similarity matching
            background_labels_arr: Background sample labels array
            k: Top K similar samples to consider
            device: Computing device (GPU/CPU)
        Returns:
            match_indices_in_sorted_tensor: Mean rank of target label matches
        """
        # Ensure test vector is 2D for distance calculation (1, D)
        test_vector = test_vector.unsqueeze(0)  # From (D,) to (1, D)
        
        # Calculate Euclidean distance (GPU accelerated)
        layer_distances = (
            (background_vectors - test_vector)
            .norm(p=2, dim=2)
            .permute(1, 0)
        )
        # Sort by distance (ascending) and get indices
        sorted_indices = torch.argsort(layer_distances, dim=1)
        # Sorted background sample labels
        sorted_background_labels = background_labels_arr[sorted_indices]
        match_indices_in_sorted_tensor = torch.empty((32), device=device)
        
        for i, s in enumerate(sorted_background_labels):
            match_indices_in_sorted_tensor[i] = (torch.where(s == target_label)[0]+1)[:10].float().mean()       
        
        return match_indices_in_sorted_tensor