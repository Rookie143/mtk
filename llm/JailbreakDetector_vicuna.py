from __future__ import annotations

from pathlib import Path

import torch


from IsolationForest import PyTorchIsolationForest

from extract_trainset_hiddenstates_vicuna import extract_dual_endpoint_activations


def rank_features(queries, background, labels, k: int, device: str, exclude_self=False, batch_size=64):
    n_queries, n_layers, _ = queries.shape
    output = torch.empty((n_queries, n_layers), dtype=torch.float32)
    background = background.to(device)
    labels = labels.to(device)
    benign_count = int((labels == 0).sum()) - int(exclude_self)
    if k > benign_count:
        raise ValueError(f"k={k} exceeds available benign references {benign_count}")
    positions = torch.arange(1, len(labels) + 1, device=device, dtype=torch.float32)
    for layer in range(n_layers):
        refs = background[:, layer, :]
        for start in range(0, n_queries, batch_size):
            stop = min(start + batch_size, n_queries)
            query = queries[start:stop, layer, :].to(device=device, dtype=background.dtype)
            distances = (refs.unsqueeze(0) - query.unsqueeze(1)).norm(p=2, dim=2)
            if exclude_self:
                local = torch.arange(stop - start, device=device)
                distances[local, torch.arange(start, stop, device=device)] = torch.inf
            order = distances.argsort(dim=1)
            benign_sorted = labels[order].eq(0)
            ranked_positions = positions.expand(stop - start, -1).masked_fill(~benign_sorted, torch.inf)
            nearest = ranked_positions.topk(k, largest=False, dim=1).values
            output[start:stop, layer] = nearest.mean(dim=1).cpu()
        print(f"rank features layer {layer + 1}/{n_layers}", flush=True)
    return output


class JailbreakDetector:
    def __init__(
        self,
        model,
        tokenizer,
        background_layered_activations,
        all_labels,
        your_flag,
        colon_settings: dict,
        ist_settings: dict,
        ist_weight: float,
        random_state: int,
        device: str,
        colon_batch_size: int,
        ist_batch_size: int,
        rank_batch_size: int,
    ):
        if not 0.0 <= ist_weight <= 1.0:
            raise ValueError("ist_weight must be in [0, 1]")
        self.model = model
        self.tokenizer = tokenizer
        self.background = background_layered_activations
        self.labels = all_labels
        self.your_flag = your_flag
        self.colon_settings = colon_settings
        self.ist_settings = ist_settings
        self.ist_weight = ist_weight
        self.random_state = random_state
        self.device = device
        self.colon_batch_size = colon_batch_size
        self.ist_batch_size = ist_batch_size
        self.rank_batch_size = rank_batch_size
        self.models = {}
        self.normalizers = {}
        for endpoint, settings in (("colon", colon_settings), ("ist", ist_settings)):
            sequences = self._training_sequences(endpoint, settings["k"])
            benign = sequences[all_labels == 0]
            mean = benign.mean(dim=0, keepdim=True)
            std = benign.std(dim=0, keepdim=True) + 1e-8
            train = (benign - mean) / std
            forest = PyTorchIsolationForest(
                n_estimators=settings["n_estimators"],
                max_samples=settings["max_samples"],
                random_state=random_state,
            ).fit(train)
            self.models[endpoint] = forest
            self.normalizers[endpoint] = (mean, std)

    def _training_sequences(self, endpoint: str, k: int):
        path = Path(self.your_flag) / f"training_sequences_{endpoint}.pt"
        if path.exists():
            return torch.load(path, map_location="cpu", weights_only=False)
        sequences = rank_features(
            self.background[endpoint],
            self.background[endpoint],
            self.labels,
            k,
            self.device,
            exclude_self=True,
            batch_size=self.rank_batch_size,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(sequences, path)
        return sequences

    def score_activations(self, activations):
        scores = {}
        for endpoint, settings in (("colon", self.colon_settings), ("ist", self.ist_settings)):
            ranks = rank_features(
                activations[endpoint],
                self.background[endpoint],
                self.labels,
                settings["k"],
                self.device,
                exclude_self=False,
                batch_size=self.rank_batch_size,
            )
            mean, std = self.normalizers[endpoint]
            normalized = (ranks - mean) / std
            scores[endpoint] = -self.models[endpoint].decision_function(normalized).cpu().numpy()
        fused = (1.0 - self.ist_weight) * scores["colon"] + self.ist_weight * scores["ist"]
        return fused, scores

    def predict_batch(self, prompts: list[str]):
        activations = extract_dual_endpoint_activations(
            self.model,
            self.tokenizer,
            prompts,
            self.colon_batch_size,
            self.ist_batch_size,
            self.device,
        )
        return self.predict_activations(activations)

    def predict_activations(self, activations):
        fused, endpoint_scores = self.score_activations(activations)
        labels = (fused > 0).astype("int64")
        return fused, labels, endpoint_scores
