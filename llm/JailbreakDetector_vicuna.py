from __future__ import annotations

import torch

from IsolationForest import PyTorchIsolationForest
from extract_trainset_hiddenstates_vicuna import extract_endpoint_activations


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
    def __init__(self, model, tokenizer, background_layered_activations, all_labels,
                 settings: dict, random_state: int, device: str, batch_size: int, rank_batch_size: int):
        self.model = model
        self.tokenizer = tokenizer
        self.background = background_layered_activations
        self.labels = all_labels
        self.settings = settings
        self.device = device
        self.batch_size = batch_size
        self.rank_batch_size = rank_batch_size
        sequences = rank_features(
            self.background, self.background, self.labels, settings["k"], device,
            exclude_self=True, batch_size=rank_batch_size,
        )
        benign = sequences[all_labels == 0]
        self.mean = benign.mean(dim=0, keepdim=True)
        self.std = benign.std(dim=0, keepdim=True) + 1e-8
        self.forest = PyTorchIsolationForest(
            n_estimators=settings["n_estimators"], max_samples=settings["max_samples"], random_state=random_state,
        ).fit((benign - self.mean) / self.std)
        self.forest_stats = {
            "tree_count": len(self.forest.estimators_),
            "total_nodes": sum(len(t["node_samples"]) for t in self.forest.estimators_),
            "internal_nodes": sum((len(t["node_samples"]) - 1) // 2 for t in self.forest.estimators_),
            "empty_nodes": sum(int((t["node_samples"] == 0).sum()) for t in self.forest.estimators_),
        }
        if self.forest_stats["empty_nodes"]:
            raise RuntimeError("Forest has empty nodes")

    def predict_batch(self, prompts: list[str]):
        activations = extract_endpoint_activations(
            self.model, self.tokenizer, prompts, self.batch_size, self.device,
        )
        ranks = rank_features(
            activations, self.background, self.labels, self.settings["k"], self.device,
            exclude_self=False, batch_size=self.rank_batch_size,
        )
        scores = -self.forest.decision_function((ranks - self.mean) / self.std).cpu().numpy()
        return scores, (scores > 0).astype("int64"), {"ant": scores}
