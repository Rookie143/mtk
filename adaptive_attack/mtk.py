"""K-NB Rank, MTK detection, and L1/L2/L3 adaptive attacks."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Optional, Union

import torch
from torch import Tensor
import transformers

from .config import AdaptiveGCGConfig
from .engine import AdaptiveAttackResult, run_adaptive_attack
from .feature_store import HiddenStateLibrary
from .objectives import stack_last_token_features


def average_path_length(sample_count: int) -> float:
    """Compute the average path length for an MTK tree leaf."""
    if sample_count <= 1:
        return 0.0
    if sample_count == 2:
        return 1.0
    return (
        2.0 * (math.log(sample_count - 1) + 0.5772156649015329)
        - 2.0 * (sample_count - 1) / sample_count
    )


@dataclass
class _TreeNode:
    """Recursive MTK tree node storing split information."""

    depth: int
    sample_count: int
    feature: Optional[int] = None
    threshold: Optional[float] = None
    left: Optional["_TreeNode"] = None
    right: Optional["_TreeNode"] = None

    @property
    def is_leaf(self) -> bool:
        """Return whether this node is a leaf."""
        return self.left is None or self.right is None


class TorchMTKScorer:
    """PyTorch-only MTK inference implementation.

    This implementation reproduces the hard detector used in the experiments. It is
    not used for GCG backpropagation because tree traversal is discrete. The attack
    uses the hidden-state surrogate objective defined later in this file.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_samples: Union[str, int, float] = "auto",
        contamination: Union[str, float] = "auto",
        max_features: Union[int, float] = 1.0,
        bootstrap: bool = False,
        random_state: Optional[int] = None,
    ) -> None:
        """Initialize MTK scorer parameters."""
        if n_estimators <= 0:
            raise ValueError("n_estimators must be greater than 0")
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        self._rng = random.Random(random_state)
        self.estimators_: list[_TreeNode] = []
        self.estimators_features_: list[list[int]] = []
        self.max_samples_: Optional[int] = None
        self.offset_: Optional[Tensor] = None
        self.n_features_in_: Optional[int] = None
        self.device = torch.device("cpu")
        self.feature_importances_: Optional[Tensor] = None

    def _resolve_max_samples(self, sample_count: int) -> int:
        """Resolve ``max_samples`` into a concrete sample count."""
        if self.max_samples == "auto":
            return max(1, min(256, sample_count))
        if isinstance(self.max_samples, int):
            if self.max_samples <= 0:
                raise ValueError("integer max_samples must be greater than 0")
            return min(self.max_samples, sample_count)
        if isinstance(self.max_samples, float) and 0 < self.max_samples <= 1:
            return max(1, int(self.max_samples * sample_count))
        raise ValueError("max_samples must be 'auto', a positive integer, or a float in (0, 1]")

    def _resolve_feature_count(self, feature_count: int) -> int:
        """Resolve ``max_features`` into the number of features per tree."""
        if isinstance(self.max_features, int):
            if self.max_features <= 0:
                raise ValueError("integer max_features must be greater than 0")
            return min(self.max_features, feature_count)
        if isinstance(self.max_features, float) and 0 < self.max_features <= 1:
            return max(1, int(self.max_features * feature_count))
        raise ValueError("max_features must be a positive integer or a float in (0, 1]")

    def _build_tree(
        self,
        features: Tensor,
        sample_indices: Tensor,
        depth: int,
        max_depth: int,
        candidate_features: list[int],
    ) -> _TreeNode:
        """Recursively build one MTK tree."""
        sample_count = int(sample_indices.numel())
        if sample_count <= 1 or depth >= max_depth:
            return _TreeNode(depth=depth, sample_count=sample_count)

        shuffled_features = list(candidate_features)
        self._rng.shuffle(shuffled_features)
        for feature_index in shuffled_features:
            values = features[sample_indices, feature_index]
            unique_values = torch.unique(values)
            if unique_values.numel() <= 1:
                continue

            sorted_values = torch.sort(unique_values).values
            split_index = self._rng.randrange(sorted_values.numel() - 1)
            threshold = float(
                ((sorted_values[split_index] + sorted_values[split_index + 1]) / 2)
                .item()
            )
            left_indices = sample_indices[values < threshold]
            right_indices = sample_indices[values >= threshold]
            if left_indices.numel() == 0 or right_indices.numel() == 0:
                continue

            return _TreeNode(
                depth=depth,
                sample_count=sample_count,
                feature=feature_index,
                threshold=threshold,
                left=self._build_tree(
                    features,
                    left_indices,
                    depth + 1,
                    max_depth,
                    candidate_features,
                ),
                right=self._build_tree(
                    features,
                    right_indices,
                    depth + 1,
                    max_depth,
                    candidate_features,
                ),
            )

        return _TreeNode(depth=depth, sample_count=sample_count)

    def fit(self, features: Tensor) -> "TorchMTKScorer":
        """Fit the MTK scorer with a 2-D feature matrix."""
        if not isinstance(features, Tensor):
            raise TypeError("features must be a torch.Tensor")
        if features.ndim != 2:
            raise ValueError("features must be a [samples, features] 2-D tensor")
        if features.shape[0] == 0 or features.shape[1] == 0:
            raise ValueError("features cannot be empty")

        features = features.detach().float()
        self.device = features.device
        self.n_features_in_ = int(features.shape[1])
        self.max_samples_ = self._resolve_max_samples(int(features.shape[0]))
        feature_count = self._resolve_feature_count(self.n_features_in_)
        candidate_features = list(range(self.n_features_in_))
        max_depth = math.ceil(math.log2(max(self.max_samples_, 2)))
        self.estimators_.clear()
        self.estimators_features_.clear()

        # Each tree sees a random sample subset and random feature subset, matching
        # the MTK randomized partitioning idea while keeping all trees in Python
        # objects that can be evaluated without scikit-learn at inference time.
        for _ in range(self.n_estimators):
            if self.bootstrap:
                sampled = [
                    self._rng.randrange(features.shape[0])
                    for _ in range(self.max_samples_)
                ]
            else:
                sampled = self._rng.sample(
                    range(features.shape[0]),
                    k=min(self.max_samples_, features.shape[0]),
                )
            tree_samples = torch.tensor(sampled, device=self.device, dtype=torch.long)
            tree_features = self._rng.sample(candidate_features, k=feature_count)
            tree = self._build_tree(
                features,
                tree_samples,
                depth=0,
                max_depth=max_depth,
                candidate_features=tree_features,
            )
            self.estimators_.append(tree)
            self.estimators_features_.append(tree_features)

        self._compute_feature_importances()
        if self.contamination == "auto":
            self.offset_ = torch.tensor(-0.5, device=self.device)
        elif isinstance(self.contamination, float) and 0 < self.contamination < 0.5:
            # score_samples only needs the trees. A temporary zero offset lets the
            # fitted checks stay shared while computing the contamination quantile.
            self.offset_ = torch.tensor(0.0, device=self.device)
            scores = self.score_samples(features)
            self.offset_ = torch.quantile(scores, self.contamination)
        else:
            raise ValueError("contamination must be 'auto' or a float in (0, 0.5)")
        return self

    def _compute_feature_importances(self) -> None:
        """Compute simple feature importances from split counts."""
        if self.n_features_in_ is None:
            return
        counts = torch.zeros(self.n_features_in_, device=self.device)
        for root in self.estimators_:
            pending = [root]
            while pending:
                node = pending.pop()
                if node.feature is not None:
                    counts[node.feature] += 1
                if node.left is not None:
                    pending.append(node.left)
                if node.right is not None:
                    pending.append(node.right)
        total = counts.sum()
        self.feature_importances_ = counts / total if total > 0 else counts

    def _path_length(self, sample: Tensor, node: _TreeNode) -> float:
        """Compute the corrected path length for one sample in one tree."""
        current = node
        while not current.is_leaf:
            if current.feature is None or current.threshold is None:
                break
            if sample[current.feature] < current.threshold:
                current = current.left  # type: ignore[assignment]
            else:
                current = current.right  # type: ignore[assignment]
        return current.depth + average_path_length(current.sample_count)

    def _check_fitted(self) -> None:
        """Check whether the scorer has been fitted."""
        if (
            not self.estimators_
            or self.max_samples_ is None
            or self.n_features_in_ is None
            or self.offset_ is None
        ):
            raise RuntimeError("MTK scorer has not been fitted")

    def _prepare_features(self, features: Tensor) -> Tensor:
        """Validate and convert inference features."""
        self._check_fitted()
        if not isinstance(features, Tensor):
            raise TypeError("features must be a torch.Tensor")
        if features.ndim == 1:
            features = features.unsqueeze(0)
        if features.ndim != 2:
            raise ValueError("features must be a [samples, features] 2-D tensor")
        if features.shape[1] != self.n_features_in_:
            raise ValueError("inference feature dimension differs from training dimension")
        return features.to(device=self.device, dtype=torch.float32)

    def score_samples(self, features: Tensor) -> Tensor:
        """Return anomaly scores; lower values are more likely anomalous."""
        prepared = self._prepare_features(features)
        assert self.max_samples_ is not None
        normalizer = average_path_length(self.max_samples_)
        if normalizer == 0:
            return -torch.ones(prepared.shape[0], device=self.device)

        depths = torch.zeros(prepared.shape[0], device=self.device)
        for tree in self.estimators_:
            for index in range(prepared.shape[0]):
                depths[index] += self._path_length(prepared[index], tree)
        mean_depth = depths / len(self.estimators_)
        return torch.pow(2.0, -mean_depth / normalizer) * -1.0

    def decision_function(self, features: Tensor) -> Tensor:
        """Return decision scores after subtracting the fitted offset."""
        self._check_fitted()
        assert self.offset_ is not None
        return self.score_samples(features) - self.offset_

    def predict(self, features: Tensor) -> Tensor:
        """Return 1 for benign and 0 for anomalous or malicious."""
        return torch.where(self.decision_function(features) < 0, 0, 1)


class KNNRankEncoder:
    """Encode multi-layer hidden states into K-NB rank sequences."""

    def __init__(
        self,
        background_features: Tensor,
        background_labels: Tensor,
        target_label: int = 1,
        k: int = 10,
    ) -> None:
        """Initialize background features and rank parameters."""
        if background_features.ndim != 3:
            raise ValueError("background_features must be [samples, layers, hidden]")
        if background_labels.ndim != 1:
            raise ValueError("background_labels must be 1-D")
        if background_features.shape[0] != background_labels.shape[0]:
            raise ValueError("background feature and label counts differ")
        if k <= 0:
            raise ValueError("k must be greater than 0")
        self.background_features = background_features.float()
        self.background_labels = background_labels.to(
            device=background_features.device
        )
        self.target_label = target_label
        self.k = k

    def _encode_one(
        self,
        query: Tensor,
        background_features: Tensor,
        background_labels: Tensor,
    ) -> Tensor:
        """Compute per-layer mean target-label rank for one query."""
        ranks = []
        positions = torch.arange(
            1,
            background_features.shape[0] + 1,
            device=query.device,
            dtype=torch.float32,
        )
        for layer_index in range(background_features.shape[1]):
            distances = torch.linalg.vector_norm(
                background_features[:, layer_index, :]
                - query[layer_index].unsqueeze(0),
                dim=1,
            )
            sorted_labels = background_labels[torch.argsort(distances)]
            target_positions = positions[sorted_labels == self.target_label]
            if target_positions.numel() == 0:
                ranks.append(positions[-1] + 1)
            else:
                ranks.append(target_positions[: self.k].mean())
        return torch.stack(ranks)

    def encode(
        self,
        query_features: Tensor,
        exclude_self: bool = False,
    ) -> Tensor:
        """Convert ``[batch, layers, hidden]`` into ``[batch, layers]`` ranks."""
        if query_features.ndim != 3:
            raise ValueError("query_features must be [batch, layers, hidden]")
        if query_features.shape[1:] != self.background_features.shape[1:]:
            raise ValueError("query_features layer or hidden dimensions do not match background")

        query_features = query_features.to(self.background_features.device).float()
        encoded = []
        for sample_index, query in enumerate(query_features):
            background = self.background_features
            labels = self.background_labels
            if exclude_self and query_features.shape[0] == background.shape[0]:
                mask = torch.ones(background.shape[0], device=background.device).bool()
                mask[sample_index] = False
                background = background[mask]
                labels = labels[mask]
            encoded.append(self._encode_one(query, background, labels))
        return torch.stack(encoded)


class MTKDetector:
    """K-NB Rank + MTK detector from the original experiments."""

    def __init__(
        self,
        background_features: Tensor,
        background_labels: Tensor,
        k: int = 10,
        n_estimators: int = 500,
        max_samples: Union[str, int, float] = 512,
        random_state: Optional[int] = 42,
    ) -> None:
        """Build and fit the detector."""
        self.encoder = KNNRankEncoder(
            background_features=background_features,
            background_labels=background_labels,
            target_label=1,
            k=k,
        )
        self.background_labels = background_labels.to(background_features.device)
        # The training set is the reference bank itself. With exclude_self=True,
        # each anchor is removed before computing its own rank trajectory, so the
        # nearest-neighbor signal is not dominated by the sample itself.
        training_sequences = self.encoder.encode(
            self.encoder.background_features,
            exclude_self=True,
        )
        benign_mask = self.background_labels == 1
        if not torch.any(benign_mask):
            raise ValueError("background bank contains no benign samples")
        benign_sequences = training_sequences[benign_mask]
        self.mean = benign_sequences.mean(dim=0, keepdim=True)
        self.std = benign_sequences.std(dim=0, keepdim=True).clamp_min(1e-8)
        self.scorer = TorchMTKScorer(
            n_estimators=n_estimators,
            max_samples=max_samples,
            random_state=random_state,
        ).fit((benign_sequences - self.mean) / self.std)

    @classmethod
    def from_library(
        cls: type["MTKDetector"],
        library: HiddenStateLibrary,
        k: int = 10,
        n_estimators: int = 500,
        max_samples: Union[str, int, float] = 512,
        random_state: Optional[int] = 42,
    ) -> "MTKDetector":
        """Create a detector from a labeled feature library."""
        if library.labels is None:
            raise ValueError("MTK detector requires a labeled feature library")
        return cls(
            background_features=library.features,
            background_labels=library.labels,
            k=k,
            n_estimators=n_estimators,
            max_samples=max_samples,
            random_state=random_state,
        )

    def decision_function(self, query_features: Tensor) -> Tensor:
        """Compute detector decision scores for query features."""
        rank_sequence = self.encoder.encode(query_features)
        scaled = (rank_sequence - self.mean) / self.std
        return self.scorer.decision_function(scaled)

    def predict(self, query_features: Tensor) -> Tensor:
        """Return 1 for benign and 0 for detector-flagged anomalous samples."""
        return torch.where(self.decision_function(query_features) < 0, 0, 1)


def _pairwise_mse(query: Tensor, references: Tensor) -> Tensor:
    """Compute pairwise MSE from query to reference bank as ``[batch, anchors]``.

    Materializing ``[batch, anchors, hidden]`` differences can consume a large
    amount of memory. The quadratic expansion
    ``||x-y||^2 = ||x||^2 + ||y||^2 - 2xy`` gives the same elementwise MSE while
    preserving gradients with respect to the query.
    """
    if query.ndim != 2 or references.ndim != 2:
        raise ValueError("query and references must both be 2-D tensors")
    if query.shape[1] != references.shape[1]:
        raise ValueError("query and references hidden dimensions do not match")

    hidden_size = float(query.shape[1])
    squared_distance = (
        query.pow(2).sum(dim=1, keepdim=True)
        + references.pow(2).sum(dim=1).unsqueeze(0)
        - 2.0 * query @ references.T
    )
    return squared_distance.clamp_min(0.0) / hidden_size


@dataclass
class MTKAdaptiveObjective:
    """Differentiable L1/L2/L3 surrogate for K-NB Rank + MTK.

    The K-NB rank trajectory requires sorting, and MTK also contains tree-path
    comparisons. Neither is a stable direct gradient target. The adaptive attack
    therefore does not backpropagate through the hard detector. Instead, it applies
    one of three hidden-state surrogates to the target model:

    * ``l1``: pull each layer toward the mean MSE of all benign anchors;
    * ``l2``: pull each layer toward the nearest benign anchor;
    * ``l3``: pull toward the benign bank and push away from the malicious bank.

    ``hidden_states`` are read from the target model forward pass and stacked at the
    position before the first target-output token as ``[batch, layers, hidden]``.
    """

    benign_features: Tensor
    malicious_features: Tensor
    loss_type: str = "l3"
    scale: float = 1.0

    def __post_init__(self) -> None:
        """Validate reference-bank layout and surrogate type."""
        if self.loss_type not in {"l1", "l2", "l3"}:
            raise ValueError("loss_type must be l1, l2, or l3")
        if self.benign_features.ndim != 3:
            raise ValueError("benign_features must be [samples, layers, hidden]")
        if self.malicious_features.ndim != 3:
            raise ValueError("malicious_features must be [samples, layers, hidden]")
        if self.benign_features.shape[1:] != self.malicious_features.shape[1:]:
            raise ValueError("benign and malicious feature dimensions do not match")
        if self.benign_features.shape[0] == 0:
            raise ValueError("benign_features cannot be empty")
        if self.malicious_features.shape[0] == 0:
            raise ValueError("malicious_features cannot be empty")
        if self.scale < 0:
            raise ValueError("scale must be non-negative")

    @property
    def name(self) -> str:
        """Return an objective name that includes the selected loss type."""
        return "mtk_%s" % self.loss_type

    @classmethod
    def from_library(
        cls: type["MTKAdaptiveObjective"],
        library: HiddenStateLibrary,
        benign_label: int = 1,
        malicious_label: int = 0,
        loss_type: str = "l3",
        scale: float = 1.0,
    ) -> "MTKAdaptiveObjective":
        """Build the surrogate objective from a labeled reference bank."""
        benign, malicious = library.split_by_label(benign_label, malicious_label)
        return cls(
            benign_features=benign,
            malicious_features=malicious,
            loss_type=loss_type,
            scale=scale,
        )

    def __call__(
        self,
        hidden_states: list[Tensor] | tuple[Tensor, ...],
        last_token_index: int,
    ) -> Tensor:
        """Return one paper-aligned surrogate loss per candidate sample."""
        query = stack_last_token_features(hidden_states, last_token_index).float()
        if query.shape[1:] != self.benign_features.shape[1:]:
            raise ValueError(
                "query hidden-state layer or hidden dimensions do not match the reference bank"
            )

        benign = self.benign_features.to(
            device=query.device,
            dtype=query.dtype,
        )
        malicious = self.malicious_features.to(
            device=query.device,
            dtype=query.dtype,
        )
        layer_losses = []
        for layer_index in range(query.shape[1]):
            query_layer = query[:, layer_index, :]
            benign_layer = benign[:, layer_index, :]

            # Process one layer at a time to avoid materializing batch, layer,
            # anchor, and hidden dimensions together. L1/L2/L3 differ only in how
            # the anchor dimension is aggregated.
            benign_mse = _pairwise_mse(query_layer, benign_layer)
            if self.loss_type == "l1":
                layer_loss = benign_mse.mean(dim=1)
            elif self.loss_type == "l2":
                layer_loss = benign_mse.amin(dim=1)
            else:
                malicious_layer = malicious[:, layer_index, :]
                malicious_mse = _pairwise_mse(query_layer, malicious_layer)
                layer_loss = benign_mse.mean(dim=1) - malicious_mse.mean(dim=1)
            layer_losses.append(layer_loss)

        # The paper lambda weights L_adv against L_evasion. Averaging over layers
        # prevents the evasion loss from growing linearly with the number of
        # layers, keeping lambda comparable across model families.
        return torch.stack(layer_losses, dim=1).mean(dim=1) * self.scale


def run_mtk_attack(
    model: transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizer,
    messages: str | list[dict[str, str]],
    target: str,
    feature_library: HiddenStateLibrary,
    loss_type: str = "l3",
    config: Optional[AdaptiveGCGConfig] = None,
    scale: float = 1.0,
    benign_label: int = 1,
    malicious_label: int = 0,
) -> AdaptiveAttackResult:
    """Run one MTK adaptive attack with the L1/L2/L3 paper surrogate."""
    objective = MTKAdaptiveObjective.from_library(
        feature_library,
        benign_label=benign_label,
        malicious_label=malicious_label,
        loss_type=loss_type,
        scale=scale,
    )
    return run_adaptive_attack(
        model=model,
        tokenizer=tokenizer,
        messages=messages,
        target=target,
        objective=objective,
        config=config,
    )
