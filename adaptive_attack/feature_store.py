"""Loading and validation utilities for hidden-state feature libraries."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
from torch import Tensor


@dataclass
class HiddenStateLibrary:
    """Sample-wise hidden-state feature library.

    ``features`` must use shape ``[samples, layers, hidden_size]``, matching the
    original ``background_layered_activations`` experiment files.
    """

    features: Tensor
    labels: Optional[Tensor] = None

    def __post_init__(self) -> None:
        """Validate shape constraints before the library is used by losses."""
        if self.features.ndim != 3:
            raise ValueError(
                "features must be [samples, layers, hidden_size], got shape %s"
                % (tuple(self.features.shape),)
            )
        if self.features.shape[0] == 0:
            raise ValueError("features cannot be empty")
        if self.labels is not None:
            if self.labels.ndim != 1:
                raise ValueError("labels must be a 1-D tensor")
            if self.labels.shape[0] != self.features.shape[0]:
                raise ValueError("labels must have the same sample count as features")

    @property
    def num_samples(self) -> int:
        """Return the number of samples."""
        return int(self.features.shape[0])

    @property
    def num_layers(self) -> int:
        """Return the number of Transformer layers."""
        return int(self.features.shape[1])

    def split_by_label(
        self,
        benign_label: int = 1,
        malicious_label: int = 0,
    ) -> tuple[Tensor, Tensor]:
        """Split features by label and return ``(benign, malicious)``."""
        if self.labels is None:
            raise ValueError("the feature library has no labels to split by")
        benign_mask = self.labels == benign_label
        malicious_mask = self.labels == malicious_label
        if not torch.any(benign_mask):
            raise ValueError("no samples found for benign_label=%d" % benign_label)
        if not torch.any(malicious_mask):
            raise ValueError(
                "no samples found for malicious_label=%d" % malicious_label
            )
        return self.features[benign_mask], self.features[malicious_mask]


def _torch_load(path: Path, map_location: Optional[str]) -> Any:
    """Load tensor-only experiment files across PyTorch versions."""
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_hidden_state_library(
    path: str | Path,
    map_location: Optional[str] = None,
    feature_key: str = "background_layered_activations",
    label_key: str = "labels",
) -> HiddenStateLibrary:
    """Load a ``.pt`` feature library produced by the experiments.

    Supported payloads:

    * a directly saved ``Tensor``;
    * a dictionary, reading ``background_layered_activations`` and ``labels`` by
      default.
    """
    source_path = Path(path)
    if not source_path.is_file():
        raise FileNotFoundError("feature library file not found: %s" % source_path)

    payload = _torch_load(source_path, map_location)
    labels: Optional[Tensor] = None
    if isinstance(payload, Tensor):
        features = payload
    elif isinstance(payload, dict):
        if feature_key not in payload:
            raise KeyError("feature library is missing key: %s" % feature_key)
        features = payload[feature_key]
        candidate_labels = payload.get(label_key)
        if candidate_labels is not None:
            labels = candidate_labels
    else:
        raise TypeError("feature library must be a Tensor or dict, got %s" % type(payload))

    if not isinstance(features, Tensor):
        raise TypeError("feature field must be a torch.Tensor")
    if labels is not None and not isinstance(labels, Tensor):
        raise TypeError("label field must be a torch.Tensor")
    return HiddenStateLibrary(features=features, labels=labels)
