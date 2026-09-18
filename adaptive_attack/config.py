"""Configuration objects for adaptive attacks.

This module extends the base GCG configuration with weights for the target sequence
loss and the defense-feature loss. The token sampling, candidate filtering, and
prefix-cache behavior remain inherited from the original GCG implementation.
"""

from __future__ import annotations

from dataclasses import dataclass

from .gcg_core import GCGConfig


@dataclass
class AdaptiveGCGConfig(GCGConfig):
    """Configuration for joint GCG and defense-feature optimization.

    ``sequence_weight`` and ``feature_weight`` are direct multipliers. They do not
    need to sum to 1. For the MTK adaptive attack setting, they are usually set to
    ``1 - lambda`` and ``lambda``.
    """

    # Direct default for users who instantiate AdaptiveGCGConfig without CLI
    # overrides.
    num_steps: int = 1000
    sequence_weight: float = 1.0
    feature_weight: float = 1.0
    last_token_offset: int = -1

    def __post_init__(self) -> None:
        """Validate loss weights before the search starts."""
        if self.sequence_weight < 0:
            raise ValueError("sequence_weight must be non-negative")
        if self.feature_weight < 0:
            raise ValueError("feature_weight must be non-negative")
        if self.sequence_weight == 0 and self.feature_weight == 0:
            raise ValueError("sequence_weight and feature_weight cannot both be 0")
