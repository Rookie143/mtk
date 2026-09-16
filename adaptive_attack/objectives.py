"""Common protocols and tensor utilities for defense-feature objectives."""

from __future__ import annotations

from typing import Protocol, Sequence

import torch
from torch import Tensor


class HiddenStateObjective(Protocol):
    """Protocol for hidden-state feature objectives that can be injected into GCG.

    Implementations receive per-layer hidden states plus the last position before
    the target tokens, and return one scalar loss per candidate. The return value
    must keep its computation graph so it can participate in the one-hot token
    gradient used by GCG.
    """

    @property
    def name(self) -> str:
        """Return the objective name for logging and result records."""

    def __call__(
        self,
        hidden_states: Sequence[Tensor],
        last_token_index: int,
    ) -> Tensor:
        """Compute one defense-feature loss per candidate sample."""


def stack_last_token_features(
    hidden_states: Sequence[Tensor],
    last_token_index: int,
) -> Tensor:
    """Stack a selected hidden-state position as ``[batch, layers, hidden]``.

    Models usually return a list of ``[batch, sequence, hidden]`` tensors. This
    normalizes them into the shared 3-D input contract used by MTK surrogates and
    other detector adapters.
    """
    if not hidden_states:
        raise ValueError("hidden_states cannot be empty")

    layer_features = []
    for layer_index, layer_state in enumerate(hidden_states):
        if layer_state.ndim != 3:
            raise ValueError(
                "hidden state at layer %d must be 3-D, got shape %s"
                % (layer_index, tuple(layer_state.shape))
            )
        if last_token_index < 0 or last_token_index >= layer_state.shape[1]:
            raise IndexError(
                "last_token_index=%d is out of bounds for layer %d with length %d"
                % (last_token_index, layer_index, layer_state.shape[1])
            )
        layer_features.append(layer_state[:, last_token_index, :])

    return torch.stack(layer_features, dim=1)


def reduce_per_sample_loss(value: Tensor, batch_size: int) -> Tensor:
    """Normalize common loss shapes to ``[batch]``.

    Some experiment code collapses the feature loss across the whole batch and then
    adds it to a ``[batch]`` sequence loss, which can trigger silent broadcasting.
    This helper keeps one comparable scalar per candidate sample.
    """
    if value.ndim == 0:
        return value.expand(batch_size)
    if value.shape[0] != batch_size:
        raise ValueError(
            "feature loss first dimension must equal batch_size, got shape %s, "
            "batch_size=%d"
            % (tuple(value.shape), batch_size)
        )
    return value.reshape(batch_size, -1).mean(dim=1)
