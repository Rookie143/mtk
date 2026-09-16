"""Sequence-loss computation for adaptive GCG."""

from __future__ import annotations

import torch
from torch import Tensor

from .gcg_utils import mellowmax


def target_sequence_loss(
    logits: Tensor,
    target_ids: Tensor,
    input_length: int,
    use_mellowmax: bool = False,
    mellowmax_alpha: float = 1.0,
) -> Tensor:
    """Compute the target-token loss for each candidate sequence.

    ``input_length`` is the current embedding sequence length passed to the model.
    With a prefix cache it does not include the cached fixed prefix, but the target
    token span is still located by ``input_length - target_length``.
    """
    if logits.ndim != 3:
        raise ValueError("logits must be a [batch, sequence, vocab] 3-D tensor")
    if target_ids.ndim == 1:
        target_ids = target_ids.unsqueeze(0)
    if target_ids.ndim != 2:
        raise ValueError("target_ids must be [batch, target_length] or [1, target_length]")

    batch_size = logits.shape[0]
    target_length = target_ids.shape[1]
    shift = input_length - target_length
    if shift <= 0:
        raise ValueError(
            "input_length must be greater than target_length, got %d and %d"
            % (input_length, target_length)
        )

    labels = target_ids.expand(batch_size, -1)
    shift_logits = logits[..., shift - 1 : -1, :].contiguous()
    if shift_logits.shape[1] != target_length:
        raise ValueError(
            "target logits length mismatch: got %d, expected %d"
            % (shift_logits.shape[1], target_length)
        )

    if use_mellowmax:
        label_logits = torch.gather(
            shift_logits,
            dim=-1,
            index=labels.unsqueeze(-1),
        ).squeeze(-1)
        return mellowmax(-label_logits, alpha=mellowmax_alpha, dim=-1)

    token_loss = torch.nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.shape[-1]),
        labels.reshape(-1),
        reduction="none",
    )
    return token_loss.view(batch_size, target_length).mean(dim=1)
