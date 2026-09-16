"""Shared adaptive GCG search engine."""

from __future__ import annotations

import gc
import logging
from dataclasses import dataclass
from typing import Any, Optional, Union

import torch
import torch.nn.functional as F
from torch import Tensor
import transformers

from .gcg_core import GCG, GCGResult

from .config import AdaptiveGCGConfig
from .losses import target_sequence_loss
from .objectives import HiddenStateObjective, reduce_per_sample_loss


logger = logging.getLogger("adaptive_attack")


@dataclass
class AdaptiveAttackResult(GCGResult):
    """GCG result extended with per-objective losses for the best candidate."""

    objective_name: str
    sequence_loss: float
    feature_loss: float


class AdaptiveGCG(GCG):
    """Inject a differentiable hidden-state objective into the GCG search loop.

    The base class owns prompt splitting, prefix caching, candidate sampling,
    retokenization filtering, and probe sampling. This subclass only overrides the
    forward pass that needs hidden states and the joint loss composition.
    """

    def __init__(
        self,
        model: transformers.PreTrainedModel,
        tokenizer: transformers.PreTrainedTokenizer,
        config: AdaptiveGCGConfig,
        objective: HiddenStateObjective,
    ) -> None:
        """Initialize the shared engine with one defense objective."""
        super().__init__(model, tokenizer, config)
        self.adaptive_config = config
        self.objective = objective

    def _expand_prefix_cache(self, batch_size: int) -> Any:
        """Expand a single-sample prefix cache to a candidate batch.

        The transformers versions used by this code represent each layer as
        ``(key, value)``. Keeping the expansion here avoids duplicating cache logic
        in every defense adapter.
        """
        if self.prefix_cache is None:
            return None
        return tuple(
            tuple(state.expand(batch_size, -1, -1, -1) for state in layer)
            for layer in self.prefix_cache
        )

    def _forward_with_hidden_states(self, input_embeds: Tensor) -> Any:
        """Run the target model and always request hidden states."""
        if self.prefix_cache is not None:
            batch_size = int(input_embeds.shape[0])
            past_key_values = self.prefix_cache
            if batch_size != 1:
                past_key_values = self._expand_prefix_cache(batch_size)
            return self.model(
                inputs_embeds=input_embeds,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
                output_hidden_states=True,
            )
        return self.model(
            inputs_embeds=input_embeds,
            return_dict=True,
            output_hidden_states=True,
        )

    def _compute_loss_components(
        self,
        outputs: Any,
        input_length: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Compute sequence loss, feature loss, and weighted total loss."""
        sequence_loss = target_sequence_loss(
            logits=outputs.logits,
            target_ids=self.target_ids,
            input_length=input_length,
            use_mellowmax=self.config.use_mellowmax,
            mellowmax_alpha=self.config.mellowmax_alpha,
        )

        target_start = input_length - self.target_ids.shape[1]
        last_token_index = target_start + self.adaptive_config.last_token_offset
        hidden_states = outputs.hidden_states[1:]
        feature_loss = self.objective(hidden_states, last_token_index)
        feature_loss = reduce_per_sample_loss(
            feature_loss,
            batch_size=outputs.logits.shape[0],
        )

        total_loss = (
            self.adaptive_config.sequence_weight * sequence_loss
            + self.adaptive_config.feature_weight * feature_loss
        )
        return sequence_loss, feature_loss, total_loss

    def compute_token_gradient(self, optim_ids: Tensor) -> Tensor:
        """Compute the joint-loss gradient with respect to optimized token one-hots."""
        embedding_layer = self.embedding_layer
        one_hot = F.one_hot(
            optim_ids,
            num_classes=embedding_layer.num_embeddings,
        ).to(self.model.device, self.model.dtype)
        one_hot.requires_grad_()

        optim_embeds = one_hot @ embedding_layer.weight
        input_embeds = torch.cat(
            [optim_embeds, self.after_embeds, self.target_embeds],
            dim=1,
        )
        if self.prefix_cache is None:
            input_embeds = torch.cat(
                [
                    self.before_embeds,
                    optim_embeds,
                    self.after_embeds,
                    self.target_embeds,
                ],
                dim=1,
            )

        outputs = self._forward_with_hidden_states(input_embeds)
        _, _, total_loss = self._compute_loss_components(
            outputs,
            input_length=input_embeds.shape[1],
        )
        return torch.autograd.grad(total_loss.sum(), one_hot)[0]

    def _compute_candidates_loss_original(
        self,
        search_batch_size: int,
        input_embeds: Tensor,
    ) -> Tensor:
        """Compute joint loss for candidate suffixes in batches."""
        losses = []
        for start in range(0, input_embeds.shape[0], search_batch_size):
            with torch.no_grad():
                input_batch = input_embeds[start : start + search_batch_size]
                outputs = self._forward_with_hidden_states(input_batch)
                _, _, total_loss = self._compute_loss_components(
                    outputs,
                    input_length=input_batch.shape[1],
                )
                losses.append(total_loss)

                if self.config.early_stop:
                    target_length = self.target_ids.shape[1]
                    shift = input_batch.shape[1] - target_length
                    shift_logits = outputs.logits[..., shift - 1 : -1, :]
                    labels = self.target_ids.expand(input_batch.shape[0], -1)
                    if torch.any(
                        torch.all(
                            torch.argmax(shift_logits, dim=-1) == labels,
                            dim=-1,
                        )
                    ):
                        self.stop_flag = True

                del outputs
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        return torch.cat(losses, dim=0)

    def _evaluate_best_candidate(
        self,
        best_string: str,
    ) -> tuple[float, float, float]:
        """Re-evaluate the best suffix and return unweighted component losses."""
        best_ids = self.tokenizer(
            best_string,
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"].to(self.model.device)
        input_embeds = torch.cat(
            [
                self.embedding_layer(best_ids),
                self.after_embeds,
                self.target_embeds,
            ],
            dim=1,
        )
        if self.prefix_cache is None:
            input_embeds = torch.cat(
                [
                    self.before_embeds,
                    self.embedding_layer(best_ids),
                    self.after_embeds,
                    self.target_embeds,
                ],
                dim=1,
            )

        with torch.no_grad():
            outputs = self._forward_with_hidden_states(input_embeds)
            sequence_loss, feature_loss, total_loss = self._compute_loss_components(
                outputs,
                input_length=input_embeds.shape[1],
            )
        return (
            float(sequence_loss.mean().item()),
            float(feature_loss.mean().item()),
            float(total_loss.mean().item()),
        )

    def run(
        self,
        messages: Union[str, list[dict[str, str]]],
        target: str,
    ) -> AdaptiveAttackResult:
        """Run GCG and return the best candidate with component losses."""
        result = super().run(messages, target)
        sequence_loss, feature_loss, _ = self._evaluate_best_candidate(
            result.best_string
        )
        return AdaptiveAttackResult(
            best_loss=result.best_loss,
            best_string=result.best_string,
            losses=result.losses,
            strings=result.strings,
            objective_name=getattr(
                self.objective,
                "name",
                type(self.objective).__name__,
            ),
            sequence_loss=sequence_loss,
            feature_loss=feature_loss,
        )


def run_adaptive_attack(
    model: transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizer,
    messages: Union[str, list[dict[str, str]]],
    target: str,
    objective: HiddenStateObjective,
    config: Optional[AdaptiveGCGConfig] = None,
) -> AdaptiveAttackResult:
    """Run one adaptive attack through the functional API."""
    attack_config = config or AdaptiveGCGConfig()
    logger.setLevel(getattr(logging, attack_config.verbosity))
    attacker = AdaptiveGCG(
        model=model,
        tokenizer=tokenizer,
        config=attack_config,
        objective=objective,
    )
    return attacker.run(messages, target)
