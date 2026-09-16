"""Minimal tests for MTK adaptive attacks, losses, and feature libraries."""

from __future__ import annotations

import pytest
from pathlib import Path


torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from torch import Tensor

from adaptive_attack.config import AdaptiveGCGConfig
from adaptive_attack.feature_store import load_hidden_state_library
from adaptive_attack.mtk import (
    MTKAdaptiveObjective,
    MTKDetector,
    TorchMTKScorer,
)
from adaptive_attack.losses import target_sequence_loss


def _hidden_states(
    batch_size: int = 2,
    sequence_length: int = 5,
    hidden_size: int = 4,
) -> list[Tensor]:
    """Create test hidden states for two Transformer layers."""
    return [
        torch.randn(
            batch_size,
            sequence_length,
            hidden_size,
            requires_grad=True,
        )
        for _ in range(2)
    ]


def test_target_sequence_loss_returns_one_value_per_candidate() -> None:
    """Target sequence loss should preserve the candidate batch dimension."""
    logits = torch.randn(3, 6, 11)
    target_ids = torch.tensor([[1, 2]])

    loss = target_sequence_loss(logits, target_ids, input_length=6)

    assert loss.shape == (3,)
    assert torch.all(torch.isfinite(loss))


def test_mtk_and_rank_detector() -> None:
    """The hard detector and rank encoder should handle a small feature library."""
    features = torch.randn(8, 3, 2)
    labels = torch.tensor([1, 1, 1, 1, 0, 0, 0, 0])
    scorer = TorchMTKScorer(n_estimators=5, random_state=7)
    scorer.fit(torch.randn(8, 3))

    assert scorer.decision_function(torch.randn(2, 3)).shape == (2,)
    assert scorer.predict(torch.randn(2, 3)).shape == (2,)
    quantile_scorer = TorchMTKScorer(
        n_estimators=5,
        contamination=0.2,
        random_state=7,
    )
    quantile_scorer.fit(torch.randn(8, 3))
    assert quantile_scorer.decision_function(torch.randn(2, 3)).shape == (2,)

    detector = MTKDetector(
        background_features=features,
        background_labels=labels,
        k=2,
        n_estimators=5,
        random_state=7,
    )
    assert detector.decision_function(features[:2]).shape == (2,)
    assert detector.predict(features[:2]).shape == (2,)


def test_mtk_losses_are_differentiable() -> None:
    """MTK L1/L2/L3 surrogates must preserve batch shape and gradients."""
    benign = torch.zeros(3, 2, 4)
    malicious = torch.ones(3, 2, 4) * 4.0

    for loss_type in ("l1", "l2", "l3"):
        query = [torch.ones(2, 5, 4, requires_grad=True) for _ in range(2)]
        objective = MTKAdaptiveObjective(
            benign_features=benign,
            malicious_features=malicious,
            loss_type=loss_type,
        )

        loss = objective(query, last_token_index=4)
        loss.sum().backward()

        assert loss.shape == (2,)
        assert torch.all(torch.isfinite(loss))
        assert query[-1].grad is not None
        assert objective.name == "mtk_%s" % loss_type


def test_mtk_surrogate_matches_paper_directions() -> None:
    """MTK L1/L2/L3 should pull toward benign states and away from malicious ones."""
    benign = torch.tensor(
        [
            [[0.0, 0.0], [0.0, 0.0]],
            [[2.0, 0.0], [2.0, 0.0]],
        ]
    )
    malicious = torch.tensor(
        [
            [[5.0, 0.0], [5.0, 0.0]],
            [[6.0, 0.0], [6.0, 0.0]],
        ]
    )
    near_benign = [torch.zeros(1, 3, 2, requires_grad=True) for _ in range(2)]
    near_malicious = [torch.full((1, 3, 2), 5.5, requires_grad=True) for _ in range(2)]

    for loss_type in ("l1", "l2", "l3"):
        objective = MTKAdaptiveObjective(
            benign_features=benign,
            malicious_features=malicious,
            loss_type=loss_type,
        )

        benign_like_loss = objective(near_benign, last_token_index=2)
        malicious_like_loss = objective(near_malicious, last_token_index=2)

        assert benign_like_loss.item() < malicious_like_loss.item()


def test_feature_library_loader_supports_original_dict_format(tmp_path: Path) -> None:
    """The feature-library loader should support the original dictionary format."""
    path = tmp_path / "features.pt"
    torch.save(
        {
            "background_layered_activations": torch.randn(4, 2, 3),
            "labels": torch.tensor([1, 1, 0, 0]),
        },
        path,
    )

    library = load_hidden_state_library(path)

    assert library.features.shape == (4, 2, 3)
    assert library.labels is not None
    assert library.labels.tolist() == [1, 1, 0, 0]


def test_adaptive_config_rejects_empty_loss() -> None:
    """The joint loss cannot have both weights set to zero."""
    with pytest.raises(ValueError):
        AdaptiveGCGConfig(sequence_weight=0.0, feature_weight=0.0)
