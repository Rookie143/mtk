"""Reusable adaptive GCG attack modules for MTK-aware evaluation."""

from .config import AdaptiveGCGConfig
from .engine import AdaptiveAttackResult, AdaptiveGCG, run_adaptive_attack
from .feature_store import HiddenStateLibrary, load_hidden_state_library
from .mtk import (
    MTKAdaptiveObjective,
    KNNRankEncoder,
    MTKDetector,
    TorchMTKScorer,
    run_mtk_attack,
)

__all__ = [
    "AdaptiveAttackResult",
    "AdaptiveGCG",
    "AdaptiveGCGConfig",
    "HiddenStateLibrary",
    "MTKAdaptiveObjective",
    "KNNRankEncoder",
    "MTKDetector",
    "TorchMTKScorer",
    "load_hidden_state_library",
    "run_adaptive_attack",
    "run_mtk_attack",
]
