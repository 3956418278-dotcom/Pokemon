"""Causal transaction-level semantic self-play training primitives."""

from .config import SCHEMA_VERSION, SelfPlayConfig, load_config
from .league import EvaluationResult, LeagueController, LeagueState
from .model import SemanticSelfPlayModel, TargetSemanticRewardModule
from .phase import CalibrationMetrics, PhaseController, TrainingPhase
from .reward import TerminalReason, TransactionReward, transaction_gae, transaction_reward
from .semantic import (
    NUM_SEMANTIC_CONCEPTS,
    SEMANTIC_CONCEPT_NAMES,
    SemanticConceptOutput,
    SemanticConceptTargets,
    SemanticPotentialExplanation,
    TrajectoryLabelBuilder,
)
from .transactions import CausalEventLink, Transaction, TransactionAssembler

__all__ = [
    "CalibrationMetrics",
    "CausalEventLink",
    "EvaluationResult",
    "LeagueController",
    "LeagueState",
    "NUM_SEMANTIC_CONCEPTS",
    "PhaseController",
    "SCHEMA_VERSION",
    "SEMANTIC_CONCEPT_NAMES",
    "SemanticConceptOutput",
    "SemanticConceptTargets",
    "SemanticPotentialExplanation",
    "SemanticSelfPlayModel",
    "SelfPlayConfig",
    "TargetSemanticRewardModule",
    "TerminalReason",
    "TrainingPhase",
    "Transaction",
    "TransactionAssembler",
    "TransactionReward",
    "TrajectoryLabelBuilder",
    "load_config",
    "transaction_gae",
    "transaction_reward",
]
