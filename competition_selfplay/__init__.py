"""Fixed-deck, competition-oriented self-play primitives."""

from .config import SelfPlayConfig, load_config
from .league import EvaluationResult, LeagueController, LeagueState
from .reward import BattleSnapshot, RewardVector, TerminalReason, VectorReward

__all__ = [
    "BattleSnapshot",
    "EvaluationResult",
    "LeagueController",
    "LeagueState",
    "RewardVector",
    "SelfPlayConfig",
    "TerminalReason",
    "VectorReward",
    "load_config",
]
