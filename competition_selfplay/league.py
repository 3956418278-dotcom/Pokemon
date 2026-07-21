from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .config import PromotionConfig


class LeagueAction(str, Enum):
    KEEP_TRAINING = "KEEP_TRAINING"
    PROMOTE_AND_COPY = "PROMOTE_AND_COPY"
    FREEZE_CONVERGED = "FREEZE_CONVERGED"


@dataclass(frozen=True)
class EvaluationResult:
    wins: int
    losses: int
    draws: int

    @property
    def games(self) -> int:
        return self.wins + self.losses + self.draws

    def score_rate(self, draws_as_half_win: bool) -> float:
        if self.games == 0:
            return 0.0
        draw_value = 0.5 if draws_as_half_win else 0.0
        return (self.wins + draw_value * self.draws) / self.games


@dataclass(frozen=True)
class LeagueState:
    generation: int = 0
    learner_revision: int = 0
    frozen_opponent_revision: int = 0
    promotions: int = 0
    evaluations_without_promotion: int = 0
    converged: bool = False


@dataclass(frozen=True)
class LeagueDecision:
    action: LeagueAction
    state: LeagueState
    score_rate: float
    reason: str


class LeagueController:
    """Learner moves; opponent stays frozen until threshold promotion."""

    def __init__(self, config: PromotionConfig) -> None:
        self.config = config

    def record_training_iteration(self, state: LeagueState) -> LeagueState:
        if state.converged:
            return state
        return LeagueState(
            generation=state.generation,
            learner_revision=state.learner_revision + 1,
            frozen_opponent_revision=state.frozen_opponent_revision,
            promotions=state.promotions,
            evaluations_without_promotion=state.evaluations_without_promotion,
            converged=False,
        )

    def evaluate(self, state: LeagueState, result: EvaluationResult) -> LeagueDecision:
        if state.converged:
            return LeagueDecision(LeagueAction.FREEZE_CONVERGED, state, 0.0, "already converged")
        if result.games < self.config.minimum_games:
            raise ValueError(
                f"evaluation has {result.games} games; minimum is {self.config.minimum_games}"
            )
        score_rate = result.score_rate(self.config.draws_as_half_win)
        if score_rate > self.config.win_rate_threshold:
            promotions = state.promotions + 1
            converged = promotions >= self.config.max_promotions
            promoted = LeagueState(
                generation=state.generation + 1,
                learner_revision=state.learner_revision,
                frozen_opponent_revision=state.learner_revision,
                promotions=promotions,
                evaluations_without_promotion=0,
                converged=converged,
            )
            action = LeagueAction.FREEZE_CONVERGED if converged else LeagueAction.PROMOTE_AND_COPY
            return LeagueDecision(action, promoted, score_rate, "learner exceeded promotion threshold")

        misses = state.evaluations_without_promotion + 1
        converged = misses >= self.config.max_evaluations_without_promotion
        retained = LeagueState(
            generation=state.generation,
            learner_revision=state.learner_revision,
            frozen_opponent_revision=state.frozen_opponent_revision,
            promotions=state.promotions,
            evaluations_without_promotion=misses,
            converged=converged,
        )
        action = LeagueAction.FREEZE_CONVERGED if converged else LeagueAction.KEEP_TRAINING
        reason = "promotion stalled; freeze current opponent" if converged else "threshold not reached"
        return LeagueDecision(action, retained, score_rate, reason)
