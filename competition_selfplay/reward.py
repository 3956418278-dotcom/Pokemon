from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

from .config import RewardConfig


class TerminalReason(IntEnum):
    PRIZES = 1
    DECK_OUT = 2
    NO_ACTIVE_POKEMON = 3
    CARD_EFFECT = 4


@dataclass(frozen=True)
class BattleSnapshot:
    """Minimal shaping state; deck and hand counts are intentionally absent."""

    turn: int
    self_prize_count: int
    opponent_prize_count: int
    self_active_count: int
    self_bench_count: int
    self_attached_energy_count: int
    opponent_active_count: int
    opponent_bench_count: int


@dataclass(frozen=True)
class RewardVector:
    """[terminal outcome, prize race, setup tempo]."""

    outcome: float
    prize_progress: float
    setup_tempo: float

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.outcome, self.prize_progress, self.setup_tempo)

    def scalarize(self, weights: tuple[float, float, float]) -> float:
        return sum(value * weight for value, weight in zip(self.as_tuple(), weights))


class VectorReward:
    def __init__(self, config: RewardConfig) -> None:
        self.config = config

    def _setup_potential(self, state: BattleSnapshot) -> float:
        cfg = self.config
        return (
            state.self_active_count * cfg.own_active_weight
            + state.self_bench_count * cfg.own_bench_weight
            + state.self_attached_energy_count * cfg.own_energy_weight
            + state.opponent_active_count * cfg.opponent_active_weight
            + state.opponent_bench_count * cfg.opponent_bench_weight
        )

    def transition(
        self,
        before: BattleSnapshot,
        after: BattleSnapshot,
        *,
        perspective_player: int,
        winner: int | None = None,
        terminal_reason: TerminalReason | int | None = None,
    ) -> RewardVector:
        if perspective_player not in (0, 1):
            raise ValueError("perspective_player must be 0 or 1")
        cfg = self.config
        prize_gain = before.opponent_prize_count - after.opponent_prize_count
        prize_loss = before.self_prize_count - after.self_prize_count
        prize_progress = (prize_gain - prize_loss) * cfg.prize_scale
        setup_delta = self._setup_potential(after) - self._setup_potential(before)
        setup_delta = max(-cfg.setup_delta_clip, min(cfg.setup_delta_clip, setup_delta))

        outcome = 0.0
        if winner is not None:
            reason = TerminalReason(terminal_reason) if terminal_reason is not None else None
            if winner == 2:
                outcome = cfg.draw
            elif winner == perspective_player:
                # A win caused solely by the opponent failing to draw is not a
                # learnable shaping target for this deck's own behaviour.
                outcome = cfg.opponent_deck_out_win if reason is TerminalReason.DECK_OUT else cfg.win
            else:
                # Self deck-out is controllable enough to deserve its explicit
                # penalty even though opponent deck remaining is never observed.
                outcome = cfg.self_deck_out_loss if reason is TerminalReason.DECK_OUT else cfg.loss
        return RewardVector(outcome, prize_progress, setup_delta)
