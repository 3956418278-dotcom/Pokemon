from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Iterable, Sequence

import torch
from torch import Tensor

from .config import RewardConfig
from .transactions import Transaction


class TerminalReason(IntEnum):
    PRIZES = 1
    DECK_OUT = 2
    NO_ACTIVE_POKEMON = 3
    CARD_EFFECT = 4


@dataclass(frozen=True)
class TransactionReward:
    terminal_outcome: float
    shaping: float

    @property
    def total(self) -> float:
        return self.terminal_outcome + self.shaping


def perspective_outcome(
    *,
    perspective_player: int,
    winner: int | None,
    config: RewardConfig,
) -> float:
    if perspective_player not in (0, 1):
        raise ValueError("perspective_player must be 0 or 1")
    if winner is None:
        return 0.0
    if winner == 2:
        return config.terminal_draw
    if winner not in (0, 1):
        raise ValueError("winner must be 0, 1, 2 (draw), or None")
    return config.terminal_win if winner == perspective_player else config.terminal_loss


def transaction_reward(
    *,
    terminal: bool,
    outcome: float,
    target_phi_before: float,
    target_phi_after: float,
    gamma: float,
    shaping_alpha: float,
    potential_clip: float = 0.8,
) -> TransactionReward:
    before = max(-potential_clip, min(potential_clip, float(target_phi_before)))
    after = 0.0 if terminal else max(
        -potential_clip, min(potential_clip, float(target_phi_after))
    )
    terminal_value = float(outcome) if terminal else 0.0
    shaping = float(shaping_alpha) * (float(gamma) * after - before)
    return TransactionReward(terminal_value, shaping)


def rewards_from_transactions(
    transactions: Sequence[Transaction],
    *,
    gamma: float,
    shaping_alpha: float,
    potential_clip: float = 0.8,
) -> Tensor:
    rewards = [
        transaction_reward(
            terminal=transaction.terminal,
            outcome=transaction.outcome,
            target_phi_before=transaction.target_phi_before,
            target_phi_after=transaction.target_phi_after,
            gamma=gamma,
            shaping_alpha=shaping_alpha,
            potential_clip=potential_clip,
        ).total
        for transaction in transactions
    ]
    return torch.tensor(rewards, dtype=torch.float32)


def transaction_gae(
    rewards: Tensor | Sequence[float],
    values: Tensor | Sequence[float],
    terminals: Tensor | Sequence[bool],
    *,
    gamma: float,
    gae_lambda: float,
    next_values: Tensor | Sequence[float] | None = None,
) -> tuple[Tensor, Tensor]:
    """GAE over transaction steps. Nested simulator selects never appear here."""

    rewards_t = torch.as_tensor(rewards, dtype=torch.float32)
    values_t = torch.as_tensor(values, dtype=torch.float32, device=rewards_t.device)
    terminals_t = torch.as_tensor(terminals, dtype=torch.bool, device=rewards_t.device)
    if rewards_t.ndim != 1 or values_t.shape != rewards_t.shape or terminals_t.shape != rewards_t.shape:
        raise ValueError("rewards, values, and terminals must be equal-length vectors")
    if next_values is None:
        next_values_t = torch.cat((values_t[1:], values_t.new_zeros(1)))
    else:
        next_values_t = torch.as_tensor(next_values, dtype=torch.float32, device=rewards_t.device)
        if next_values_t.shape != rewards_t.shape:
            raise ValueError("next_values must match rewards")

    advantages = torch.zeros_like(rewards_t)
    running = rewards_t.new_zeros(())
    for index in range(len(rewards_t) - 1, -1, -1):
        continuation = (~terminals_t[index]).to(rewards_t.dtype)
        delta = (
            rewards_t[index]
            + gamma * continuation * next_values_t[index]
            - values_t[index]
        )
        running = delta + gamma * gae_lambda * continuation * running
        advantages[index] = running
    return advantages, advantages + values_t


def discounted_outcome_returns(
    transactions: Sequence[Transaction],
    *,
    gamma: float,
) -> Tensor:
    """Build the locked ``G_k = gamma**(K-1-k) * z`` target per seat."""

    result = torch.zeros(len(transactions), dtype=torch.float32)
    for seat in (0, 1):
        segment: list[int] = []
        for index, transaction in enumerate(transactions):
            if transaction.seat != seat:
                continue
            segment.append(index)
            if not transaction.terminal:
                continue
            terminal_utility = float(transaction.outcome)
            count = len(segment)
            for own_index, transaction_index in enumerate(segment):
                result[transaction_index] = (
                    gamma ** (count - 1 - own_index)
                ) * terminal_utility
            segment = []
        if segment:
            raise ValueError("each seat trajectory must end in a terminal transaction")
    return result


def mark_terminal_outcomes(
    transactions: Iterable[Transaction],
    *,
    winner: int | None,
    config: RewardConfig,
) -> None:
    """Close each seat's final temporal edge at the common terminal state."""

    by_seat: dict[int, list[Transaction]] = {0: [], 1: []}
    for transaction in transactions:
        by_seat[transaction.seat].append(transaction)
    for seat, seat_transactions in by_seat.items():
        if not seat_transactions:
            continue
        for transaction in seat_transactions:
            transaction.terminal = False
            transaction.outcome = 0
        final = seat_transactions[-1]
        final.terminal = True
        final.outcome = int(perspective_outcome(
            perspective_player=seat,
            winner=winner,
            config=config,
        ))
        final.target_phi_after = 0.0
