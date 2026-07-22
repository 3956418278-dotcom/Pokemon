from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .transactions import (
    Transaction,
    TransactionEvent,
    is_root_decision,
    is_terminal_state,
    state_turn,
)


SEMANTIC_CONCEPT_NAMES = (
    "self_attack_current_turn",
    "opponent_attack_before_next_self_turn",
    "self_attack_next_self_turn",
    "net_prize_swing_current_turn",
    "net_prize_swing_opponent_response",
    "net_prize_swing_next_self_turn",
    "net_prize_swing_self_turns_2_to_3",
    "net_prize_swing_self_turns_4_to_6",
    "terminal_reached_by_end_self_turn_6",
    "terminal_utility_if_reached",
)
NUM_SEMANTIC_CONCEPTS = 10

SELF_ATTACK_CURRENT = 0
OPPONENT_ATTACK_RESPONSE = 1
SELF_ATTACK_NEXT = 2
PRIZE_CURRENT = 3
PRIZE_OPPONENT_RESPONSE = 4
PRIZE_NEXT_SELF = 5
PRIZE_SELF_TURNS_2_TO_3 = 6
PRIZE_SELF_TURNS_4_TO_6 = 7
TERMINAL_REACHED_H6 = 8
TERMINAL_UTILITY_H6 = 9

BINARY_CONCEPT_INDICES = (0, 1, 2, 8)
CONTINUOUS_CONCEPT_INDICES = (3, 4, 5, 6, 7, 9)

SEMANTIC_INTERACTIONS = (
    (SELF_ATTACK_CURRENT, PRIZE_CURRENT),
    (OPPONENT_ATTACK_RESPONSE, PRIZE_OPPONENT_RESPONSE),
    (SELF_ATTACK_NEXT, PRIZE_NEXT_SELF),
)


@dataclass(frozen=True)
class SemanticConceptOutput:
    values: Tensor
    applicable: Tensor
    confidence: Tensor
    ensemble_values: Tensor | None = None


@dataclass(frozen=True)
class SemanticConceptTargets:
    values: Tensor
    applicable: Tensor


@dataclass(frozen=True)
class SemanticPotentialExplanation:
    bias: float
    unary_contributions: dict[str, float]
    interaction_contributions: dict[str, float]
    confidence_gate: float
    pre_tanh_logit: float
    potential: float


@dataclass(frozen=True)
class TurnGroup:
    turn_id: int
    owner_seat: int
    transaction_indices: tuple[int, ...]


@dataclass(frozen=True)
class TransactionWindow:
    transaction_indices: tuple[int, ...]
    applicable: bool
    start_state: Any = None
    end_state: Any = None


@dataclass(frozen=True)
class SemanticTimeWindows:
    current_turn: TransactionWindow
    opponent_response: TransactionWindow
    next_self_turn: TransactionWindow
    self_turns_2_to_3: TransactionWindow
    self_turns_4_to_6: TransactionWindow

    @property
    def prize_windows(self) -> tuple[TransactionWindow, ...]:
        return (
            self.current_turn,
            self.opponent_response,
            self.next_self_turn,
            self.self_turns_2_to_3,
            self.self_turns_4_to_6,
        )


@dataclass(frozen=True)
class SemanticStateFacts:
    seat: int
    turn: int | None
    active_serial: int | None
    living_pokemon_serials: tuple[int, ...]
    self_prize_count: int
    opponent_prize_count: int
    self_deck_count: int | None
    rule_lock_ids: tuple[str, ...]
    armed_delayed_effect_ids: tuple[str, ...]
    recovery_needed: bool


def _as_int(value: Any, default: int | None = None) -> int | None:
    try:
        return default if value is None else int(value)
    except (TypeError, ValueError):
        return default


def _serial(card: Any) -> int | None:
    if not isinstance(card, Mapping):
        return None
    return _as_int(card.get("serial"))


def _pokemon_instance_serial(card: Any) -> int | None:
    if not isinstance(card, Mapping):
        return None
    lineage = [
        serial
        for serial in (_serial(ancestor) for ancestor in (card.get("preEvolution") or []))
        if serial is not None
    ]
    return lineage[0] if lineage else _serial(card)


def _fact_ids(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, Mapping):
        value = value.values()
    if isinstance(value, (str, int)):
        return (str(value),)
    result = []
    for item in value:
        if isinstance(item, Mapping):
            identifier = item.get("effect_id", item.get("id", item.get("serial", item.get("kind"))))
        else:
            identifier = item
        if identifier is not None:
            result.append(str(identifier))
    return tuple(sorted(set(result)))


def _observable_rule_lock_ids(current: Mapping[str, Any], players: Sequence[Any]) -> tuple[str, ...]:
    """Generic in-play rule objects; no concrete Card ID table is involved."""

    identifiers: list[str] = []
    for stadium in current.get("stadium") or []:
        serial = _serial(stadium)
        if serial is not None:
            identifiers.append(f"stadium:{serial}")
    condition_names = ("poisoned", "burned", "asleep", "paralyzed", "confused")
    for player_index, player in enumerate(players):
        if not isinstance(player, Mapping):
            continue
        active = (player.get("active") or [None])[0]
        active_serial = _pokemon_instance_serial(active)
        for condition in condition_names:
            if player.get(condition, False):
                identifiers.append(f"condition:{player_index}:{condition}:{active_serial}")
        for pokemon in (*(player.get("active") or []), *(player.get("bench") or [])):
            if not isinstance(pokemon, Mapping):
                continue
            for tool in pokemon.get("tools") or []:
                serial = _serial(tool)
                if serial is not None:
                    identifiers.append(f"tool:{serial}")
    return tuple(sorted(set(identifiers)))


def semantic_state_facts(state: Mapping[str, Any] | None, seat: int) -> SemanticStateFacts:
    """Extract deterministic applicability facts from a seat-relative state.

    Runtime adapters may supply a ``semantic_facts`` mapping for engine facts
    that are not represented in the public JSON. The model never predicts any
    field consumed here.
    """

    if seat not in (0, 1):
        raise ValueError("seat must be 0 or 1")
    state = state or {}
    current = state.get("current") or {}
    supplied = state.get("semantic_facts") or current.get("semantic_facts") or {}
    players = current.get("players") or ({}, {})
    own = players[seat] if seat < len(players) and isinstance(players[seat], Mapping) else {}
    opponent_seat = 1 - seat
    opponent = (
        players[opponent_seat]
        if opponent_seat < len(players) and isinstance(players[opponent_seat], Mapping)
        else {}
    )
    active_cards = own.get("active") or []
    active_serial = _as_int(supplied.get("active_serial"))
    if active_serial is None and active_cards:
        active_serial = _pokemon_instance_serial(active_cards[0])
    supplied_living = supplied.get("living_pokemon_serials")
    if supplied_living is None:
        living_serials = tuple(
            serial
            for serial in (
                _pokemon_instance_serial(card)
                for card in (*active_cards, *(own.get("bench") or []))
            )
            if serial is not None
        )
    else:
        living_serials = tuple(
            int(serial) for serial in supplied_living if _as_int(serial) is not None
        )

    rule_locks = supplied.get("rule_lock_ids")
    if rule_locks is None:
        rule_locks = (
            own.get("ruleLocks")
            or current.get("ruleLocks")
            or current.get("activeRuleLocks")
            or _observable_rule_lock_ids(current, players)
        )
    armed = supplied.get("armed_delayed_effect_ids")
    if armed is None:
        armed = (
            own.get("armedDelayedEffects")
            or current.get("armedDelayedEffects")
            or current.get("delayedEffects")
            or ()
        )
    blocked = any(
        bool(own.get(name, False))
        for name in ("asleep", "paralyzed", "confused", "blocked")
    )
    recovery_needed = bool(
        supplied.get("recovery_needed", active_serial is None or blocked)
    )
    self_prizes = _as_int(supplied.get("self_prize_count"))
    if self_prizes is None:
        self_prizes = len(own.get("prize") or [])
    opponent_prizes = _as_int(supplied.get("opponent_prize_count"))
    if opponent_prizes is None:
        opponent_prizes = len(opponent.get("prize") or [])
    deck_count = _as_int(supplied.get("self_deck_count", own.get("deckCount")))
    return SemanticStateFacts(
        seat=seat,
        turn=_as_int(supplied.get("turn", current.get("turn"))),
        active_serial=active_serial,
        living_pokemon_serials=living_serials,
        self_prize_count=int(self_prizes or 0),
        opponent_prize_count=int(opponent_prizes or 0),
        self_deck_count=deck_count,
        rule_lock_ids=_fact_ids(rule_locks),
        armed_delayed_effect_ids=_fact_ids(armed),
        recovery_needed=recovery_needed,
    )


def semantic_applicability(state: Mapping[str, Any] | None, seat: int) -> Tensor:
    # All ten coordinates are structurally available to a live model state.
    # Completed-trajectory supervision supplies the narrower temporal mask,
    # including the conditional applicability of terminal utility.
    semantic_state_facts(state, seat)
    return torch.ones(NUM_SEMANTIC_CONCEPTS, dtype=torch.bool)


def _event_kind(event: TransactionEvent) -> str:
    return event.kind.lower().replace("-", "_")


def build_turn_groups(transactions: Sequence[Transaction]) -> tuple[TurnGroup, ...]:
    """Group consecutive transactions by the real simulator turn field."""

    groups: list[TurnGroup] = []
    pending_turn: int | None = None
    pending_indices: list[int] = []

    def finish_pending() -> None:
        if not pending_indices or pending_turn is None:
            return
        root = next(
            (
                transactions[index]
                for index in pending_indices
                if is_root_decision(transactions[index].start_state)
            ),
            transactions[pending_indices[0]],
        )
        groups.append(
            TurnGroup(
                turn_id=pending_turn,
                owner_seat=root.seat,
                transaction_indices=tuple(pending_indices),
            )
        )

    for index, transaction in enumerate(transactions):
        turn_id = state_turn(transaction.start_state)
        if turn_id is None:
            raise ValueError(
                "complete training trajectory is missing current.turn at "
                f"transaction {transaction.transaction_id}"
            )
        if pending_indices and turn_id != pending_turn:
            finish_pending()
            pending_indices = []
        pending_turn = turn_id
        pending_indices.append(index)
    finish_pending()
    return tuple(groups)


class _TrajectoryTimeline:
    def __init__(self, transactions: Sequence[Transaction]) -> None:
        if not transactions:
            raise ValueError("cannot label an empty trajectory")
        self.transactions = transactions
        self.groups = build_turn_groups(transactions)
        self.group_position_by_transaction: dict[int, int] = {}
        for group_position, group in enumerate(self.groups):
            for transaction_index in group.transaction_indices:
                self.group_position_by_transaction[transaction_index] = group_position

        terminal_indices = [
            index
            for index, transaction in enumerate(transactions)
            if is_terminal_state(transaction.end_state)
        ]
        if not terminal_indices:
            terminal_indices = [
                index
                for index, transaction in enumerate(transactions)
                if is_terminal_state(transaction.trajectory_terminal_state)
            ]
        if not terminal_indices:
            terminal_indices = [
                index
                for index, transaction in enumerate(transactions)
                if transaction.terminal
            ]
        if not terminal_indices:
            raise ValueError("complete training trajectory has no terminal boundary")
        self.terminal_index = max(terminal_indices)
        if self.terminal_index != len(transactions) - 1:
            raise ValueError("complete training trajectory contains transactions after terminal")
        terminal_transaction = transactions[self.terminal_index]
        if is_terminal_state(terminal_transaction.end_state):
            self.terminal_state = terminal_transaction.end_state
        else:
            self.terminal_state = (
                terminal_transaction.trajectory_terminal_state
                or terminal_transaction.end_state
            )
        if self.terminal_state is None:
            raise ValueError("terminal transaction has no terminal state")

    def _state_after(self, transaction_index: int) -> Any:
        if transaction_index == self.terminal_index:
            return self.terminal_state
        state = self.transactions[transaction_index].end_state
        if state is None and transaction_index + 1 < len(self.transactions):
            state = self.transactions[transaction_index + 1].start_state
        if state is None:
            raise ValueError(
                "trajectory window has no end state at transaction "
                f"{self.transactions[transaction_index].transaction_id}"
            )
        return state

    def _window(
        self,
        start_index: int,
        end_index: int,
        *,
        empty_after: int | None = None,
    ) -> TransactionWindow:
        if start_index > self.terminal_index:
            return TransactionWindow((), False)
        if end_index < start_index:
            if empty_after is None:
                return TransactionWindow((), False)
            boundary = self._state_after(empty_after)
            return TransactionWindow((), True, boundary, boundary)
        bounded_end = min(end_index, self.terminal_index)
        indices = tuple(range(start_index, bounded_end + 1))
        start_state = self.transactions[start_index].start_state
        if start_state is None:
            raise ValueError(
                "trajectory window has no start state at transaction "
                f"{self.transactions[start_index].transaction_id}"
            )
        return TransactionWindow(
            transaction_indices=indices,
            applicable=True,
            start_state=start_state,
            end_state=self._state_after(bounded_end),
        )

    def windows(self, transaction_index: int) -> SemanticTimeWindows:
        transaction = self.transactions[transaction_index]
        seat = transaction.seat
        group_position = self.group_position_by_transaction[transaction_index]
        current_group = self.groups[group_position]
        current_end = current_group.transaction_indices[-1]
        future_self_groups = [
            position
            for position in range(group_position + 1, len(self.groups))
            if self.groups[position].owner_seat == seat
        ]

        current = self._window(transaction_index, current_end)
        response_start = current_end + 1
        if future_self_groups:
            next_self_start = self.groups[future_self_groups[0]].transaction_indices[0]
            response = self._window(
                response_start,
                next_self_start - 1,
                empty_after=current_end,
            )
            next_self_group = self.groups[future_self_groups[0]]
            next_self = self._window(
                next_self_group.transaction_indices[0],
                next_self_group.transaction_indices[-1],
            )
            turns_2_to_3_end = (
                self.groups[future_self_groups[2]].transaction_indices[-1]
                if len(future_self_groups) >= 3
                else self.terminal_index
            )
            turns_2_to_3 = self._window(
                next_self_group.transaction_indices[-1] + 1,
                turns_2_to_3_end,
            )
        else:
            response = self._window(response_start, self.terminal_index)
            next_self = TransactionWindow((), False)
            turns_2_to_3 = TransactionWindow((), False)

        if len(future_self_groups) >= 3:
            third_self_end = self.groups[future_self_groups[2]].transaction_indices[-1]
            turns_4_to_6_end = (
                self.groups[future_self_groups[5]].transaction_indices[-1]
                if len(future_self_groups) >= 6
                else self.terminal_index
            )
            turns_4_to_6 = self._window(third_self_end + 1, turns_4_to_6_end)
        else:
            turns_4_to_6 = TransactionWindow((), False)

        return SemanticTimeWindows(
            current_turn=current,
            opponent_response=response,
            next_self_turn=next_self,
            self_turns_2_to_3=turns_2_to_3,
            self_turns_4_to_6=turns_4_to_6,
        )

    def terminal_reached_h6(self, transaction_index: int) -> bool:
        seat = self.transactions[transaction_index].seat
        group_position = self.group_position_by_transaction[transaction_index]
        future_self_groups = [
            position
            for position in range(group_position + 1, len(self.groups))
            if self.groups[position].owner_seat == seat
        ]
        if len(future_self_groups) < 6:
            return True
        sixth_self_end = self.groups[future_self_groups[5]].transaction_indices[-1]
        return self.terminal_index <= sixth_self_end

    def terminal_utility(self, seat: int) -> float:
        current = (
            self.terminal_state.get("current") or {}
            if isinstance(self.terminal_state, Mapping)
            else {}
        )
        winner = _as_int(current.get("result"))
        if winner in (0, 1, 2):
            if winner == 2:
                return 0.0
            return 1.0 if winner == seat else -1.0
        seat_terminals = [
            transaction
            for transaction in self.transactions
            if transaction.seat == seat and transaction.terminal
        ]
        if not seat_terminals or seat_terminals[-1].outcome not in (-1, 0, 1):
            raise ValueError("terminal trajectory has no valid seat-relative utility")
        return float(seat_terminals[-1].outcome)


class TrajectoryLabelBuilder:
    """Build the fixed ten concepts from a completed transactional game."""

    ATTACK_KINDS = {"attack_executed", "15"}

    @staticmethod
    def time_windows(
        transactions: Sequence[Transaction],
        index: int,
    ) -> SemanticTimeWindows:
        return _TrajectoryTimeline(transactions).windows(index)

    def build(self, transactions: Sequence[Transaction]) -> SemanticConceptTargets:
        timeline = _TrajectoryTimeline(transactions)
        values = torch.zeros((len(transactions), NUM_SEMANTIC_CONCEPTS), dtype=torch.float32)
        applicable = torch.zeros_like(values, dtype=torch.bool)
        for index, transaction in enumerate(transactions):
            result, mask = self._build_one(timeline, index)
            values[index] = result
            applicable[index] = mask
            transaction.semantic_target_values = values[index].clone()
            transaction.semantic_target_applicable = mask.clone()
            if transaction.semantic_applicable is None:
                transaction.semantic_applicable = semantic_applicability(
                    transaction.start_state,
                    transaction.seat,
                )
            elif transaction.semantic_applicable.shape != (NUM_SEMANTIC_CONCEPTS,):
                raise ValueError(
                    "transaction semantic applicability does not match the v3 "
                    "ten-dimensional schema"
                )
        return SemanticConceptTargets(values=values, applicable=applicable)

    def _build_one(
        self,
        timeline: _TrajectoryTimeline,
        index: int,
    ) -> tuple[Tensor, Tensor]:
        transaction = timeline.transactions[index]
        seat = transaction.seat
        result = torch.zeros(NUM_SEMANTIC_CONCEPTS, dtype=torch.float32)
        mask = torch.zeros(NUM_SEMANTIC_CONCEPTS, dtype=torch.bool)
        windows = timeline.windows(index)

        for concept_index, window, event_seat in (
            (SELF_ATTACK_CURRENT, windows.current_turn, seat),
            (OPPONENT_ATTACK_RESPONSE, windows.opponent_response, 1 - seat),
            (SELF_ATTACK_NEXT, windows.next_self_turn, seat),
        ):
            mask[concept_index] = window.applicable
            if window.applicable:
                result[concept_index] = float(
                    self._window_has_attack(timeline, window, event_seat)
                )

        for concept_index, window in zip(
            (
                PRIZE_CURRENT,
                PRIZE_OPPONENT_RESPONSE,
                PRIZE_NEXT_SELF,
                PRIZE_SELF_TURNS_2_TO_3,
                PRIZE_SELF_TURNS_4_TO_6,
            ),
            windows.prize_windows,
        ):
            mask[concept_index] = window.applicable
            if window.applicable:
                result[concept_index] = self._net_prize_swing(window, seat)

        reached_terminal = timeline.terminal_reached_h6(index)
        result[TERMINAL_REACHED_H6] = float(reached_terminal)
        mask[TERMINAL_REACHED_H6] = True
        if reached_terminal:
            result[TERMINAL_UTILITY_H6] = timeline.terminal_utility(seat)
            mask[TERMINAL_UTILITY_H6] = True
        return result, mask

    def _window_has_attack(
        self,
        timeline: _TrajectoryTimeline,
        window: TransactionWindow,
        seat: int,
    ) -> bool:
        for transaction_index in window.transaction_indices:
            transaction = timeline.transactions[transaction_index]
            for event in transaction.event_records:
                if event.seat not in (None, seat):
                    continue
                if _event_kind(event) in self.ATTACK_KINDS:
                    return True
        return False

    @staticmethod
    def _net_prize_swing(window: TransactionWindow, seat: int) -> float:
        if window.start_state is None or window.end_state is None:
            raise ValueError("applicable prize window is missing a state boundary")
        start = semantic_state_facts(window.start_state, seat)
        end = semantic_state_facts(window.end_state, seat)
        self_gain = start.self_prize_count - end.self_prize_count
        opponent_gain = start.opponent_prize_count - end.opponent_prize_count
        return max(-1.0, min(1.0, (self_gain - opponent_gain) / 6.0))


class SemanticConceptHeads(nn.Module):
    """Small ensemble whose disagreement supplies non-gameable confidence."""

    def __init__(self, hidden_dim: int, ensemble_size: int = 3) -> None:
        super().__init__()
        if ensemble_size < 2:
            raise ValueError("semantic confidence requires at least two ensemble members")
        self.members = nn.ModuleList(
            [nn.Linear(hidden_dim, NUM_SEMANTIC_CONCEPTS) for _ in range(ensemble_size)]
        )
        self.register_buffer("holdout_reliability", torch.ones(NUM_SEMANTIC_CONCEPTS))

    def forward(self, encoded: Tensor, applicable: Tensor) -> SemanticConceptOutput:
        if applicable.shape != encoded.shape[:-1] + (NUM_SEMANTIC_CONCEPTS,):
            raise ValueError(
                f"applicable must end in {NUM_SEMANTIC_CONCEPTS} semantic concepts"
            )
        logits = torch.stack([member(encoded) for member in self.members], dim=0)
        binary_index = torch.tensor(BINARY_CONCEPT_INDICES, device=encoded.device)
        continuous_index = torch.tensor(CONTINUOUS_CONCEPT_INDICES, device=encoded.device)
        predictions = torch.empty_like(logits)
        predictions[..., binary_index] = torch.sigmoid(logits[..., binary_index])
        predictions[..., continuous_index] = torch.tanh(logits[..., continuous_index])
        values = predictions.mean(dim=0)
        disagreement = predictions.var(dim=0, unbiased=False)
        confidence = torch.exp(-4.0 * disagreement) * self.holdout_reliability.clamp(0.0, 1.0)
        confidence = confidence.clamp(0.0, 1.0).detach()
        return SemanticConceptOutput(
            values=values,
            applicable=applicable.to(dtype=torch.bool),
            confidence=confidence,
            ensemble_values=predictions,
        )

    @torch.no_grad()
    def set_holdout_reliability(self, reliability: Tensor) -> None:
        if reliability.shape != (NUM_SEMANTIC_CONCEPTS,):
            raise ValueError(
                f"reliability must contain {NUM_SEMANTIC_CONCEPTS} values"
            )
        self.holdout_reliability.copy_(reliability.clamp(0.0, 1.0))


def semantic_concept_loss(output: SemanticConceptOutput, targets: SemanticConceptTargets) -> Tensor:
    if output.values.shape != targets.values.shape or output.values.shape != targets.applicable.shape:
        raise ValueError("semantic prediction and target shapes must match")
    # Applicability is an engine fact. The target mask is authoritative and no
    # predicted mask exists anywhere in this loss.
    mask = targets.applicable.to(dtype=output.values.dtype)
    binary_index = list(BINARY_CONCEPT_INDICES)
    continuous_index = list(CONTINUOUS_CONCEPT_INDICES)
    predictions = (
        output.values.unsqueeze(0)
        if output.ensemble_values is None
        else output.ensemble_values
    )
    expanded_targets = targets.values.unsqueeze(0).expand_as(predictions)
    binary_loss = F.binary_cross_entropy(
        predictions[..., binary_index].clamp(1e-6, 1.0 - 1e-6),
        expanded_targets[..., binary_index],
        reduction="none",
    ).mean(dim=0)
    continuous_loss = F.huber_loss(
        predictions[..., continuous_index],
        expanded_targets[..., continuous_index],
        reduction="none",
    ).mean(dim=0)
    losses = torch.zeros_like(output.values)
    losses[..., binary_index] = binary_loss
    losses[..., continuous_index] = continuous_loss
    denominator = mask.sum().clamp_min(1.0)
    return (losses * mask).sum() / denominator


class PiecewiseLinearFunction(nn.Module):
    def __init__(self, knot_count: int = 5) -> None:
        super().__init__()
        if knot_count < 2:
            raise ValueError("knot_count must be at least two")
        self.knot_values = nn.Parameter(torch.zeros(knot_count))

    def forward(self, value: Tensor) -> Tensor:
        scaled = ((value.clamp(-1.0, 1.0) + 1.0) * 0.5) * (len(self.knot_values) - 1)
        left = scaled.floor().long().clamp(0, len(self.knot_values) - 2)
        fraction = scaled - left.to(scaled.dtype)
        lower = self.knot_values[left]
        upper = self.knot_values[left + 1]
        return lower + fraction * (upper - lower)


class SemanticPotentialHead(nn.Module):
    def __init__(self, *, knot_count: int = 5, potential_clip: float = 0.8) -> None:
        super().__init__()
        self.unary_functions = nn.ModuleList(
            [PiecewiseLinearFunction(knot_count) for _ in range(NUM_SEMANTIC_CONCEPTS)]
        )
        self.bias = nn.Parameter(torch.zeros(()))
        self.interaction_weights = nn.Parameter(torch.zeros(len(SEMANTIC_INTERACTIONS)))
        self.potential_clip = float(potential_clip)

    def components(
        self,
        values: Tensor,
        applicable: Tensor,
        confidence: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if values.shape[-1] != NUM_SEMANTIC_CONCEPTS:
            raise ValueError(
                f"semantic values must have {NUM_SEMANTIC_CONCEPTS} fixed positions"
            )
        if applicable.shape != values.shape or confidence.shape != values.shape:
            raise ValueError("values, applicable and confidence must have identical shapes")
        mask = applicable.to(dtype=values.dtype)
        detached_confidence = confidence.detach().to(dtype=values.dtype)
        unary = torch.stack(
            [function(values[..., index]) for index, function in enumerate(self.unary_functions)],
            dim=-1,
        )
        unary = unary * mask * detached_confidence
        interactions = torch.stack(
            [
                self.interaction_weights[index] * unary[..., left] * unary[..., right]
                for index, (left, right) in enumerate(SEMANTIC_INTERACTIONS)
            ],
            dim=-1,
        )
        logit = self.bias + unary.sum(dim=-1) + interactions.sum(dim=-1)
        gate = (mask * detached_confidence).sum(dim=-1) / mask.sum(dim=-1).clamp_min(1.0)
        return unary, interactions, logit, gate

    def forward(
        self,
        values: Tensor,
        applicable: Tensor,
        confidence: Tensor,
        *,
        terminal: Tensor | None = None,
    ) -> Tensor:
        _, _, logit, gate = self.components(values, applicable, confidence)
        potential = (gate * torch.tanh(logit)).clamp(-self.potential_clip, self.potential_clip)
        if terminal is not None:
            potential = torch.where(terminal.to(dtype=torch.bool), torch.zeros_like(potential), potential)
        return potential

    @torch.no_grad()
    def explain(
        self,
        values: Tensor,
        applicable: Tensor,
        confidence: Tensor,
    ) -> SemanticPotentialExplanation:
        if values.ndim == 1:
            values = values.unsqueeze(0)
            applicable = applicable.unsqueeze(0)
            confidence = confidence.unsqueeze(0)
        if values.shape[0] != 1:
            raise ValueError("explain accepts exactly one semantic state")
        unary, interactions, logit, gate = self.components(values, applicable, confidence)
        potential = self.forward(values, applicable, confidence)
        unary_values = {
            name: float(unary[0, index].cpu())
            for index, name in enumerate(SEMANTIC_CONCEPT_NAMES)
        }
        interaction_values = {
            f"{SEMANTIC_CONCEPT_NAMES[left]}__x__{SEMANTIC_CONCEPT_NAMES[right]}": float(
                interactions[0, index].cpu()
            )
            for index, (left, right) in enumerate(SEMANTIC_INTERACTIONS)
        }
        return SemanticPotentialExplanation(
            bias=float(self.bias.cpu()),
            unary_contributions=unary_values,
            interaction_contributions=interaction_values,
            confidence_gate=float(gate[0].cpu()),
            pre_tanh_logit=float(logit[0].cpu()),
            potential=float(potential[0].cpu()),
        )
