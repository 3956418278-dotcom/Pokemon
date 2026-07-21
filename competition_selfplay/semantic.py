from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .transactions import Transaction, TransactionEvent


SEMANTIC_CONCEPT_NAMES = (
    "attack_available_next_own_turn",
    "active_survival_to_next_own_turn",
    "net_prize_gain_h1",
    "net_prize_gain_h3",
    "net_prize_gain_h6",
    "self_deckout_risk_h6",
    "rule_lock_persists_to_opponent_turn",
    "recovery_path_realized_next_own_turn",
    "armed_delayed_trigger_realized",
)
NUM_SEMANTIC_CONCEPTS = 9

ATTACK_AVAILABLE = 0
ACTIVE_SURVIVAL = 1
NET_PRIZE_H1 = 2
NET_PRIZE_H3 = 3
NET_PRIZE_H6 = 4
SELF_DECKOUT = 5
RULE_LOCK = 6
RECOVERY_PATH = 7
DELAYED_TRIGGER = 8

BINARY_CONCEPT_INDICES = (0, 1, 5, 6, 7, 8)
CONTINUOUS_CONCEPT_INDICES = (2, 3, 4)

SEMANTIC_INTERACTIONS = (
    (ATTACK_AVAILABLE, NET_PRIZE_H1),
    (ACTIVE_SURVIVAL, NET_PRIZE_H3),
    (RECOVERY_PATH, ACTIVE_SURVIVAL),
    (RULE_LOCK, NET_PRIZE_H3),
    (DELAYED_TRIGGER, NET_PRIZE_H3),
    (SELF_DECKOUT, NET_PRIZE_H6),
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
    facts = semantic_state_facts(state, seat)
    mask = torch.ones(NUM_SEMANTIC_CONCEPTS, dtype=torch.bool)
    mask[ACTIVE_SURVIVAL] = facts.active_serial is not None
    mask[RULE_LOCK] = bool(facts.rule_lock_ids)
    mask[RECOVERY_PATH] = facts.recovery_needed
    mask[DELAYED_TRIGGER] = bool(facts.armed_delayed_effect_ids)
    return mask


def _event_kind(event: TransactionEvent) -> str:
    return event.kind.lower().replace("-", "_")


def _has_event(transactions: Iterable[Transaction], kinds: set[str], *, seat: int) -> bool:
    for transaction in transactions:
        for event in transaction.event_records:
            if event.seat not in (None, seat):
                continue
            if _event_kind(event) in kinds:
                return True
    return False


class TrajectoryLabelBuilder:
    """Build all concept targets by looking back over a completed game."""

    ATTACK_KINDS = {"attack", "attack_executed", "15"}
    DECKOUT_KINDS = {"self_deckout", "deck_out", "deckout"}
    RECOVERY_KINDS = {"recovery_path_realized", "recovery_realized"}
    DELAYED_KINDS = {"delayed_trigger", "armed_delayed_trigger_realized"}

    def build(self, transactions: Sequence[Transaction]) -> SemanticConceptTargets:
        values = torch.zeros((len(transactions), NUM_SEMANTIC_CONCEPTS), dtype=torch.float32)
        applicable = torch.zeros_like(values, dtype=torch.bool)
        for index, transaction in enumerate(transactions):
            mask = semantic_applicability(transaction.start_state, transaction.seat)
            applicable[index] = mask
            values[index] = self._build_one(transactions, index)
            transaction.semantic_target_values = values[index].clone()
            transaction.semantic_target_applicable = mask.clone()
            if transaction.semantic_applicable is None:
                transaction.semantic_applicable = mask.clone()
        return SemanticConceptTargets(values=values, applicable=applicable)

    def _build_one(self, transactions: Sequence[Transaction], index: int) -> Tensor:
        transaction = transactions[index]
        seat = transaction.seat
        start = semantic_state_facts(transaction.start_state, seat)
        own_indices = [
            candidate
            for candidate in range(index, len(transactions))
            if transactions[candidate].seat == seat
        ]
        result = torch.zeros(NUM_SEMANTIC_CONCEPTS, dtype=torch.float32)

        first_window = [transactions[own_indices[0]]] if own_indices else []
        result[ATTACK_AVAILABLE] = float(
            _has_event(first_window, self.ATTACK_KINDS, seat=seat)
        )

        next_own_index = next(
            (
                candidate
                for candidate in range(index + 1, len(transactions))
                if transactions[candidate].seat == seat
            ),
            None,
        )
        if start.active_serial is not None and next_own_index is not None:
            next_facts = semantic_state_facts(transactions[next_own_index].start_state, seat)
            result[ACTIVE_SURVIVAL] = float(
                start.active_serial in next_facts.living_pokemon_serials
            )
        elif start.active_serial is not None and not transaction.terminal:
            end_facts = semantic_state_facts(transaction.end_state, seat)
            result[ACTIVE_SURVIVAL] = float(
                start.active_serial in end_facts.living_pokemon_serials
            )

        for target_index, horizon in (
            (NET_PRIZE_H1, 1),
            (NET_PRIZE_H3, 3),
            (NET_PRIZE_H6, 6),
        ):
            horizon_indices = own_indices[:horizon]
            if not horizon_indices:
                continue
            horizon_transaction = transactions[horizon_indices[-1]]
            end_state = (
                horizon_transaction.trajectory_terminal_state
                if horizon_transaction.terminal
                and horizon_transaction.trajectory_terminal_state is not None
                else horizon_transaction.end_state
            )
            end = semantic_state_facts(end_state, seat)
            own_gain = start.opponent_prize_count - end.opponent_prize_count
            opponent_gain = start.self_prize_count - end.self_prize_count
            result[target_index] = max(-1.0, min(1.0, (own_gain - opponent_gain) / 6.0))

        window_end = own_indices[min(5, len(own_indices) - 1)] if own_indices else index
        chronological_window = transactions[index : window_end + 1]
        result[SELF_DECKOUT] = float(
            _has_event(chronological_window, self.DECKOUT_KINDS, seat=seat)
            or any(
                tx.terminal
                and tx.outcome < 0
                and (
                    self._terminal_reason(tx) == "deck_out"
                    or self._terminal_deck_count(tx, seat) == 0
                )
                for tx in chronological_window
                if tx.seat == seat
            )
        )

        next_opponent = next(
            (tx for tx in transactions[index + 1 :] if tx.seat != seat),
            None,
        )
        if start.rule_lock_ids and next_opponent is not None:
            opponent_facts = semantic_state_facts(next_opponent.start_state, next_opponent.seat)
            present = set(start.rule_lock_ids)
            result[RULE_LOCK] = float(bool(present.intersection(opponent_facts.rule_lock_ids)))

        until_next_own = transactions[index : next_own_index if next_own_index is not None else index + 1]
        if start.recovery_needed:
            event_realized = _has_event(until_next_own, self.RECOVERY_KINDS, seat=seat)
            if next_own_index is not None:
                next_facts = semantic_state_facts(transactions[next_own_index].start_state, seat)
                recovered_state = next_facts.active_serial is not None and not next_facts.recovery_needed
            else:
                recovered_state = False
            result[RECOVERY_PATH] = float(event_realized or recovered_state)

        if start.armed_delayed_effect_ids:
            result[DELAYED_TRIGGER] = float(
                self._delayed_realized(
                    until_next_own,
                    cause_transaction_id=transaction.transaction_id,
                    armed_ids=set(start.armed_delayed_effect_ids),
                )
            )
        return result

    @staticmethod
    def _terminal_reason(transaction: Transaction) -> str:
        if transaction.terminal_reason is not None:
            return {
                1: "prizes",
                2: "deck_out",
                3: "no_active_pokemon",
                4: "card_effect",
            }.get(transaction.terminal_reason, str(transaction.terminal_reason))
        state = transaction.trajectory_terminal_state or transaction.end_state or {}
        current = state.get("current") or {} if isinstance(state, Mapping) else {}
        value = current.get("terminalReason", current.get("terminal_reason", ""))
        return str(value).lower().replace("-", "_")

    @staticmethod
    def _terminal_deck_count(transaction: Transaction, seat: int) -> int | None:
        state = transaction.trajectory_terminal_state or transaction.end_state
        return semantic_state_facts(state, seat).self_deck_count

    def _delayed_realized(
        self,
        transactions: Iterable[Transaction],
        *,
        cause_transaction_id: int,
        armed_ids: set[str],
    ) -> bool:
        for transaction in transactions:
            for link in transaction.causal_links:
                if link.cause_transaction_id == cause_transaction_id:
                    return True
                if link.source_card_serial is not None and str(link.source_card_serial) in armed_ids:
                    return True
            for event in transaction.event_records:
                if _event_kind(event) not in self.DELAYED_KINDS:
                    continue
                if event.cause_transaction_id == cause_transaction_id:
                    return True
                if event.source_card_serial is not None and str(event.source_card_serial) in armed_ids:
                    return True
        return False


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
            raise ValueError("applicable must have shape [batch, 9]")
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
            raise ValueError("reliability must contain nine values")
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
            raise ValueError("semantic values must have nine fixed positions")
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


def reverse_net_prize_perspective(values: Tensor) -> Tensor:
    """Transform the three signed prize concepts under a seat swap."""

    if values.shape[-1] != NUM_SEMANTIC_CONCEPTS:
        raise ValueError("values must end in the nine semantic concepts")
    swapped = values.clone()
    swapped[..., list(CONTINUOUS_CONCEPT_INDICES)] *= -1.0
    return swapped
