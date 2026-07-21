from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

import torch
from torch import Tensor


MAIN_CONTEXT = 0


@dataclass(frozen=True)
class TransactionEvent:
    """Normalized event used by trajectory labels and causal diagnostics."""

    kind: str
    seat: int | None = None
    turn: int | None = None
    source_card_serial: int | None = None
    subject_serial: int | None = None
    cause_transaction_id: int | None = None
    details: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CausalEventLink:
    cause_transaction_id: int
    trigger_transaction_id: int
    cause_kind: str
    source_card_serial: int | None = None


@dataclass
class SelectRecord:
    """One simulator ``select`` inside a transaction.

    Features and chosen indices are retained so the joint probability can be
    recomputed under the current learner during PPO. Forced records remain in
    the event stream but are deliberately excluded from that probability.
    """

    state_features: Tensor
    option_features: Tensor
    action_indices: tuple[int, ...]
    old_log_prob: float
    forced: bool
    entropy: float = 0.0
    minimum_count: int = 1
    maximum_count: int = 1
    stopped_early: bool = False


@dataclass
class Transaction:
    transaction_id: int
    seat: int
    start_state: Any
    end_state: Any = None
    non_forced_log_probs: list[Tensor] = field(default_factory=list)
    old_log_prob_sum: float = 0.0
    terminal: bool = False
    outcome: int = 0
    terminal_reason: int | None = None
    event_records: list[TransactionEvent] = field(default_factory=list)
    cause_transaction_ids: list[int] = field(default_factory=list)
    select_records: list[SelectRecord] = field(default_factory=list)
    causal_links: list[CausalEventLink] = field(default_factory=list)
    learner_controlled: bool = False
    old_full_value: float = 0.0
    target_phi_before: float = 0.0
    target_phi_after: float = 0.0
    semantic_applicable: Tensor | None = None
    semantic_target_values: Tensor | None = None
    semantic_target_applicable: Tensor | None = None
    trajectory_terminal_state: Any = None
    seat_swapped_state: Any = None

    @property
    def non_forced_select_count(self) -> int:
        return sum(not record.forced for record in self.select_records)

    @property
    def forced_select_count(self) -> int:
        return sum(record.forced for record in self.select_records)

    def append_select(
        self,
        *,
        log_prob: Tensor | float,
        forced: bool,
        record: SelectRecord | None = None,
    ) -> None:
        if record is not None:
            if bool(record.forced) != bool(forced):
                raise ValueError("record.forced disagrees with forced")
            self.select_records.append(record)
        if forced:
            return
        value = log_prob if isinstance(log_prob, Tensor) else torch.tensor(float(log_prob))
        scalar = value.reshape(())
        self.non_forced_log_probs.append(scalar)
        self.old_log_prob_sum += float(scalar.detach().cpu())

    def add_event(self, event: TransactionEvent | Mapping[str, Any]) -> None:
        self.event_records.append(normalize_event(event))


def normalize_event(event: TransactionEvent | Mapping[str, Any]) -> TransactionEvent:
    if isinstance(event, TransactionEvent):
        return event
    known = {
        "kind",
        "seat",
        "turn",
        "source_card_serial",
        "subject_serial",
        "cause_transaction_id",
    }
    details = dict(event.get("details") or {})
    details.update({key: value for key, value in event.items() if key not in known and key != "details"})
    return TransactionEvent(
        kind=str(event.get("kind", event.get("type", "unknown"))),
        seat=_optional_int(event.get("seat", event.get("player_index"))),
        turn=_optional_int(event.get("turn")),
        source_card_serial=_optional_int(event.get("source_card_serial", event.get("serial"))),
        subject_serial=_optional_int(event.get("subject_serial")),
        cause_transaction_id=_optional_int(event.get("cause_transaction_id")),
        details=details,
    )


def _optional_int(value: Any) -> int | None:
    try:
        return None if value is None else int(value)
    except (TypeError, ValueError):
        return None


def joint_log_probability(log_probs: Iterable[Tensor]) -> Tensor:
    values = [value.reshape(()) for value in log_probs]
    if not values:
        return torch.tensor(0.0)
    return torch.stack(values).sum()


def selection_is_forced(observation: Mapping[str, Any]) -> bool:
    """Return a deterministic rule fact; this is never predicted by the model."""

    select = observation.get("select") or {}
    options = list(select.get("option") or [])
    if not options:
        return True
    # Some effects require every available option. There is only one legal set
    # when order is explicitly irrelevant; otherwise multiple permutations are
    # policy choices and must retain their probability.
    minimum = _optional_int(select.get("minCount", select.get("min_count")))
    maximum = _optional_int(select.get("maxCount", select.get("max_count")))
    if maximum == 0:
        return True
    if len(options) == 1:
        return minimum is not None and minimum >= 1
    unordered = bool(select.get("unordered", False))
    return unordered and minimum == maximum == len(options)


def state_seat(state: Mapping[str, Any] | None) -> int | None:
    current = (state or {}).get("current") or {}
    value = _optional_int(current.get("yourIndex", current.get("currentPlayer")))
    return value if value in (0, 1) else None


def state_turn(state: Mapping[str, Any] | None) -> int | None:
    return _optional_int(((state or {}).get("current") or {}).get("turn"))


def is_terminal_state(state: Mapping[str, Any] | None) -> bool:
    current = (state or {}).get("current") or {}
    result = _optional_int(current.get("result"))
    select = (state or {}).get("select")
    return result is not None and result >= 0 and (select is None or not (select.get("option") or []))


def is_root_decision(state: Mapping[str, Any] | None) -> bool:
    select = (state or {}).get("select") or {}
    return _optional_int(select.get("context")) == MAIN_CONTEXT


def closes_transaction(
    transaction: Transaction,
    next_state: Mapping[str, Any],
) -> bool:
    """Implement the causal transaction boundary from the locked contract."""

    if is_terminal_state(next_state):
        return True
    next_seat = state_seat(next_state)
    if next_seat is not None and next_seat != transaction.seat:
        return True
    before_turn = state_turn(transaction.start_state)
    after_turn = state_turn(next_state)
    if before_turn is not None and after_turn is not None and before_turn != after_turn:
        return True
    return next_seat == transaction.seat and is_root_decision(next_state)


class TransactionAssembler:
    """Stateful assembler used directly by rollout collection."""

    def __init__(self, first_transaction_id: int = 0) -> None:
        self._next_id = int(first_transaction_id)
        self.current: Transaction | None = None
        self.completed: list[Transaction] = []

    def begin(
        self,
        *,
        seat: int,
        start_state: Any,
        learner_controlled: bool = False,
        old_full_value: float = 0.0,
        target_phi_before: float = 0.0,
        semantic_applicable: Tensor | None = None,
        seat_swapped_state: Any = None,
    ) -> Transaction:
        if self.current is not None:
            raise RuntimeError("cannot begin a transaction before closing the current one")
        if seat not in (0, 1):
            raise ValueError("seat must be 0 or 1")
        transaction = Transaction(
            transaction_id=self._next_id,
            seat=seat,
            start_state=start_state,
            learner_controlled=learner_controlled,
            old_full_value=float(old_full_value),
            target_phi_before=float(target_phi_before),
            semantic_applicable=semantic_applicable,
            seat_swapped_state=seat_swapped_state,
        )
        self._next_id += 1
        self.current = transaction
        return transaction

    def record_select(
        self,
        *,
        log_prob: Tensor | float,
        forced: bool,
        record: SelectRecord | None = None,
        events: Iterable[TransactionEvent | Mapping[str, Any]] = (),
    ) -> None:
        if self.current is None:
            raise RuntimeError("no active transaction")
        self.current.append_select(log_prob=log_prob, forced=forced, record=record)
        for event in events:
            self.current.add_event(event)

    def close(
        self,
        *,
        end_state: Any,
        terminal: bool = False,
        outcome: int = 0,
        final_events: Iterable[TransactionEvent | Mapping[str, Any]] = (),
    ) -> Transaction:
        if self.current is None:
            raise RuntimeError("no active transaction")
        transaction = self.current
        transaction.end_state = end_state
        transaction.terminal = bool(terminal)
        transaction.outcome = int(outcome)
        for event in final_events:
            transaction.add_event(event)
        self._link_trigger_events(transaction)
        self.completed.append(transaction)
        self.current = None
        return transaction

    @staticmethod
    def _link_trigger_events(transaction: Transaction) -> None:
        for event in transaction.event_records:
            if event.cause_transaction_id is None:
                continue
            event_kind = event.kind.lower().replace("-", "_")
            if event_kind not in {"delayed_trigger", "armed_delayed_trigger_realized"}:
                continue
            link = CausalEventLink(
                cause_transaction_id=event.cause_transaction_id,
                trigger_transaction_id=transaction.transaction_id,
                cause_kind=str(event.details.get("cause_kind", "delayed_effect")),
                source_card_serial=event.source_card_serial,
            )
            transaction.causal_links.append(link)
            if link.cause_transaction_id not in transaction.cause_transaction_ids:
                transaction.cause_transaction_ids.append(link.cause_transaction_id)
