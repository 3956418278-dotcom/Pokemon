from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum


class RelativeOwner(IntEnum):
    UNKNOWN = 0
    SELF = 1
    OPPONENT = 2


class SelectionMode(str, Enum):
    SINGLE = "SINGLE"
    ORDERED_SEQUENCE = "ORDERED_SEQUENCE"
    UNORDERED_UNIQUE_SUBSET = "UNORDERED_UNIQUE_SUBSET"
    UNKNOWN = "UNKNOWN"


class TerminalOutcome(IntEnum):
    LOSS = 0
    DRAW = 1
    WIN = 2


@dataclass(frozen=True)
class CardInstanceView:
    card_id: int | None
    card_index: int
    serial: int | None
    attached_to_serial: int | None
    relative_owner: int
    zone: int
    zone_position: int
    is_active: bool
    is_bench: bool
    hp: float
    max_hp: float
    attached_energy_counts: tuple[int, ...]
    special_condition_flags: tuple[bool, ...]
    appear_this_turn: bool
    pre_evolution_count: int
    tool_count: int
    field_visibility_mask: tuple[float, ...]

    def dynamic_features(self) -> tuple[float, ...]:
        hp_scale = max(self.max_hp, 1.0)
        values = (
            self.hp / 400.0,
            self.max_hp / 400.0,
            max(self.max_hp - self.hp, 0.0) / hp_scale,
            float(self.is_active),
            float(self.is_bench),
            float(self.appear_this_turn),
            self.pre_evolution_count / 3.0,
            self.tool_count / 4.0,
            *(count / 8.0 for count in self.attached_energy_counts),
            *(float(flag) for flag in self.special_condition_flags),
            *self.field_visibility_mask,
        )
        return tuple(float(value) for value in values)


@dataclass(frozen=True)
class GlobalStateView:
    turn: int
    turn_action_count: int
    is_first_player: bool
    relative_current_player: int
    self_hand_count: int
    opponent_hand_count: int
    self_deck_count: int
    opponent_deck_count: int
    self_prize_count: int
    opponent_prize_count: int
    self_bench_count: int
    opponent_bench_count: int
    energy_attached: bool
    supporter_played: bool
    stadium_played: bool
    retreated: bool
    select_type: int
    select_context: int
    remain_damage_counter: int
    remain_energy_cost: int
    min_count: int
    max_count: int
    field_visibility_mask: tuple[float, ...]

    def features(self) -> tuple[float, ...]:
        values = (
            self.turn / 100.0,
            self.turn_action_count / 30.0,
            float(self.is_first_player),
            float(self.relative_current_player),
            self.self_hand_count / 30.0,
            self.opponent_hand_count / 30.0,
            self.self_deck_count / 60.0,
            self.opponent_deck_count / 60.0,
            self.self_prize_count / 6.0,
            self.opponent_prize_count / 6.0,
            self.self_bench_count / 5.0,
            self.opponent_bench_count / 5.0,
            float(self.energy_attached),
            float(self.supporter_played),
            float(self.stadium_played),
            float(self.retreated),
            self.select_type / 10.0,
            self.select_context / 50.0,
            self.remain_damage_counter / 300.0,
            self.remain_energy_cost / 10.0,
            self.min_count / 10.0,
            self.max_count / 10.0,
            *self.field_visibility_mask,
        )
        return tuple(float(value) for value in values)


@dataclass(frozen=True)
class PublicEventView:
    event_type: int
    relative_player: int
    card_index: int
    serial: int | None
    from_zone: int
    to_zone: int
    turn_age: int


@dataclass(frozen=True)
class HistoryView:
    recent_events: tuple[PublicEventView, ...]
    event_type_counts: tuple[int, ...]
    public_card_id_counts: tuple[tuple[int, int], ...]
    public_serial_zone_changes: tuple[tuple[int, int, int], ...]
    current_turn_action_position: int

    def features(self) -> tuple[float, ...]:
        event_counts = tuple(value / 20.0 for value in self.event_type_counts[:24])
        # Keep the summary fixed-width while retaining Card ID occurrence
        # information.  Exact counts remain in ``public_card_id_counts`` for
        # auditability; stable hash buckets make them usable by the V1 token.
        card_id_count_buckets = [0.0] * 16
        for card_id, count in self.public_card_id_counts:
            card_id_count_buckets[int(card_id) % len(card_id_count_buckets)] += count / 8.0
        return (
            *event_counts,
            len(self.recent_events) / 32.0,
            len(self.public_card_id_counts) / 60.0,
            len(self.public_serial_zone_changes) / 60.0,
            self.current_turn_action_position / 30.0,
            *card_id_count_buckets,
        )


@dataclass(frozen=True)
class OptionView:
    original_option_index: int
    option_type: int
    select_type: int
    select_context: int
    relative_player: int
    area: int
    position_index: int
    card_id: int | None
    card_index: int
    serial: int | None
    energy_type: int
    damage_value: float
    has_card_reference: bool
    has_serial_reference: bool
    equivalence_group: int
    field_visibility_mask: tuple[float, ...]


@dataclass(frozen=True)
class DecisionSampleV1:
    episode_id: str
    source_date: str | None
    agent_index: int
    decision_index: int
    step: int
    turn: int
    turn_action_count: int
    cards: tuple[CardInstanceView, ...]
    global_state: GlobalStateView
    history: HistoryView
    options: tuple[OptionView, ...]
    selected_option_indices: tuple[int, ...]
    selected_equivalence_groups: tuple[int, ...]
    min_count: int
    max_count: int
    selection_mode: SelectionMode
    terminal_outcome: TerminalOutcome
    episode_decision_count: int
    policy_supervision: bool
    policy_mask_reason: str
    visibility_sources: tuple[str, ...] = field(
        default=("observation.current", "observation.logs", "observation.select")
    )
