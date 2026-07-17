from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Any


class FieldState(IntEnum):
    """Presence state for every nullable model-facing scalar or reference."""

    PRESENT = 0
    MISSING = 1
    UNKNOWN = 2
    NOT_APPLICABLE = 3
    EXPLICIT_NULL = 4


class DetailUsageState(IntEnum):
    TRUE = 0
    FALSE = 1
    UNKNOWN = 2
    NOT_APPLICABLE = 3


class ActionSemantics(str, Enum):
    SINGLE_INDEX = "SINGLE_INDEX"
    UNORDERED_UNIQUE_SUBSET = "UNORDERED_UNIQUE_SUBSET"
    ORDERED_INDEX_SEQUENCE = "ORDERED_INDEX_SEQUENCE"
    INDEX_MULTISET = "INDEX_MULTISET"
    COUNT_VALUE = "COUNT_VALUE"


@dataclass(frozen=True)
class DecisionKey:
    episode_id: int | None
    decision_step_index: int
    action_step_index: int
    player_index: int


@dataclass(frozen=True)
class OptionalField:
    value: Any
    state: FieldState


@dataclass(frozen=True)
class ResidualMarginals:
    """Contract for the unresolved transport problem used by residual IPF.

    ``unresolved_by_card`` is u_c after exact hidden copies are removed.
    ``unresolved_by_zone`` is q_z after exact hidden copies are removed.
    The IPF output is an expected-count matrix, not a presence probability.
    """

    unresolved_by_card: dict[int, float]
    unresolved_by_zone: dict[str, float]

    def validate(self, tolerance: float = 1e-5) -> None:
        if any(value < -tolerance for value in self.unresolved_by_card.values()):
            raise ValueError("residual card marginals must be non-negative")
        if any(value < -tolerance for value in self.unresolved_by_zone.values()):
            raise ValueError("residual zone marginals must be non-negative")
        row_mass = sum(max(value, 0.0) for value in self.unresolved_by_card.values())
        column_mass = sum(max(value, 0.0) for value in self.unresolved_by_zone.values())
        if abs(row_mass - column_mass) > tolerance:
            raise ValueError(
                f"residual IPF marginals disagree: card mass={row_mass}, zone mass={column_mass}"
            )


@dataclass(frozen=True)
class LegalActionTarget:
    semantics: ActionSemantics
    chosen_option_indices: tuple[int, ...]
    equivalence_class_ids: tuple[int, ...]
    equivalence_class_capacities: dict[int, int]
    chosen_class_counts: dict[int, int]
    selected_count: int
    ordered_class_sequence: tuple[int, ...] = ()
    count_value: int | None = None
    count_value_to_option_indices: dict[int, tuple[int, ...]] | None = None


@dataclass(frozen=True)
class MatchContextRecord:
    turn: OptionalField
    turn_action_count: OptionalField
    is_starting_player: OptionalField
    is_turn_owner: OptionalField


@dataclass(frozen=True)
class SideResourceRecord:
    deck_count: OptionalField
    hand_count: OptionalField
    prize_count: OptionalField
    discard_count: OptionalField
    bench_free_slots: OptionalField


@dataclass(frozen=True)
class TurnUsageRecord:
    energy_attached: OptionalField
    supporter_played: OptionalField
    stadium_played: OptionalField
    retreated: OptionalField
    turn_owner_relative: OptionalField


@dataclass(frozen=True)
class ResourceContextRecord:
    self_resources: SideResourceRecord
    opponent_resources: SideResourceRecord
    turn_usage: TurnUsageRecord


@dataclass(frozen=True)
class SerialRegistryRecord:
    serial: int
    card_id: int | None
    owner_relative: int | None
    exact_zone: int | None
    previous_exact_zone: int | None
    possible_hidden_zone_mask: int
    currently_visible: bool
    last_seen_turn: int
    last_seen_observation: int
    last_event_type: int | None
    field_states: dict[str, FieldState]


@dataclass(frozen=True)
class CardIdMemoryRecord:
    owner_relative: int
    card_id: int
    exact_zone_counts: dict[str, int]
    ambiguous_hidden_count: int
    expected_zone_counts: dict[str, float]
    presence_prediction: float | None
    uncertainty: float | None
    revealed_unique_copy_count: int
    historical_seen_count: int
    historical_move_count: int
    first_seen_turn: int | None
    last_seen_turn: int | None


@dataclass(frozen=True)
class AnonymousHiddenPoolsRecord:
    self_unresolved_deck_prize_count: int
    opponent_unknown_hand_count: int
    opponent_unknown_deck_count: int
    opponent_unknown_prize_count: int
    anonymous_zone_transitions_by_side: dict[int, dict[str, int]]


@dataclass(frozen=True)
class HiddenBeliefRecord:
    deck_archetype_compatibility_distribution: tuple[float, ...]
    residual_marginals_by_side: dict[int, ResidualMarginals]
    expected_zone_counts: dict[tuple[int, int], dict[str, float]]
    presence_predictions: dict[tuple[int, int], float]
    unresolved_zone_entropy: dict[tuple[int, int], float]
    recalled_card_keys: tuple[tuple[int, int], ...]
    recall_slot_reasons: dict[tuple[int, int], tuple[str, ...]]
    tail_statistics: dict[str, float]


@dataclass(frozen=True)
class DecisionContextRecord:
    select_type: OptionalField
    select_context: OptionalField
    min_count: OptionalField
    max_count: OptionalField
    remain_energy_cost: OptionalField
    remain_damage_counter: OptionalField
    effect_reference: OptionalField
    context_card_reference: OptionalField


@dataclass(frozen=True)
class TrainingTargetsRecord:
    legal_action: LegalActionTarget
    final_outcome: int | None
    hidden_truth_reference: str | None
    policy_loss_mask: bool


@dataclass(frozen=True)
class ReplayDecisionContract:
    """Frozen eight-part record; raw observation remains in the replay archive."""

    schema_version: str
    key: DecisionKey
    match: MatchContextRecord
    resources: ResourceContextRecord
    card_instances: tuple[Any, ...]
    recent_events: tuple[Any, ...]
    serial_registry: tuple[SerialRegistryRecord, ...]
    anonymous_hidden_pools: AnonymousHiddenPoolsRecord
    card_id_memory: tuple[CardIdMemoryRecord, ...]
    instance_card_id_memory_edges: tuple[tuple[int, int], ...]
    hidden_belief: HiddenBeliefRecord | None
    hidden_belief_state: FieldState
    decision: DecisionContextRecord
    legal_options: tuple[dict[str, Any], ...]
    targets: TrainingTargetsRecord
