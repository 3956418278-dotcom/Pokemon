from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, TYPE_CHECKING

from .decision_schema import DetailUsageState, FieldState

if TYPE_CHECKING:
    import torch


AREA_IDS = {
    "DECK": 1,
    "HAND": 2,
    "DISCARD": 3,
    "ACTIVE": 4,
    "BENCH": 5,
    "PRIZE": 6,
    "STADIUM": 7,
    "ENERGY": 8,
    "TOOL": 9,
    "PRE_EVOLUTION": 10,
    "PLAYER": 11,
    "LOOKING": 12,
}

OWNER_IDS = {
    "UNKNOWN": 0,
    "SELF": 1,
    "OPPONENT": 2,
}

# Keep zone IDs aligned with cabt's stable area IDs. Zero is reserved for
# missing/unknown zones so it can be used as an embedding padding index.
ZONE_IDS = {"UNKNOWN": 0, **AREA_IDS}

FIELD_ROLE_IDS = {
    "UNKNOWN": 0,
    "NONE": 1,
    "ACTIVE": 2,
    "BENCH": 3,
    "ATTACHMENT": 4,
}

ATTACHMENT_KIND_IDS = {
    "UNKNOWN": 0,
    "NONE": 1,
    "ENERGY": 2,
    "TOOL": 3,
    "PRE_EVOLUTION": 4,
}

KNOWLEDGE_IDS = {
    "UNKNOWN": 0,
    "VISIBLE_KNOWN": 1,
    "VISIBLE_UNKNOWN": 2,
    "HIDDEN_ANONYMOUS": 3,
    "HIDDEN_KNOWN": 4,
}

OWNER_VOCAB_SIZE = max(OWNER_IDS.values()) + 1
ZONE_VOCAB_SIZE = max(ZONE_IDS.values()) + 1
FIELD_ROLE_VOCAB_SIZE = max(FIELD_ROLE_IDS.values()) + 1
ATTACHMENT_KIND_VOCAB_SIZE = max(ATTACHMENT_KIND_IDS.values()) + 1
KNOWLEDGE_VOCAB_SIZE = max(KNOWLEDGE_IDS.values()) + 1

ENERGY_TYPE_NAMES = [
    "COLORLESS",
    "GRASS",
    "FIRE",
    "WATER",
    "LIGHTNING",
    "PSYCHIC",
    "FIGHTING",
    "DARKNESS",
    "METAL",
    "DRAGON",
    "RAINBOW",
    "TEAM_ROCKET",
]

SPECIAL_CONDITION_NAMES = ["poisoned", "burned", "asleep", "paralyzed", "confused"]

CARD_APPEARANCE_FEATURE_DIM = 32
GLOBAL_FEATURE_DIM = 32
DECISION_FEATURE_DIM = 16
MATCH_FEATURE_DIM = 16
LEDGER_FEATURE_DIM = 20
EVENT_FEATURE_DIM = 19
MAX_RECENT_EVENTS = 32

NUMERICAL_FEATURE_NAMES = (
    "current_hp",
    "max_hp",
    "damage",
    "hp_ratio",
    "damage_ratio",
    "copy_count",
    "energy_card_count",
    "tool_count",
    "pre_evolution_count",
)

BOOLEAN_FEATURE_NAMES = (
    "is_visible",
    "is_face_down",
    "is_pokemon",
    "appear_this_turn",
    "has_tool",
    "has_pre_evolution",
    "is_attachment",
)


@dataclass
class CardInstanceState:
    card_id: int | None
    serial: int | None
    player_index: int | None
    relative_player: int | None
    area: int
    zone: str
    slot: int
    is_visible: bool = True
    is_face_down: bool = False
    is_pokemon: bool = False
    hp: int | None = None
    max_hp: int | None = None
    appear_this_turn: bool = False
    appear_this_turn_valid: bool = False
    energy_counts: list[int] = field(default_factory=lambda: [0] * 12)
    energy_counts_valid: bool = False
    energy_card_count: int = 0
    energy_cards_valid: bool = False
    energy_card_ids: list[int] = field(default_factory=list)
    tool_count: int = 0
    tools_valid: bool = False
    tool_card_ids: list[int] = field(default_factory=list)
    pre_evolution_count: int = 0
    pre_evolution_valid: bool = False
    pre_evolution_card_ids: list[int] = field(default_factory=list)
    special_conditions: list[bool] = field(default_factory=lambda: [False] * 5)
    special_conditions_valid: bool = False
    attached_to_serial: int | None = None
    attachment_kind: int | None = 0
    copy_count: int | None = None
    energy_payment_resolved: bool | None = None
    detail_exists: bool | None = None
    static_artifact_known: bool | None = None
    source: str = "current"
    field_states: dict[str, FieldState] = field(default_factory=dict)
    used_detail_states: list[DetailUsageState] = field(default_factory=list)
    used_detail_inference_sources: list[str | None] = field(default_factory=list)

    @property
    def static_card_id(self) -> int:
        return int(self.card_id or 0)

    def numerical_values_and_mask(self) -> tuple[list[float], list[float]]:
        hp_valid = self.hp is not None
        max_hp_valid = self.max_hp is not None
        damage_valid = hp_valid and max_hp_valid
        ratio_valid = damage_valid and float(self.max_hp) > 0
        hp = float(self.hp) if hp_valid else 0.0
        max_hp = float(self.max_hp) if max_hp_valid else 0.0
        damage = max(max_hp - hp, 0.0) if damage_valid else 0.0
        hp_ratio = hp / max_hp if ratio_valid else 0.0
        damage_ratio = damage / max_hp if ratio_valid else 0.0
        values = [
            hp,
            max_hp,
            damage,
            hp_ratio,
            damage_ratio,
            float(self.copy_count or 0),
            float(self.energy_card_count),
            float(self.tool_count),
            float(self.pre_evolution_count),
        ]
        mask = [
            float(hp_valid),
            float(max_hp_valid),
            float(damage_valid),
            float(ratio_valid),
            float(ratio_valid),
            float(self.copy_count is not None),
            float(self.energy_cards_valid),
            float(self.tools_valid),
            float(self.pre_evolution_valid),
        ]
        return values, mask

    def boolean_values_and_mask(self) -> tuple[list[float], list[float]]:
        is_attachment = self.attachment_kind not in (None, 0)
        values = [
            float(self.is_visible),
            float(self.is_face_down),
            float(self.is_pokemon),
            float(self.appear_this_turn),
            float(self.tool_count > 0),
            float(self.pre_evolution_count > 0),
            float(is_attachment),
        ]
        mask = [
            1.0,
            1.0,
            1.0,
            float(self.is_pokemon and self.appear_this_turn_valid),
            float(self.is_pokemon and self.tools_valid),
            float(self.is_pokemon and self.pre_evolution_valid),
            float(self.attachment_kind is not None),
        ]
        return values, mask

    def detail_usage(self, detail_count: int) -> tuple[list[DetailUsageState], list[str | None]]:
        """Return aligned detail-use states, defaulting conservatively to UNKNOWN."""

        default = DetailUsageState.UNKNOWN if self.is_pokemon else DetailUsageState.NOT_APPLICABLE
        states = (self.used_detail_states + [default] * detail_count)[:detail_count]
        sources = (self.used_detail_inference_sources + [None] * detail_count)[:detail_count]
        return states, sources


@dataclass
class GlobalSnapshot:
    turn: int = 0
    turn_action_count: int = 0
    your_index: int = 0
    first_player: int = -1
    supporter_played: bool = False
    stadium_played: bool = False
    energy_attached: bool = False
    retreated: bool = False
    result: int = -1
    stadium_count: int = 0
    looking_count: int = 0
    select_type: int = -1
    select_context: int = -1
    select_min_count: int = 0
    select_max_count: int = 0
    remain_damage_counter: int = 0
    remain_energy_cost: int = 0
    player_counts: list[dict[str, int]] = field(default_factory=list)
    current_log_count: int = 0
    current_reverse_log_count: int = 0
    current_public_card_log_count: int = 0
    field_states: dict[str, FieldState] = field(default_factory=dict)

    def features(self) -> list[float]:
        players = (self.player_counts + [{} for _ in range(2)])[:2]
        first_player = (
            self.first_player
            if self.field_states.get("firstPlayer") is FieldState.PRESENT and self.first_player >= 0
            else 0
        )
        result = (
            self.result
            if self.field_states.get("result") is FieldState.PRESENT and self.result >= 0
            else 0
        )
        values = [
            self.turn / 100.0,
            self.turn_action_count / 30.0,
            float(self.your_index),
            first_player / 2.0,
            float(self.supporter_played),
            float(self.stadium_played),
            float(self.energy_attached),
            float(self.retreated),
            result / 2.0,
            self.stadium_count,
            self.looking_count / 10.0,
            self.select_type / 12.0,
            self.select_context / 50.0,
            self.select_min_count / 10.0,
            self.select_max_count / 10.0,
            self.remain_damage_counter / 100.0,
            self.remain_energy_cost / 10.0,
        ]
        for counts in players:
            values.extend(
                [
                    counts.get("active", 0),
                    counts.get("bench", 0) / 5.0,
                    counts.get("deck", 0) / 60.0,
                    counts.get("discard", 0) / 60.0,
                    counts.get("prize", 0) / 6.0,
                    counts.get("hand", 0) / 20.0,
                ]
            )
        values.extend([0.0] * GLOBAL_FEATURE_DIM)
        return values[:GLOBAL_FEATURE_DIM]

    def decision_features(self) -> list[float]:
        def present(name: str, value: int) -> int:
            return value if self.field_states.get(name) is FieldState.PRESENT else 0

        select_type = present("select.type", self.select_type)
        select_context = present("select.context", self.select_context)
        values = [
            select_type / 12.0,
            select_context / 50.0,
            present("select.minCount", self.select_min_count) / 10.0,
            present("select.maxCount", self.select_max_count) / 10.0,
            present("select.remainDamageCounter", self.remain_damage_counter) / 100.0,
            present("select.remainEnergyCost", self.remain_energy_cost) / 10.0,
            float(self.field_states.get("select.type") is FieldState.PRESENT),
            float(self.field_states.get("select.context") is FieldState.PRESENT),
        ]
        values.extend([0.0] * DECISION_FEATURE_DIM)
        return values[:DECISION_FEATURE_DIM]

    def match_features(self) -> list[float]:
        first_player = (
            self.first_player
            if self.field_states.get("firstPlayer") is FieldState.PRESENT and self.first_player >= 0
            else 0
        )
        result = (
            self.result
            if self.field_states.get("result") is FieldState.PRESENT and self.result >= 0
            else 0
        )
        values = [
            self.turn / 100.0,
            self.turn_action_count / 30.0,
            float(self.your_index),
            first_player / 2.0,
            result / 2.0,
            float(self.supporter_played),
            float(self.stadium_played),
            float(self.energy_attached),
            float(self.retreated),
            self.stadium_count,
            self.looking_count / 10.0,
            self.current_log_count / 64.0,
            self.current_reverse_log_count / 64.0,
            self.current_public_card_log_count / 64.0,
        ]
        values.extend([0.0] * MATCH_FEATURE_DIM)
        return values[:MATCH_FEATURE_DIM]


@dataclass
class GameEvent:
    event_type: int
    player_index: int | None = None
    actor_relative: int | None = None
    card_id: int | None = None
    serial: int | None = None
    from_area: int | None = None
    to_area: int | None = None
    target_card_id: int | None = None
    target_serial: int | None = None
    attack_id: int | None = None
    value: int | None = None
    coin_result: int | None = None
    is_reverse: bool = False
    identity_visible: bool = False
    observation_age: int = 0
    batch_position: int = 0
    observed_at_turn: int = 0
    observed_at_turn_action_count: int = 0
    turn_age: int = 0
    field_states: dict[str, FieldState] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class ParsedObservation:
    global_snapshot: GlobalSnapshot
    card_instances: list[CardInstanceState]
    events: list[GameEvent]
    select_options: list[dict[str, Any]]
    effect_reference: dict[str, Any] | None = None
    context_card_reference: dict[str, Any] | None = None
    effect_presence: FieldState = FieldState.NOT_APPLICABLE
    context_card_presence: FieldState = FieldState.NOT_APPLICABLE
    option_field_states: list[dict[str, FieldState]] = field(default_factory=list)


@dataclass
class CardDynamicBatch:
    card_ids: "torch.Tensor"
    serials: "torch.Tensor"
    owner_ids: "torch.Tensor"
    zone_ids: "torch.Tensor"
    field_role_ids: "torch.Tensor"
    attachment_kind_ids: "torch.Tensor"
    knowledge_ids: "torch.Tensor"
    numerical_features: "torch.Tensor"
    numerical_mask: "torch.Tensor"
    energy_counts: "torch.Tensor"
    energy_valid_mask: "torch.Tensor"
    condition_flags: "torch.Tensor"
    condition_valid_mask: "torch.Tensor"
    boolean_features: "torch.Tensor"
    boolean_mask: "torch.Tensor"
    visibility_mask: "torch.Tensor"
    static_known_mask: "torch.Tensor"
    detail_exists_mask: "torch.Tensor"
    energy_resolved_mask: "torch.Tensor"
    appearance_features: "torch.Tensor | None" = None

    @property
    def batch_size(self) -> int:
        return int(self.card_ids.shape[0])

    def to(self, device: "torch.device | str") -> "CardDynamicBatch":
        values = {}
        for item in fields(self):
            value = getattr(self, item.name)
            values[item.name] = value.to(device) if value is not None else None
        return CardDynamicBatch(**values)


def _owner_id(instance: CardInstanceState) -> int:
    if instance.relative_player == 0:
        return OWNER_IDS["SELF"]
    if instance.relative_player == 1:
        return OWNER_IDS["OPPONENT"]
    return OWNER_IDS["UNKNOWN"]


def _zone_id(instance: CardInstanceState) -> int:
    return int(instance.area) if int(instance.area) in AREA_IDS.values() else ZONE_IDS["UNKNOWN"]


def _field_role_id(instance: CardInstanceState) -> int:
    if instance.attachment_kind not in (None, 0):
        return FIELD_ROLE_IDS["ATTACHMENT"]
    if instance.zone == "active":
        return FIELD_ROLE_IDS["ACTIVE"]
    if instance.zone == "bench":
        return FIELD_ROLE_IDS["BENCH"]
    if instance.zone:
        return FIELD_ROLE_IDS["NONE"]
    return FIELD_ROLE_IDS["UNKNOWN"]


def _attachment_kind_id(instance: CardInstanceState) -> int:
    if instance.attachment_kind is None:
        return ATTACHMENT_KIND_IDS["UNKNOWN"]
    return {
        0: ATTACHMENT_KIND_IDS["NONE"],
        1: ATTACHMENT_KIND_IDS["ENERGY"],
        2: ATTACHMENT_KIND_IDS["TOOL"],
        3: ATTACHMENT_KIND_IDS["PRE_EVOLUTION"],
    }.get(int(instance.attachment_kind), ATTACHMENT_KIND_IDS["UNKNOWN"])


def _knowledge_id(instance: CardInstanceState) -> int:
    if instance.is_visible:
        return KNOWLEDGE_IDS["VISIBLE_KNOWN"] if instance.card_id is not None else KNOWLEDGE_IDS["VISIBLE_UNKNOWN"]
    return KNOWLEDGE_IDS["HIDDEN_KNOWN"] if instance.card_id is not None else KNOWLEDGE_IDS["HIDDEN_ANONYMOUS"]


def collate_card_dynamic(instances: list[CardInstanceState], appearance_features: list[list[float]] | None = None) -> CardDynamicBatch:
    import torch

    if not instances:
        return CardDynamicBatch(
            card_ids=torch.zeros(0, dtype=torch.long),
            serials=torch.zeros(0, dtype=torch.long),
            owner_ids=torch.zeros(0, dtype=torch.long),
            zone_ids=torch.zeros(0, dtype=torch.long),
            field_role_ids=torch.zeros(0, dtype=torch.long),
            attachment_kind_ids=torch.zeros(0, dtype=torch.long),
            knowledge_ids=torch.zeros(0, dtype=torch.long),
            numerical_features=torch.zeros(0, len(NUMERICAL_FEATURE_NAMES), dtype=torch.float32),
            numerical_mask=torch.zeros(0, len(NUMERICAL_FEATURE_NAMES), dtype=torch.float32),
            energy_counts=torch.zeros(0, len(ENERGY_TYPE_NAMES), dtype=torch.float32),
            energy_valid_mask=torch.zeros(0, 1, dtype=torch.float32),
            condition_flags=torch.zeros(0, len(SPECIAL_CONDITION_NAMES), dtype=torch.float32),
            condition_valid_mask=torch.zeros(0, 1, dtype=torch.float32),
            boolean_features=torch.zeros(0, len(BOOLEAN_FEATURE_NAMES), dtype=torch.float32),
            boolean_mask=torch.zeros(0, len(BOOLEAN_FEATURE_NAMES), dtype=torch.float32),
            visibility_mask=torch.zeros(0, dtype=torch.float32),
            static_known_mask=torch.zeros(0, dtype=torch.float32),
            detail_exists_mask=torch.zeros(0, dtype=torch.float32),
            energy_resolved_mask=torch.zeros(0, dtype=torch.float32),
            appearance_features=(
                torch.zeros(0, CARD_APPEARANCE_FEATURE_DIM, dtype=torch.float32)
                if appearance_features is not None
                else None
            ),
        )
    card_ids = torch.tensor([instance.static_card_id for instance in instances], dtype=torch.long)
    serials = torch.tensor([int(instance.serial) if instance.serial is not None else -1 for instance in instances], dtype=torch.long)
    numerical_rows = [instance.numerical_values_and_mask() for instance in instances]
    boolean_rows = [instance.boolean_values_and_mask() for instance in instances]
    appearance = torch.tensor(appearance_features, dtype=torch.float32) if appearance_features is not None else None
    visible = torch.tensor([float(instance.is_visible) for instance in instances], dtype=torch.float32)
    return CardDynamicBatch(
        card_ids=card_ids,
        serials=serials,
        owner_ids=torch.tensor([_owner_id(instance) for instance in instances], dtype=torch.long),
        zone_ids=torch.tensor([_zone_id(instance) for instance in instances], dtype=torch.long),
        field_role_ids=torch.tensor([_field_role_id(instance) for instance in instances], dtype=torch.long),
        attachment_kind_ids=torch.tensor([_attachment_kind_id(instance) for instance in instances], dtype=torch.long),
        knowledge_ids=torch.tensor([_knowledge_id(instance) for instance in instances], dtype=torch.long),
        numerical_features=torch.tensor([row[0] for row in numerical_rows], dtype=torch.float32),
        numerical_mask=torch.tensor([row[1] for row in numerical_rows], dtype=torch.float32),
        energy_counts=torch.tensor([(instance.energy_counts + [0] * 12)[:12] for instance in instances], dtype=torch.float32),
        energy_valid_mask=torch.tensor([[float(instance.energy_counts_valid)] for instance in instances], dtype=torch.float32),
        condition_flags=torch.tensor(
            [(instance.special_conditions + [False] * 5)[:5] for instance in instances], dtype=torch.float32
        ),
        condition_valid_mask=torch.tensor(
            [[float(instance.special_conditions_valid)] for instance in instances], dtype=torch.float32
        ),
        boolean_features=torch.tensor([row[0] for row in boolean_rows], dtype=torch.float32),
        boolean_mask=torch.tensor([row[1] for row in boolean_rows], dtype=torch.float32),
        visibility_mask=visible,
        static_known_mask=torch.tensor(
            [
                float(instance.static_artifact_known)
                if instance.static_artifact_known is not None
                else float(instance.card_id is not None)
                for instance in instances
            ],
            dtype=torch.float32,
        ),
        detail_exists_mask=torch.tensor([float(instance.detail_exists or False) for instance in instances], dtype=torch.float32),
        energy_resolved_mask=torch.tensor(
            [float(instance.energy_payment_resolved or False) for instance in instances], dtype=torch.float32
        ),
        appearance_features=appearance,
    )
