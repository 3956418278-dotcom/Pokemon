from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

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

CARD_DYNAMIC_FEATURE_DIM = 32
CARD_APPEARANCE_FEATURE_DIM = 32
GLOBAL_FEATURE_DIM = 32
DECISION_FEATURE_DIM = 16
MATCH_FEATURE_DIM = 16
LEDGER_FEATURE_DIM = 20
EVENT_FEATURE_DIM = 16
MAX_RECENT_EVENTS = 16


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
    energy_counts: list[int] = field(default_factory=lambda: [0] * 12)
    energy_card_count: int = 0
    tool_count: int = 0
    pre_evolution_count: int = 0
    special_conditions: list[bool] = field(default_factory=lambda: [False] * 5)
    attached_to_serial: int | None = None
    attachment_kind: int = 0
    source: str = "current"

    @property
    def static_card_id(self) -> int:
        return int(self.card_id or 0)

    def board_features(self) -> list[float]:
        hp = float(self.hp or 0)
        max_hp = float(self.max_hp or 0)
        damage = max(max_hp - hp, 0.0) if max_hp else 0.0
        hp_fraction = hp / max_hp if max_hp > 0 else 0.0
        owner = 0.5 if self.relative_player is None else float(self.relative_player)
        energy_counts = (self.energy_counts + [0] * 12)[:12]
        features = [
            float(self.area) / 12.0,
            owner,
            float(max(self.slot, 0)) / 10.0,
            float(self.is_visible),
            float(self.is_face_down),
            float(self.is_pokemon),
            hp / 400.0,
            max_hp / 400.0,
            damage / 400.0,
            hp_fraction,
            float(self.appear_this_turn),
            float(self.energy_card_count) / 10.0,
            float(self.tool_count) / 4.0,
            float(self.pre_evolution_count) / 4.0,
            float(self.attachment_kind) / 3.0,
        ]
        features.extend(float(value) for value in (self.special_conditions + [False] * 5)[:5])
        features.extend(float(value) / 10.0 for value in energy_counts)
        return features[:CARD_DYNAMIC_FEATURE_DIM]


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

    def features(self) -> list[float]:
        players = (self.player_counts + [{} for _ in range(2)])[:2]
        values = [
            self.turn / 100.0,
            self.turn_action_count / 30.0,
            float(self.your_index),
            self.first_player / 2.0,
            float(self.supporter_played),
            float(self.stadium_played),
            float(self.energy_attached),
            float(self.retreated),
            self.result / 2.0,
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
        values = [
            self.select_type / 12.0,
            self.select_context / 50.0,
            self.select_min_count / 10.0,
            self.select_max_count / 10.0,
            self.remain_damage_counter / 100.0,
            self.remain_energy_cost / 10.0,
            float(self.select_type >= 0),
            float(self.select_context >= 0),
        ]
        values.extend([0.0] * DECISION_FEATURE_DIM)
        return values[:DECISION_FEATURE_DIM]

    def match_features(self) -> list[float]:
        values = [
            self.turn / 100.0,
            self.turn_action_count / 30.0,
            float(self.your_index),
            self.first_player / 2.0,
            self.result / 2.0,
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
    card_id: int | None = None
    serial: int | None = None
    from_area: int | None = None
    to_area: int | None = None
    target_card_id: int | None = None
    target_serial: int | None = None
    attack_id: int | None = None
    value: int | None = None
    is_reverse: bool = False
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class ParsedObservation:
    global_snapshot: GlobalSnapshot
    card_instances: list[CardInstanceState]
    events: list[GameEvent]
    select_options: list[dict[str, Any]]


@dataclass
class CardDynamicBatch:
    card_ids: "torch.Tensor"
    dynamic_features: "torch.Tensor"
    appearance_features: "torch.Tensor"
    visibility_mask: "torch.Tensor"
    serials: "torch.Tensor | None" = None


def collate_card_dynamic(instances: list[CardInstanceState], appearance_features: list[list[float]] | None = None) -> CardDynamicBatch:
    import torch

    if appearance_features is None:
        appearance_features = [[0.0] * CARD_APPEARANCE_FEATURE_DIM for _ in instances]
    if not instances:
        return CardDynamicBatch(
            card_ids=torch.zeros(0, dtype=torch.long),
            dynamic_features=torch.zeros(0, CARD_DYNAMIC_FEATURE_DIM, dtype=torch.float32),
            appearance_features=torch.zeros(0, CARD_APPEARANCE_FEATURE_DIM, dtype=torch.float32),
            visibility_mask=torch.zeros(0, dtype=torch.float32),
            serials=torch.zeros(0, dtype=torch.long),
        )
    card_ids = torch.tensor([instance.static_card_id for instance in instances], dtype=torch.long)
    serials = torch.tensor([int(instance.serial or -1) for instance in instances], dtype=torch.long)
    dynamic = torch.tensor([instance.board_features() for instance in instances], dtype=torch.float32)
    appearance = torch.tensor(appearance_features, dtype=torch.float32)
    visible = torch.tensor([float(instance.is_visible) for instance in instances], dtype=torch.float32)
    return CardDynamicBatch(
        card_ids=card_ids,
        dynamic_features=dynamic,
        appearance_features=appearance,
        visibility_mask=visible,
        serials=serials,
    )
