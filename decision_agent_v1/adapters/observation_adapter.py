from __future__ import annotations

from collections import Counter

from data.decision_schema import FieldState
from data.game_memory import GameMemoryState
from data.state_schema import AREA_IDS, CardInstanceState, ParsedObservation

from decision_agent_v1.contracts.schemas import (
    CardInstanceView,
    GlobalStateView,
    HistoryView,
    OptionView,
    PublicEventView,
    RelativeOwner,
)

from .card_vocab_adapter import CardVocabulary


def _int(value: object, default: int = -1) -> int:
    if isinstance(value, bool):
        return default
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _present(state: FieldState | None) -> float:
    return float(state is FieldState.PRESENT)


class ObservationAdapter:
    """Convert the existing visibility-filtered parser output into V1 views.

    This adapter intentionally receives no raw replay envelope and never reads
    ``visualize``. Raw option dictionaries are accepted only because the existing
    parser exposes them as the policy-visible ``select.option`` list.
    """

    def __init__(self, vocabulary: CardVocabulary) -> None:
        self.vocabulary = vocabulary

    @staticmethod
    def _card_sort_key(card: CardInstanceState) -> tuple[int, ...]:
        owner_order = {0: 0, 1: 1, None: 2}.get(card.relative_player, 2)
        zone_order = {
            AREA_IDS["ACTIVE"]: 0,
            AREA_IDS["BENCH"]: 1,
            AREA_IDS["HAND"]: 2,
            AREA_IDS["DISCARD"]: 3,
            AREA_IDS["PRIZE"]: 4,
            AREA_IDS["STADIUM"]: 5,
            AREA_IDS["ENERGY"]: 6,
            AREA_IDS["TOOL"]: 7,
            AREA_IDS["PRE_EVOLUTION"]: 8,
            AREA_IDS["LOOKING"]: 9,
        }.get(card.area, 10)
        return (
            owner_order,
            zone_order,
            int(card.slot),
            int(card.attached_to_serial or -1),
            int(card.card_id or -1),
            int(card.serial or -1),
        )

    def cards(self, parsed: ParsedObservation) -> tuple[CardInstanceView, ...]:
        views = []
        for card in sorted(parsed.card_instances, key=self._card_sort_key):
            owner = (
                RelativeOwner.SELF
                if card.relative_player == 0
                else RelativeOwner.OPPONENT
                if card.relative_player == 1
                else RelativeOwner.UNKNOWN
            )
            visibility = (
                float(card.is_visible and card.card_id is not None),
                float(card.hp is not None),
                float(card.max_hp is not None),
                float(card.energy_counts_valid),
                float(card.special_conditions_valid),
                float(card.appear_this_turn_valid),
                float(card.pre_evolution_valid),
                float(card.tools_valid),
            )
            views.append(
                CardInstanceView(
                    card_id=card.card_id if card.is_visible else None,
                    card_index=self.vocabulary.encode(card.card_id if card.is_visible else None),
                    serial=card.serial if card.is_visible else None,
                    attached_to_serial=card.attached_to_serial if card.is_visible else None,
                    relative_owner=int(owner),
                    zone=int(card.area),
                    zone_position=int(card.slot),
                    is_active=card.area == AREA_IDS["ACTIVE"] and card.attachment_kind in (None, 0),
                    is_bench=card.area == AREA_IDS["BENCH"] and card.attachment_kind in (None, 0),
                    hp=float(card.hp or 0),
                    max_hp=float(card.max_hp or 0),
                    attached_energy_counts=tuple(int(value) for value in card.energy_counts[:12]),
                    special_condition_flags=tuple(bool(value) for value in card.special_conditions[:5]),
                    appear_this_turn=bool(card.appear_this_turn),
                    pre_evolution_count=int(card.pre_evolution_count),
                    tool_count=int(card.tool_count),
                    field_visibility_mask=visibility,
                )
            )
        return tuple(views)

    def global_state(self, parsed: ParsedObservation) -> GlobalStateView:
        snapshot = parsed.global_snapshot
        players = (snapshot.player_counts + [{}, {}])[:2]
        your_index = snapshot.your_index if snapshot.your_index in (0, 1) else 0
        opponent_index = 1 - your_index
        self_counts, opponent_counts = players[your_index], players[opponent_index]
        if snapshot.turn >= 1 and snapshot.first_player in (0, 1):
            current = snapshot.first_player if snapshot.turn % 2 == 1 else 1 - snapshot.first_player
            relative_current = 0 if current == your_index else 1
        else:
            relative_current = -1
        tracked = (
            "turn",
            "turnActionCount",
            "firstPlayer",
            f"players[{your_index}].handCount",
            f"players[{opponent_index}].handCount",
            f"players[{your_index}].deckCount",
            f"players[{opponent_index}].deckCount",
            f"players[{your_index}].prize",
            f"players[{opponent_index}].prize",
            "select.type",
            "select.context",
            "select.minCount",
            "select.maxCount",
        )
        return GlobalStateView(
            turn=snapshot.turn,
            turn_action_count=snapshot.turn_action_count,
            is_first_player=snapshot.first_player == your_index,
            relative_current_player=relative_current,
            self_hand_count=int(self_counts.get("hand", 0)),
            opponent_hand_count=int(opponent_counts.get("hand", 0)),
            self_deck_count=int(self_counts.get("deck", 0)),
            opponent_deck_count=int(opponent_counts.get("deck", 0)),
            self_prize_count=int(self_counts.get("prize", 0)),
            opponent_prize_count=int(opponent_counts.get("prize", 0)),
            self_bench_count=int(self_counts.get("bench", 0)),
            opponent_bench_count=int(opponent_counts.get("bench", 0)),
            energy_attached=snapshot.energy_attached,
            supporter_played=snapshot.supporter_played,
            stadium_played=snapshot.stadium_played,
            retreated=snapshot.retreated,
            select_type=snapshot.select_type,
            select_context=snapshot.select_context,
            remain_damage_counter=snapshot.remain_damage_counter,
            remain_energy_cost=snapshot.remain_energy_cost,
            min_count=snapshot.select_min_count,
            max_count=snapshot.select_max_count,
            field_visibility_mask=tuple(_present(snapshot.field_states.get(name)) for name in tracked),
        )

    def history(self, parsed: ParsedObservation, memory: GameMemoryState) -> HistoryView:
        counts = Counter(event.event_type for event in memory.recent_events if 0 <= event.event_type < 24)
        public_cards = Counter(
            int(event.card_id)
            for event in memory.recent_events
            if event.identity_visible and event.card_id is not None
        )
        recent = tuple(
            PublicEventView(
                event_type=int(event.event_type),
                relative_player=int(event.actor_relative if event.actor_relative in (0, 1) else -1),
                card_index=self.vocabulary.encode(event.card_id if event.identity_visible else None),
                serial=event.serial if event.identity_visible else None,
                from_zone=int(event.from_area or 0),
                to_zone=int(event.to_area or 0),
                turn_age=int(event.turn_age),
            )
            for event in memory.recent_events[-32:]
        )
        changes = tuple(
            (int(serial), int(item.previous_area or 0), int(item.current_area or 0))
            for serial, item in sorted(memory.serials.items())
            if item.previous_area is not None and item.current_area != item.previous_area
        )
        return HistoryView(
            recent_events=recent,
            event_type_counts=tuple(counts.get(index, 0) for index in range(24)),
            public_card_id_counts=tuple(sorted(public_cards.items())),
            public_serial_zone_changes=changes,
            current_turn_action_position=parsed.global_snapshot.turn_action_count,
        )

    @staticmethod
    def _option_reference(
        option: dict[str, object],
        cards: tuple[CardInstanceView, ...],
        your_index: int,
    ) -> CardInstanceView | None:
        serial = _int(option.get("serial"), -1)
        if serial >= 0:
            matches = [card for card in cards if card.serial == serial]
            if len(matches) == 1:
                return matches[0]
        option_type = _int(option.get("type"))
        absolute_player = _int(option.get("playerIndex"), your_index)
        relative_owner = int(RelativeOwner.SELF if absolute_player == your_index else RelativeOwner.OPPONENT)
        if option_type in {7, 8, 9}:
            area = AREA_IDS["HAND"]
            relative_owner = int(RelativeOwner.SELF)
            position = _int(option.get("index"))
        elif option_type in {3, 10, 11}:
            area = _int(option.get("area"))
            if area == AREA_IDS["DECK"]:
                area = AREA_IDS["LOOKING"]
            position = _int(option.get("index"))
        elif option_type in {4, 5}:
            parent_area = _int(option.get("area"))
            parent_position = _int(option.get("index"))
            parents = [
                card
                for card in cards
                if card.relative_owner == relative_owner
                and card.zone == parent_area
                and card.zone_position == parent_position
                and card.attached_to_serial is None
            ]
            if len(parents) != 1 or parents[0].serial is None:
                return None
            area = AREA_IDS["TOOL"] if option_type == 4 else AREA_IDS["ENERGY"]
            child_field = "toolIndex" if option_type == 4 else "energyIndex"
            child_position = _int(option.get(child_field))
            children = [
                card
                for card in cards
                if card.relative_owner == relative_owner
                and card.zone == area
                and card.zone_position == child_position
                and card.attached_to_serial == parents[0].serial
            ]
            return children[0] if len(children) == 1 else None
        else:
            return None
        matches = [
            card
            for card in cards
            if card.relative_owner == relative_owner
            and card.zone == area
            and card.zone_position == position
        ]
        return matches[0] if len(matches) == 1 else None

    def options(
        self,
        parsed: ParsedObservation,
        cards: tuple[CardInstanceView, ...],
        equivalence_groups: tuple[int, ...],
    ) -> tuple[OptionView, ...]:
        snapshot = parsed.global_snapshot
        views = []
        for index, option in enumerate(parsed.select_options):
            reference = self._option_reference(option, cards, snapshot.your_index)
            states = parsed.option_field_states[index]
            fields = ("type", "area", "index", "playerIndex", "cardId", "serial", "count", "number")
            absolute_player = _int(option.get("playerIndex"), snapshot.your_index)
            relative_player = 0 if absolute_player == snapshot.your_index else 1
            direct_card_id = _int(option.get("cardId"), -1)
            card_id = reference.card_id if reference is not None else direct_card_id if direct_card_id >= 0 else None
            direct_serial = _int(option.get("serial"), -1)
            serial = reference.serial if reference is not None else direct_serial if direct_serial >= 0 else None
            damage = next(
                (
                    float(_int(option.get(name), 0))
                    for name in ("damage", "value", "count", "number")
                    if name in option
                ),
                0.0,
            )
            views.append(
                OptionView(
                    original_option_index=index,
                    option_type=_int(option.get("type")),
                    select_type=snapshot.select_type,
                    select_context=snapshot.select_context,
                    relative_player=relative_player,
                    area=_int(option.get("area"), 0),
                    position_index=_int(option.get("index"), -1),
                    card_id=card_id,
                    card_index=self.vocabulary.encode(card_id),
                    serial=serial,
                    energy_type=_int(option.get("energyType"), -1),
                    damage_value=damage,
                    has_card_reference=reference is not None or card_id is not None,
                    has_serial_reference=serial is not None,
                    equivalence_group=(equivalence_groups[index] if index < len(equivalence_groups) else index),
                    field_visibility_mask=tuple(_present(states.get(name)) for name in fields),
                )
            )
        return tuple(views)
