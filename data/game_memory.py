from __future__ import annotations

import copy
from dataclasses import dataclass, field

from .state_schema import (
    CARD_APPEARANCE_FEATURE_DIM,
    EVENT_FEATURE_DIM,
    LEDGER_FEATURE_DIM,
    CardInstanceState,
    GameEvent,
    ParsedObservation,
    MAX_RECENT_EVENTS,
)
from .decision_schema import AnonymousHiddenPoolsRecord, CardIdMemoryRecord


@dataclass
class SerialMemory:
    serial: int
    card_id: int | None = None
    player_index: int | None = None
    first_seen_turn: int = 0
    last_seen_turn: int = 0
    current_area: int | None = None
    previous_area: int | None = None
    possible_hidden_zone_mask: int = 0
    currently_visible: bool = False
    last_seen_observation: int = 0
    last_event_type: int | None = None
    seen_count: int = 0
    moved_count: int = 0
    played: bool = False
    attached: bool = False
    evolved: bool = False
    attacked: bool = False
    damaged: bool = False
    discarded_public: bool = False


@dataclass
class GameMemoryState:
    serials: dict[int, SerialMemory] = field(default_factory=dict)
    recent_events: list[GameEvent] = field(default_factory=list)
    public_counts_by_player: dict[int, dict[str, int]] = field(default_factory=dict)
    max_recent_events: int = MAX_RECENT_EVENTS
    observation_count: int = 0
    anonymous_zone_transitions: dict[int, dict[str, int]] = field(default_factory=dict)

    def update_from_parsed(self, parsed: ParsedObservation) -> "GameMemoryState":
        self.observation_count += 1
        turn = parsed.global_snapshot.turn
        for event in self.recent_events:
            event.observation_age += 1
            event.turn_delta = max(0, turn - event.observed_turn)
        for player_index, counts in enumerate(parsed.global_snapshot.player_counts):
            self.public_counts_by_player[player_index] = dict(counts)
        # The batch logs describe transitions leading to the current snapshot.
        # Apply them first, then let exact current visibility set the final zone.
        for event in parsed.events:
            self._apply_event(event, turn)
        for memory in self.serials.values():
            memory.currently_visible = False
        for instance in parsed.card_instances:
            if instance.serial is None:
                continue
            memory = self.serials.get(instance.serial)
            if memory is None:
                memory = SerialMemory(
                    serial=instance.serial,
                    card_id=instance.card_id,
                    player_index=instance.player_index,
                    first_seen_turn=turn,
                )
                self.serials[instance.serial] = memory
            if instance.card_id is not None:
                memory.card_id = instance.card_id
            memory.player_index = instance.player_index
            memory.last_seen_turn = turn
            memory.last_seen_observation = self.observation_count
            if memory.current_area != instance.area:
                memory.previous_area = memory.current_area
            memory.current_area = instance.area
            memory.possible_hidden_zone_mask = 0
            memory.currently_visible = bool(instance.is_visible)
            memory.seen_count += 1
            if instance.zone == "discard":
                memory.discarded_public = True
        if parsed.events:
            # Memory owns event age/turn-delta updates; ParsedObservation remains
            # an immutable snapshot of the arrival batch.
            self.recent_events.extend(copy.deepcopy(parsed.events))
            self.recent_events = self.recent_events[-self.max_recent_events :]
        return self

    def _memory_for_event(self, event: GameEvent, turn: int) -> SerialMemory | None:
        if event.serial is None:
            return None
        memory = self.serials.get(event.serial)
        if memory is None:
            memory = SerialMemory(
                serial=event.serial,
                card_id=event.card_id,
                player_index=event.player_index,
                first_seen_turn=turn,
            )
            self.serials[event.serial] = memory
        if event.card_id is not None:
            memory.card_id = event.card_id
        if event.player_index is not None:
            memory.player_index = event.player_index
        memory.last_seen_turn = turn
        return memory

    def _apply_event(self, event: GameEvent, turn: int) -> None:
        if event.card_id is None and event.player_index is not None:
            zone_names = {1: "deck", 2: "hand", 6: "prize"}
            quantity = event.raw.get("quantity", 1)
            try:
                quantity = max(0, int(quantity))
            except (TypeError, ValueError):
                quantity = 1
            counters = self.anonymous_zone_transitions.setdefault(event.player_index, {})
            if event.from_area in zone_names:
                key = f"anonymous_{zone_names[event.from_area]}_out_count"
                counters[key] = counters.get(key, 0) + quantity
            if event.to_area in zone_names:
                key = f"anonymous_{zone_names[event.to_area]}_in_count"
                counters[key] = counters.get(key, 0) + quantity
        memory = self._memory_for_event(event, turn)
        if memory is None:
            if event.from_area is not None and event.player_index is not None:
                possible_mask = 1 << int(event.from_area)
                if event.to_area is not None:
                    possible_mask |= 1 << int(event.to_area)
                for candidate in self.serials.values():
                    if (
                        candidate.player_index == event.player_index
                        and candidate.current_area == event.from_area
                    ):
                        candidate.previous_area = candidate.current_area
                        candidate.current_area = None
                        candidate.possible_hidden_zone_mask |= possible_mask
                        candidate.last_event_type = event.event_type
            return
        memory.last_event_type = event.event_type
        memory.last_seen_observation = self.observation_count
        if event.to_area is not None:
            if memory.current_area != event.to_area:
                memory.previous_area = memory.current_area
            memory.current_area = event.to_area
            memory.possible_hidden_zone_mask = 0
        if event.from_area is not None or event.to_area is not None:
            memory.moved_count += 1
        if event.event_type == 10:
            memory.played = True
        elif event.event_type == 11:
            memory.attached = True
        elif event.event_type == 12:
            memory.evolved = True
        elif event.event_type == 15:
            memory.attacked = True
        elif event.event_type == 16:
            memory.damaged = True
        if event.to_area == 3:
            memory.discarded_public = True

    def anonymous_hidden_pools_record(self, your_index: int) -> AnonymousHiddenPoolsRecord:
        """Derive current unknown pool sizes and retain anonymous flow facts."""

        opponent_index = 1 - your_index if your_index in (0, 1) else -1

        def public(player_index: int, zone: str) -> int:
            return int(self.public_counts_by_player.get(player_index, {}).get(zone, 0))

        def exact(player_index: int, area: int) -> int:
            return sum(
                memory.player_index == player_index and memory.current_area == area
                for memory in self.serials.values()
            )

        transitions: dict[int, dict[str, int]] = {}
        for absolute_player, counts in self.anonymous_zone_transitions.items():
            relative = 0 if absolute_player == your_index else 1
            transitions[relative] = dict(counts)
        return AnonymousHiddenPoolsRecord(
            self_unresolved_deck_prize_count=max(
                0,
                public(your_index, "deck")
                + public(your_index, "prize")
                - exact(your_index, 1)
                - exact(your_index, 6),
            ),
            opponent_unknown_hand_count=max(
                0, public(opponent_index, "hand") - exact(opponent_index, 2)
            ),
            opponent_unknown_deck_count=max(
                0, public(opponent_index, "deck") - exact(opponent_index, 1)
            ),
            opponent_unknown_prize_count=max(
                0, public(opponent_index, "prize") - exact(opponent_index, 6)
            ),
            anonymous_zone_transitions_by_side=transitions,
        )

    def card_id_memory_records(
        self,
        your_index: int,
        *,
        expected_zone_counts: dict[tuple[int, int], dict[str, float]] | None = None,
        presence_predictions: dict[tuple[int, int], float] | None = None,
        uncertainty: dict[tuple[int, int], float] | None = None,
    ) -> list[CardIdMemoryRecord]:
        """Derive Card ID memory from the serial registry without mutable ledger state."""

        expected_zone_counts = expected_zone_counts or {}
        presence_predictions = presence_predictions or {}
        uncertainty = uncertainty or {}

        area_names = {
            1: "DECK",
            2: "HAND",
            3: "DISCARD",
            4: "ACTIVE",
            5: "BENCH",
            6: "PRIZE",
            7: "STADIUM",
            8: "ENERGY",
            9: "TOOL",
            10: "PRE_EVOLUTION",
            12: "LOOKING",
        }
        grouped: dict[tuple[int, int], list[SerialMemory]] = {}
        for memory in self.serials.values():
            if memory.card_id is None or memory.player_index is None:
                continue
            owner_relative = 0 if memory.player_index == your_index else 1
            grouped.setdefault((owner_relative, int(memory.card_id)), []).append(memory)
        keys = set(grouped) | set(expected_zone_counts) | set(presence_predictions) | set(uncertainty)
        records = []
        for owner_relative, card_id in sorted(keys):
            memories = grouped.get((owner_relative, card_id), [])
            exact_counts: dict[str, int] = {}
            for memory in memories:
                if memory.current_area in area_names:
                    name = area_names[int(memory.current_area)]
                    exact_counts[name] = exact_counts.get(name, 0) + 1
            records.append(
                CardIdMemoryRecord(
                    owner_relative=owner_relative,
                    card_id=card_id,
                    exact_zone_counts=exact_counts,
                    ambiguous_hidden_count=sum(
                        memory.current_area is None and memory.possible_hidden_zone_mask != 0
                        for memory in memories
                    ),
                    expected_zone_counts=dict(
                        expected_zone_counts.get((owner_relative, card_id), {})
                    ),
                    presence_prediction=presence_predictions.get((owner_relative, card_id)),
                    uncertainty=uncertainty.get((owner_relative, card_id)),
                    revealed_unique_copy_count=len(memories),
                    historical_seen_count=sum(memory.seen_count for memory in memories),
                    historical_move_count=sum(memory.moved_count for memory in memories),
                    first_seen_turn=(
                        min(memory.first_seen_turn for memory in memories) if memories else None
                    ),
                    last_seen_turn=(
                        max(memory.last_seen_turn for memory in memories) if memories else None
                    ),
                )
            )
        return records

    def appearance_features(self, instances: list[CardInstanceState]) -> list[list[float]]:
        rows: list[list[float]] = []
        moved_serials = {event.serial for event in self.recent_events if event.serial is not None}
        for instance in instances:
            memory = self.serials.get(instance.serial) if instance.serial is not None else None
            features = [0.0] * CARD_APPEARANCE_FEATURE_DIM
            features[0] = float(memory is not None)
            features[1] = float(instance.zone in {"active", "bench", "discard", "stadium"})
            features[2] = float(instance.is_face_down or instance.zone in {"prize"})
            features[3] = float(instance.zone in {"active", "bench"})
            features[4] = float(instance.attached_to_serial is not None)
            features[5] = float(instance.relative_player == 0)
            features[6] = float(instance.relative_player == 1)
            features[7] = float(instance.serial is not None)
            features[8] = float(instance.card_id is not None)
            if memory is not None:
                features[9] = memory.first_seen_turn / 100.0
                features[10] = memory.last_seen_turn / 100.0
                features[11] = float(instance.serial in moved_serials)
                features[12] = float(memory.played)
                features[13] = float(memory.attached)
                features[14] = float(memory.evolved)
                features[15] = float(memory.attacked)
                features[16] = float(memory.damaged)
                features[17] = float(memory.discarded_public)
                features[18] = min(memory.seen_count, 10) / 10.0
                features[19] = min(memory.moved_count, 10) / 10.0
            rows.append(features)
        return rows

    def ledger_features(self, your_index: int) -> list[list[float]]:
        rows = []
        for relative_player in [0, 1]:
            player_index = your_index if relative_player == 0 else 1 - your_index
            memories = [memory for memory in self.serials.values() if memory.player_index == player_index]
            features = [0.0] * LEDGER_FEATURE_DIM
            features[0] = len(memories) / 60.0
            features[1] = sum(memory.played for memory in memories) / 60.0
            features[2] = sum(memory.attached for memory in memories) / 60.0
            features[3] = sum(memory.evolved for memory in memories) / 20.0
            features[4] = sum(memory.attacked for memory in memories) / 20.0
            features[5] = sum(memory.damaged for memory in memories) / 20.0
            features[6] = sum(memory.discarded_public for memory in memories) / 60.0
            features[7] = sum(memory.current_area == 2 for memory in memories) / 20.0
            features[8] = sum(memory.current_area == 3 for memory in memories) / 60.0
            features[9] = sum(memory.current_area in {4, 5} for memory in memories) / 6.0
            features[10] = sum(memory.current_area in {8, 9, 10} for memory in memories) / 20.0
            features[11] = sum(memory.current_area == 7 for memory in memories)
            features[12] = min(sum(memory.seen_count for memory in memories), 100) / 100.0
            features[13] = min(sum(memory.moved_count for memory in memories), 100) / 100.0
            public_counts = self.public_counts_by_player.get(player_index, {})
            features[14] = public_counts.get("deck", 0) / 60.0
            features[15] = public_counts.get("hand", 0) / 20.0
            features[16] = public_counts.get("prize", 0) / 6.0
            features[17] = public_counts.get("bench", 0) / 5.0
            features[18] = public_counts.get("active", 0)
            features[19] = float(relative_player)
            rows.append(features)
        return rows

    def recent_event_features(self) -> list[list[float]]:
        rows: list[list[float]] = []
        for event in self.recent_events[-self.max_recent_events :]:
            features = [0.0] * EVENT_FEATURE_DIM
            features[0] = event.event_type / 24.0
            features[1] = 0.0 if event.actor_relative is None else float(event.actor_relative)
            features[2] = float(event.card_id or 0) / 1300.0
            features[3] = float(event.serial or 0) / 100.0
            features[4] = float(event.from_area or 0) / 12.0
            features[5] = float(event.to_area or 0) / 12.0
            features[6] = float(event.target_card_id or 0) / 1300.0
            features[7] = float(event.target_serial or 0) / 100.0
            features[8] = float(event.attack_id or 0) / 500.0
            features[9] = float(event.value or 0) / 400.0
            features[10] = float(event.is_reverse)
            features[11] = float(event.event_type in {10, 11, 12, 15})
            features[12] = float(event.event_type in {16, 17, 18, 19, 20, 21})
            features[13] = float(event.event_type in {4, 5, 6, 7})
            features[14] = min(event.observation_age, 32) / 32.0
            features[15] = min(event.event_position_in_batch, 32) / 32.0
            rows.append(features)
        return rows
