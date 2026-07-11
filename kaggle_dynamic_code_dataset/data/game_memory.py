from __future__ import annotations

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


@dataclass
class SerialMemory:
    serial: int
    card_id: int | None = None
    player_index: int | None = None
    first_seen_turn: int = 0
    last_seen_turn: int = 0
    current_area: int | None = None
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

    def update_from_parsed(self, parsed: ParsedObservation) -> "GameMemoryState":
        turn = parsed.global_snapshot.turn
        for player_index, counts in enumerate(parsed.global_snapshot.player_counts):
            self.public_counts_by_player[player_index] = dict(counts)
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
            memory.current_area = instance.area
            memory.seen_count += 1
            if instance.zone == "discard":
                memory.discarded_public = True
        for event in parsed.events:
            self._apply_event(event, turn)
        if parsed.events:
            self.recent_events.extend(parsed.events)
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
        memory = self._memory_for_event(event, turn)
        if memory is None:
            return
        if event.to_area is not None:
            memory.current_area = event.to_area
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
            features[1] = -1.0 if event.player_index is None else float(event.player_index)
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
            rows.append(features)
        return rows
