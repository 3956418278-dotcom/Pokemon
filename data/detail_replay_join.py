from __future__ import annotations

import copy
import csv
import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Hashable
from dataclasses import asdict, dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any, Iterable

from .game_memory import GameMemoryState
from .legal_options import option_equivalence_key
from .replay_dataset import ReplayDecisionSample
from .state_schema import CardInstanceState, GameEvent, ParsedObservation
from .static_detail_catalog import DetailResolution, EngineReference, StaticDetailCatalog


class OperationKind(str, Enum):
    DETAIL_DIRECT = "DETAIL_DIRECT"
    DETAIL_TRIGGER = "DETAIL_TRIGGER"
    NON_DETAIL_GAME_ACTION = "NON_DETAIL_GAME_ACTION"
    UNKNOWN_EFFECT = "UNKNOWN_EFFECT"


class DetailMappingStatus(str, Enum):
    EXACT = "EXACT"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    UNRESOLVED = "UNRESOLVED"
    CONFLICT = "CONFLICT"
    UNKNOWN_CARD = "UNKNOWN_CARD"


@dataclass
class DetailReplayJoinRecord:
    episode_id: int | None
    decision_index: int
    step_start: int
    step_end: int
    actor: int
    operation_kind: str
    source_serial: int | None
    source_card_id: int | None
    engine_detail_reference: dict[str, Any] | None
    resolved_detail_id: int | None
    resolved_detail_index: int
    detail_mapping_status: str
    detail_supervision_mask: bool
    transition_supervision_mask: bool
    state_before: dict[str, Any]
    state_after: dict[str, Any] | None
    state_delta: dict[str, Any]
    recent_events: list[dict[str, Any]]
    replay_key: str
    source_date: str | None
    split: str
    reason: str | None = None
    candidate_detail_ids: list[int] | None = None
    decision_span_count: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class _ReferenceMatch:
    reference: EngineReference
    raw_reference: dict[str, Any]
    resolution: DetailResolution
    origin: str
    source_card_id: int | None = None
    source_serial: int | None = None


def assign_episode_splits(
    samples: Iterable[ReplayDecisionSample],
    *,
    validation_fraction: float = 0.1,
    test_fraction: float = 0.1,
    held_out_date: str | None = None,
) -> dict[str, str]:
    """Assign whole episodes to one deterministic split.

    A supplied held-out date takes precedence for test. Remaining episodes are
    split by a stable replay-key hash, never by individual transition.
    """

    if validation_fraction < 0 or test_fraction < 0 or validation_fraction + test_fraction >= 1:
        raise ValueError("validation_fraction and test_fraction must be non-negative and sum below 1")
    episode_dates: dict[str, str | None] = {}
    for sample in samples:
        previous = episode_dates.get(sample.replay_key)
        if previous is not None and sample.source_date is not None and previous != sample.source_date:
            raise ValueError(f"episode {sample.replay_key!r} has multiple source dates")
        if previous is None and sample.source_date is not None:
            episode_dates[sample.replay_key] = sample.source_date
        else:
            episode_dates.setdefault(sample.replay_key, previous)
    result: dict[str, str] = {}
    validation_limit = int(round(validation_fraction * 10_000))
    test_limit = validation_limit + int(round(test_fraction * 10_000))
    for replay_key, source_date in sorted(episode_dates.items()):
        if held_out_date is not None and source_date is not None and source_date >= held_out_date:
            split = "test"
        else:
            bucket = int.from_bytes(hashlib.sha256(replay_key.encode("utf-8")).digest()[:8], "big") % 10_000
            if held_out_date is not None:
                split = "validation" if bucket < validation_limit else "train"
            elif bucket < validation_limit:
                split = "validation"
            elif bucket < test_limit:
                split = "test"
            else:
                split = "train"
        result[replay_key] = split
    return result


def _int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _selected_options(sample: ReplayDecisionSample) -> list[dict[str, Any]]:
    options = (sample.observation.get("select") or {}).get("option") or []
    selected: list[dict[str, Any]] = []
    for index in sample.action:
        if 0 <= int(index) < len(options) and isinstance(options[int(index)], dict):
            selected.append(options[int(index)])
    return selected


def _source_from_options(sample: ReplayDecisionSample) -> tuple[int | None, int | None]:
    identities: set[tuple[int | None, int | None]] = set()
    for option in _selected_options(sample):
        key = option_equivalence_key(sample.observation, option)
        source = key.get("source_identity_and_dynamic_state")
        if isinstance(source, dict):
            card_id = _int(source.get("id", source.get("cardId")))
            serial = _int(source.get("serial"))
            if card_id is not None or serial is not None:
                identities.add((card_id, serial))
        direct_card_id = _int(option.get("cardId"))
        direct_serial = _int(option.get("serial"))
        if direct_card_id is not None or direct_serial is not None:
            identities.add((direct_card_id, direct_serial))
    if len(identities) == 1:
        return next(iter(identities))
    return None, None


def _source_from_effect(sample: ReplayDecisionSample) -> tuple[int | None, int | None]:
    reference = sample.parsed.effect_reference
    if not isinstance(reference, dict):
        return None, None
    return _int(reference.get("id", reference.get("cardId"))), _int(reference.get("serial"))


def _unique_instance_for_card(
    parsed: ParsedObservation,
    *,
    card_id: int,
    actor: int,
) -> CardInstanceState | None:
    candidates = [
        instance
        for instance in parsed.card_instances
        if instance.card_id == card_id and instance.player_index == actor and instance.serial is not None
    ]
    active = [instance for instance in candidates if instance.zone == "active" or instance.area == 4]
    if len(active) == 1:
        return active[0]
    return candidates[0] if len(candidates) == 1 else None


def _reference_matches(
    sample: ReplayDecisionSample,
    catalog: StaticDetailCatalog,
) -> list[_ReferenceMatch]:
    source_card_id, source_serial = _source_from_options(sample)
    matches: list[_ReferenceMatch] = []
    seen: set[tuple[EngineReference, str]] = set()
    selected_attack_ids = {
        attack_id
        for option in _selected_options(sample)
        if (attack_id := _int(option.get("attackId"))) is not None
    }

    def add(
        reference: EngineReference,
        raw: dict[str, Any],
        origin: str,
        *,
        card_id: int | None = None,
        serial: int | None = None,
    ) -> None:
        key = (reference, origin)
        if key in seen:
            return
        seen.add(key)
        matches.append(
            _ReferenceMatch(
                reference=reference,
                raw_reference=copy.deepcopy(raw),
                resolution=catalog.resolve(reference, card_id=card_id),
                origin=origin,
                source_card_id=card_id,
                source_serial=serial,
            )
        )

    for option in _selected_options(sample):
        card_id = _int(option.get("cardId")) or source_card_id
        serial = _int(option.get("serial")) or source_serial
        option_type = _int(option.get("type"))
        card_record = catalog.card_record(card_id)
        if (
            option_type == 7
            and card_id is not None
            and card_record is not None
            and card_record.get("card_type") in {"ITEM", "TOOL", "SUPPORTER", "STADIUM"}
        ):
            add(
                ("card_effect", card_id),
                {"cardId": card_id, "serial": serial, "optionType": option_type},
                "SELECTED_OPTION",
                card_id=card_id,
                serial=serial,
            )
        attack_id = _int(option.get("attackId"))
        if attack_id is not None:
            reference = (
                ("card_attack_id", card_id, attack_id)
                if card_id is not None
                else ("attack_id", attack_id)
            )
            add(reference, {"attackId": attack_id, "cardId": card_id, "serial": serial}, "SELECTED_OPTION", card_id=card_id, serial=serial)
        local_index = _int(option.get("detailLocalIndex"))
        if local_index is None:
            local_index = _int(option.get("detailIndex"))
        if local_index is not None and card_id is not None:
            add(
                ("card_detail_local_index", card_id, local_index),
                {"detailLocalIndex": local_index, "cardId": card_id, "serial": serial},
                "SELECTED_OPTION",
                card_id=card_id,
                serial=serial,
            )

    after = sample.transition_parsed_after
    if after is not None:
        for event in after.events:
            raw = event.raw if isinstance(event.raw, dict) else {}
            card_id = event.card_id if event.card_id is not None else _int(raw.get("cardId"))
            serial = event.serial if event.serial is not None else _int(raw.get("serial"))
            if event.attack_id is not None and int(event.attack_id) in selected_attack_ids:
                reference = (
                    ("card_attack_id", card_id, int(event.attack_id))
                    if card_id is not None
                    else ("attack_id", int(event.attack_id))
                )
                add(reference, raw, "POST_EVENT", card_id=card_id, serial=serial)
            local_index = _int(raw.get("detailLocalIndex"))
            if local_index is None:
                local_index = _int(raw.get("detailIndex"))
            if local_index is not None and card_id is not None:
                add(
                    ("card_detail_local_index", card_id, local_index),
                    raw,
                    "POST_EVENT",
                    card_id=card_id,
                    serial=serial,
                )
    return matches


def _unmapped_engine_reference(sample: ReplayDecisionSample) -> dict[str, Any] | None:
    selected_refs = []
    for option in _selected_options(sample):
        reference = {
            key: option[key]
            for key in ("cardId", "serial", "skillId", "abilityId", "detailId")
            if key in option and option[key] not in (None, -1)
        }
        if any(key in reference for key in ("skillId", "abilityId", "detailId")):
            selected_refs.append(reference)
    if selected_refs:
        return {"origin": "SELECTED_OPTION", "references": selected_refs}
    if sample.parsed.effect_reference is not None:
        return {"origin": "SELECT_EFFECT", "reference": copy.deepcopy(sample.parsed.effect_reference)}
    after = sample.transition_parsed_after
    if after is not None:
        raw_refs = []
        for event in after.events:
            raw = event.raw if isinstance(event.raw, dict) else {}
            reference = {
                key: raw[key]
                for key in ("cardId", "serial", "skillId", "abilityId", "detailId")
                if key in raw and raw[key] not in (None, -1)
            }
            if any(key in reference for key in ("skillId", "abilityId", "detailId")):
                raw_refs.append(reference)
        if raw_refs:
            return {"origin": "POST_EVENT", "references": raw_refs}
    return None


def _effectful_card_play(
    sample: ReplayDecisionSample,
    catalog: StaticDetailCatalog,
    source_card_id: int | None,
) -> bool:
    if source_card_id is None:
        return False
    record = catalog.card_record(source_card_id)
    if record is None or not record.get("detail_ids"):
        return False
    selected_types = {_int(option.get("type")) for option in _selected_options(sample)}
    return 7 in selected_types and record.get("card_type") in {
        "ITEM",
        "TOOL",
        "SUPPORTER",
        "STADIUM",
        "SPECIAL_ENERGY",
    }


def _snapshot(parsed: ParsedObservation, memory: GameMemoryState) -> dict[str, Any]:
    instances = []
    for instance in parsed.card_instances:
        instances.append(
            {
                "card_id": instance.card_id,
                "serial": instance.serial,
                "player_index": instance.player_index,
                "area": instance.area,
                "zone": instance.zone,
                "slot": instance.slot,
                "hp": instance.hp,
                "max_hp": instance.max_hp,
                "energy_card_ids": list(instance.energy_card_ids),
                "tool_card_ids": list(instance.tool_card_ids),
                "pre_evolution_card_ids": list(instance.pre_evolution_card_ids),
                "attached_to_serial": instance.attached_to_serial,
                "attachment_kind": instance.attachment_kind,
            }
        )
    serial_registry = {
        str(serial): {
            "card_id": item.card_id,
            "player_index": item.player_index,
            "current_area": item.current_area,
            "previous_area": item.previous_area,
            "played": item.played,
            "attached": item.attached,
            "evolved": item.evolved,
            "attacked": item.attacked,
            "damaged": item.damaged,
        }
        for serial, item in sorted(memory.serials.items())
    }
    global_snapshot = parsed.global_snapshot
    return {
        "turn": global_snapshot.turn,
        "turn_action_count": global_snapshot.turn_action_count,
        "your_index": global_snapshot.your_index,
        "player_counts": copy.deepcopy(global_snapshot.player_counts),
        "card_instances": instances,
        "serial_registry": serial_registry,
        "memory_observation_count": memory.observation_count,
    }


def _state_delta(before: dict[str, Any], after: dict[str, Any] | None) -> dict[str, Any]:
    if after is None:
        return {"status": "MISSING_POST_ACTION_STATE"}
    before_instances = {
        str(item["serial"]): item for item in before["card_instances"] if item["serial"] is not None
    }
    after_instances = {
        str(item["serial"]): item for item in after["card_instances"] if item["serial"] is not None
    }
    changed = []
    for serial in sorted(set(before_instances) & set(after_instances), key=int):
        fields = {
            key: {"before": before_instances[serial].get(key), "after": after_instances[serial].get(key)}
            for key in sorted(set(before_instances[serial]) | set(after_instances[serial]))
            if before_instances[serial].get(key) != after_instances[serial].get(key)
        }
        if fields:
            changed.append({"serial": int(serial), "fields": fields})
    return {
        "status": "COMPLETE",
        "perspective_changed": before["your_index"] != after["your_index"],
        "turn_before": before["turn"],
        "turn_after": after["turn"],
        "turn_action_count_before": before["turn_action_count"],
        "turn_action_count_after": after["turn_action_count"],
        "added_serials": sorted(int(value) for value in set(after_instances) - set(before_instances)),
        "removed_serials": sorted(int(value) for value in set(before_instances) - set(after_instances)),
        "changed_serials": changed,
    }


def _event_payload(event: GameEvent) -> dict[str, Any]:
    return {
        "event_type": event.event_type,
        "player_index": event.player_index,
        "card_id": event.card_id,
        "serial": event.serial,
        "target_card_id": event.target_card_id,
        "target_serial": event.target_serial,
        "attack_id": event.attack_id,
        "observation_age": event.observation_age,
        "turn_age": event.turn_age,
        "raw": copy.deepcopy(event.raw),
    }


def _choose_exact(matches: list[_ReferenceMatch]) -> tuple[_ReferenceMatch | None, DetailResolution | None]:
    exact = [match for match in matches if match.resolution.status == "EXACT"]
    exact_indices = {match.resolution.detail_index for match in exact}
    if len(exact_indices) == 1:
        preferred = next((match for match in exact if match.origin == "SELECTED_OPTION"), exact[0])
        return preferred, preferred.resolution
    if len(exact_indices) > 1:
        ids = sorted({int(match.resolution.detail_id) for match in exact if match.resolution.detail_id is not None})
        return None, DetailResolution(
            status="CONFLICT",
            candidate_detail_ids=tuple(ids),
            reason="one transition contains multiple exact detail references",
        )
    conflicts = [match.resolution for match in matches if match.resolution.status == "CONFLICT"]
    if conflicts:
        candidates = sorted({detail_id for item in conflicts for detail_id in item.candidate_detail_ids})
        return None, DetailResolution(
            status="CONFLICT",
            candidate_detail_ids=tuple(candidates),
            reason="stable engine reference is non-unique",
        )
    unknown_cards = [match.resolution for match in matches if match.resolution.status == "UNKNOWN_CARD"]
    if unknown_cards:
        return None, unknown_cards[0]
    if matches:
        return None, matches[0].resolution
    return None, None


def join_replay_samples(
    samples: Iterable[ReplayDecisionSample],
    catalog: StaticDetailCatalog,
    *,
    episode_splits: dict[str, str] | None = None,
) -> list[DetailReplayJoinRecord]:
    samples = list(samples)
    splits = episode_splits or assign_episode_splits(samples)
    decision_counters: dict[str, int] = {}
    records: list[DetailReplayJoinRecord] = []
    for sample in samples:
        decision_index = decision_counters.get(sample.replay_key, 0)
        decision_counters[sample.replay_key] = decision_index + 1
        source_card_id, source_serial = _source_from_options(sample)
        effect_card_id, effect_serial = _source_from_effect(sample)
        if effect_card_id is not None or effect_serial is not None:
            source_card_id, source_serial = effect_card_id, effect_serial
        matches = _reference_matches(sample, catalog)
        match, resolution = _choose_exact(matches)
        raw_unmapped = _unmapped_engine_reference(sample)

        if match is not None and resolution is not None:
            entry = catalog.entries[resolution.detail_index]
            source_card_id = match.source_card_id or entry.parent_card_id
            source_serial = match.source_serial or source_serial
            if source_serial is None:
                instance = _unique_instance_for_card(
                    sample.parsed,
                    card_id=entry.parent_card_id,
                    actor=sample.agent_index,
                )
                source_serial = instance.serial if instance is not None else None
            operation = (
                OperationKind.DETAIL_DIRECT
                if match.origin == "SELECTED_OPTION" or entry.detail_type == "ATTACK"
                else OperationKind.DETAIL_TRIGGER
            )
            status = DetailMappingStatus.EXACT
            engine_reference = {
                "kind": match.reference[0],
                "value": list(match.reference[1:]),
                "origin": match.origin,
                "raw": match.raw_reference,
            }
            detail_supervision = True
            transition_supervision = True
            reason = None
        else:
            selected_source = catalog.card_record(source_card_id)
            unknown_source_card = source_card_id is not None and selected_source is None
            effect_related = bool(matches or raw_unmapped or _effectful_card_play(sample, catalog, source_card_id))
            if effect_related or unknown_source_card:
                operation = OperationKind.UNKNOWN_EFFECT
                if resolution is not None and resolution.status in {
                    "CONFLICT",
                    "UNKNOWN_CARD",
                }:
                    status = DetailMappingStatus(resolution.status)
                elif unknown_source_card:
                    status = DetailMappingStatus.UNKNOWN_CARD
                else:
                    status = DetailMappingStatus.UNRESOLVED
                engine_reference = raw_unmapped
                if engine_reference is None and matches:
                    engine_reference = {
                        "references": [
                            {
                                "kind": item.reference[0],
                                "value": list(item.reference[1:]),
                                "origin": item.origin,
                                "raw": item.raw_reference,
                            }
                            for item in matches
                        ]
                    }
                detail_supervision = False
                transition_supervision = True
                reason = (
                    resolution.reason
                    if resolution is not None
                    else "effect-related operation has no stable detail reference"
                )
            else:
                operation = OperationKind.NON_DETAIL_GAME_ACTION
                status = DetailMappingStatus.NOT_APPLICABLE
                engine_reference = None
                detail_supervision = False
                transition_supervision = False
                reason = None
            resolution = resolution or DetailResolution(status=status.value, reason=reason)

        before = _snapshot(sample.parsed, sample.memory_after)
        after = (
            _snapshot(sample.transition_parsed_after, sample.transition_memory_after)
            if sample.transition_parsed_after is not None and sample.transition_memory_after is not None
            else None
        )
        recent_memory = sample.transition_memory_after or sample.memory_after
        records.append(
            DetailReplayJoinRecord(
                episode_id=sample.episode_id,
                decision_index=decision_index,
                step_start=sample.step_index,
                step_end=sample.action_step_index if sample.action_step_index is not None else sample.step_index,
                actor=sample.agent_index,
                operation_kind=operation.value,
                source_serial=source_serial,
                source_card_id=source_card_id,
                engine_detail_reference=engine_reference,
                resolved_detail_id=resolution.detail_id,
                resolved_detail_index=resolution.detail_index,
                detail_mapping_status=status.value,
                detail_supervision_mask=detail_supervision,
                transition_supervision_mask=transition_supervision,
                state_before=before,
                state_after=after,
                state_delta=_state_delta(before, after),
                recent_events=[_event_payload(event) for event in recent_memory.recent_events],
                replay_key=sample.replay_key,
                source_date=sample.source_date,
                split=splits[sample.replay_key],
                reason=reason,
                candidate_detail_ids=list(resolution.candidate_detail_ids),
            )
        )
    return _coalesce_pending_chains(samples, records)


def _post_has_pending_selection(sample: ReplayDecisionSample) -> bool:
    observation = sample.transition_observation_after
    if not isinstance(observation, dict):
        return False
    select = observation.get("select")
    if not isinstance(select, dict):
        return False
    return (_int(select.get("type")), _int(select.get("context"))) != (0, 0)


def _sample_is_pending_selection(sample: ReplayDecisionSample) -> bool:
    select = sample.observation.get("select")
    if not isinstance(select, dict):
        return False
    return (_int(select.get("type")), _int(select.get("context"))) != (0, 0)


def _merge_chain(
    chain: list[DetailReplayJoinRecord],
) -> DetailReplayJoinRecord:
    first, last = chain[0], chain[-1]
    exact = [record for record in chain if record.detail_mapping_status == DetailMappingStatus.EXACT.value]
    exact_indices = {record.resolved_detail_index for record in exact}
    if len(exact_indices) > 1:
        candidate_ids = sorted(
            {int(record.resolved_detail_id) for record in exact if record.resolved_detail_id is not None}
        )
        selected = next(
            (record for record in chain if record.detail_mapping_status != DetailMappingStatus.NOT_APPLICABLE.value),
            first,
        )
        updates = {
            "operation_kind": OperationKind.UNKNOWN_EFFECT.value,
            "resolved_detail_id": None,
            "resolved_detail_index": -1,
            "detail_mapping_status": DetailMappingStatus.CONFLICT.value,
            "detail_supervision_mask": False,
            "transition_supervision_mask": True,
            "reason": "pending action chain contains multiple exact detail identities",
            "candidate_detail_ids": candidate_ids,
        }
    elif exact:
        selected = next(
            (record for record in exact if record.operation_kind == OperationKind.DETAIL_DIRECT.value),
            exact[0],
        )
        updates = {}
    else:
        status_priority = {
            DetailMappingStatus.CONFLICT.value: 0,
            DetailMappingStatus.UNKNOWN_CARD.value: 1,
            DetailMappingStatus.UNRESOLVED.value: 2,
            DetailMappingStatus.NOT_APPLICABLE.value: 3,
        }
        selected = min(chain, key=lambda record: status_priority[record.detail_mapping_status])
        updates = {}
    merged = replace(
        selected,
        decision_index=first.decision_index,
        step_start=first.step_start,
        step_end=last.step_end,
        state_before=first.state_before,
        state_after=last.state_after,
        state_delta=_state_delta(first.state_before, last.state_after),
        recent_events=last.recent_events,
        decision_span_count=sum(record.decision_span_count for record in chain),
        **updates,
    )
    return merged


def _coalesce_pending_chains(
    samples: list[ReplayDecisionSample],
    records: list[DetailReplayJoinRecord],
) -> list[DetailReplayJoinRecord]:
    if len(samples) != len(records):
        raise ValueError("sample and preliminary join record counts disagree")
    result: list[DetailReplayJoinRecord] = []
    index = 0
    while index < len(samples):
        chain_records = [records[index]]
        cursor = index
        while _post_has_pending_selection(samples[cursor]) and cursor + 1 < len(samples):
            current = samples[cursor]
            following = samples[cursor + 1]
            if (
                following.replay_key != current.replay_key
                or following.step_index != current.action_step_index
                or (
                    following.agent_index != current.agent_index
                    and not _sample_is_pending_selection(following)
                )
            ):
                break
            cursor += 1
            chain_records.append(records[cursor])
        result.append(_merge_chain(chain_records))
        index = cursor + 1
    return result


def _summary_for_records(
    records: list[DetailReplayJoinRecord],
    catalog: StaticDetailCatalog,
) -> dict[str, Any]:
    operation_counts = Counter(record.operation_kind for record in records)
    status_counts = Counter(record.detail_mapping_status for record in records)
    exact = [record for record in records if record.detail_mapping_status == "EXACT"]
    observed = {record.resolved_detail_index for record in exact}
    direct = {
        record.resolved_detail_index
        for record in exact
        if record.operation_kind == OperationKind.DETAIL_DIRECT.value
    }
    triggered = {
        record.resolved_detail_index
        for record in exact
        if record.operation_kind == OperationKind.DETAIL_TRIGGER.value
    }
    unknown_card_observations = {
        (record.replay_key, record.decision_index, card_id, serial)
        for record in records
        for card_id, serial, _location in _unknown_cards_in_record(record, catalog)
    }
    unknown_mapping_without_identity = sum(
        record.detail_mapping_status == DetailMappingStatus.UNKNOWN_CARD.value
        and record.source_card_id is None
        for record in records
    )
    return {
        "episode_count": len({record.replay_key for record in records}),
        "decision_count": sum(record.decision_span_count for record in records),
        "transition_count": len(records),
        "detail_direct_count": operation_counts[OperationKind.DETAIL_DIRECT.value],
        "detail_trigger_count": operation_counts[OperationKind.DETAIL_TRIGGER.value],
        "non_detail_game_action_count": operation_counts[OperationKind.NON_DETAIL_GAME_ACTION.value],
        "unknown_effect_count": operation_counts[OperationKind.UNKNOWN_EFFECT.value],
        "exact_mapping_count": status_counts[DetailMappingStatus.EXACT.value],
        "unresolved_mapping_count": status_counts[DetailMappingStatus.UNRESOLVED.value],
        "conflict_mapping_count": status_counts[DetailMappingStatus.CONFLICT.value],
        "unknown_card_count": len(unknown_card_observations) + unknown_mapping_without_identity,
        "unique_unknown_card_ids": len({item[2] for item in unknown_card_observations}),
        "unique_detail_total": len(catalog.entries),
        "unique_detail_observed": len(observed),
        "unique_detail_directly_used": len(direct),
        "unique_detail_triggered": len(triggered),
        "unique_detail_without_supervision": len(catalog.entries) - len(observed),
    }


def build_detail_join_audit(
    catalog: StaticDetailCatalog,
    records: Iterable[DetailReplayJoinRecord],
) -> dict[str, Any]:
    records = list(records)
    summary = _summary_for_records(records, catalog)
    summary["splits"] = {
        split: _summary_for_records(
            [record for record in records if record.split == split],
            catalog,
        )
        for split in ("train", "validation", "test")
    }
    return summary


def _coverage_rows(
    catalog: StaticDetailCatalog,
    records: list[DetailReplayJoinRecord],
) -> list[dict[str, Any]]:
    rows = []
    for split in ("all", "train", "validation", "test"):
        scoped = records if split == "all" else [record for record in records if record.split == split]
        by_detail: dict[int, list[DetailReplayJoinRecord]] = defaultdict(list)
        for record in scoped:
            if record.resolved_detail_index >= 0 and record.detail_mapping_status == "EXACT":
                by_detail[record.resolved_detail_index].append(record)
        for entry in catalog.entries:
            seen = by_detail.get(entry.detail_index, [])
            episodes = sorted({record.replay_key for record in seen})
            direct_count = sum(record.operation_kind == OperationKind.DETAIL_DIRECT.value for record in seen)
            trigger_count = sum(record.operation_kind == OperationKind.DETAIL_TRIGGER.value for record in seen)
            rows.append(
                {
                    "split": split,
                    "detail_id": entry.detail_id,
                    "detail_index": entry.detail_index,
                    "parent_card_id": entry.parent_card_id,
                    "detail_type": entry.detail_type,
                    "direct_use_count": direct_count,
                    "trigger_count": trigger_count,
                    "exact_supervision_count": direct_count + trigger_count,
                    "episode_count": len(episodes),
                    "first_seen_episode": episodes[0] if episodes else "",
                    "last_seen_episode": episodes[-1] if episodes else "",
                    "has_supervision": bool(seen),
                }
            )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _unknown_cards_in_record(
    record: DetailReplayJoinRecord,
    catalog: StaticDetailCatalog,
) -> list[tuple[int, int | None, str]]:
    found: set[tuple[int, int | None, str]] = set()
    if (
        record.detail_mapping_status == DetailMappingStatus.UNKNOWN_CARD.value
        and record.source_card_id is not None
    ):
        found.add((int(record.source_card_id), record.source_serial, "SOURCE"))
    for location, state in (("STATE_BEFORE", record.state_before), ("STATE_AFTER", record.state_after)):
        if not isinstance(state, dict):
            continue
        for instance in state.get("card_instances") or []:
            if not isinstance(instance, dict):
                continue
            card_id = _int(instance.get("card_id"))
            if card_id is not None and card_id not in catalog.card_id_to_index:
                found.add((card_id, _int(instance.get("serial")), location))
    return sorted(found, key=lambda item: (item[0], item[1] if item[1] is not None else -1, item[2]))


def _record_engine_references(record: DetailReplayJoinRecord) -> set[EngineReference]:
    payload = record.engine_detail_reference
    if not isinstance(payload, dict):
        return set()
    items = payload.get("references") if isinstance(payload.get("references"), list) else [payload]
    result = set()
    for item in items:
        if not isinstance(item, dict) or not isinstance(item.get("kind"), str):
            continue
        values = item.get("value") if isinstance(item.get("value"), list) else []
        if all(isinstance(value, Hashable) for value in values):
            result.add((item["kind"], *values))
    return result


def write_detail_join_audit(
    output_dir: str | Path,
    catalog: StaticDetailCatalog,
    records: Iterable[DetailReplayJoinRecord],
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = list(records)
    summary = build_detail_join_audit(catalog, records)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "joined_transitions.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False, separators=(",", ":")) + "\n")

    coverage = _coverage_rows(catalog, records)
    _write_csv(
        output_dir / "detail_coverage.csv",
        coverage,
        [
            "split",
            "detail_id",
            "detail_index",
            "parent_card_id",
            "detail_type",
            "direct_use_count",
            "trigger_count",
            "exact_supervision_count",
            "episode_count",
            "first_seen_episode",
            "last_seen_episode",
            "has_supervision",
        ],
    )
    unresolved_rows = [
        {
            "episode_id": record.episode_id,
            "decision_index": record.decision_index,
            "source_card_id": record.source_card_id,
            "source_serial": record.source_serial,
            "operation_kind": record.operation_kind,
            "raw_engine_reference": json.dumps(record.engine_detail_reference, ensure_ascii=False),
            "reason": record.reason,
            "candidate_detail_ids": json.dumps(record.candidate_detail_ids or []),
        }
        for record in records
        if record.detail_mapping_status == DetailMappingStatus.UNRESOLVED.value
    ]
    unresolved_fields = [
        "episode_id",
        "decision_index",
        "source_card_id",
        "source_serial",
        "operation_kind",
        "raw_engine_reference",
        "reason",
        "candidate_detail_ids",
    ]
    _write_csv(output_dir / "unresolved_references.csv", unresolved_rows, unresolved_fields)

    conflict_rows = [
        {
            "engine_reference": json.dumps(list(reference), ensure_ascii=False),
            "candidate_detail_indices": json.dumps(list(indices)),
            "candidate_detail_ids": json.dumps([catalog.entries[index].detail_id for index in indices]),
            "observed_transition_count": sum(
                record.detail_mapping_status == DetailMappingStatus.CONFLICT.value
                and reference in _record_engine_references(record)
                for record in records
            ),
        }
        for reference, indices in sorted(
            catalog.mapping_conflicts.items(), key=lambda item: json.dumps(list(item[0]), ensure_ascii=False)
        )
    ]
    _write_csv(
        output_dir / "mapping_conflicts.csv",
        conflict_rows,
        ["engine_reference", "candidate_detail_indices", "candidate_detail_ids", "observed_transition_count"],
    )
    unknown_rows = []
    for record in records:
        for card_id, serial, location in _unknown_cards_in_record(record, catalog):
            unknown_rows.append(
                {
                    "episode_id": record.episode_id,
                    "decision_index": record.decision_index,
                    "source_card_id": card_id,
                    "source_serial": serial,
                    "location": location,
                    "raw_engine_reference": json.dumps(record.engine_detail_reference, ensure_ascii=False),
                    "reason": record.reason or "Replay Card ID is absent from the static catalog",
                }
            )
    _write_csv(
        output_dir / "unknown_card_ids.csv",
        unknown_rows,
        [
            "episode_id",
            "decision_index",
            "source_card_id",
            "source_serial",
            "location",
            "raw_engine_reference",
            "reason",
        ],
    )
    distribution_rows = []
    for split in ("all", "train", "validation", "test"):
        scoped = records if split == "all" else [record for record in records if record.split == split]
        counts = Counter(record.operation_kind for record in scoped)
        for operation in OperationKind:
            distribution_rows.append(
                {"split": split, "operation_kind": operation.value, "count": counts[operation.value]}
            )
    _write_csv(
        output_dir / "operation_distribution.csv",
        distribution_rows,
        ["split", "operation_kind", "count"],
    )
    return summary
