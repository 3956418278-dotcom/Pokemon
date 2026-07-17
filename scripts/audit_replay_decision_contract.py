from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import multiprocessing as mp
import sys
import zipfile
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.decision_schema import ActionSemantics, FieldState
from data.legal_options import (
    build_action_target,
    infer_action_semantics,
    option_equivalence_signature,
    policy_loss_mask,
)

ORDER_AUDIT_CONTEXTS = {2, 15, 21, 22, 26, 27, 34}
CONTEXT_SEMANTICS = {
    2: {
        "effect_kind": "game setup",
        "candidate_meaning": "remaining Basic Pokemon in hand to place on the Bench",
        "order_changes_final_effect": "NO",
        "final_semantics": "UNORDERED_UNIQUE_SUBSET",
        "semantic_evidence": "bench placement is a set; both ascending and reversed legal orders occur",
    },
    15: {
        "effect_kind": "Slowking Seek Inspiration copying Kyurem Trifrost",
        "candidate_meaning": "three opposing Pokemon that each receive 110 damage",
        "order_changes_final_effect": "NO",
        "final_semantics": "UNORDERED_UNIQUE_SUBSET",
        "semantic_evidence": "Trifrost applies identical damage to a chosen target set",
    },
    21: {
        "effect_kind": "Glass Trumpet or Janine's Secret Art",
        "candidate_meaning": "Pokemon that each receive one equivalent Basic Energy",
        "order_changes_final_effect": "NO",
        "final_semantics": "UNORDERED_UNIQUE_SUBSET",
        "semantic_evidence": "card text chooses a Pokemon set; poison depends on membership, not order",
    },
    22: {
        "effect_kind": "multi-Energy search or recovery effect",
        "candidate_meaning": "Energy sources processed by subsequent target-selection decisions",
        "order_changes_final_effect": "YES",
        "final_semantics": "ORDERED_INDEX_SEQUENCE",
        "semantic_evidence": "the engine consumes selected Energy in action order and opens one target select per Energy",
    },
    26: {
        "effect_kind": "Erasure Ball or Bellowing Thunder attack cost/effect",
        "candidate_meaning": "attached Energy cards to discard",
        "order_changes_final_effect": "NO",
        "final_semantics": "UNORDERED_UNIQUE_SUBSET",
        "semantic_evidence": "damage depends on selected count; remaining attachment state depends on the set",
    },
    27: {
        "effect_kind": "Tool Scrapper",
        "candidate_meaning": "attached Pokemon Tools to discard",
        "order_changes_final_effect": "NO",
        "final_semantics": "UNORDERED_UNIQUE_SUBSET",
        "semantic_evidence": "the card resolves one chosen Tool set",
    },
    34: {
        "effect_kind": "Risky Ruins pending Stadium triggers",
        "candidate_meaning": "forced equivalent trigger occurrences",
        "order_changes_final_effect": "NO",
        "final_semantics": "UNORDERED_UNIQUE_SUBSET",
        "semantic_evidence": "all options are forced and equivalent; the policy loss is masked",
    },
}

_ZIP: zipfile.ZipFile | None = None
_WRITE_REFERENCE_DATASET = False


def _init_worker(archive: str, write_reference_dataset: bool) -> None:
    global _WRITE_REFERENCE_DATASET, _ZIP
    _ZIP = zipfile.ZipFile(archive)
    _WRITE_REFERENCE_DATASET = write_reference_dataset


def _int(value: Any, default: int = -1) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _field_state(obj: Any, name: str) -> str:
    if not isinstance(obj, dict) or name not in obj:
        return FieldState.MISSING.name
    value = obj[name]
    if value is None:
        return FieldState.EXPLICIT_NULL.name
    if isinstance(value, int) and value < 0:
        return FieldState.UNKNOWN.name
    return FieldState.PRESENT.name


def _distribution(counter: Counter[int]) -> dict[str, Any]:
    count = sum(counter.values())
    if count:
        targets = [0.5, 0.9, 0.99]
        quantiles: dict[float, float] = {}
        cumulative = 0
        ordered = sorted(counter.items())
        for value, frequency in ordered:
            cumulative += frequency
            for target in targets:
                if target not in quantiles and cumulative >= round((count - 1) * target) + 1:
                    quantiles[target] = float(value)
        return {
            "count": count,
            "mean": sum(value * frequency for value, frequency in counter.items()) / count,
            "min": min(counter),
            "p50": quantiles.get(0.5, float(max(counter))),
            "p90": quantiles.get(0.9, float(max(counter))),
            "p99": quantiles.get(0.99, float(max(counter))),
            "max": max(counter),
            "counts": {str(key): value for key, value in sorted(counter.items())},
        }
    return {"count": 0, "mean": 0.0, "min": 0, "p50": 0.0, "p90": 0.0, "p99": 0.0, "max": 0, "counts": {}}


def _payload(observation: dict[str, Any]) -> dict[str, Any]:
    return {
        "current": observation.get("current"),
        "logs": observation.get("logs"),
        "select": observation.get("select"),
    }


def _known_cards(value: Any, result: list[tuple[int, int | None, int | None]]) -> None:
    if isinstance(value, list):
        for item in value:
            _known_cards(item, result)
        return
    if not isinstance(value, dict):
        return
    if value.get("id") is not None:
        result.append((_int(value.get("id")), value.get("serial"), value.get("playerIndex")))
    for child in value.values():
        if isinstance(child, (list, dict)):
            _known_cards(child, result)


def _current_card_count(observation: dict[str, Any]) -> int:
    cards: list[tuple[int, int | None, int | None]] = []
    _known_cards(observation.get("current") or {}, cards)
    _known_cards((observation.get("select") or {}).get("deck") or [], cards)
    serials = {(serial, player) for _, serial, player in cards if serial is not None}
    no_serial = sum(serial is None for _, serial, _ in cards)
    return len(serials) + no_serial


def _card_dynamic_key(card: Any) -> tuple[Any, ...] | None:
    if not isinstance(card, dict) or card.get("id") is None:
        return None
    return (
        _int(card.get("id")),
        card.get("hp"),
        card.get("maxHp"),
        card.get("appearThisTurn"),
        tuple(sorted(_int(value) for value in (card.get("energies") or []))),
        tuple(sorted(_int(item.get("id")) for item in (card.get("energyCards") or []) if isinstance(item, dict))),
        tuple(sorted(_int(item.get("id")) for item in (card.get("tools") or []) if isinstance(item, dict))),
        tuple(_int(item.get("id")) for item in (card.get("preEvolution") or []) if isinstance(item, dict)),
    )


def _normalized_zone_state(observation: dict[str, Any]) -> tuple[tuple[Any, ...], ...]:
    current = observation.get("current") or {}
    your_index = _int(current.get("yourIndex"), 0)
    counts: Counter[tuple[Any, ...]] = Counter()
    players = current.get("players") or []
    for absolute_owner, player in enumerate(players):
        relative_owner = 0 if absolute_owner == your_index else 1
        for zone in ("active", "bench", "hand", "discard", "prize"):
            cards = player.get(zone) if isinstance(player, dict) else None
            for card in cards or []:
                key = _card_dynamic_key(card)
                if key is not None:
                    counts[(relative_owner, zone, *key)] += 1
        if isinstance(player, dict):
            counts[(relative_owner, "deck_count", _int(player.get("deckCount"), -1))] += 1
            counts[(relative_owner, "hand_count", _int(player.get("handCount"), -1))] += 1
            counts[(relative_owner, "prize_count", len(player.get("prize") or []))] += 1
    for zone in ("stadium", "looking"):
        for card in current.get(zone) or []:
            key = _card_dynamic_key(card)
            if key is not None:
                owner = _int(card.get("playerIndex"), your_index)
                counts[(0 if owner == your_index else 1, zone, *key)] += 1
    return tuple(sorted([(*key, count) for key, count in counts.items()], key=repr))


def _normalized_zone_delta(before: dict[str, Any], after: dict[str, Any]) -> tuple[tuple[Any, ...], ...]:
    before_counts = Counter({row[:-1]: row[-1] for row in _normalized_zone_state(before)})
    after_counts = Counter({row[:-1]: row[-1] for row in _normalized_zone_state(after)})
    delta = after_counts.copy()
    delta.subtract(before_counts)
    return tuple(sorted([(*key, count) for key, count in delta.items() if count], key=repr))


def _normalized_log_multiset(observation: dict[str, Any]) -> tuple[tuple[str, int], ...]:
    current = observation.get("current") or {}
    your_index = _int(current.get("yourIndex"), 0)
    counts: Counter[str] = Counter()
    for log in observation.get("logs") or []:
        if not isinstance(log, dict):
            continue
        normalized = {}
        for key, value in sorted(log.items()):
            if key.startswith("serial"):
                continue
            if key == "playerIndex" and value is not None:
                normalized[key] = 0 if _int(value) == your_index else 1
            else:
                normalized[key] = value
        counts[json.dumps(normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":"))] += 1
    return tuple(sorted(counts.items()))


def _stable_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _observation_fingerprint(observation: dict[str, Any]) -> str:
    payload = json.dumps(
        _payload(observation), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _semantics(select: dict[str, Any], action: list[int]) -> str:
    return infer_action_semantics(select, action).value


def _counter_dict() -> dict[str, Counter[Any]]:
    return {
        "option_count": Counter(),
        "action_count": Counter(),
        "multi_option_count": Counter(),
        "multi_action_count": Counter(),
        "projected_board_upper": Counter(),
        "current_card_count": Counter(),
        "recent_event_count": Counter(),
        "equivalence_group_size": Counter(),
        "count_number_values": Counter(),
    }


def _audit_member(name: str) -> dict[str, Any]:
    assert _ZIP is not None
    try:
        replay = json.loads(_ZIP.read(name))
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}", "name": name}

    info = replay.get("info") or {}
    episode_id = info.get("EpisodeId")
    counters = _counter_dict()
    table: Counter[tuple[Any, ...]] = Counter()
    eq_types: Counter[tuple[Any, ...]] = Counter()
    ordered_stats: Counter[tuple[Any, ...]] = Counter()
    summary = Counter()
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    order_records: list[tuple[str, str, str, int, int]] = []
    settlement_records: list[tuple[Any, ...]] = []
    decision_rows: list[dict[str, Any]] = []
    deck_rows: list[dict[str, Any]] = []
    pending: dict[int, dict[str, Any]] = {}
    last_payload: dict[int, dict[str, Any]] = {}
    last_turn_action: dict[int, tuple[int, int]] = {}
    recent_event_total: Counter[int] = Counter()
    self_decks: dict[int, set[int]] = defaultdict(set)
    opponent_revealed: dict[int, set[int]] = defaultdict(set)

    for step_index, step in enumerate(replay.get("steps") or []):
        if not isinstance(step, list):
            summary["bad_steps"] += 1
            continue
        for player_index, agent_step in enumerate(step):
            if not isinstance(agent_step, dict):
                continue
            raw_action = agent_step.get("action")
            action = [_int(value, 0) for value in raw_action] if isinstance(raw_action, list) else ([] if raw_action is None else [_int(raw_action, 0)])
            decision = pending.pop(player_index, None)
            if decision is not None:
                select = decision["select"]
                options = select.get("option") or []
                semantics = _semantics(select, action)
                summary["decisions"] += 1
                summary[f"semantics:{semantics}"] += 1
                counters["option_count"][len(options)] += 1
                counters["action_count"][len(action)] += 1
                if _int(select.get("maxCount"), 0) > 1:
                    counters["multi_option_count"][len(options)] += 1
                    counters["multi_action_count"][len(action)] += 1
                if len(action) != len(set(action)):
                    summary["duplicate_action_index_decisions"] += 1
                    if len(examples["duplicate_action_index"]) < 3:
                        examples["duplicate_action_index"].append({"episode_id": episode_id, "decision_step": decision["step"], "action_step": step_index, "player": player_index, "action": action, "select": select})
                if action != sorted(action):
                    summary["nonascending_action_decisions"] += 1
                invalid = [value for value in action if value < 0 or value >= len(options)]
                if invalid:
                    summary["invalid_action_index_decisions"] += 1
                    if len(examples["invalid_action_index"]) < 3:
                        examples["invalid_action_index"].append({"episode_id": episode_id, "decision_step": decision["step"], "action_step": step_index, "player": player_index, "action": action, "option_count": len(options)})
                else:
                    target = build_action_target(
                        decision["observation"], action, ActionSemantics(semantics)
                    )
                    final_rewards = replay.get("rewards") or []
                    final_reward = (
                        final_rewards[player_index]
                        if player_index < len(final_rewards) and final_rewards[player_index] is not None
                        else None
                    )
                    if _WRITE_REFERENCE_DATASET:
                        decision_rows.append(
                            {
                            "schema_version": "replay_decision_reference_v1",
                            "decision_key": {
                                "episode_id": _int(episode_id),
                                "decision_step_index": decision["step"],
                                "action_step_index": step_index,
                                "player_index": player_index,
                            },
                            "source": {"replay_member": name},
                            "observation_fingerprint": decision["fingerprint"],
                            "action": action,
                            "action_semantics": semantics,
                            "action_target": asdict(target),
                            "select_type": _int(select.get("type")),
                            "select_context": _int(select.get("context")),
                            "min_count": _int(select.get("minCount"), 0),
                            "min_count_state": _field_state(select, "minCount"),
                            "max_count": _int(select.get("maxCount"), 0),
                            "max_count_state": _field_state(select, "maxCount"),
                            "effect_reference": select.get("effect"),
                            "effect_presence": _field_state(select, "effect"),
                            "context_card_reference": select.get("contextCard"),
                            "context_card_presence": _field_state(select, "contextCard"),
                            "policy_loss_mask": policy_loss_mask(select, target),
                            "reward": agent_step.get("reward"),
                            "status": agent_step.get("status"),
                            "final_reward": final_reward,
                            "final_outcome": (
                                None
                                if final_reward is None
                                else 1 if final_reward > 0 else -1 if final_reward < 0 else 0
                            ),
                            }
                        )

                option_types = sorted({_int(option.get("type")) for option in options}) or [-1]
                for option_type in option_types:
                    key = (_int(select.get("type")), _int(select.get("context")), option_type, semantics)
                    table[key + ("decisions",)] += 1
                    table[key + ("selected",)] += len(action)
                    table[key + ("empty",)] += int(not action)
                    table[key + ("duplicate_index",)] += int(len(action) != len(set(action)))
                    table[key + ("nonascending",)] += int(action != sorted(action))
                    table[key + ("max_options",)] = max(table[key + ("max_options",)], len(options))
                    table[key + ("max_selected",)] = max(table[key + ("max_selected",)], len(action))

                signatures = [
                    option_equivalence_signature(decision["observation"], option)
                    for option in options
                ]
                groups: dict[str, list[int]] = defaultdict(list)
                for index, signature in enumerate(signatures):
                    groups[signature].append(index)
                has_equivalent_options = False
                for signature, indices in groups.items():
                    if len(indices) <= 1:
                        continue
                    has_equivalent_options = True
                    summary["equivalent_option_groups"] += 1
                    counters["equivalence_group_size"][len(indices)] += 1
                    group_types = sorted({_int(options[index].get("type")) for index in indices})
                    eq_types[(_int(select.get("type")), _int(select.get("context")), tuple(group_types), len(indices))] += 1
                    chosen = [value for value in action if value in indices]
                    if chosen:
                        summary["selected_equivalent_option_groups"] += 1
                        if any(value != indices[0] for value in chosen):
                            summary["selected_nonfirst_equivalent_index_groups"] += 1
                            if len(examples["equivalent_nonfirst"]) < 3:
                                examples["equivalent_nonfirst"].append({"episode_id": episode_id, "decision_step": decision["step"], "player": player_index, "indices": indices, "chosen": chosen, "options": [options[index] for index in indices]})
                summary["decisions_with_equivalent_options"] += int(has_equivalent_options)
                if len(action) > 1 and not invalid:
                    ordered = ",".join(signatures[index] for index in action)
                    multiset = ",".join(sorted(signatures[index] for index in action))
                    context_key = f"{_int(select.get('type'))}:{_int(select.get('context'))}:{','.join(map(str, option_types))}"
                    order_records.append((context_key, hashlib.sha1(multiset.encode()).hexdigest(), hashlib.sha1(ordered.encode()).hexdigest(), _int(episode_id), decision["step"]))

                context = _int(select.get("context"))
                if context in ORDER_AUDIT_CONTEXTS and _int(select.get("maxCount"), 0) > 1:
                    select_type = _int(select.get("type"))
                    option_type_key = tuple(option_types)
                    stat_key = (select_type, context, option_type_key)
                    ordered_stats[stat_key + ("samples",)] += 1
                    ordered_stats[stat_key + ("max_options",)] = max(
                        ordered_stats[stat_key + ("max_options",)], len(options)
                    )
                    ordered_stats[stat_key + ("max_selected",)] = max(
                        ordered_stats[stat_key + ("max_selected",)], len(action)
                    )
                    for option in options:
                        ordered_stats[stat_key + ("source_area", _int(option.get("area")))] += 1
                        ordered_stats[stat_key + ("target_area", _int(option.get("inPlayArea")))] += 1
                    effect = select.get("effect")
                    context_card = select.get("contextCard")
                    example_key = f"ordered_context_{context}"
                    if len(examples[example_key]) < 3:
                        examples[example_key].append(
                            {
                                "episode_id": _int(episode_id),
                                "decision_step": decision["step"],
                                "action_step": step_index,
                                "player_index": player_index,
                                "action": action,
                                "select": select,
                                "next_logs": (agent_step.get("observation") or {}).get("logs") or [],
                            }
                        )
                    if isinstance(effect, dict) and effect.get("id") is not None:
                        ordered_stats[stat_key + ("effect_card", _int(effect.get("id")))] += 1
                    if isinstance(context_card, dict) and context_card.get("id") is not None:
                        ordered_stats[stat_key + ("context_card", _int(context_card.get("id")))] += 1
                    if not invalid and len(action) > 1:
                        after_observation = agent_step.get("observation") or {}
                        selected_signatures = [signatures[index] for index in action]
                        selected_multiset_hash = _stable_hash(sorted(selected_signatures))
                        pre_key = _stable_hash(
                            {
                                "state": _normalized_zone_state(decision["observation"]),
                                "select_type": select_type,
                                "context": context,
                                "effect_id": _int(effect.get("id")) if isinstance(effect, dict) else -1,
                                "context_card_id": (
                                    _int(context_card.get("id")) if isinstance(context_card, dict) else -1
                                ),
                                "selected_multiset": sorted(selected_signatures),
                            }
                        )
                        order_hash = _stable_hash(selected_signatures)
                        state_delta_hash = _stable_hash(
                            _normalized_zone_delta(decision["observation"], after_observation)
                        )
                        log_hash = _stable_hash(_normalized_log_multiset(after_observation))
                        settlement_records.append(
                            (
                                select_type,
                                context,
                                option_type_key,
                                pre_key,
                                selected_multiset_hash,
                                order_hash,
                                state_delta_hash,
                                log_hash,
                                _int(episode_id),
                                decision["step"],
                            )
                        )

                if _int(select.get("type")) == 8 or any(_int(option.get("type")) == 0 for option in options):
                    for option in options:
                        if _int(option.get("type")) == 0 and option.get("number") is not None:
                            counters["count_number_values"][_int(option.get("number"))] += 1
                    if len(examples["count_number"]) < 3:
                        examples["count_number"].append({"episode_id": episode_id, "decision_step": decision["step"], "player": player_index, "action": action, "select": select})

                selected_ability = [index for index in action if 0 <= index < len(options) and _int(options[index].get("type")) == 10]
                if selected_ability:
                    summary["selected_ability_actions"] += len(selected_ability)
                    next_observation = agent_step.get("observation") or {}
                    logs = next_observation.get("logs") or []
                    explicit = sum(any(key in log for key in ("abilityId", "detailId", "skillId")) for log in logs if isinstance(log, dict))
                    summary["selected_ability_with_explicit_detail_log"] += int(explicit > 0)
                    if len(examples["selected_ability"]) < 3:
                        examples["selected_ability"].append({"episode_id": episode_id, "decision_step": decision["step"], "action_step": step_index, "player": player_index, "selected_options": [options[index] for index in selected_ability], "next_log_types": [_int(log.get("type")) for log in logs if isinstance(log, dict)], "explicit_detail_log_fields": explicit})
            elif action:
                summary["unpaired_nonempty_actions"] += 1
                if len(action) == 60:
                    summary["initial_deck_actions"] += 1
                    self_decks[player_index].update(action)
                    if _WRITE_REFERENCE_DATASET:
                        deck_rows.append(
                            {
                            "schema_version": "replay_deck_configuration_v1",
                            "episode_id": _int(episode_id),
                            "player_index": player_index,
                            "action_step_index": step_index,
                            "deck": action,
                            "source": {"replay_member": name},
                            }
                        )
                else:
                    summary["unexpected_unpaired_actions"] += 1

            observation = agent_step.get("observation")
            if not isinstance(observation, dict):
                continue
            current_payload = _payload(observation)
            if last_payload.get(player_index) == current_payload:
                summary["duplicate_observations"] += 1
                continue
            last_payload[player_index] = current_payload
            logs = observation.get("logs") or []
            recent_event_total[player_index] += len(logs)
            select = observation.get("select")
            current = observation.get("current")
            if select is None:
                summary["effective_no_select_observations"] += 1
                continue
            summary["effective_select_observations"] += 1
            if current is None:
                summary["select_without_current"] += 1
            else:
                if "turnActionCount" not in current:
                    summary["turn_action_count_missing"] += 1
                elif current.get("turnActionCount") is None:
                    summary["turn_action_count_null"] += 1
                else:
                    turn = _int(current.get("turn"))
                    tac = _int(current.get("turnActionCount"))
                    previous_turn_action = last_turn_action.get(player_index)
                    if previous_turn_action is not None:
                        previous_turn, previous_tac = previous_turn_action
                        if turn == previous_turn:
                            relation = "increase" if tac > previous_tac else "equal" if tac == previous_tac else "decrease"
                            summary[f"turn_action_same_turn_{relation}"] += 1
                            if relation != "increase" and len(examples[f"turn_action_{relation}"]) < 3:
                                examples[f"turn_action_{relation}"].append({"episode_id": episode_id, "step": step_index, "player": player_index, "turn": turn, "previous": previous_tac, "current": tac, "select_type": select.get("type"), "context": select.get("context")})
                        else:
                            summary["turn_action_turn_changed"] += 1
                    last_turn_action[player_index] = (turn, tac)

            if select.get("effect") is not None and select.get("contextCard") is not None:
                summary["effect_and_context_card"] += 1
                if len(examples["effect_and_context_card"]) < 5:
                    examples["effect_and_context_card"].append({"episode_id": episode_id, "step": step_index, "player": player_index, "select_type": select.get("type"), "select_context": select.get("context"), "effect": select.get("effect"), "contextCard": select.get("contextCard")})

            known: list[tuple[int, int | None, int | None]] = []
            _known_cards(current or {}, known)
            _known_cards(logs, known)
            your_index = _int((current or {}).get("yourIndex"), player_index)
            for card_id, _, owner in known:
                if owner is not None and _int(owner) != your_index:
                    opponent_revealed[player_index].add(card_id)
            current_cards = _current_card_count(observation)
            recent_events = min(recent_event_total[player_index], 32)
            projected = 5 + current_cards + recent_events + len(self_decks[player_index]) + len(opponent_revealed[player_index]) + 24 + 3
            counters["current_card_count"][current_cards] += 1
            counters["recent_event_count"][recent_events] += 1
            counters["projected_board_upper"][projected] += 1
            pending[player_index] = {
                "step": step_index,
                "select": select,
                "observation": observation,
                "fingerprint": _observation_fingerprint(observation),
            }

    summary["unpaired_terminal_decisions"] += len(pending)
    return {
        "name": name,
        "summary": summary,
        "counters": counters,
        "table": table,
        "eq_types": eq_types,
        "ordered_stats": ordered_stats,
        "examples": dict(examples),
        "order_records": order_records,
        "settlement_records": settlement_records,
        "decision_rows": decision_rows,
        "deck_rows": deck_rows,
    }


def _merge_counter_map(target: dict[str, Counter[Any]], source: dict[str, Counter[Any]]) -> None:
    for name, counter in source.items():
        target[name].update(counter)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the frozen replay decision data contract over a replay ZIP.")
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/replay_decision_contract_audit"))
    parser.add_argument("--workers", type=int, default=max(1, min(4, mp.cpu_count())))
    parser.add_argument("--max-replays", type=int)
    parser.add_argument(
        "--write-reference-dataset",
        action="store_true",
        help="Write a compact gzip decision index that references replay members instead of copying observations.",
    )
    args = parser.parse_args()

    with zipfile.ZipFile(args.archive) as archive:
        names = [name for name in archive.namelist() if name.endswith(".json")]
    if args.max_replays is not None:
        names = names[: args.max_replays]

    summary = Counter()
    counters = _counter_dict()
    table: Counter[tuple[Any, ...]] = Counter()
    eq_types: Counter[tuple[Any, ...]] = Counter()
    ordered_stats: Counter[tuple[Any, ...]] = Counter()
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    order_map: dict[tuple[str, str], set[str]] = defaultdict(set)
    order_examples: dict[tuple[str, str], list[tuple[int, int]]] = defaultdict(list)
    settlement_groups: dict[tuple[Any, ...], dict[str, set[tuple[str, str]]]] = defaultdict(
        lambda: defaultdict(set)
    )
    settlement_examples: dict[tuple[Any, ...], list[tuple[int, int]]] = defaultdict(list)
    coarse_settlement_groups: dict[
        tuple[Any, ...], dict[str, set[tuple[str, str]]]
    ] = defaultdict(lambda: defaultdict(set))
    coarse_settlement_examples: dict[tuple[Any, ...], list[tuple[int, int]]] = defaultdict(list)
    errors: list[dict[str, str]] = []

    dataset_dir = args.output_dir / "dataset"
    decision_handle = None
    deck_handle = None
    if args.write_reference_dataset:
        dataset_dir.mkdir(parents=True, exist_ok=True)
        decision_handle = gzip.open(dataset_dir / "decisions.jsonl.gz", "wt", encoding="utf-8")
        deck_handle = gzip.open(dataset_dir / "deck_configurations.jsonl.gz", "wt", encoding="utf-8")

    try:
        with mp.Pool(
            args.workers,
            initializer=_init_worker,
            initargs=(str(args.archive), args.write_reference_dataset),
        ) as pool:
            # Preserve archive order so the compact index is reproducible.
            for index, result in enumerate(pool.imap(_audit_member, names, chunksize=1), start=1):
                if "error" in result:
                    errors.append({"name": result["name"], "error": result["error"]})
                    continue
                if decision_handle is not None:
                    for row in result["decision_rows"]:
                        decision_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    for row in result["deck_rows"]:
                        deck_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                summary.update(result["summary"])
                _merge_counter_map(counters, result["counters"])
                for key, value in result["table"].items():
                    if key[-1].startswith("max_"):
                        table[key] = max(table[key], value)
                    else:
                        table[key] += value
                eq_types.update(result["eq_types"])
                for key, value in result["ordered_stats"].items():
                    if any(isinstance(part, str) and part.startswith("max_") for part in key):
                        ordered_stats[key] = max(ordered_stats[key], value)
                    else:
                        ordered_stats[key] += value
                for kind, rows in result["examples"].items():
                    remaining = 100 - len(examples[kind])
                    if remaining > 0:
                        examples[kind].extend(rows[:remaining])
                for context_key, multiset_hash, order_hash, episode_id, step in result["order_records"]:
                    key = (context_key, multiset_hash)
                    order_map[key].add(order_hash)
                    if len(order_examples[key]) < 5:
                        order_examples[key].append((episode_id, step))
                for record in result["settlement_records"]:
                    select_type, context, option_types, pre_key, selected_multiset_hash, order_hash, state_hash, log_hash, episode_id, step = record
                    group_key = (select_type, context, option_types, pre_key)
                    settlement_groups[group_key][order_hash].add((state_hash, log_hash))
                    if len(settlement_examples[group_key]) < 5:
                        settlement_examples[group_key].append((episode_id, step))
                    coarse_key = (select_type, context, option_types, selected_multiset_hash)
                    coarse_settlement_groups[coarse_key][order_hash].add((state_hash, log_hash))
                    if len(coarse_settlement_examples[coarse_key]) < 5:
                        coarse_settlement_examples[coarse_key].append((episode_id, step))
                if index % 100 == 0:
                    print(json.dumps({"processed": index, "total": len(names), "decisions": summary["decisions"]}), flush=True)
    finally:
        if decision_handle is not None:
            decision_handle.close()
        if deck_handle is not None:
            deck_handle.close()

    differing_order_groups = [(key, values) for key, values in order_map.items() if len(values) > 1]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table_rows = []
    base_keys = sorted({key[:4] for key in table})
    for base in base_keys:
        select_type, context, option_type, semantics = base
        table_rows.append({
            "select_type": select_type,
            "select_context": context,
            "option_type": option_type,
            "action_semantics": semantics,
            "decision_count": table[base + ("decisions",)],
            "selected_index_count": table[base + ("selected",)],
            "empty_action_count": table[base + ("empty",)],
            "duplicate_index_decision_count": table[base + ("duplicate_index",)],
            "nonascending_action_count": table[base + ("nonascending",)],
            "max_option_count": table[base + ("max_options",)],
            "max_selected_count": table[base + ("max_selected",)],
        })
    with (args.output_dir / "action_semantics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table_rows[0]) if table_rows else ["select_type"])
        writer.writeheader()
        writer.writerows(table_rows)

    eq_rows = [
        {"select_type": key[0], "select_context": key[1], "option_types": list(key[2]), "group_size": key[3], "group_count": count}
        for key, count in sorted(eq_types.items())
    ]
    with (args.output_dir / "equivalent_option_groups.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = list(eq_rows[0]) if eq_rows else ["select_type"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(eq_rows)

    comparison_by_semantic: Counter[tuple[Any, ...]] = Counter()
    comparison_examples: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for group_key, orders in settlement_groups.items():
        if len(orders) <= 1:
            continue
        semantic_key = group_key[:3]
        outcomes = [outcome for values in orders.values() for outcome in values]
        state_hashes = {state_hash for state_hash, _ in outcomes}
        log_hashes = {log_hash for _, log_hash in outcomes}
        pair_hashes = set(outcomes)
        comparison_by_semantic[semantic_key + ("comparable",)] += 1
        comparison_by_semantic[semantic_key + ("state_invariant",)] += int(len(state_hashes) == 1)
        comparison_by_semantic[semantic_key + ("log_invariant",)] += int(len(log_hashes) == 1)
        comparison_by_semantic[semantic_key + ("fully_invariant",)] += int(len(pair_hashes) == 1)
        if len(comparison_examples[semantic_key]) < 5:
            comparison_examples[semantic_key].append(
                {
                    "order_count": len(orders),
                    "normalized_outcome_count": len(pair_hashes),
                    "episode_steps": settlement_examples[group_key],
                }
            )

    coarse_comparison: Counter[tuple[Any, ...]] = Counter()
    coarse_examples: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for group_key, orders in coarse_settlement_groups.items():
        if len(orders) <= 1:
            continue
        semantic_key = group_key[:3]
        outcome_sets = list(orders.values())
        shared_pairs = set.intersection(*outcome_sets)
        state_sets = [{state_hash for state_hash, _ in values} for values in outcome_sets]
        log_sets = [{log_hash for _, log_hash in values} for values in outcome_sets]
        coarse_comparison[semantic_key + ("comparable",)] += 1
        coarse_comparison[semantic_key + ("shared_pair",)] += int(bool(shared_pairs))
        coarse_comparison[semantic_key + ("shared_state",)] += int(bool(set.intersection(*state_sets)))
        coarse_comparison[semantic_key + ("shared_log",)] += int(bool(set.intersection(*log_sets)))
        if len(coarse_examples[semantic_key]) < 5:
            coarse_examples[semantic_key].append(
                {
                    "order_count": len(orders),
                    "has_shared_normalized_outcome": bool(shared_pairs),
                    "episode_steps": coarse_settlement_examples[group_key],
                }
            )

    ordered_rows = []
    ordered_keys = sorted({key[:3] for key in ordered_stats})
    for semantic_key in ordered_keys:
        select_type, context, option_types = semantic_key
        comparable = comparison_by_semantic[semantic_key + ("comparable",)]
        fully_invariant = comparison_by_semantic[semantic_key + ("fully_invariant",)]
        state_invariant = comparison_by_semantic[semantic_key + ("state_invariant",)]
        log_invariant = comparison_by_semantic[semantic_key + ("log_invariant",)]
        coarse_comparable = coarse_comparison[semantic_key + ("comparable",)]
        coarse_shared_pair = coarse_comparison[semantic_key + ("shared_pair",)]
        coarse_shared_state = coarse_comparison[semantic_key + ("shared_state",)]
        coarse_shared_log = coarse_comparison[semantic_key + ("shared_log",)]
        source_areas = {
            key[-1]: value
            for key, value in ordered_stats.items()
            if key[:3] == semantic_key and len(key) == 5 and key[3] == "source_area"
        }
        target_areas = {
            key[-1]: value
            for key, value in ordered_stats.items()
            if key[:3] == semantic_key and len(key) == 5 and key[3] == "target_area"
        }
        effect_cards = Counter(
            {
                key[-1]: value
                for key, value in ordered_stats.items()
                if key[:3] == semantic_key and len(key) == 5 and key[3] == "effect_card"
            }
        )
        context_cards = Counter(
            {
                key[-1]: value
                for key, value in ordered_stats.items()
                if key[:3] == semantic_key and len(key) == 5 and key[3] == "context_card"
            }
        )
        verdict = (
            "EXACT_NORMALIZED_OUTCOME_INVARIANT"
            if comparable > 0 and fully_invariant == comparable
            else "NO_CONTROLLED_PRESTATE_PAIR"
            if comparable == 0
            else "CONTROLLED_NORMALIZED_OUTCOMES_DIFFER"
        )
        contract = CONTEXT_SEMANTICS[context]
        ordered_rows.append(
            {
                "select_type": select_type,
                "select_context": context,
                "option_types": json.dumps(option_types),
                "effect_kind": contract["effect_kind"],
                "candidate_meaning": contract["candidate_meaning"],
                "order_changes_final_effect": contract["order_changes_final_effect"],
                "final_action_semantics": contract["final_semantics"],
                "semantic_evidence": contract["semantic_evidence"],
                "sample_count": ordered_stats[semantic_key + ("samples",)],
                "max_option_count": ordered_stats[semantic_key + ("max_options",)],
                "max_selected_count": ordered_stats[semantic_key + ("max_selected",)],
                "source_areas": json.dumps(source_areas, sort_keys=True),
                "target_areas": json.dumps(target_areas, sort_keys=True),
                "top_effect_card_ids": json.dumps(effect_cards.most_common(10)),
                "top_context_card_ids": json.dumps(context_cards.most_common(10)),
                "comparable_prestate_groups": comparable,
                "normalized_state_invariant_groups": state_invariant,
                "normalized_log_invariant_groups": log_invariant,
                "fully_invariant_groups": fully_invariant,
                "same_selected_set_multiple_order_groups": coarse_comparable,
                "groups_with_shared_normalized_state_delta": coarse_shared_state,
                "groups_with_shared_normalized_log_multiset": coarse_shared_log,
                "groups_with_shared_full_outcome": coarse_shared_pair,
                "normalized_comparison_verdict": verdict,
            }
        )
    with (args.output_dir / "ordered_context_settlement.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        fields = list(ordered_rows[0]) if ordered_rows else ["select_type"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(ordered_rows)

    report = {
        "schema_version": "replay_decision_contract_audit_v1",
        "archive": str(args.archive),
        "requested_replays": len(names),
        "successful_replays": len(names) - len(errors),
        "errors": errors[:100],
        "summary": dict(summary),
        "distributions": {name: _distribution(counter) for name, counter in counters.items()},
        "semantic_row_count": len(table_rows),
        "equivalence_row_count": len(eq_rows),
        "ordered_context_settlement": {
            "row_count": len(ordered_rows),
            "comparison_normalization": "Same normalized visible pre-state, decision references, and selected equivalence multiset; compare serial-free zone deltas and order-insensitive log multisets after the action.",
            "examples": {
                f"{key[0]}:{key[1]}:{','.join(map(str, key[2]))}": value
                for key, value in comparison_examples.items()
            },
            "same_selected_set_examples": {
                f"{key[0]}:{key[1]}:{','.join(map(str, key[2]))}": value
                for key, value in coarse_examples.items()
            },
        },
        "equivalence_key_fields": [
            "select_type",
            "select_context",
            "option_type",
            "source_zone",
            "target_zone",
            "source_identity_and_dynamic_state_without_serial",
            "target_identity_and_dynamic_state_without_serial",
            "detail_reference",
            "effect_reference",
            "effect_presence_state",
            "context_card_reference",
            "context_card_presence_state",
            "option_field_missingness_states",
            "energy_quantity_and_other_resolution_fields",
            "source_owner",
            "target_owner",
            "unresolved_raw_indices_as_conservative_fallback",
            "unresolved_direct_card_id_and_serial_as_conservative_fallback",
        ],
        "order_evidence": {
            "multi_action_semantic_multisets": len(order_map),
            "multisets_seen_in_multiple_orders": len(differing_order_groups),
            "examples": [
                {"context": key[0], "multiset_hash": key[1], "order_count": len(values), "episode_steps": order_examples[key]}
                for key, values in differing_order_groups[:100]
            ],
            "limitation": "Multiple accepted orders are evidence of acceptance, not proof that every rule resolves order-independently.",
        },
        "ability_mapping": {
            "explicit_log_fields_checked": ["abilityId", "detailId", "skillId"],
            "stable_general_mapping": summary["selected_ability_actions"] > 0 and summary["selected_ability_actions"] == summary["selected_ability_with_explicit_detail_log"],
        },
        "board_token_estimate_contract": "5 fixed facts + visible current serial cards + last 32 events + self deck unique Card IDs + revealed opponent Card IDs + 24 opponent recall slots + 3 belief global/tail tokens; this is a conservative upper estimate because recalled IDs may overlap revealed IDs.",
        "examples": examples,
    }
    (args.output_dir / "audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    equivalence_contract = {
        "schema_version": "legal_option_equivalence_v1",
        "rule": "Two options share a class only when every normalized field below is equal.",
        "serial_rule": "Resolved entity serials and names are removed; unresolved references retain raw indices or direct identity conservatively.",
        "fields": report["equivalence_key_fields"],
        "single_target": "sum probability over all members of the demonstrated class",
        "unordered_target": "selected member count per class plus total selected count",
        "ordered_target": "ordered class sequence",
        "count_target": "option.number with a value-to-option-index mapping",
    }
    (args.output_dir / "equivalence_contract.json").write_text(
        json.dumps(equivalence_contract, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if args.write_reference_dataset:
        manifest = {
            "schema_version": "replay_decision_reference_dataset_v1",
            "storage_contract": "Raw observations remain in the replay ZIP. Decision rows store keys, targets, labels, and replay_member references only.",
            "replay_archive": str(args.archive),
            "decision_file": "decisions.jsonl.gz",
            "deck_configuration_file": "deck_configurations.jsonl.gz",
            "decision_count": summary["decisions"],
            "unpaired_pending": summary["unpaired_terminal_decisions"],
            "illegal_action_index": summary["invalid_action_index_decisions"],
            "deck_configuration_action": summary["initial_deck_actions"],
            "duplicate_old_observation": summary["duplicate_observations"],
            "unexpected_unpaired_action": summary["unexpected_unpaired_actions"],
            "action_semantics_counts": {
                semantics: summary[f"semantics:{semantics}"]
                for semantics in (
                    "SINGLE_INDEX",
                    "COUNT_VALUE",
                    "UNORDERED_UNIQUE_SUBSET",
                    "ORDERED_INDEX_SEQUENCE",
                    "INDEX_MULTISET",
                )
            },
        }
        (dataset_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(json.dumps({"output": str(args.output_dir), "replays": report["successful_replays"], "decisions": summary["decisions"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
