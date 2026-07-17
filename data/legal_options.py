from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Any

from .decision_schema import ActionSemantics, FieldState, LegalActionTarget


INDEX_FIELDS = {"index", "inPlayIndex", "toolIndex", "energyIndex"}
DIRECT_IDENTITY_FIELDS = {"cardId", "serial"}
AUDITED_UNORDERED_SELECT_CONTEXTS = frozenset(
    {
        (1, 2),   # setup: choose the remaining Basic Pokémon for the Bench
        (1, 5),
        (1, 7),
        (1, 8),
        (1, 9),
        (1, 15),  # Trifrost: three Pokémon receive the same simultaneous damage
        (1, 21),  # choose Pokémon receiving one equivalent Energy each
        (2, 26),  # discard an Energy subset; damage depends on count
        (2, 27),  # Tool Scrapper target subset
        (5, 34),  # forced Risky Ruins trigger set
    }
)


def _int(value: Any, default: int = -1) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _field_state(obj: Any, key: str) -> str:
    if not isinstance(obj, dict) or key not in obj:
        return FieldState.MISSING.name
    value = obj[key]
    if value is None:
        return FieldState.EXPLICIT_NULL.name
    if isinstance(value, int) and value < 0:
        return FieldState.UNKNOWN.name
    return FieldState.PRESENT.name


def _area_cards(observation: dict[str, Any], player_index: int, area: int) -> list[Any] | None:
    current = observation.get("current") or {}
    players = current.get("players") or []
    player = players[player_index] if 0 <= player_index < len(players) else {}
    return {
        1: (observation.get("select") or {}).get("deck") or current.get("looking"),
        2: player.get("hand"),
        3: player.get("discard"),
        4: player.get("active"),
        5: player.get("bench"),
        6: player.get("prize"),
        7: current.get("stadium"),
        12: current.get("looking"),
    }.get(area)


def _indexed(cards: list[Any] | None, value: Any) -> Any | None:
    index = _int(value)
    return cards[index] if cards is not None and 0 <= index < len(cards) else None


def _direct_card_reference(
    observation: dict[str, Any], option: dict[str, Any]
) -> tuple[Any | None, int | None, int | None]:
    serial = option.get("serial")
    card_id = option.get("cardId")
    if serial is None and card_id is None:
        return None, None, None
    current = observation.get("current") or {}
    select = observation.get("select") or {}
    candidates: list[tuple[int, int | None, Any]] = []
    for owner, player in enumerate(current.get("players") or []):
        if not isinstance(player, dict):
            continue
        for key, zone in (("hand", 2), ("discard", 3), ("active", 4), ("bench", 5), ("prize", 6)):
            for card in player.get(key) or []:
                candidates.append((zone, owner, card))
    your_index = _int(current.get("yourIndex"), 0)
    for key, zone in (("stadium", 7), ("looking", 12)):
        for card in current.get(key) or []:
            owner = _int(card.get("playerIndex"), your_index) if isinstance(card, dict) else None
            candidates.append((zone, owner, card))
    for card in select.get("deck") or []:
        owner = _int(card.get("playerIndex"), your_index) if isinstance(card, dict) else None
        candidates.append((1, owner, card))

    def walk(card: Any, zone: int, owner: int | None) -> tuple[Any, int, int | None] | None:
        if not isinstance(card, dict):
            return None
        if (serial is None or card.get("serial") == serial) and (
            card_id is None or card.get("id") == card_id
        ):
            return card, zone, owner
        for child_key, child_zone in (("energyCards", 8), ("tools", 9), ("preEvolution", 10)):
            for child in card.get(child_key) or []:
                found = walk(child, child_zone, owner)
                if found is not None:
                    return found
        return None

    for zone, owner, card in candidates:
        found = walk(card, zone, owner)
        if found is not None:
            return found
    return None, None, None


def _without_serial(value: Any) -> Any:
    if isinstance(value, list):
        return [_without_serial(item) for item in value]
    if not isinstance(value, dict):
        return value
    result = {}
    for key, item in sorted(value.items()):
        if key in {"serial", "name"}:
            continue
        normalized = _without_serial(item)
        if key in {"energies", "energyCards", "tools"} and isinstance(normalized, list):
            normalized = sorted(
                normalized,
                key=lambda child: json.dumps(child, sort_keys=True, separators=(",", ":")),
            )
        result[key] = normalized
    return result


def option_equivalence_key(observation: dict[str, Any], option: dict[str, Any]) -> dict[str, Any]:
    """Return the auditable, settlement-aware equivalence key components."""

    current = observation.get("current") or {}
    select = observation.get("select") or {}
    your_index = _int(current.get("yourIndex"), 0)
    option_type = _int(option.get("type"))
    player_index = _int(option.get("playerIndex"), your_index)
    area = _int(option.get("area"))
    source = None
    target = None
    source_zone = area if area >= 0 else (2 if option_type in {7, 8, 9} else None)
    source_owner = player_index
    target_owner = None
    if option_type == 7:
        source = _indexed(_area_cards(observation, your_index, 2), option.get("index"))
    elif option_type in {8, 9}:
        source = _indexed(_area_cards(observation, your_index, 2), option.get("index"))
        target = _indexed(
            _area_cards(observation, your_index, _int(option.get("inPlayArea"))),
            option.get("inPlayIndex"),
        )
    elif option_type in {3, 10, 11, 15}:
        source = _indexed(_area_cards(observation, player_index, area), option.get("index"))
    elif option_type in {4, 5}:
        parent = _indexed(_area_cards(observation, player_index, area), option.get("index"))
        child_key = "tools" if option_type == 4 else "energyCards"
        child_index = option.get("toolIndex") if option_type == 4 else option.get("energyIndex")
        source = _indexed(parent.get(child_key) if isinstance(parent, dict) else None, child_index)
        target = parent
        target_owner = player_index
    if source is None and any(field in option for field in DIRECT_IDENTITY_FIELDS):
        direct, direct_zone, direct_owner = _direct_card_reference(observation, option)
        if direct is not None:
            source = direct
            source_zone = direct_zone
            source_owner = direct_owner
    if target is not None and target_owner is None:
        target_owner = your_index

    normalized = {
        "select_type": _int(select.get("type")),
        "select_context": _int(select.get("context")),
        "option_type": option_type,
        "source_zone": source_zone,
        "target_zone": _int(option.get("inPlayArea"), -1),
        "source_identity_and_dynamic_state": _without_serial(source),
        "target_identity_and_dynamic_state": _without_serial(target),
        "detail_reference": {
            key: option.get(key)
            for key in ("attackId", "skillId", "abilityId", "detailId")
            if key in option
        },
        "effect_reference": _without_serial(select.get("effect")),
        "effect_presence": _field_state(select, "effect"),
        "context_card_reference": _without_serial(select.get("contextCard")),
        "context_card_presence": _field_state(select, "contextCard"),
        "option_field_states": {
            key: _field_state(option, key)
            for key in sorted(
                INDEX_FIELDS
                | DIRECT_IDENTITY_FIELDS
                | {
                    "type",
                    "area",
                    "inPlayArea",
                    "playerIndex",
                    "attackId",
                    "skillId",
                    "abilityId",
                    "detailId",
                    "energyType",
                    "quantity",
                    "number",
                    "count",
                    "specialConditionType",
                }
            )
        },
        "resolution_fields": {
            key: value
            for key, value in sorted(option.items())
            if key
            not in INDEX_FIELDS
            | DIRECT_IDENTITY_FIELDS
            | {"type", "area", "inPlayArea", "playerIndex"}
        },
        "source_owner": source_owner,
        "target_owner": target_owner,
    }
    if any(field in option for field in INDEX_FIELDS) and source is None and target is None:
        normalized["unresolved_indices"] = {
            field: option.get(field) for field in sorted(INDEX_FIELDS) if field in option
        }
    if any(field in option for field in DIRECT_IDENTITY_FIELDS) and source is None:
        normalized["unresolved_direct_identity"] = {
            field: option.get(field)
            for field in sorted(DIRECT_IDENTITY_FIELDS)
            if field in option
        }
    return normalized


def option_equivalence_signature(observation: dict[str, Any], option: dict[str, Any]) -> str:
    """Hash the formal key; unresolved references retain indices conservatively."""

    normalized = option_equivalence_key(observation, option)
    payload = json.dumps(normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def equivalence_class_ids(observation: dict[str, Any], options: list[dict[str, Any]]) -> list[int]:
    signature_to_class: dict[str, int] = {}
    result: list[int] = []
    for option in options:
        signature = option_equivalence_signature(observation, option)
        if signature not in signature_to_class:
            signature_to_class[signature] = len(signature_to_class)
        result.append(signature_to_class[signature])
    return result


def infer_action_semantics(select: dict[str, Any], action: list[int]) -> ActionSemantics:
    """Route a select to its audited action contract.

    Multi-select contexts remain ordered unless settlement semantics have been
    established as set-valued. ``maxCount`` alone is not enough evidence.
    """

    select_type = _int(select.get("type"))
    select_context = _int(select.get("context"))
    option_types = {_int(option.get("type")) for option in (select.get("option") or [])}
    max_count = _int(select.get("maxCount"), 0)
    if select_type == 8 or 0 in option_types:
        return ActionSemantics.COUNT_VALUE
    if max_count <= 1:
        return ActionSemantics.SINGLE_INDEX
    if len(action) != len(set(action)):
        return ActionSemantics.INDEX_MULTISET
    if (select_type, select_context) in AUDITED_UNORDERED_SELECT_CONTEXTS:
        return ActionSemantics.UNORDERED_UNIQUE_SUBSET
    return ActionSemantics.ORDERED_INDEX_SEQUENCE


def build_action_target(
    observation: dict[str, Any],
    action: list[int],
    semantics: ActionSemantics,
) -> LegalActionTarget:
    options = (observation.get("select") or {}).get("option") or []
    classes = equivalence_class_ids(observation, options)
    if any(index < 0 or index >= len(options) for index in action):
        raise ValueError("action contains an out-of-range legal option index")
    ordered_classes = tuple(classes[index] for index in action)
    count_value = None
    count_mapping: dict[int, list[int]] | None = None
    if semantics is ActionSemantics.COUNT_VALUE and action:
        raw_value = options[action[0]].get("number")
        count_value = _int(raw_value) if raw_value is not None else None
    if semantics is ActionSemantics.COUNT_VALUE:
        count_mapping = {}
        for index, option in enumerate(options):
            if option.get("number") is not None:
                count_mapping.setdefault(_int(option["number"]), []).append(index)
    capacities = Counter(classes)
    return LegalActionTarget(
        semantics=semantics,
        chosen_option_indices=tuple(action),
        equivalence_class_ids=tuple(classes),
        equivalence_class_capacities=dict(capacities),
        chosen_class_counts=dict(Counter(ordered_classes)),
        selected_count=len(action),
        ordered_class_sequence=(
            ordered_classes if semantics is ActionSemantics.ORDERED_INDEX_SEQUENCE else ()
        ),
        count_value=count_value,
        count_value_to_option_indices=(
            {value: tuple(indices) for value, indices in count_mapping.items()}
            if count_mapping is not None
            else None
        ),
    )


def policy_loss_mask(select: dict[str, Any], target: LegalActionTarget) -> bool:
    """Whether the equivalence-aware action space contains a real choice."""

    minimum = _int(select.get("minCount"), 0)
    maximum = _int(select.get("maxCount"), 0)
    class_count = len(target.equivalence_class_capacities)
    if target.semantics is ActionSemantics.COUNT_VALUE:
        return len(target.count_value_to_option_indices or {}) > 1
    if target.semantics is ActionSemantics.SINGLE_INDEX:
        return minimum == 0 or class_count > 1
    if minimum != maximum:
        return True
    if minimum == 0:
        return False
    if target.semantics is ActionSemantics.ORDERED_INDEX_SEQUENCE:
        return class_count > 1

    # For a fixed-cardinality unordered target, count feasible class-count
    # vectors. Stop once two distinct targets are known to exist.
    reachable = {0: 1}
    for capacity in target.equivalence_class_capacities.values():
        updated: dict[int, int] = {}
        for subtotal, ways in reachable.items():
            for chosen in range(min(capacity, maximum - subtotal) + 1):
                total = subtotal + chosen
                updated[total] = min(2, updated.get(total, 0) + ways)
        reachable = updated
    return reachable.get(minimum, 0) > 1
