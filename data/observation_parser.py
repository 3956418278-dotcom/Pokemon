from __future__ import annotations

from collections import Counter
from dataclasses import asdict, is_dataclass
from typing import Any

from .decision_schema import FieldState
from .state_schema import AREA_IDS, CardInstanceState, GameEvent, GlobalSnapshot, ParsedObservation


def _value(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _has_field(obj: Any, name: str) -> bool:
    if obj is None:
        return False
    if isinstance(obj, dict):
        return name in obj
    return hasattr(obj, name)


def _field_state(obj: Any, name: str, *, applicable: bool = True) -> FieldState:
    if not applicable:
        return FieldState.NOT_APPLICABLE
    if not _has_field(obj, name):
        return FieldState.MISSING
    value = _value(obj, name)
    if value is None:
        return FieldState.EXPLICIT_NULL
    if isinstance(value, int) and value < 0:
        return FieldState.UNKNOWN
    return FieldState.PRESENT


def _int(value: Any, default: int | None = 0) -> int | None:
    if value is None:
        return default
    if hasattr(value, "value"):
        return int(value.value)
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _raw(obj: Any) -> dict[str, Any]:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return dict(obj)
    if is_dataclass(obj):
        return asdict(obj)
    return dict(getattr(obj, "__dict__", {}))


def _relative(player_index: int | None, your_index: int) -> int | None:
    if player_index is None:
        return None
    return 0 if player_index == your_index else 1


def _card_instance(card: Any, area: int, zone: str, slot: int, your_index: int, *, source: str = "current") -> CardInstanceState:
    player_index = _int(_value(card, "playerIndex"), None)
    card_id = _int(_value(card, "id"), None)
    return CardInstanceState(
        card_id=card_id,
        serial=_int(_value(card, "serial"), None),
        player_index=player_index,
        relative_player=_relative(player_index, your_index),
        area=area,
        zone=zone,
        slot=slot,
        copy_count=1 if card_id is not None else None,
        source=source,
        field_states={
            name: _field_state(card, name)
            for name in ("id", "serial", "playerIndex")
        },
    )


def _hidden_instance(area: int, zone: str, slot: int, player_index: int | None, your_index: int) -> CardInstanceState:
    return CardInstanceState(
        card_id=None,
        serial=None,
        player_index=player_index,
        relative_player=_relative(player_index, your_index),
        area=area,
        zone=zone,
        slot=slot,
        is_visible=False,
        is_face_down=True,
        field_states={
            "id": FieldState.EXPLICIT_NULL,
            "serial": FieldState.EXPLICIT_NULL,
            "playerIndex": (
                FieldState.PRESENT if player_index is not None else FieldState.UNKNOWN
            ),
        },
    )


def _energy_counts(energies: list[Any]) -> list[int]:
    counts = [0] * 12
    for energy in energies or []:
        index = _int(energy, None)
        if index is not None and 0 <= index < len(counts):
            counts[index] += 1
    return counts


def _energy_counts_are_valid(energies: list[Any]) -> bool:
    return all(
        (index := _int(energy, None)) is not None and 0 <= index < 12
        for energy in energies
    )


def _card_ids(cards: list[Any]) -> list[int]:
    result: list[int] = []
    for card in cards:
        card_id = _int(_value(card, "id"), None)
        if card_id is not None:
            result.append(card_id)
    return result


def _pokemon_instance(
    pokemon: Any,
    area: int,
    zone: str,
    slot: int,
    player_index: int,
    your_index: int,
    conditions: list[bool] | None = None,
    conditions_valid: bool = False,
) -> list[CardInstanceState]:
    if pokemon is None:
        return [_hidden_instance(area, zone, slot, player_index, your_index)]
    serial = _int(_value(pokemon, "serial"), None)
    card_id = _int(_value(pokemon, "id"), None)
    energy_cards = list(_value(pokemon, "energyCards", []) or [])
    tools = list(_value(pokemon, "tools", []) or [])
    pre_evolution = list(_value(pokemon, "preEvolution", []) or [])
    raw_energies = list(_value(pokemon, "energies", []) or [])
    instance = CardInstanceState(
        card_id=card_id,
        serial=serial,
        player_index=player_index,
        relative_player=_relative(player_index, your_index),
        area=area,
        zone=zone,
        slot=slot,
        is_pokemon=True,
        hp=_int(_value(pokemon, "hp"), None),
        max_hp=_int(_value(pokemon, "maxHp"), None),
        appear_this_turn=bool(_value(pokemon, "appearThisTurn", False)),
        appear_this_turn_valid=_has_field(pokemon, "appearThisTurn"),
        energy_counts=_energy_counts(raw_energies),
        energy_counts_valid=_has_field(pokemon, "energies") and _energy_counts_are_valid(raw_energies),
        energy_card_count=len(energy_cards),
        energy_cards_valid=_has_field(pokemon, "energyCards"),
        energy_card_ids=_card_ids(energy_cards),
        tool_count=len(tools),
        tools_valid=_has_field(pokemon, "tools"),
        tool_card_ids=_card_ids(tools),
        pre_evolution_count=len(pre_evolution),
        pre_evolution_valid=_has_field(pokemon, "preEvolution"),
        pre_evolution_card_ids=_card_ids(pre_evolution),
        special_conditions=(conditions or [False] * 5),
        special_conditions_valid=conditions_valid,
        copy_count=1 if card_id is not None else None,
        field_states={
            name: _field_state(pokemon, name)
            for name in (
                "id",
                "serial",
                "playerIndex",
                "hp",
                "maxHp",
                "appearThisTurn",
                "energies",
                "energyCards",
                "tools",
                "preEvolution",
            )
        },
    )
    attached: list[CardInstanceState] = [instance]
    for index, card in enumerate(energy_cards):
        card_state = _card_instance(card, AREA_IDS["ENERGY"], f"{zone}.energy", index, your_index)
        card_state.attached_to_serial = serial
        card_state.attachment_kind = 1
        attached.append(card_state)
    for index, card in enumerate(tools):
        card_state = _card_instance(card, AREA_IDS["TOOL"], f"{zone}.tool", index, your_index)
        card_state.attached_to_serial = serial
        card_state.attachment_kind = 2
        attached.append(card_state)
    for index, card in enumerate(pre_evolution):
        card_state = _card_instance(card, AREA_IDS["PRE_EVOLUTION"], f"{zone}.pre_evolution", index, your_index)
        card_state.attached_to_serial = serial
        card_state.attachment_kind = 3
        attached.append(card_state)
    return attached


def parse_global_snapshot(observation: Any) -> GlobalSnapshot:
    state = _value(observation, "current")
    select = _value(observation, "select")
    tracked_state_fields = (
        "turn",
        "turnActionCount",
        "yourIndex",
        "firstPlayer",
        "supporterPlayed",
        "stadiumPlayed",
        "energyAttached",
        "retreated",
        "result",
        "stadium",
        "looking",
        "players",
    )
    tracked_select_fields = (
        "type",
        "context",
        "minCount",
        "maxCount",
        "remainDamageCounter",
        "remainEnergyCost",
        "effect",
        "contextCard",
    )
    field_states = {
        name: _field_state(state, name, applicable=state is not None)
        for name in tracked_state_fields
    }
    field_states.update(
        {
            f"select.{name}": _field_state(select, name, applicable=select is not None)
            for name in tracked_select_fields
        }
    )
    if state is None:
        return GlobalSnapshot(field_states=field_states)
    if _int(_value(state, "firstPlayer"), -1) == -1:
        field_states["firstPlayer"] = FieldState.UNKNOWN
    if _int(_value(state, "result"), -1) == -1:
        field_states["result"] = FieldState.UNKNOWN
    your_index = _int(_value(state, "yourIndex"), 0) or 0
    players = _value(state, "players", []) or []
    player_counts = []
    for player in players[:2]:
        player_counts.append(
            {
                "active": len(_value(player, "active", []) or []),
                "bench": len(_value(player, "bench", []) or []),
                "deck": _int(_value(player, "deckCount"), 0) or 0,
                "discard": len(_value(player, "discard", []) or []),
                "prize": len(_value(player, "prize", []) or []),
                "hand": _int(_value(player, "handCount"), 0) or 0,
            }
        )
    for player_index, player in enumerate(players[:2]):
        for name in ("active", "bench", "benchMax", "deckCount", "discard", "prize", "handCount", "hand"):
            field_states[f"players[{player_index}].{name}"] = _field_state(player, name)
    looking = _value(state, "looking")
    logs = _value(observation, "logs", []) or []
    log_count = len(logs)
    reverse_log_count = 0
    public_card_log_count = 0
    for log in logs:
        log_type = _int(_value(log, "type"), -1)
        if log_type in {5, 7}:
            reverse_log_count += 1
        if _value(log, "cardId") is not None:
            public_card_log_count += 1
    return GlobalSnapshot(
        turn=_int(_value(state, "turn"), 0) or 0,
        turn_action_count=_int(_value(state, "turnActionCount"), 0) or 0,
        your_index=your_index,
        first_player=_int(_value(state, "firstPlayer"), -1),
        supporter_played=bool(_value(state, "supporterPlayed", False)),
        stadium_played=bool(_value(state, "stadiumPlayed", False)),
        energy_attached=bool(_value(state, "energyAttached", False)),
        retreated=bool(_value(state, "retreated", False)),
        result=_int(_value(state, "result"), -1),
        stadium_count=len(_value(state, "stadium", []) or []),
        looking_count=len(looking or []),
        select_type=_int(_value(select, "type"), -1) if select is not None else -1,
        select_context=_int(_value(select, "context"), -1) if select is not None else -1,
        select_min_count=_int(_value(select, "minCount"), 0) if select is not None else 0,
        select_max_count=_int(_value(select, "maxCount"), 0) if select is not None else 0,
        remain_damage_counter=_int(_value(select, "remainDamageCounter"), 0) if select is not None else 0,
        remain_energy_cost=_int(_value(select, "remainEnergyCost"), 0) if select is not None else 0,
        player_counts=player_counts,
        current_log_count=log_count,
        current_reverse_log_count=reverse_log_count,
        current_public_card_log_count=public_card_log_count,
        field_states=field_states,
    )


def parse_card_instances(observation: Any) -> list[CardInstanceState]:
    state = _value(observation, "current")
    if state is None:
        return []
    your_index = _int(_value(state, "yourIndex"), 0) or 0
    instances: list[CardInstanceState] = []
    for player_index, player in enumerate(_value(state, "players", []) or []):
        condition_fields = ("poisoned", "burned", "asleep", "paralyzed", "confused")
        active_conditions = [
            bool(_value(player, field_name, False))
            for field_name in condition_fields
        ]
        active_conditions_valid = all(_has_field(player, field_name) for field_name in condition_fields)
        for index, pokemon in enumerate(_value(player, "active", []) or []):
            instances.extend(
                _pokemon_instance(
                    pokemon,
                    AREA_IDS["ACTIVE"],
                    "active",
                    index,
                    player_index,
                    your_index,
                    active_conditions,
                    active_conditions_valid,
                )
            )
        for index, pokemon in enumerate(_value(player, "bench", []) or []):
            instances.extend(_pokemon_instance(pokemon, AREA_IDS["BENCH"], "bench", index, player_index, your_index))
        hand = _value(player, "hand")
        if hand is not None:
            for index, card in enumerate(hand or []):
                instances.append(_card_instance(card, AREA_IDS["HAND"], "hand", index, your_index))
        for index, card in enumerate(_value(player, "discard", []) or []):
            instances.append(_card_instance(card, AREA_IDS["DISCARD"], "discard", index, your_index))
        for index, card in enumerate(_value(player, "prize", []) or []):
            if card is None:
                instances.append(_hidden_instance(AREA_IDS["PRIZE"], "prize", index, player_index, your_index))
            else:
                instances.append(_card_instance(card, AREA_IDS["PRIZE"], "prize", index, your_index))
    for index, card in enumerate(_value(state, "stadium", []) or []):
        instances.append(_card_instance(card, AREA_IDS["STADIUM"], "stadium", index, your_index))
    looking = _value(state, "looking")
    if looking is not None:
        for index, card in enumerate(looking or []):
            if card is None:
                instances.append(_hidden_instance(AREA_IDS["LOOKING"], "looking", index, None, your_index))
            else:
                instances.append(_card_instance(card, AREA_IDS["LOOKING"], "looking", index, your_index))
    select = _value(observation, "select")
    select_deck = _value(select, "deck")
    if select_deck is not None:
        existing_serials = {instance.serial for instance in instances if instance.serial is not None}
        for index, card in enumerate(select_deck or []):
            if card is None:
                instances.append(_hidden_instance(AREA_IDS["LOOKING"], "select.deck", index, your_index, your_index))
                continue
            serial = _int(_value(card, "serial"), None)
            if serial is not None and serial in existing_serials:
                continue
            instances.append(_card_instance(card, AREA_IDS["LOOKING"], "select.deck", index, your_index, source="select.deck"))
    counts = Counter(
        (instance.relative_player, instance.area, int(instance.card_id))
        for instance in instances
        if instance.card_id is not None
    )
    for instance in instances:
        if instance.card_id is not None:
            instance.copy_count = counts[(instance.relative_player, instance.area, int(instance.card_id))]
    return instances


def parse_events(observation: Any) -> list[GameEvent]:
    events: list[GameEvent] = []
    current = _value(observation, "current")
    your_index = _int(_value(current, "yourIndex"), 0) or 0
    turn = _int(_value(current, "turn"), 0) or 0
    turn_action_count = _int(_value(current, "turnActionCount"), 0) or 0
    for event_position, log in enumerate(_value(observation, "logs", []) or []):
        event_type = _int(_value(log, "type"), -1)
        from_area = _int(_value(log, "fromArea"), None)
        to_area = _int(_value(log, "toArea"), None)
        player_index = _int(_value(log, "playerIndex"), None)
        card_id = _int(_value(log, "cardId"), None)
        serial = _int(_value(log, "serial"), None)
        raw = _raw(log)
        events.append(
            GameEvent(
                event_type=event_type,
                player_index=player_index,
                actor_relative=_relative(player_index, your_index),
                card_id=card_id,
                serial=serial,
                from_area=from_area,
                to_area=to_area,
                target_card_id=_int(_value(log, "cardIdTarget"), None),
                target_serial=_int(_value(log, "serialTarget"), None),
                attack_id=_int(_value(log, "attackId"), None),
                value=_int(_value(log, "value"), None),
                coin_result=_int(_value(log, "coinResult"), None),
                is_reverse=event_type in {5, 7},
                identity_visible=card_id is not None,
                event_position_in_batch=event_position,
                observed_turn=turn,
                position_in_turn=turn_action_count,
                field_states={
                    canonical: _field_state(log, raw_name)
                    for canonical, raw_name in (
                        ("event_type", "type"),
                        ("player_index", "playerIndex"),
                        ("card_id", "cardId"),
                        ("serial", "serial"),
                        ("from_area", "fromArea"),
                        ("to_area", "toArea"),
                        ("target_card_id", "cardIdTarget"),
                        ("target_serial", "serialTarget"),
                        ("attack_id", "attackId"),
                        ("value", "value"),
                        ("coin_result", "coinResult"),
                    )
                },
                raw=raw,
            )
        )
    return events


def parse_select_options(observation: Any) -> list[dict[str, Any]]:
    select = _value(observation, "select")
    if select is None:
        return []
    return [_raw(option) for option in (_value(select, "option", []) or [])]


def parse_observation(observation: Any) -> ParsedObservation:
    select = _value(observation, "select")
    effect = _value(select, "effect")
    context_card = _value(select, "contextCard")
    return ParsedObservation(
        global_snapshot=parse_global_snapshot(observation),
        card_instances=parse_card_instances(observation),
        events=parse_events(observation),
        select_options=parse_select_options(observation),
        effect_reference=_raw(effect) if effect is not None else None,
        context_card_reference=_raw(context_card) if context_card is not None else None,
        effect_presence=_field_state(select, "effect", applicable=select is not None),
        context_card_presence=_field_state(select, "contextCard", applicable=select is not None),
        option_field_states=[
            {
                name: _field_state(option, name)
                for name in (
                    "type",
                    "number",
                    "area",
                    "index",
                    "playerIndex",
                    "toolIndex",
                    "energyIndex",
                    "count",
                    "inPlayArea",
                    "inPlayIndex",
                    "attackId",
                    "cardId",
                    "serial",
                    "specialConditionType",
                )
            }
            for option in (_value(select, "option", []) or [])
        ],
    )
