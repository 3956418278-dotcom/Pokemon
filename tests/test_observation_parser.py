from __future__ import annotations

import pytest

from data.decision_schema import DetailUsageState, FieldState
from data.game_memory import GameMemoryState
from data.observation_parser import parse_observation
from data.state_schema import (
    AREA_IDS,
    BOOLEAN_FEATURE_NAMES,
    CARD_APPEARANCE_FEATURE_DIM,
    EVENT_FEATURE_DIM,
    LEDGER_FEATURE_DIM,
    NUMERICAL_FEATURE_NAMES,
    CardInstanceState,
    collate_card_dynamic,
)


def _card(card_id: int, serial: int, player: int) -> dict:
    return {"id": card_id, "serial": serial, "playerIndex": player}


def _pokemon(card_id: int, serial: int, player: int, hp: int = 100) -> dict:
    return {
        "id": card_id,
        "serial": serial,
        "playerIndex": player,
        "hp": hp,
        "maxHp": 120,
        "appearThisTurn": True,
        "energies": [0, 2],
        "energyCards": [_card(1, serial + 100, player)],
        "tools": [_card(9, serial + 200, player)],
        "preEvolution": [_card(20, serial + 300, player)],
    }


def _observation() -> dict:
    return {
        "current": {
            "turn": 3,
            "turnActionCount": 2,
            "yourIndex": 0,
            "firstPlayer": 0,
            "supporterPlayed": False,
            "stadiumPlayed": False,
            "energyAttached": True,
            "retreated": False,
            "result": -1,
            "stadium": [_card(100, 900, 0)],
            "looking": [None, _card(101, 901, 0)],
            "players": [
                {
                    "active": [_pokemon(21, 10, 0)],
                    "bench": [_pokemon(22, 11, 0, hp=80)],
                    "benchMax": 5,
                    "deckCount": 45,
                    "discard": [_card(30, 12, 0)],
                    "prize": [None, None],
                    "handCount": 5,
                    "hand": [_card(40, 13, 0)],
                    "poisoned": True,
                    "burned": False,
                    "asleep": False,
                    "paralyzed": False,
                    "confused": True,
                },
                {
                    "active": [None],
                    "bench": [],
                    "benchMax": 5,
                    "deckCount": 46,
                    "discard": [],
                    "prize": [None],
                    "handCount": 6,
                    "hand": None,
                    "poisoned": False,
                    "burned": False,
                    "asleep": False,
                    "paralyzed": False,
                    "confused": False,
                },
            ],
        },
        "select": {
            "type": 0,
            "context": 0,
            "minCount": 1,
            "maxCount": 1,
            "remainDamageCounter": 0,
            "remainEnergyCost": 0,
            "option": [{"type": 13, "attackId": 1}],
            "deck": None,
            "contextCard": None,
            "effect": None,
        },
        "logs": [
            {"type": 11, "playerIndex": 0, "cardId": 1, "serial": 110, "cardIdTarget": 21, "serialTarget": 10},
            {"type": 5, "playerIndex": 1},
        ],
    }


def test_parse_observation_preserves_visibility_and_instances() -> None:
    parsed = parse_observation(_observation())
    assert parsed.global_snapshot.turn == 3
    assert parsed.global_snapshot.player_counts[1]["hand"] == 6
    assert parsed.global_snapshot.current_log_count == 2
    assert parsed.global_snapshot.current_reverse_log_count == 1
    assert parsed.global_snapshot.current_public_card_log_count == 1
    assert len(parsed.select_options) == 1
    opponent_hidden_active = [x for x in parsed.card_instances if x.zone == "active" and x.relative_player == 1][0]
    assert opponent_hidden_active.card_id is None
    assert not opponent_hidden_active.is_visible
    own_hand = [x for x in parsed.card_instances if x.zone == "hand"]
    assert len(own_hand) == 1
    assert all(x.relative_player == 0 for x in own_hand)
    assert not any(x.zone == "hand" and x.relative_player == 1 for x in parsed.card_instances)
    active = [x for x in parsed.card_instances if x.serial == 10][0]
    assert active.special_conditions[0] is True
    assert active.special_conditions[4] is True
    assert active.energy_counts[0] == 1
    assert active.energy_counts[2] == 1
    assert active.energy_card_ids == [1]
    assert active.tool_card_ids == [9]
    assert active.pre_evolution_card_ids == [20]
    assert any(x.attachment_kind == 1 and x.attached_to_serial == 10 for x in parsed.card_instances)
    assert any(x.area == AREA_IDS["LOOKING"] and not x.is_visible for x in parsed.card_instances)
    states, sources = active.detail_usage(2)
    assert states == [DetailUsageState.UNKNOWN, DetailUsageState.UNKNOWN]
    assert sources == [None, None]


def test_dynamic_batch_and_memory_shapes() -> None:
    pytest.importorskip("torch")
    parsed = parse_observation(_observation())
    memory = GameMemoryState().update_from_parsed(parsed)
    appearance = memory.appearance_features(parsed.card_instances)
    batch = collate_card_dynamic(parsed.card_instances, appearance)
    assert batch.numerical_features.shape == (len(parsed.card_instances), len(NUMERICAL_FEATURE_NAMES))
    assert batch.numerical_mask.shape == batch.numerical_features.shape
    assert batch.energy_counts.shape == (len(parsed.card_instances), 12)
    assert batch.condition_flags.shape == (len(parsed.card_instances), 5)
    assert batch.boolean_features.shape == (len(parsed.card_instances), len(BOOLEAN_FEATURE_NAMES))
    assert batch.appearance_features.shape == (len(parsed.card_instances), CARD_APPEARANCE_FEATURE_DIM)
    assert len(memory.recent_events) == 2
    assert memory.serials[110].attached is True
    assert len(memory.ledger_features(parsed.global_snapshot.your_index)) == 2
    assert len(memory.ledger_features(parsed.global_snapshot.your_index)[0]) == LEDGER_FEATURE_DIM
    assert len(memory.recent_event_features()[0]) == EVENT_FEATURE_DIM


def test_log_type_zero_is_preserved() -> None:
    obs = _observation()
    obs["logs"] = [{"type": 0, "playerIndex": 0}]
    parsed = parse_observation(obs)
    assert parsed.events[0].event_type == 0


def test_events_keep_missingness_batch_position_and_player_relative_age() -> None:
    first = parse_observation(_observation())
    event = first.events[0]
    assert event.actor_relative == 0
    assert event.batch_position == 0
    assert event.observed_at_turn_action_count == 2
    assert event.identity_visible
    assert event.field_states["card_id"] is FieldState.PRESENT
    assert event.field_states["coin_result"] is FieldState.MISSING

    memory = GameMemoryState().update_from_parsed(first)
    second_observation = _observation()
    second_observation["current"]["turn"] = 4
    second_observation["logs"] = []
    memory.update_from_parsed(parse_observation(second_observation))
    assert memory.recent_events[0].observation_age == 1
    assert memory.recent_events[0].turn_age == 1
    # Memory owns temporal mutation; the arrival batch remains unchanged.
    assert first.events[0].observation_age == 0


def test_anonymous_hidden_pool_flows_are_separate_from_card_id_memory() -> None:
    observation = _observation()
    observation["logs"] = [
        {"type": 7, "playerIndex": 1, "fromArea": 1, "toArea": 2, "quantity": 2}
    ]
    memory = GameMemoryState().update_from_parsed(parse_observation(observation))
    pools = memory.anonymous_hidden_pools_record(your_index=0)
    assert pools.self_unknown_deck_count == 45
    assert pools.self_unknown_prize_count == 2
    assert pools.cumulative_anonymous_zone_transitions_by_side[1] == {
        "anonymous_deck_out_count": 2,
        "anonymous_hand_in_count": 2,
    }
    assert pools.opponent_unknown_hand_count == 6
    records = memory.card_id_memory_records(your_index=0)
    assert records
    assert records[0].known_serial_count >= 1
    assert records[0].visible_observation_count >= records[0].known_serial_count


def test_anonymous_single_move_does_not_clear_known_serial_locations() -> None:
    initial = _observation()
    initial["logs"] = []
    initial["current"]["players"][0]["hand"] = [
        _card(40, 13, 0),
        _card(41, 14, 0),
        _card(42, 15, 0),
    ]
    initial["current"]["players"][0]["handCount"] = 3
    memory = GameMemoryState().update_from_parsed(parse_observation(initial))

    moved = _observation()
    moved["current"]["players"][0]["hand"] = initial["current"]["players"][0]["hand"]
    moved["current"]["players"][0]["handCount"] = 3
    moved["logs"] = [
        {"type": 6, "playerIndex": 0, "fromArea": 2, "toArea": 3, "quantity": 1}
    ]
    memory.update_from_parsed(parse_observation(moved))
    assert [memory.serials[serial].current_area for serial in (13, 14, 15)] == [2, 2, 2]
    pools = memory.anonymous_hidden_pools_record(your_index=0)
    assert pools.cumulative_anonymous_zone_transitions_by_side[0][
        "anonymous_hand_out_count"
    ] == 1


def test_field_specific_unknown_sentinels_do_not_hide_valid_negative_values() -> None:
    observation = _observation()
    observation["current"]["firstPlayer"] = -1
    observation["current"]["result"] = -1
    observation["select"]["remainDamageCounter"] = -10
    parsed = parse_observation(observation)
    states = parsed.global_snapshot.field_states
    assert states["firstPlayer"] is FieldState.UNKNOWN
    assert states["result"] is FieldState.UNKNOWN
    assert states["select.remainDamageCounter"] is FieldState.PRESENT


def test_missing_null_and_unknown_field_states_are_distinct() -> None:
    observation = _observation()
    del observation["current"]["energyAttached"]
    observation["select"]["contextCard"] = None
    observation["current"]["firstPlayer"] = -1
    states = parse_observation(observation).global_snapshot.field_states
    assert states["energyAttached"] is FieldState.MISSING
    assert states["select.contextCard"] is FieldState.EXPLICIT_NULL
    assert states["firstPlayer"] is FieldState.UNKNOWN


def test_dynamic_batch_distinguishes_missing_hp_from_real_zero_and_preserves_serial_zero() -> None:
    torch = pytest.importorskip("torch")
    instances = [
        CardInstanceState(21, 0, 0, 0, AREA_IDS["ACTIVE"], "active", 0, is_pokemon=True, hp=0, max_hp=120),
        CardInstanceState(40, 7, 0, 0, AREA_IDS["HAND"], "hand", 0, hp=None, max_hp=None),
    ]
    batch = collate_card_dynamic(instances)
    assert batch.serials.tolist() == [0, 7]
    assert batch.numerical_features[0, 0].item() == 0.0
    assert batch.numerical_mask[0, 0].item() == 1.0
    assert batch.numerical_features[1, 0].item() == 0.0
    assert batch.numerical_mask[1, 0].item() == 0.0
    assert torch.isfinite(batch.numerical_features).all()


def test_same_card_id_with_different_serials_stays_as_two_instances() -> None:
    obs = _observation()
    obs["current"]["players"][0]["bench"] = [
        _pokemon(21, 100, 0, hp=120),
        _pokemon(21, 101, 0, hp=80),
    ]
    parsed = parse_observation(obs)
    matches = [item for item in parsed.card_instances if item.card_id == 21 and item.zone == "bench"]
    assert [item.serial for item in matches] == [100, 101]
    assert [item.hp for item in matches] == [120, 80]


def test_copy_count_is_grouped_without_losing_serials() -> None:
    obs = _observation()
    obs["current"]["players"][0]["hand"] = [_card(40, 13, 0), _card(40, 14, 0)]
    parsed = parse_observation(obs)
    hand = [item for item in parsed.card_instances if item.zone == "hand" and item.card_id == 40]
    assert [item.serial for item in hand] == [13, 14]
    assert [item.copy_count for item in hand] == [2, 2]


def test_select_deck_card_instance_is_currently_in_looking_zone() -> None:
    observation = _observation()
    observation["select"]["deck"] = [_card(77, 707, 0)]
    parsed = parse_observation(observation)
    instance = next(item for item in parsed.card_instances if item.serial == 707)
    assert instance.area == AREA_IDS["LOOKING"]
    assert instance.zone == "select.deck"
    assert instance.source == "select.deck"


def test_unknown_energy_enum_invalidates_energy_supervision() -> None:
    obs = _observation()
    obs["current"]["players"][0]["active"][0]["energies"] = [5, 99]
    parsed = parse_observation(obs)
    active = [item for item in parsed.card_instances if item.serial == 10][0]
    assert active.energy_counts[5] == 1
    assert not active.energy_counts_valid


def test_empty_energy_list_is_valid_zero_but_missing_list_is_unknown() -> None:
    empty_obs = _observation()
    empty_obs["current"]["players"][0]["active"][0]["energies"] = []
    empty = parse_observation(empty_obs).card_instances[0]
    assert empty.energy_counts == [0] * 12
    assert empty.energy_counts_valid

    missing_obs = _observation()
    del missing_obs["current"]["players"][0]["active"][0]["energies"]
    missing = parse_observation(missing_obs).card_instances[0]
    assert missing.energy_counts == [0] * 12
    assert not missing.energy_counts_valid


def test_missing_or_partial_condition_fields_are_not_valid_zeroes() -> None:
    missing_obs = _observation()
    player = missing_obs["current"]["players"][0]
    for name in ("poisoned", "burned", "asleep", "paralyzed", "confused"):
        player.pop(name)
    missing = parse_observation(missing_obs).card_instances[0]
    assert missing.special_conditions == [False] * 5
    assert not missing.special_conditions_valid

    partial_obs = _observation()
    del partial_obs["current"]["players"][0]["confused"]
    partial = parse_observation(partial_obs).card_instances[0]
    assert partial.special_conditions[0]
    assert not partial.special_conditions_valid
