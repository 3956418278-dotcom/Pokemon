from __future__ import annotations

import pytest

from data.game_memory import GameMemoryState
from data.observation_parser import parse_observation
from data.state_schema import (
    AREA_IDS,
    CARD_APPEARANCE_FEATURE_DIM,
    CARD_DYNAMIC_FEATURE_DIM,
    EVENT_FEATURE_DIM,
    LEDGER_FEATURE_DIM,
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
    assert any(x.attachment_kind == 1 and x.attached_to_serial == 10 for x in parsed.card_instances)
    assert any(x.area == AREA_IDS["LOOKING"] and not x.is_visible for x in parsed.card_instances)


def test_dynamic_batch_and_memory_shapes() -> None:
    pytest.importorskip("torch")
    parsed = parse_observation(_observation())
    memory = GameMemoryState().update_from_parsed(parsed)
    appearance = memory.appearance_features(parsed.card_instances)
    batch = collate_card_dynamic(parsed.card_instances, appearance)
    assert batch.dynamic_features.shape == (len(parsed.card_instances), CARD_DYNAMIC_FEATURE_DIM)
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
