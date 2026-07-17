from __future__ import annotations

import pytest

from data.decision_schema import ActionSemantics, ResidualMarginals
from data.hidden_belief import (
    build_residual_marginals,
    residual_ipf_expected_counts,
    unresolved_zone_entropy,
)
from data.legal_options import (
    build_action_target,
    option_equivalence_key,
    policy_loss_mask,
    policy_mask_decision,
)
from data.replay_dataset import infer_action_semantics


def test_residual_marginals_require_matching_nonnegative_mass() -> None:
    ResidualMarginals({1: 1.0, 2: 2.0}, {"HAND": 1.0, "DECK": 2.0}).validate()
    with pytest.raises(ValueError, match="disagree"):
        ResidualMarginals({1: 2.0}, {"HAND": 1.0}).validate()
    with pytest.raises(ValueError, match="non-negative"):
        ResidualMarginals({1: -1.0}, {"HAND": -1.0}).validate()


def test_action_semantics_are_not_inferred_only_from_max_count() -> None:
    for context in (5, 7, 8, 9):
        assert infer_action_semantics(
            {"type": 1, "context": context, "maxCount": 2, "option": [{"type": 3}, {"type": 3}]},
            [1, 0],
        ) is ActionSemantics.ORDERED_INDEX_SEQUENCE
    assert infer_action_semantics(
        {"type": 1, "context": 22, "maxCount": 2, "option": [{"type": 3}, {"type": 3}]},
        [0, 1],
    ) is ActionSemantics.ORDERED_INDEX_SEQUENCE
    for select_type, context, option_type in (
        (1, 2, 3),
        (1, 15, 3),
        (1, 21, 3),
        (2, 26, 5),
        (2, 27, 4),
        (5, 34, 15),
    ):
        assert infer_action_semantics(
            {
                "type": select_type,
                "context": context,
                "maxCount": 2,
                "option": [{"type": option_type}, {"type": option_type}],
            },
            [1, 0],
        ) is ActionSemantics.UNORDERED_UNIQUE_SUBSET
    assert infer_action_semantics(
        {"type": 8, "context": 38, "maxCount": 1, "option": [{"type": 0}]},
        [0],
    ) is ActionSemantics.COUNT_VALUE


def test_residual_ipf_returns_expected_counts_with_both_marginals() -> None:
    marginals = build_residual_marginals(
        {10: 3.0, 20: 1.0},
        {10: {"HAND": 1.0}},
        {"HAND": 2.0, "DECK": 1.5, "PRIZE": 0.5, "OTHER": 0.0},
    )
    assert marginals.unresolved_by_card == {10: 2.0, 20: 1.0}
    assert marginals.unresolved_by_zone["HAND"] == 1.0
    result = residual_ipf_expected_counts(
        marginals,
        {
            10: {"HAND": 2.0, "DECK": 0.0, "PRIZE": -1.0},
            20: {"HAND": -1.0, "DECK": 1.0, "PRIZE": 0.0},
        },
    )
    assert sum(result[10].values()) == pytest.approx(2.0, abs=1e-5)
    assert sum(result[20].values()) == pytest.approx(1.0, abs=1e-5)
    for zone, expected in marginals.unresolved_by_zone.items():
        assert sum(result[card_id][zone] for card_id in result) == pytest.approx(expected, abs=1e-5)
    entropy = unresolved_zone_entropy(result, marginals.unresolved_by_card)
    assert set(entropy) == {10, 20}
    assert all(value >= 0.0 for value in entropy.values())


def test_equivalence_key_contains_decision_entities_and_resolution_fields() -> None:
    observation = {
        "current": {
            "yourIndex": 0,
            "players": [
                {"hand": [{"id": 1, "serial": 11, "playerIndex": 0}]},
                {"hand": None},
            ],
        },
        "select": {
            "type": 0,
            "context": 0,
            "effect": {"id": 100, "serial": 50, "playerIndex": 0},
            "contextCard": None,
            "option": [{"type": 8, "index": 0, "inPlayArea": 4, "inPlayIndex": 0, "count": 1}],
        },
    }
    key = option_equivalence_key(observation, observation["select"]["option"][0])
    assert key["select_type"] == 0
    assert key["select_context"] == 0
    assert key["option_type"] == 8
    assert key["source_zone"] == 2
    assert key["effect_reference"] == {"id": 100, "playerIndex": 0, "serial": 50}
    assert key["resolution_fields"] == {"count": 1}


def test_count_target_uses_number_and_subset_target_uses_group_counts() -> None:
    count_observation = {
        "current": {"yourIndex": 0, "players": [{}, {}]},
        "select": {
            "type": 8,
            "context": 38,
            "option": [{"type": 0, "number": 1}, {"type": 0, "number": 3}],
        },
    }
    count_target = build_action_target(count_observation, [1], ActionSemantics.COUNT_VALUE)
    assert count_target.count_value == 3
    assert count_target.count_value_to_option_indices == {1: (0,), 3: (1,)}


def test_direct_card_reference_retains_serial_without_interchangeability_rule() -> None:
    observation = {
        "current": {
            "yourIndex": 0,
            "players": [{}, {}],
            "stadium": [{"id": 1260, "serial": 52, "playerIndex": 0}],
        },
        "select": {
            "type": 5,
            "context": 34,
            "option": [
                {"type": 15, "cardId": 1260, "serial": 52},
                {"type": 15, "cardId": 1260, "serial": 52},
            ],
        },
    }
    key = option_equivalence_key(observation, observation["select"]["option"][0])
    assert key["source_zone"] == 7
    assert key["source_identity_and_dynamic_state"] == {"id": 1260, "playerIndex": 0, "serial": 52}
    assert key["resolution_fields"] == {}
    assert "unresolved_direct_identity" not in key
    target = build_action_target(
        observation, [1, 0], ActionSemantics.UNORDERED_UNIQUE_SUBSET
    )
    assert target.equivalence_class_ids == (0, 0)
    assert target.chosen_class_counts == {0: 2}
    assert policy_loss_mask(observation["select"], target) is False


def test_select_deck_resolution_uses_looking_zone() -> None:
    observation = {
        "current": {"yourIndex": 0, "players": [{}, {}], "looking": None},
        "select": {
            "type": 0,
            "context": 0,
            "deck": [{"id": 55, "serial": 9, "playerIndex": 0}],
            "option": [{"type": 3, "area": 1, "index": 0, "playerIndex": 0}],
        },
    }
    key = option_equivalence_key(observation, observation["select"]["option"][0])
    assert key["source_zone"] == 12
    assert key["equivalence_resolution_status"] == "FULLY_RESOLVED"


def test_unresolved_equivalence_never_masks_a_real_policy_choice() -> None:
    observation = {
        "current": {"yourIndex": 0, "players": [{"hand": []}, {}]},
        "select": {
            "type": 0,
            "context": 0,
            "minCount": 1,
            "maxCount": 1,
            "option": [
                {"type": 7, "index": 4},
                {"type": 7, "index": 4},
            ],
        },
    }
    target = build_action_target(observation, [0], ActionSemantics.SINGLE_INDEX)
    assert target.equivalence_resolution_status == "UNRESOLVED"
    assert target.equivalence_class_ids == (0, 1)
    assert policy_mask_decision(observation["select"], target) == (
        True,
        "UNRESOLVED_EQUIVALENCE",
    )


def test_explicit_empty_select_deck_does_not_fall_back_to_current_looking() -> None:
    observation = {
        "current": {
            "yourIndex": 0,
            "players": [{}, {}],
            "looking": [{"id": 99, "serial": 1, "playerIndex": 0}],
        },
        "select": {
            "type": 0,
            "context": 0,
            "deck": [],
            "option": [{"type": 3, "area": 1, "index": 0, "playerIndex": 0}],
        },
    }
    key = option_equivalence_key(observation, observation["select"]["option"][0])
    assert key["source_identity_and_dynamic_state"] is None
    assert key["equivalence_resolution_status"] == "UNRESOLVED"


def test_attachment_option_targets_its_actual_parent_zone() -> None:
    observation = {
        "current": {
            "yourIndex": 0,
            "players": [
                {
                    "active": [
                        {
                            "id": 10,
                            "serial": 1,
                            "playerIndex": 0,
                            "tools": [{"id": 20, "serial": 2, "playerIndex": 0}],
                        }
                    ]
                },
                {},
            ],
        },
        "select": {
            "type": 2,
            "context": 27,
            "option": [
                {"type": 4, "area": 4, "index": 0, "toolIndex": 0, "playerIndex": 0}
            ],
        },
    }
    key = option_equivalence_key(observation, observation["select"]["option"][0])
    assert key["source_zone"] == 9
    assert key["target_zone"] == 4


def test_missing_card_id_does_not_merge_named_serials() -> None:
    observation = {
        "current": {
            "yourIndex": 0,
            "players": [
                {
                    "hand": [
                        {"name": "Alpha", "serial": 1, "playerIndex": 0},
                        {"name": "Beta", "serial": 2, "playerIndex": 0},
                    ]
                },
                {},
            ],
        },
        "select": {
            "type": 0,
            "context": 0,
            "minCount": 1,
            "maxCount": 1,
            "option": [{"type": 7, "index": 0}, {"type": 7, "index": 1}],
        },
    }
    target = build_action_target(observation, [0], ActionSemantics.SINGLE_INDEX)
    assert target.equivalence_class_ids == (0, 1)


def test_incomplete_effect_reference_prevents_equivalence_masking() -> None:
    observation = {
        "current": {"yourIndex": 0, "players": [{}, {}]},
        "select": {
            "type": 0,
            "context": 0,
            "minCount": 1,
            "maxCount": 1,
            "effect": {"id": 100},
            "option": [{"type": 1}, {"type": 1}],
        },
    }
    target = build_action_target(observation, [0], ActionSemantics.SINGLE_INDEX)
    assert target.equivalence_resolution_status == "PARTIALLY_RESOLVED"
    assert policy_mask_decision(observation["select"], target) == (
        True,
        "UNRESOLVED_EQUIVALENCE",
    )
