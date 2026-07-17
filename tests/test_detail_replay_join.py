from __future__ import annotations

import json
from pathlib import Path

from data.detail_replay_join import (
    assign_episode_splits,
    join_replay_samples,
    write_detail_join_audit,
)
from data.replay_dataset import ReplayDecisionDataset
from data.state_schema import GameEvent
from data.static_detail_catalog import StaticDetailCatalog


def _write_catalog(root: Path) -> StaticDetailCatalog:
    cards = [
        {"card_id": 10, "name": "Attacker", "card_type": "POKEMON", "detail_ids": [0]},
        {"card_id": 20, "name": "Effect Card", "card_type": "ITEM", "detail_ids": [1]},
        {"card_id": 30, "name": "Ambiguous Effect", "card_type": "ITEM", "detail_ids": [2, 3]},
    ]
    details = [
        {
            "detail_id": 0,
            "detail_index": 0,
            "card_id": 10,
            "detail_type": "ATTACK",
            "detail_subtype": "ATTACK",
            "detail_name": "Test Attack",
            "attack_id": 77,
            "source_row": 2,
            "source_line": 3,
            "source_fields": {},
        },
        {
            "detail_id": 1,
            "detail_index": 1,
            "card_id": 20,
            "detail_type": "CARD_EFFECT",
            "detail_subtype": "TRAINER_EFFECT",
            "detail_name": "",
            "attack_id": None,
            "source_row": 3,
            "source_line": 4,
            "source_fields": {},
        },
        {
            "detail_id": 2,
            "detail_index": 2,
            "card_id": 30,
            "detail_type": "CARD_EFFECT",
            "detail_subtype": "TRAINER_EFFECT",
            "detail_name": "First",
            "attack_id": None,
            "source_row": 4,
            "source_line": 5,
            "source_fields": {},
        },
        {
            "detail_id": 3,
            "detail_index": 3,
            "card_id": 30,
            "detail_type": "CARD_EFFECT",
            "detail_subtype": "TRAINER_EFFECT",
            "detail_name": "Second",
            "attack_id": None,
            "source_row": 5,
            "source_line": 6,
            "source_fields": {},
        },
    ]
    (root / "cards.json").write_text(json.dumps(cards), encoding="utf-8")
    (root / "details.json").write_text(json.dumps(details), encoding="utf-8")
    (root / "card_id_to_index.json").write_text(
        json.dumps({"10": 0, "20": 1, "30": 2}), encoding="utf-8"
    )
    return StaticDetailCatalog.from_artifact_dir(root)


def _observation(select: dict | None) -> dict:
    return {"current": None, "logs": [], "select": select}


def _select(option: dict, *, select_type: int = 0, context: int = 0, effect=None) -> dict:
    result = {
        "type": select_type,
        "context": context,
        "minCount": 1,
        "maxCount": 1,
        "option": [option],
    }
    if effect is not None:
        result["effect"] = effect
    return result


def _write_replay(path: Path, episode_id: int, selections: list[dict]) -> None:
    steps = []
    for index, selection in enumerate(selections):
        steps.append(
            [
                {
                    "action": [] if index == 0 else [0],
                    "observation": _observation(selection),
                    "reward": 0,
                    "status": "ACTIVE",
                }
            ]
        )
    steps.append(
        [
            {
                "action": [0],
                "observation": _observation(None),
                "reward": 0,
                "status": "DONE",
            }
        ]
    )
    path.write_text(
        json.dumps({"id": str(episode_id), "info": {"EpisodeId": episode_id}, "steps": steps}),
        encoding="utf-8",
    )


def test_catalog_and_join_use_only_stable_detail_references(tmp_path: Path) -> None:
    catalog = _write_catalog(tmp_path)
    assert catalog.card_id_to_detail_indices == {10: [0], 20: [1], 30: [2, 3]}
    assert catalog.resolve(("attack_id", 77)).detail_index == 0
    assert catalog.resolve(("card_detail_local_index", 20, 0)).detail_index == 1
    assert catalog.resolve(("card_effect", 20)).detail_index == 1
    assert catalog.resolve(("card_effect", 30)).status == "CONFLICT"

    attack_path = tmp_path / "attack.json"
    ordinary_path = tmp_path / "ordinary.json"
    unknown_path = tmp_path / "unknown.json"
    trigger_path = tmp_path / "trigger.json"
    trainer_path = tmp_path / "trainer.json"
    conflict_path = tmp_path / "trainer-conflict.json"
    _write_replay(attack_path, 1, [_select({"type": 13, "attackId": 77})])
    _write_replay(ordinary_path, 2, [_select({"type": 14})])
    _write_replay(
        unknown_path,
        3,
        [_select({"type": 3}, select_type=1, context=7, effect={"id": 20, "serial": 9})],
    )
    _write_replay(trigger_path, 4, [_select({"type": 14})])
    _write_replay(trainer_path, 5, [_select({"type": 7, "cardId": 20, "serial": 7})])
    _write_replay(conflict_path, 6, [_select({"type": 7, "cardId": 30, "serial": 8})])
    dataset = ReplayDecisionDataset.from_paths(
        [attack_path, ordinary_path, unknown_path, trigger_path, trainer_path, conflict_path]
    )
    trigger_sample = next(sample for sample in dataset.samples if sample.episode_id == 4)
    assert trigger_sample.transition_parsed_after is not None
    trigger_sample.transition_parsed_after.events.append(
        GameEvent(
            event_type=99,
            card_id=20,
            serial=12,
            raw={"cardId": 20, "serial": 12, "detailLocalIndex": 0},
        )
    )
    trainer_sample = next(sample for sample in dataset.samples if sample.episode_id == 5)
    assert trainer_sample.transition_parsed_after is not None
    trainer_sample.transition_parsed_after.events.append(
        GameEvent(
            event_type=15,
            card_id=10,
            serial=99,
            attack_id=77,
            raw={"cardId": 10, "serial": 99, "attackId": 77},
        )
    )
    splits = assign_episode_splits(dataset.samples)
    joined = join_replay_samples(dataset.samples, catalog, episode_splits=splits)
    by_episode = {record.episode_id: record for record in joined}

    direct = by_episode[1]
    assert (direct.operation_kind, direct.detail_mapping_status) == ("DETAIL_DIRECT", "EXACT")
    assert (direct.resolved_detail_id, direct.resolved_detail_index) == (0, 0)
    assert direct.detail_supervision_mask and direct.transition_supervision_mask

    ordinary = by_episode[2]
    assert ordinary.operation_kind == "NON_DETAIL_GAME_ACTION"
    assert ordinary.detail_mapping_status == "NOT_APPLICABLE"
    assert not ordinary.detail_supervision_mask and not ordinary.transition_supervision_mask

    unknown = by_episode[3]
    assert unknown.operation_kind == "UNKNOWN_EFFECT"
    assert unknown.detail_mapping_status == "UNRESOLVED"
    assert not unknown.detail_supervision_mask and unknown.transition_supervision_mask
    assert unknown.resolved_detail_index == -1

    trigger = by_episode[4]
    assert (trigger.operation_kind, trigger.detail_mapping_status) == ("DETAIL_TRIGGER", "EXACT")
    assert (trigger.resolved_detail_id, trigger.resolved_detail_index) == (1, 1)
    assert trigger.source_card_id == 20 and trigger.source_serial == 12

    trainer = by_episode[5]
    assert (trainer.operation_kind, trainer.detail_mapping_status) == ("DETAIL_DIRECT", "EXACT")
    assert (trainer.resolved_detail_id, trainer.resolved_detail_index) == (1, 1)
    assert trainer.engine_detail_reference["kind"] == "card_effect"

    conflict = by_episode[6]
    assert (conflict.operation_kind, conflict.detail_mapping_status) == (
        "UNKNOWN_EFFECT",
        "CONFLICT",
    )
    assert conflict.candidate_detail_ids == [2, 3]
    assert all(record.split == splits[record.replay_key] for record in joined)

    summary = write_detail_join_audit(tmp_path / "audit", catalog, joined)
    assert summary["exact_mapping_count"] == 3
    assert summary["detail_trigger_count"] == 1
    assert summary["unknown_effect_count"] == 2
    assert summary["conflict_mapping_count"] == 1
    assert {
        "summary.json",
        "detail_coverage.csv",
        "unresolved_references.csv",
        "mapping_conflicts.csv",
        "unknown_card_ids.csv",
        "operation_distribution.csv",
        "joined_transitions.jsonl",
    } <= {path.name for path in (tmp_path / "audit").iterdir()}


def test_pending_selection_chain_uses_final_post_action_state(tmp_path: Path) -> None:
    catalog = _write_catalog(tmp_path)
    replay_path = tmp_path / "pending.json"
    _write_replay(
        replay_path,
        4,
        [
            _select({"type": 13, "attackId": 77}),
            _select({"type": 3}, select_type=1, context=7),
            _select({"type": 14}),
        ],
    )
    dataset = ReplayDecisionDataset.from_paths([replay_path])
    assert len(dataset) == 3
    joined = join_replay_samples(dataset.samples, catalog)
    assert len(joined) == 2
    attack = joined[0]
    assert attack.operation_kind == "DETAIL_DIRECT"
    assert attack.decision_span_count == 2
    assert (attack.step_start, attack.step_end) == (0, 2)
    assert attack.state_after is not None
    assert attack.state_delta["status"] == "COMPLETE"
