from __future__ import annotations

import json
import importlib.util
import gzip
import hashlib
import zipfile
from pathlib import Path

import pytest

from data.replay_dataset import (
    ReplayDecisionDataset,
    build_replay_decision_contract,
    collate_replay_decisions,
    export_replay_decisions,
    iter_replay_paths,
    rebuild_replay_decision_from_reference,
    stable_replay_key,
)

_KERNEL_PATH = Path("kaggle/kernels/replay_extract/extract_popular_decks.py")
_KERNEL_SPEC = importlib.util.spec_from_file_location("local_extract_popular_decks", _KERNEL_PATH)
assert _KERNEL_SPEC is not None and _KERNEL_SPEC.loader is not None
_KERNEL = importlib.util.module_from_spec(_KERNEL_SPEC)
_KERNEL_SPEC.loader.exec_module(_KERNEL)
build_card_frequency_rows = _KERNEL.build_card_frequency_rows
build_card_pair_frequency_rows = _KERNEL.build_card_pair_frequency_rows
parse_replay = _KERNEL.parse_replay
stable_daily_replay_selection = _KERNEL.stable_daily_replay_selection
write_daily_outputs = _KERNEL.write_daily_outputs
build_popular_decks = _KERNEL.build_popular_decks


REPLAY_PATH = Path("tests/fixtures/replay/episode-84817357-replay.json")


def test_replay_decision_dataset_from_real_replay() -> None:
    if not REPLAY_PATH.exists():
        return
    dataset = ReplayDecisionDataset.from_paths([REPLAY_PATH], max_samples=20)
    assert len(dataset) == 20
    assert dataset.summary.parser_errors == []
    first = dataset[0]
    assert first.observation.get("select") is not None
    assert first.option_count == len(first.parsed.select_options)
    assert len(first.parsed.card_instances) >= 0
    assert first.memory_after.max_recent_events == 32
    assert first.transition_observation_after is not None
    assert first.transition_parsed_after is not None
    assert first.transition_memory_after is not None
    assert first.action_step_index == first.step_index + 1
    batch = collate_replay_decisions([dataset[0], dataset[1]])
    assert len(batch["observations"]) == 2
    assert len(batch["metadata"]) == 2
    assert {"episode_key", "source_path", "source_date", "done"} <= batch["metadata"][0].keys()


def test_replay_decision_dataset_supports_controlled_agent_filter() -> None:
    if not REPLAY_PATH.exists():
        return
    dataset = ReplayDecisionDataset.from_paths([REPLAY_PATH], controlled_agents={1}, max_samples=10)
    assert len(dataset) == 10
    assert {sample.agent_index for sample in dataset.samples} == {1}


def test_iter_replay_paths_accepts_daily_dataset_json_names(tmp_path: Path) -> None:
    daily = tmp_path / "pokemon-tcg-ai-battle-episodes-2026-07-10"
    daily.mkdir()
    replay = daily / "85109456.json"
    replay.write_text("{}", encoding="utf-8")
    assert iter_replay_paths([daily]) == [replay]


def _minimal_replay(*, episode_id: int | None = 85109456, replay_id: str | None = "replay-id") -> dict:
    info = {"EpisodeId": episode_id} if episode_id is not None else {}
    return {
        "id": replay_id,
        "info": info,
        "steps": [
            [
                {
                    "action": [],
                    "observation": {
                        "current": None,
                        "logs": [],
                        "select": {
                            "type": 0,
                            "context": 0,
                            "minCount": 0,
                            "maxCount": 0,
                            "option": [],
                        },
                    },
                    "reward": 0,
                    "status": "ACTIVE",
                }
            ],
            [
                {
                    "action": [],
                    "observation": {"current": None, "logs": [], "select": None},
                    "reward": 1,
                    "status": "DONE",
                }
            ]
        ],
    }


def test_replay_provenance_is_preserved_in_index_and_collate(tmp_path: Path) -> None:
    daily = tmp_path / "pokemon-tcg-ai-battle-episodes-2026-07-10"
    daily.mkdir()
    replay = daily / "85109456.json"
    replay.write_text(json.dumps(_minimal_replay()), encoding="utf-8")
    dataset = ReplayDecisionDataset.from_paths([daily])
    assert len(dataset) == 1
    sample = dataset[0]
    assert sample.source_date == "2026-07-10"
    assert sample.source_path == str(replay)
    assert sample.step_index == 0
    assert sample.action_step_index == 1
    assert sample.action == []
    assert sample.episode_key == "episode:85109456"

    row = dataset.to_index_rows()[0]
    assert row["source_date"] == "2026-07-10"
    assert row["source_path"] == str(replay)
    assert row["action_step_index"] == 1
    assert row["episode_key"] == "episode:85109456"
    assert row["done"] is True
    metadata = collate_replay_decisions([sample])["metadata"][0]
    assert metadata["source_date"] == "2026-07-10"

    contract = build_replay_decision_contract(sample, final_outcome=1)
    assert contract.schema_version == "replay_decision_contract_v2"
    assert contract.key.decision_step_index == 0
    assert contract.targets.final_outcome == 1
    assert contract.targets.legal_action.chosen_option_indices == ()
    assert contract.match.is_turn_owner.state.name == "UNKNOWN"
    assert contract.hidden_belief is None
    assert contract.hidden_belief_state.name == "NOT_APPLICABLE"
    assert build_replay_decision_contract(
        sample, hidden_belief_enabled=True
    ).hidden_belief_state.name == "UNKNOWN"
    assert metadata["episode_key"] == "episode:85109456"


def test_replay_key_falls_back_to_replay_then_canonical_content(tmp_path: Path) -> None:
    daily = tmp_path / "pokemon-tcg-ai-battle-episodes-2026-07-10"
    daily.mkdir()
    replay_id_path = daily / "with-replay-id.json"
    replay_id_path.write_text(
        json.dumps(_minimal_replay(episode_id=None, replay_id="fallback-replay")),
        encoding="utf-8",
    )
    path_only_path = daily / "path-only.json"
    path_only_path.write_text(
        json.dumps(_minimal_replay(episode_id=None, replay_id=None)),
        encoding="utf-8",
    )
    dataset = ReplayDecisionDataset.from_paths([replay_id_path, path_only_path])
    keys = {sample.episode_key for sample in dataset.samples}
    assert "replay:fallback-replay" in keys
    assert stable_replay_key(_minimal_replay(episode_id=None, replay_id=None)).startswith(
        "content:"
    )
    assert stable_replay_key(_minimal_replay(episode_id=None, replay_id=None)) in keys


def test_bad_replay_is_recorded_without_aborting_other_files(tmp_path: Path) -> None:
    daily = tmp_path / "pokemon-tcg-ai-battle-episodes-2026-07-10"
    daily.mkdir()
    (daily / "bad.json").write_text("{not json", encoding="utf-8")
    (daily / "metadata.json").write_text("{}", encoding="utf-8")
    (daily / "valid.json").write_text(json.dumps(_minimal_replay()), encoding="utf-8")
    dataset = ReplayDecisionDataset.from_paths([daily])
    assert len(dataset) == 1
    assert dataset.summary.replay_count == 1
    assert {error["stage"] for error in dataset.summary.parser_errors} == {
        "replay_load",
        "replay_structure",
    }


def test_streaming_export_writes_compact_decision_references(tmp_path: Path) -> None:
    replay = tmp_path / "episode-85109456-replay.json"
    payload = _minimal_replay()
    payload["rewards"] = [1]
    replay.write_text(json.dumps(payload), encoding="utf-8")
    output_dir = tmp_path / "out"
    summary = export_replay_decisions([replay], output_dir)
    assert summary["decision_sample_count"] == 1
    assert {(output_dir / name).exists() for name in [
        "decisions/decision_references.jsonl.gz",
        "reports/replay_feature_audit.json", "reports/extraction_errors.jsonl",
    ]} == {True}
    import gzip

    with gzip.open(
        output_dir / "decisions/decision_references.jsonl.gz", "rt", encoding="utf-8"
    ) as handle:
        row = json.loads(handle.read().strip())
    assert row["decision_key"] == {
        "replay_key": "episode:85109456",
        "episode_id": 85109456,
        "decision_step_index": 0,
        "action_step_index": 1,
        "player_index": 0,
    }
    assert row["final_outcome"] == 1
    assert row["source"]["source_path"] == str(replay)
    assert row["source"]["raw_content_hash"]
    assert row["parser_contract_version"] == "replay_observation_parser_v2"
    assert not ({"card_instances", "global_snapshot", "memory_summary", "recent_events", "select_options"} & row.keys())

    rebuilt = rebuild_replay_decision_from_reference(row)
    assert rebuilt.decision_key == dataset_key_from_row(row)
    moved = tmp_path / "moved-replay.json"
    moved.write_bytes(replay.read_bytes())
    rebuilt_after_move = rebuild_replay_decision_from_reference(row, source_path=moved)
    assert rebuilt_after_move.decision_key == dataset_key_from_row(row)
    tampered = json.loads(json.dumps(row))
    tampered["source"]["raw_content_hash"] = "0" * 64
    with pytest.raises(ValueError, match="content hash mismatch"):
        rebuild_replay_decision_from_reference(tampered)

    fingerprint_changed = json.loads(json.dumps(payload))
    fingerprint_changed["steps"][0][0]["observation"]["select"]["context"] = 99
    changed_path = tmp_path / "fingerprint-changed.json"
    changed_raw = json.dumps(fingerprint_changed).encode("utf-8")
    changed_path.write_bytes(changed_raw)
    tampered_source = json.loads(json.dumps(row))
    tampered_source["source"]["raw_content_hash"] = hashlib.sha256(changed_raw).hexdigest()
    with pytest.raises(ValueError, match="observation fingerprint mismatch"):
        rebuild_replay_decision_from_reference(tampered_source, source_path=changed_path)


def dataset_key_from_row(row: dict):
    from data.decision_schema import DecisionKey

    return DecisionKey(**row["decision_key"])


def test_missing_episode_ids_do_not_collide_within_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "replays.jsonl"
    rows = [
        _minimal_replay(episode_id=None, replay_id=None),
        _minimal_replay(episode_id=None, replay_id=None),
    ]
    rows[1]["steps"][1][0]["reward"] = -1
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    dataset = ReplayDecisionDataset.from_paths([path])
    assert len(dataset) == 2
    assert len({sample.decision_key for sample in dataset.samples}) == 2
    assert {sample.source_record_line for sample in dataset.samples} == {1, 2}


def test_missing_episode_ids_ignore_zip_member_identity(tmp_path: Path) -> None:
    replay = _minimal_replay(episode_id=None, replay_id=None)
    archive = tmp_path / "replays.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("a.json", json.dumps(replay))
        handle.writestr("nested/b.json", json.dumps(replay, indent=2))
    with zipfile.ZipFile(archive) as handle:
        first_replay = json.loads(handle.read("a.json"))
        second_replay = json.loads(handle.read("nested/b.json"))
    first = stable_replay_key(first_replay, archive, archive_member="a.json")
    second = stable_replay_key(second_replay, archive, archive_member="nested/b.json")
    assert first == second == f"content:{stable_replay_key(replay).split(':', 1)[1]}"


def test_replay_dataset_streams_multiple_zip_members_with_provenance(tmp_path: Path) -> None:
    archive = tmp_path / "replays.zip"
    first = _minimal_replay(episode_id=1, replay_id="first")
    second = _minimal_replay(episode_id=2, replay_id="second")
    third = _minimal_replay(episode_id=3, replay_id="third")
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("2026-07-01-1.json", json.dumps(first))
        handle.writestr("2026-07-02-2.json", json.dumps(second))
        handle.writestr("2026-07-03-3.json", json.dumps(third))
    dataset = ReplayDecisionDataset.from_paths([archive], max_replays=2)
    assert len(dataset) == 2
    assert dataset.summary.replay_count == 2
    assert {sample.source_kind for sample in dataset.samples} == {"ZIP_MEMBER"}
    assert {sample.source_archive_member for sample in dataset.samples} == {
        "2026-07-01-1.json",
        "2026-07-02-2.json",
    }
    assert {sample.source_date for sample in dataset.samples} == {"2026-07-01", "2026-07-02"}
    assert ReplayDecisionDataset.from_paths([archive], max_replays=1).summary.replay_count == 1
    evenly_spaced = ReplayDecisionDataset.from_paths(
        [archive], max_replays=2, archive_member_selection="EVENLY_SPACED"
    )
    assert {sample.source_archive_member for sample in evenly_spaced.samples} == {
        "2026-07-01-1.json",
        "2026-07-03-3.json",
    }


def test_content_replay_key_survives_move_and_changes_with_content(tmp_path: Path) -> None:
    replay = _minimal_replay(episode_id=None, replay_id=None)
    path_a = tmp_path / "a.json"
    path_b = tmp_path / "nested/b.json"
    path_b.parent.mkdir()
    path_a.write_text(json.dumps(replay), encoding="utf-8")
    path_b.write_text(json.dumps(replay), encoding="utf-8")
    assert ReplayDecisionDataset.from_paths([path_a])[0].replay_key == (
        ReplayDecisionDataset.from_paths([path_b])[0].replay_key
    )
    changed = json.loads(json.dumps(replay))
    changed["steps"][1][0]["reward"] = -1
    assert stable_replay_key(changed) != stable_replay_key(replay)


def test_identical_idless_replays_are_deduplicated(tmp_path: Path) -> None:
    replay = _minimal_replay(episode_id=None, replay_id=None)
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text(json.dumps(replay), encoding="utf-8")
    second.write_text(json.dumps(replay, indent=2), encoding="utf-8")
    dataset = ReplayDecisionDataset.from_paths([first, second])
    assert len(dataset) == 1
    assert dataset.summary.replay_count == 1
    assert dataset.summary.duplicate_replay_content_count == 1


@pytest.mark.parametrize("bad_action", [["bad"], [None], [{"index": 0}]])
def test_invalid_action_payload_never_becomes_option_zero(
    tmp_path: Path, bad_action: list[object]
) -> None:
    replay = _minimal_replay()
    replay["steps"][0][0]["observation"]["select"]["option"] = [{"type": 1}]
    replay["steps"][0][0]["observation"]["select"]["minCount"] = 1
    replay["steps"][0][0]["observation"]["select"]["maxCount"] = 1
    replay["steps"][1][0]["action"] = bad_action
    path = tmp_path / "bad-action.json"
    path.write_text(json.dumps(replay), encoding="utf-8")
    dataset = ReplayDecisionDataset.from_paths([path])
    assert len(dataset) == 0
    assert any(error["stage"] == "action_alignment" for error in dataset.summary.parser_errors)


def _replay_with_turn(
    *,
    agent_index: int = 0,
    your_index: int = 0,
    turn: int = 3,
    first_player: int = 0,
    explicit_owner: int | None = None,
) -> dict:
    replay = _minimal_replay()
    if agent_index == 1:
        replay["steps"] = [[{}, replay["steps"][0][0]], [{}, replay["steps"][1][0]]]
    current = {
        "turn": turn,
        "turnActionCount": 0,
        "yourIndex": your_index,
        "firstPlayer": first_player,
        "players": [{}, {}],
    }
    if explicit_owner is not None:
        current["turnOwner"] = explicit_owner
    replay["steps"][0][agent_index]["observation"]["current"] = current
    return replay


@pytest.mark.parametrize(
    ("turn", "first_player", "your_index", "expected_owner", "expected_is_owner"),
    [(3, 0, 0, 0, True), (4, 0, 0, 1, False), (3, 1, 1, 1, True), (4, 1, 1, 0, False)],
)
def test_turn_owner_uses_first_player_formula_and_current_view(
    tmp_path: Path,
    turn: int,
    first_player: int,
    your_index: int,
    expected_owner: int,
    expected_is_owner: bool,
) -> None:
    replay = _replay_with_turn(
        agent_index=your_index,
        your_index=your_index,
        turn=turn,
        first_player=first_player,
    )
    path = tmp_path / "turn-owner.json"
    path.write_text(json.dumps(replay), encoding="utf-8")
    sample = ReplayDecisionDataset.from_paths([path])[0]
    contract = build_replay_decision_contract(sample)
    owner = contract.match.is_turn_owner
    relative = contract.resources.turn_usage.turn_owner_relative
    assert expected_owner == (your_index if expected_is_owner else 1 - your_index)
    assert owner.value is expected_is_owner
    assert relative.value == (0 if expected_is_owner else 1)
    assert owner.state.name == "PRESENT"
    assert owner.inference_source == "INFERRED_TURN_FORMULA"


def test_setup_turn_owner_is_unknown(tmp_path: Path) -> None:
    replay = _replay_with_turn(turn=0, first_player=-1)
    path = tmp_path / "setup.json"
    path.write_text(json.dumps(replay), encoding="utf-8")
    contract = build_replay_decision_contract(ReplayDecisionDataset.from_paths([path])[0])
    assert contract.match.is_turn_owner.state.name == "UNKNOWN"
    assert contract.resources.turn_usage.turn_owner_relative.state.name == "UNKNOWN"


def test_agent_perspective_mismatch_is_recorded_without_sample(tmp_path: Path) -> None:
    replay = _replay_with_turn(agent_index=0, your_index=1)
    path = tmp_path / "perspective-mismatch.json"
    path.write_text(json.dumps(replay), encoding="utf-8")
    dataset = ReplayDecisionDataset.from_paths([path])
    assert len(dataset) == 0
    assert dataset.summary.agent_perspective_mismatch_count == 1
    assert any(error["stage"] == "agent_perspective" for error in dataset.summary.parser_errors)


def test_agent_perspective_match_generates_sample(tmp_path: Path) -> None:
    replay = _replay_with_turn(agent_index=1, your_index=1, first_player=1)
    path = tmp_path / "perspective-match.json"
    path.write_text(json.dumps(replay), encoding="utf-8")
    dataset = ReplayDecisionDataset.from_paths([path])
    assert len(dataset) == 1
    assert dataset.summary.agent_perspective_match_count == 1


def test_explicit_turn_owner_conflict_is_recorded_without_sample(tmp_path: Path) -> None:
    replay = _replay_with_turn(turn=3, first_player=0, explicit_owner=1)
    path = tmp_path / "turn-owner-conflict.json"
    path.write_text(json.dumps(replay), encoding="utf-8")
    dataset = ReplayDecisionDataset.from_paths([path])
    assert len(dataset) == 0
    assert dataset.summary.turn_owner_conflict_count == 1
    assert any(error["stage"] == "turn_owner_conflict" for error in dataset.summary.parser_errors)


def test_full_deck_variant_is_order_insensitive() -> None:
    previous = _KERNEL.MIN_GROUP_GAMES
    _KERNEL.MIN_GROUP_GAMES = 1
    try:
        deck = [1] * 30 + [2] * 30
        rows = [
            {"signature": "fixture", "deck": deck, "won": True, "reward": 1},
            {"signature": "fixture", "deck": list(reversed(deck)), "won": False, "reward": -1},
        ]
        group = build_popular_decks(rows)[0]
        assert group["trainer_variant_count"] == 1
        assert group["representative_count"] == 2
    finally:
        _KERNEL.MIN_GROUP_GAMES = previous


def test_missing_card_metadata_fails_instead_of_building_empty_signatures(monkeypatch) -> None:
    monkeypatch.setattr(_KERNEL, "find_card_data_csv", lambda: None)
    with pytest.raises(FileNotFoundError, match="required.*searched"):
        _KERNEL.load_card_table()


def test_complete_deck_observations_and_presence_statistics(tmp_path: Path) -> None:
    replay = tmp_path / "episode-42-replay.json"
    deck_a = [1] * 4 + [2] * 2 + [3] * 54
    deck_b = [1] + [2] * 4 + [4] * 55
    replay.write_text(
        json.dumps(
            {
                "info": {"EpisodeId": 42, "TeamNames": ["a", "b"]},
                "rewards": [1, -1],
                "steps": [
                    [{"visualize": []}, {"visualize": []}],
                    [{"action": deck_a}, {"action": deck_b}],
                ],
            }
        ),
        encoding="utf-8",
    )
    rows = parse_replay(replay, {"source_submission_id": 7})
    assert [(row["episode_id"], row["seat"]) for row in rows] == [(42, 0), (42, 1)]
    assert len({row["deck_fingerprint"] for row in rows}) == 2
    frequencies = {row["card_id"]: row for row in build_card_frequency_rows(rows)}
    assert frequencies[1]["deck_presence_count"] == 2
    assert frequencies[1]["total_copy_count"] == 5
    assert frequencies[3]["deck_presence_count"] == 1
    pairs = build_card_pair_frequency_rows(rows)
    keys = [(row["card_id_a"], row["card_id_b"]) for row in pairs]
    assert len(keys) == len(set(keys))
    assert all(left < right for left, right in keys)
    assert (1, 2) in keys and (2, 1) not in keys
    assert all(left != right for left, right in keys)


def test_incomplete_deck_does_not_enter_frequency_denominator() -> None:
    valid = {"deck": [1] * 60, "deck_size": 60}
    invalid = {"deck": [2] * 59, "deck_size": 59}
    frequencies = build_card_frequency_rows([valid, invalid])
    assert [(row["card_id"], row["deck_presence_frequency"]) for row in frequencies] == [(1, 1.0)]


def test_daily_statistics_are_partitioned_and_replay_selection_is_stable(tmp_path: Path) -> None:
    paths = [tmp_path / f"{index}.json" for index in range(10)]
    selected_a = stable_daily_replay_selection(paths, "2026-07-15", 3)
    selected_b = stable_daily_replay_selection(list(reversed(paths)), "2026-07-15", 3)
    assert selected_a == selected_b
    deck_rows = [
        {
            "source_date": "2026-07-15", "episode_id": 1, "seat": 0, "team": "a",
            "source_submission_id": None, "winner": "a", "won": True, "reward": 1,
            "deck_size": 60, "deck_fingerprint": "x", "card_counts": {"1": 60},
            "deck": [1] * 60, "signature": "1:60", "archetype": "fixture",
        }
    ]
    summary = write_daily_outputs(tmp_path / "out", "2026-07-15", deck_rows, 1, [])
    assert summary["valid_complete_deck_count"] == 1
    assert (tmp_path / "out/decks/2026-07-15/deck_observations.jsonl").exists()
    assert (tmp_path / "out/statistics/2026-07-15/card_frequency.csv").exists()
    assert (tmp_path / "out/statistics/2026-07-15/card_pair_frequency.csv").exists()


def test_daily_dataset_discovery_supports_namespaced_kaggle_mount(tmp_path: Path) -> None:
    mount = (
        tmp_path
        / "input/datasets/organizations/kaggle/pokemon-tcg-ai-battle-episodes-2026-07-15"
    )
    mount.mkdir(parents=True)
    previous = _KERNEL.KAGGLE_INPUT
    _KERNEL.KAGGLE_INPUT = tmp_path / "input"
    try:
        assert _KERNEL.daily_dataset_dirs() == [("2026-07-15", mount)]
    finally:
        _KERNEL.KAGGLE_INPUT = previous
