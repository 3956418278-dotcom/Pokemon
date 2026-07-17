from __future__ import annotations

import json
import importlib.util
from pathlib import Path

from data.replay_dataset import (
    ReplayDecisionDataset,
    build_replay_decision_contract,
    collate_replay_decisions,
    export_replay_decisions,
    iter_replay_paths,
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


REPLAY_PATH = Path("data_from_submission/replays/episode-84817357-replay.json")


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
    assert contract.schema_version == "replay_decision_contract_v1"
    assert contract.key.decision_step_index == 0
    assert contract.targets.final_outcome == 1
    assert contract.targets.legal_action.chosen_option_indices == ()
    assert contract.match.is_turn_owner.state.name == "UNKNOWN"
    assert contract.hidden_belief is None
    assert contract.hidden_belief_state.name == "NOT_APPLICABLE"
    assert metadata["episode_key"] == "episode:85109456"


def test_episode_key_falls_back_to_replay_then_source_path(tmp_path: Path) -> None:
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
    assert f"path:{path_only_path}" in keys


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
        "episode_id": 85109456,
        "decision_step_index": 0,
        "action_step_index": 1,
        "player_index": 0,
    }
    assert row["final_outcome"] == 1
    assert row["source"]["replay_path"] == str(replay)
    assert not ({"card_instances", "global_snapshot", "memory_summary", "recent_events", "select_options"} & row.keys())


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
