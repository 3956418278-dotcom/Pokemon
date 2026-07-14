from __future__ import annotations

import json
from pathlib import Path

from data.replay_dataset import ReplayDecisionDataset, collate_replay_decisions, iter_replay_paths


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
    assert first.memory_after.max_recent_events == 16
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
                    "action": [0],
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
    assert sample.episode_key == "episode:85109456"

    row = dataset.to_index_rows()[0]
    assert row["source_date"] == "2026-07-10"
    assert row["source_path"] == str(replay)
    assert row["episode_key"] == "episode:85109456"
    assert row["done"] is True
    metadata = collate_replay_decisions([sample])["metadata"][0]
    assert metadata["source_date"] == "2026-07-10"
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
