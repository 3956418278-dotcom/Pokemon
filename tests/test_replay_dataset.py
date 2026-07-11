from __future__ import annotations

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
