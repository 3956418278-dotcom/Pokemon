from __future__ import annotations

import json
from pathlib import Path

from data.online_replay_importer import (
    OnlineReplayImportConfig,
    ReplayEpisodeRef,
    import_mounted_daily_replay_dataset,
    import_online_replay_dataset,
    select_daily_dataset_refs,
    select_episode_refs,
)


class FakeReplayClient:
    def __init__(self, replay_bytes: bytes) -> None:
        self.replay_bytes = replay_bytes
        self.downloaded: list[int] = []

    def list_recent_submission_ids(self, competition: str, page_size: int, limit: int) -> list[int]:
        return [101, 102][:limit]

    def list_episodes(self, competition: str, submission_id: int) -> list[ReplayEpisodeRef]:
        return [
            ReplayEpisodeRef(
                episode_id=9000 + submission_id,
                source_submission_id=submission_id,
                state="COMPLETE",
                episode_type="PUBLIC",
                team_names=["a", "b"],
            )
        ]

    def download_replay(self, competition: str, episode_id: int) -> bytes:
        self.downloaded.append(episode_id)
        return self.replay_bytes


def test_online_replay_importer_uses_client_and_builds_dataset(tmp_path: Path) -> None:
    replay_path = Path("tests/fixtures/replay/episode-84817357-replay.json")
    if replay_path.exists():
        replay_bytes = replay_path.read_bytes()
    else:
        replay_bytes = json.dumps(
            {
                "id": "fake",
                "info": {"EpisodeId": 1},
                "steps": [
                    [
                        {
                                "action": [],
                            "observation": {
                                "current": None,
                                "logs": [],
                                "select": {"type": 9, "context": 41, "minCount": 1, "maxCount": 1, "option": [{"type": 1}]},
                            },
                            "reward": 0,
                            "status": "ACTIVE",
                        }
                    ]
                ],
            }
        ).encode("utf-8")
    client = FakeReplayClient(replay_bytes)
    config = OnlineReplayImportConfig(max_replays=1, output_dir=tmp_path, download_sleep_seconds=0)
    dataset, metadata = import_online_replay_dataset(config, client=client, max_samples=5)
    assert client.downloaded
    assert metadata["downloaded_replay_paths"]
    assert len(dataset) > 0
    assert dataset.summary.parser_errors == []
    assert (tmp_path / "reports/online_import_manifest.json").exists()


def test_mounted_episode_index_reserves_recent_days(tmp_path: Path) -> None:
    index_dir = tmp_path / "pokemon-tcg-ai-battle-episodes-index"
    index_dir.mkdir()
    (index_dir / "episodes.csv").write_text(
        "\n".join(
            [
                "episode_id,submission_id,state,type,created_at",
                "1001,77,COMPLETE,PUBLIC,2026-07-01T00:00:00Z",
                "1002,77,COMPLETE,PUBLIC,2026-07-03T00:00:00Z",
                "1003,77,COMPLETE,PUBLIC,2026-07-05T00:00:00Z",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    train_config = OnlineReplayImportConfig(
        episodes_index_dir=index_dir,
        reserve_recent_days=2,
        import_split="train",
        max_replays=10,
    )
    reserved_config = OnlineReplayImportConfig(
        episodes_index_dir=index_dir,
        reserve_recent_days=2,
        import_split="reserved",
        max_replays=10,
    )
    train_refs = select_episode_refs(FakeReplayClient(b"{}"), train_config)
    reserved_refs = select_episode_refs(FakeReplayClient(b"{}"), reserved_config)
    assert [ref.episode_id for ref in train_refs] == [1001, 1002]
    assert [ref.episode_id for ref in reserved_refs] == [1003]


def test_manifest_selects_mounted_daily_dataset_refs(tmp_path: Path) -> None:
    index_dir = tmp_path / "pokemon-tcg-ai-battle-episodes-index"
    mount_root = tmp_path / "input"
    index_dir.mkdir()
    (index_dir / "manifest.csv").write_text(
        "\n".join(
            [
                "date,daily_dataset_slug,daily_dataset_url,episode_count,total_bytes,top_avg_score,median_avg_score",
                "2026-07-07,pokemon-tcg-ai-battle-episodes-2026-07-07,https://example/7,10,100,1,1",
                "2026-07-10,pokemon-tcg-ai-battle-episodes-2026-07-10,https://example/10,20,200,1,1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    train_refs = select_daily_dataset_refs(index_dir, mount_root=mount_root, reserve_recent_days=3, import_split="train")
    reserved_refs = select_daily_dataset_refs(index_dir, mount_root=mount_root, reserve_recent_days=3, import_split="reserved")
    assert [ref.daily_dataset_slug for ref in train_refs] == ["pokemon-tcg-ai-battle-episodes-2026-07-07"]
    assert [ref.daily_dataset_slug for ref in reserved_refs] == ["pokemon-tcg-ai-battle-episodes-2026-07-10"]
    assert train_refs[0].mount_path == mount_root / "pokemon-tcg-ai-battle-episodes-2026-07-07"


def test_import_mounted_daily_replay_dataset_accepts_plain_json_names(tmp_path: Path) -> None:
    daily_dir = tmp_path / "pokemon-tcg-ai-battle-episodes-2026-07-10"
    daily_dir.mkdir()
    (daily_dir / "85109456.json").write_text(
        json.dumps(
            {
                "id": "plain-json",
                "info": {"EpisodeId": 85109456},
                "steps": [
                    [
                        {
                            "action": [],
                            "observation": {"current": None, "logs": [], "select": None},
                            "reward": 0,
                            "status": "ACTIVE",
                        }
                    ],
                    [
                        {
                            "action": [],
                            "observation": {
                                "current": None,
                                "logs": [],
                                "select": {"type": 9, "context": 41, "minCount": 1, "maxCount": 1, "option": [{"type": 1}]},
                            },
                            "reward": 0,
                            "status": "ACTIVE",
                        }
                    ],
                    [
                        {
                            "action": [0],
                            "observation": {
                                "current": None,
                                "logs": [],
                                "select": None,
                            },
                            "reward": 0,
                            "status": "ACTIVE",
                        }
                    ],
                ],
            }
        ),
        encoding="utf-8",
    )
    dataset, metadata = import_mounted_daily_replay_dataset([daily_dir], output_dir=tmp_path / "out", max_samples=5)
    assert len(dataset) == 1
    assert dataset.samples[0].episode_id == 85109456
    assert metadata["source"] == "mounted_daily_replay_dirs"
    assert metadata["replay_paths"][0].endswith("85109456.json")
