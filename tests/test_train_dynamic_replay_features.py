from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from scripts.train_dynamic_replay_features import choose_daily_dirs


def _args(**overrides):
    values = {
        "daily_replay_dir": [],
        "use_daily_manifest": False,
        "episodes_index_dir": None,
        "daily_dataset_mount_root": Path("/kaggle/input"),
        "reserve_recent_days": 0,
        "import_split": "train",
        "max_days": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_choose_daily_dirs_accepts_direct_mounted_dirs(tmp_path: Path) -> None:
    daily = tmp_path / "pokemon-tcg-ai-battle-episodes-2026-07-07"
    assert choose_daily_dirs(_args(daily_replay_dir=[daily])) == [daily]


def test_choose_daily_dirs_uses_manifest_for_mounted_daily_paths(tmp_path: Path) -> None:
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
    paths = choose_daily_dirs(
        _args(
            use_daily_manifest=True,
            episodes_index_dir=index_dir,
            daily_dataset_mount_root=mount_root,
            reserve_recent_days=3,
            import_split="reserved",
            max_days=1,
        )
    )
    assert paths == [mount_root / "pokemon-tcg-ai-battle-episodes-2026-07-10"]


def test_train_dynamic_replay_features_help_does_not_require_torch() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/train_dynamic_replay_features.py", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--daily-replay-dir" in result.stdout
