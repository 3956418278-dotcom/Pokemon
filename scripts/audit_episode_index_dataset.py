from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.online_replay_importer import (
    assign_time_splits,
    iter_episode_index_files,
    load_episode_refs_from_index,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit a mounted Kaggle PTCG episode-index dataset without downloading replays.")
    parser.add_argument(
        "episodes_index_dir",
        type=Path,
        nargs="?",
        default=Path("/kaggle/input/pokemon-tcg-ai-battle-episodes-index"),
    )
    parser.add_argument("--reserve-recent-days", type=int, default=3)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    files = iter_episode_index_files(args.episodes_index_dir)
    refs = assign_time_splits(load_episode_refs_from_index(args.episodes_index_dir), args.reserve_recent_days)
    by_split = Counter(ref.split for ref in refs)
    by_state = Counter(ref.state or "" for ref in refs)
    by_type = Counter(ref.episode_type or "" for ref in refs)
    by_file = Counter(ref.source_index_file or "" for ref in refs)
    columns_by_file: dict[str, list[str]] = {}
    for ref in refs:
        if ref.source_index_file and ref.source_index_file not in columns_by_file:
            columns_by_file[ref.source_index_file] = sorted(str(key) for key in ref.raw.keys())
    dated = [ref.created_at for ref in refs if ref.created_at]
    examples_by_split: dict[str, list[int]] = defaultdict(list)
    for ref in refs:
        if len(examples_by_split[ref.split]) < 5:
            examples_by_split[ref.split].append(ref.episode_id)
    report = {
        "episodes_index_dir": str(args.episodes_index_dir),
        "reserve_recent_days": args.reserve_recent_days,
        "index_file_count": len(files),
        "index_files": [str(path) for path in files[:20]],
        "episode_count": len(refs),
        "split_counts": dict(by_split),
        "state_counts": dict(by_state),
        "type_counts": dict(by_type),
        "source_file_episode_counts": dict(by_file),
        "date_min": min(dated) if dated else None,
        "date_max": max(dated) if dated else None,
        "example_episode_ids_by_split": dict(examples_by_split),
        "columns_by_file": columns_by_file,
    }
    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
