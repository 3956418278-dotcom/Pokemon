from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.detail_replay_join import (  # noqa: E402
    assign_episode_splits,
    join_replay_samples,
    write_detail_join_audit,
)
from data.replay_dataset import ReplayDecisionDataset  # noqa: E402
from data.static_detail_catalog import StaticDetailCatalog  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Join replay decision transitions to the canonical static detail catalog."
    )
    parser.add_argument("paths", nargs="+", type=Path, help="Replay JSON/JSONL files or directories.")
    parser.add_argument(
        "--static-artifacts",
        type=Path,
        default=ROOT / "static_card/artifacts/card_data",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs/detail_join_audit",
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--max-replays",
        type=int,
        default=None,
        help="Maximum number of complete replays to read, including ZIP members.",
    )
    parser.add_argument(
        "--archive-member-selection",
        choices=("FIRST", "EVENLY_SPACED"),
        default="EVENLY_SPACED",
        help="How a replay limit samples members inside a ZIP archive.",
    )
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument(
        "--held-out-date",
        type=str,
        default=None,
        help="Optional YYYY-MM-DD boundary; episodes on/after it are test data.",
    )
    args = parser.parse_args()

    catalog = StaticDetailCatalog.from_artifact_dir(args.static_artifacts)
    catalog.write(args.static_artifacts / "detail_catalog.json")
    dataset = ReplayDecisionDataset.from_paths(
        args.paths,
        max_samples=args.max_samples,
        max_replays=args.max_replays,
        archive_member_selection=args.archive_member_selection,
    )
    splits = assign_episode_splits(
        dataset.samples,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
        held_out_date=args.held_out_date,
    )
    joined = join_replay_samples(dataset.samples, catalog, episode_splits=splits)
    summary = write_detail_join_audit(args.output_dir, catalog, joined)
    summary["replay_parser_error_count"] = len(dataset.summary.parser_errors)
    summary["unpaired_pending_count"] = dataset.summary.unpaired_pending
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"wrote {args.output_dir}")


if __name__ == "__main__":
    main()
