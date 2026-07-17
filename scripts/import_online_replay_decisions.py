from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.online_replay_importer import (
    OnlineReplayImportConfig,
    prepare_mounted_daily_replays,
    prepare_online_replays,
    select_daily_dataset_refs,
)
from data.replay_dataset import export_replay_decisions


def main() -> None:
    parser = argparse.ArgumentParser(description="Import online Kaggle PTCG replay observations into decision samples.")
    parser.add_argument("--competition", default="pokemon-tcg-ai-battle")
    parser.add_argument("--submission-id", type=int, action="append", default=[])
    parser.add_argument("--recent-submissions", type=int, default=4)
    parser.add_argument("--submission-page-size", type=int, default=20)
    parser.add_argument("--max-replays", type=int, default=40)
    parser.add_argument("--download-sleep", type=float, default=0.5)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs/replay_extract",
    )
    parser.add_argument(
        "--episodes-index-dir",
        type=Path,
        default=None,
        help="Mounted Kaggle dataset directory, e.g. /kaggle/input/pokemon-tcg-ai-battle-episodes-index.",
    )
    parser.add_argument(
        "--use-daily-manifest",
        action="store_true",
        help="Use episodes-index/manifest.csv to select mounted daily replay datasets instead of episode API refs.",
    )
    parser.add_argument(
        "--daily-dataset-mount-root",
        type=Path,
        default=Path("/kaggle/input"),
        help="Root where daily datasets from manifest are mounted.",
    )
    parser.add_argument(
        "--daily-replay-dir",
        type=Path,
        action="append",
        default=[],
        help="Mounted daily replay dataset directory containing episode JSON files. Can be repeated.",
    )
    parser.add_argument("--max-days", type=int, default=None)
    parser.add_argument(
        "--reserve-recent-days",
        type=int,
        default=0,
        help="Reserve the most recent N days from the mounted index for validation/test instead of training.",
    )
    parser.add_argument("--import-split", choices=["train", "reserved"], default="train")
    parser.add_argument("--include-no-select", action="store_true")
    parser.add_argument("--controlled-agent", type=int, action="append")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--include-private-episodes", action="store_true")
    args = parser.parse_args()

    controlled = set(args.controlled_agent) if args.controlled_agent is not None else None
    daily_dirs = list(args.daily_replay_dir)
    daily_refs = []
    if args.use_daily_manifest:
        if args.episodes_index_dir is None:
            raise SystemExit("--use-daily-manifest requires --episodes-index-dir")
        daily_refs = select_daily_dataset_refs(
            args.episodes_index_dir,
            mount_root=args.daily_dataset_mount_root,
            reserve_recent_days=args.reserve_recent_days,
            import_split=args.import_split,
            max_days=args.max_days,
        )
        daily_dirs.extend([ref.mount_path for ref in daily_refs if ref.mount_path is not None])
    if daily_dirs:
        replay_paths, metadata = prepare_mounted_daily_replays(daily_dirs, output_dir=args.output_dir)
        metadata["daily_dataset_refs"] = [
            {
                "date": ref.date,
                "daily_dataset_slug": ref.daily_dataset_slug,
                "split": ref.split,
                "mount_path": str(ref.mount_path) if ref.mount_path is not None else None,
                "episode_count": ref.episode_count,
                "total_bytes": ref.total_bytes,
            }
            for ref in daily_refs
        ]
        reports_dir = args.output_dir / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        (reports_dir / "online_import_manifest.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    else:
        config = OnlineReplayImportConfig(
            competition=args.competition,
            submission_ids=list(args.submission_id),
            recent_submissions_to_use=args.recent_submissions,
            submission_page_size=args.submission_page_size,
            max_replays=args.max_replays,
            download_sleep_seconds=args.download_sleep,
            include_private_episodes=args.include_private_episodes,
            episodes_index_dir=args.episodes_index_dir,
            reserve_recent_days=args.reserve_recent_days,
            import_split=args.import_split,
            output_dir=args.output_dir,
        )
        replay_paths, metadata = prepare_online_replays(config)
    decision_summary = export_replay_decisions(
        replay_paths,
        args.output_dir,
        include_no_select=args.include_no_select,
        controlled_agents=controlled,
        max_samples=args.max_samples,
    )
    reports_dir = args.output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    summary_path = reports_dir / "extraction_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    summary.update(
        {
            "online_import": metadata,
            "decision_sample_count": decision_summary["decision_sample_count"],
            "steps_per_replay": decision_summary["steps_per_replay"],
            "decisions_per_replay": decision_summary["decisions_per_replay"],
            "max_card_instances": decision_summary["max_card_instances"],
            "max_legal_options": decision_summary["max_legal_options"],
            "observation_parser_error_count": decision_summary["observation_parser_error_count"],
            "decision_export": decision_summary,
        }
    )
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(decision_summary, indent=2, ensure_ascii=False))
    print("wrote", summary_path)


if __name__ == "__main__":
    main()
