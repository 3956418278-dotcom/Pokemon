from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.replay_dataset import ReplayDecisionDataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("data_from_submission/replay_dataset"))
    parser.add_argument("--include-no-select", action="store_true")
    parser.add_argument("--controlled-agent", type=int, action="append")
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    controlled = set(args.controlled_agent) if args.controlled_agent is not None else None
    dataset = ReplayDecisionDataset.from_paths(
        args.paths,
        include_no_select=args.include_no_select,
        controlled_agents=controlled,
        max_samples=args.max_samples,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "summary.json"
    index_path = args.output_dir / "decision_index.csv"
    summary = {
        "replay_paths": [str(path) for path in dataset.replay_paths],
        "replay_count": dataset.summary.replay_count,
        "sample_count": dataset.summary.sample_count,
        "skipped_no_select": dataset.summary.skipped_no_select,
        "parser_errors": dataset.summary.parser_errors[:20],
        "max_instances": dataset.summary.max_instances,
        "max_options": dataset.summary.max_options,
        "max_events": dataset.summary.max_events,
        "max_token_estimate": dataset.summary.max_token_estimate,
        "include_no_select": args.include_no_select,
        "controlled_agents": sorted(controlled) if controlled is not None else None,
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    rows = dataset.to_index_rows()
    with index_path.open("w", newline="", encoding="utf-8-sig") as handle:
        fieldnames = [
            "index",
            "replay_id",
            "episode_id",
            "step_index",
            "agent_index",
            "action",
            "reward",
            "status",
            "option_count",
            "select_type",
            "select_context",
            "instance_count",
            "event_count",
            "recent_event_count",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            out["action"] = json.dumps(out["action"])
            writer.writerow(out)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("wrote", summary_path)
    print("wrote", index_path)


if __name__ == "__main__":
    main()
