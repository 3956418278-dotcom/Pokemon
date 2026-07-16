from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.dynamic_card_dataset import AttackCostCatalog  # noqa: E402
from data.replay_dataset import ReplayDecisionDataset  # noqa: E402
from data.replay_feature_audit import build_report, load_known_static_ids  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit model-facing fields across replay decision points.")
    parser.add_argument("paths", nargs="+", type=Path, help="Replay JSON/JSONL files or directories.")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--static-id-map", type=Path, required=True)
    parser.add_argument("--card-records", type=Path, required=True)
    parser.add_argument("--detail-metadata", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/replay_extract/replay_feature_audit.json",
    )
    args = parser.parse_args()

    known_ids = load_known_static_ids(args.static_id_map)
    catalog = None
    if args.card_records.exists() and args.detail_metadata.exists() and args.static_id_map.exists():
        catalog = AttackCostCatalog.from_files(args.card_records, args.detail_metadata, args.static_id_map)
    dataset = ReplayDecisionDataset.from_paths(args.paths, max_samples=args.max_samples)
    report = build_report(dataset, known_ids, catalog)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print("wrote", args.output)


if __name__ == "__main__":
    main()
