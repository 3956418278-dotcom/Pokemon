from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.dynamic_card_dataset import AttackCostCatalog, collate_dynamic_card_samples
from data.replay_dataset import ReplayDecisionDataset
from models.static_card_adapter import StaticCardAdapter
from training.train_dynamic_card_fusion import cpu_benchmark
from training.train_dynamic_card_fusion import require_static_adapter_ready


def main() -> None:
    static_adapter = StaticCardAdapter()
    require_static_adapter_ready(static_adapter)
    parser = argparse.ArgumentParser(description="Benchmark a trained dynamic CardInstanceFusion checkpoint on CPU.")
    parser.add_argument("replay_paths", type=Path, nargs="+")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--static-artifact-dir", type=Path, required=True)
    parser.add_argument("--card-records", type=Path, required=True)
    parser.add_argument("--detail-metadata", type=Path, required=True)
    parser.add_argument("--decision-samples", type=int, default=8)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/benchmarks/dynamic_card_fusion.json",
    )
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    catalog = AttackCostCatalog.from_files(
        args.card_records,
        args.detail_metadata,
        args.static_artifact_dir / "card_id_to_index.json",
    )
    dataset = ReplayDecisionDataset.from_paths(args.replay_paths, max_samples=args.decision_samples)
    if not dataset.samples:
        raise RuntimeError("no replay decisions were available for the benchmark")
    batch = collate_dynamic_card_samples(
        dataset.samples,
        catalog,
        int(config["model"]["max_details"]),
    )
    result = cpu_benchmark(args.static_artifact_dir, args.checkpoint, batch, config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
