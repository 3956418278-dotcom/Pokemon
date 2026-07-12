from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.replay_dataset import ReplayDecisionDataset


def percentile(values: list[int], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * q)))
    return float(ordered[index])


def distribution(values: list[int]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "mean": float(mean(values)) if values else 0.0,
        "p50": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "p99": percentile(values, 0.99),
        "max": max(values) if values else 0,
    }


def load_known_static_ids(path: Path) -> set[int]:
    if not path.exists():
        return set()
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "cards" in raw:
        return {int(card["card_id"]) for card in raw["cards"] if card.get("card_id") is not None}
    if isinstance(raw, dict):
        return {int(card_id) for card_id in raw}
    return set()


def build_report(dataset: ReplayDecisionDataset, known_static_ids: set[int]) -> dict[str, Any]:
    values: dict[str, list[int]] = {
        "instances": [],
        "visible_instances": [],
        "hidden_instances": [],
        "events": [],
        "recent_events": [],
        "options": [],
    }
    counters: dict[str, Counter] = {
        "zone": Counter(),
        "event_type": Counter(),
        "option_type": Counter(),
        "select_type": Counter(),
        "select_context": Counter(),
    }
    visible_card_ids: set[int] = set()
    static_lookup_total = 0
    static_lookup_known = 0
    anonymous_instances = 0

    for sample in dataset.samples:
        parsed = sample.parsed
        instances = parsed.card_instances
        values["instances"].append(len(instances))
        values["visible_instances"].append(sum(item.is_visible for item in instances))
        values["hidden_instances"].append(sum(not item.is_visible for item in instances))
        values["events"].append(len(parsed.events))
        values["recent_events"].append(len(sample.memory_after.recent_events))
        values["options"].append(len(parsed.select_options))
        counters["select_type"].update([sample.select_type])
        counters["select_context"].update([sample.select_context])

        for instance in instances:
            counters["zone"].update([instance.zone])
            if instance.card_id is None:
                anonymous_instances += 1
                continue
            if instance.is_visible:
                visible_card_ids.add(int(instance.card_id))
                static_lookup_total += 1
                if not known_static_ids or int(instance.card_id) in known_static_ids:
                    static_lookup_known += 1
        for event in parsed.events:
            counters["event_type"].update([event.event_type])
        for option in parsed.select_options:
            counters["option_type"].update([option.get("type", -1)])

    return {
        "replay_count": dataset.summary.replay_count,
        "decision_sample_count": len(dataset),
        "skipped_no_select": dataset.summary.skipped_no_select,
        "parser_error_count": len(dataset.summary.parser_errors),
        "parser_errors": dataset.summary.parser_errors[:20],
        "unique_visible_card_ids": len(visible_card_ids),
        "anonymous_instance_count": anonymous_instances,
        "static_lookup": {
            "checked": bool(known_static_ids),
            "known": static_lookup_known,
            "total_visible": static_lookup_total,
            "coverage": static_lookup_known / static_lookup_total if static_lookup_total else 1.0,
        },
        "distributions": {name: distribution(rows) for name, rows in values.items()},
        "counters": {name: rows.most_common(50) for name, rows in counters.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit model-facing fields across replay decision points.")
    parser.add_argument("paths", nargs="+", type=Path, help="Replay JSON/JSONL files or directories.")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--static-id-map",
        type=Path,
        default=Path("outputs/card_pretrain/artifacts/card_id_to_index.json"),
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/replay_feature_audit.json"))
    args = parser.parse_args()

    dataset = ReplayDecisionDataset.from_paths(args.paths, max_samples=args.max_samples)
    report = build_report(dataset, load_known_static_ids(args.static_id_map))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print("wrote", args.output)


if __name__ == "__main__":
    main()
