from __future__ import annotations

import argparse
import filecmp
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "kaggle_dynamic_code_dataset"

CANONICAL_FILES = [
    "data/__init__.py",
    "data/state_schema.py",
    "data/observation_parser.py",
    "data/game_memory.py",
    "data/replay_dataset.py",
    "data/online_replay_importer.py",
    "models/__init__.py",
    "models/static_card_adapter.py",
    "models/static_detail_aggregator.py",
    "models/dynamic_instance_encoder.py",
    "models/card_instance_fusion.py",
    "models/board_tokenizer.py",
    "models/board_transformer.py",
    "models/dynamic_state_encoder.py",
    "scripts/audit_replay_features.py",
    "scripts/benchmark_dynamic_state.py",
    "scripts/import_online_replay_decisions.py",
]


def sync() -> None:
    for directory in ("data", "models", "scripts"):
        shutil.rmtree(TARGET / directory, ignore_errors=True)
    for relative in CANONICAL_FILES:
        source = ROOT / relative
        destination = TARGET / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        print(f"synced {relative}")


def check() -> bool:
    missing_or_changed = []
    for relative in CANONICAL_FILES:
        source = ROOT / relative
        destination = TARGET / relative
        if not destination.exists() or not filecmp.cmp(source, destination, shallow=False):
            missing_or_changed.append(relative)
    if missing_or_changed:
        print("dynamic code dataset needs sync:")
        for relative in missing_or_changed:
            print(f"  - {relative}")
        return False
    print("dynamic code dataset is synchronized")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the Kaggle dynamic-code Dataset from canonical source files.")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        raise SystemExit(0 if check() else 1)
    sync()


if __name__ == "__main__":
    main()
