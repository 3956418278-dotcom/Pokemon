from __future__ import annotations

import argparse
import filecmp
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
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
    "data/dynamic_card_dataset.py",
    "models/__init__.py",
    "models/static_card_adapter.py",
    "models/dynamic_instance_encoder.py",
    "models/card_instance_fusion.py",
    "models/dynamic_card_auxiliary.py",
    "models/board_tokenizer.py",
    "models/board_transformer.py",
    "models/dynamic_state_encoder.py",
    "training/__init__.py",
    "training/train_dynamic_card_fusion.py",
    "configs/dynamic_card_fusion.json",
    "configs/dynamic_card_fusion_smoke.json",
    "scripts/audit_replay_features.py",
    "scripts/benchmark_dynamic_card_fusion.py",
    "scripts/benchmark_dynamic_state.py",
    "scripts/import_online_replay_decisions.py",
    "tests/conftest.py",
    "tests/test_observation_parser.py",
    "tests/test_replay_dataset.py",
    "tests/test_dynamic_card_dataset.py",
    "tests/test_dynamic_card_models.py",
    "tests/test_dynamic_training.py",
]

PUBLICATION_LINEAGE_FILES = [
    "kaggle_dynamic_code_dataset/dataset-metadata.json",
    "kaggle_dynamic_training/kernel-metadata.json",
    "kaggle_dynamic_training/run_dynamic_card_training.py",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_output(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def _source_manifest() -> dict[str, object]:
    dirty_lines = _git_output("status", "--porcelain=v1", "--untracked-files=all").splitlines()
    return {
        "schema_version": "dynamic_code_source_manifest_v2",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_output("rev-parse", "HEAD"),
        "git_dirty": bool(dirty_lines),
        "dirty_paths": [line[3:] if len(line) > 3 else line for line in dirty_lines],
        "files": {
            relative: {
                "sha256": _sha256(ROOT / relative),
                "size_bytes": (ROOT / relative).stat().st_size,
            }
            for relative in CANONICAL_FILES
        },
        "publication_lineage": {
            "files": {
                relative: {
                    "sha256": _sha256(ROOT / relative),
                    "size_bytes": (ROOT / relative).stat().st_size,
                }
                for relative in PUBLICATION_LINEAGE_FILES
            }
        },
    }


def sync() -> None:
    for directory in ("data", "models", "training", "configs", "scripts", "tests"):
        shutil.rmtree(TARGET / directory, ignore_errors=True)
    for relative in CANONICAL_FILES:
        source = ROOT / relative
        destination = TARGET / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        print(f"synced {relative}")
    (TARGET / "source_manifest.json").write_text(
        json.dumps(_source_manifest(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print("wrote source_manifest.json")


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
    manifest_path = TARGET / "source_manifest.json"
    if not manifest_path.exists():
        print("dynamic code dataset is missing source_manifest.json")
        return False
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    recorded_files = manifest.get("files", {})
    hash_mismatches = [
        relative
        for relative in CANONICAL_FILES
        if recorded_files.get(relative, {}).get("sha256") != _sha256(ROOT / relative)
    ]
    if hash_mismatches:
        print("dynamic code source manifest has stale hashes:")
        for relative in hash_mismatches:
            print(f"  - {relative}")
        return False
    recorded_lineage = manifest.get("publication_lineage", {}).get("files", {})
    lineage_mismatches = [
        relative
        for relative in PUBLICATION_LINEAGE_FILES
        if recorded_lineage.get(relative, {}).get("sha256") != _sha256(ROOT / relative)
    ]
    if lineage_mismatches:
        print("dynamic code publication lineage has stale hashes:")
        for relative in lineage_mismatches:
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
