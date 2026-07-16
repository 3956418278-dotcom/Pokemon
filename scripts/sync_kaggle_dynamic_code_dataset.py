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
TARGET_RELATIVE = Path("kaggle/datasets/dynamic_code")
TARGET = ROOT / TARGET_RELATIVE
REPORT_DIR = ROOT / "outputs/dynamic_code_dataset"

SOURCE_DIRECTORIES = ("data", "models", "training", "scripts", "configs", "tests")


def _canonical_files() -> list[str]:
    files = []
    for directory in SOURCE_DIRECTORIES:
        for path in (ROOT / directory).rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts and path.suffix in {".py", ".json"}:
                files.append(path.relative_to(ROOT).as_posix())
    return sorted(files)


CANONICAL_FILES = _canonical_files()

PUBLICATION_LINEAGE_FILES = [
    "kaggle/datasets/dynamic_code/dataset-metadata.json",
    "kaggle/kernels/dynamic_training/kernel-metadata.json",
    "kaggle/kernels/dynamic_training/run_dynamic_card_training.py",
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
    return result.stdout.rstrip() if result.returncode == 0 else "unavailable"


def _source_manifest() -> dict[str, object]:
    dirty_lines = _git_output("status", "--porcelain=v1", "--untracked-files=all").splitlines()
    dirty_paths = [line[3:] if len(line) > 3 else line for line in dirty_lines]
    existing_dirty_paths = [path for path in dirty_paths if (ROOT / path).exists()]
    return {
        "schema_version": "dynamic_code_source_manifest_v2",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_output("rev-parse", "HEAD"),
        "git_dirty": bool(dirty_lines),
        "dirty_paths": existing_dirty_paths,
        "deleted_dirty_path_count": len(dirty_paths) - len(existing_dirty_paths),
        "source_root": ".",
        "target_root": TARGET_RELATIVE.as_posix(),
        "files": {
            relative: {
                "source_path": relative,
                "target_path": (TARGET_RELATIVE / relative).as_posix(),
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
    TARGET.mkdir(parents=True, exist_ok=True)
    for directory in SOURCE_DIRECTORIES:
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
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "last_sync.json").write_text(
        json.dumps(
            {"target_root": TARGET_RELATIVE.as_posix(), "file_count": len(CANONICAL_FILES)},
            indent=2,
        )
        + "\n",
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
    if manifest.get("source_root") != "." or manifest.get("target_root") != TARGET_RELATIVE.as_posix():
        print("dynamic code source manifest records incorrect source or target roots")
        return False
    recorded_files = manifest.get("files", {})
    hash_mismatches = [
        relative
        for relative in CANONICAL_FILES
        if (
            recorded_files.get(relative, {}).get("source_path") != relative
            or recorded_files.get(relative, {}).get("target_path")
            != (TARGET_RELATIVE / relative).as_posix()
            or recorded_files.get(relative, {}).get("sha256") != _sha256(ROOT / relative)
        )
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
