from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import traceback
import zipfile
from pathlib import Path


INPUT_ROOT = Path("/kaggle/input")
WORKING_ROOT = Path("/kaggle/working")
OUTPUT_ROOT = WORKING_ROOT / "outputs"
CONFIG_FILENAME = "dynamic_card_fusion_smoke.json"
RUNNER_LINEAGE_PATH = "kaggle_dynamic_training/run_dynamic_card_training.py"
CODE_ARCHIVE_DIRECTORIES = ("data", "models", "training", "configs", "scripts", "tests")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _is_code_root(candidate: Path) -> bool:
    return (candidate / "training" / "train_dynamic_card_fusion.py").exists()


def _extract_code_archives(dataset_root: Path) -> Path | None:
    """Materialize directory-mode Dataset archives when Kaggle mounts them as zip files."""
    archives = {
        directory: dataset_root / f"{directory}.zip"
        for directory in CODE_ARCHIVE_DIRECTORIES
    }
    if not archives["training"].exists():
        return None
    target = WORKING_ROOT / "dynamic_card_source"
    shutil.rmtree(target, ignore_errors=True)
    target.mkdir(parents=True, exist_ok=True)
    for directory, archive in archives.items():
        if not archive.exists():
            raise FileNotFoundError(f"dynamic code Dataset is missing {archive.name}")
        destination = target / directory
        destination.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as handle:
            handle.extractall(destination)
    manifest = dataset_root / "source_manifest.json"
    if not manifest.exists():
        raise FileNotFoundError("dynamic code Dataset is missing source_manifest.json")
    shutil.copy2(manifest, target / manifest.name)
    return target


def _find_code_root() -> Path:
    dataset_root = INPUT_ROOT / "ptcg-dynamic-code-dataset"
    for candidate in (dataset_root, Path(__file__).resolve().parents[1]):
        if _is_code_root(candidate):
            return candidate
    extracted = _extract_code_archives(dataset_root) if dataset_root.exists() else None
    if extracted is not None and _is_code_root(extracted):
        return extracted
    for candidate in INPUT_ROOT.rglob("source_manifest.json"):
        extracted = _extract_code_archives(candidate.parent)
        if extracted is not None and _is_code_root(extracted):
            return extracted
    for path in INPUT_ROOT.rglob("train_dynamic_card_fusion.py"):
        candidate = path.parent.parent
        if _is_code_root(candidate):
            return candidate
    raise FileNotFoundError("could not locate the synchronized dynamic training source")


def _find_static_artifact_dir() -> Path:
    static_root = INPUT_ROOT / "ptcg-card-pretrain"
    if static_root.exists():
        return static_root
    return INPUT_ROOT


def _find_card_records() -> Path:
    static_root = INPUT_ROOT / "ptcg-card-pretrain"
    search_root = static_root if static_root.exists() else INPUT_ROOT
    paths = sorted(search_root.rglob("card_records.json"))
    if paths:
        return paths[0]
    return search_root / "card_records.json"


def _find_detail_metadata(static_artifact_dir: Path) -> Path:
    direct = static_artifact_dir / "card_detail_metadata.json"
    if direct.exists():
        return direct
    return static_artifact_dir / "card_detail_metadata.json"


def _daily_replay_dir(date: str) -> Path:
    slug = f"pokemon-tcg-ai-battle-episodes-{date}"
    candidates = (
        INPUT_ROOT / slug,
        INPUT_ROOT / "datasets" / "kaggle" / slug,
    )
    for path in candidates:
        if path.is_dir():
            return path
    matches = sorted(path for path in INPUT_ROOT.rglob(slug) if path.is_dir())
    if matches:
        return matches[0]
    checked = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"missing mounted replay Dataset for {date}; checked: {checked}")


def _write_failure(stage: str, exc: BaseException) -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    wrapper_error = {
        "wrapper_completed_stage": stage,
        "error_type": type(exc).__name__,
        "error": str(exc),
        "traceback": traceback.format_exc(),
    }
    summary_path = OUTPUT_ROOT / "run_summary.json"
    if summary_path.exists():
        try:
            existing = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            existing = None
        if isinstance(existing, dict) and existing.get("success") is False:
            existing["kernel_wrapper_error"] = wrapper_error
            _write_json(summary_path, existing)
            return
    payload = {
        "success": False,
        "completed_stage": stage,
        **{key: value for key, value in wrapper_error.items() if key != "wrapper_completed_stage"},
    }
    _write_json(summary_path, payload)


def _record_publication_lineage(code_root: Path) -> None:
    source_manifest_path = code_root / "source_manifest.json"
    if not source_manifest_path.exists():
        raise FileNotFoundError("dynamic code Dataset is missing source_manifest.json")
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    lineage_files = source_manifest.get("publication_lineage", {}).get("files", {})
    expected_runner_hash = lineage_files.get(RUNNER_LINEAGE_PATH, {}).get("sha256")
    expected_kernel_metadata = lineage_files.get("kaggle_dynamic_training/kernel-metadata.json")
    expected_code_dataset_metadata = lineage_files.get(
        "kaggle_dynamic_code_dataset/dataset-metadata.json"
    )
    actual_runner_hash = _sha256(Path(__file__).resolve())
    payload = {
        "schema_version": "dynamic_kernel_publication_lineage_v1",
        "source_manifest": str(source_manifest_path),
        "source_manifest_sha256": _sha256(source_manifest_path),
        "source_manifest_schema_version": source_manifest.get("schema_version"),
        "source_git_commit": source_manifest.get("git_commit"),
        "source_git_dirty": source_manifest.get("git_dirty"),
        "runner_path": RUNNER_LINEAGE_PATH,
        "runner_source_sha256": expected_runner_hash,
        "runner_runtime_path": str(Path(__file__).resolve()),
        "runner_runtime_sha256": actual_runner_hash,
        "kernel_metadata": expected_kernel_metadata,
        "code_dataset_metadata": expected_code_dataset_metadata,
    }
    _write_json(OUTPUT_ROOT / "metadata" / "kernel_wrapper_lineage.json", payload)
    if (
        source_manifest.get("schema_version") != "dynamic_code_source_manifest_v2"
        or not expected_runner_hash
        or actual_runner_hash != expected_runner_hash
        or not expected_kernel_metadata
        or not expected_code_dataset_metadata
    ):
        raise RuntimeError("dynamic code Dataset has incomplete Kernel publication lineage metadata")


def main() -> None:
    stage = "environment_discovery"
    try:
        code_root = _find_code_root()
        stage = "publication_lineage"
        _record_publication_lineage(code_root)
        sys.path.insert(0, str(code_root))
        config_path = code_root / "configs" / CONFIG_FILENAME
        config = json.loads(config_path.read_text(encoding="utf-8"))
        static_artifact_dir = _find_static_artifact_dir()
        card_records_path = _find_card_records()
        detail_metadata_path = _find_detail_metadata(static_artifact_dir)

        stage = "automated_tests"
        test_paths = [
            code_root / "tests" / "test_observation_parser.py",
            code_root / "tests" / "test_replay_dataset.py",
            code_root / "tests" / "test_dynamic_card_dataset.py",
            code_root / "tests" / "test_dynamic_card_models.py",
            code_root / "tests" / "test_dynamic_training.py",
        ]
        subprocess.run(
            [sys.executable, "-m", "pytest", "-q", *[str(path) for path in test_paths]],
            cwd=code_root,
            check=True,
        )

        stage = "dynamic_training"
        command = [
            sys.executable,
            "-m",
            "training.train_dynamic_card_fusion",
            "--config",
            str(config_path),
            "--static-artifact-dir",
            str(static_artifact_dir),
            "--card-records",
            str(card_records_path),
            "--detail-metadata",
            str(detail_metadata_path),
            "--output-dir",
            str(OUTPUT_ROOT),
        ]
        for split_name in ("train", "validation", "test"):
            for date in config["data"][f"{split_name}_dates"]:
                command.extend([f"--{split_name}-replay-dir", str(_daily_replay_dir(date))])
        env = dict(os.environ)
        env["PYTHONPATH"] = str(code_root) + os.pathsep + env.get("PYTHONPATH", "")
        subprocess.run(command, cwd=code_root, env=env, check=True)
    except BaseException as exc:
        _write_failure(stage, exc)
        raise


if __name__ == "__main__":
    main()
