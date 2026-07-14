from __future__ import annotations

import argparse
import base64
import hashlib
import json
import textwrap
import zipfile
from io import BytesIO
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILES = [
    "configs/card_pretrain.yaml",
    "data/__init__.py",
    "data/card_dataset.py",
    "data/card_preprocessing.py",
    "models/__init__.py",
    "models/card_encoder.py",
    "models/card_pretrain_heads.py",
    "models/static_card_adapter.py",
    "training/__init__.py",
    "training/evaluate_card_embeddings.py",
    "training/export_card_embeddings.py",
    "training/pretrain_card_encoder.py",
    "training/validate_card_artifacts.py",
    "tests/conftest.py",
    "tests/test_card_preprocessing.py",
    "tests/test_card_dataset.py",
    "tests/test_card_encoder.py",
    "tests/test_pretrain_tasks.py",
    "tests/test_static_card_export.py",
    "tests/test_evaluate_card_embeddings.py",
]


RUNNER_TEMPLATE = '''#!/usr/bin/env python3
from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
import traceback
import zipfile
from io import BytesIO
from pathlib import Path
from xml.etree import ElementTree


ARCHIVE_B64 = """
__ARCHIVE_B64__
"""
ARCHIVE_SHA256 = "__ARCHIVE_SHA256__"
SOURCE_FILE_HASHES = __SOURCE_FILE_HASHES__
RUN_MODE = "static_v2_formal"
WORK_DIR = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path.cwd()
SOURCE_DIR = WORK_DIR / "card_pretrain_v2_formal_src"
FORMAL_ROOT = WORK_DIR / "outputs" / "static_v2_formal"
SUMMARY_PATH = FORMAL_ROOT / "formal_run_summary.json"
CACHE_DIR = WORK_DIR / "artifacts" / "card_data_v2"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\\n", encoding="utf-8")


def run(command: list[str], commands: list[list[str]]) -> None:
    print("+", " ".join(command), flush=True)
    commands.append(command)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SOURCE_DIR) + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run(command, check=True, cwd=WORK_DIR, env=env)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def all_finite(value: object) -> bool:
    if isinstance(value, dict):
        return all(all_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(all_finite(item) for item in value)
    return not isinstance(value, float) or math.isfinite(value)


def read_junit(path: Path) -> dict[str, object]:
    root = ElementTree.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("./testsuite"))
    require(bool(suites), "pytest JUnit report contains no test suites")
    result = {
        key: sum(int(suite.attrib.get(key, 0)) for suite in suites)
        for key in ("tests", "errors", "failures", "skipped")
    }
    result["time_seconds"] = sum(float(suite.attrib.get("time", 0.0)) for suite in suites)
    result["path"] = str(path)
    result["sha256"] = sha256_file(path)
    return result


def load_checkpoint(path: Path) -> dict[str, object]:
    import torch

    try:
        value = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    require(isinstance(value, dict), f"checkpoint is not a mapping: {path}")
    return value


def verify_embedded_source(archive_payload: bytes) -> None:
    require(sha256_bytes(archive_payload) == ARCHIVE_SHA256, "embedded source archive SHA256 mismatch")
    shutil.rmtree(SOURCE_DIR, ignore_errors=True)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(BytesIO(archive_payload)) as archive:
        names = sorted(name for name in archive.namelist() if not name.endswith("/"))
        require(names == sorted(SOURCE_FILE_HASHES), "embedded source archive member list mismatch")
        archive.extractall(SOURCE_DIR)
    for relative, expected_hash in SOURCE_FILE_HASHES.items():
        path = SOURCE_DIR / relative
        require(path.is_file(), f"embedded source file was not extracted: {relative}")
        require(sha256_file(path) == expected_hash, f"embedded source file SHA256 mismatch: {relative}")


def validate_config(config: dict[str, object]) -> None:
    require(config.get("schema_version") == "static_card_training_v2", "formal config has the wrong schema")
    require(config.get("run_mode") == RUN_MODE, "formal config has the wrong run_mode")
    data = config.get("data") or {}
    training = config.get("training") or {}
    early = training.get("early_stopping") or {}
    refit = config.get("production_refit") or {}
    tasks = config.get("tasks") or {}
    require(data.get("split_mode") == "card_id", "formal data split must be card_id")
    require(float(data.get("validation_ratio", 0.0)) > 0.0, "formal validation split is disabled")
    require(float(data.get("test_ratio", 0.0)) > 0.0, "formal test split is disabled")
    require(training.get("device") == "cuda", "formal config must require CUDA")
    require(int(training.get("max_epochs", 0)) == 400, "formal training must allow exactly 400 epochs")
    require(int(training.get("min_epochs", 0)) == 80, "formal training must require at least 80 epochs")
    require(int(training.get("eval_every_epochs", 0)) == 5, "formal validation cadence must be 5 epochs")
    require(
        [int(value) for value in training.get("evaluation_seed_offsets", [])]
        == [1_000_000, 2_000_000, 3_000_000],
        "formal validation must use the three fixed evaluation seed offsets",
    )
    require(bool(early.get("enabled")), "formal early stopping must be enabled")
    require(int(early.get("patience_epochs", 0)) == 40, "formal early-stopping patience must be 40 epochs")
    require(bool(early.get("restore_best_checkpoint")), "formal evaluation must restore the best checkpoint")
    require(bool(refit.get("enabled")), "production refit must be enabled")
    require(refit.get("epoch_source") == "best_validation_epoch", "production refit must use best validation epoch")
    require(bool(tasks.get("identity_relation_masking")), "identity relation masking must be enabled")
    require(bool(tasks.get("leave_one_detail_out")), "leave-one-detail-out ownership must be enabled")
    require(bool(tasks.get("hard_negative_same_type")), "same-type hard negatives must be enabled")
    require(
        Path(str(training.get("checkpoint_dir"))) == Path("outputs/static_v2_formal/checkpoints"),
        "formal checkpoint directory is not isolated",
    )
    require(
        Path(str(training.get("log_dir"))) == Path("outputs/static_v2_formal/logs"),
        "formal log directory is not isolated",
    )
    require(
        Path(str(refit.get("checkpoint_path")))
        == Path("outputs/static_v2_formal/checkpoints/card_encoder_production_refit.pt"),
        "production refit checkpoint path is not isolated",
    )


def main() -> None:
    started = time.time()
    commands: list[list[str]] = []
    device: dict[str, object] = {}
    stage = "fresh_output"
    promotion_ready = False
    source_cleaned = False
    formal_training_started = False
    test_results: dict[str, object] | None = None
    shutil.rmtree(FORMAL_ROOT, ignore_errors=True)
    FORMAL_ROOT.mkdir(parents=True, exist_ok=True)
    try:
        stage = "source_archive"
        archive_payload = base64.b64decode("".join(ARCHIVE_B64.split()))
        verify_embedded_source(archive_payload)

        stage = "formal_config_guard"
        import yaml

        config_path = SOURCE_DIR / "configs" / "card_pretrain.yaml"
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        validate_config(config)
        config_sha256 = sha256_file(config_path)

        stage = "t4_environment_guard"
        import torch

        require(torch.cuda.is_available(), "static v2 formal training requires CUDA")
        name = torch.cuda.get_device_name(0)
        capability = tuple(int(value) for value in torch.cuda.get_device_capability(0))
        device = {
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "cuda_available": True,
            "device_name": name,
            "device_capability": list(capability),
        }
        require("T4" in name.upper(), f"static v2 formal training requires T4, got {name!r}")
        require(capability == (7, 5), f"static v2 formal training requires CUDA capability 7.5, got {capability}")

        stage = "static_v2_tests"
        junit_path = FORMAL_ROOT / "static_v2_tests.xml"
        test_paths = [
            SOURCE_DIR / "tests" / "test_card_preprocessing.py",
            SOURCE_DIR / "tests" / "test_card_dataset.py",
            SOURCE_DIR / "tests" / "test_card_encoder.py",
            SOURCE_DIR / "tests" / "test_pretrain_tasks.py",
            SOURCE_DIR / "tests" / "test_static_card_export.py",
            SOURCE_DIR / "tests" / "test_evaluate_card_embeddings.py",
        ]
        try:
            run(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "-q",
                    f"--junitxml={junit_path}",
                    *[str(path) for path in test_paths],
                ],
                commands,
            )
        finally:
            if junit_path.is_file():
                test_results = read_junit(junit_path)
        require(test_results is not None, "pytest did not produce a JUnit report")
        require(int(test_results["tests"]) > 0, "pytest collected no static v2 tests")
        require(int(test_results["errors"]) == 0, "static v2 tests reported errors")
        require(int(test_results["failures"]) == 0, "static v2 tests reported failures")
        require(int(test_results["skipped"]) == 0, "static v2 tests unexpectedly skipped coverage")

        stage = "formal_training"
        formal_training_started = True
        run(
            [
                sys.executable,
                "-m",
                "training.pretrain_card_encoder",
                "--config",
                str(config_path),
                "--rebuild-cache",
            ],
            commands,
        )

        stage = "formal_training_validation"
        preprocess_path = CACHE_DIR / "preprocess_manifest.json"
        training_summary_path = FORMAL_ROOT / "formal_training_summary.json"
        split_manifest_path = FORMAL_ROOT / "split_manifest.json"
        require(preprocess_path.is_file(), "formal preprocessing manifest is missing")
        require(training_summary_path.is_file(), "formal training summary is missing")
        require(split_manifest_path.is_file(), "formal split manifest is missing")
        preprocess = json.loads(preprocess_path.read_text(encoding="utf-8"))
        training_summary = json.loads(training_summary_path.read_text(encoding="utf-8"))
        split_manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
        require(preprocess.get("schema_version") == "static_card_v2", "wrong preprocessing schema")
        require(int(preprocess.get("source_row_count", -1)) == 2022, "wrong source row count")
        require(int(preprocess.get("card_count", -1)) == 1267, "wrong card count")
        require(int(preprocess.get("detail_count", -1)) == 2014, "wrong detail count")
        require(
            preprocess.get("detail_type_counts")
            == {"ABILITY": 218, "ATTACK": 1556, "CARD_EFFECT": 240},
            "wrong detail type counts",
        )
        require(int(preprocess.get("unresolved_count", -1)) == 0, "preprocessing has unresolved rows")
        require(training_summary.get("schema_version") == "static_card_formal_training_v2", "wrong formal summary schema")
        require(training_summary.get("success") is True, "formal training summary reports failure")
        require(training_summary.get("run_mode") == RUN_MODE, "formal training summary has wrong run mode")
        require(all_finite(training_summary), "formal training summary contains non-finite metrics")

        tiny = training_summary.get("tiny_overfit") or {}
        require(tiny.get("schema_version") == "static_card_tiny_overfit_v2", "wrong tiny-overfit schema")
        require(tiny.get("success") is True, "formal-run tiny-overfit did not pass")
        require(int(tiny.get("card_count", -1)) == 32, "formal-run tiny-overfit used the wrong card count")
        require(int(tiny.get("steps", -1)) == 300, "formal-run tiny-overfit used the wrong step count")
        require(
            math.isfinite(float(tiny.get("loss_ratio", math.nan)))
            and float(tiny["loss_ratio"]) <= float(tiny["required_loss_ratio"]),
            "formal-run tiny-overfit loss ratio failed",
        )
        require(
            all(
                math.isfinite(float(value)) and float(value) > 0
                for value in (tiny.get("first_gradient_norms") or {}).values()
            )
            and bool(tiny.get("first_gradient_norms")),
            "formal-run tiny-overfit gradient gate failed",
        )
        require(
            all(math.isfinite(float(value)) for value in (tiny.get("final_task_losses") or {}).values())
            and bool(tiny.get("final_task_losses")),
            "formal-run tiny-overfit task loss is non-finite or missing",
        )
        tiny_checkpoint = FORMAL_ROOT / "tiny_overfit" / "tiny_overfit.pt"
        require(tiny_checkpoint.is_file(), "formal-run tiny-overfit checkpoint is missing")
        tiny_payload = load_checkpoint(tiny_checkpoint)
        require(tiny_payload.get("stage") == "tiny_overfit", "tiny-overfit checkpoint has wrong stage")
        require(
            tiny_payload.get("lineage", {}).get("split_manifest_sha256")
            == sha256_file(split_manifest_path),
            "tiny-overfit checkpoint split lineage mismatch",
        )

        selection = training_summary.get("selection_training") or {}
        best_epoch = int(selection.get("best_epoch", -1))
        completed_epochs = int(selection.get("completed_epochs", 0))
        require(best_epoch >= 0, "selection training has no best epoch")
        require(completed_epochs >= 80, "selection training completed fewer than 80 formal epochs")
        require(completed_epochs >= best_epoch + 1, "selection training ended before its best epoch")
        best_checkpoint = WORK_DIR / str(selection.get("best_checkpoint"))
        last_checkpoint = WORK_DIR / str(selection.get("last_checkpoint"))
        require(best_checkpoint.is_file(), "selection best checkpoint is missing")
        require(last_checkpoint.is_file(), "selection last checkpoint is missing")
        require(
            sha256_file(best_checkpoint) == str(selection.get("best_checkpoint_sha256")),
            "selection best checkpoint hash mismatch",
        )
        best_payload = load_checkpoint(best_checkpoint)
        require(best_payload.get("stage") == "split_selection_best", "selection checkpoint has wrong stage")
        require(
            best_payload.get("lineage", {}).get("split_manifest_sha256") == sha256_file(split_manifest_path),
            "selection checkpoint split lineage mismatch",
        )

        require(split_manifest.get("schema_version") == "static_card_split_v2", "wrong split schema")
        require(split_manifest.get("mode") == "card_id", "split mode is not card_id")
        require(split_manifest.get("transductive_catalog_schema") is True, "transductive schema is not declared")
        split_sets = {
            key: {int(value) for value in split_manifest[f"{key}_indices"]}
            for key in ("train", "validation", "test")
        }
        require(all(split_sets.values()), "formal split contains an empty partition")
        require(not (split_sets["train"] & split_sets["validation"]), "train/validation split overlap")
        require(not (split_sets["train"] & split_sets["test"]), "train/test split overlap")
        require(not (split_sets["validation"] & split_sets["test"]), "validation/test split overlap")
        require(set.union(*split_sets.values()) == set(range(1267)), "formal split is not a complete partition")

        refit = training_summary.get("production_refit") or {}
        require(refit.get("enabled") is True, "production refit summary is disabled")
        require(int(refit.get("card_count", -1)) == 1267, "production refit did not use all cards")
        require(int(refit.get("epochs", -1)) == best_epoch + 1, "production refit epochs do not match best_epoch + 1")
        production_checkpoint = WORK_DIR / str(refit.get("checkpoint"))
        require(production_checkpoint.is_file(), "production refit checkpoint is missing")
        require(
            sha256_file(production_checkpoint) == str(refit.get("checkpoint_sha256")),
            "production refit checkpoint hash mismatch",
        )
        production_payload = load_checkpoint(production_checkpoint)
        require(production_payload.get("stage") == "full_catalog_production_refit", "production checkpoint has wrong stage")
        require(int(production_payload.get("epoch", -1)) + 1 == best_epoch + 1, "production checkpoint epoch mismatch")

        stage = "selection_export"
        selection_artifacts = FORMAL_ROOT / "selection_artifacts"
        run(
            [
                sys.executable,
                "-m",
                "training.export_card_embeddings",
                "--checkpoint",
                str(best_checkpoint),
                "--cache-dir",
                str(CACHE_DIR),
                "--output-dir",
                str(selection_artifacts),
                "--batch-size",
                "64",
            ],
            commands,
        )

        stage = "selection_probe"
        evaluation_dir = FORMAL_ROOT / "evaluation"
        run(
            [
                sys.executable,
                "-m",
                "training.evaluate_card_embeddings",
                "--cache-dir",
                str(CACHE_DIR),
                "--artifacts-dir",
                str(selection_artifacts),
                "--split-manifest",
                str(split_manifest_path),
                "--output-dir",
                str(evaluation_dir),
            ],
            commands,
        )
        evaluation_path = evaluation_dir / "evaluation.json"
        require(evaluation_path.is_file(), "frozen probe evaluation report is missing")
        evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
        require(
            evaluation.get("schema_version") == "static_card_embedding_evaluation_v2",
            "wrong frozen probe evaluation schema",
        )
        require(all_finite(evaluation), "frozen probe evaluation contains non-finite values")
        acceptance = evaluation.get("acceptance") or {}
        hard_checks = acceptance.get("hard_checks") or {}
        structural_checks = {
            key: value
            for key, value in hard_checks.items()
            if key != "online_target_masked_recovery"
        }
        require(structural_checks, "frozen probe evaluation has no structural acceptance checks")
        require(
            all(value is True for value in structural_checks.values()),
            f"frozen probe structural acceptance failed: {structural_checks}",
        )
        require(
            "online_target_masked_recovery" in hard_checks,
            "online target-masked evaluation has no quality acceptance check",
        )

        stage = "production_export_batch64"
        production_artifacts64 = FORMAL_ROOT / "production_artifacts_batch64"
        run(
            [
                sys.executable,
                "-m",
                "training.export_card_embeddings",
                "--checkpoint",
                str(production_checkpoint),
                "--cache-dir",
                str(CACHE_DIR),
                "--output-dir",
                str(production_artifacts64),
                "--batch-size",
                "64",
            ],
            commands,
        )

        stage = "production_export_batch257"
        production_artifacts257 = FORMAL_ROOT / "production_artifacts_batch257"
        run(
            [
                sys.executable,
                "-m",
                "training.export_card_embeddings",
                "--checkpoint",
                str(production_checkpoint),
                "--cache-dir",
                str(CACHE_DIR),
                "--output-dir",
                str(production_artifacts257),
                "--batch-size",
                "257",
            ],
            commands,
        )

        stage = "production_canonical_validation"
        alignment_path = FORMAL_ROOT / "production_alignment_report.json"
        run(
            [
                sys.executable,
                "-m",
                "training.validate_card_artifacts",
                "--cache-dir",
                str(CACHE_DIR),
                "--artifact-dir",
                str(production_artifacts64),
                "--artifact-dir",
                str(production_artifacts257),
                "--output",
                str(alignment_path),
            ],
            commands,
        )
        alignment = json.loads(alignment_path.read_text(encoding="utf-8"))
        require(alignment.get("success") is True, "production artifact canonical validation failed")
        require(all_finite(alignment), "production alignment report contains non-finite values")

        promotion_ready = bool(acceptance.get("passed"))
        stage = "source_cleanup"
        shutil.rmtree(SOURCE_DIR)
        source_cleaned = not SOURCE_DIR.exists()
        require(source_cleaned, "successful formal run did not clean extracted source")
        shutil.rmtree(WORK_DIR / ".pytest_cache", ignore_errors=True)
        require(not (WORK_DIR / ".pytest_cache").exists(), "successful formal run did not clean .pytest_cache")

        stage = "complete"
        write_json(
            SUMMARY_PATH,
            {
                "schema_version": "static_card_v2_formal_run_summary_v1",
                "success": True,
                "promotion_ready": promotion_ready,
                "run_mode": RUN_MODE,
                "completed_stage": stage,
                "formal_training_started": formal_training_started,
                "device": device,
                "source_archive_sha256": ARCHIVE_SHA256,
                "source_file_hashes": SOURCE_FILE_HASHES,
                "source_cleaned": source_cleaned,
                "runner_sha256": sha256_file(Path(__file__).resolve()),
                "config_sha256": config_sha256,
                "commands": commands,
                "tests": test_results,
                "preprocess": preprocess,
                "split": {
                    "schema_version": split_manifest.get("schema_version"),
                    "sha256": sha256_file(split_manifest_path),
                    "train_cards": len(split_sets["train"]),
                    "validation_cards": len(split_sets["validation"]),
                    "test_cards": len(split_sets["test"]),
                },
                "selection_training": selection,
                "test_metrics": training_summary.get("test_metrics"),
                "production_refit": refit,
                "evaluation": evaluation,
                "production_alignment": alignment,
                "quality_gate_note": (
                    "Promotion records the online masked split-test evaluation together with frozen "
                    "embedding diagnostics. A quality-gate failure preserves a COMPLETE run and "
                    "all artifacts, but promotion_ready remains false."
                ),
                "elapsed_seconds": time.time() - started,
            },
        )
        print(
            f"STATIC_V2_FORMAL_COMPLETE promotion_ready={promotion_ready} summary={SUMMARY_PATH}",
            flush=True,
        )
    except BaseException as exc:
        write_json(
            SUMMARY_PATH,
            {
                "schema_version": "static_card_v2_formal_run_summary_v1",
                "success": False,
                "promotion_ready": False,
                "run_mode": RUN_MODE,
                "completed_stage": stage,
                "formal_training_started": formal_training_started,
                "device": device,
                "source_archive_sha256": ARCHIVE_SHA256,
                "source_cleaned": source_cleaned,
                "commands": commands,
                "tests": test_results,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "elapsed_seconds": time.time() - started,
            },
        )
        raise


if __name__ == "__main__":
    main()
'''


def build_archive() -> tuple[bytes, dict[str, str]]:
    source_hashes: dict[str, str] = {}
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for relative in SOURCE_FILES:
            path = ROOT / relative
            if not path.is_file():
                raise FileNotFoundError(f"Kaggle static v2 formal source file is missing: {relative}")
            payload = path.read_bytes()
            source_hashes[relative] = hashlib.sha256(payload).hexdigest()
            info = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, payload)
    return buffer.getvalue(), source_hashes


def build_runner() -> str:
    payload, source_hashes = build_archive()
    encoded = "\n".join(textwrap.wrap(base64.b64encode(payload).decode("ascii"), 100))
    return (
        RUNNER_TEMPLATE.replace("__ARCHIVE_B64__", encoded)
        .replace("__ARCHIVE_SHA256__", hashlib.sha256(payload).hexdigest())
        .replace("__SOURCE_FILE_HASHES__", json.dumps(source_hashes, sort_keys=True))
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the self-contained Kaggle static v2 formal runner")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "kaggle_card_pretrain_formal" / "run_card_pretrain_formal.py",
    )
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    generated = build_runner()
    if args.check:
        raise SystemExit(
            0 if args.output.exists() and args.output.read_text(encoding="utf-8") == generated else 1
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(generated, encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
