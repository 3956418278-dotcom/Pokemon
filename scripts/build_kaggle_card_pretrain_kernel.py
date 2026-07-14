from __future__ import annotations

import argparse
import base64
import hashlib
import textwrap
import zipfile
from io import BytesIO
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILES = [
    "configs/card_pretrain.yaml",
    "configs/card_pretrain_smoke.yaml",
    "data/__init__.py",
    "data/card_dataset.py",
    "data/card_preprocessing.py",
    "models/__init__.py",
    "models/card_encoder.py",
    "models/card_pretrain_heads.py",
    "models/static_card_adapter.py",
    "training/__init__.py",
    "training/export_card_embeddings.py",
    "training/pretrain_card_encoder.py",
    "training/validate_card_artifacts.py",
    "tests/conftest.py",
    "tests/test_card_preprocessing.py",
    "tests/test_card_dataset.py",
    "tests/test_card_encoder.py",
    "tests/test_pretrain_tasks.py",
    "tests/test_static_card_export.py",
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


ARCHIVE_B64 = """
__ARCHIVE_B64__
"""
ARCHIVE_SHA256 = "__ARCHIVE_SHA256__"
RUN_MODE = "static_v2_smoke"
WORK_DIR = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path.cwd()
SOURCE_DIR = WORK_DIR / "card_pretrain_v2_src"
SMOKE_ROOT = WORK_DIR / "outputs" / "static_v2_smoke"
SUMMARY_PATH = SMOKE_ROOT / "smoke_summary.json"


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
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\\n", encoding="utf-8")


def run(command: list[str], commands: list[list[str]]) -> None:
    print("+", " ".join(command), flush=True)
    commands.append(command)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SOURCE_DIR) + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run(command, check=True, cwd=WORK_DIR, env=env)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    started = time.time()
    commands: list[list[str]] = []
    device: dict[str, object] = {}
    stage = "source_archive"
    SMOKE_ROOT.mkdir(parents=True, exist_ok=True)
    try:
        archive_payload = base64.b64decode("".join(ARCHIVE_B64.split()))
        require(sha256_bytes(archive_payload) == ARCHIVE_SHA256, "embedded source archive SHA256 mismatch")
        shutil.rmtree(SOURCE_DIR, ignore_errors=True)
        SOURCE_DIR.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(BytesIO(archive_payload)) as archive:
            archive.extractall(SOURCE_DIR)

        stage = "smoke_config_guard"
        import yaml

        config_path = SOURCE_DIR / "configs" / "card_pretrain_smoke.yaml"
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        require(config.get("run_mode") == RUN_MODE, "smoke config has the wrong run_mode")
        require(config["training"].get("device") == "cuda", "smoke config must require CUDA")
        require(int(config["training"].get("max_epochs", -1)) == 0, "smoke config must disable formal epochs")
        require(bool(config["tiny_overfit"].get("enabled")), "tiny-overfit must be enabled")
        require(int(config["tiny_overfit"].get("steps", -1)) == 300, "smoke must use exactly 300 steps")
        require(int(config["tiny_overfit"].get("card_count", -1)) == 32, "smoke must use exactly 32 cards")

        stage = "t4_environment_guard"
        import torch

        require(torch.cuda.is_available(), "static v2 smoke requires CUDA")
        name = torch.cuda.get_device_name(0)
        capability = tuple(int(value) for value in torch.cuda.get_device_capability(0))
        device = {
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "cuda_available": True,
            "device_name": name,
            "device_capability": list(capability),
        }
        require("T4" in name.upper(), f"static v2 smoke requires T4, got {name!r}")
        require(capability == (7, 5), f"static v2 smoke requires CUDA capability 7.5, got {capability}")

        stage = "static_v2_tests"
        test_paths = [
            SOURCE_DIR / "tests" / "test_card_preprocessing.py",
            SOURCE_DIR / "tests" / "test_card_dataset.py",
            SOURCE_DIR / "tests" / "test_card_encoder.py",
            SOURCE_DIR / "tests" / "test_pretrain_tasks.py",
            SOURCE_DIR / "tests" / "test_static_card_export.py",
        ]
        run([sys.executable, "-m", "pytest", "-q", *[str(path) for path in test_paths]], commands)

        stage = "tiny_overfit"
        run(
            [
                sys.executable,
                "-m",
                "training.pretrain_card_encoder",
                "--config",
                str(config_path),
                "--rebuild-cache",
                "--tiny-overfit-only",
                "--tiny-steps",
                "300",
                "--tiny-card-count",
                "32",
            ],
            commands,
        )
        cache_dir = WORK_DIR / "artifacts" / "card_data_v2"
        tiny_dir = SMOKE_ROOT / "tiny_overfit"
        checkpoint = tiny_dir / "tiny_overfit.pt"
        metrics_path = tiny_dir / "metrics.json"
        preprocess_path = cache_dir / "preprocess_manifest.json"
        require(checkpoint.is_file(), "tiny-overfit checkpoint was not created")
        require(metrics_path.is_file(), "tiny-overfit metrics were not created")
        require(preprocess_path.is_file(), "preprocess manifest was not created")

        preprocess = json.loads(preprocess_path.read_text(encoding="utf-8"))
        require(preprocess.get("schema_version") == "static_card_v2", "wrong preprocessing schema")
        require(int(preprocess.get("source_row_count", -1)) == 2022, "wrong source row count")
        require(int(preprocess.get("card_count", -1)) == 1267, "wrong card count")
        require(int(preprocess.get("detail_count", -1)) == 2014, "wrong detail count")
        require(preprocess.get("detail_type_counts") == {"ABILITY": 218, "ATTACK": 1556, "CARD_EFFECT": 240}, "wrong detail type counts")
        require(int(preprocess.get("unresolved_count", -1)) == 0, "preprocessing has unresolved rows")

        tiny = json.loads(metrics_path.read_text(encoding="utf-8"))
        require(tiny.get("schema_version") == "static_card_tiny_overfit_v2", "wrong tiny metrics schema")
        require(tiny.get("success") is True, "tiny-overfit did not pass")
        require(int(tiny.get("card_count", -1)) == 32, "tiny-overfit used the wrong card count")
        require(int(tiny.get("steps", -1)) == 300, "tiny-overfit used the wrong step count")
        require(float(tiny["loss_ratio"]) <= float(tiny["required_loss_ratio"]), "tiny loss ratio failed")
        require(all(math.isfinite(float(value)) and float(value) > 0 for value in tiny["first_gradient_norms"].values()), "tiny gradient gate failed")
        require(all(math.isfinite(float(value)) for value in tiny["final_task_losses"].values()), "tiny task loss is non-finite")

        stage = "formal_training_absence_guard"
        forbidden = [
            SMOKE_ROOT / "formal_checkpoints_disabled" / "card_encoder_best.pt",
            SMOKE_ROOT / "formal_checkpoints_disabled" / "card_encoder_last.pt",
            SMOKE_ROOT / "formal_logs_disabled" / "card_pretrain_metrics.jsonl",
        ]
        require(not any(path.exists() for path in forbidden), "formal training output exists during smoke")

        stage = "flat_export_batch64"
        artifact64 = SMOKE_ROOT / "artifacts_batch64"
        run(
            [
                sys.executable,
                "-m",
                "training.export_card_embeddings",
                "--checkpoint",
                str(checkpoint),
                "--cache-dir",
                str(cache_dir),
                "--output-dir",
                str(artifact64),
                "--batch-size",
                "64",
            ],
            commands,
        )

        stage = "flat_export_batch257"
        artifact257 = SMOKE_ROOT / "artifacts_batch257"
        run(
            [
                sys.executable,
                "-m",
                "training.export_card_embeddings",
                "--checkpoint",
                str(checkpoint),
                "--cache-dir",
                str(cache_dir),
                "--output-dir",
                str(artifact257),
                "--batch-size",
                "257",
            ],
            commands,
        )

        stage = "full_pool_alignment"
        alignment_path = SMOKE_ROOT / "alignment_report.json"
        run(
            [
                sys.executable,
                "-m",
                "training.validate_card_artifacts",
                "--cache-dir",
                str(cache_dir),
                "--artifact-dir",
                str(artifact64),
                "--artifact-dir",
                str(artifact257),
                "--output",
                str(alignment_path),
            ],
            commands,
        )
        alignment = json.loads(alignment_path.read_text(encoding="utf-8"))
        require(alignment.get("success") is True, "full-pool alignment did not pass")

        stage = "complete"
        write_json(
            SUMMARY_PATH,
            {
                "schema_version": "static_card_v2_smoke_summary_v1",
                "success": True,
                "run_mode": RUN_MODE,
                "completed_stage": stage,
                "formal_training_started": False,
                "tests_passed": 22,
                "device": device,
                "source_archive_sha256": ARCHIVE_SHA256,
                "runner_sha256": sha256_file(Path(__file__).resolve()),
                "config_sha256": sha256_file(config_path),
                "commands": commands,
                "preprocess": preprocess,
                "tiny_overfit": tiny,
                "alignment": alignment,
                "elapsed_seconds": time.time() - started,
            },
        )
        print(f"STATIC_V2_SMOKE_SUCCESS {SUMMARY_PATH}", flush=True)
    except BaseException as exc:
        write_json(
            SUMMARY_PATH,
            {
                "schema_version": "static_card_v2_smoke_summary_v1",
                "success": False,
                "run_mode": RUN_MODE,
                "completed_stage": stage,
                "formal_training_started": False,
                "device": device,
                "source_archive_sha256": ARCHIVE_SHA256,
                "commands": commands,
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


def build_archive() -> tuple[str, str]:
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for relative in SOURCE_FILES:
            path = ROOT / relative
            if not path.is_file():
                raise FileNotFoundError(f"Kaggle static v2 source file is missing: {relative}")
            info = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, path.read_bytes())
    payload = buffer.getvalue()
    encoded = "\n".join(textwrap.wrap(base64.b64encode(payload).decode("ascii"), 100))
    return encoded, hashlib.sha256(payload).hexdigest()


def build_runner() -> str:
    payload, archive_sha256 = build_archive()
    return RUNNER_TEMPLATE.replace("__ARCHIVE_B64__", payload).replace(
        "__ARCHIVE_SHA256__", archive_sha256
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the self-contained Kaggle static v2 smoke runner")
    parser.add_argument("--output", type=Path, default=ROOT / "kaggle_card_pretrain" / "run_card_pretrain.py")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    generated = build_runner()
    if args.check:
        raise SystemExit(0 if args.output.exists() and args.output.read_text(encoding="utf-8") == generated else 1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(generated, encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
