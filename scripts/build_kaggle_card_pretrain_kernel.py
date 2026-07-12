from __future__ import annotations

import argparse
import base64
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
    "training/__init__.py",
    "training/evaluate_card_embeddings.py",
    "training/export_card_embeddings.py",
    "training/pretrain_card_encoder.py",
]


def build_archive() -> str:
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for relative in SOURCE_FILES:
            info = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, (ROOT / relative).read_bytes())
    return "\n".join(textwrap.wrap(base64.b64encode(buffer.getvalue()).decode("ascii"), 100))


def build_runner() -> str:
    payload = build_archive()
    return f'''#!/usr/bin/env python3
from __future__ import annotations

import base64
import os
import subprocess
import sys
import zipfile
from io import BytesIO
from pathlib import Path

ARCHIVE_B64 = """
{payload}
"""


def run(cmd: list[str], source_dir: Path, work_dir: Path) -> None:
    print("+", " ".join(cmd), flush=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(source_dir) + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run(cmd, check=True, cwd=work_dir, env=env)


def main() -> None:
    work_dir = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path.cwd()
    source_dir = work_dir / "card_pretrain_src"
    source_dir.mkdir(parents=True, exist_ok=True)
    payload = base64.b64decode("".join(ARCHIVE_B64.split()))
    with zipfile.ZipFile(BytesIO(payload)) as archive:
        archive.extractall(source_dir)

    config = source_dir / "configs" / "card_pretrain.yaml"
    checkpoint = work_dir / "checkpoints" / "card_encoder_best.pt"
    output = work_dir / "artifacts" / "card_embeddings.pt"
    run([sys.executable, "-m", "training.pretrain_card_encoder", "--config", str(config), "--rebuild-cache"], source_dir, work_dir)
    run([sys.executable, "-m", "training.export_card_embeddings", "--checkpoint", str(checkpoint), "--output", str(output)], source_dir, work_dir)
    run([sys.executable, "-m", "training.evaluate_card_embeddings", "--embeddings", str(output)], source_dir, work_dir)


if __name__ == "__main__":
    main()
'''


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the self-contained Kaggle CardEncoder training script.")
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
