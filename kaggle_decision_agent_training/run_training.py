from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
from pathlib import Path

import yaml


KAGGLE_INPUT = Path("/kaggle/input")
KAGGLE_WORKING = Path("/kaggle/working")
OUTPUT_ROOT = KAGGLE_WORKING / "decision_agent_v1_outputs"


def find_code_root(explicit: Path | None = None) -> Path:
    candidates = []
    if explicit is not None:
        candidates.append(explicit)
    if os.environ.get("PTCG_DECISION_CODE_ROOT"):
        candidates.append(Path(os.environ["PTCG_DECISION_CODE_ROOT"]))
    candidates.append(Path(__file__).resolve().parents[1])
    if KAGGLE_INPUT.exists():
        candidates.extend(path.parent for path in KAGGLE_INPUT.glob("**/decision_agent_v1/__init__.py"))
    for candidate in candidates:
        if (candidate / "decision_agent_v1").is_dir() and (candidate / "data/replay_dataset.py").is_file():
            return candidate.resolve()
    raise FileNotFoundError("could not locate the packaged decision_agent_v1 code root")


def discover_daily_replays(code_root: Path, output_root: Path) -> list[Path]:
    if KAGGLE_INPUT.exists():
        daily_dirs = sorted(KAGGLE_INPUT.glob("pokemon-tcg-ai-battle-episodes-2026-*"))
        if daily_dirs:
            sys.path.insert(0, str(code_root))
            from data.online_replay_importer import prepare_mounted_daily_replays

            replay_paths, _ = prepare_mounted_daily_replays(
                daily_dirs, output_root / "mounted_replay_import"
            )
            return replay_paths
    local_archive = code_root / "outputs/replay_extract/replays/replays.zip"
    return [local_archive] if local_archive.exists() else []


def write_loose_index(paths: list[Path], output: Path) -> None:
    rows = []
    for path in paths:
        if path.suffix.lower() == ".zip":
            existing = path.parent / "replay_index.csv"
            if existing.exists():
                output.write_bytes(existing.read_bytes())
                return
            continue
        date_match = re.search(r"(2026-\d{2}-\d{2})", str(path))
        episode_match = re.search(r"(\d{7,})", path.stem)
        if date_match and episode_match:
            rows.append(
                {
                    "source_date": date_match.group(1),
                    "episode_id": episode_match.group(1),
                    "archive_name": str(path.resolve()),
                    "source_dataset": path.parent.name,
                    "size_bytes": path.stat().st_size,
                }
            )
    if not rows:
        raise RuntimeError("no dated Replay JSON files were discovered")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def run(command: list[str], code_root: Path) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=code_root, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-root", type=Path, default=None)
    parser.add_argument("--scale", choices=("small", "medium", "full"), default="full")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    code_root = find_code_root(args.code_root)
    output_root = (
        OUTPUT_ROOT
        if KAGGLE_WORKING.exists()
        else code_root / "outputs/decision_agent_v1/kaggle_kernel_smoke"
    )
    paths = discover_daily_replays(code_root, output_root)
    if not paths:
        raise RuntimeError("no Replay source is mounted")
    output_root.mkdir(parents=True, exist_ok=True)
    index_path = output_root / "replay_index.csv"
    write_loose_index(paths, index_path)
    config = yaml.safe_load(
        (code_root / "decision_agent_v1/configs/policy_value_v1.yaml").read_text(encoding="utf-8")
    )
    config["data"]["replay_paths"] = [str(paths[0] if paths[0].suffix.lower() == ".zip" else KAGGLE_INPUT)]
    config["data"]["replay_index_path"] = str(index_path)
    config["data"]["output_root"] = str(output_root)
    config["data"]["card_vocab_path"] = str(
        code_root / "static_card/artifacts/card_data/card_id_to_index.json"
    )
    config["data"]["action_contract_path"] = str(
        code_root / "decision_agent_v1/contracts/action_semantics.json"
    )
    config["training"]["device"] = "cuda"
    runtime_config = output_root / "policy_value_v1_kaggle.yaml"
    runtime_config.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    print(f"code_root={code_root}")
    print(f"replay_sources={len(paths)}")
    print(f"runtime_config={runtime_config}")
    if args.dry_run:
        return
    python = sys.executable
    run(
        [
            python,
            "-m",
            "decision_agent_v1.scripts.build_training_cache",
            "--config",
            str(runtime_config),
            "--scale",
            args.scale,
        ],
        code_root,
    )
    cache_roots = sorted((output_root / "cache").glob(f"policy_value_v1_*/{args.scale}"))
    if len(cache_roots) != 1:
        raise RuntimeError(f"expected exactly one cache root, found {cache_roots}")
    cache_dir = cache_roots[0]
    run(
        [python, "-m", "decision_agent_v1.scripts.audit_training_cache", str(cache_dir)],
        code_root,
    )
    run(
        [
            python,
            "-m",
            "decision_agent_v1.training.train_policy_value",
            "--config",
            str(runtime_config),
            "--cache-dir",
            str(cache_dir),
            "--checkpoint-dir",
            str(output_root / "checkpoints"),
            "--metrics-dir",
            str(output_root / "metrics/full_training"),
            "--device",
            "cuda",
            "--epochs",
            str(args.epochs),
        ],
        code_root,
    )


if __name__ == "__main__":
    main()
