from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_MODULES = [
    "data/__init__.py",
    "data/state_schema.py",
    "data/observation_parser.py",
    "data/game_memory.py",
    "data/replay_dataset.py",
    "data/online_replay_importer.py",
    "data/replay_training_features.py",
    "models/__init__.py",
    "models/static_card_adapter.py",
    "models/static_detail_aggregator.py",
    "models/dynamic_instance_encoder.py",
    "models/card_instance_fusion.py",
    "models/board_tokenizer.py",
    "models/board_transformer.py",
    "models/dynamic_state_encoder.py",
]


def build_kernel_source(repo_root: Path, train_args: list[str]) -> str:
    modules = {name: (repo_root / name).read_text(encoding="utf-8") for name in DEFAULT_MODULES}
    train_source = (repo_root / "scripts/train_dynamic_replay_features.py").read_text(encoding="utf-8")
    train_source = train_source.replace("from __future__ import annotations\n\n", "", 1)
    train_source = train_source.rsplit('\nif __name__ == "__main__":\n    main()\n', 1)[0]
    return (
        "from pathlib import Path\n"
        "import sys\n\n"
        f"MODULES = {modules!r}\n\n"
        "for name, source in MODULES.items():\n"
        "    path = Path(name)\n"
        "    path.parent.mkdir(parents=True, exist_ok=True)\n"
        "    path.write_text(source, encoding='utf-8')\n\n"
        "sys.path.insert(0, str(Path.cwd()))\n\n"
        f"TRAIN_ARGS = {train_args!r}\n\n"
        + train_source
        + "\n\n"
        "if __name__ == '__main__':\n"
        "    sys.argv = [sys.argv[0]] + TRAIN_ARGS\n"
        "    main()\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a single-file Kaggle dynamic replay feature training script.")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--kernel-dir", type=Path, default=Path("kaggle_dynamic_state_tests"))
    parser.add_argument("--output-file", default="run_dynamic_state_training.py")
    parser.add_argument("--daily-replay-dir", type=Path, action="append", default=[])
    parser.add_argument("--episodes-index-dir", type=Path, default=Path("/kaggle/input/pokemon-tcg-ai-battle-episodes-index"))
    parser.add_argument("--use-daily-manifest", action="store_true")
    parser.add_argument("--daily-dataset-mount-root", type=Path, default=Path("/kaggle/input"))
    parser.add_argument("--reserve-recent-days", type=int, default=3)
    parser.add_argument("--import-split", choices=["train", "reserved"], default="train")
    parser.add_argument("--max-days", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output-dir", type=Path, default=Path("/kaggle/working/dynamic_replay_features"))
    parser.add_argument("--static-artifact-dir", type=Path, default=Path("outputs/card_pretrain/artifacts"))
    args = parser.parse_args()

    train_args: list[str] = []
    for path in args.daily_replay_dir:
        train_args.extend(["--daily-replay-dir", str(path)])
    if args.use_daily_manifest:
        train_args.append("--use-daily-manifest")
        train_args.extend(["--episodes-index-dir", str(args.episodes_index_dir)])
        train_args.extend(["--daily-dataset-mount-root", str(args.daily_dataset_mount_root)])
        train_args.extend(["--reserve-recent-days", str(args.reserve_recent_days)])
        train_args.extend(["--import-split", args.import_split])
        train_args.extend(["--max-days", str(args.max_days)])
    train_args.extend(["--max-samples", str(args.max_samples)])
    train_args.extend(["--epochs", str(args.epochs)])
    train_args.extend(["--batch-size", str(args.batch_size)])
    train_args.extend(["--output-dir", str(args.output_dir)])
    train_args.extend(["--static-artifact-dir", str(args.static_artifact_dir)])

    source = build_kernel_source(args.repo_root, train_args)
    output_path = args.kernel_dir / args.output_file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(source, encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
