#!/usr/bin/env python3
"""Build a Kaggle submission package from trained PPO weights.

This is intentionally separate from train_agent.py. Use train_agent.py for
cloud training and this script when you want to convert a trained output into
`submission.tar.gz`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from train_agent import (
    KAGGLE_INPUT,
    KAGGLE_WORKING,
    RULE_WEIGHTS,
    choose_submission_deck,
    copy_cg_runtime,
    discover_competition_root,
    kaggle_paths,
    load_baseline_decks,
    local_competition_zip,
    main_py_source,
    write_deck_csv,
    write_submission_tar,
)


def find_artifact(filename: str, working_dir: Path) -> Path | None:
    candidates = [
        working_dir / filename,
        KAGGLE_INPUT / filename,
        Path.cwd() / filename,
    ]
    for path in candidates:
        if path.exists():
            return path
    if KAGGLE_INPUT.exists():
        matches = sorted(KAGGLE_INPUT.rglob(filename))
        if matches:
            return matches[0]
    return None


def build_submission(
    working_dir: Path,
    cg_dir: Path,
    selected_deck: dict[str, Any],
    model_source: Path | None = None,
) -> None:
    main_path = working_dir / "main.py"
    deck_path = working_dir / "deck.csv"
    model_path = working_dir / "model.json"
    weights_path = working_dir / "weights.json"
    tar_path = working_dir / "submission.tar.gz"

    main_path.write_text(main_py_source(), encoding="utf-8")
    write_deck_csv(deck_path, selected_deck["deck_ids"])

    if model_source is not None and model_source.exists() and model_source.resolve() != model_path.resolve():
        model_path.write_text(model_source.read_text(encoding="utf-8"), encoding="utf-8")
    elif not model_path.exists():
        model_path.write_text(json.dumps({"layers": [], "fallback": "rule_score"}, indent=2) + "\n", encoding="utf-8")

    weights_path.write_text(
        json.dumps(
            {
                "rule_weights": RULE_WEIGHTS,
                "selected_deck": selected_deck["name"],
                "note": "Submission metadata; trained PPO weights are in model.json.",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    write_submission_tar(tar_path, [main_path, deck_path, model_path, weights_path], [cg_dir])

    print("Wrote submission artifacts:")
    for path in (main_path, deck_path, model_path, weights_path, tar_path):
        print(f"  - {path} ({path.stat().st_size} bytes)")


def run() -> None:
    input_dir, working_dir = kaggle_paths()
    competition_root = discover_competition_root(input_dir)
    zip_path = local_competition_zip(input_dir)
    cg_dir = copy_cg_runtime(working_dir, competition_root, zip_path)
    sys.path.insert(0, str(working_dir))

    summary_path = find_artifact("training_summary.json", working_dir) or (working_dir / "training_summary.json")
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {"episodes": []}
    decks = load_baseline_decks()
    selected_deck = choose_submission_deck(decks, summary)
    model_source = find_artifact("ppo_weights.json", working_dir)
    if model_source is None:
        model_source = find_artifact("model.json", working_dir)
    build_submission(working_dir, cg_dir, selected_deck, model_source)


if __name__ == "__main__":
    run()
