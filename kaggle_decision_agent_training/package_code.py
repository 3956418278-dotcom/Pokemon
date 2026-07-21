from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_DATA_FILES = (
    "__init__.py",
    "decision_schema.py",
    "legal_options.py",
    "state_schema.py",
    "observation_parser.py",
    "game_memory.py",
    "replay_dataset.py",
    "online_replay_importer.py",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs/decision_agent_v1/kaggle_code_dataset",
    )
    args = parser.parse_args()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        ROOT / "decision_agent_v1",
        output / "decision_agent_v1",
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    data_output = output / "data"
    data_output.mkdir(exist_ok=True)
    for name in REQUIRED_DATA_FILES:
        shutil.copy2(ROOT / "data" / name, data_output / name)
    vocab_output = output / "static_card/artifacts/card_data"
    vocab_output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        ROOT / "static_card/artifacts/card_data/card_id_to_index.json",
        vocab_output / "card_id_to_index.json",
    )
    metadata = {
        "title": "ptcg-decision-agent-v1-code",
        "id": "f7e6n5g4/ptcg-decision-agent-v1-code",
        "licenses": [{"name": "other"}],
    }
    (output / "dataset-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(output)


if __name__ == "__main__":
    main()
