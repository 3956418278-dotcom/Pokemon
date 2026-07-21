from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import yaml

from decision_agent_v1.state_upgrade.deck_prior import build_deck_prior


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "decision_agent_v1/configs/policy_value_v2.yaml")
    parser.add_argument("--max-templates", type=int, default=None)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    rows = []
    for value in sorted(glob.glob(str(ROOT / "outputs/replay_extract/decks/*/deck_observations.jsonl"))):
        with open(value, encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    card_records = json.loads((ROOT / "static_card/artifacts/card_data/card_records.json").read_text(encoding="utf-8"))
    payload, cooccurrence = build_deck_prior(
        rows,
        card_records,
        config["split"]["train_dates"],
        max_templates=args.max_templates or int(config["state_upgrade"]["max_templates"]),
    )
    output = ROOT / config["data"]["belief_output_dir"]
    output.mkdir(parents=True, exist_ok=True)
    (output / "deck_templates.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    metadata = {
        "schema_version": payload["schema_version"],
        "source_split": payload["source_split"],
        "source_dates": payload["source_dates"],
        "complete_deck_observations": payload["complete_deck_observations"],
        "unique_fingerprints": payload["unique_fingerprints"],
        "template_count": len(payload["templates"]),
        "template_hash": payload["template_hash"],
        "actor_input_contract": "posterior updated from public opponent Card IDs only",
        "complete_opponent_deck_usage": ["offline template construction", "auxiliary archetype label", "belief evaluation"],
    }
    (output / "deck_template_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(output / "card_cooccurrence.npz", card_ids=np.asarray(payload["card_ids"]), cooccurrence=cooccurrence)
    print(output / "deck_templates.json")


if __name__ == "__main__":
    main()
