from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config
from .deck import load_target_deck
from .league import EvaluationResult, LeagueController, LeagueState
from .semantic import NUM_SEMANTIC_CONCEPTS, SEMANTIC_CONCEPT_NAMES


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "competition_selfplay/configs/raging_bolt_fast.yaml"


def dry_run(config_path: Path) -> dict[str, object]:
    config = load_config(config_path)
    deck = load_target_deck(config.target_deck, ROOT)
    controller = LeagueController(config.promotion)
    trained = controller.record_training_iteration(LeagueState())
    wins = int(config.promotion.minimum_games * (config.promotion.win_rate_threshold + 0.02))
    wins = min(wins, config.promotion.minimum_games)
    decision = controller.evaluate(
        trained,
        EvaluationResult(wins=wins, losses=config.promotion.minimum_games - wins, draws=0),
    )
    return {
        "schema_version": config.schema_version,
        "target_deck": deck.name,
        "card_count": len(deck.card_ids),
        "unique_card_count": len(set(deck.card_ids)),
        "deck_sha256": deck.sha256,
        "source_submission_ref": config.target_deck.source_submission_ref,
        "reward": "terminal_outcome_plus_calibrated_semantic_potential_delta",
        "semantic_concepts": list(SEMANTIC_CONCEPT_NAMES),
        "semantic_concept_count": NUM_SEMANTIC_CONCEPTS,
        "prize_windows": ["I0", "I1", "I2", "I3", "I4"],
        "phase_b_gate_value": "semantic_value",
        "value_dimensions": config.model.value_dimensions,
        "phase_a_games": config.reward.phase_a_games,
        "phase_b_calibration_gated": True,
        "simulated_league_action": decision.action.value,
        "passed": decision.state.frozen_opponent_revision == trained.learner_revision,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the fixed-deck self-play configuration")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dry-run", action="store_true", required=True)
    args = parser.parse_args()
    print(json.dumps(dry_run(args.config), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
