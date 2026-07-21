from __future__ import annotations

import argparse
import json
import random
from datetime import datetime
from pathlib import Path

import torch

from .config import load_config
from .deck import load_target_deck
from .league import EvaluationResult
from .model import SemanticSelfPlayModel
from .phase import PhaseController
from .rollout import AsymmetricRolloutCollector, CgBattleEnvironment
from .training import AsymmetricSelfPlayTrainer, FixedCalibrationHoldout


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "competition_selfplay/configs/raging_bolt_fast.yaml"


def _device(requested: str) -> torch.device:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested by config/CLI but is unavailable")
    return torch.device(requested)


def _evaluation_result(batch) -> EvaluationResult:
    terminal = [
        transaction
        for transaction in batch.transactions
        if transaction.learner_controlled and transaction.terminal
    ]
    return EvaluationResult(
        wins=sum(transaction.outcome > 0 for transaction in terminal),
        losses=sum(transaction.outcome < 0 for transaction in terminal),
        draws=sum(transaction.outcome == 0 for transaction in terminal),
    )


def run(args: argparse.Namespace) -> dict[str, object]:
    config = load_config(args.config)
    device = _device(args.device or config.training.device)
    random.seed(config.training.seed)
    torch.manual_seed(config.training.seed)
    deck = load_target_deck(config.target_deck, ROOT)

    learner = SemanticSelfPlayModel(config.model).to(device)
    target_semantic = learner.build_target_semantic_module().to(device)
    frozen_opponent = SemanticSelfPlayModel(config.model).to(device)
    frozen_opponent.load_state_dict(learner.state_dict())
    optimizer = torch.optim.Adam(learner.parameters(), lr=config.training.learning_rate)
    phase_controller = PhaseController(config.reward)
    trainer = AsymmetricSelfPlayTrainer(
        config=config,
        learner=learner,
        frozen_opponent=frozen_opponent,
        target_semantic=target_semantic,
        optimizer=optimizer,
        phase_controller=phase_controller,
        device=device,
    )
    collector = AsymmetricRolloutCollector(
        online_model=learner,
        frozen_opponent=frozen_opponent,
        target_semantic=target_semantic,
        model_config=config.model,
        reward_config=config.reward,
        phase_controller=phase_controller,
        device=device,
    )
    if args.resume is not None:
        trainer.load_checkpoint(args.resume)

    runtime_root = str(args.runtime_root.resolve()) if args.runtime_root is not None else None
    environment_factory = lambda: CgBattleEnvironment(runtime_root)
    holdout = trainer.calibration_holdout
    if holdout is None:
        holdout_batch = collector.collect_batch(
            environment_factory=environment_factory,
            deck=deck.card_ids,
            games=args.holdout_games,
            max_selects_per_game=args.max_selects_per_game,
            count_completed_games=False,
            capture_swapped_states=True,
        )
        holdout = FixedCalibrationHoldout.from_transactions(
            holdout_batch.transactions,
            collector.vectorizer,
            gamma=config.training.gamma,
        )
        trainer.calibration_holdout = holdout

    run_id = "run-" + datetime.now().strftime("%Y%m%d-%H%M%S")
    if args.output_root is None:
        metrics_root = Path(config.training.metrics_root) / run_id
        checkpoint_root = Path(config.training.checkpoint_root) / run_id
    else:
        if args.output_root.exists():
            raise FileExistsError(f"output root already exists: {args.output_root}")
        metrics_root = args.output_root / "metrics"
        checkpoint_root = args.output_root / "checkpoints"
    if metrics_root.exists() or checkpoint_root.exists():
        raise FileExistsError(
            f"run output already exists: metrics={metrics_root}, checkpoints={checkpoint_root}"
        )
    metrics_root.mkdir(parents=True, exist_ok=False)
    checkpoint_root.mkdir(parents=True, exist_ok=False)
    metrics_path = metrics_root / "metrics.jsonl"
    checkpoint_path = checkpoint_root / "checkpoint-latest.pt"
    latest_metrics: dict[str, object] = {}
    for iteration in range(args.iterations):
        batch = collector.collect_batch(
            environment_factory=environment_factory,
            deck=deck.card_ids,
            games=args.games_per_batch,
            max_selects_per_game=args.max_selects_per_game,
        )
        update = trainer.update_online(batch)
        # Rewards for this batch remain the floats stored during collection;
        # the complete target semantic path moves only after learner updates.
        trainer.finish_rollout_batch()
        trainer.record_training_iteration()

        latest_metrics = dict(update.metrics)
        if phase_controller.completed_games >= config.reward.phase_a_games:
            calibration = holdout.evaluate(
                learner,
                device=device,
            )
            decision = phase_controller.consider_calibration(calibration)
            latest_metrics.update(calibration.as_dict())
            latest_metrics["calibration_gate_passed"] = int(decision.passed)

        if (
            trainer.league_state.learner_revision > 0
            and trainer.league_state.learner_revision % args.evaluate_every_iterations == 0
        ):
            evaluation_batch = collector.collect_batch(
                environment_factory=environment_factory,
                deck=deck.card_ids,
                games=config.promotion.evaluation_games,
                max_selects_per_game=args.max_selects_per_game,
                count_completed_games=False,
            )
            league_decision = trainer.apply_league_evaluation(
                _evaluation_result(evaluation_batch)
            )
            latest_metrics["league_action"] = league_decision.action.value
            latest_metrics["league_score_rate"] = league_decision.score_rate
        latest_metrics.update(
            {
                "iteration": iteration,
                "completed_games": phase_controller.completed_games,
                "phase": phase_controller.phase.value,
                "shaping_alpha_next_batch": phase_controller.shaping_alpha(),
                "league_converged": int(trainer.league_state.converged),
            }
        )
        with metrics_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(latest_metrics, ensure_ascii=False) + "\n")
        trainer.save_checkpoint(checkpoint_path)
        if trainer.league_state.converged:
            break

    return {
        "run_id": run_id,
        "metrics_path": str(metrics_path.resolve()),
        "checkpoint": str(checkpoint_path.resolve()),
        "metrics": latest_metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run causal transaction-level semantic self-play training"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--runtime-root", type=Path, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--games-per-batch", type=int, required=True)
    parser.add_argument("--holdout-games", type=int, default=64)
    parser.add_argument("--evaluate-every-iterations", type=int, default=10)
    parser.add_argument("--max-selects-per-game", type=int, default=2_000)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args()
    if min(
        args.iterations,
        args.games_per_batch,
        args.holdout_games,
        args.evaluate_every_iterations,
    ) <= 0:
        parser.error("iterations, game counts and evaluation interval must be positive")
    print(json.dumps(run(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
