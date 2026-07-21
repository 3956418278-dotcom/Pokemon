from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch
import yaml

from decision_agent_v1.adapters.card_vocab_adapter import CardVocabulary
from decision_agent_v1.data.cache import CachedDecisionCorpus, cache_identity
from decision_agent_v1.models.policy_value_model import PolicyValueModel
from decision_agent_v1.training.evaluate_policy_value import (
    evaluate_model,
    fit_training_baselines,
)
from .losses import joint_policy_value_loss


ROOT = Path(__file__).resolve().parents[2]


def move_batch(batch: dict[str, Any], device: torch.device | str) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def gradient_norm(module: torch.nn.Module) -> float:
    total = 0.0
    for parameter in module.parameters():
        if parameter.grad is not None:
            total += float(parameter.grad.detach().pow(2).sum())
    return total**0.5


def train_epoch(
    model: torch.nn.Module,
    batches: Iterable[dict[str, Any]],
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device | str = "cpu",
    policy_weight: float = 1.0,
    value_weight: float = 0.5,
) -> dict[str, float]:
    model.train()
    totals = {"total_loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0}
    steps = 0
    for raw_batch in batches:
        batch = move_batch(raw_batch, device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(batch)
        losses = joint_policy_value_loss(
            model.multiselect_decoder, outputs, batch, policy_weight, value_weight
        )
        losses["total_loss"].backward()
        optimizer.step()
        for key in totals:
            totals[key] += float(losses[key].detach())
        steps += 1
    return {key: value / max(steps, 1) for key, value in totals.items()}


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    torch.save(payload, temporary)
    temporary.replace(path)


def _model_config(config: dict[str, Any], vocab_size: int) -> dict[str, Any]:
    model = config["model"]
    return {
        "vocab_size": vocab_size,
        "model_dim": int(model["model_dim"]),
        "dynamic_hidden_dim": int(model["dynamic_hidden_dim"]),
        "board_layers": int(model["board_layers"]),
        "board_heads": int(model["board_heads"]),
        "board_ffn_dim": int(model["board_ffn_dim"]),
        "dropout": float(model["dropout"]),
    }


def _class_weights(counts: list[int], device: str) -> torch.Tensor:
    total = sum(counts)
    weights = [total / (len(counts) * count) if count else 0.0 for count in counts]
    positive = [weight for weight in weights if weight > 0]
    scale = sum(positive) / len(positive) if positive else 1.0
    return torch.tensor([weight / scale for weight in weights], dtype=torch.float32, device=device)


def _checkpoint_payload(
    model: PolicyValueModel,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: torch.cuda.amp.GradScaler,
    *,
    epoch: int,
    global_step: int,
    model_config: dict[str, Any],
    manifest: dict[str, Any],
    metrics: dict[str, Any],
    class_counts: list[int],
    class_weights: list[float],
) -> dict[str, Any]:
    return {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "model_config": model_config,
        "data_schema_hash": manifest["schema_hash"],
        "action_contract_hash": manifest["action_contract_hash"],
        "card_vocabulary_hash": manifest["card_vocabulary_hash"],
        "adapter_hash": manifest["adapter_hash"],
        "split_dates": manifest["source_dates"],
        "metrics": metrics,
        "value_class_counts": class_counts,
        "value_class_weights": class_weights,
    }


def apply_reduced_acceptance(
    report: dict[str, Any], history: list[dict[str, Any]]
) -> dict[str, Any]:
    """Apply the Prompt-02 core gate while retaining optional diagnostics."""

    trained_policy = report["test"]["policy"]["trained_model"][
        "equivalence_aware_accuracy"
    ]
    random_policy = report["test"]["policy"]["random_legal"][
        "equivalence_aware_accuracy"
    ]
    untrained_policy = report["untrained_test"]["policy"][
        "equivalence_aware_accuracy"
    ]
    value = report["test"]["value"]
    prior = report["test"]["value_class_prior_baseline"]
    finite_by_epoch = {
        int(row["epoch"]): all(
            math.isfinite(float(value)) for value in row.get("train", {}).values()
        )
        for row in history
    }
    nonfinite_epochs = [epoch for epoch, finite in finite_by_epoch.items() if not finite]
    recovery_start_epoch = max(nonfinite_epochs, default=-1) + 1
    post_recovery = [
        finite for epoch, finite in finite_by_epoch.items() if epoch >= recovery_start_epoch
    ]
    report["gradient_recovery"] = {
        "nonfinite_metric_epochs_before_fix": nonfinite_epochs,
        "finite_suffix_start_epoch": recovery_start_epoch,
        "finite_suffix_epochs": [
            epoch for epoch, finite in finite_by_epoch.items()
            if epoch >= recovery_start_epoch and finite
        ],
    }
    report["optional_diagnostics"] = {
        "value_log_loss_above_prior": value["value_log_loss"] < prior["value_log_loss"],
        "value_brier_above_prior": value["brier_score"] < prior["brier_score"],
        "value_expected_calibration_error": value["expected_calibration_error"],
    }
    report["acceptance"] = {
        "policy_above_random": trained_policy > random_policy,
        "policy_above_untrained": trained_policy > untrained_policy,
        "value_accuracy_above_prior": (
            value["win_draw_loss_accuracy"] > prior["win_draw_loss_accuracy"]
        ),
        "value_macro_f1_above_prior": value["macro_f1"] > prior["macro_f1"],
        "value_core_metrics_finite": all(
            math.isfinite(float(value[key]))
            for key in ("win_draw_loss_accuracy", "macro_f1", "value_log_loss")
        ),
        "post_recovery_train_metrics_finite": bool(post_recovery) and all(post_recovery),
        "checkpoint_independent_load": report["best_joint_load_verification"]["finite"],
        "cpu_checkpoint_load": report.get("cpu_checkpoint_load_verification", {}).get(
            "finite", False
        ),
    }
    report["passed"] = all(report["acceptance"].values())
    return report


def verify_checkpoint_cpu_load(checkpoint_path: Path) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = PolicyValueModel(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    nonfinite = [
        name for name, value in model.state_dict().items() if not torch.isfinite(value).all()
    ]
    return {
        "epoch": int(checkpoint["epoch"]),
        "global_step": int(checkpoint["global_step"]),
        "device": str(next(model.parameters()).device),
        "nonfinite_parameter_tensors": nonfinite,
        "finite": not nonfinite,
    }


def train_cached_corpus(
    config: dict[str, Any],
    cache_dir: Path,
    checkpoint_dir: Path,
    metrics_dir: Path,
    *,
    device: str,
    epochs: int,
    resume_path: Path | None = None,
) -> dict[str, Any]:
    identity = cache_identity(ROOT, config)
    corpus = CachedDecisionCorpus(cache_dir, identity)
    vocabulary = CardVocabulary.from_json(ROOT / config["data"]["card_vocab_path"])
    model_config = _model_config(config, len(vocabulary))
    model = PolicyValueModel(**model_config).to(device)
    training = config["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    batch_size = int(training["batch_size"])
    train_decisions = int(corpus.manifest["splits"]["train"]["decision_count"])
    steps_per_epoch = max(1, math.ceil(train_decisions / batch_size))
    total_steps = max(1, steps_per_epoch * epochs)
    warmup_steps = int(total_steps * float(training["warmup_ratio"]))

    def lr_multiplier(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return max(step, 1) / warmup_steps
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_multiplier)
    use_amp = device.startswith("cuda") and bool(training["mixed_precision"])
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    checkpoint = torch.load(resume_path, map_location=device) if resume_path is not None else None
    if checkpoint is not None:
        class_counts = [int(value) for value in checkpoint["value_class_counts"]]
        class_weights = [float(value) for value in checkpoint["value_class_weights"]]
        frequency = {}
    else:
        class_counts, frequency = fit_training_baselines(corpus)
        class_weights = []
    class_prior = [count / max(sum(class_counts), 1) for count in class_counts]
    if training["value_class_weighting"]:
        class_weight_tensor = (
            torch.tensor(class_weights, dtype=torch.float32, device=device)
            if checkpoint is not None
            else _class_weights(class_counts, device)
        )
    else:
        class_weight_tensor = None
    if not class_weights:
        class_weights = (
            class_weight_tensor.detach().cpu().tolist()
            if class_weight_tensor is not None
            else [1.0] * 3
        )
    start_epoch = 0
    global_step = 0
    if checkpoint is not None:
        for key, expected in (
            ("data_schema_hash", corpus.manifest["schema_hash"]),
            ("action_contract_hash", corpus.manifest["action_contract_hash"]),
            ("card_vocabulary_hash", corpus.manifest["card_vocabulary_hash"]),
        ):
            if checkpoint.get(key) != expected:
                raise RuntimeError(f"resume checkpoint {key} mismatch")
        if checkpoint["model_config"] != model_config:
            raise RuntimeError("resume checkpoint model config mismatch")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        scaler.load_state_dict(checkpoint.get("scaler_state_dict", {}))
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint["global_step"])

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(int(training["seed"]))
    if checkpoint is None:
        untrained_validation_model = PolicyValueModel(**model_config).to(device)
        untrained_validation = evaluate_model(
            untrained_validation_model,
            corpus,
            "validation",
            batch_size,
            device,
            frequency_baseline=frequency,
            class_prior=class_prior,
        )
        del untrained_validation_model
    else:
        untrained_validation = {
            "status": "skipped_on_resume",
            "reason": "resume reuses the existing epoch history and avoids repeated baseline work",
        }
    history_path = metrics_dir / "training_history.json"
    history = (
        json.loads(history_path.read_text(encoding="utf-8"))
        if resume_path is not None and history_path.exists()
        else []
    )
    best_policy = -float("inf")
    best_value = float("inf")
    best_joint = -float("inf")
    best_epoch = -1
    patience = 0
    elapsed_before_resume = 0.0
    resume_metadata = {
        "requested": resume_path is not None,
        "checkpoint": str(resume_path) if resume_path is not None else None,
        "restored_epoch": start_epoch - 1 if resume_path is not None else None,
        "restored_global_step": global_step if resume_path is not None else None,
    }
    if resume_path is not None and (checkpoint_dir / "training_state.json").exists():
        previous_state = json.loads(
            (checkpoint_dir / "training_state.json").read_text(encoding="utf-8")
        )
        best_policy = float(previous_state.get("best_policy_accuracy", best_policy))
        best_value = float(previous_state.get("best_value_log_loss", best_value))
        best_joint = float(previous_state.get("best_joint_score", best_joint))
        best_epoch = int(previous_state.get("best_epoch", best_epoch))
        patience = int(previous_state.get("early_stopping_patience_used", 0))
        elapsed_before_resume = float(previous_state.get("elapsed_seconds", 0.0))
    alpha = float(training["joint_score_value_alpha"])
    started = time.perf_counter()
    for epoch in range(start_epoch, epochs):
        model.train()
        sums = Counter()
        batch_count = 0
        for cpu_batch in corpus.iter_batches(
            "train", batch_size, shuffle=True, seed=int(training["seed"]) + epoch
        ):
            batch = move_batch(cpu_batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(batch)
                losses = joint_policy_value_loss(
                    model.multiselect_decoder,
                    outputs,
                    batch,
                    float(training["policy_loss_weight"]),
                    float(training["value_loss_weight"]),
                    class_weight_tensor,
                )
            scaler.scale(losses["total_loss"]).backward()
            scaler.unscale_(optimizer)
            gradient = torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(training["gradient_clip_norm"])
            )
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            global_step += 1
            batch_count += 1
            for key in ("total_loss", "policy_loss", "value_loss"):
                sums[key] += float(losses[key].detach())
            sums["gradient_norm"] += float(gradient)
        validation = evaluate_model(
            model,
            corpus,
            "validation",
            batch_size,
            device,
            frequency_baseline=frequency,
            class_prior=class_prior,
            seed=int(training["seed"]),
        )
        policy_metric = float(validation["policy"]["trained_model"]["equivalence_aware_accuracy"])
        value_metric = float(validation["value"]["value_log_loss"])
        joint_score = policy_metric - alpha * value_metric
        epoch_metrics = {
            "epoch": epoch,
            "global_step": global_step,
            "learning_rate": scheduler.get_last_lr()[0],
            "train": {key: value / max(batch_count, 1) for key, value in sums.items()},
            "validation": validation,
            "joint_score": joint_score,
        }
        history.append(epoch_metrics)
        payload = _checkpoint_payload(
            model,
            optimizer,
            scheduler,
            scaler,
            epoch=epoch,
            global_step=global_step,
            model_config=model_config,
            manifest=corpus.manifest,
            metrics=epoch_metrics,
            class_counts=class_counts,
            class_weights=class_weights,
        )
        _atomic_torch_save(payload, checkpoint_dir / "last.pt")
        improved = False
        if policy_metric > best_policy:
            best_policy = policy_metric
            _atomic_torch_save(payload, checkpoint_dir / "best_policy.pt")
        if value_metric < best_value:
            best_value = value_metric
            _atomic_torch_save(payload, checkpoint_dir / "best_value.pt")
        if joint_score > best_joint:
            best_joint = joint_score
            best_epoch = epoch
            patience = 0
            improved = True
            _atomic_torch_save(payload, checkpoint_dir / "best_joint.pt")
        if not improved:
            patience += 1
        state = {
            "status": "running",
            "epoch": epoch,
            "global_step": global_step,
            "best_epoch": best_epoch,
            "best_policy_accuracy": best_policy,
            "best_value_log_loss": best_value,
            "best_joint_score": best_joint,
            "joint_score_formula": f"equivalence_aware_policy_accuracy - {alpha} * validation_value_log_loss",
            "early_stopping_patience_used": patience,
            "elapsed_seconds": elapsed_before_resume + time.perf_counter() - started,
            "resume": resume_metadata,
        }
        (checkpoint_dir / "training_state.json").write_text(
            json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        history_path.write_text(
            json.dumps(history, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        if patience >= int(training["early_stopping_patience"]):
            break

    best_checkpoint = torch.load(checkpoint_dir / "best_joint.pt", map_location=device)
    loaded_model = PolicyValueModel(**best_checkpoint["model_config"]).to(device)
    loaded_model.load_state_dict(best_checkpoint["model_state_dict"])
    verification_batch = next(corpus.iter_batches("validation", min(batch_size, 4)))
    with torch.no_grad():
        verification_output = loaded_model(move_batch(verification_batch, device))
    load_verification = {
        "policy_shape": list(verification_output["policy_logits"].shape),
        "value_shape": list(verification_output["value_logits"].shape),
        "finite": bool(
            torch.isfinite(verification_output["value_logits"]).all()
            and torch.isfinite(
                verification_output["policy_logits"][
                    move_batch(verification_batch, device)["option_mask"]
                ]
            ).all()
        ),
    }
    validation = evaluate_model(
        loaded_model,
        corpus,
        "validation",
        batch_size,
        device,
        frequency_baseline=frequency,
        class_prior=class_prior,
    )
    test = evaluate_model(
        loaded_model,
        corpus,
        "test",
        batch_size,
        device,
        frequency_baseline=frequency,
        class_prior=class_prior,
    )
    torch.manual_seed(int(training["seed"]))
    untrained_model = PolicyValueModel(**model_config).to(device)
    untrained_test = evaluate_model(
        untrained_model,
        corpus,
        "test",
        batch_size,
        device,
        frequency_baseline=frequency,
        class_prior=class_prior,
    )
    report = {
        "status": "complete",
        "cache_dir": str(cache_dir),
        "checkpoint_dir": str(checkpoint_dir),
        "best_epoch": best_epoch,
        "global_step": global_step,
        "training_seconds": elapsed_before_resume + time.perf_counter() - started,
        "resume": resume_metadata,
        "value_class_counts": class_counts,
        "value_class_weights": class_weights,
        "joint_score_alpha": alpha,
        "untrained_validation": untrained_validation,
        "validation": validation,
        "test": test,
        "untrained_test": {
            "policy": untrained_test["policy"]["trained_model"],
            "value": untrained_test["value"],
        },
        "best_joint_load_verification": load_verification,
        "cpu_checkpoint_load_verification": verify_checkpoint_cpu_load(
            checkpoint_dir / "best_joint.pt"
        ),
    }
    apply_reduced_acceptance(report, history)
    (metrics_dir / "final_evaluation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    final_state = json.loads((checkpoint_dir / "training_state.json").read_text(encoding="utf-8"))
    final_state["status"] = "complete"
    final_state["passed"] = report["passed"]
    (checkpoint_dir / "training_state.json").write_text(
        json.dumps(final_state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "decision_agent_v1/configs/policy_value_v1.yaml")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, default=ROOT / "outputs/decision_agent_v1/checkpoints")
    parser.add_argument("--metrics-dir", type=Path, default=ROOT / "outputs/decision_agent_v1/metrics/full_training")
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    device = args.device or str(config["training"]["device"])
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    torch.manual_seed(int(config["training"]["seed"]))
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(int(config["training"]["seed"]))
    report = train_cached_corpus(
        config,
        args.cache_dir,
        args.checkpoint_dir,
        args.metrics_dir,
        device=device,
        epochs=args.epochs or int(config["training"]["epochs"]),
        resume_path=args.resume,
    )
    if not report["passed"]:
        raise RuntimeError(f"training acceptance failed; see {args.metrics_dir / 'final_evaluation.json'}")
    print(args.metrics_dir / "final_evaluation.json")


if __name__ == "__main__":
    main()
