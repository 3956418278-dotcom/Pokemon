from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import torch
import yaml

from decision_agent_v1.adapters.card_vocab_adapter import CardVocabulary
from decision_agent_v1.data.v2_cache import StateUpgradeCorpus
from decision_agent_v1.models.policy_value_model import PolicyValueModel
from decision_agent_v1.models.policy_value_v2_model import PolicyValueV2Model
from decision_agent_v1.training.evaluate_policy_value import evaluate_model
from decision_agent_v1.training.state_upgrade_losses import state_upgrade_loss
from decision_agent_v1.training.train_policy_value import move_batch


ROOT = Path(__file__).resolve().parents[2]


def _model_config(config: dict[str, Any], vocab_size: int, template_count: int) -> dict[str, Any]:
    model = config["model"]
    return {
        "vocab_size": vocab_size,
        "template_count": template_count,
        "model_dim": int(model["model_dim"]),
        "dynamic_hidden_dim": int(model["dynamic_hidden_dim"]),
        "board_layers": int(model["board_layers"]),
        "board_heads": int(model["board_heads"]),
        "board_ffn_dim": int(model["board_ffn_dim"]),
        "dropout": float(model["dropout"]),
    }


def _weights(counts: list[int], device: str) -> torch.Tensor:
    total = sum(counts)
    raw = [total / (3 * count) if count else 0.0 for count in counts]
    scale = sum(value for value in raw if value) / max(sum(bool(value) for value in raw), 1)
    return torch.tensor([value / scale for value in raw], device=device)


def _loss(model: PolicyValueV2Model, outputs: dict[str, torch.Tensor], batch: dict[str, Any], config: dict[str, Any], class_weights: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
    training = config["training"]
    upgrade = config["state_upgrade"]
    return state_upgrade_loss(
        model,
        outputs,
        batch,
        policy_weight=float(training["policy_loss_weight"]),
        value_weight=float(training["value_loss_weight"]),
        archetype_weight=float(upgrade["archetype_loss_weight"]),
        next_public_weight=float(upgrade["next_public_loss_weight"]),
        value_class_weight=class_weights,
    )


@torch.no_grad()
def _auxiliary_metrics(model: PolicyValueV2Model, corpus: StateUpgradeCorpus, split: str, batch_size: int, device: str) -> dict[str, Any]:
    model.eval()
    archetype_correct = archetype_total = next_correct = next_total = 0
    for raw in corpus.iter_batches(split, batch_size):
        batch = move_batch(raw, device)
        output = model(batch)
        archetype_correct += int((output["archetype_logits"].argmax(-1) == batch["archetype_target"]).sum())
        archetype_total += len(batch["metadata"])
        mask = batch["next_public_mask"]
        next_correct += int((output["next_public_logits"].argmax(-1)[mask] == batch["next_public_target"][mask]).sum())
        next_total += int(mask.sum())
    return {
        "archetype_accuracy": archetype_correct / max(archetype_total, 1),
        "archetype_samples": archetype_total,
        "next_public_accuracy": next_correct / max(next_total, 1),
        "next_public_samples": next_total,
    }


def _limited_batches(corpus: StateUpgradeCorpus, split: str, batch_size: int, limit: int, **kwargs: Any) -> Iterable[dict[str, Any]]:
    for index, batch in enumerate(corpus.iter_batches(split, batch_size, **kwargs)):
        if index >= limit:
            break
        yield batch


def _preflight(model_config: dict[str, Any], v1: dict[str, Any], corpus: StateUpgradeCorpus, config: dict[str, Any], device: str) -> dict[str, Any]:
    training = config["training"]
    batch_size = min(64, int(training["batch_size"]))
    real = next(corpus.iter_batches("train", batch_size))
    model = PolicyValueV2Model(**{**model_config, "dropout": 0.0}).to(device)
    model.load_v1_state_dict(v1["model_state_dict"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.0)
    batch = move_batch(real, device)
    model.train()
    output = model(batch)
    initial = _loss(model, output, batch, config)
    initial_values = {key: float(initial[key].detach()) for key in ("total_loss", "policy_loss", "value_loss", "archetype_loss", "next_public_loss")}
    initial["total_loss"].backward()
    gradients = {
        "self_deck_encoder": float(model.board_encoder.self_summary[0].weight.grad.norm()),
        "belief_encoder": float(model.board_encoder.belief_summary[0].weight.grad.norm()),
        "ledger_encoder": float(model.board_encoder.ledger_summary[0].weight.grad.norm()),
        "event_encoder": float(model.board_encoder.event_numeric.weight.grad.norm()),
        "archetype_head": float(model.archetype_head.weight.grad.norm()),
        "next_public_head": float(model.next_public_head.weight.grad.norm()),
    }
    optimizer.zero_grad(set_to_none=True)
    trace = []
    for step in range(100):
        output = model(batch)
        losses = _loss(model, output, batch, config)
        optimizer.zero_grad(set_to_none=True)
        losses["total_loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step in {0, 24, 49, 99}:
            trace.append({key: float(losses[key].detach()) for key in ("total_loss", "policy_loss", "value_loss", "archetype_loss", "next_public_loss")})
    model.eval()
    with torch.no_grad():
        final_losses = _loss(model, model(batch), batch, config)
    final_values = {key: float(final_losses[key]) for key in initial_values}
    # Short validation is intentionally bounded and is a health check, not a
    # model-selection result.
    validation_losses = Counter()
    validation_batches = 0
    with torch.no_grad():
        for raw in _limited_batches(corpus, "validation", batch_size, 8):
            validation = move_batch(raw, device)
            losses = _loss(model, model(validation), validation, config)
            for key in initial_values:
                validation_losses[key] += float(losses[key])
            validation_batches += 1
    finite = all(torch.isfinite(value).all() for value in model.parameters()) and all(math.isfinite(value) for value in gradients.values())
    return {
        "real_batch_forward_backward": {"batch_size": len(real["metadata"]), "finite": finite, "gradients": gradients},
        "tiny_overfit": {"steps": 100, "initial": initial_values, "final": final_values, "trace": trace, "passed": final_values["total_loss"] < initial_values["total_loss"] * 0.75 and all(value > 0 for value in gradients.values())},
        "short_validation": {"batches": validation_batches, "metrics": {key: value / max(validation_batches, 1) for key, value in validation_losses.items()}, "finite": all(math.isfinite(value) for value in validation_losses.values())},
    }


def _benchmark(model: torch.nn.Module, batch: dict[str, Any], device: str, runs: int = 50) -> float:
    model.eval()
    one = {key: value[:1] if isinstance(value, torch.Tensor) else value[:1] if isinstance(value, list) else value for key, value in batch.items()}
    one = move_batch(one, device)
    with torch.no_grad():
        for _ in range(5):
            model(one)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(runs):
            model(one)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
    return (time.perf_counter() - started) * 1000.0 / runs


def _representative_batch(corpus: StateUpgradeCorpus) -> dict[str, Any]:
    best_batch: dict[str, Any] | None = None
    best_row = 0
    best_score = -1
    for batch_index, batch in enumerate(corpus.iter_batches("validation", 64)):
        scores = batch["card_mask"].sum(dim=1) + batch["recent_event_mask"].sum(dim=1)
        score, row = scores.max(dim=0)
        if int(score) > best_score:
            best_score = int(score)
            best_batch = batch
            best_row = int(row)
        if batch_index >= 7:
            break
    if best_batch is None:
        raise RuntimeError("validation corpus is empty")
    return {
        key: value[best_row : best_row + 1]
        if isinstance(value, torch.Tensor)
        else value[best_row : best_row + 1]
        if isinstance(value, list)
        else value
        for key, value in best_batch.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "decision_agent_v1/configs/policy_value_v2.yaml")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    device = args.device or str(config["training"]["device"])
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.manual_seed(int(config["training"]["seed"]))
    corpus = StateUpgradeCorpus(args.cache_dir)
    vocabulary = CardVocabulary.from_json(ROOT / config["data"]["card_vocab_path"])
    prior = json.loads((ROOT / config["data"]["belief_output_dir"] / "deck_template_metadata.json").read_text(encoding="utf-8"))
    model_config = _model_config(config, len(vocabulary), int(prior["template_count"]))
    v1_path = ROOT / config["data"]["base_checkpoint"]
    v1 = torch.load(v1_path, map_location="cpu")
    output_root = ROOT / config["data"]["output_root"]
    checkpoint_dir = output_root / "checkpoints/state_upgrade"
    metrics_dir = output_root / "metrics/state_upgrade"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    preflight = _preflight(model_config, v1, corpus, config, device)
    if not (preflight["real_batch_forward_backward"]["finite"] and preflight["tiny_overfit"]["passed"] and preflight["short_validation"]["finite"]):
        raise RuntimeError("V2 preflight failed")
    (metrics_dir / "preflight.json").write_text(json.dumps(preflight, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    model = PolicyValueV2Model(**model_config).to(device)
    missing, unexpected = model.load_v1_state_dict(v1["model_state_dict"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["training"]["learning_rate"]), weight_decay=float(config["training"]["weight_decay"]))
    class_counts = corpus.class_counts("train")
    class_weights = _weights(class_counts, device)
    epochs = args.epochs or int(config["training"]["epochs"])
    history = []
    best_score = -float("inf")
    best_epoch = -1
    batch_size = int(config["training"]["batch_size"])
    started = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        sums = Counter()
        batches = 0
        for raw in corpus.iter_batches("train", batch_size, shuffle=True, seed=int(config["training"]["seed"]) + epoch):
            batch = move_batch(raw, device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch)
            losses = _loss(model, outputs, batch, config, class_weights)
            losses["total_loss"].backward()
            gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["training"]["gradient_clip_norm"]))
            optimizer.step()
            for key in ("total_loss", "policy_loss", "value_loss", "archetype_loss", "next_public_loss"):
                sums[key] += float(losses[key].detach())
            sums["gradient_norm"] += float(gradient)
            batches += 1
        validation = evaluate_model(model, corpus, "validation", batch_size, device)
        aux = _auxiliary_metrics(model, corpus, "validation", batch_size, device)
        score = float(validation["policy"]["trained_model"]["equivalence_aware_accuracy"]) - 0.25 * float(validation["value"]["value_log_loss"])
        row = {"epoch": epoch, "train": {key: value / max(batches, 1) for key, value in sums.items()}, "validation": validation, "auxiliary": aux, "joint_score": score}
        history.append(row)
        payload = {
            "schema_version": "policy_value_v2_checkpoint",
            "epoch": epoch,
            "model_config": model_config,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "data_schema_hash": corpus.manifest["schema_hash"],
            "belief_template_hash": corpus.manifest["belief_template_hash"],
            "base_checkpoint": str(v1_path),
            "metrics": row,
        }
        temporary = checkpoint_dir / "last.pt.incomplete"
        torch.save(payload, temporary)
        temporary.replace(checkpoint_dir / "last.pt")
        if score > best_score:
            best_score, best_epoch = score, epoch
            temporary = checkpoint_dir / "best_joint_v2.pt.incomplete"
            torch.save(payload, temporary)
            temporary.replace(checkpoint_dir / "best_joint_v2.pt")
        (metrics_dir / "training_history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    best = torch.load(checkpoint_dir / "best_joint_v2.pt", map_location=device)
    loaded = PolicyValueV2Model(**best["model_config"]).to(device)
    loaded.load_state_dict(best["model_state_dict"])
    validation = evaluate_model(loaded, corpus, "validation", batch_size, device)
    test = evaluate_model(loaded, corpus, "test", batch_size, device)
    auxiliary = {split: _auxiliary_metrics(loaded, corpus, split, batch_size, device) for split in ("validation", "test")}
    v1_report = json.loads((output_root / "metrics/full_training/final_evaluation.json").read_text(encoding="utf-8"))
    first_batch = _representative_batch(corpus)
    v1_model = PolicyValueModel(**v1["model_config"]).to(device)
    v1_model.load_state_dict(v1["model_state_dict"])
    v1_latency = _benchmark(v1_model, first_batch, device)
    v2_latency = _benchmark(loaded, first_batch, device)
    validation_policy_improved = validation["policy"]["trained_model"]["equivalence_aware_accuracy"] > v1_report["validation"]["policy"]["trained_model"]["equivalence_aware_accuracy"]
    validation_calibration_improved = validation["value"]["value_log_loss"] < v1_report["validation"]["value"]["value_log_loss"]
    default_v2 = bool(validation_policy_improved or validation_calibration_improved)
    result = {
        "status": "complete",
        "schema_version": "state_upgrade_results_v1",
        "reduction_applied": {"comparison": "V1_vs_complete_V2", "omitted_full_retrains": ["B", "C", "D"], "markdown_reports_generated": False},
        "cache_dir": str(args.cache_dir),
        "base_checkpoint": str(v1_path),
        "v2_checkpoint": str(checkpoint_dir / "best_joint_v2.pt"),
        "best_epoch": best_epoch,
        "training_seconds": time.perf_counter() - started,
        "preflight": preflight,
        "v1": {"validation": v1_report["validation"], "test": v1_report["test"]},
        "v2": {"validation": validation, "test": test, "auxiliary": auxiliary},
        "comparison": {
            "validation_policy_improved": validation_policy_improved,
            "validation_value_log_loss_improved": validation_calibration_improved,
            "v2_default": default_v2,
        },
        "model_size": {
            "v1_parameters": sum(value.numel() for value in v1_model.parameters()),
            "v2_parameters": sum(value.numel() for value in loaded.parameters()),
        },
        "inference": {
            "v1_fixed_prefix_tokens": 2,
            "v2_fixed_prefix_and_event_slots": 20,
            "visible_card_slots_in_sample": int(first_batch["card_mask"][0].sum()),
            "public_event_tokens_in_sample": int(first_batch["recent_event_mask"][0].sum()),
            "v1_latency_ms": v1_latency,
            "v2_latency_ms": v2_latency,
        },
        "v1_initialization": {"missing_v2_parameters": missing, "unexpected_parameters": unexpected},
        "belief_public_only": True,
        "passed": bool(preflight["tiny_overfit"]["passed"] and torch.isfinite(torch.tensor([validation["policy_loss"], validation["value_loss"], test["policy_loss"], test["value_loss"]])).all()),
    }
    target = output_root / "metrics/state_upgrade_results.json"
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    # Required public path; this is a copy, not the V1 default checkpoint.
    public_checkpoint = output_root / "checkpoints/best_joint_v2.pt"
    public_checkpoint.write_bytes((checkpoint_dir / "best_joint_v2.pt").read_bytes())
    print(target)


if __name__ == "__main__":
    main()
