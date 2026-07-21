from __future__ import annotations

import argparse

import torch

from decision_agent_v1.data.collate import collate_decision_samples
from decision_agent_v1.models.policy_value_model import PolicyValueModel
from decision_agent_v1.training.evaluate_policy_value import batch_metrics
from decision_agent_v1.training.losses import joint_policy_value_loss
from decision_agent_v1.training.train_policy_value import gradient_norm

from ._common import OUTPUT_ROOT, add_data_arguments, load_samples, seed_everything, stratified_tiny_samples, write_json


def _measure(model: PolicyValueModel, batch: dict[str, object]) -> tuple[dict[str, float], dict[str, object]]:
    model.eval()
    with torch.no_grad():
        outputs = model(batch)
        losses = joint_policy_value_loss(model.multiselect_decoder, outputs, batch)
        metrics = batch_metrics(model, outputs, batch)
    return (
        {key: float(losses[key]) for key in ("total_loss", "policy_loss", "value_loss")},
        metrics,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    add_data_arguments(parser, default_replays=8)
    parser.add_argument("--samples", type=int, default=32, choices=range(32, 129))
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    args = parser.parse_args()
    seed_everything()
    torch.set_num_threads(min(torch.get_num_threads(), 8))
    samples, report, vocabulary, _ = load_samples(args)
    selected = stratified_tiny_samples(samples, args.samples)
    if len(selected) < 32:
        raise RuntimeError("tiny-overfit requires at least 32 real decisions")
    outcomes = {sample.terminal_outcome.name for sample in selected}
    episode_streams = {(sample.episode_id, sample.agent_index) for sample in selected}
    if not {"WIN", "LOSS"}.issubset(outcomes) or len(episode_streams) < 2:
        raise RuntimeError("tiny batch must contain two agent episode streams and WIN/LOSS")
    batch = collate_decision_samples(selected)
    model = PolicyValueModel(len(vocabulary), dropout=0.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.0)
    initial_losses, initial_metrics = _measure(model, batch)
    trace = []
    model.train()
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        outputs = model(batch)
        losses = joint_policy_value_loss(model.multiselect_decoder, outputs, batch)
        losses["total_loss"].backward()
        optimizer.step()
        if step in {0, args.steps // 4, args.steps // 2, args.steps - 1}:
            trace.append(
                {
                    "step": step + 1,
                    "total_loss": float(losses["total_loss"].detach()),
                    "policy_loss": float(losses["policy_loss"].detach()),
                    "value_loss": float(losses["value_loss"].detach()),
                }
            )
    gradient_metrics = {
        "shared_encoder_gradient_norm": gradient_norm(model.board_encoder),
        "card_id_embedding_gradient_norm": float(
            model.card_instance_encoder.card_id_embedding.weight.grad.norm()
            if model.card_instance_encoder.card_id_embedding.weight.grad is not None
            else 0.0
        ),
        "policy_head_gradient_norm": gradient_norm(model.policy_head),
        "value_head_gradient_norm": gradient_norm(model.value_head),
    }
    final_losses, final_metrics = _measure(model, batch)
    payload = {
        "real_decision_samples": len(selected),
        "episode_streams": len(episode_streams),
        "outcomes": sorted(outcomes),
        "selection_modes": sorted({sample.selection_mode.value for sample in selected}),
        "initial": {**initial_losses, **initial_metrics},
        "final": {**final_losses, **final_metrics},
        "trace": trace,
        "gradients": gradient_metrics,
        "finite": all(torch.isfinite(parameter).all() for parameter in model.parameters()),
    }
    payload["passed"] = (
        payload["finite"]
        and final_losses["policy_loss"] < initial_losses["policy_loss"] * 0.7
        and final_losses["value_loss"] < initial_losses["value_loss"] * 0.7
        and all(value > 0 for value in gradient_metrics.values())
        and final_metrics["sequence_exact_match"] >= 0.7
        and final_metrics["win_draw_loss_accuracy"] >= 0.7
    )
    output = OUTPUT_ROOT / "tiny_overfit/tiny_overfit_metrics.json"
    write_json(output, payload)
    torch.save(
        {"model_state_dict": model.state_dict(), "vocab_size": len(vocabulary), "metrics": payload},
        OUTPUT_ROOT / "tiny_overfit/policy_value_v1_tiny.pt",
    )
    if not payload["passed"]:
        raise RuntimeError(f"tiny-overfit gate failed; see {output}")
    print(output)


if __name__ == "__main__":
    main()
