from __future__ import annotations

import argparse
import math

import torch

from decision_agent_v1.data.collate import collate_decision_samples
from decision_agent_v1.models.policy_value_model import PolicyValueModel
from decision_agent_v1.training.losses import joint_policy_value_loss
from decision_agent_v1.training.train_policy_value import gradient_norm

from ._common import OUTPUT_ROOT, add_data_arguments, load_samples, seed_everything, stratified_tiny_samples, write_json


def _module_norms(model: PolicyValueModel) -> dict[str, float]:
    return {
        "shared_encoder_gradient_norm": gradient_norm(model.board_encoder),
        "card_id_embedding_gradient_norm": float(
            model.card_instance_encoder.card_id_embedding.weight.grad.norm()
            if model.card_instance_encoder.card_id_embedding.weight.grad is not None
            else 0.0
        ),
        "policy_head_gradient_norm": gradient_norm(model.policy_head),
        "value_head_gradient_norm": gradient_norm(model.value_head),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    add_data_arguments(parser, default_replays=4)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    seed_everything()
    samples, report, vocabulary, _ = load_samples(args)
    selected = stratified_tiny_samples(samples, args.batch_size)
    batch = collate_decision_samples(selected)
    model = PolicyValueModel(len(vocabulary), dropout=0.0)

    model.zero_grad(set_to_none=True)
    outputs = model(batch)
    losses = joint_policy_value_loss(model.multiselect_decoder, outputs, batch)
    losses["policy_loss"].backward()
    policy_gradient = gradient_norm(model.board_encoder)

    model.zero_grad(set_to_none=True)
    outputs = model(batch)
    losses = joint_policy_value_loss(model.multiselect_decoder, outputs, batch)
    losses["value_loss"].backward()
    value_gradient = gradient_norm(model.board_encoder)

    model.zero_grad(set_to_none=True)
    outputs = model(batch)
    losses = joint_policy_value_loss(model.multiselect_decoder, outputs, batch)
    losses["total_loss"].backward()
    gradients = _module_norms(model)
    valid_policy = outputs["policy_logits"][batch["option_mask"]]
    tensors = [
        outputs["value_logits"],
        outputs["board_embedding"],
        outputs["contextual_card_tokens"],
        valid_policy,
        *[losses[key].reshape(1) for key in ("policy_loss", "value_loss", "total_loss")],
    ]
    finite = all(bool(torch.isfinite(tensor).all()) for tensor in tensors)
    payload = {
        "source": "real ReplayDecisionDataset samples",
        "episodes": report.episodes,
        "batch_size": len(selected),
        "tensor_shapes": {
            "cards": list(batch["card_index"].shape),
            "options": list(batch["option_type"].shape),
            "policy_logits": list(outputs["policy_logits"].shape),
            "value_logits": list(outputs["value_logits"].shape),
            "board_embedding": list(outputs["board_embedding"].shape),
            "contextual_card_tokens": list(outputs["contextual_card_tokens"].shape),
        },
        "policy_loss": float(losses["policy_loss"].detach()),
        "value_loss": float(losses["value_loss"].detach()),
        "total_loss": float(losses["total_loss"].detach()),
        "board_encoder_policy_only_gradient_norm": policy_gradient,
        "board_encoder_value_only_gradient_norm": value_gradient,
        **gradients,
        "finite": finite,
        "passed": finite
        and policy_gradient > 0
        and value_gradient > 0
        and all(value > 0 for value in gradients.values()),
    }
    if not payload["passed"] or not all(math.isfinite(float(value)) for value in gradients.values()):
        raise RuntimeError(f"smoke backward gate failed: {payload}")
    output = OUTPUT_ROOT / "metrics/smoke_pipeline.json"
    write_json(output, payload)
    print(output)


if __name__ == "__main__":
    main()
