from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from .losses import joint_policy_value_loss


def state_upgrade_loss(
    model: Any,
    outputs: dict[str, torch.Tensor],
    batch: dict[str, Any],
    *,
    policy_weight: float = 1.0,
    value_weight: float = 0.5,
    archetype_weight: float = 0.05,
    next_public_weight: float = 0.05,
    value_class_weight: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    base = joint_policy_value_loss(
        model.multiselect_decoder,
        outputs,
        batch,
        policy_weight,
        value_weight,
        value_class_weight,
    )
    archetype = F.cross_entropy(outputs["archetype_logits"], batch["archetype_target"])
    mask = batch["next_public_mask"]
    if bool(mask.any()):
        next_public = F.cross_entropy(
            outputs["next_public_logits"][mask], batch["next_public_target"][mask]
        )
    else:
        next_public = outputs["next_public_logits"].sum() * 0.0
    total = base["total_loss"] + archetype_weight * archetype + next_public_weight * next_public
    return {
        **base,
        "total_loss": total,
        "archetype_loss": archetype,
        "next_public_loss": next_public,
    }
