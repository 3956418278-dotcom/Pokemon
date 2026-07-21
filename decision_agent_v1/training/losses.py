from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from decision_agent_v1.models.multiselect_decoder import MultiSelectDecoder


def autoregressive_policy_loss(
    decoder: MultiSelectDecoder,
    outputs: dict[str, torch.Tensor],
    batch: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    base_logits = outputs["policy_logits"]
    option_embeddings = outputs["option_embeddings"]
    option_mask = batch["option_mask"]
    batch_size, option_count = option_mask.shape
    max_steps = max((len(values) for values in batch["target_sequences"]), default=0)
    if max_steps == 0:
        zero = outputs["option_embeddings"].sum() * 0.0
        return zero, torch.empty(0, device=zero.device)
    target_indices = torch.full(
        (batch_size, max_steps), -1, dtype=torch.long, device=base_logits.device
    )
    target_groups = torch.full_like(target_indices, -1)
    target_step_mask = torch.zeros(
        batch_size, max_steps, dtype=torch.bool, device=base_logits.device
    )
    for row, (indices, groups) in enumerate(
        zip(batch["target_sequences"], batch["target_equivalence_groups"])
    ):
        length = min(len(indices), len(groups))
        if length:
            target_indices[row, :length] = torch.tensor(
                indices[:length], dtype=torch.long, device=base_logits.device
            )
            target_groups[row, :length] = torch.tensor(
                groups[:length], dtype=torch.long, device=base_logits.device
            )
            target_step_mask[row, :length] = True
    target_step_mask &= batch["policy_sample_mask"].unsqueeze(1)
    selected = torch.zeros_like(option_mask)
    per_sample_sum = torch.zeros(batch_size, device=base_logits.device)
    per_sample_count = torch.zeros(batch_size, device=base_logits.device)
    rows = torch.arange(batch_size, device=base_logits.device)
    equivalence = batch["option_equivalence_group"]
    for step in range(max_steps):
        active = target_step_mask[:, step]
        if not bool(active.any()):
            continue
        logits = decoder.step_logits(base_logits, option_embeddings, selected, option_mask)
        group_mask = (
            equivalence == target_groups[:, step].unsqueeze(1)
        ) & option_mask & ~selected
        # Never evaluate logsumexp on inactive rows or rows without a target
        # equivalence member.  Computing (-inf - -inf) and masking it later can
        # still backpropagate NaN through LogsumexpBackward (0 * NaN).
        candidate = active & group_mask.any(dim=1) & (option_mask & ~selected).any(dim=1)
        candidate_rows = torch.nonzero(candidate, as_tuple=False).flatten()
        valid = torch.zeros(batch_size, dtype=torch.bool, device=base_logits.device)
        if candidate_rows.numel():
            candidate_logits = logits.index_select(0, candidate_rows)
            candidate_group_mask = group_mask.index_select(0, candidate_rows)
            numerator = torch.logsumexp(
                candidate_logits.masked_fill(~candidate_group_mask, float("-inf")), dim=1
            )
            denominator = torch.logsumexp(candidate_logits, dim=1)
            finite = torch.isfinite(numerator) & torch.isfinite(denominator)
            valid_rows = candidate_rows[finite]
            if valid_rows.numel():
                step_loss = -(numerator[finite] - denominator[finite])
                per_sample_sum = per_sample_sum.index_add(0, valid_rows, step_loss)
                per_sample_count = per_sample_count.index_add(
                    0, valid_rows, torch.ones_like(step_loss)
                )
                valid[valid_rows] = True
        target = target_indices[:, step]
        update = valid & (target >= 0) & (target < option_count)
        selected = selected.clone()
        selected[rows[update], target[update]] = True
    supervised = per_sample_count > 0
    if not bool(supervised.any()):
        zero = outputs["option_embeddings"].sum() * 0.0
        return zero, torch.empty(0, device=zero.device)
    values = per_sample_sum[supervised] / per_sample_count[supervised]
    return values.mean(), values


def weighted_value_loss(
    value_logits: torch.Tensor,
    batch: dict[str, Any],
    class_weight: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    per_sample = F.cross_entropy(
        value_logits, batch["value_target"], reduction="none", weight=class_weight
    )
    weight = batch["episode_weight"]
    return (per_sample * weight).sum() / weight.sum().clamp(min=1e-8), per_sample


def joint_policy_value_loss(
    decoder: MultiSelectDecoder,
    outputs: dict[str, torch.Tensor],
    batch: dict[str, Any],
    policy_weight: float = 1.0,
    value_weight: float = 0.5,
    value_class_weight: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    policy, policy_per_sample = autoregressive_policy_loss(decoder, outputs, batch)
    value, value_per_sample = weighted_value_loss(
        outputs["value_logits"], batch, value_class_weight
    )
    return {
        "total_loss": policy_weight * policy + value_weight * value,
        "policy_loss": policy,
        "value_loss": value,
        "policy_loss_per_sample": policy_per_sample,
        "value_loss_per_sample": value_per_sample,
    }
