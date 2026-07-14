from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .card_encoder import infer_head_sizes


def _zero(reference: torch.Tensor) -> torch.Tensor:
    return reference.sum() * 0.0


def _accuracy(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    valid = mask.bool()
    if not bool(valid.any().item()):
        return 0.0
    return float((logits[valid].argmax(dim=-1) == target[valid]).float().mean().detach().cpu())


class CardFieldRecoveryHeads(nn.Module):
    """Recover grouped card fields; exact card name is intentionally absent."""

    def __init__(self, schema: dict[str, Any], embedding_dim: int = 128) -> None:
        super().__init__()
        sizes = infer_head_sizes(schema)
        self.fields = tuple(schema["card_field_slots"])
        self.field_heads = nn.ModuleDict(
            {field: nn.Linear(embedding_dim, size) for field, size in sizes["card_fields"].items()}
        )
        energy_count_classes = int(sizes["profile_energy_count"])
        self.energy_count_heads = nn.ModuleList(
            [nn.Linear(embedding_dim, energy_count_classes) for _ in schema["energy_symbols"]]
        )

    def forward(self, card_summary: torch.Tensor) -> dict[str, Any]:
        return {
            "fields": {field: head(card_summary) for field, head in self.field_heads.items()},
            "energy_counts": torch.stack([head(card_summary) for head in self.energy_count_heads], dim=1),
        }


def card_field_recovery_loss(
    predictions: dict[str, Any],
    batch: dict[str, torch.Tensor],
    field_masks: dict[str, torch.Tensor],
    field_slots: tuple[str, ...],
) -> tuple[torch.Tensor, dict[str, float], int]:
    field_indices = {field: index for index, field in enumerate(field_slots)}
    applicability = batch["card_field_applicability_mask"].bool()
    total_sum: torch.Tensor | None = None
    total_count = 0
    metrics: dict[str, float] = {}
    for field, logits in predictions["fields"].items():
        mask = field_masks.get(field)
        if not isinstance(mask, torch.Tensor):
            continue
        valid = mask.bool() & applicability[:, field_indices[field]]
        if not bool(valid.any().item()):
            continue
        target = batch["card_field_value_ids"][:, field_indices[field]].long()
        loss_sum = F.cross_entropy(logits[valid], target[valid], reduction="sum")
        total_sum = loss_sum if total_sum is None else total_sum + loss_sum
        count = int(valid.sum().item())
        total_count += count
        metrics[f"{field}_accuracy"] = _accuracy(logits, target, valid)
        metrics[f"{field}_count"] = float(count)

    energy_mask = field_masks.get("energy_printed_type")
    if isinstance(energy_mask, torch.Tensor):
        valid_cards = energy_mask.bool() & applicability[:, field_indices["energy_printed_type"]]
        energy_logits = predictions["energy_counts"]
        energy_target = batch["provided_energy_count_ids"].long()
        for symbol_index in range(energy_logits.shape[1]):
            if not bool(valid_cards.any().item()):
                continue
            logits = energy_logits[:, symbol_index]
            target = energy_target[:, symbol_index]
            loss_sum = F.cross_entropy(logits[valid_cards], target[valid_cards], reduction="sum")
            total_sum = loss_sum if total_sum is None else total_sum + loss_sum
            count = int(valid_cards.sum().item())
            total_count += count
            metrics[f"energy_symbol_{symbol_index}_accuracy"] = _accuracy(logits, target, valid_cards)
            metrics[f"energy_symbol_{symbol_index}_count"] = float(count)

    reference = predictions["energy_counts"]
    loss = _zero(reference) if total_sum is None else total_sum / max(1, total_count)
    metrics["valid_prediction_count"] = float(total_count)
    metrics["loss"] = float(loss.detach().cpu())
    return loss, metrics, total_count


class DetailAttributeHeads(nn.Module):
    """Recover discrete cost counts and damage from complete independent details."""

    def __init__(self, schema: dict[str, Any], detail_dim: int = 128) -> None:
        super().__init__()
        sizes = infer_head_sizes(schema)
        self.cost_heads = nn.ModuleList(
            [nn.Linear(detail_dim, int(sizes["attack_energy_count"])) for _ in schema["attack_energy_symbols"]]
        )
        self.damage_value = nn.Linear(detail_dim, int(sizes["damage_value"]))
        self.damage_mode = nn.Linear(detail_dim, int(sizes["damage_mode"]))

    def forward(self, detail_tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "cost_counts": torch.stack([head(detail_tokens) for head in self.cost_heads], dim=2),
            "damage_value": self.damage_value(detail_tokens),
            "damage_mode": self.damage_mode(detail_tokens),
        }


def detail_attribute_loss(
    predictions: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, float], int]:
    valid = batch["detail_mask"].bool() & batch["attack_mask"].bool()
    total_sum: torch.Tensor | None = None
    total_count = 0
    metrics: dict[str, float] = {}
    cost_logits = predictions["cost_counts"]
    cost_target = batch["attack_energy_count_ids"].long()
    for symbol_index in range(cost_logits.shape[2]):
        logits = cost_logits[:, :, symbol_index]
        target = cost_target[:, :, symbol_index]
        if bool(valid.any().item()):
            loss_sum = F.cross_entropy(logits[valid], target[valid], reduction="sum")
            total_sum = loss_sum if total_sum is None else total_sum + loss_sum
            count = int(valid.sum().item())
            total_count += count
            metrics[f"cost_symbol_{symbol_index}_accuracy"] = _accuracy(logits, target, valid)

    for name, target_name in (
        ("damage_value", "attack_damage_value_ids"),
        ("damage_mode", "attack_damage_mode_ids"),
    ):
        logits = predictions[name]
        target = batch[target_name].long()
        if bool(valid.any().item()):
            loss_sum = F.cross_entropy(logits[valid], target[valid], reduction="sum")
            total_sum = loss_sum if total_sum is None else total_sum + loss_sum
            count = int(valid.sum().item())
            total_count += count
            metrics[f"{name}_accuracy"] = _accuracy(logits, target, valid)

    reference = predictions["damage_mode"]
    loss = _zero(reference) if total_sum is None else total_sum / max(1, total_count)
    if bool(valid.any().item()):
        exact = torch.ones_like(valid)
        for symbol_index in range(cost_logits.shape[2]):
            exact &= cost_logits[:, :, symbol_index].argmax(dim=-1) == cost_target[:, :, symbol_index]
        metrics["cost_exact_vector_accuracy"] = float(exact[valid].float().mean().detach().cpu())
        nonzero = valid.unsqueeze(-1) & (cost_target > 0)
        if bool(nonzero.any().item()):
            predicted = cost_logits.argmax(dim=-1)
            metrics["cost_nonzero_accuracy"] = float((predicted[nonzero] == cost_target[nonzero]).float().mean().detach().cpu())
    metrics["valid_prediction_count"] = float(total_count)
    metrics["loss"] = float(loss.detach().cpu())
    return loss, metrics, total_count


class TextMLMHead(nn.Module):
    def __init__(self, schema: dict[str, Any], hidden_dim: int = 128) -> None:
        super().__init__()
        self.projection = nn.Linear(hidden_dim, int(infer_head_sizes(schema)["text"]))

    def forward(self, token_states: torch.Tensor) -> torch.Tensor:
        return self.projection(token_states)


def text_mlm_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float], int]:
    valid = mask.bool()
    count = int(valid.sum().item())
    if count == 0:
        loss = _zero(logits)
        return loss, {"accuracy": 0.0, "perplexity": 1.0, "valid_prediction_count": 0.0, "loss": 0.0}, 0
    loss = F.cross_entropy(logits[valid], labels.long()[valid])
    return loss, {
        "accuracy": _accuracy(logits, labels.long(), valid),
        "perplexity": float(torch.exp(loss.detach().clamp(max=20)).cpu()),
        "valid_prediction_count": float(count),
        "loss": float(loss.detach().cpu()),
    }, count


class StructureReferenceHeads(nn.Module):
    """Recover a fully masked structure-reference unit."""

    def __init__(self, schema: dict[str, Any], hidden_dim: int = 128) -> None:
        super().__init__()
        sizes = infer_head_sizes(schema)
        self.reference_type = nn.Linear(hidden_dim, int(sizes["reference_type"]))
        self.reference_values = nn.ModuleDict(
            {field: nn.Linear(hidden_dim, int(size)) for field, size in sizes["reference_values"].items()}
        )

    def forward(self, token_states: torch.Tensor) -> dict[str, Any]:
        return {
            "reference_type": self.reference_type(token_states),
            "reference_values": {
                field: head(token_states) for field, head in self.reference_values.items()
            },
        }


def structure_reference_loss(
    predictions: dict[str, Any],
    type_labels: torch.Tensor,
    field_labels: torch.Tensor,
    value_labels: torch.Tensor,
    mask: torch.Tensor,
    reference_fields: tuple[str, ...],
) -> tuple[torch.Tensor, dict[str, float], int]:
    valid = mask.bool()
    total_sum: torch.Tensor | None = None
    total_count = 0
    metrics: dict[str, float] = {}
    type_logits = predictions["reference_type"]
    if bool(valid.any().item()):
        type_sum = F.cross_entropy(type_logits[valid], type_labels.long()[valid], reduction="sum")
        total_sum = type_sum
        total_count = int(valid.sum().item())
        metrics["reference_type_accuracy"] = _accuracy(type_logits, type_labels.long(), valid)
    for field_index, field in enumerate(reference_fields, start=1):
        field_valid = valid & (field_labels.long() == field_index)
        if not bool(field_valid.any().item()):
            continue
        logits = predictions["reference_values"][field]
        loss_sum = F.cross_entropy(logits[field_valid], value_labels.long()[field_valid], reduction="sum")
        total_sum = loss_sum if total_sum is None else total_sum + loss_sum
        count = int(field_valid.sum().item())
        total_count += count
        metrics[f"{field}_accuracy"] = _accuracy(logits, value_labels.long(), field_valid)
        metrics[f"{field}_count"] = float(count)
    loss = _zero(type_logits) if total_sum is None else total_sum / max(1, total_count)
    metrics["valid_prediction_count"] = float(total_count)
    metrics["loss"] = float(loss.detach().cpu())
    return loss, metrics, total_count


class CardDetailMatchingHead(nn.Module):
    def __init__(self, embedding_dim: int = 128, temperature: float = 0.07) -> None:
        super().__init__()
        self.card_projection = nn.Linear(embedding_dim, embedding_dim)
        self.detail_projection = nn.Linear(embedding_dim, embedding_dim)
        self.temperature = float(temperature)

    def forward(self, card_summaries: torch.Tensor, held_out_details: torch.Tensor) -> torch.Tensor:
        cards = F.normalize(self.card_projection(card_summaries), dim=-1)
        details = F.normalize(self.detail_projection(held_out_details), dim=-1)
        return details @ cards.t() / max(self.temperature, 1e-6)


def masked_info_nce_loss(
    logits: torch.Tensor,
    valid_examples: torch.Tensor,
    negative_mask: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float], int]:
    if logits.dim() != 2 or logits.shape[0] != logits.shape[1]:
        raise ValueError("ownership InfoNCE logits must be square [B, B]")
    batch_size = logits.shape[0]
    valid = valid_examples.bool()
    count = int(valid.sum().item())
    if count == 0:
        loss = _zero(logits)
        return loss, {"recall_at_1": 0.0, "recall_at_5": 0.0, "mrr": 0.0, "valid_prediction_count": 0.0, "loss": 0.0}, 0
    diagonal = torch.eye(batch_size, dtype=torch.bool, device=logits.device)
    allowed = negative_mask.bool() | diagonal
    allowed &= valid[:, None] & valid[None, :]
    masked_logits = logits.masked_fill(~allowed, torch.finfo(logits.dtype).min)
    rows = masked_logits[valid]
    targets = torch.arange(batch_size, device=logits.device)[valid]
    loss = F.cross_entropy(rows, targets)
    order = rows.argsort(dim=1, descending=True)
    ranks = (order == targets.unsqueeze(1)).nonzero(as_tuple=False)[:, 1] + 1
    metrics = {
        "recall_at_1": float((ranks <= 1).float().mean().detach().cpu()),
        "recall_at_5": float((ranks <= 5).float().mean().detach().cpu()),
        "mrr": float((1.0 / ranks.float()).mean().detach().cpu()),
        "valid_prediction_count": float(count),
        "loss": float(loss.detach().cpu()),
    }
    return loss, metrics, count
