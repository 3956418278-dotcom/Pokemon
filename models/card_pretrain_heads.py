from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .card_encoder import infer_head_sizes


class MaskedFieldHeads(nn.Module):
    def __init__(self, schema: dict[str, Any], embedding_dim: int = 128) -> None:
        super().__init__()
        sizes = infer_head_sizes(schema)
        self.card_type = nn.Linear(embedding_dim, sizes["card_type"])
        self.pokemon_type = nn.Linear(embedding_dim, sizes["pokemon_type"])
        self.stage = nn.Linear(embedding_dim, sizes["stage"])
        self.trainer_type = nn.Linear(embedding_dim, sizes["trainer_type"])
        self.energy_type = nn.Linear(embedding_dim, sizes["energy_type"])
        self.hp = nn.Linear(embedding_dim, 1)
        self.retreat = nn.Linear(embedding_dim, 6)
        self.energy_cost = nn.Linear(embedding_dim, 1)
        self.damage = nn.Linear(embedding_dim, 1)

    def forward(self, embedding: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "card_type": self.card_type(embedding),
            "pokemon_type": self.pokemon_type(embedding),
            "stage": self.stage(embedding),
            "trainer_type": self.trainer_type(embedding),
            "energy_type": self.energy_type(embedding),
            "hp": self.hp(embedding).squeeze(-1),
            "retreat": self.retreat(embedding),
            "energy_cost": self.energy_cost(embedding).squeeze(-1),
            "damage": self.damage(embedding).squeeze(-1),
        }


class RelationHead(nn.Module):
    def __init__(self, embedding_dim: int = 128, relation_types: int = 4) -> None:
        super().__init__()
        self.relation_type = nn.Embedding(relation_types, 16)
        self.net = nn.Sequential(
            nn.Linear(embedding_dim * 4 + 16, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, left: torch.Tensor, right: torch.Tensor, relation_type: torch.Tensor) -> torch.Tensor:
        rel = self.relation_type(relation_type)
        features = torch.cat([left, right, torch.abs(left - right), left * right, rel], dim=-1)
        return self.net(features).squeeze(-1)


class AttackOwnerHead(nn.Module):
    def __init__(self, embedding_dim: int = 128, attack_feature_dim: int = 16) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim + attack_feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, card_embedding: torch.Tensor, attack_features: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([card_embedding, attack_features], dim=-1)).squeeze(-1)


def info_nce_loss(text_embedding: torch.Tensor, structure_embedding: torch.Tensor, temperature: float) -> tuple[torch.Tensor, dict[str, float]]:
    logits = text_embedding @ structure_embedding.t() / max(temperature, 1e-6)
    target = torch.arange(logits.size(0), device=logits.device)
    loss_ts = F.cross_entropy(logits, target)
    loss_st = F.cross_entropy(logits.t(), target)
    top1_ts = (logits.argmax(dim=1) == target).float().mean().item()
    top1_st = (logits.argmax(dim=0) == target).float().mean().item()
    return (loss_ts + loss_st) * 0.5, {
        "text_to_structure_top1": top1_ts,
        "structure_to_text_top1": top1_st,
    }


def safe_cross_entropy(logits: torch.Tensor, target: torch.Tensor, ignore_index: int | None = None) -> torch.Tensor:
    if ignore_index is not None and bool((target != ignore_index).sum().item() == 0):
        return logits.sum() * 0.0
    if ignore_index is None:
        return F.cross_entropy(logits, target)
    return F.cross_entropy(logits, target, ignore_index=ignore_index)


def masked_field_loss(pred: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
    losses = {
        "card_type": safe_cross_entropy(pred["card_type"], batch["card_type_target"]),
        "pokemon_type": safe_cross_entropy(pred["pokemon_type"], batch["pokemon_type_target"], ignore_index=-100),
        "stage": safe_cross_entropy(pred["stage"], batch["stage_target"], ignore_index=-100),
        "trainer_type": safe_cross_entropy(pred["trainer_type"], batch["trainer_type_target"], ignore_index=-100),
        "energy_type": safe_cross_entropy(pred["energy_type"], batch["energy_type_target"], ignore_index=-100),
        "hp": F.smooth_l1_loss(pred["hp"], batch["hp_target"]),
        "retreat": F.cross_entropy(pred["retreat"], batch["retreat_target"]),
        "energy_cost": F.smooth_l1_loss(pred["energy_cost"], batch["energy_cost_target"]),
        "damage": F.smooth_l1_loss(pred["damage"], batch["damage_target"]),
    }
    total = sum(losses.values()) / len(losses)
    metrics: dict[str, float] = {f"mask_{name}_loss": float(value.detach().cpu()) for name, value in losses.items()}
    metrics["mask_card_type_acc"] = float((pred["card_type"].argmax(dim=1) == batch["card_type_target"]).float().mean().detach().cpu())
    metrics["mask_retreat_acc"] = float((pred["retreat"].argmax(dim=1) == batch["retreat_target"]).float().mean().detach().cpu())
    valid_energy = batch["energy_type_target"] != -100
    if bool(valid_energy.any().item()):
        metrics["mask_energy_type_acc"] = float(
            (pred["energy_type"].argmax(dim=1)[valid_energy] == batch["energy_type_target"][valid_energy])
            .float()
            .mean()
            .detach()
            .cpu()
        )
    return total, metrics


def energy_separation_loss(embedding: torch.Tensor, labels: torch.Tensor, margin: float = 0.25) -> torch.Tensor:
    valid = labels != -100
    if int(valid.sum().item()) < 2:
        return embedding.sum() * 0.0
    z = F.normalize(embedding[valid], dim=-1)
    y = labels[valid]
    sim = z @ z.t()
    same = y[:, None] == y[None, :]
    eye = torch.eye(sim.size(0), dtype=torch.bool, device=sim.device)
    different = (~same) & (~eye)
    if not bool(different.any().item()):
        return embedding.sum() * 0.0
    return F.relu(sim[different] - margin).mean()


def binary_metrics(logits: torch.Tensor, labels: torch.Tensor, prefix: str) -> dict[str, float]:
    probs = torch.sigmoid(logits)
    pred = probs >= 0.5
    labels_bool = labels >= 0.5
    tp = (pred & labels_bool).sum().item()
    tn = ((~pred) & (~labels_bool)).sum().item()
    fp = (pred & (~labels_bool)).sum().item()
    fn = ((~pred) & labels_bool).sum().item()
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-8, precision + recall)
    return {
        f"{prefix}_accuracy": (tp + tn) / max(1, tp + tn + fp + fn),
        f"{prefix}_precision": precision,
        f"{prefix}_recall": recall,
        f"{prefix}_f1": f1,
    }
