from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CardEncoderOutput:
    card_summary: torch.Tensor
    detail_tokens: torch.Tensor
    detail_mask: torch.Tensor
    detail_type_ids: torch.Tensor


class HashingTextEncoder(nn.Module):
    """Small offline text encoder using deterministic token hashes."""

    def __init__(self, hash_dim: int = 2048, token_dim: int = 128, output_dim: int = 128, freeze: bool = False) -> None:
        super().__init__()
        self.hash_dim = hash_dim
        self.token_dim = token_dim
        self.output_dim = output_dim
        self.embedding = nn.Embedding(hash_dim, token_dim)
        self.proj = nn.Sequential(nn.LayerNorm(token_dim), nn.Linear(token_dim, output_dim), nn.Tanh())
        if freeze:
            for parameter in self.parameters():
                parameter.requires_grad = False

    def forward(self, token_hashes: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
        token_emb = self.embedding(token_hashes.clamp_min(0).clamp_max(self.hash_dim - 1))
        mask = token_mask.unsqueeze(-1)
        pooled = (token_emb * mask).sum(dim=-2) / mask.sum(dim=-2).clamp_min(1.0)
        return self.proj(pooled)


class AttackEncoder(nn.Module):
    def __init__(self, schema: dict[str, Any], detail_dim: int = 128, text_dim: int = 128, energy_dim: int = 32) -> None:
        super().__init__()
        self.energy_type_embedding = nn.Parameter(torch.randn(12, energy_dim) * 0.02)
        self.damage_mode_embedding = nn.Embedding(len(schema["vocab"]["damage_mode"]), 16)
        self.name_encoder = HashingTextEncoder(int(schema.get("text_hash_dim", 2048)), output_dim=text_dim)
        self.effect_encoder = HashingTextEncoder(int(schema.get("text_hash_dim", 2048)), output_dim=text_dim)
        input_dim = 12 + energy_dim + 1 + 1 + 1 + 16 + text_dim + text_dim
        self.proj = nn.Sequential(
            nn.Linear(input_dim, detail_dim),
            nn.ReLU(),
            nn.LayerNorm(detail_dim),
        )

    def forward(self, attack_batch: dict[str, torch.Tensor]) -> torch.Tensor:
        counts = attack_batch["energy_counts"].float()
        energy_repr = (counts.unsqueeze(-1) * self.energy_type_embedding.unsqueeze(0).unsqueeze(0)).sum(dim=-2)
        name = self.name_encoder(attack_batch["name_hashes"], attack_batch["name_mask"])
        effect = self.effect_encoder(attack_batch["effect_hashes"], attack_batch["effect_mask"])
        parts = [
            counts,
            energy_repr,
            attack_batch["total_energy_cost"].float().unsqueeze(-1),
            attack_batch["damage"].float().unsqueeze(-1),
            attack_batch["damage_mask"].float().unsqueeze(-1),
            self.damage_mode_embedding(attack_batch["damage_mode"].long()),
            name,
            effect,
        ]
        return self.proj(torch.cat(parts, dim=-1))


class AbilityEncoder(nn.Module):
    def __init__(self, schema: dict[str, Any], detail_dim: int = 128, text_dim: int = 128) -> None:
        super().__init__()
        self.source_type_embedding = nn.Embedding(len(schema["vocab"]["effect_source_type"]), 16)
        self.name_encoder = HashingTextEncoder(int(schema.get("text_hash_dim", 2048)), output_dim=text_dim)
        self.effect_encoder = HashingTextEncoder(int(schema.get("text_hash_dim", 2048)), output_dim=text_dim)
        self.proj = nn.Sequential(
            nn.Linear(text_dim + text_dim + 16, detail_dim),
            nn.ReLU(),
            nn.LayerNorm(detail_dim),
        )

    def forward(self, ability_batch: dict[str, torch.Tensor]) -> torch.Tensor:
        name = self.name_encoder(ability_batch["name_hashes"], ability_batch["name_mask"])
        effect = self.effect_encoder(ability_batch["effect_hashes"], ability_batch["effect_mask"])
        source = self.source_type_embedding(ability_batch["source_type"].long())
        return self.proj(torch.cat([name, effect, source], dim=-1))


class SpecialEffectEncoder(nn.Module):
    def __init__(self, schema: dict[str, Any], detail_dim: int = 128, text_dim: int = 128) -> None:
        super().__init__()
        self.source_type_embedding = nn.Embedding(len(schema["vocab"]["effect_source_type"]), 16)
        self.name_encoder = HashingTextEncoder(int(schema.get("text_hash_dim", 2048)), output_dim=text_dim)
        self.effect_encoder = HashingTextEncoder(int(schema.get("text_hash_dim", 2048)), output_dim=text_dim)
        self.proj = nn.Sequential(
            nn.Linear(text_dim + text_dim + 16, detail_dim),
            nn.ReLU(),
            nn.LayerNorm(detail_dim),
        )

    def forward(self, effect_batch: dict[str, torch.Tensor]) -> torch.Tensor:
        name = self.name_encoder(effect_batch["name_hashes"], effect_batch["name_mask"])
        effect = self.effect_encoder(effect_batch["effect_hashes"], effect_batch["effect_mask"])
        source = self.source_type_embedding(effect_batch["source_type"].long())
        return self.proj(torch.cat([name, effect, source], dim=-1))


class DetailAttentionPooler(nn.Module):
    def __init__(self, detail_dim: int = 128) -> None:
        super().__init__()
        self.score = nn.Linear(detail_dim, 1)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        valid = mask > 0
        scores = self.score(tokens).squeeze(-1).masked_fill(~valid, -1e9)
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)
        pooled = (tokens * weights * mask.unsqueeze(-1)).sum(dim=1)
        return torch.where(valid.any(dim=1, keepdim=True), pooled, torch.zeros_like(pooled))


class CardEncoder(nn.Module):
    def __init__(
        self,
        schema: dict[str, Any],
        embedding_dim: int = 128,
        cat_dim: int = 32,
        numeric_dim: int = 64,
        text_dim: int = 128,
        detail_token_dim: int = 128,
        freeze_text_encoder: bool = False,
    ) -> None:
        super().__init__()
        self.schema = schema
        self.embedding_dim = embedding_dim
        self.detail_token_dim = detail_token_dim
        self.single_fields = list(schema["single_fields"])
        self.multi_fields = list(schema["multi_fields"]) + ["attack_energy_types"]
        self.numeric_fields = list(schema["numeric_fields"])

        self.single_embeddings = nn.ModuleList(
            [nn.Embedding(len(schema["vocab"]["single"][field]), cat_dim) for field in self.single_fields]
        )
        self.multi_embeddings = nn.ModuleList(
            [nn.Embedding(len(schema["vocab"]["multi"][field]), cat_dim) for field in self.multi_fields]
        )
        self.cat_proj = nn.Sequential(
            nn.Linear(cat_dim * (len(self.single_fields) + len(self.multi_fields)), 128),
            nn.ReLU(),
            nn.LayerNorm(128),
        )
        self.numeric_proj = nn.Sequential(
            nn.Linear(len(self.numeric_fields) * 2, numeric_dim),
            nn.ReLU(),
            nn.Linear(numeric_dim, numeric_dim),
            nn.ReLU(),
            nn.LayerNorm(numeric_dim),
        )
        self.text_encoder = HashingTextEncoder(
            hash_dim=int(schema.get("text_hash_dim", 2048)),
            token_dim=text_dim,
            output_dim=text_dim,
            freeze=freeze_text_encoder,
        )
        self.attack_encoder = AttackEncoder(schema, detail_token_dim, text_dim)
        self.ability_encoder = AbilityEncoder(schema, detail_token_dim, text_dim)
        self.special_effect_encoder = SpecialEffectEncoder(schema, detail_token_dim, text_dim)
        self.attack_pooler = DetailAttentionPooler(detail_token_dim)
        self.ability_pooler = DetailAttentionPooler(detail_token_dim)
        self.special_effect_pooler = DetailAttentionPooler(detail_token_dim)
        self.fusion = nn.Sequential(
            nn.Linear(128 + numeric_dim + text_dim + detail_token_dim * 3, embedding_dim),
            nn.ReLU(),
            nn.LayerNorm(embedding_dim),
        )
        self.energy_residual = nn.Linear(16, embedding_dim)
        self.summary_norm = nn.LayerNorm(embedding_dim)
        self.structure_projection = nn.Sequential(
            nn.Linear(128 + numeric_dim, embedding_dim),
            nn.ReLU(),
            nn.LayerNorm(embedding_dim),
        )
        self.text_projection = nn.Sequential(
            nn.Linear(text_dim, embedding_dim),
            nn.ReLU(),
            nn.LayerNorm(embedding_dim),
        )

    def encode_categories(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        single = batch["single_cats"]
        parts = []
        for i, emb in enumerate(self.single_embeddings):
            parts.append(emb(single[:, i]))
        values = batch["multi_values"]
        mask = batch["multi_mask"].unsqueeze(-1)
        offsets = batch["multi_offsets"]
        for i, emb in enumerate(self.multi_embeddings):
            start = offsets[:, i]
            pooled_rows = []
            for row_idx in range(values.size(0)):
                row_start = int(start[row_idx].item())
                row_end = int(offsets[row_idx, i + 1].item()) if i + 1 < offsets.size(1) else values.size(1)
                row_values = values[row_idx, row_start:row_end].clamp_min(0).clamp_max(emb.num_embeddings - 1)
                row_emb = emb(row_values)
                row_mask = mask[row_idx, row_start:row_end]
                pooled_rows.append((row_emb * row_mask).sum(dim=0) / row_mask.sum().clamp_min(1.0))
            parts.append(torch.stack(pooled_rows, dim=0))
        return self.cat_proj(torch.cat(parts, dim=-1))

    def encode_numeric(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.numeric_proj(torch.cat([batch["numeric"], batch["numeric_mask"]], dim=-1))

    def encode_text(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.text_encoder(batch["text_hashes"], batch["text_mask"])

    def encode_details(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        attack_tokens = self.attack_encoder(
            {
                "name_hashes": batch["attack_name_hashes"],
                "name_mask": batch["attack_name_mask"],
                "effect_hashes": batch["attack_effect_hashes"],
                "effect_mask": batch["attack_effect_mask"],
                "energy_counts": batch["attack_energy_counts"],
                "total_energy_cost": batch["attack_total_energy_cost"],
                "damage": batch["attack_damage"],
                "damage_mask": batch["attack_damage_mask"],
                "damage_mode": batch["attack_damage_mode"],
            }
        )
        ability_tokens = self.ability_encoder(
            {
                "name_hashes": batch["ability_name_hashes"],
                "name_mask": batch["ability_name_mask"],
                "effect_hashes": batch["ability_effect_hashes"],
                "effect_mask": batch["ability_effect_mask"],
                "source_type": torch.full_like(batch["ability_mask"].long(), int(self.schema["vocab"]["effect_source_type"]["ability"])),
            }
        )
        effect_tokens = self.special_effect_encoder(
            {
                "name_hashes": batch["effect_name_hashes"],
                "name_mask": batch["effect_name_mask"],
                "effect_hashes": batch["effect_text_hashes"],
                "effect_mask": batch["effect_text_mask"],
                "source_type": batch["effect_source_type"],
            }
        )
        tokens = torch.cat([attack_tokens, ability_tokens, effect_tokens], dim=1)
        mask = torch.cat([batch["attack_mask"], batch["ability_mask"], batch["effect_mask"]], dim=1)
        type_ids = torch.cat(
            [
                torch.full_like(batch["attack_mask"].long(), 1),
                torch.full_like(batch["ability_mask"].long(), 2),
                torch.full_like(batch["effect_mask"].long(), 3),
            ],
            dim=1,
        )
        type_ids = type_ids * mask.long()
        return tokens, mask, type_ids

    def structure_embedding(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        cats = self.encode_categories(batch)
        nums = self.encode_numeric(batch)
        return F.normalize(self.structure_projection(torch.cat([cats, nums], dim=-1)), dim=-1)

    def text_embedding(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return F.normalize(self.text_projection(self.encode_text(batch)), dim=-1)

    def forward(self, batch: dict[str, torch.Tensor], return_details: bool = False) -> torch.Tensor | CardEncoderOutput:
        cats = self.encode_categories(batch)
        nums = self.encode_numeric(batch)
        text = self.encode_text(batch)
        detail_tokens, detail_mask, detail_type_ids = self.encode_details(batch)
        attack_summary = self.attack_pooler(detail_tokens, detail_mask * (detail_type_ids == 1).float())
        ability_summary = self.ability_pooler(detail_tokens, detail_mask * (detail_type_ids == 2).float())
        effect_summary = self.special_effect_pooler(detail_tokens, detail_mask * (detail_type_ids == 3).float())
        learned_summary = self.fusion(torch.cat([cats, nums, text, attack_summary, ability_summary, effect_summary], dim=-1))
        energy_features = torch.cat(
            [
                batch["is_energy"].float().unsqueeze(-1),
                batch["is_basic_energy"].float().unsqueeze(-1),
                batch["is_special_energy"].float().unsqueeze(-1),
                batch["provided_energy_multihot"].float(),
                batch["provided_energy_amount"].float().unsqueeze(-1),
            ],
            dim=-1,
        )
        card_summary = self.summary_norm(learned_summary + self.energy_residual(energy_features))
        if not return_details:
            return card_summary
        return CardEncoderOutput(
            card_summary=card_summary,
            detail_tokens=detail_tokens,
            detail_mask=detail_mask,
            detail_type_ids=detail_type_ids,
        )


def infer_head_sizes(schema: dict[str, Any]) -> dict[str, int]:
    return {
        "card_type": len(schema["vocab"]["single"]["card_type"]),
        "pokemon_type": len(schema["vocab"]["single"]["pokemon_type"]),
        "stage": len(schema["vocab"]["single"]["stage"]),
        "trainer_type": len(schema["vocab"]["single"]["trainer_type"]),
        "energy_type": len(schema["vocab"]["energy_type_target"]),
        "damage_mode": len(schema["vocab"]["damage_mode"]),
        "effect_source_type": len(schema["vocab"]["effect_source_type"]),
        "detail_type": len(schema["vocab"]["detail_type"]),
    }
