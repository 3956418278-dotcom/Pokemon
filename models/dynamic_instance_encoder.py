from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from data.state_schema import (
    ATTACHMENT_KIND_VOCAB_SIZE,
    BOOLEAN_FEATURE_NAMES,
    FIELD_ROLE_VOCAB_SIZE,
    KNOWLEDGE_VOCAB_SIZE,
    NUMERICAL_FEATURE_NAMES,
    OWNER_VOCAB_SIZE,
    ZONE_VOCAB_SIZE,
    CardDynamicBatch,
)


class DynamicInstanceEncoder(nn.Module):
    """Encode explicitly structured, mask-aware card-instance state."""

    def __init__(self, output_dim: int = 64, dropout: float = 0.1) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.owner_embedding = nn.Embedding(OWNER_VOCAB_SIZE, 8, padding_idx=0)
        self.zone_embedding = nn.Embedding(ZONE_VOCAB_SIZE, 12, padding_idx=0)
        self.role_embedding = nn.Embedding(FIELD_ROLE_VOCAB_SIZE, 8, padding_idx=0)
        self.attachment_embedding = nn.Embedding(ATTACHMENT_KIND_VOCAB_SIZE, 8, padding_idx=0)
        self.knowledge_embedding = nn.Embedding(KNOWLEDGE_VOCAB_SIZE, 8, padding_idx=0)
        self.categorical_projection = nn.Sequential(nn.Linear(44, 32), nn.GELU(), nn.LayerNorm(32))

        numerical_dim = len(NUMERICAL_FEATURE_NAMES)
        boolean_dim = len(BOOLEAN_FEATURE_NAMES)
        self.register_buffer(
            "numerical_scales",
            torch.tensor([400.0, 400.0, 400.0, 1.0, 1.0, 4.0, 10.0, 4.0, 4.0]),
            persistent=False,
        )
        self.numerical_projection = nn.Sequential(
            nn.Linear(numerical_dim * 2, 32),
            nn.GELU(),
            nn.LayerNorm(32),
        )
        self.count_projection = nn.Sequential(nn.Linear(12 + 5 + 2, 32), nn.GELU(), nn.LayerNorm(32))
        self.boolean_projection = nn.Sequential(
            nn.Linear(boolean_dim * 2, 16),
            nn.GELU(),
            nn.LayerNorm(16),
        )
        self.validity_projection = nn.Sequential(
            nn.Linear(numerical_dim + 6, 16),
            nn.GELU(),
            nn.LayerNorm(16),
        )
        self.fusion = nn.Sequential(
            nn.Linear(128, output_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim),
        )

    @staticmethod
    def _field(batch: CardDynamicBatch | dict[str, torch.Tensor], name: str) -> torch.Tensor:
        return batch[name] if isinstance(batch, dict) else getattr(batch, name)

    def forward(self, dynamic_batch: CardDynamicBatch | dict[str, torch.Tensor]) -> torch.Tensor:
        numerical_features = self._field(dynamic_batch, "numerical_features").float()
        if numerical_features.shape[0] == 0:
            anchor = sum((parameter.sum() * 0.0 for parameter in self.parameters()), numerical_features.sum() * 0.0)
            return numerical_features.new_zeros((0, self.output_dim)) + anchor

        categorical = torch.cat(
            [
                self.owner_embedding(self._field(dynamic_batch, "owner_ids").long()),
                self.zone_embedding(self._field(dynamic_batch, "zone_ids").long()),
                self.role_embedding(self._field(dynamic_batch, "field_role_ids").long()),
                self.attachment_embedding(self._field(dynamic_batch, "attachment_kind_ids").long()),
                self.knowledge_embedding(self._field(dynamic_batch, "knowledge_ids").long()),
            ],
            dim=-1,
        )
        categorical = self.categorical_projection(categorical)

        numerical_mask = torch.nan_to_num(self._field(dynamic_batch, "numerical_mask").float()).clamp(0.0, 1.0)
        numerical_features = torch.where(numerical_mask > 0, numerical_features, torch.zeros_like(numerical_features))
        normalized = (numerical_features / self.numerical_scales).clamp(min=-4.0, max=4.0)
        numerical = self.numerical_projection(torch.cat([normalized * numerical_mask, numerical_mask], dim=-1))

        energy_mask = torch.nan_to_num(self._field(dynamic_batch, "energy_valid_mask").float()).clamp(0.0, 1.0)
        condition_mask = torch.nan_to_num(self._field(dynamic_batch, "condition_valid_mask").float()).clamp(0.0, 1.0)
        raw_energy = self._field(dynamic_batch, "energy_counts").float()
        raw_conditions = self._field(dynamic_batch, "condition_flags").float()
        energy = torch.log1p(torch.where(energy_mask > 0, raw_energy, torch.zeros_like(raw_energy)).clamp_min(0.0))
        conditions = torch.where(condition_mask > 0, raw_conditions, torch.zeros_like(raw_conditions))
        counts = self.count_projection(torch.cat([energy, conditions, energy_mask, condition_mask], dim=-1))

        boolean_mask = torch.nan_to_num(self._field(dynamic_batch, "boolean_mask").float()).clamp(0.0, 1.0)
        booleans = self._field(dynamic_batch, "boolean_features").float()
        booleans = torch.where(boolean_mask > 0, booleans, torch.zeros_like(booleans))
        booleans = self.boolean_projection(torch.cat([booleans * boolean_mask, boolean_mask], dim=-1))

        validity = torch.cat(
            [
                numerical_mask,
                energy_mask,
                condition_mask,
                torch.nan_to_num(self._field(dynamic_batch, "static_known_mask").float())
                .clamp(0.0, 1.0)
                .unsqueeze(-1),
                torch.nan_to_num(self._field(dynamic_batch, "detail_exists_mask").float())
                .clamp(0.0, 1.0)
                .unsqueeze(-1),
                torch.nan_to_num(self._field(dynamic_batch, "energy_resolved_mask").float())
                .clamp(0.0, 1.0)
                .unsqueeze(-1),
                torch.nan_to_num(self._field(dynamic_batch, "visibility_mask").float())
                .clamp(0.0, 1.0)
                .unsqueeze(-1),
            ],
            dim=-1,
        )
        validity = self.validity_projection(validity)
        return self.fusion(torch.cat([categorical, numerical, counts, booleans, validity], dim=-1))
