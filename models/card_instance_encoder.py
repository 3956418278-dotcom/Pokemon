from __future__ import annotations

import torch
import torch.nn as nn


class CardInstanceEncoder(nn.Module):
    """Minimal interface reserved for future board-state models."""

    def __init__(self, static_dim: int = 128, dynamic_dim: int = 32, output_dim: int = 128) -> None:
        super().__init__()
        self.zone_embedding = nn.Embedding(16, 8)
        self.owner_embedding = nn.Embedding(4, 4)
        self.position_embedding = nn.Embedding(8, 4)
        self.dynamic_proj = nn.Sequential(
            nn.Linear(1 + 12 + 8 + 8 + 4 + 4, dynamic_dim),
            nn.ReLU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(static_dim + dynamic_dim, output_dim),
            nn.ReLU(),
            nn.LayerNorm(output_dim),
        )

    def forward(
        self,
        card_static_embedding: torch.Tensor,
        hp_ratio: torch.Tensor,
        attached_energy_counts: torch.Tensor,
        special_condition_flags: torch.Tensor,
        zone: torch.Tensor,
        owner: torch.Tensor,
        position: torch.Tensor,
    ) -> torch.Tensor:
        dynamic = torch.cat(
            [
                hp_ratio.unsqueeze(-1),
                attached_energy_counts.float(),
                special_condition_flags.float(),
                self.zone_embedding(zone.long()),
                self.owner_embedding(owner.long()),
                self.position_embedding(position.long()),
            ],
            dim=-1,
        )
        return self.fusion(torch.cat([card_static_embedding, self.dynamic_proj(dynamic)], dim=-1))

