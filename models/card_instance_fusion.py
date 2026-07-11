from __future__ import annotations

import torch
import torch.nn as nn


class CardInstanceFusion(nn.Module):
    def __init__(self, static_dim: int = 128, dynamic_dim: int = 64, output_dim: int = 128) -> None:
        super().__init__()
        self.static_dim = static_dim
        self.dynamic_dim = dynamic_dim
        self.output_dim = output_dim
        self.fusion = nn.Sequential(
            nn.Linear(static_dim + dynamic_dim, output_dim),
            nn.ReLU(),
            nn.LayerNorm(output_dim),
        )

    def forward(self, static_card_embedding: torch.Tensor, dynamic_embedding: torch.Tensor) -> torch.Tensor:
        return self.fusion(torch.cat([static_card_embedding.float(), dynamic_embedding.float()], dim=-1))
