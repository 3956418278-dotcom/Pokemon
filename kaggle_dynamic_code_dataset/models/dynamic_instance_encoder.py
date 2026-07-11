from __future__ import annotations

import torch
import torch.nn as nn

from data.state_schema import CARD_APPEARANCE_FEATURE_DIM, CARD_DYNAMIC_FEATURE_DIM, CardDynamicBatch


class DynamicInstanceEncoder(nn.Module):
    def __init__(
        self,
        dynamic_dim: int = CARD_DYNAMIC_FEATURE_DIM,
        appearance_dim: int = CARD_APPEARANCE_FEATURE_DIM,
        output_dim: int = 64,
    ) -> None:
        super().__init__()
        self.dynamic_dim = dynamic_dim
        self.appearance_dim = appearance_dim
        self.output_dim = output_dim
        self.dynamic_proj = nn.Sequential(
            nn.Linear(dynamic_dim, output_dim),
            nn.ReLU(),
        )
        self.appearance_proj = nn.Sequential(
            nn.Linear(appearance_dim, output_dim),
            nn.ReLU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim),
            nn.ReLU(),
            nn.LayerNorm(output_dim),
        )

    def forward(self, dynamic_batch: CardDynamicBatch | dict[str, torch.Tensor]) -> torch.Tensor:
        if isinstance(dynamic_batch, dict):
            dynamic_features = dynamic_batch["dynamic_features"]
            appearance_features = dynamic_batch["appearance_features"]
        else:
            dynamic_features = dynamic_batch.dynamic_features
            appearance_features = dynamic_batch.appearance_features
        dynamic = self.dynamic_proj(dynamic_features.float())
        appearance = self.appearance_proj(appearance_features.float())
        return self.fusion(torch.cat([dynamic, appearance], dim=-1))
