from __future__ import annotations

import torch
from torch import nn


class CardInstanceEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        model_dim: int = 128,
        dynamic_dim: int = 33,
        dynamic_hidden_dim: int = 64,
        max_position: int = 128,
    ) -> None:
        super().__init__()
        self.card_id_embedding = nn.Embedding(vocab_size, model_dim, padding_idx=0)
        self.owner_embedding = nn.Embedding(4, model_dim, padding_idx=0)
        self.zone_embedding = nn.Embedding(16, model_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(max_position, model_dim)
        self.dynamic = nn.Sequential(
            nn.Linear(dynamic_dim, dynamic_hidden_dim),
            nn.GELU(),
            nn.Linear(dynamic_hidden_dim, model_dim),
        )
        self.norm = nn.LayerNorm(model_dim)

    def forward(
        self,
        card_index: torch.Tensor,
        owner: torch.Tensor,
        zone: torch.Tensor,
        position: torch.Tensor,
        dynamic: torch.Tensor,
    ) -> torch.Tensor:
        encoded = (
            self.card_id_embedding(card_index)
            + self.owner_embedding(owner.clamp(0, 3))
            + self.zone_embedding(zone.clamp(0, 15))
            + self.position_embedding(position.clamp(0, self.position_embedding.num_embeddings - 1))
            + self.dynamic(dynamic)
        )
        return self.norm(encoded)
