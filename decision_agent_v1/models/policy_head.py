from __future__ import annotations

import torch
from torch import nn


class PolicyHead(nn.Module):
    def __init__(self, model_dim: int = 128) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(model_dim * 4, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, 1),
        )

    def forward(
        self, board_embedding: torch.Tensor, option_embeddings: torch.Tensor, option_mask: torch.Tensor
    ) -> torch.Tensor:
        board = board_embedding[:, None].expand_as(option_embeddings)
        features = torch.cat((board, option_embeddings, board * option_embeddings, board - option_embeddings), dim=-1)
        logits = self.mlp(features).squeeze(-1)
        return logits.masked_fill(~option_mask, float("-inf"))
