from __future__ import annotations

import torch
from torch import nn


class ValueHead(nn.Module):
    def __init__(self, model_dim: int = 128) -> None:
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(model_dim, model_dim), nn.GELU(), nn.Linear(model_dim, 3))

    def forward(self, board_embedding: torch.Tensor) -> torch.Tensor:
        return self.mlp(board_embedding)

    @staticmethod
    def scalar(value_logits: torch.Tensor) -> torch.Tensor:
        probabilities = value_logits.softmax(dim=-1)
        return probabilities[..., 2] - probabilities[..., 0]
