from __future__ import annotations

import torch
import torch.nn as nn


class CardInstanceEncoder(nn.Module):
    """Fuse static card embeddings with reserved per-card feature groups.

    The dimensions are intentionally generic. Board-state and appearance
    features should be defined by the downstream state encoder rather than
    hard-coded here.
    """

    def __init__(
        self,
        static_dim: int = 128,
        board_state_dim: int = 32,
        appearance_dim: int = 32,
        output_dim: int = 128,
    ) -> None:
        super().__init__()
        self.static_dim = static_dim
        self.board_state_dim = board_state_dim
        self.appearance_dim = appearance_dim
        self.board_state_proj = nn.Sequential(
            nn.Linear(board_state_dim, board_state_dim),
            nn.ReLU(),
        )
        self.appearance_proj = nn.Sequential(
            nn.Linear(appearance_dim, appearance_dim),
            nn.ReLU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(static_dim + board_state_dim + appearance_dim, output_dim),
            nn.ReLU(),
            nn.LayerNorm(output_dim),
        )

    def forward(
        self,
        card_static_embedding: torch.Tensor,
        board_state_features: torch.Tensor | None = None,
        appearance_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if board_state_features is None:
            board_state_features = card_static_embedding.new_zeros(
                card_static_embedding.size(0),
                self.board_state_dim,
            )
        if appearance_features is None:
            appearance_features = card_static_embedding.new_zeros(
                card_static_embedding.size(0),
                self.appearance_dim,
            )
        board_state = self.board_state_proj(board_state_features.float())
        appearance = self.appearance_proj(appearance_features.float())
        return self.fusion(
            torch.cat(
                [
                    card_static_embedding,
                    board_state,
                    appearance,
                ],
                dim=-1,
            )
        )
