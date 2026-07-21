from __future__ import annotations

import torch
from torch import nn


class BoardEncoder(nn.Module):
    def __init__(
        self,
        model_dim: int = 128,
        global_dim: int = 35,
        history_dim: int = 44,
        layers: int = 2,
        heads: int = 4,
        ffn_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.global_encoder = nn.Sequential(nn.Linear(global_dim, model_dim), nn.GELU(), nn.Linear(model_dim, model_dim))
        self.history_encoder = nn.Sequential(nn.Linear(history_dim, model_dim), nn.GELU(), nn.Linear(model_dim, model_dim))
        self.token_type_embedding = nn.Embedding(3, model_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=layers,
            norm=nn.LayerNorm(model_dim),
            enable_nested_tensor=False,
        )

    def forward(
        self,
        card_tokens: torch.Tensor,
        card_mask: torch.Tensor,
        global_features: torch.Tensor,
        history_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, card_count, _ = card_tokens.shape
        global_token = self.global_encoder(global_features) + self.token_type_embedding.weight[0]
        history_token = self.history_encoder(history_features) + self.token_type_embedding.weight[1]
        typed_cards = card_tokens + self.token_type_embedding.weight[2]
        tokens = torch.cat((global_token[:, None], history_token[:, None], typed_cards), dim=1)
        valid_mask = torch.cat(
            (
                torch.ones(batch_size, 2, dtype=torch.bool, device=card_mask.device),
                card_mask,
            ),
            dim=1,
        )
        encoded = self.transformer(tokens, src_key_padding_mask=~valid_mask)
        return encoded[:, 0], encoded[:, 2 : 2 + card_count], valid_mask
