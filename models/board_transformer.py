from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class BoardEncoderOutput:
    tokens: torch.Tensor
    mask: torch.Tensor
    pooled: torch.Tensor


class BoardTransformer(nn.Module):
    def __init__(
        self,
        token_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=num_heads,
            dim_feedforward=token_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(token_dim)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> BoardEncoderOutput:
        if tokens.dim() == 2:
            tokens = tokens.unsqueeze(0)
        if mask.dim() == 1:
            mask = mask.unsqueeze(0)
        key_padding_mask = mask <= 0
        encoded = self.encoder(tokens, src_key_padding_mask=key_padding_mask)
        encoded = self.norm(encoded)
        pooled = (encoded * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        return BoardEncoderOutput(tokens=encoded, mask=mask, pooled=pooled)
