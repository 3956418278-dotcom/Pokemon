from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class CardInstanceFusionOutput:
    card_instance_token: torch.Tensor
    attention_weights: torch.Tensor


class CardInstanceFusion(nn.Module):
    """Fuse static card details with a dynamic-state-conditioned query."""

    def __init__(
        self,
        static_dim: int = 128,
        dynamic_dim: int = 64,
        output_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.1,
        detail_type_count: int = 4,
    ) -> None:
        super().__init__()
        if static_dim != output_dim:
            raise ValueError("CardInstanceFusion requires static_dim == output_dim for its residual path")
        if output_dim % num_heads:
            raise ValueError("output_dim must be divisible by num_heads")
        self.static_dim = static_dim
        self.dynamic_dim = dynamic_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.dynamic_projection = nn.Linear(dynamic_dim, output_dim)
        self.query_norm = nn.LayerNorm(output_dim)
        self.detail_type_embedding = nn.Embedding(detail_type_count, output_dim, padding_idx=0)
        self.cross_attention = nn.MultiheadAttention(
            output_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attention_dropout = nn.Dropout(dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(output_dim, output_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim * 4, output_dim),
            nn.Dropout(dropout),
        )
        self.output_norm = nn.LayerNorm(output_dim)

    def forward(
        self,
        card_summary: torch.Tensor,
        dynamic_repr: torch.Tensor,
        detail_tokens: torch.Tensor | None = None,
        detail_mask: torch.Tensor | None = None,
        detail_type_ids: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> torch.Tensor | CardInstanceFusionOutput:
        query = self.query_norm(card_summary.float() + self.dynamic_projection(dynamic_repr.float()))
        batch_size = query.shape[0]
        detail_count = int(detail_tokens.shape[1]) if detail_tokens is not None else 0
        if batch_size == 0:
            token = query
            weights = query.new_zeros((0, self.num_heads, detail_count))
            return CardInstanceFusionOutput(token, weights) if return_attention else token

        context = torch.zeros_like(query)
        weights = query.new_zeros((batch_size, self.num_heads, detail_count))
        if detail_tokens is not None and detail_mask is not None and detail_count > 0:
            valid = detail_mask > 0
            if detail_type_ids is None:
                detail_type_ids = torch.zeros(valid.shape, dtype=torch.long, device=detail_tokens.device)
            finite_details = torch.where(
                valid.unsqueeze(-1),
                detail_tokens.float(),
                torch.zeros_like(detail_tokens, dtype=torch.float32),
            )
            key_value = finite_details + self.detail_type_embedding(
                detail_type_ids.long().clamp(min=0, max=self.detail_type_embedding.num_embeddings - 1)
            )
            has_detail = valid.any(dim=1)
            safe_key_value = torch.where(valid.unsqueeze(-1), key_value, torch.zeros_like(key_value))
            safe_padding_mask = ~valid.clone()
            if (~has_detail).any():
                safe_key_value[~has_detail] = 0.0
                safe_padding_mask[~has_detail] = True
                safe_padding_mask[~has_detail, 0] = False
            attended, raw_weights = self.cross_attention(
                query.unsqueeze(1),
                safe_key_value,
                safe_key_value,
                key_padding_mask=safe_padding_mask,
                need_weights=True,
                average_attn_weights=False,
            )
            context = torch.where(has_detail.unsqueeze(-1), attended.squeeze(1), torch.zeros_like(query))
            weights = raw_weights.squeeze(2)
            weights = torch.where(has_detail[:, None, None], weights, torch.zeros_like(weights))

        residual = query + self.attention_dropout(context)
        token = self.output_norm(residual + self.feed_forward(residual))
        if not torch.isfinite(token).all():
            raise FloatingPointError("CardInstanceFusion produced non-finite values")
        return CardInstanceFusionOutput(token, weights) if return_attention else token
