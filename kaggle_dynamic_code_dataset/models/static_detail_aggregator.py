from __future__ import annotations

import torch
import torch.nn as nn


class StaticDetailAggregator(nn.Module):
    """Aggregate a card's static detail tokens for dynamic instance fusion."""

    def __init__(self, summary_dim: int = 128, detail_dim: int = 128, output_dim: int = 128) -> None:
        super().__init__()
        self.query = nn.Linear(summary_dim, detail_dim)
        self.type_embedding = nn.Embedding(4, detail_dim)
        self.output = nn.Sequential(
            nn.Linear(summary_dim + detail_dim, output_dim),
            nn.ReLU(),
            nn.LayerNorm(output_dim),
        )

    def forward(
        self,
        card_summary: torch.Tensor,
        detail_tokens: torch.Tensor | None,
        detail_mask: torch.Tensor | None,
        detail_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if detail_tokens is None or detail_mask is None:
            return card_summary
        if detail_type_ids is not None:
            detail_tokens = detail_tokens + self.type_embedding(detail_type_ids.long().clamp_min(0).clamp_max(3))
        query = self.query(card_summary).unsqueeze(1)
        scores = (detail_tokens * query).sum(dim=-1) / (detail_tokens.size(-1) ** 0.5)
        valid = detail_mask > 0
        scores = scores.masked_fill(~valid, -1e9)
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)
        pooled = (detail_tokens * weights * detail_mask.unsqueeze(-1)).sum(dim=1)
        pooled = torch.where(valid.any(dim=1, keepdim=True), pooled, torch.zeros_like(pooled))
        return self.output(torch.cat([card_summary, pooled], dim=-1))
