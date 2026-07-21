from __future__ import annotations

import torch
from torch import nn


class MultiSelectDecoder(nn.Module):
    """Autoregressive without-replacement decoder over the current option list."""

    def __init__(self, model_dim: int = 128) -> None:
        super().__init__()
        self.start = nn.Parameter(torch.empty(model_dim))
        nn.init.normal_(self.start, std=0.02)
        self.history_projection = nn.Linear(model_dim, model_dim)

    def step_logits(
        self,
        base_logits: torch.Tensor,
        option_embeddings: torch.Tensor,
        selected_mask: torch.Tensor,
        option_mask: torch.Tensor,
    ) -> torch.Tensor:
        count = selected_mask.sum(dim=1, keepdim=True).clamp(min=1)
        selected_summary = (option_embeddings * selected_mask.unsqueeze(-1)).sum(dim=1) / count
        start = self.start.unsqueeze(0).expand_as(selected_summary)
        history = torch.where(selected_mask.any(dim=1, keepdim=True), selected_summary, start)
        adjustment = torch.einsum(
            "bd,bod->bo", self.history_projection(history), option_embeddings
        ) / option_embeddings.shape[-1] ** 0.5
        return (base_logits + adjustment).masked_fill(~option_mask | selected_mask, float("-inf"))

    def decode(
        self,
        base_logits: torch.Tensor,
        option_embeddings: torch.Tensor,
        option_mask: torch.Tensor,
        count: int,
    ) -> list[list[int]]:
        selected_mask = torch.zeros_like(option_mask)
        sequences = [[] for _ in range(base_logits.shape[0])]
        for _ in range(min(count, option_mask.shape[1])):
            logits = self.step_logits(base_logits, option_embeddings, selected_mask, option_mask)
            choice = logits.argmax(dim=1)
            for row, index in enumerate(choice.tolist()):
                if option_mask[row, index] and not selected_mask[row, index]:
                    sequences[row].append(index)
                    selected_mask[row, index] = True
        return sequences
