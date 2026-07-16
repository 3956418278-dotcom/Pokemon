from __future__ import annotations

import torch
import torch.nn as nn

from models.static_card_adapter import StaticCardFeatureOutput


class FakeStaticCardAdapter(nn.Module):
    """Controllable static features for dynamic model unit tests."""

    def __init__(
        self,
        embedding_weight: torch.Tensor | None = None,
        card_id_to_index: dict[str, int] | None = None,
        *,
        freeze: bool = True,
        detail_tokens: torch.Tensor | None = None,
        detail_mask: torch.Tensor | None = None,
        detail_type_ids: torch.Tensor | None = None,
        embedding_dim: int = 128,
        max_details: int = 4,
        detail_dim: int = 128,
    ) -> None:
        super().__init__()
        if embedding_weight is None:
            embedding_weight = torch.zeros(1, embedding_dim)
        padded_weight = torch.cat(
            [embedding_weight.new_zeros(1, embedding_weight.size(-1)), embedding_weight],
            dim=0,
        )
        self.embedding = nn.Embedding.from_pretrained(padded_weight, freeze=freeze)
        self.card_id_to_index = card_id_to_index or {}
        self.embedding_dim = int(embedding_weight.shape[-1])
        self.max_details = int(detail_tokens.shape[1]) if detail_tokens is not None else max_details
        self.detail_dim = int(detail_tokens.shape[2]) if detail_tokens is not None else detail_dim
        self.register_buffer("detail_tokens", self._pad(detail_tokens), persistent=False)
        self.register_buffer("detail_mask", self._pad(detail_mask), persistent=False)
        self.register_buffer("detail_type_ids", self._pad(detail_type_ids), persistent=False)
        self._build_lookup()

    @staticmethod
    def _pad(tensor: torch.Tensor | None) -> torch.Tensor | None:
        if tensor is None:
            return None
        return torch.cat([tensor.new_zeros(1, *tensor.shape[1:]), tensor], dim=0)

    @property
    def ready(self) -> bool:
        return True

    def _build_lookup(self) -> None:
        numeric_mapping = {
            int(card_id): int(index)
            for card_id, index in self.card_id_to_index.items()
            if str(card_id).isdigit()
        }
        size = max(numeric_mapping, default=0) + 1
        lookup = torch.zeros(size, dtype=torch.long)
        known = torch.zeros(size, dtype=torch.bool)
        for card_id, index in numeric_mapping.items():
            lookup[card_id] = index + 1
            known[card_id] = True
        self.register_buffer("_index_lookup", lookup, persistent=False)
        self.register_buffer("_known_lookup", known, persistent=False)

    def forward_features(self, card_ids: torch.Tensor) -> StaticCardFeatureOutput:
        card_ids = card_ids.to(self.embedding.weight.device).long()
        in_range = (card_ids >= 0) & (card_ids < self._index_lookup.numel())
        safe_ids = torch.where(in_range, card_ids, torch.zeros_like(card_ids))
        embedding_indices = self._index_lookup[safe_ids]
        known_mask = (in_range & self._known_lookup[safe_ids] & (card_ids > 0)).float()
        detail_tokens = self.detail_tokens[embedding_indices] if self.detail_tokens is not None else None
        detail_mask = self.detail_mask[embedding_indices] if self.detail_mask is not None else None
        detail_type_ids = self.detail_type_ids[embedding_indices] if self.detail_type_ids is not None else None
        return StaticCardFeatureOutput(
            card_summary=self.embedding(embedding_indices),
            known_mask=known_mask,
            detail_tokens=detail_tokens,
            detail_mask=detail_mask,
            detail_type_ids=detail_type_ids,
        )

    def forward(self, card_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        output = self.forward_features(card_ids)
        return output.card_summary, output.known_mask
