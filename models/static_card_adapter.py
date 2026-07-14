from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import torch
import torch.nn as nn


@dataclass
class StaticCardFeatureOutput:
    card_summary: torch.Tensor
    detail_tokens: torch.Tensor | None
    detail_mask: torch.Tensor | None
    detail_type_ids: torch.Tensor | None
    known_mask: torch.Tensor

    @property
    def summary(self) -> torch.Tensor:
        """Alias for compatibility with existing dynamic fusion components."""
        return self.card_summary


class StaticCardAdapter(nn.Module):
    """Integration layer for static card features exported by colleague scripts.

    This adapter acts as the clean interface between colleague static artifacts
    and the dynamic Pokémon battle model.
    """

    def __init__(
        self,
        embedding_weight: torch.Tensor | None = None,
        card_id_to_index: dict[str, int] | None = None,
        freeze: bool = True,
        detail_tokens: torch.Tensor | None = None,
        detail_mask: torch.Tensor | None = None,
        detail_type_ids: torch.Tensor | None = None,
        embedding_dim: int = 128,
        max_details: int = 4,
        detail_dim: int = 128,
        *args,
        **kwargs,
    ) -> None:
        super().__init__()
        
        # If embedding_weight is not provided, mock it or use dummy
        if embedding_weight is None:
            embedding_weight = torch.zeros(1, embedding_dim)
            
        # Pad with 1 row at index 0 to act as padding row (same as the real adapter's padded_weight)
        padded_weight = torch.cat([
            embedding_weight.new_zeros(1, embedding_weight.size(-1)),
            embedding_weight
        ], dim=0)
        self.embedding = nn.Embedding.from_pretrained(padded_weight, freeze=freeze)
        
        self.card_id_to_index = card_id_to_index or {}
        
        # Pad details at index 0 for alignment with embedding indices
        if detail_tokens is not None:
            self._detail_tokens = torch.cat([
                detail_tokens.new_zeros(1, *detail_tokens.shape[1:]),
                detail_tokens
            ], dim=0)
        else:
            self._detail_tokens = None
            
        if detail_mask is not None:
            self._detail_mask = torch.cat([
                detail_mask.new_zeros(1, *detail_mask.shape[1:]),
                detail_mask
            ], dim=0)
        else:
            self._detail_mask = None
            
        if detail_type_ids is not None:
            self._detail_type_ids = torch.cat([
                detail_type_ids.new_zeros(1, *detail_type_ids.shape[1:]),
                detail_type_ids
            ], dim=0)
        else:
            self._detail_type_ids = None

        self.embedding_dim = embedding_weight.shape[-1]
        self.max_details = detail_tokens.shape[1] if detail_tokens is not None else max_details
        self.detail_dim = detail_tokens.shape[2] if detail_tokens is not None else detail_dim
        
        # A dummy parameter ensures device is tracked
        self.dummy_param = nn.Parameter(torch.zeros(1), requires_grad=False)

    @classmethod
    def from_artifacts(cls, artifact_dir: str | Path) -> StaticCardAdapter:
        """Load colleague static artifacts and initialize the adapter.

        Currently implemented as a clean interface shell awaiting formal artifact specifications.
        """
        # Placeholder initialization
        return cls()

    @property
    def detail_tokens(self) -> torch.Tensor | None:
        return self._detail_tokens

    @property
    def detail_mask(self) -> torch.Tensor | None:
        return self._detail_mask

    @property
    def detail_type_ids(self) -> torch.Tensor | None:
        return self._detail_type_ids

    def forward_features(self, card_ids: torch.Tensor) -> StaticCardFeatureOutput:
        """Map card IDs to their static feature representations."""
        device = self.dummy_param.device
        card_ids = card_ids.to(device)
        
        # Build index lookup tensor dynamically if not already built
        if not hasattr(self, "_index_lookup"):
            max_id = max([int(k) for k in self.card_id_to_index if str(k).isdigit()] + [0])
            lookup = torch.zeros(max_id + 1, dtype=torch.long, device=device)
            known = torch.zeros(max_id + 1, dtype=torch.bool, device=device)
            for card_id, idx in self.card_id_to_index.items():
                if str(card_id).isdigit():
                    lookup[int(card_id)] = int(idx) + 1  # 1-based index (0 is padding)
                    known[int(card_id)] = True
            self.register_buffer("_index_lookup", lookup, persistent=False)
            self.register_buffer("_known_lookup", known, persistent=False)
            
        safe_ids = card_ids.long().clamp(0, self._index_lookup.numel() - 1)
        in_range = (card_ids.long() >= 0) & (card_ids.long() < self._index_lookup.numel())
        
        lookup_ids = torch.where(in_range, safe_ids, torch.zeros_like(safe_ids))
        embedding_indices = self._index_lookup[lookup_ids]
        if not self.card_id_to_index:
            known_mask = (card_ids > 0).float()
        else:
            known_mask = (in_range & self._known_lookup[lookup_ids] & (card_ids.long() > 0)).float()
        
        card_summary = self.embedding(embedding_indices)
        
        if self._detail_tokens is not None:
            detail_tokens = self._detail_tokens[embedding_indices]
            detail_mask = self._detail_mask[embedding_indices] if self._detail_mask is not None else None
            detail_type_ids = self._detail_type_ids[embedding_indices] if self._detail_type_ids is not None else None
        else:
            # Mock details
            detail_tokens = torch.zeros(*card_ids.shape, self.max_details, self.detail_dim, device=device)
            detail_mask = torch.zeros(*card_ids.shape, self.max_details, device=device)
            detail_type_ids = torch.zeros(*card_ids.shape, self.max_details, dtype=torch.long, device=device)
            
        return StaticCardFeatureOutput(
            card_summary=card_summary,
            detail_tokens=detail_tokens,
            detail_mask=detail_mask,
            detail_type_ids=detail_type_ids,
            known_mask=known_mask,
        )

    def forward(self, card_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.forward_features(card_ids)
        return features.card_summary, features.known_mask


# Alias for backwards compatibility during transition
StaticCardEmbeddingAdapter = StaticCardAdapter
