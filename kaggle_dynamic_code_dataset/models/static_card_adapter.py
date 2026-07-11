from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn


@dataclass
class StaticCardFeatureOutput:
    summary: torch.Tensor
    known_mask: torch.Tensor
    detail_tokens: torch.Tensor | None = None
    detail_mask: torch.Tensor | None = None
    detail_type_ids: torch.Tensor | None = None


class StaticCardEmbeddingAdapter(nn.Module):
    """Map public card ids to pretrained static card summary embeddings."""

    def __init__(
        self,
        embedding_weight: torch.Tensor,
        card_id_to_index: dict[str, int],
        freeze: bool = True,
        detail_tokens: torch.Tensor | None = None,
        detail_mask: torch.Tensor | None = None,
        detail_type_ids: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if embedding_weight.dim() != 2:
            raise ValueError("embedding_weight must have shape [num_cards, embedding_dim]")
        self.card_id_to_index = {str(key): int(value) for key, value in card_id_to_index.items()}
        max_card_id = max([int(key) for key in self.card_id_to_index if str(key).isdigit()] + [0])
        index_lookup = torch.zeros(max_card_id + 1, dtype=torch.long)
        known_lookup = torch.zeros(max_card_id + 1, dtype=torch.bool)
        for card_id, index in self.card_id_to_index.items():
            if card_id.isdigit():
                index_lookup[int(card_id)] = int(index) + 1
                known_lookup[int(card_id)] = True
        padded_weight = torch.cat([embedding_weight.new_zeros(1, embedding_weight.size(1)), embedding_weight.float()], dim=0)
        self.embedding = nn.Embedding.from_pretrained(padded_weight, freeze=freeze, padding_idx=0)
        if detail_tokens is not None:
            padded_details = torch.cat(
                [
                    detail_tokens.new_zeros(1, detail_tokens.size(1), detail_tokens.size(2)),
                    detail_tokens.float(),
                ],
                dim=0,
            )
            self.register_buffer("detail_tokens", padded_details, persistent=False)
            self.register_buffer(
                "detail_mask",
                torch.cat([detail_mask.new_zeros(1, detail_mask.size(1)), detail_mask.float()], dim=0)
                if detail_mask is not None
                else torch.ones(padded_details.shape[:2], dtype=torch.float32),
                persistent=False,
            )
            self.register_buffer(
                "detail_type_ids",
                torch.cat([detail_type_ids.new_zeros(1, detail_type_ids.size(1)), detail_type_ids.long()], dim=0)
                if detail_type_ids is not None
                else torch.zeros(padded_details.shape[:2], dtype=torch.long),
                persistent=False,
            )
        else:
            self.detail_tokens = None
            self.detail_mask = None
            self.detail_type_ids = None
        self.register_buffer("index_lookup", index_lookup, persistent=False)
        self.register_buffer("known_lookup", known_lookup, persistent=False)

    @classmethod
    def from_artifacts(cls, artifact_dir: str | Path, freeze: bool = True) -> "StaticCardEmbeddingAdapter":
        artifact_dir = Path(artifact_dir)
        weight_path = artifact_dir / "card_embeddings.pt"
        mapping_path = artifact_dir / "card_id_to_index.json"
        weight_obj = torch.load(weight_path, map_location="cpu")
        if isinstance(weight_obj, dict):
            embedding_weight = weight_obj.get("embeddings")
            if embedding_weight is None:
                embedding_weight = weight_obj.get("card_embeddings")
            if embedding_weight is None:
                raise KeyError(f"{weight_path} does not contain embeddings")
        else:
            embedding_weight = weight_obj
        mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
        detail_tokens = torch.load(artifact_dir / "card_detail_tokens.pt", map_location="cpu") if (artifact_dir / "card_detail_tokens.pt").exists() else None
        detail_mask = torch.load(artifact_dir / "card_detail_masks.pt", map_location="cpu") if (artifact_dir / "card_detail_masks.pt").exists() else None
        detail_type_ids = torch.load(artifact_dir / "card_detail_type_ids.pt", map_location="cpu") if (artifact_dir / "card_detail_type_ids.pt").exists() else None
        return cls(
            embedding_weight.float(),
            mapping,
            freeze=freeze,
            detail_tokens=detail_tokens,
            detail_mask=detail_mask,
            detail_type_ids=detail_type_ids,
        )

    @property
    def embedding_dim(self) -> int:
        return int(self.embedding.embedding_dim)

    def forward(self, card_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedding_indices, known = self.lookup_indices(card_ids)
        return self.embedding(embedding_indices), known.float()

    def lookup_indices(self, card_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        safe_ids = card_ids.long().clamp_min(0)
        in_range = safe_ids < self.index_lookup.numel()
        lookup_ids = torch.where(in_range, safe_ids, torch.zeros_like(safe_ids))
        embedding_indices = self.index_lookup[lookup_ids]
        known = in_range & self.known_lookup[lookup_ids] & (card_ids.long() > 0)
        return embedding_indices, known

    def forward_features(self, card_ids: torch.Tensor) -> StaticCardFeatureOutput:
        embedding_indices, known = self.lookup_indices(card_ids)
        summary = self.embedding(embedding_indices)
        if self.detail_tokens is None:
            return StaticCardFeatureOutput(summary=summary, known_mask=known.float())
        return StaticCardFeatureOutput(
            summary=summary,
            known_mask=known.float(),
            detail_tokens=self.detail_tokens[embedding_indices],
            detail_mask=self.detail_mask[embedding_indices],
            detail_type_ids=self.detail_type_ids[embedding_indices],
        )
