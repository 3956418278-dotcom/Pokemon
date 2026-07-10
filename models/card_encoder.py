from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class HashingTextEncoder(nn.Module):
    """Small offline text encoder using deterministic token hashes.

    This is the default because Kaggle competition kernels may not have network
    access for downloading external language models. It still supports
    precomputing/caching text embeddings and can be replaced behind the same
    interface later.
    """

    def __init__(self, hash_dim: int = 2048, token_dim: int = 128, output_dim: int = 128, freeze: bool = False) -> None:
        super().__init__()
        self.hash_dim = hash_dim
        self.token_dim = token_dim
        self.output_dim = output_dim
        self.embedding = nn.Embedding(hash_dim, token_dim)
        self.proj = nn.Sequential(nn.LayerNorm(token_dim), nn.Linear(token_dim, output_dim), nn.Tanh())
        if freeze:
            for parameter in self.parameters():
                parameter.requires_grad = False

    def forward(self, token_hashes: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
        token_emb = self.embedding(token_hashes.clamp_min(0).clamp_max(self.hash_dim - 1))
        mask = token_mask.unsqueeze(-1)
        pooled = (token_emb * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return self.proj(pooled)


class CardEncoder(nn.Module):
    def __init__(
        self,
        schema: dict[str, Any],
        embedding_dim: int = 128,
        cat_dim: int = 32,
        numeric_dim: int = 64,
        text_dim: int = 128,
        freeze_text_encoder: bool = False,
    ) -> None:
        super().__init__()
        self.schema = schema
        self.embedding_dim = embedding_dim
        self.single_fields = list(schema["single_fields"])
        self.multi_fields = list(schema["multi_fields"]) + ["attack_energy_types"]
        self.numeric_fields = list(schema["numeric_fields"])

        self.single_embeddings = nn.ModuleList(
            [nn.Embedding(len(schema["vocab"]["single"][field]), cat_dim) for field in self.single_fields]
        )
        self.multi_embeddings = nn.ModuleList(
            [nn.Embedding(len(schema["vocab"]["multi"][field]), cat_dim) for field in self.multi_fields]
        )
        self.cat_proj = nn.Sequential(
            nn.Linear(cat_dim * (len(self.single_fields) + len(self.multi_fields)), 128),
            nn.ReLU(),
            nn.LayerNorm(128),
        )
        self.numeric_proj = nn.Sequential(
            nn.Linear(len(self.numeric_fields) * 2, numeric_dim),
            nn.ReLU(),
            nn.Linear(numeric_dim, numeric_dim),
            nn.ReLU(),
            nn.LayerNorm(numeric_dim),
        )
        self.text_encoder = HashingTextEncoder(
            hash_dim=int(schema.get("text_hash_dim", 2048)),
            token_dim=text_dim,
            output_dim=text_dim,
            freeze=freeze_text_encoder,
        )
        self.fusion = nn.Sequential(
            nn.Linear(128 + numeric_dim + text_dim, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Linear(256, embedding_dim),
        )
        self.structure_projection = nn.Sequential(
            nn.Linear(128 + numeric_dim, embedding_dim),
            nn.ReLU(),
            nn.LayerNorm(embedding_dim),
        )
        self.text_projection = nn.Sequential(
            nn.Linear(text_dim, embedding_dim),
            nn.ReLU(),
            nn.LayerNorm(embedding_dim),
        )

    def encode_categories(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        single = batch["single_cats"]
        parts = []
        for i, emb in enumerate(self.single_embeddings):
            parts.append(emb(single[:, i]))
        values = batch["multi_values"]
        mask = batch["multi_mask"].unsqueeze(-1)
        offsets = batch["multi_offsets"]
        for i, emb in enumerate(self.multi_embeddings):
            start = offsets[:, i]
            pooled_rows = []
            for row_idx in range(values.size(0)):
                row_start = int(start[row_idx].item())
                row_end = int(offsets[row_idx, i + 1].item()) if i + 1 < offsets.size(1) else values.size(1)
                row_values = values[row_idx, row_start:row_end].clamp_min(0).clamp_max(emb.num_embeddings - 1)
                row_emb = emb(row_values)
                row_mask = mask[row_idx, row_start:row_end]
                pooled_rows.append((row_emb * row_mask).sum(dim=0) / row_mask.sum().clamp_min(1.0))
            parts.append(torch.stack(pooled_rows, dim=0))
        return self.cat_proj(torch.cat(parts, dim=-1))

    def encode_numeric(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.numeric_proj(torch.cat([batch["numeric"], batch["numeric_mask"]], dim=-1))

    def encode_text(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.text_encoder(batch["text_hashes"], batch["text_mask"])

    def structure_embedding(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        cats = self.encode_categories(batch)
        nums = self.encode_numeric(batch)
        return F.normalize(self.structure_projection(torch.cat([cats, nums], dim=-1)), dim=-1)

    def text_embedding(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return F.normalize(self.text_projection(self.encode_text(batch)), dim=-1)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        cats = self.encode_categories(batch)
        nums = self.encode_numeric(batch)
        text = self.encode_text(batch)
        return self.fusion(torch.cat([cats, nums, text], dim=-1))


def infer_head_sizes(schema: dict[str, Any]) -> dict[str, int]:
    return {
        "card_type": len(schema["vocab"]["single"]["card_type"]),
        "pokemon_type": len(schema["vocab"]["single"]["pokemon_type"]),
        "stage": len(schema["vocab"]["single"]["stage"]),
        "trainer_type": len(schema["vocab"]["single"]["trainer_type"]),
        "energy_type": len(schema["vocab"]["energy_type_target"]),
    }
