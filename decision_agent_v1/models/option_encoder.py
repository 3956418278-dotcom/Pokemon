from __future__ import annotations

import torch
from torch import nn


class OptionEncoder(nn.Module):
    def __init__(
        self,
        card_id_embedding: nn.Embedding,
        model_dim: int = 128,
        numeric_dim: int = 12,
        max_position: int = 128,
    ) -> None:
        super().__init__()
        self.card_id_embedding = card_id_embedding
        self.option_type = nn.Embedding(24, model_dim)
        self.select_type = nn.Embedding(16, model_dim)
        self.select_context = nn.Embedding(64, model_dim)
        self.owner = nn.Embedding(4, model_dim)
        self.area = nn.Embedding(16, model_dim)
        self.position = nn.Embedding(max_position, model_dim)
        self.numeric = nn.Sequential(nn.Linear(numeric_dim, model_dim), nn.GELU(), nn.Linear(model_dim, model_dim))
        self.no_card_reference = nn.Parameter(torch.empty(model_dim))
        nn.init.normal_(self.no_card_reference, std=0.02)
        self.reference_projection = nn.Linear(model_dim, model_dim)
        self.norm = nn.LayerNorm(model_dim)

    def forward(self, batch: dict[str, torch.Tensor], contextual_cards: torch.Tensor) -> torch.Tensor:
        encoded = (
            self.option_type((batch["option_type"] + 1).clamp(0, 23))
            + self.select_type((batch["option_select_type"] + 1).clamp(0, 15))
            + self.select_context((batch["option_context"] + 1).clamp(0, 63))
            + self.owner((batch["option_owner"] + 1).clamp(0, 3))
            + self.area((batch["option_area"] + 1).clamp(0, 15))
            + self.position((batch["option_position"] + 1).clamp(0, self.position.num_embeddings - 1))
            + self.card_id_embedding(batch["option_card_index"])
            + self.numeric(batch["option_numeric"])
        )
        reference_index = batch["option_card_token_index"]
        safe_index = reference_index.clamp(min=0)
        gathered = contextual_cards.gather(
            1, safe_index.unsqueeze(-1).expand(-1, -1, contextual_cards.shape[-1])
        )
        no_reference = self.no_card_reference.view(1, 1, -1).expand_as(gathered)
        reference = torch.where((reference_index >= 0).unsqueeze(-1), gathered, no_reference)
        return self.norm(encoded + self.reference_projection(reference))
