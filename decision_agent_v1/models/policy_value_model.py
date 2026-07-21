from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .board_encoder import BoardEncoder
from .card_instance_encoder import CardInstanceEncoder
from .multiselect_decoder import MultiSelectDecoder
from .option_encoder import OptionEncoder
from .policy_head import PolicyHead
from .value_head import ValueHead


class PolicyValueModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        model_dim: int = 128,
        dynamic_hidden_dim: int = 64,
        board_layers: int = 2,
        board_heads: int = 4,
        board_ffn_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.card_instance_encoder = CardInstanceEncoder(
            vocab_size, model_dim, 33, dynamic_hidden_dim
        )
        self.board_encoder = BoardEncoder(
            model_dim, 35, 44, board_layers, board_heads, board_ffn_dim, dropout
        )
        self.option_encoder = OptionEncoder(
            self.card_instance_encoder.card_id_embedding, model_dim
        )
        self.policy_head = PolicyHead(model_dim)
        self.value_head = ValueHead(model_dim)
        self.multiselect_decoder = MultiSelectDecoder(model_dim)

    def forward(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        cards = self.card_instance_encoder(
            batch["card_index"],
            batch["card_owner"],
            batch["card_zone"],
            batch["card_position"],
            batch["card_dynamic"],
        )
        board, contextual_cards, board_mask = self.board_encoder(
            cards, batch["card_mask"], batch["global_features"], batch["history_features"]
        )
        options = self.option_encoder(batch, contextual_cards)
        policy_logits = self.policy_head(board, options, batch["option_mask"])
        value_logits = self.value_head(board)
        return {
            "policy_logits": policy_logits,
            "value_logits": value_logits,
            "board_embedding": board,
            "contextual_card_tokens": contextual_cards,
            "option_embeddings": options,
            "option_mask": batch["option_mask"],
            "board_token_mask": board_mask,
        }
