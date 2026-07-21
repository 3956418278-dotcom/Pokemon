from __future__ import annotations

from typing import Any

import torch
from torch import nn

from decision_agent_v1.state_upgrade.features import (
    LEDGER_CARD_NUMERIC_DIM,
    LEDGER_SUMMARY_DIM,
    SELF_DECK_SUMMARY_DIM,
)

from .card_instance_encoder import CardInstanceEncoder
from .multiselect_decoder import MultiSelectDecoder
from .option_encoder import OptionEncoder
from .policy_head import PolicyHead
from .value_head import ValueHead


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = mask.unsqueeze(-1).to(values.dtype)
    return (values * weight).sum(dim=1) / weight.sum(dim=1).clamp(min=1.0)


class StateUpgradeBoardEncoder(nn.Module):
    """Fixed V2 order: global, self deck, belief, ledger, K events, cards."""

    def __init__(
        self,
        card_embedding: nn.Embedding,
        template_count: int,
        model_dim: int,
        layers: int,
        heads: int,
        ffn_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.card_embedding = card_embedding
        self.global_encoder = nn.Sequential(nn.Linear(35, model_dim), nn.GELU(), nn.Linear(model_dim, model_dim))
        self.self_summary = nn.Sequential(nn.Linear(SELF_DECK_SUMMARY_DIM, model_dim), nn.GELU(), nn.Linear(model_dim, model_dim))
        self.self_card_numeric = nn.Linear(3, model_dim)
        self.belief_summary = nn.Sequential(nn.Linear(8, model_dim), nn.GELU(), nn.Linear(model_dim, model_dim))
        self.belief_template = nn.Embedding(template_count + 1, model_dim, padding_idx=0)
        self.ledger_summary = nn.Sequential(nn.Linear(LEDGER_SUMMARY_DIM, model_dim), nn.GELU(), nn.Linear(model_dim, model_dim))
        self.ledger_card_numeric = nn.Linear(LEDGER_CARD_NUMERIC_DIM, model_dim)
        self.event_type = nn.Embedding(26, model_dim, padding_idx=0)
        self.event_player = nn.Embedding(3, model_dim, padding_idx=0)
        self.event_zone = nn.Embedding(16, model_dim, padding_idx=0)
        self.event_numeric = nn.Linear(6, model_dim)
        self.event_position = nn.Embedding(16, model_dim)
        self.token_type = nn.Embedding(6, model_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=layers, norm=nn.LayerNorm(model_dim), enable_nested_tensor=False
        )

    def forward(
        self, card_tokens: torch.Tensor, card_mask: torch.Tensor, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        global_token = self.global_encoder(batch["global_features"]) + self.token_type.weight[0]
        self_cards = self.card_embedding(batch["self_deck_card_index"]) + self.self_card_numeric(batch["self_deck_card_numeric"])
        self_token = self.self_summary(batch["self_deck_summary"]) + _masked_mean(self_cards, batch["self_deck_card_mask"]) + self.token_type.weight[1]
        template = self.belief_template(batch["belief_template_index"])
        template_weight = batch["belief_template_probability"].unsqueeze(-1)
        template_token = (template * template_weight).sum(dim=1)
        belief_cards = self.card_embedding(batch["belief_card_index"]) * batch["belief_card_expected"].unsqueeze(-1)
        belief_token = self.belief_summary(batch["belief_summary"]) + template_token + _masked_mean(belief_cards, batch["belief_card_mask"]) + self.token_type.weight[2]
        ledger_cards = self.card_embedding(batch["ledger_card_index"]) + self.ledger_card_numeric(batch["ledger_card_numeric"])
        ledger_token = self.ledger_summary(batch["ledger_summary"]) + _masked_mean(ledger_cards, batch["ledger_card_mask"]) + self.token_type.weight[3]
        positions = torch.arange(batch["recent_event_type"].shape[1], device=card_tokens.device)
        events = (
            self.event_type(batch["recent_event_type"].clamp(0, 25))
            + self.event_player(batch["recent_event_player"].clamp(0, 2))
            + self.card_embedding(batch["recent_event_card_index"])
            + self.event_zone(batch["recent_event_source"].clamp(0, 15))
            + self.event_zone(batch["recent_event_target"].clamp(0, 15))
            + self.event_numeric(batch["recent_event_numeric"])
            + self.event_position(positions)[None, :, :]
            + self.token_type.weight[4]
        )
        typed_cards = card_tokens + self.token_type.weight[5]
        prefix = torch.stack((global_token, self_token, belief_token, ledger_token), dim=1)
        tokens = torch.cat((prefix, events, typed_cards), dim=1)
        batch_size = card_mask.shape[0]
        valid = torch.cat(
            (
                torch.ones(batch_size, 4, dtype=torch.bool, device=card_mask.device),
                batch["recent_event_mask"],
                card_mask,
            ),
            dim=1,
        )
        encoded = self.transformer(tokens, src_key_padding_mask=~valid)
        card_start = 4 + events.shape[1]
        return encoded[:, 0], encoded[:, card_start:], valid


class PolicyValueV2Model(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        template_count: int,
        model_dim: int = 128,
        dynamic_hidden_dim: int = 64,
        board_layers: int = 2,
        board_heads: int = 4,
        board_ffn_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.card_instance_encoder = CardInstanceEncoder(vocab_size, model_dim, 33, dynamic_hidden_dim)
        self.board_encoder = StateUpgradeBoardEncoder(
            self.card_instance_encoder.card_id_embedding,
            template_count,
            model_dim,
            board_layers,
            board_heads,
            board_ffn_dim,
            dropout,
        )
        self.option_encoder = OptionEncoder(self.card_instance_encoder.card_id_embedding, model_dim)
        self.policy_head = PolicyHead(model_dim)
        self.value_head = ValueHead(model_dim)
        self.multiselect_decoder = MultiSelectDecoder(model_dim)
        self.archetype_head = nn.Linear(model_dim, template_count)
        self.next_public_head = nn.Linear(model_dim, vocab_size)

    def load_v1_state_dict(self, state: dict[str, torch.Tensor]) -> tuple[list[str], list[str]]:
        compatible = {
            key: value for key, value in state.items()
            if key in self.state_dict() and self.state_dict()[key].shape == value.shape
        }
        missing, unexpected = self.load_state_dict(compatible, strict=False)
        return list(missing), list(unexpected)

    def forward(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        cards = self.card_instance_encoder(
            batch["card_index"], batch["card_owner"], batch["card_zone"],
            batch["card_position"], batch["card_dynamic"],
        )
        board, contextual_cards, board_mask = self.board_encoder(cards, batch["card_mask"], batch)
        options = self.option_encoder(batch, contextual_cards)
        return {
            "policy_logits": self.policy_head(board, options, batch["option_mask"]),
            "value_logits": self.value_head(board),
            "archetype_logits": self.archetype_head(board),
            "next_public_logits": self.next_public_head(board),
            "board_embedding": board,
            "contextual_card_tokens": contextual_cards,
            "option_embeddings": options,
            "option_mask": batch["option_mask"],
            "board_token_mask": board_mask,
        }
