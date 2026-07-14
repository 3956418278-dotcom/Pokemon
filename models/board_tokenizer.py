from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from data.state_schema import DECISION_FEATURE_DIM, EVENT_FEATURE_DIM, GLOBAL_FEATURE_DIM, LEDGER_FEATURE_DIM, MATCH_FEATURE_DIM


@dataclass
class BoardTokenizerOutput:
    tokens: torch.Tensor
    mask: torch.Tensor
    type_ids: torch.Tensor


class BoardTokenizer(nn.Module):
    def __init__(
        self,
        card_instance_dim: int = 128,
        global_feature_dim: int = GLOBAL_FEATURE_DIM,
        decision_feature_dim: int = DECISION_FEATURE_DIM,
        match_feature_dim: int = MATCH_FEATURE_DIM,
        ledger_feature_dim: int = LEDGER_FEATURE_DIM,
        event_feature_dim: int = EVENT_FEATURE_DIM,
        token_dim: int = 128,
        max_area_id: int = 12,
    ) -> None:
        super().__init__()
        self.token_dim = token_dim
        self.card_projection = nn.Sequential(
            nn.Linear(card_instance_dim, token_dim),
            nn.ReLU(),
            nn.LayerNorm(token_dim),
        )
        self.global_projection = nn.Sequential(
            nn.Linear(global_feature_dim, token_dim),
            nn.ReLU(),
            nn.LayerNorm(token_dim),
        )
        self.decision_projection = nn.Sequential(
            nn.Linear(decision_feature_dim, token_dim),
            nn.ReLU(),
            nn.LayerNorm(token_dim),
        )
        self.match_projection = nn.Sequential(
            nn.Linear(match_feature_dim, token_dim),
            nn.ReLU(),
            nn.LayerNorm(token_dim),
        )
        self.ledger_projection = nn.Sequential(
            nn.Linear(ledger_feature_dim, token_dim),
            nn.ReLU(),
            nn.LayerNorm(token_dim),
        )
        self.event_projection = nn.Sequential(
            nn.Linear(event_feature_dim, token_dim),
            nn.ReLU(),
            nn.LayerNorm(token_dim),
        )
        self.area_embedding = nn.Embedding(max_area_id + 1, token_dim)
        self.type_embedding = nn.Embedding(6, token_dim)

    def forward(
        self,
        card_instance_embeddings: torch.Tensor,
        card_mask: torch.Tensor,
        global_features: torch.Tensor,
        area_ids: torch.Tensor | None = None,
        decision_features: torch.Tensor | None = None,
        match_features: torch.Tensor | None = None,
        ledger_features: torch.Tensor | None = None,
        event_features: torch.Tensor | None = None,
        event_mask: torch.Tensor | None = None,
    ) -> BoardTokenizerOutput:
        if card_instance_embeddings.dim() == 2:
            card_instance_embeddings = card_instance_embeddings.unsqueeze(0)
        if card_mask.dim() == 1:
            card_mask = card_mask.unsqueeze(0)
        if global_features.dim() == 1:
            global_features = global_features.unsqueeze(0)
        batch_size, card_count, _ = card_instance_embeddings.shape
        if area_ids is None:
            area_ids = torch.zeros(batch_size, card_count, dtype=torch.long, device=card_instance_embeddings.device)
        elif area_ids.dim() == 1:
            area_ids = area_ids.unsqueeze(0)
        card_tokens = self.card_projection(card_instance_embeddings)
        card_tokens = card_tokens + self.area_embedding(area_ids.long().clamp_min(0).clamp_max(self.area_embedding.num_embeddings - 1))
        card_tokens = card_tokens + self.type_embedding(torch.ones_like(area_ids.long()).clamp_max(3))
        global_token = self.global_projection(global_features.float()).unsqueeze(1)
        global_token = global_token + self.type_embedding(torch.zeros(batch_size, 1, dtype=torch.long, device=global_token.device))
        
        token_parts = [global_token]
        global_mask = torch.ones(batch_size, 1, dtype=card_mask.dtype, device=card_mask.device)
        mask_parts = [global_mask]
        type_parts = [
            torch.zeros(batch_size, 1, dtype=torch.long, device=card_instance_embeddings.device),
        ]
        
        if decision_features is not None:
            if decision_features.dim() == 1:
                decision_features = decision_features.unsqueeze(0)
            decision_token = self.decision_projection(decision_features.float()).unsqueeze(1)
            decision_token = decision_token + self.type_embedding(torch.full((batch_size, 1), 4, dtype=torch.long, device=decision_token.device))
            token_parts.append(decision_token)
            mask_parts.append(torch.ones(batch_size, 1, dtype=card_mask.dtype, device=card_mask.device))
            type_parts.append(torch.full((batch_size, 1), 4, dtype=torch.long, device=decision_token.device))
        if match_features is not None:
            if match_features.dim() == 1:
                match_features = match_features.unsqueeze(0)
            match_token = self.match_projection(match_features.float()).unsqueeze(1)
            match_token = match_token + self.type_embedding(torch.full((batch_size, 1), 5, dtype=torch.long, device=match_token.device))
            token_parts.append(match_token)
            mask_parts.append(torch.ones(batch_size, 1, dtype=card_mask.dtype, device=card_mask.device))
            type_parts.append(torch.full((batch_size, 1), 5, dtype=torch.long, device=match_token.device))
        if ledger_features is not None:
            if ledger_features.dim() == 2:
                ledger_features = ledger_features.unsqueeze(0)
            ledger_tokens = self.ledger_projection(ledger_features.float())
            ledger_tokens = ledger_tokens + self.type_embedding(torch.full((batch_size, ledger_tokens.size(1)), 2, dtype=torch.long, device=ledger_tokens.device))
            token_parts.append(ledger_tokens)
            mask_parts.append(torch.ones(batch_size, ledger_tokens.size(1), dtype=card_mask.dtype, device=card_mask.device))
            type_parts.append(torch.full((batch_size, ledger_tokens.size(1)), 2, dtype=torch.long, device=ledger_tokens.device))
        if event_features is not None:
            if event_features.dim() == 2:
                event_features = event_features.unsqueeze(0)
            event_tokens = self.event_projection(event_features.float())
            event_tokens = event_tokens + self.type_embedding(torch.full((batch_size, event_tokens.size(1)), 3, dtype=torch.long, device=event_tokens.device))
            if event_mask is None:
                event_mask = torch.ones(batch_size, event_tokens.size(1), dtype=card_mask.dtype, device=card_mask.device)
            elif event_mask.dim() == 1:
                event_mask = event_mask.unsqueeze(0)
            token_parts.append(event_tokens)
            mask_parts.append(event_mask.float())
            type_parts.append(torch.full((batch_size, event_tokens.size(1)), 3, dtype=torch.long, device=event_tokens.device))
            
        token_parts.append(card_tokens)
        mask_parts.append(card_mask.float())
        type_parts.append(torch.ones(batch_size, card_count, dtype=torch.long, device=card_instance_embeddings.device))
        
        tokens = torch.cat(token_parts, dim=1)
        mask = torch.cat(mask_parts, dim=1)
        type_ids = torch.cat(type_parts, dim=1)
        type_ids = type_ids * mask.long()
        return BoardTokenizerOutput(tokens=tokens, mask=mask, type_ids=type_ids)
