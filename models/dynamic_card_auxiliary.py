from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from data.state_schema import FIELD_ROLE_VOCAB_SIZE, ZONE_VOCAB_SIZE


@dataclass
class DynamicCardAuxiliaryOutput:
    payable_logits: torch.Tensor
    energy_remaining: torch.Tensor
    hp_state: torch.Tensor
    zone_logits: torch.Tensor
    role_logits: torch.Tensor


class DynamicCardAuxiliaryHeads(nn.Module):
    def __init__(self, token_dim: int = 128, detail_dim: int = 128, energy_types: int = 12) -> None:
        super().__init__()
        self.detail_dim = detail_dim
        pair_dim = token_dim + detail_dim
        self.payable_head = nn.Sequential(nn.Linear(pair_dim, 128), nn.GELU(), nn.Linear(128, 1))
        self.energy_remaining_head = nn.Sequential(
            nn.Linear(pair_dim, 128),
            nn.GELU(),
            nn.Linear(128, energy_types),
        )
        self.hp_head = nn.Sequential(nn.Linear(token_dim, 64), nn.GELU(), nn.Linear(64, 2))
        self.zone_head = nn.Sequential(nn.Linear(token_dim, 64), nn.GELU(), nn.Linear(64, ZONE_VOCAB_SIZE))
        self.role_head = nn.Sequential(
            nn.Linear(token_dim, 64),
            nn.GELU(),
            nn.Linear(64, FIELD_ROLE_VOCAB_SIZE),
        )

    def forward(
        self,
        instance_tokens: torch.Tensor,
        detail_tokens: torch.Tensor | None,
        detail_mask: torch.Tensor | None = None,
        detail_type_ids: torch.Tensor | None = None,
    ) -> DynamicCardAuxiliaryOutput:
        del detail_type_ids
        count = instance_tokens.shape[0]
        if detail_tokens is None:
            detail_tokens = instance_tokens.new_zeros((count, 0, self.detail_dim))
        if detail_tokens.shape[-1] != self.detail_dim:
            raise ValueError(
                f"detail token width {detail_tokens.shape[-1]} does not match configured detail_dim {self.detail_dim}"
            )
        detail_count = detail_tokens.shape[1]
        if detail_mask is not None:
            valid = detail_mask > 0
            detail_tokens = torch.where(
                valid.unsqueeze(-1), detail_tokens.float(), torch.zeros_like(detail_tokens, dtype=torch.float32)
            )
        else:
            valid = None
        expanded = instance_tokens.unsqueeze(1).expand(-1, detail_count, -1)
        pair = torch.cat([expanded, detail_tokens.float()], dim=-1)
        payable_logits = self.payable_head(pair).squeeze(-1)
        energy_remaining = F.softplus(self.energy_remaining_head(pair))
        if valid is not None:
            valid_float = valid.to(payable_logits.dtype)
            payable_logits = payable_logits * valid_float
            energy_remaining = energy_remaining * valid_float.unsqueeze(-1)
        return DynamicCardAuxiliaryOutput(
            payable_logits=payable_logits,
            energy_remaining=energy_remaining,
            hp_state=torch.sigmoid(self.hp_head(instance_tokens)),
            zone_logits=self.zone_head(instance_tokens),
            role_logits=self.role_head(instance_tokens),
        )
