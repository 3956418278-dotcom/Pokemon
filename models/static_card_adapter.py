from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn


class StaticArtifactContractNotConfigured(RuntimeError):
    pass


@dataclass
class StaticCardFeatureOutput:
    card_summary: torch.Tensor
    known_mask: torch.Tensor
    detail_tokens: torch.Tensor | None = None
    detail_mask: torch.Tensor | None = None
    detail_type_ids: torch.Tensor | None = None


class StaticCardAdapter(nn.Module):
    """唯一的 colleague 静态产物接入边界。"""

    def __init__(self) -> None:
        super().__init__()
        self._ready = False

    @property
    def ready(self) -> bool:
        return self._ready

    @classmethod
    def from_artifacts(cls, artifact_dir: str | Path) -> "StaticCardAdapter":
        raise StaticArtifactContractNotConfigured(
            "colleague static artifact contract has not been integrated"
        )

    def forward_features(
        self,
        card_ids: torch.Tensor,
    ) -> StaticCardFeatureOutput:
        raise StaticArtifactContractNotConfigured(
            "StaticCardAdapter cannot encode cards before colleague artifacts "
            "and their loading contract are integrated"
        )

    def forward(self, card_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        output = self.forward_features(card_ids)
        return output.card_summary, output.known_mask
