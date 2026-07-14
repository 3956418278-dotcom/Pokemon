from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from data.game_memory import GameMemoryState
from data.observation_parser import parse_observation
from data.state_schema import ParsedObservation, collate_card_dynamic
from .board_transformer import BoardEncoderOutput, BoardTransformer
from .board_tokenizer import BoardTokenizer, BoardTokenizerOutput
from .card_instance_fusion import CardInstanceFusion
from .dynamic_instance_encoder import DynamicInstanceEncoder
from .static_card_adapter import StaticCardEmbeddingAdapter


@dataclass
class DynamicStateEncoderOutput:
    parsed: ParsedObservation
    memory: GameMemoryState
    static_known_mask: torch.Tensor
    card_instance_embeddings: torch.Tensor
    tokenized: BoardTokenizerOutput
    board: BoardEncoderOutput


class DynamicStateEncoder(nn.Module):
    def __init__(
        self,
        static_adapter: StaticCardEmbeddingAdapter,
        dynamic_encoder: DynamicInstanceEncoder | None = None,
        instance_fusion: CardInstanceFusion | None = None,
        board_tokenizer: BoardTokenizer | None = None,
        board_transformer: BoardTransformer | None = None,
    ) -> None:
        super().__init__()
        self.static_adapter = static_adapter
        self.dynamic_encoder = dynamic_encoder or DynamicInstanceEncoder()
        self.instance_fusion = instance_fusion or CardInstanceFusion(
            static_dim=static_adapter.embedding_dim,
            dynamic_dim=self.dynamic_encoder.output_dim,
            output_dim=128,
        )
        self.board_tokenizer = board_tokenizer or BoardTokenizer()
        self.board_transformer = board_transformer or BoardTransformer()

    def forward_parsed(self, parsed: ParsedObservation, memory: GameMemoryState) -> DynamicStateEncoderOutput:
        """Encode a parsed replay observation with its already-updated memory state."""
        dynamic_batch = collate_card_dynamic(parsed.card_instances, memory.appearance_features(parsed.card_instances))
        dynamic_batch = dynamic_batch.to(self.static_adapter.embedding.weight.device)
        static_features = self.static_adapter.forward_features(dynamic_batch.card_ids)
        known_mask = static_features.known_mask
        dynamic_batch.static_known_mask = known_mask.float()
        if static_features.detail_mask is not None:
            dynamic_batch.detail_exists_mask = (static_features.detail_mask > 0).any(dim=1).float()
        dynamic_embeddings = self.dynamic_encoder(dynamic_batch)
        instance_embeddings = self.instance_fusion(
            static_features.summary,
            dynamic_embeddings,
            static_features.detail_tokens,
            static_features.detail_mask,
            static_features.detail_type_ids,
        )
        device = instance_embeddings.device
        event_features = memory.recent_event_features()
        tokenized = self.board_tokenizer(
            instance_embeddings,
            dynamic_batch.visibility_mask.to(device),
            torch.tensor(parsed.global_snapshot.features(), dtype=torch.float32, device=device),
            area_ids=torch.tensor([instance.area for instance in parsed.card_instances], dtype=torch.long, device=device),
            decision_features=torch.tensor(parsed.global_snapshot.decision_features(), dtype=torch.float32, device=device),
            match_features=torch.tensor(parsed.global_snapshot.match_features(), dtype=torch.float32, device=device),
            ledger_features=torch.tensor(memory.ledger_features(parsed.global_snapshot.your_index), dtype=torch.float32, device=device),
            event_features=torch.tensor(event_features, dtype=torch.float32, device=device) if event_features else None,
        )
        board = self.board_transformer(tokenized.tokens, tokenized.mask)
        return DynamicStateEncoderOutput(
            parsed=parsed,
            memory=memory,
            static_known_mask=known_mask,
            card_instance_embeddings=instance_embeddings,
            tokenized=tokenized,
            board=board,
        )

    def forward(self, observation: Any, memory: GameMemoryState | None = None) -> DynamicStateEncoderOutput:
        parsed = parse_observation(observation)
        memory = memory or GameMemoryState()
        memory.update_from_parsed(parsed)
        return self.forward_parsed(parsed, memory)
