from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from data.game_memory import GameMemoryState
from data.observation_parser import parse_observation
from data.state_schema import collate_card_dynamic
from models.board_tokenizer import BoardTokenizer
from models.board_transformer import BoardTransformer
from models.card_instance_fusion import CardInstanceFusion
from models.dynamic_instance_encoder import DynamicInstanceEncoder
from models.dynamic_state_encoder import DynamicStateEncoder
from models.static_detail_aggregator import StaticDetailAggregator
from models.static_card_adapter import StaticCardEmbeddingAdapter

from .test_observation_parser import _observation


def test_static_adapter_maps_unknown_to_padding() -> None:
    weights = torch.randn(2, 128)
    details = torch.randn(2, 3, 128)
    detail_mask = torch.ones(2, 3)
    detail_type_ids = torch.ones(2, 3, dtype=torch.long)
    adapter = StaticCardEmbeddingAdapter(
        weights,
        {"21": 0, "22": 1},
        detail_tokens=details,
        detail_mask=detail_mask,
        detail_type_ids=detail_type_ids,
    )
    embedded, known = adapter(torch.tensor([21, 22, 999, 0]))
    assert embedded.shape == (4, 128)
    assert known.tolist() == [1.0, 1.0, 0.0, 0.0]
    assert torch.allclose(embedded[2], torch.zeros(128))
    features = adapter.forward_features(torch.tensor([21, 999]))
    assert features.summary.shape == (2, 128)
    assert features.detail_tokens.shape == (2, 3, 128)
    assert features.detail_mask[1].sum().item() == 0.0


def test_static_detail_aggregator_uses_detail_tokens() -> None:
    summary = torch.randn(2, 128)
    details = torch.randn(2, 3, 128)
    mask = torch.tensor([[1.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    type_ids = torch.tensor([[1, 2, 0], [0, 0, 0]])
    aggregator = StaticDetailAggregator()
    output = aggregator(summary, details, mask, type_ids)
    assert output.shape == (2, 128)
    assert not torch.allclose(output[0], summary[0])


def test_board_tokenizer_and_transformer_shapes() -> None:
    parsed = parse_observation(_observation())
    memory = GameMemoryState().update_from_parsed(parsed)
    dynamic_batch = collate_card_dynamic(parsed.card_instances, memory.appearance_features(parsed.card_instances))
    static_adapter = StaticCardEmbeddingAdapter(torch.randn(4, 128), {"1": 0, "9": 1, "21": 2, "22": 3})
    static_embeddings, known = static_adapter(dynamic_batch.card_ids)
    dynamic_embeddings = DynamicInstanceEncoder()(dynamic_batch)
    instance_embeddings = CardInstanceFusion()(static_embeddings, dynamic_embeddings)
    tokenizer = BoardTokenizer()
    tokenized = tokenizer(
        instance_embeddings,
        dynamic_batch.visibility_mask,
        torch.tensor(parsed.global_snapshot.features(), dtype=torch.float32),
        area_ids=torch.tensor([instance.area for instance in parsed.card_instances], dtype=torch.long),
        decision_features=torch.tensor(parsed.global_snapshot.decision_features(), dtype=torch.float32),
        match_features=torch.tensor(parsed.global_snapshot.match_features(), dtype=torch.float32),
        ledger_features=torch.tensor(memory.ledger_features(parsed.global_snapshot.your_index), dtype=torch.float32),
        event_features=torch.tensor(memory.recent_event_features(), dtype=torch.float32),
    )
    board = BoardTransformer(dropout=0.0)(tokenized.tokens, tokenized.mask)
    assert known.shape[0] == len(parsed.card_instances)
    assert tokenized.tokens.shape[1] == len(parsed.card_instances) + 1 + 1 + 1 + 2 + len(memory.recent_events)
    assert set(tokenized.type_ids.flatten().tolist()) >= {0, 1, 2, 3, 4, 5}
    assert board.tokens.shape == tokenized.tokens.shape
    assert board.pooled.shape == (1, 128)


def test_dynamic_state_encoder_end_to_end() -> None:
    adapter = StaticCardEmbeddingAdapter(
        torch.randn(4, 128),
        {"1": 0, "9": 1, "21": 2, "22": 3},
        detail_tokens=torch.randn(4, 3, 128),
        detail_mask=torch.ones(4, 3),
        detail_type_ids=torch.ones(4, 3, dtype=torch.long),
    )
    encoder = DynamicStateEncoder(adapter)
    output = encoder(_observation())
    assert output.board.pooled.shape == (1, 128)
    assert output.card_instance_embeddings.shape[0] == len(output.parsed.card_instances)
    loss = output.board.pooled.square().mean()
    loss.backward()


def test_dynamic_state_encoder_handles_empty_observation() -> None:
    adapter = StaticCardEmbeddingAdapter(torch.randn(4, 128), {"1": 0, "9": 1, "21": 2, "22": 3})
    encoder = DynamicStateEncoder(adapter)
    output = encoder({"current": None, "select": None, "logs": []})
    assert output.card_instance_embeddings.shape == (0, 128)
    assert output.tokenized.tokens.shape == (1, 5, 128)
    assert output.board.pooled.shape == (1, 128)
