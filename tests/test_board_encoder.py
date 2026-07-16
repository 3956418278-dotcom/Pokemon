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
from models.static_card_adapter import StaticArtifactContractNotConfigured, StaticCardAdapter
from tests.fakes.static_card_adapter import FakeStaticCardAdapter

from tests.test_observation_parser import _observation


def test_static_adapter_fails_before_contract_integration() -> None:
    adapter = StaticCardAdapter()
    assert not adapter.ready
    with pytest.raises(StaticArtifactContractNotConfigured):
        adapter.forward_features(torch.tensor([21, 22, 0]))
    with pytest.raises(StaticArtifactContractNotConfigured):
        StaticCardAdapter.from_artifacts("unused")


def test_board_tokenizer_and_transformer_shapes() -> None:
    parsed = parse_observation(_observation())
    memory = GameMemoryState().update_from_parsed(parsed)
    dynamic_batch = collate_card_dynamic(parsed.card_instances, memory.appearance_features(parsed.card_instances))
    static_adapter = FakeStaticCardAdapter(torch.randn(2, 128), {"21": 0, "22": 1})
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
    assert board.state_embedding.shape == (1, 128)


def test_dynamic_state_encoder_end_to_end() -> None:
    adapter = FakeStaticCardAdapter(torch.randn(2, 128), {"21": 0, "22": 1})
    encoder = DynamicStateEncoder(adapter)
    output = encoder(_observation())
    assert output.board.state_embedding.shape == (1, 128)
    assert output.card_instance_embeddings.shape[0] == len(output.parsed.card_instances)
    loss = output.board.state_embedding.square().mean()
    loss.backward()


def test_dynamic_state_encoder_handles_empty_observation() -> None:
    adapter = FakeStaticCardAdapter(torch.randn(1, 128), {})
    encoder = DynamicStateEncoder(adapter)
    output = encoder({"current": None, "select": None, "logs": []})
    assert output.card_instance_embeddings.shape == (0, 128)
    assert output.tokenized.tokens.shape == (1, 5, 128)
    assert output.board.state_embedding.shape == (1, 128)
