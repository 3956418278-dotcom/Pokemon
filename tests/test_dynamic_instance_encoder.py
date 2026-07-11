from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from data.game_memory import GameMemoryState
from data.observation_parser import parse_observation
from data.state_schema import collate_card_dynamic
from models.card_instance_fusion import CardInstanceFusion
from models.dynamic_instance_encoder import DynamicInstanceEncoder

from .test_observation_parser import _observation


def test_dynamic_instance_encoder_and_fusion_backward() -> None:
    parsed = parse_observation(_observation())
    memory = GameMemoryState().update_from_parsed(parsed)
    batch = collate_card_dynamic(parsed.card_instances, memory.appearance_features(parsed.card_instances))
    encoder = DynamicInstanceEncoder(output_dim=64)
    fusion = CardInstanceFusion(static_dim=128, dynamic_dim=64, output_dim=128)
    dynamic = encoder(batch)
    static = torch.randn(dynamic.size(0), 128, requires_grad=True)
    fused = fusion(static, dynamic)
    assert dynamic.shape == (len(parsed.card_instances), 64)
    assert fused.shape == (len(parsed.card_instances), 128)
    loss = fused.square().mean()
    loss.backward()
    assert static.grad is not None
