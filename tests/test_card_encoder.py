from __future__ import annotations

import pytest

pytest.importorskip("torch")

from data.card_dataset import CardDataset, collate_cards
from models.card_encoder import CardEncoder
from models.card_instance_encoder import CardInstanceEncoder


def test_card_encoder_output_shape() -> None:
    dataset = CardDataset.from_cache()
    items = [dataset[i] for i in range(min(4, len(dataset)))]
    batch = collate_cards(items, dataset.schema)
    model = CardEncoder(dataset.schema, embedding_dim=128)
    output = model(batch)
    assert tuple(output.shape) == (len(items), 128)


def test_card_encoder_detail_output_shape() -> None:
    dataset = CardDataset.from_cache()
    items = [dataset[i] for i in range(min(4, len(dataset)))]
    batch = collate_cards(items, dataset.schema)
    model = CardEncoder(dataset.schema, embedding_dim=128)
    output = model(batch, return_details=True)
    assert tuple(output.card_summary.shape) == (len(items), 128)
    assert output.detail_tokens.shape[0] == len(items)
    assert output.detail_tokens.shape[-1] == 128
    assert output.detail_mask.shape == output.detail_type_ids.shape
    assert output.detail_mask.shape == output.detail_tokens.shape[:2]
    assert bool((output.detail_type_ids[output.detail_mask == 0] == 0).all().item())


def test_card_instance_encoder_shape() -> None:
    import torch

    encoder = CardInstanceEncoder(static_dim=128, board_state_dim=32, appearance_dim=32, output_dim=128)
    static = torch.randn(3, 128)
    output = encoder(
        static,
        board_state_features=torch.zeros(3, 32),
        appearance_features=torch.zeros(3, 32),
    )
    assert tuple(output.shape) == (3, 128)


def test_card_instance_encoder_defaults_reserved_features() -> None:
    import torch

    encoder = CardInstanceEncoder(static_dim=128, board_state_dim=32, appearance_dim=32, output_dim=128)
    static = torch.randn(3, 128)
    output = encoder(static)
    assert tuple(output.shape) == (3, 128)
