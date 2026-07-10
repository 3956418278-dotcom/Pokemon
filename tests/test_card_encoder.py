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


def test_card_instance_encoder_shape() -> None:
    import torch

    encoder = CardInstanceEncoder(static_dim=128, output_dim=128)
    static = torch.randn(3, 128)
    output = encoder(
        static,
        hp_ratio=torch.ones(3),
        attached_energy_counts=torch.zeros(3, 12),
        special_condition_flags=torch.zeros(3, 8),
        zone=torch.zeros(3, dtype=torch.long),
        owner=torch.zeros(3, dtype=torch.long),
        position=torch.zeros(3, dtype=torch.long),
    )
    assert tuple(output.shape) == (3, 128)
