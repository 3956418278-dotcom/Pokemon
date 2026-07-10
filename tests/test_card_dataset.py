from __future__ import annotations

import pytest

pytest.importorskip("torch")

from data.card_dataset import CardDataset, collate_cards


def test_card_dataset_batch_shapes() -> None:
    dataset = CardDataset.from_cache(rebuild=True)
    items = [dataset[i] for i in range(min(4, len(dataset)))]
    batch = collate_cards(items, dataset.schema)
    assert batch["single_cats"].shape[0] == len(items)
    assert batch["numeric"].shape[0] == len(items)
    assert batch["text_hashes"].shape[0] == len(items)
    assert len(batch["card_ids"]) == len(items)
