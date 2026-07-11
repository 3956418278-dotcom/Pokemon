from __future__ import annotations

import pytest

pytest.importorskip("torch")

import torch

from data.card_dataset import CardDataset, collate_cards
from models.card_encoder import CardEncoder
from models.card_pretrain_heads import DetailPredictionHead, MaskedFieldHeads, RelationHead, detail_prediction_loss, info_nce_loss, masked_field_loss


def test_pretrain_losses_backward() -> None:
    dataset = CardDataset.from_cache()
    items = [dataset[i] for i in range(min(8, len(dataset)))]
    batch = collate_cards(items, dataset.schema)
    encoder = CardEncoder(dataset.schema, embedding_dim=128)
    mask_heads = MaskedFieldHeads(dataset.schema, embedding_dim=128)
    relation_head = RelationHead(embedding_dim=128)
    detail_head = DetailPredictionHead(dataset.schema, detail_dim=128)

    output = encoder(batch, return_details=True)
    embedding = output.card_summary
    mask_loss, _ = masked_field_loss(mask_heads(embedding), batch)
    detail_loss, _ = detail_prediction_loss(detail_head(output.detail_tokens), output.detail_mask, output.detail_type_ids, batch)
    contrastive_loss, _ = info_nce_loss(encoder.text_embedding(batch), encoder.structure_embedding(batch), temperature=0.07)
    rel_logits = relation_head(embedding[:2], embedding[2:4], torch.zeros(2, dtype=torch.long))
    rel_loss = torch.nn.functional.binary_cross_entropy_with_logits(rel_logits, torch.tensor([1.0, 0.0]))
    total = mask_loss + detail_loss + contrastive_loss + rel_loss
    total.backward()
    assert torch.isfinite(total)
