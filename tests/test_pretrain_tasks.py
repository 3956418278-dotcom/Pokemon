from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from models.card_encoder import CardEncoder
from models.card_pretrain_heads import (
    CardDetailOwnershipHead,
    CardRelationHead,
    MaskedCardFieldHeads,
    MaskedDetailHeads,
    card_detail_ownership_loss,
    card_relation_loss,
    masked_card_field_loss,
    masked_detail_loss,
)
from training.pretrain_card_encoder import mask_training_inputs


def schema() -> dict:
    return {
        "vocab": {
            "card_category": {"<PAD>": 0, "POKEMON": 1, "TRAINER": 2, "ENERGY": 3},
            "card_name": {"<PAD>": 0, "A": 1, "B": 2},
            "species": {"<PAD>": 0, "A": 1},
            "stage": {"<PAD>": 0, "BASIC": 1, "STAGE1": 2, "<MASK>": 3},
            "pokemon_type": {"<PAD>": 0, "G": 1, "R": 2, "<MASK>": 3},
            "weakness_type": {"<PAD>": 0, "G": 1, "R": 2, "<MASK>": 3},
            "resistance_type": {"<PAD>": 0, "G": 1, "R": 2, "<MASK>": 3},
            "trainer_subtype": {"<PAD>": 0, "ITEM": 1, "<MASK>": 2},
            "energy_subtype": {"<PAD>": 0, "BASIC": 1, "<MASK>": 2},
            "hp_applicability": {"<PAD>": 0, "POKEMON": 1, "PLAYABLE_AS_POKEMON": 2},
            "rule_flags": {"EX": 0, "TERA": 1},
            "card_tags": {"FOSSIL": 0, "TERA": 1, "TERA_TYPE_FIRE": 2, "ANCIENT": 3},
            "detail_type": {"<PAD>": 0, "ATTACK": 1, "ABILITY": 2, "CARD_EFFECT": 3},
            "detail_subtype": {"<PAD>": 0, "ATTACK": 1, "ABILITY": 2, "ITEM": 3, "<MASK>": 4},
            "damage_mode": {"<PAD>": 0, "NONE": 1, "FIXED": 2, "PLUS": 3, "<MASK>": 4},
        },
        "rule_flag_vocab": {"EX": 0, "TERA": 1},
        "card_tag_vocab": {"FOSSIL": 0, "TERA": 1, "TERA_TYPE_FIRE": 2, "ANCIENT": 3},
        "text_vocab_size": 24,
        "text_pad_id": 0,
        "text_mask_id": 1,
        "max_text_length": 4,
        "max_details": 3,
        "energy_type_count": 12,
        "energy_types": ["C", "G", "R", "W", "L", "P", "F", "D", "M", "N", "Y", "A"],
        "provided_energy_types": ["C", "G", "R", "W", "L", "P", "F", "D", "M", "N", "Y", "A", "TEAM_ROCKET"],
        "text_encoder_layers": 1,
    }


def batch() -> dict[str, torch.Tensor]:
    return {
        "card_category_ids": torch.tensor([1, 2]),
        "card_name_ids": torch.tensor([1, 2]),
        "species_ids": torch.tensor([1, 0]),
        "previous_species_ids": torch.tensor([0, 0]),
        "evolves_from_name_ids": torch.tensor([0, 0]),
        "stage_ids": torch.tensor([1, 0]),
        "pokemon_type_ids": torch.tensor([1, 0]),
        "weakness_type_ids": torch.tensor([2, 0]),
        "resistance_type_ids": torch.tensor([0, 0]),
        "trainer_subtype_ids": torch.tensor([0, 1]),
        "energy_subtype_ids": torch.tensor([0, 0]),
        "hp_applicability_ids": torch.tensor([1, 2]),
        "printed_hp": torch.tensor([100.0, 60.0]),
        "printed_hp_mask": torch.tensor([1.0, 1.0]),
        "retreat": torch.tensor([2.0, 0.0]),
        "retreat_mask": torch.tensor([1.0, 0.0]),
        "rule_flag_multihot": torch.tensor([[1.0, 0.0], [0.0, 0.0]]),
        "card_tag_multihot": torch.tensor([[0.0, 1.0, 1.0, 1.0], [1.0, 0.0, 0.0, 0.0]]),
        "provided_energy_counts": torch.zeros(2, 13),
        # Attack is deliberately at source position 1 rather than position 0.
        "detail_type_ids": torch.tensor([[3, 1, 2], [3, 0, 0]]),
        "detail_subtype_ids": torch.tensor([[3, 1, 2], [3, 0, 0]]),
        "detail_mask": torch.tensor([[1.0, 1.0, 1.0], [1.0, 0.0, 0.0]]),
        "detail_text_ids": torch.tensor(
            [[[2, 3, 0], [4, 5, 6], [7, 8, 0]], [[9, 10, 0], [0, 0, 0], [0, 0, 0]]]
        ),
        "detail_text_mask": torch.tensor(
            [[[1, 1, 0], [1, 1, 1], [1, 1, 0]], [[1, 1, 0], [0, 0, 0], [0, 0, 0]]],
            dtype=torch.float32,
        ),
        "attack_energy_counts": torch.tensor(
            [
                [[0.0] * 12, [1.0, 2.0] + [0.0] * 10, [0.0] * 12],
                [[0.0] * 12, [0.0] * 12, [0.0] * 12],
            ]
        ),
        "base_damage": torch.tensor([[0.0, 90.0, 0.0], [0.0, 0.0, 0.0]]),
        "base_damage_mask": torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]),
        "damage_mode_ids": torch.tensor([[1, 3, 1], [1, 0, 0]]),
    }


def mask_ready_batch() -> dict[str, torch.Tensor]:
    data = batch()
    data["attack_energy_mask"] = torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    data["attack_base_damage"] = data["base_damage"].clone()
    data["attack_damage_mode"] = data["damage_mode_ids"].clone()
    data["attack_damage_mask"] = data["base_damage_mask"].clone()
    return data


def test_rule_mask_clears_tera_alias_tags_but_preserves_unrelated_tags() -> None:
    data = mask_ready_batch()
    masked = mask_training_inputs(
        data,
        schema(),
        {
            "categorical_probability": 0.0,
            "numeric_probability": 0.0,
            "rule_probability": 1.0,
            "card_tag_probability": 0.0,
            "detail_field_probability": 0.0,
            "text_token_probability": 0.0,
        },
        seed=17,
    )
    tags = masked.batch["card_tag_multihot"]
    vocab = schema()["card_tag_vocab"]
    assert tags[0, vocab["TERA"]].item() == 0.0
    assert tags[0, vocab["TERA_TYPE_FIRE"]].item() == 0.0
    assert tags[0, vocab["ANCIENT"]].item() == 1.0
    assert tags[1, vocab["FOSSIL"]].item() == 1.0
    assert torch.equal(masked.card_masks["rule_flags"], torch.tensor([True, True]))
    assert not masked.card_masks["card_tags"].any()


def test_masked_card_loss_ignores_every_unmasked_target() -> None:
    data = batch()
    encoder = CardEncoder(schema(), dropout=0.0)
    output = encoder(data, return_details=True)
    heads = MaskedCardFieldHeads(schema())
    predictions = heads(output.card_summary)
    masks = {"stage": torch.tensor([1, 0], dtype=torch.bool)}
    loss_a, metrics = masked_card_field_loss(predictions, data, masks)
    changed = dict(data)
    changed["stage_ids"] = torch.tensor([1, 2])
    loss_b, _ = masked_card_field_loss(predictions, changed, masks)
    assert torch.allclose(loss_a, loss_b)
    assert metrics["active_loss_count"] == 1.0

    zero, zero_metrics = masked_card_field_loss(predictions, data, {})
    assert zero.item() == 0.0
    assert zero_metrics["active_loss_count"] == 0.0


def test_detail_losses_follow_source_positions_and_only_explicit_masks() -> None:
    data = batch()
    encoder = CardEncoder(schema(), dropout=0.0)
    output = encoder(data, return_details=True)
    heads = MaskedDetailHeads(schema())
    predictions = heads(output.detail_tokens, output.text_token_states)
    detail_position = torch.tensor([[0, 1, 0], [0, 0, 0]], dtype=torch.bool)
    token_position = torch.zeros_like(data["detail_text_ids"], dtype=torch.bool)
    token_position[0, 1, 1] = True
    masks = {
        "energy_counts": detail_position,
        "base_damage": detail_position,
        "damage_mode": detail_position,
    }
    labels = data["detail_text_ids"].clone()
    loss_a, metrics = masked_detail_loss(
        predictions,
        data,
        masks,
        mlm_labels=labels,
        mlm_mask=token_position,
    )
    changed = dict(data)
    changed_energy = data["attack_energy_counts"].clone()
    changed_energy[0, 0] = 99.0  # unmasked CARD_EFFECT at source position zero
    changed["attack_energy_counts"] = changed_energy
    changed_labels = labels.clone()
    changed_labels[0, 0, 0] = 23
    loss_b, _ = masked_detail_loss(
        predictions,
        changed,
        masks,
        mlm_labels=changed_labels,
        mlm_mask=token_position,
    )
    assert torch.allclose(loss_a, loss_b)
    assert metrics["active_loss_count"] == 4.0
    assert torch.isfinite(loss_a)


def test_v2_pretrain_heads_backward() -> None:
    data = batch()
    encoder = CardEncoder(schema(), dropout=0.0)
    output = encoder(data, return_details=True)
    card_heads = MaskedCardFieldHeads(schema())
    detail_heads = MaskedDetailHeads(schema())
    ownership_head = CardDetailOwnershipHead()
    relation_head = CardRelationHead()

    card_predictions = card_heads(output.card_summary)
    card_loss, _ = masked_card_field_loss(
        card_predictions,
        data,
        {
            "stage": torch.tensor([1, 0], dtype=torch.bool),
            "rule_flags": torch.tensor([0, 1], dtype=torch.bool),
            "printed_hp": torch.tensor([1, 0], dtype=torch.bool),
        },
    )
    detail_predictions = detail_heads(output.detail_tokens, output.text_token_states)
    detail_position = torch.tensor([[0, 1, 0], [0, 0, 0]], dtype=torch.bool)
    token_position = torch.zeros_like(data["detail_text_ids"], dtype=torch.bool)
    token_position[0, 1, 1] = True
    detail_loss, _ = masked_detail_loss(
        detail_predictions,
        data,
        {
            "energy_counts": detail_position,
            "base_damage": detail_position,
            "damage_mode": detail_position,
        },
        mlm_labels=data["detail_text_ids"],
        mlm_mask=token_position,
    )

    ownership_logits = ownership_head(output.card_summary, output.detail_tokens)
    ownership_loss = card_detail_ownership_loss(
        ownership_logits,
        torch.tensor([[1.0, 1.0, 1.0], [1.0, 0.0, 0.0]]),
        data["detail_mask"],
    )
    relation_predictions = relation_head(output.card_summary, output.card_summary.flip(0))
    relation_labels = {
        "same_name": torch.tensor([0.0, 0.0]),
        "same_species": torch.tensor([0.0, 0.0]),
        "direct_evolution": torch.tensor([1.0, 0.0]),
    }
    relation_masks = {name: torch.ones(2, dtype=torch.bool) for name in relation_labels}
    relation_loss, relation_metrics = card_relation_loss(
        relation_predictions,
        relation_labels,
        relation_masks,
    )

    total = card_loss + detail_loss + ownership_loss + relation_loss
    total.backward()
    assert torch.isfinite(total)
    assert relation_metrics["active_loss_count"] == 3.0
    assert card_heads.rule_flags.weight.grad is not None
    assert detail_heads.mlm.weight.grad is not None
    assert ownership_head.net[0].weight.grad is not None
    assert relation_head.heads["direct_evolution"].weight.grad is not None
