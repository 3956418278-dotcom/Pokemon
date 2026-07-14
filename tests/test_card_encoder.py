from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from models.card_encoder import CardEncoder, CardEncoderOutput


def v2_schema() -> dict:
    return {
        "schema_version": "static_card_v2",
        "vocab": {
            "card_category": {"<PAD>": 0, "POKEMON": 1, "TRAINER": 2, "ENERGY": 3},
            "card_name": {"<PAD>": 0, "A": 1, "B": 2, "C": 3, "D": 4},
            "species": {"<PAD>": 0, "A": 1, "B": 2, "C": 3},
            "stage": {"<PAD>": 0, "BASIC": 1, "STAGE1": 2, "STAGE2": 3},
            "pokemon_type": {"<PAD>": 0, "G": 1, "R": 2, "W": 3},
            "weakness_type": {"<PAD>": 0, "G": 1, "R": 2, "W": 3},
            "resistance_type": {"<PAD>": 0, "G": 1, "R": 2, "W": 3},
            "trainer_subtype": {"<PAD>": 0, "ITEM": 1, "TOOL": 2},
            "energy_subtype": {"<PAD>": 0, "BASIC": 1, "SPECIAL": 2},
            "hp_applicability": {"<PAD>": 0, "POKEMON": 1, "PLAYABLE_AS_POKEMON": 2, "NOT_APPLICABLE": 3},
            "rule_flags": {"EX": 0, "TERA": 1},
            "card_tags": {"FOSSIL": 0, "ANCIENT": 1},
            "detail_type": {"<PAD>": 0, "ATTACK": 1, "ABILITY": 2, "CARD_EFFECT": 3},
            "detail_subtype": {"<PAD>": 0, "ATTACK": 1, "ABILITY": 2, "ITEM": 3, "TERA": 4},
            "damage_mode": {"<PAD>": 0, "NONE": 1, "FIXED": 2, "PLUS": 3, "TIMES": 4, "MINUS": 5},
        },
        "text_vocab_size": 32,
        "text_pad_id": 0,
        "text_mask_id": 1,
        "max_text_length": 6,
        "max_details": 4,
        "energy_type_count": 12,
        "energy_types": ["C", "G", "R", "W", "L", "P", "F", "D", "M", "N", "Y", "A"],
        "provided_energy_types": ["C", "G", "R", "W", "L", "P", "F", "D", "M", "N", "Y", "A", "TEAM_ROCKET"],
        "text_encoder_layers": 1,
    }


def v2_batch() -> dict[str, torch.Tensor]:
    # Detail order is intentionally mixed: attack/effect/ability and effect/attack.
    return {
        "card_category_ids": torch.tensor([1, 2, 3]),
        "card_name_ids": torch.tensor([1, 2, 3]),
        "species_ids": torch.tensor([1, 0, 0]),
        "previous_species_ids": torch.tensor([0, 0, 0]),
        "evolves_from_name_ids": torch.tensor([0, 0, 0]),
        "stage_ids": torch.tensor([1, 0, 0]),
        "pokemon_type_ids": torch.tensor([1, 0, 0]),
        "weakness_type_ids": torch.tensor([2, 0, 0]),
        "resistance_type_ids": torch.tensor([0, 0, 0]),
        "trainer_subtype_ids": torch.tensor([0, 2, 0]),
        "energy_subtype_ids": torch.tensor([0, 0, 1]),
        "hp_applicability_ids": torch.tensor([1, 2, 3]),
        "printed_hp": torch.tensor([100.0, 60.0, 0.0]),
        "printed_hp_mask": torch.tensor([1.0, 1.0, 0.0]),
        "retreat": torch.tensor([2.0, 0.0, 0.0]),
        "retreat_mask": torch.tensor([1.0, 0.0, 0.0]),
        "rule_flag_multihot": torch.tensor([[1.0, 0.0], [0.0, 0.0], [0.0, 0.0]]),
        "card_tag_multihot": torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 0.0]]),
        "provided_energy_counts": torch.tensor(
            [[0.0] * 13, [0.0] * 13, [0.0, 1.0] + [0.0] * 10 + [1.0]]
        ),
        "detail_type_ids": torch.tensor([[1, 3, 2], [3, 1, 0], [0, 0, 0]]),
        "detail_subtype_ids": torch.tensor([[1, 4, 2], [3, 1, 0], [0, 0, 0]]),
        "detail_mask": torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 0.0], [0.0, 0.0, 0.0]]),
        "detail_text_ids": torch.tensor(
            [
                [[2, 3, 4, 0], [5, 6, 0, 0], [7, 8, 9, 0]],
                [[10, 11, 0, 0], [12, 13, 14, 0], [0, 0, 0, 0]],
                [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            ]
        ),
        "detail_text_mask": torch.tensor(
            [
                [[1, 1, 1, 0], [1, 1, 0, 0], [1, 1, 1, 0]],
                [[1, 1, 0, 0], [1, 1, 1, 0], [0, 0, 0, 0]],
                [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            ],
            dtype=torch.float32,
        ),
        "attack_energy_counts": torch.tensor(
            [
                [[1.0] + [0.0] * 11, [0.0] * 12, [0.0] * 12],
                [[0.0] * 12, [0.0, 2.0] + [0.0] * 10, [0.0] * 12],
                [[0.0] * 12, [0.0] * 12, [0.0] * 12],
            ]
        ),
        "base_damage": torch.tensor([[30.0, 0.0, 0.0], [0.0, 90.0, 0.0], [0.0, 0.0, 0.0]]),
        "base_damage_mask": torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]),
        "damage_mode_ids": torch.tensor([[2, 1, 1], [1, 3, 0], [0, 0, 0]]),
    }


def slice_batch(batch: dict[str, torch.Tensor], row: int) -> dict[str, torch.Tensor]:
    return {name: value[row : row + 1].clone() for name, value in batch.items()}


def test_card_encoder_v2_shapes_and_source_order() -> None:
    batch = v2_batch()
    model = CardEncoder(v2_schema(), dropout=0.0)
    output = model(batch, return_details=True)
    assert isinstance(output, CardEncoderOutput)
    assert output.card_summary.shape == (3, 128)
    assert output.detail_tokens.shape == (3, 3, 128)
    assert output.pre_fusion_detail_tokens.shape == (3, 3, 128)
    assert output.text_token_states.shape == (3, 3, 4, 128)
    assert torch.equal(output.detail_type_ids, batch["detail_type_ids"])
    assert torch.equal(output.detail_mask, batch["detail_mask"])
    assert torch.count_nonzero(output.detail_tokens[batch["detail_mask"] == 0]) == 0
    assert torch.count_nonzero(output.text_token_states[batch["detail_text_mask"] == 0]) == 0


def test_trainer_branch_honors_fossil_hp_applicability() -> None:
    schema = v2_schema()
    model = CardEncoder(schema, dropout=0.0).eval()
    fossil = slice_batch(v2_batch(), 1)
    changed = {name: value.clone() for name, value in fossil.items()}
    changed["printed_hp"][:] = 120.0
    with torch.no_grad():
        fossil_summary = model(fossil)
        changed_summary = model(changed)
    assert not torch.allclose(fossil_summary, changed_summary)

    # A masked-out HP value is inapplicable and must not leak into the card token.
    fossil["printed_hp_mask"][:] = 0.0
    changed["printed_hp_mask"][:] = 0.0
    with torch.no_grad():
        masked_a = model(fossil)
        masked_b = model(changed)
    assert torch.allclose(masked_a, masked_b, atol=1e-6)


def test_card_encoder_backward_reaches_shared_text_and_all_detail_adapters() -> None:
    model = CardEncoder(v2_schema(), dropout=0.0)
    output = model(v2_batch(), return_details=True)
    loss = output.card_summary.square().mean() + output.detail_tokens.square().mean()
    loss.backward()
    assert model.text_encoder.token_embedding.weight.grad is not None
    assert model.attack_encoder.projection[0].weight.grad is not None
    assert model.ability_encoder.projection[0].weight.grad is not None
    assert model.card_effect_encoder.projection[0].weight.grad is not None
