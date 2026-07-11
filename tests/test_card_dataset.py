from __future__ import annotations

import pytest

pytest.importorskip("torch")

from data.card_dataset import CardDataset, collate_cards, split_record_indices
from data.card_dataset import make_feature_schema


def sample_records() -> list[dict]:
    return [
        {
            "card_id": "1",
            "name": "Sample Pokemon",
            "card_type": "POKEMON",
            "subtype": "Basic",
            "pokemon_type": "G",
            "stage": "BASIC",
            "hp": 70,
            "retreat_cost": 1,
            "weakness_type": "R",
            "weakness_value": 2.0,
            "resistance_type": None,
            "resistance_value": None,
            "evolves_from": None,
            "rule_flags": [],
            "ability_texts": ["Overgrow Draw a card."],
            "attack_texts": ["Hit once.", "Hit twice."],
            "attack_names": ["Leaf Hit", "Leaf Burst"],
            "attack_damage": [30, 90],
            "attack_energy_costs": [{"G": 1}, {"G": 2, "C": 1}],
            "trainer_type": None,
            "provided_energy_types": [],
            "full_effect_text": "",
            "attack_ids": [101, 102],
            "attacks": [
                {
                    "attack_id": "101",
                    "name": "Leaf Hit",
                    "effect_text": "Hit once.",
                    "damage_raw": "30",
                    "damage_value": 30.0,
                    "damage_mode": "fixed",
                    "energy_costs": {"G": 1},
                },
                {
                    "attack_id": "102",
                    "name": "Leaf Burst",
                    "effect_text": "Hit twice.",
                    "damage_raw": "90+",
                    "damage_value": 90.0,
                    "damage_mode": "plus",
                    "energy_costs": {"G": 2, "C": 1},
                },
            ],
            "abilities": [{"name": "Overgrow", "effect_text": "Draw a card."}],
            "special_effects": [],
        },
        {
            "card_id": "2",
            "name": "Grass Energy",
            "card_type": "BASIC_ENERGY",
            "subtype": "Basic Energy",
            "pokemon_type": None,
            "stage": None,
            "hp": None,
            "retreat_cost": None,
            "weakness_type": None,
            "weakness_value": None,
            "resistance_type": None,
            "resistance_value": None,
            "evolves_from": None,
            "rule_flags": [],
            "ability_texts": [],
            "attack_texts": [],
            "attack_names": [],
            "attack_damage": [],
            "attack_energy_costs": [],
            "trainer_type": None,
            "provided_energy_types": ["G"],
            "full_effect_text": "",
            "attack_ids": [],
            "attacks": [],
            "abilities": [],
            "special_effects": [],
        },
    ]


def test_card_dataset_batch_shapes() -> None:
    dataset = CardDataset.from_cache(rebuild=True)
    items = [dataset[i] for i in range(min(4, len(dataset)))]
    batch = collate_cards(items, dataset.schema)
    assert batch["single_cats"].shape[0] == len(items)
    assert batch["numeric"].shape[0] == len(items)
    assert batch["text_hashes"].shape[0] == len(items)
    assert len(batch["card_ids"]) == len(items)


def test_static_detail_schema_and_binding() -> None:
    import torch

    records = sample_records()
    schema = make_feature_schema(records)
    forbidden = {"mean_attack_damage", "mean_attack_energy_cost", "text_length", "energy_cost_G"}
    assert forbidden.isdisjoint(set(schema["numeric_fields"]))
    batch = collate_cards(
        [{"index": index, "card_id": record["card_id"], "record": record} for index, record in enumerate(records)],
        schema,
    )
    assert torch.equal(batch["attack_mask"][0], torch.tensor([1.0, 1.0]))
    assert batch["attack_mask"][0].sum().item() == 2
    assert batch["ability_mask"][0].sum().item() == 1
    assert batch["effect_mask"][0].sum().item() == 0
    assert batch["attack_energy_counts"][0, 0, 1].item() == 1.0
    assert batch["attack_energy_counts"][0, 1, 0].item() == 1.0
    assert batch["attack_energy_counts"][0, 1, 1].item() == 2.0
    assert batch["attack_raw_damage"][0, 0].item() == 30.0
    assert batch["attack_raw_damage"][0, 1].item() == 90.0
    assert batch["is_basic_energy"][1].item() == 1.0
    assert batch["provided_energy_multihot"][1, 1].item() == 1.0


def test_card_id_split_keeps_duplicate_ids_together() -> None:
    records = [{"card_id": "a"}, {"card_id": "a"}, {"card_id": "b"}, {"card_id": "c"}]
    train_idx, val_idx = split_record_indices(records, val_ratio=0.5, seed=1, mode="card_id")
    train_ids = {records[index]["card_id"] for index in train_idx}
    val_ids = {records[index]["card_id"] for index in val_idx}
    assert train_ids.isdisjoint(val_ids)
