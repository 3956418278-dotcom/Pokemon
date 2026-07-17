from __future__ import annotations

from pathlib import Path

import pytest

from static_card.data.card_dataset import (
    MASK_TOKEN,
    NULL_TOKEN,
    PAD_TOKEN,
    UNK_TOKEN,
    CardDataset,
    collate_cards,
    split_train_validation_test,
)
from static_card.data.card_preprocessing import DEFAULT_CACHE_DIR, ENERGY_TYPES


@pytest.fixture(scope="module")
def dataset() -> CardDataset:
    return CardDataset.from_cache(DEFAULT_CACHE_DIR)


def _item(dataset: CardDataset, card_id: int) -> dict:
    canonical_index = dataset.card_id_to_index[str(card_id)]
    return dataset[dataset.indices.index(canonical_index)]


def test_schema_identity_and_canonical_aggregation(dataset: CardDataset) -> None:
    for name in [
        "name_vocab", "major_role_vocab", "card_type_vocab", "subtype_vocab", "stage_vocab",
        "rule_flag_vocab", "category_vocab", "evolution_name_vocab",
    ]:
        vocab = dataset.schema[name]
        assert [vocab[token] for token in [PAD_TOKEN, NULL_TOKEN, UNK_TOKEN, MASK_TOKEN]] == [0, 1, 2, 3]

    assert len(dataset) == 1267
    assert len({record["card_id"] for record in dataset.records}) == len(dataset)
    item = _item(dataset, 1180)
    assert "card_id" not in item["card"]
    detail_types = [dataset.details[row["detail_index"]]["detail_type"] for row in item["details"]]
    assert detail_types == ["CARD_EFFECT", "ATTACK"]
    attack = item["details"][1]
    raw_attack = dataset.details[attack["detail_index"]]
    assert attack["attack_energy_counts"] == [float(value) for value in raw_attack["energy_counts"]]
    assert attack["damage_raw_text"] == raw_attack["damage_raw"]


def test_fossil_ability_energy_and_empty_basic_details(dataset: CardDataset) -> None:
    fossil = _item(dataset, 1099)
    fossil_record = dataset.records[dataset.card_id_to_index["1099"]]
    assert fossil_record["hp"] == 60
    assert fossil_record["evolves_to"] == ["Lileep"]
    fossil_details = [dataset.details[row["detail_index"]] for row in fossil["details"]]
    assert [(row["detail_type"], row["detail_name"]) for row in fossil_details] == [
        ("CARD_EFFECT", ""), ("ABILITY", "Primal Root")
    ]
    assert fossil_record["source_fields"]["Category"] == "Fossil"

    basic = _item(dataset, 1)
    special = _item(dataset, 15)
    special_record = dataset.records[dataset.card_id_to_index["15"]]
    assert basic["details"] == []
    assert special_record["provided_energy_amount"] == 2
    assert special_record["provided_energy_allowed_types"] == ["P", "D"]
    assert special_record["provided_energy_mode"] == "CHOOSE_ANY_COMBINATION"
    assert special_record["attachment_restriction"] == "TEAM_ROCKET_POKEMON_ONLY"
    assert special_record["invalid_attachment_effect"] == "DISCARD"
    allowed_mask = special["card"]["provided_energy_allowed_type_mask"]
    assert allowed_mask[ENERGY_TYPES.index("P")] == 1
    assert allowed_mask[ENERGY_TYPES.index("D")] == 1
    assert allowed_mask[ENERGY_TYPES.index("A")] == 0
    assert sum(allowed_mask) == 2
    assert special["card"]["provided_energy_counts"] == [0.0] * len(ENERGY_TYPES)
    assert special["card"]["provided_energy_amount"] == 2
    assert special["card"]["provided_energy_mask"] == 1.0
    neo_upper = _item(dataset, 10)["card"]
    ignition = _item(dataset, 17)["card"]
    assert neo_upper["provided_energy_counts"][ENERGY_TYPES.index("A")] == 2
    assert neo_upper["provided_energy_allowed_type_mask"][ENERGY_TYPES.index("A")] == 1
    assert ignition["provided_energy_counts"][ENERGY_TYPES.index("C")] == 3
    assert ignition["provided_energy_allowed_type_mask"][ENERGY_TYPES.index("C")] == 1
    batch = collate_cards([basic, fossil, special])
    assert batch["detail_mask"][0].sum().item() == 0
    assert batch["detail_mask"][1].sum().item() == 2
    assert batch["detail_text_ids"].shape[:2] == batch["detail_mask"].shape


def test_multiple_attacks_remain_independent_and_padding_is_dynamic(dataset: CardDataset) -> None:
    candidate = next(
        index for index, record in enumerate(dataset.records)
        if sum(dataset.details[pos]["detail_type"] == "ATTACK" for pos in range(record["detail_start"], record["detail_end"])) >= 2
    )
    item = dataset[dataset.indices.index(candidate)]
    attacks = [detail for detail in item["details"] if dataset.details[detail["detail_index"]]["detail_type"] == "ATTACK"]
    assert len(attacks) >= 2
    assert len({detail["detail_index"] for detail in attacks}) == len(attacks)
    for encoded in attacks:
        raw = dataset.details[encoded["detail_index"]]
        identity = f"ATTACK::{raw['detail_name']}"
        assert encoded["detail_identity_id"] == dataset.schema["detail_identity_vocab"][identity]
        assert encoded["attack_energy_counts"] == [float(value) for value in raw["energy_counts"]]
        assert encoded["damage_raw_text"] == raw["damage_raw"]

    batch = collate_cards([_item(dataset, 1), item])
    assert batch["detail_mask"].shape[1] == len(item["details"])
    assert batch["detail_text_ids"].shape[2] >= 1


def test_card_id_split_keeps_all_details_with_parent(dataset: CardDataset) -> None:
    first = split_train_validation_test(dataset, seed=73)
    second = split_train_validation_test(dataset, seed=73)
    assert [part.indices for part in first] == [part.indices for part in second]
    index_sets = [set(part.indices) for part in first]
    assert not (index_sets[0] & index_sets[1] or index_sets[0] & index_sets[2] or index_sets[1] & index_sets[2])
    assert set.union(*index_sets) == set(dataset.indices)
    for part in first:
        # This split monitors optimization within one frozen card pool; it is
        # deliberately not an unseen-card/train-only-vocabulary benchmark.
        assert part.schema is dataset.schema
        for canonical_index in part.indices[:20]:
            record = dataset.records[canonical_index]
            assert all(
                detail["card_id"] == record["card_id"]
                for detail in dataset.details[record["detail_start"] : record["detail_end"]]
            )
