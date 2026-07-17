from __future__ import annotations

from dataclasses import fields

import pytest

from static_card.data.card_dataset import (
    NULL_TOKEN,
    PAD_TOKEN,
    UNK_TOKEN,
    CardDataset,
    collate_cards,
    encode_card,
    split_train_validation_test,
)
from static_card.data.card_preprocessing import (
    CORPUS_SCHEMA_VERSION,
    DEFAULT_CACHE_DIR,
    CardRecord,
    DetailRecord,
)


@pytest.fixture(scope="module")
def dataset() -> CardDataset:
    return CardDataset.from_cache(DEFAULT_CACHE_DIR)


def _item(dataset: CardDataset, card_id: int) -> dict:
    canonical_index = dataset.card_id_to_index[str(card_id)]
    return dataset[dataset.indices.index(canonical_index)]


def test_corpus_and_card_record_contract(dataset: CardDataset) -> None:
    card_fields = [
        "card_id", "name", "card_type", "stage", "rule", "category", "type", "hp",
        "weakness_type", "resistance_type", "retreat_cost", "evolves_from", "evolves_to",
        "detail_ids", "expansion", "collection_no", "source_fields",
    ]
    assert [field.name for field in fields(CardRecord)] == card_fields
    assert dataset.manifest["card_record_fields"] == card_fields
    assert [field.name for field in fields(DetailRecord)] == dataset.manifest["detail_record_fields"]
    assert dataset.manifest["schema_version"] == CORPUS_SCHEMA_VERSION
    assert dataset.manifest["source_rows"] == 2022
    assert len(dataset.records) == len({record["card_id"] for record in dataset.records}) == 1267
    assert len(dataset.details) == 2014
    assert dataset.manifest["detail_type_counts"] == {"CARD_EFFECT": 235, "ATTACK": 1556, "ABILITY": 223}

    deleted = {
        "pokemon_type", "energy_profile", "basic_energy_type", "provided_energy_types",
        "provided_energy_counts", "provided_energy_amount", "provided_energy_allowed_types",
        "provided_energy_mode", "attachment_restriction", "invalid_attachment_effect",
        "rule_flags", "card_tags", "detail_start", "detail_end",
    }
    assert all(not (deleted & record.keys()) for record in dataset.records)


def test_shared_type_and_special_energy_details(dataset: CardDataset) -> None:
    actual_types = {
        value
        for record in dataset.records
        for value in (record["type"], record["weakness_type"], record["resistance_type"])
        if value is not None
    }
    assert actual_types == {"C", "G", "R", "W", "L", "P", "F", "D", "M", "DRAGON"}
    assert set(dataset.schema["type_vocab"]) == {PAD_TOKEN, NULL_TOKEN, UNK_TOKEN, *actual_types}
    assert "N" not in dataset.schema["type_vocab"]
    assert "Y" not in dataset.schema["type_vocab"]
    assert "TEAM_ROCKET" not in dataset.schema["type_vocab"]

    basic_fire = _item(dataset, 2)
    charizard = _item(dataset, 790)
    assert basic_fire["record"]["name"] == "Basic {R} Energy"
    assert basic_fire["record"]["card_type"] == "BASIC_ENERGY"
    assert basic_fire["record"]["type"] == "R"
    assert basic_fire["record"]["detail_ids"] == []
    assert charizard["record"]["name"] == "Mega Charizard X ex"
    assert charizard["record"]["card_type"] == "POKEMON"
    assert charizard["record"]["type"] == "R"
    assert basic_fire["card"]["type_id"] == charizard["card"]["type_id"] == dataset.schema["type_vocab"]["R"]

    special_cards = [record for record in dataset.records if record["card_type"] == "SPECIAL_ENERGY"]
    special_detail_ids = [detail_id for record in special_cards for detail_id in record["detail_ids"]]
    assert len(special_cards) == len(special_detail_ids) == len(set(special_detail_ids)) == 12
    assert all(record["type"] is None for record in special_cards)
    assert all(
        dataset.details[detail_id]["detail_id"] == dataset.details[detail_id]["detail_index"] == detail_id
        and dataset.details[detail_id]["detail_type"] == "CARD_EFFECT"
        and dataset.details[detail_id]["detail_subtype"] == "SPECIAL_ENERGY_EFFECT"
        for detail_id in special_detail_ids
    )

    for card_id in (10, 12, 15, 17):
        record = _item(dataset, card_id)["record"]
        assert record["type"] is None
        assert len(record["detail_ids"]) == 1
    team_rocket = _item(dataset, 15)["record"]
    assert team_rocket["source_fields"]["Type"] == "{Team Rocket}{Team Rocket}"


def test_non_detail_schema_and_fixed_numeric_scaling(dataset: CardDataset) -> None:
    for vocab_name in ["name_vocab", "card_type_vocab", "stage_vocab", "rule_vocab", "category_vocab", "type_vocab"]:
        vocab = dataset.schema[vocab_name]
        assert [vocab[token] for token in (PAD_TOKEN, NULL_TOKEN, UNK_TOKEN)] == [0, 1, 2]
    assert dataset.schema["numeric_scaling"] == {
        "mode": "fixed_divisors", "hp_divisor": 100.0, "retreat_divisor": 4.0,
    }
    assert "normalization" not in dataset.schema
    assert "energy_profile_vocab" not in dataset.schema
    assert "evolution_name_vocab" not in dataset.schema
    assert "energy_type_vocab" not in dataset.schema
    assert "detail_identity_vocab" not in dataset.schema

    expected_card_keys = {
        "name_id", "card_type_id", "stage_id", "rule_id", "category_id", "type_id",
        "weakness_type_id", "resistance_type_id", "hp_normalized", "hp_mask",
        "retreat_normalized", "retreat_mask", "evolves_from_name_id", "evolves_from_mask",
        "evolves_to_name_ids",
    }
    charizard = _item(dataset, 790)
    fossil = _item(dataset, 1099)
    basic_energy = _item(dataset, 2)
    assert set(charizard["card"]) == expected_card_keys
    assert charizard["record"]["hp"] == 360
    assert charizard["record"]["retreat_cost"] == 2
    assert charizard["card"]["hp_normalized"] == pytest.approx(3.6)
    assert charizard["card"]["retreat_normalized"] == pytest.approx(0.5)
    assert charizard["card"]["hp_mask"] == charizard["card"]["retreat_mask"] == 1.0
    assert fossil["record"]["hp"] == 60
    assert fossil["card"]["hp_normalized"] == pytest.approx(0.6)
    assert fossil["card"]["hp_mask"] == 1.0
    assert fossil["card"]["retreat_normalized"] == 0.0
    assert fossil["card"]["retreat_mask"] == 0.0
    assert basic_energy["card"]["hp_normalized"] == basic_energy["card"]["retreat_normalized"] == 0.0
    assert basic_energy["card"]["hp_mask"] == basic_energy["card"]["retreat_mask"] == 0.0

    zero_retreat = next(record for record in dataset.records if record["card_type"] == "POKEMON" and record["retreat_cost"] == 0)
    encoded_zero = encode_card(zero_retreat, dataset.schema)
    assert encoded_zero["retreat_normalized"] == 0.0
    assert encoded_zero["retreat_mask"] == 1.0


def test_name_category_rule_and_evolution_vocab_contract(dataset: CardDataset) -> None:
    name_vocab = dataset.schema["name_vocab"]
    assert all(record["name"] in name_vocab for record in dataset.records)
    assert all(
        value in name_vocab
        for record in dataset.records
        for value in [record["evolves_from"], *record["evolves_to"]]
        if value is not None
    )
    assert "Trainer's Pokémon（N）" in dataset.schema["category_vocab"]
    assert "Trainer's Pokémon（Team Rocket）" in dataset.schema["category_vocab"]
    assert "Tera(Fire)" in dataset.schema["category_vocab"]
    assert {record["rule"] for record in dataset.records} == {None, "POKEMON_EX", "MEGA_POKEMON_EX", "ACE_SPEC"}

    fossil = _item(dataset, 1099)
    assert fossil["record"]["evolves_to"] == ["Lileep"]
    assert fossil["card"]["evolves_to_name_ids"] == [name_vocab["Lileep"]]
    lileep = next(record for record in dataset.records if record["name"] == "Lileep")
    encoded_lileep = encode_card(lileep, dataset.schema)
    assert encoded_lileep["evolves_from_name_id"] == name_vocab["Antique Root Fossil"]
    assert encoded_lileep["evolves_from_mask"] == 1.0


def test_collate_detail_ids_shapes_and_loader_split(dataset: CardDataset) -> None:
    basic = _item(dataset, 2)
    fossil = _item(dataset, 1099)
    charizard = _item(dataset, 790)
    batch = collate_cards([basic, fossil, charizard])
    for field in [
        "name_ids", "card_type_ids", "stage_ids", "rule_ids", "category_ids", "type_ids",
        "weakness_type_ids", "resistance_type_ids", "hp_normalized", "hp_mask",
        "retreat_normalized", "retreat_mask", "evolves_from_name_ids", "evolves_from_mask",
    ]:
        assert batch[field].shape == (3,)
    assert batch["evolves_to_name_ids"].shape == batch["evolves_to_mask"].shape
    assert batch["evolves_to_mask"][1].sum().item() == 1.0
    assert "card_ids" not in batch and "card_indices" not in batch
    assert "detail_ids" in batch and "detail_indices" not in batch and "detail_identity_ids" not in batch
    assert batch["detail_ids"].shape == batch["detail_mask"].shape
    assert batch["detail_ids"][1, 0].item() == fossil["record"]["detail_ids"][0]

    for index, record in enumerate(dataset.records):
        start, end = dataset.detail_offsets[index : index + 2]
        assert record["detail_ids"] == list(range(start, end))
        assert all(dataset.details[detail_id]["card_id"] == record["card_id"] for detail_id in record["detail_ids"])

    first = split_train_validation_test(dataset, seed=73)
    second = split_train_validation_test(dataset, seed=73)
    assert [part.indices for part in first] == [part.indices for part in second]
    index_sets = [set(part.indices) for part in first]
    assert set.union(*index_sets) == set(dataset.indices)
    assert not (index_sets[0] & index_sets[1] or index_sets[0] & index_sets[2] or index_sets[1] & index_sets[2])
