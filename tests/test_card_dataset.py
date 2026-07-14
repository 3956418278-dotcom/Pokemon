from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from data.card_dataset import CardDataset, collate_cards, make_feature_schema, split_record_indices, text_tokens
from data.card_preprocessing import MASK_TOKEN, PAD_TOKEN, build_corpus
from models.card_pretrain_heads import CardRelationHead
from training.pretrain_card_encoder import build_relation_inputs


@pytest.fixture(scope="module")
def full_dataset() -> CardDataset:
    cards, details, offsets, _manifest = build_corpus()
    mapping = {card["card_id"]: index for index, card in enumerate(cards)}
    return CardDataset(cards, details, offsets, mapping, make_feature_schema(cards, details))


def test_schema_v2_exposes_complete_model_contract(full_dataset: CardDataset) -> None:
    schema = full_dataset.schema
    assert schema["schema_version"] == "static_card_v2"
    assert schema["text_vocab"][PAD_TOKEN] == schema["text_pad_id"] == 0
    assert schema["text_vocab"][MASK_TOKEN] == schema["text_mask_id"] == 1
    assert schema["text_vocab_size"] == len(schema["text_vocab"])
    assert schema["text_vocab_size"] == 556
    assert schema["max_text_tokens"] == 120
    assert schema["rule_flag_width"] == len(schema["rule_flag_vocab"])
    assert schema["rule_flag_width"] == 4
    assert schema["card_tag_width"] == len(schema["card_tag_vocab"])
    assert schema["card_tag_width"] == 27
    assert schema["move_name_is_model_input"] is False
    for field in [
        "card_category",
        "card_name",
        "species",
        "previous_species",
        "evolves_from_name",
        "stage",
        "trainer_subtype",
        "energy_subtype",
        "detail_type",
        "detail_subtype",
        "damage_mode",
    ]:
        assert schema["vocab_sizes"][field] == len(schema["vocab"][field])


def test_unified_source_order_batch_and_padding(full_dataset: CardDataset) -> None:
    indices = [full_dataset.card_id_to_index[card_id] for card_id in ["1180", "1099", "1"]]
    batch = collate_cards([full_dataset[full_dataset.indices.index(index)] for index in indices], full_dataset.schema)
    required = {
        "card_category_ids",
        "card_name_ids",
        "species_ids",
        "previous_species_ids",
        "evolves_from_name_ids",
        "stage_ids",
        "pokemon_type_ids",
        "weakness_type_ids",
        "resistance_type_ids",
        "trainer_subtype_ids",
        "energy_subtype_ids",
        "printed_hp",
        "printed_hp_mask",
        "retreat",
        "retreat_mask",
        "rule_flag_multihot",
        "card_tag_multihot",
        "provided_energy_counts",
        "detail_type_ids",
        "detail_subtype_ids",
        "detail_mask",
        "detail_text_ids",
        "detail_text_mask",
        "attack_energy_counts",
        "attack_base_damage",
        "attack_damage_mode",
        "attack_damage_mask",
        "detail_metadata",
    }
    assert required <= set(batch)
    assert batch["detail_mask"].shape == (3, 2)
    assert batch["detail_mask"].tolist() == [[1.0, 1.0], [1.0, 1.0], [0.0, 0.0]]
    assert batch["detail_metadata"][0]["details"][0]["detail_type"] == "CARD_EFFECT"
    assert batch["detail_metadata"][0]["details"][1]["detail_type"] == "ATTACK"
    assert batch["detail_metadata"][0]["details"][1]["attack_id"] == 1556
    assert batch["detail_source_rows"][0, 0] < batch["detail_source_rows"][0, 1]
    assert batch["printed_hp_raw"][1].item() == 60.0
    assert batch["printed_hp_mask"][1].item() == 1.0
    assert batch["detail_global_indices"][2].tolist() == [-1, -1]


def test_move_name_is_audit_only_not_text_input(full_dataset: CardDataset) -> None:
    index = full_dataset.card_id_to_index["28"]  # Storehouse Hideaway ability
    item = full_dataset[index]
    detail = next(row for row in item["details"] if row["detail_type"] == "ABILITY")
    assert "storehouse" in text_tokens(detail["move_name"])
    assert "storehouse" not in text_tokens(detail["effect_text"])
    batch = collate_cards([item], full_dataset.schema)
    ids = batch["detail_text_ids"][0, 0][batch["detail_text_mask"][0, 0] > 0].tolist()
    inverse = {index: token for token, index in full_dataset.schema["text_vocab"].items()}
    encoded_tokens = [inverse[index] for index in ids]
    assert encoded_tokens == text_tokens(detail["effect_text"])
    assert "storehouse" not in encoded_tokens
    assert batch["detail_metadata"][0]["details"][0]["move_name"] == "[Ability] Storehouse Hideaway"


def test_card_id_split_and_subset_relations_do_not_escape_split(full_dataset: CardDataset) -> None:
    train_idx, val_idx = split_record_indices(full_dataset.cards, val_ratio=0.2, seed=17, mode="card_id")
    train_ids = {full_dataset.cards[index]["card_id"] for index in train_idx}
    val_ids = {full_dataset.cards[index]["card_id"] for index in val_idx}
    assert train_ids.isdisjoint(val_ids)
    subset = full_dataset.subset(train_idx[:100])
    allowed = set(subset.indices)
    assert all(left in allowed and right in allowed for pairs in subset.relation_samples().values() for left, right in pairs)


def test_fossil_evolves_to_pairs_are_directed_and_split_local(full_dataset: CardDataset) -> None:
    expected_ids = [
        ("1099", "146"),  # Antique Root Fossil -> Lileep
        ("1136", "503"),  # Antique Cover Fossil -> Tirtouga
        ("1138", "603"),  # Antique Plume Fossil -> Archen
        ("1150", "1053"),  # Antique Jaw Fossil -> Tyrunt
        ("1151", "1032"),  # Antique Sail Fossil -> Amaura
    ]
    expected = {
        (full_dataset.card_id_to_index[parent], full_dataset.card_id_to_index[child])
        for parent, child in expected_ids
    }
    assert expected <= set(full_dataset.relation_samples()["evolves_to"])

    # Prove these positives are sourced from the Fossil forward field itself,
    # not merely duplicated from each child's evolves_from field.
    forward_only_cards = [dict(card) for card in full_dataset.cards]
    for _parent, child in expected:
        forward_only_cards[child]["evolves_from_card_name"] = None
    forward_only = CardDataset(
        forward_only_cards,
        full_dataset.details,
        full_dataset.detail_offsets,
        full_dataset.card_id_to_index,
        full_dataset.schema,
    )
    assert expected <= set(forward_only.relation_samples()["evolves_to"])

    both_ends = sorted({index for pair in expected for index in pair})
    local_pairs = set(full_dataset.subset(both_ends).relation_samples()["evolves_to"])
    assert expected <= local_pairs

    parents_only = sorted(parent for parent, _child in expected)
    assert set(full_dataset.subset(parents_only).relation_samples()["evolves_to"]).isdisjoint(expected)


def test_relation_inputs_mask_name_and_species_identity_aliases(full_dataset: CardDataset) -> None:
    relation = build_relation_inputs(full_dataset, batch_size=64, seed=20260713)
    assert relation is not None
    aliases = (
        ("card_name_ids", "card_name"),
        ("species_ids", "species"),
        ("previous_species_ids", "previous_species"),
        ("evolves_from_name_ids", "evolves_from_name"),
        ("evolves_to_name_ids", "evolves_to_name"),
    )
    for relation_name in CardRelationHead.RELATIONS:
        rows = relation.masks[relation_name]
        assert rows.any()
        expected_aliases = aliases[:2] if relation_name == "same_name" else aliases
        for side in (relation.left, relation.right):
            for batch_key, vocab_key in expected_aliases:
                mask_id = full_dataset.schema["vocab"][vocab_key][MASK_TOKEN]
                assert (side[batch_key][rows] == mask_id).all()


def test_relation_negatives_are_split_local_deterministic_and_signature_matched(
    full_dataset: CardDataset,
) -> None:
    train_indices, _validation_indices = split_record_indices(
        full_dataset.cards,
        val_ratio=0.2,
        seed=29,
        mode="card_id",
    )
    split = full_dataset.subset(train_indices)
    relation = build_relation_inputs(split, batch_size=120, seed=20260713)
    repeated = build_relation_inputs(split, batch_size=120, seed=20260713)
    assert relation is not None and repeated is not None
    assert relation.diagnostics == repeated.diagnostics
    assert torch.equal(relation.left["card_index"], repeated.left["card_index"])
    assert torch.equal(relation.right["card_index"], repeated.right["card_index"])

    allowed = set(split.indices)
    positives = split.relation_samples()
    positive_sets = {
        "same_name": set(positives["same_name"]),
        "same_species": set(positives["same_species"]),
        "direct_evolution": set(positives["evolves_to"]),
    }

    def signature(index: int, keys: tuple[str, ...]) -> tuple[object, ...]:
        return tuple(split.cards[index].get(key) for key in keys)

    direct_signatures = {
        (
            signature(left, ("card_category", "card_type", "stage")),
            signature(right, ("card_category", "card_type", "stage")),
        )
        for left, right in positive_sets["direct_evolution"]
    }
    same_name_signatures = {
        (
            signature(left, ("card_category", "card_type", "stage")),
            signature(right, ("card_category", "card_type", "stage")),
        )
        for left, right in positive_sets["same_name"]
    }
    same_species_signatures = {
        (
            signature(left, ("stage", "pokemon_type")),
            signature(right, ("stage", "pokemon_type")),
        )
        for left, right in positive_sets["same_species"]
    }
    for relation_name in CardRelationHead.RELATIONS:
        relation_rows = relation.masks[relation_name]
        positive_rows = relation_rows & (relation.labels[relation_name] == 1)
        negative_rows = relation_rows & (relation.labels[relation_name] == 0)
        assert int(positive_rows.sum().item()) == int(negative_rows.sum().item())
        assert negative_rows.any(), relation_name
        left_indices = relation.left["card_index"][negative_rows].tolist()
        right_indices = relation.right["card_index"][negative_rows].tolist()
        for left, right in zip(left_indices, right_indices):
            pair = (int(left), int(right))
            assert pair[0] in allowed and pair[1] in allowed
            assert pair[0] != pair[1]
            assert pair not in positive_sets[relation_name]
            if relation_name == "same_name":
                keys = ("card_category", "card_type", "stage")
                assert (signature(pair[0], keys), signature(pair[1], keys)) in same_name_signatures
            elif relation_name == "same_species":
                assert split.cards[pair[0]]["card_category"] == "POKEMON"
                assert split.cards[pair[1]]["card_category"] == "POKEMON"
                keys = ("stage", "pokemon_type")
                assert (signature(pair[0], keys), signature(pair[1], keys)) in same_species_signatures
            else:
                keys = ("card_category", "card_type", "stage")
                assert (signature(pair[0], keys), signature(pair[1], keys)) in direct_signatures

    assert not any(name.endswith("_unmatched") for name in relation.diagnostics)
