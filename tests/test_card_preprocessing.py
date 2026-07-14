from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest

from data.card_preprocessing import (
    EXPECTED_SOURCE_SHA256,
    CardPreprocessingError,
    build_corpus,
    energy_cost_dict,
    load_csv_rows,
    load_csv_rows_with_hash,
    load_cg_data,
    load_or_create_corpus,
    normalize_energy_symbol,
    normalize_missing,
    parse_damage,
    parse_damage_mode,
    parse_energy_symbols,
    write_card_cache,
)


def test_strict_energy_and_damage_parsers_cover_real_notation() -> None:
    assert parse_energy_symbols("{G}{C}{C}") == ["C", "C", "G"]
    assert parse_energy_symbols("竜") == ["N"]
    assert energy_cost_dict("{G}●●") == {"C": 2, "G": 1}
    assert energy_cost_dict("No cost") == {}
    assert parse_damage("120+") == 120
    assert parse_damage_mode("120+") == "PLUS"
    assert parse_damage_mode("30×") == "TIMES"
    assert parse_damage_mode("-120") == "MINUS"
    assert normalize_energy_symbol("TEAM ROCKET") == "TEAM_ROCKET"
    with pytest.raises(CardPreprocessingError, match="unparsed energy"):
        energy_cost_dict("{G} mystery")
    with pytest.raises(CardPreprocessingError, match="damage"):
        parse_damage("many")


def test_full_corpus_v2_semantic_contract() -> None:
    cards, details, offsets, manifest = build_corpus()
    assert manifest["source_sha256"] == EXPECTED_SOURCE_SHA256
    assert len(cards) == 1267
    assert len(details) == 2014
    assert len(offsets) == 1268
    assert offsets[0] == 0
    assert offsets[-1] == len(details)
    assert manifest["detail_type_counts"] == {"ABILITY": 218, "ATTACK": 1556, "CARD_EFFECT": 240}
    assert manifest["card_category_counts"] == {"ENERGY": 20, "POKEMON": 1056, "TRAINER": 191}
    assert manifest["card_type_counts"]["TOOL"] == 27
    assert manifest["unresolved_count"] == 0

    card_by_id = {card["card_id"]: card for card in cards}
    detail_by_card: dict[str, list[dict]] = {}
    for detail in details:
        detail_by_card.setdefault(detail["card_id"], []).append(detail)

    non_pokemon = [card for card in cards if card["card_category"] != "POKEMON"]
    assert all(card["pokemon_type"] is None for card in non_pokemon)
    non_pokemon_details = [
        detail
        for card in non_pokemon
        for detail in detail_by_card.get(card["card_id"], [])
    ]
    assert not any(detail["detail_type"] == "ABILITY" for detail in non_pokemon_details)
    assert [
        (detail["card_id"], detail["move_name"], detail["attack_id"])
        for detail in non_pokemon_details
        if detail["detail_type"] == "ATTACK"
    ] == [("1180", "Geobuster", 1556)]

    expected_rule_flags = {
        "ACE_SPEC": 29,
        "MEGA_POKEMON_EX": 30,
        "POKEMON_EX": 121,
        "TERA": 32,
    }
    actual_rule_flags = {
        flag: sum(flag in card["rule_flags"] for card in cards)
        for flag in expected_rule_flags
    }
    assert actual_rule_flags == expected_rule_flags
    assert all(len(card["rule_flags"]) == len(set(card["rule_flags"])) for card in cards)
    assert {
        flag
        for card in cards
        for flag in card["rule_flags"]
    } == set(expected_rule_flags)

    fossils = [card for card in cards if "FOSSIL" in card["card_tags"]]
    assert len(fossils) == 5
    assert all(card["card_type"] == "ITEM" for card in fossils)
    assert all(card["printed_hp"] == 60 for card in fossils)
    assert all(card["hp_applicability"] == "PLAYABLE_AS_POKEMON" for card in fossils)
    assert all(len(detail_by_card[card["card_id"]]) == 2 for card in fossils)
    assert all(detail["detail_type"] == "CARD_EFFECT" for card in fossils for detail in detail_by_card[card["card_id"]])

    core = card_by_id["1180"]
    assert core["card_type"] == "TOOL"
    assert "TECHNICAL_MACHINE" in core["card_tags"]
    assert [detail["detail_type"] for detail in detail_by_card["1180"]] == ["CARD_EFFECT", "ATTACK"]
    geobuster = detail_by_card["1180"][1]
    assert geobuster["move_name"] == "Geobuster"
    assert geobuster["attack_id"] == 1556
    assert geobuster["energy_costs"] == {"F": 4}
    assert geobuster["base_damage"] == 350

    koraidon = detail_by_card["979"]
    assert [(row["move_name"], row["attack_id"]) for row in koraidon if row["detail_type"] == "ATTACK"] == [
        ("Orichalcum Fang", 1408),
        ("Impact Blow", 1409),
    ]
    corrected = [detail_by_card["480"][0], detail_by_card["481"][1]]
    assert corrected[0]["effect_text"].startswith("Flip a coin.")
    assert corrected[1]["effect_text"].startswith("You may search your deck")
    assert all(detail["source_effect_text"] != detail["effect_text"] for detail in corrected)
    corrections = manifest["known_effect_text_corrections"]
    assert [(row["card_id"], row["attack_id"]) for row in corrections] == [(480, 678), (481, 680)]
    for detail, correction in zip(corrected, corrections):
        assert correction["correction_id"]
        assert correction["reason"] == "english_csv_contains_japanese_attack_effect_text"
        assert correction["before_hash"] == hashlib.sha256(detail["source_effect_text"].encode("utf-8")).hexdigest()
        assert correction["after_hash"] == hashlib.sha256(detail["effect_text"].encode("utf-8")).hexdigest()
    assert sum(detail["detail_subtype"] == "POKEMON_TERA_RULE" for detail in details) == 32


def test_each_nonempty_source_detail_row_is_bound_once() -> None:
    rows, _source = load_csv_rows()
    _cards, details, _offsets, _manifest = build_corpus()
    expected = {
        index
        for index, row in enumerate(rows)
        if any(normalize_missing(row.get(column)) for column in ["Move Name", "Cost", "Damage", "Effect Explanation"])
    }
    actual = {int(detail["source_row_index"]) for detail in details}
    assert actual == expected
    assert len(actual) == len(details)


def test_v2_cache_uses_only_new_flat_files(tmp_path) -> None:
    manifest = write_card_cache(tmp_path)
    expected_files = {
        "cards.json",
        "details.json",
        "detail_offsets.json",
        "card_id_to_index.json",
        "preprocess_manifest.json",
    }
    assert {path.name for path in tmp_path.iterdir()} == expected_files
    cards, details, offsets, mapping, loaded_manifest = load_or_create_corpus(tmp_path)
    assert loaded_manifest == manifest
    assert len(cards) == len(mapping) == 1267
    assert offsets[-1] == len(details) == 2014
    assert json.loads((tmp_path / "details.json").read_text(encoding="utf-8"))[0] == details[0]


def test_unlisted_cg_text_mismatch_is_fatal() -> None:
    rows, source, source_hash = load_csv_rows_with_hash()
    cg_cards, cg_attacks = load_cg_data()
    bad_attacks = dict(cg_attacks)
    attack_id = next(attack_id for attack_id in sorted(bad_attacks) if attack_id not in {678, 680, 1408, 1409})
    replacement = SimpleNamespace(**vars(bad_attacks[attack_id]))
    replacement.text = f"{replacement.text} unexpected mutation"
    bad_attacks[attack_id] = replacement
    with pytest.raises(CardPreprocessingError, match="effect text mismatch outside known corrections"):
        build_corpus(
            rows=rows,
            source=source,
            source_sha256=source_hash,
            cg_cards=cg_cards,
            cg_attacks=bad_attacks,
        )
