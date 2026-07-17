from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from data.dynamic_card_dataset import AttackCostCatalog, StaticCardCatalog
from data.replay_dataset import ReplayDatasetSummary
from data.state_schema import AREA_IDS, CardInstanceState, GlobalSnapshot, ParsedObservation
from data.replay_feature_audit import build_report


def _catalog_files(tmp_path: Path) -> tuple[Path, Path, Path]:
    records = [
        {
            "card_id": 5,
            "name": "Basic Psychic Energy",
            "card_type": "BASIC_ENERGY",
            "type": "P",
            "detail_ids": [],
        },
        {
            "card_id": 10,
            "name": "Flexible Energy",
            "card_type": "SPECIAL_ENERGY",
            "type": None,
            "detail_ids": [],
        },
        {
            "card_id": 19,
            "name": "Fixed Psychic Energy",
            "card_type": "SPECIAL_ENERGY",
            "type": None,
            "detail_ids": [],
        },
        {
            "card_id": 21,
            "name": "Two Attack Pokemon",
            "card_type": "POKEMON",
            "type": "P",
            "detail_ids": [0, 1, 2],
        },
    ]
    details = {
        "cards": [
            {"card_id": str(card_id), "details": []}
            for card_id in (5, 10, 19)
        ]
        + [
            {
                "card_id": "21",
                "details": [
                    {"detail_index": 0, "detail_type": "attack", "attack_id": "1", "energy_counts": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]},
                    {"detail_index": 1, "detail_type": "attack", "attack_id": "2", "energy_counts": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]},
                    {"detail_index": 2, "detail_type": "ability", "ability_index": 0},
                ],
            }
        ]
    }
    mapping = {"5": 0, "10": 1, "19": 2, "21": 3}
    records_path = tmp_path / "card_records.json"
    details_path = tmp_path / "card_detail_metadata.json"
    mapping_path = tmp_path / "card_id_to_index.json"
    records_path.write_text(json.dumps(records), encoding="utf-8")
    details_path.write_text(json.dumps(details), encoding="utf-8")
    mapping_path.write_text(json.dumps(mapping), encoding="utf-8")
    (tmp_path / "card_embedding_metadata.json").write_text(
        json.dumps({"max_detail_count": 3, "card_count": 4}),
        encoding="utf-8",
    )
    return records_path, details_path, mapping_path


def _catalog(tmp_path: Path) -> AttackCostCatalog:
    return AttackCostCatalog.from_files(*_catalog_files(tmp_path))


def _pokemon(*, serial: int, energy_counts: list[int], energy_card_ids: list[int]) -> CardInstanceState:
    return CardInstanceState(
        card_id=21,
        serial=serial,
        player_index=0,
        relative_player=0,
        area=AREA_IDS["ACTIVE"],
        zone="active",
        slot=0,
        is_pokemon=True,
        hp=90,
        max_hp=120,
        energy_counts=energy_counts,
        energy_counts_valid=True,
        energy_card_count=len(energy_card_ids),
        energy_cards_valid=True,
        energy_card_ids=energy_card_ids,
        tool_count=0,
        tools_valid=True,
        pre_evolution_count=0,
        pre_evolution_valid=True,
        appear_this_turn_valid=True,
        special_conditions_valid=True,
        copy_count=1,
    )


def _sample(instances: list[CardInstanceState]) -> SimpleNamespace:
    memory = SimpleNamespace(appearance_features=lambda rows: [[0.0] * 32 for _ in rows])
    return SimpleNamespace(
        parsed=ParsedObservation(GlobalSnapshot(), instances, [], []),
        memory_after=memory,
        episode_id=101,
        replay_id="replay-101",
        source_path="pokemon-tcg-ai-battle-episodes-2026-07-10/101.json",
        source_date="2026-07-10",
        select_type=0,
        select_context=0,
    )


def test_attack_cost_catalog_aligns_details_and_costs(tmp_path: Path) -> None:
    catalog = _catalog(tmp_path)
    attacks = catalog.attack_details(21)
    assert [(row.detail_index, row.attack_id) for row in attacks] == [(0, 1), (1, 2)]
    assert attacks[0].energy_cost[0] == 1
    assert attacks[1].energy_cost[0] == 1
    assert attacks[1].energy_cost[5] == 1
    assert all(row.cost_known for row in attacks)
    assert catalog.max_details == 3


def test_payment_resolver_handles_colored_then_colorless() -> None:
    payable, remaining = AttackCostCatalog.resolve_payment([0, 0, 0, 0, 0, 2], [1, 0, 0, 0, 0, 1])
    assert payable
    assert sum(remaining) == 0
    payable, remaining = AttackCostCatalog.resolve_payment([0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 1])
    assert not payable
    assert remaining[0] == 1


def test_special_energy_resolution_is_conservative(tmp_path: Path) -> None:
    catalog = _catalog(tmp_path)
    counts = [0] * 12
    counts[5] = 1
    assert catalog.energy_payment_is_resolved(_pokemon(serial=1, energy_counts=counts, energy_card_ids=[5]))
    assert not catalog.energy_payment_is_resolved(_pokemon(serial=2, energy_counts=counts, energy_card_ids=[19]))
    assert not catalog.energy_payment_is_resolved(_pokemon(serial=3, energy_counts=counts, energy_card_ids=[10]))
    rainbow = [0] * 12
    rainbow[10] = 1
    assert not catalog.energy_payment_is_resolved(_pokemon(serial=4, energy_counts=rainbow, energy_card_ids=[19]))


def test_collator_builds_detail_level_targets_and_masks_unresolved(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    from data.dynamic_card_dataset import collate_dynamic_card_samples

    one_psychic = [0] * 12
    one_psychic[5] = 1
    samples = [
        _sample([_pokemon(serial=1, energy_counts=one_psychic, energy_card_ids=[5])]),
        _sample([_pokemon(serial=2, energy_counts=one_psychic, energy_card_ids=[10])]),
    ]
    batch = collate_dynamic_card_samples(samples, _catalog(tmp_path), max_details=3)
    assert batch.attack_detail_mask.shape == (2, 3)
    assert batch.detail_valid_mask.shape == (2, 3)
    assert batch.detail_valid_mask.tolist() == [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
    assert batch.attack_detail_mask.tolist() == [[1.0, 1.0, 0.0], [1.0, 1.0, 0.0]]
    assert batch.payment_supervision_mask[0].tolist() == [1.0, 1.0, 0.0]
    assert batch.payment_supervision_mask[1].sum().item() == 0.0
    assert batch.payable_targets[0].tolist() == [1.0, 0.0, 0.0]
    assert batch.energy_remaining_targets[0, 1, 0].item() == 1.0
    assert batch.hp_mask.tolist() == [1.0, 1.0]
    assert torch.isfinite(batch.attack_costs).all()


def test_catalog_rejects_mapping_without_card_record(tmp_path: Path) -> None:
    records, details, mapping = _catalog_files(tmp_path)
    mapping.write_text(json.dumps({"5": 0, "999": 1}), encoding="utf-8")
    with pytest.raises(ValueError, match="missing"):
        StaticCardCatalog.from_files(records, details, mapping)


def test_catalog_uses_physical_artifact_width_and_keeps_legacy_packed_indices_unverified(tmp_path: Path) -> None:
    records_path, details_path, mapping_path = _catalog_files(tmp_path)
    records = json.loads(records_path.read_text(encoding="utf-8"))
    records.extend(
        [
            {
                "card_id": 22,
                "name": "Ability Card",
                "card_type": "POKEMON",
                "detail_ids": [0, 1],
            },
            {
                "card_id": 23,
                "name": "Effect Card",
                "card_type": "ITEM",
                "detail_ids": [0, 1, 2],
            },
        ]
    )
    records_path.write_text(json.dumps(records), encoding="utf-8")
    detail_payload = json.loads(details_path.read_text(encoding="utf-8"))
    detail_payload["cards"].extend(
        [
            {
                "card_id": "22",
                "details": [
                    {"detail_index": 0, "detail_type": "ability", "ability_index": 0},
                    {"detail_index": 1, "detail_type": "ability", "ability_index": 1},
                ],
            },
            {
                "card_id": "23",
                "details": [
                    {"detail_index": 0, "detail_type": "special_effect", "effect_index": 0},
                    {"detail_index": 1, "detail_type": "special_effect", "effect_index": 1},
                    {"detail_index": 2, "detail_type": "special_effect", "effect_index": 2},
                ],
            },
        ]
    )
    details_path.write_text(json.dumps(detail_payload), encoding="utf-8")
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    mapping.update({"22": 4, "23": 5})
    mapping_path.write_text(json.dumps(mapping), encoding="utf-8")
    (tmp_path / "card_embedding_metadata.json").write_text(
        json.dumps({"max_detail_count": 7, "card_count": 6}),
        encoding="utf-8",
    )

    catalog = AttackCostCatalog.from_files(records_path, details_path, mapping_path)
    layout = catalog.static_catalog.detail_layout
    assert catalog.max_details == 7
    assert layout.metadata_packed_width == 3
    assert layout.type_capacities == {"attack": 2, "ability": 2, "special_effect": 3}
    assert layout.attack_slots_verified
    assert not layout.all_type_slots_verified
    assert [row.detail_index for row in catalog.attack_details(21)] == [0, 1]


def test_catalog_filters_null_attack_details_and_reports_alignment_anomaly(tmp_path: Path) -> None:
    records_path, details_path, mapping_path = _catalog_files(tmp_path)
    details = json.loads(details_path.read_text(encoding="utf-8"))
    card = next(row for row in details["cards"] if row["card_id"] == "21")
    card["details"] = [
        {"detail_index": 0, "detail_type": "attack", "attack_id": None, "attack_name": "[Ability] Fake"},
        {"detail_index": 1, "detail_type": "attack", "attack_id": "1", "energy_counts": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]},
        {"detail_index": 2, "detail_type": "attack", "attack_id": "2", "energy_counts": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]},
        {"detail_index": 3, "detail_type": "ability", "ability_index": 0},
    ]
    details_path.write_text(json.dumps(details), encoding="utf-8")
    (tmp_path / "card_embedding_metadata.json").write_text(
        json.dumps({"max_detail_count": 4, "card_count": 4}),
        encoding="utf-8",
    )

    catalog = AttackCostCatalog.from_files(records_path, details_path, mapping_path)
    assert [(row.detail_index, row.attack_id) for row in catalog.attack_details(21)] == [(1, 1), (2, 2)]
    assert catalog.invalid_detail_slots(21) == {0}
    assert catalog.invalid_detail_slots(5) == set()
    assert any(
        row["kind"] == "invalid_attack_detail" and row["card_id"] == 21
        for row in catalog.static_catalog.alignment_anomalies
    )


def test_collator_masks_only_null_attack_physical_slot(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    from data.dynamic_card_dataset import collate_dynamic_card_samples

    records_path, details_path, mapping_path = _catalog_files(tmp_path)
    details = json.loads(details_path.read_text(encoding="utf-8"))
    card = next(row for row in details["cards"] if row["card_id"] == "21")
    card["details"] = [
        {"detail_index": 0, "detail_type": "attack", "attack_id": None, "attack_name": "[Ability] Fake"},
        {"detail_index": 1, "detail_type": "attack", "attack_id": "1", "energy_counts": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]},
        {"detail_index": 2, "detail_type": "attack", "attack_id": "2", "energy_counts": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]},
        {"detail_index": 3, "detail_type": "ability", "ability_index": 0},
    ]
    details_path.write_text(json.dumps(details), encoding="utf-8")
    (tmp_path / "card_embedding_metadata.json").write_text(
        json.dumps({"max_detail_count": 4, "card_count": 4}),
        encoding="utf-8",
    )
    catalog = AttackCostCatalog.from_files(records_path, details_path, mapping_path)
    psychic = [0] * 12
    psychic[5] = 2
    batch = collate_dynamic_card_samples(
        [_sample([_pokemon(serial=9, energy_counts=psychic, energy_card_ids=[5, 5])])],
        catalog,
        max_details=4,
    )

    assert batch.detail_valid_mask.tolist() == [[0.0, 1.0, 1.0, 1.0]]
    assert batch.attack_detail_mask.tolist() == [[0.0, 1.0, 1.0, 0.0]]
    assert batch.payment_supervision_mask.tolist() == [[0.0, 1.0, 1.0, 0.0]]


def test_catalog_rejects_ambiguous_duplicate_attack_ids(tmp_path: Path) -> None:
    records_path, details_path, mapping_path = _catalog_files(tmp_path)
    details = json.loads(details_path.read_text(encoding="utf-8"))
    pokemon = next(row for row in details["cards"] if row["card_id"] == "21")
    pokemon["details"][1]["attack_id"] = 1
    details_path.write_text(json.dumps(details), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate attack_ids"):
        AttackCostCatalog.from_files(records_path, details_path, mapping_path)


def test_missing_attack_cost_is_unresolved_and_reported_not_misaligned(tmp_path: Path) -> None:
    records_path, details_path, mapping_path = _catalog_files(tmp_path)
    details = json.loads(details_path.read_text(encoding="utf-8"))
    pokemon = next(row for row in details["cards"] if row["card_id"] == "21")
    pokemon["details"][1]["energy_counts"] = None
    details_path.write_text(json.dumps(details), encoding="utf-8")
    catalog = AttackCostCatalog.from_files(records_path, details_path, mapping_path)
    assert [row.cost_known for row in catalog.attack_details(21)] == [True, False]
    assert any(row["kind"] == "attack_cost_missing" for row in catalog.static_catalog.alignment_anomalies)


def test_audit_is_fail_closed_without_static_artifacts_and_reports_detail_reasons(tmp_path: Path) -> None:
    catalog = _catalog(tmp_path)
    rainbow = [0] * 12
    rainbow[10] = 1
    sample = _sample([_pokemon(serial=7, energy_counts=rainbow, energy_card_ids=[19])])
    dataset = SimpleNamespace(
        samples=[sample],
        summary=ReplayDatasetSummary(replay_count=1, sample_count=1),
    )

    unchecked = build_report(dataset, set(), None)
    assert unchecked["static_lookup"]["checked"] is False
    assert unchecked["static_lookup"]["known"] is None
    assert unchecked["static_lookup"]["coverage"] is None
    assert unchecked["detail_alignment"]["coverage"] is None
    assert unchecked["energy_resolution"]["detail_unresolved_ratio"] is None
    assert unchecked["training_payment_supervision"]["attack_detail_count"] is None

    report = build_report(dataset, set(catalog.static_catalog.card_id_to_index), catalog)
    energy = report["energy_resolution"]
    assert energy["payment_candidate_details"] == 2
    assert energy["unresolved_details"] == 2
    assert energy["unresolved_reasons"] == {"special_energy_count_vector": 2}
    assert energy["special_energy_unresolved_details"] == 2
    assert energy["special_energy_unresolved_detail_ratio"] == 1.0
    assert report["detail_alignment"]["metadata_card_coverage"] == 1.0
    assert report["detail_alignment"]["coverage"] is None
    assert report["detail_alignment"]["physical_layout"]["physical_width"] == 3


def test_training_payment_supervision_matches_collator_rules_across_all_instances(tmp_path: Path) -> None:
    catalog = _catalog(tmp_path)
    psychic = [0] * 12
    psychic[5] = 2
    resolved = _pokemon(serial=1, energy_counts=psychic, energy_card_ids=[5, 5])
    hand = replace(
        resolved,
        serial=2,
        area=AREA_IDS["HAND"],
        zone="hand",
        is_pokemon=False,
        energy_counts_valid=False,
        energy_cards_valid=False,
        energy_card_count=0,
        energy_card_ids=[],
    )
    missing_energy = replace(
        resolved,
        serial=3,
        energy_counts_valid=False,
    )
    ambiguous_special = replace(
        resolved,
        serial=4,
        energy_counts=[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        energy_card_count=1,
        energy_card_ids=[10],
    )
    sample = _sample([resolved, hand, missing_energy, ambiguous_special])
    dataset = SimpleNamespace(
        samples=[sample],
        summary=ReplayDatasetSummary(replay_count=1, sample_count=1),
    )

    report = build_report(dataset, set(catalog.static_catalog.card_id_to_index), catalog)
    training = report["training_payment_supervision"]
    assert training["attack_detail_count"] == 8
    assert training["payment_supervision_count"] == 2
    assert training["unsupervised_attack_detail_count"] == 6
    assert training["supervision_ratio"] == 0.25
    assert training["unsupervised_ratio"] == 0.75
    assert training["unsupervised_reasons"] == {
        "not_pokemon": 2,
        "energy_counts_missing_or_invalid": 2,
        "ambiguous_special_energy_card": 2,
    }

    # The legacy energy-resolution section intentionally remains Pokémon-state-only.
    assert report["energy_resolution"]["payment_candidate_details"] == 6
    assert report["energy_resolution"]["supervised_details"] == 2
