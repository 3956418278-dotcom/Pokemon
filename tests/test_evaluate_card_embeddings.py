from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import training.evaluate_card_embeddings as evaluator
from training.evaluate_card_embeddings import (
    EVALUATION_SCHEMA_VERSION,
    EvaluationError,
    FROZEN_DIAGNOSTICS,
    ONLINE_REQUIRED_GATES,
    _categorical_probe,
    _classification_recovery_report,
    _fit_standardizer,
    _regression_recovery_report,
    _semantic_contract,
    build_argument_parser,
    evaluate_card_embeddings,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def _build_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    cache = tmp_path / "cache"
    artifacts = tmp_path / "artifacts"
    cache.mkdir()
    artifacts.mkdir()
    cards: list[dict] = []
    details: list[dict] = []
    offsets = [0]
    card_embeddings: list[list[float]] = []
    detail_embeddings: list[list[float]] = []
    split_indices = {"train": [], "validation": [], "test": []}
    split_names = ("train", "validation", "test")

    for split_number, split in enumerate(split_names):
        for local in range(30):
            index = len(cards)
            split_indices[split].append(index)
            group = local // 4
            within = local % 4
            if local < 24:
                category = "POKEMON"
                is_evolution = within >= 2
                root_name = f"Base-{split_number}-{group}"
                card_name = (
                    f"Evo-{split_number}-{group}"
                    if is_evolution
                    else root_name
                )
                species = card_name
                stage = "STAGE_1" if is_evolution else "BASIC"
                pokemon_type = ("G", "R", "W")[group % 3]
                previous = root_name if is_evolution else None
                hp = 100 + 20 * int(is_evolution) + 10 * (group % 3)
                retreat = group % 3
            elif local < 27:
                category = "TRAINER"
                card_name = f"Trainer-{split_number}-{local}"
                species = stage = pokemon_type = previous = hp = retreat = None
            else:
                category = "ENERGY"
                card_name = f"Energy-{split_number}-{local}"
                species = stage = pokemon_type = previous = hp = retreat = None
            flags = ["EX"] if local % 5 == 0 else []
            cards.append(
                {
                    "card_id": str(index + 1),
                    "card_name": card_name,
                    "card_category": category,
                    "card_type": category,
                    "card_tags": [],
                    "stage": stage,
                    "pokemon_type": pokemon_type,
                    "species": species,
                    "evolves_from_card_name": previous,
                    "evolves_to_card_name": None,
                    "printed_hp": hp,
                    "retreat": retreat,
                    "rule_flags": flags,
                }
            )
            category_id = {"POKEMON": 0.0, "TRAINER": 1.0, "ENERGY": 2.0}[category]
            signature = [((index + 3) * prime % 17) / 17.0 for prime in (2, 3, 5, 7)]
            card_vector = [0.0] * 16
            card_vector[int(category_id)] = 1.0
            card_vector[3] = float(stage == "STAGE_1")
            card_vector[4] = float({"G": 0, "R": 1, "W": 2}.get(pokemon_type, 0))
            card_vector[5] = float(hp or 0) / 200.0
            card_vector[6] = float(retreat or 0)
            card_vector[7] = float(bool(flags))
            card_vector[8] = float(split_number * 10 + group) if species else -1.0
            card_vector[12:16] = signature
            card_embeddings.append(card_vector)

            row_types = ("ATTACK", "ABILITY", "CARD_EFFECT")
            for local_detail, detail_type in enumerate(row_types):
                global_index = len(details)
                attack = detail_type == "ATTACK"
                damage_mode = ("FIXED", "PLUS", "TIMES")[group % 3] if attack else "NONE"
                detail = {
                    "global_detail_index": global_index,
                    "local_detail_index": local_detail,
                    "card_index": index,
                    "card_id": str(index + 1),
                    "source_row_index": global_index + 100,
                    "detail_type": detail_type,
                    "move_name": f"detail-{global_index}",
                    "attack_id": global_index + 1 if attack else None,
                    "energy_costs": ({"C": group % 3, "G": int(group % 2 == 0)} if attack else {}),
                    "base_damage": (20 + group * 10 if attack else None),
                    "damage_mode": damage_mode,
                }
                details.append(detail)
                detail_vector = [0.0] * 16
                detail_vector[local_detail] = 1.0
                detail_vector[3] = float(detail["energy_costs"].get("C", 0))
                detail_vector[4] = float(detail["energy_costs"].get("G", 0))
                detail_vector[5] = float(detail["base_damage"] or 0) / 100.0
                detail_vector[6] = float({"NONE": 0, "FIXED": 1, "PLUS": 2, "TIMES": 3}[damage_mode])
                detail_vector[12:16] = signature
                detail_embeddings.append(detail_vector)
            offsets.append(len(details))

    mapping = {card["card_id"]: index for index, card in enumerate(cards)}
    _write_json(cache / "cards.json", cards)
    _write_json(cache / "details.json", details)
    _write_json(cache / "detail_offsets.json", offsets)
    _write_json(cache / "card_id_to_index.json", mapping)
    _write_json(cache / "preprocess_manifest.json", {"schema_version": "static_card_v2"})

    split_manifest = {
        "schema_version": "static_card_split_v2",
        "seed": 17,
        "mode": "card_id",
        "transductive_catalog_schema": True,
        "transductive_note": "fixture catalog schema",
    }
    for split, indices in split_indices.items():
        prefix = "validation" if split == "validation" else split
        split_manifest[f"{prefix}_indices"] = indices
        split_manifest[f"{prefix}_card_ids"] = [cards[index]["card_id"] for index in indices]
    split_path = tmp_path / "split_manifest.json"
    _write_json(split_path, split_manifest)

    checkpoint_path = tmp_path / "selection.pt"
    torch.save(
        {
            "stage": "split_selection_best",
            "lineage": {"split_manifest_sha256": _sha256(split_path)},
        },
        checkpoint_path,
    )
    tensors = {
        "card_embeddings.pt": torch.tensor(card_embeddings, dtype=torch.float32),
        "detail_embeddings.pt": torch.tensor(detail_embeddings, dtype=torch.float32),
        "detail_offsets.pt": torch.tensor(offsets, dtype=torch.long),
        "detail_type_ids.pt": torch.tensor(
            [{"ATTACK": 1, "ABILITY": 2, "CARD_EFFECT": 3}[row["detail_type"]] for row in details],
            dtype=torch.long,
        ),
    }
    for filename, tensor in tensors.items():
        torch.save(tensor, artifacts / filename)
    metadata = [
        {
            "global_detail_index": row["global_detail_index"],
            "local_detail_index": row["local_detail_index"],
            "card_index": row["card_index"],
            "card_id": row["card_id"],
            "source_row": row["source_row_index"],
            "detail_type": row["detail_type"].lower(),
            "move_name": row["move_name"],
            "attack_id": row["attack_id"],
        }
        for row in details
    ]
    _write_json(artifacts / "detail_metadata.json", metadata)
    _write_json(artifacts / "card_id_to_index.json", mapping)
    files = {}
    for filename in (
        "card_embeddings.pt",
        "detail_embeddings.pt",
        "detail_offsets.pt",
        "detail_type_ids.pt",
        "detail_metadata.json",
        "card_id_to_index.json",
    ):
        path = artifacts / filename
        files[filename] = {"sha256": _sha256(path), "size_bytes": path.stat().st_size}
    _write_json(
        artifacts / "artifact_manifest.json",
        {
            "schema_version": "static_card_artifacts_v2",
            "source_schema_version": "static_card_v2",
            "model_version": "card_encoder_v2",
            "storage": "flat_details_with_offsets",
            "card_count": len(cards),
            "detail_count": len(details),
            "card_embedding_dim": 16,
            "detail_embedding_dim": 16,
            "checkpoint": {"path": str(checkpoint_path), "sha256": _sha256(checkpoint_path)},
            "detail_type_vocab": {"padding": 0, "attack": 1, "ability": 2, "card_effect": 3},
            "files": files,
        },
    )
    return cache, artifacts, split_path


def _fake_online_result(passed: bool) -> dict:
    gates = {
        name: {"name": name, "task": "fixture_online", "gate": "pass" if passed else "fail"}
        for name in ONLINE_REQUIRED_GATES
    }
    return {
        "protocol": {"target_masked_online_inference": True, "test_evaluations": 1},
        "gates": gates,
        "required_gates": list(ONLINE_REQUIRED_GATES),
        "failed_gates": [] if passed else list(ONLINE_REQUIRED_GATES),
        "passed": passed,
        "relations": {"status": "diagnostic_only"},
    }


def test_frozen_v2_evaluator_is_split_audited_and_structured(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cache, artifacts, split_path = _build_fixture(tmp_path)
    monkeypatch.setattr(evaluator, "_evaluate_online_checkpoint", lambda *_args, **_kwargs: _fake_online_result(False))
    output = tmp_path / "evaluation"
    report = evaluate_card_embeddings(
        cache_dir=cache,
        artifact_dir=artifacts,
        split_manifest_path=split_path,
        output_dir=output,
        seed=11,
        max_pairs_per_split=100,
    )

    assert report["schema_version"] == EVALUATION_SCHEMA_VERSION
    assert report["acceptance"]["hard_checks"]["artifact_alignment"] is True
    assert report["acceptance"]["hard_checks"]["split_partition"] is True
    assert report["acceptance"]["hard_checks"]["selection_checkpoint_lineage"] is True
    assert report["protocol"]["standardization_scope"] == "train_only"
    assert report["protocol"]["test_scope"] == "single_final_evaluation_after_selection"
    assert report["data_integrity"]["detail_split_counts"] == {
        "train": 90,
        "validation": 90,
        "test": 90,
    }
    assert report["probes"]["card_category"]["diagnostic_only"] is True
    assert report["probes"]["detail_type"]["diagnostic_only"] is True
    assert set(report["acceptance"]["diagnostic_only_probes"]) == FROZEN_DIAGNOSTICS
    assert all(row["gate"] == "diagnostic_only" for row in report["frozen_probes"].values())
    assert all(
        child["gate"] == "diagnostic_only"
        for child in report["frozen_probes"]["rule_flags"]["per_label"].values()
    )
    assert report["acceptance"]["passed"] is False
    assert report["acceptance"]["hard_checks"]["online_target_masked_recovery"] is False
    assert report["probes"]["same_species"]["fit_protocol"]["probe_fit"] == "train_only"
    assert "Frozen pair probe" in report["probes"]["same_species"]["limitations"][0]
    assert "leave-one-detail-out" in report["probes"]["ownership"]["limitations"][2]
    assert report["probes"]["attack_cost"]["outputs"]["A"]["status"] == "unsupported"
    saved = json.loads((output / "evaluation.json").read_text(encoding="utf-8"))
    assert saved == report


def test_split_overlap_is_rejected_before_probe_fitting(tmp_path: Path) -> None:
    cache, artifacts, split_path = _build_fixture(tmp_path)
    split = json.loads(split_path.read_text(encoding="utf-8"))
    split["test_indices"][0] = split["train_indices"][0]
    split["test_card_ids"][0] = split["train_card_ids"][0]
    _write_json(split_path, split)

    with pytest.raises(EvaluationError, match="multiple splits"):
        evaluate_card_embeddings(
            cache_dir=cache,
            artifact_dir=artifacts,
            split_manifest_path=split_path,
        )


def test_standardizer_and_ridge_selection_do_not_read_test_statistics() -> None:
    train = np.arange(120, dtype=np.float64).reshape(60, 2)
    validation = np.arange(40, dtype=np.float64).reshape(20, 2)
    test = np.full((20, 2), 1.0e9)
    mean, std = _fit_standardizer(train)
    assert np.allclose(mean, train.mean(axis=0))
    assert np.allclose(std, train.std(axis=0))

    labels = {
        "train": ["a", "b"] * 30,
        "validation": ["a", "b"] * 10,
        "test": ["a"] * 19 + ["rare"],
    }
    report = _categorical_probe(
        "rare_fixture",
        {"train": train, "validation": validation, "test": test},
        labels,
    )
    assert report["status"] == "low_confidence"
    assert any("absent from train" in reason for reason in report["confidence_reasons"])
    assert report["selection"]["criterion"] == "validation_balanced_accuracy"
    assert report["fit_protocol"]["test_evaluations"] == 1


def test_perfect_frozen_inputs_cannot_override_failed_online_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache, artifacts, split_path = _build_fixture(tmp_path)
    monkeypatch.setattr(
        evaluator,
        "_semantic_contract",
        lambda *_args, **_kwargs: {"passed": True, "checks": {"fixture": True}, "limitations": []},
    )
    monkeypatch.setattr(evaluator, "_evaluate_online_checkpoint", lambda *_args, **_kwargs: _fake_online_result(False))
    report = evaluate_card_embeddings(
        cache_dir=cache,
        artifact_dir=artifacts,
        split_manifest_path=split_path,
        seed=11,
        max_pairs_per_split=100,
    )
    assert report["frozen_probes"]["pokemon_stage"]["test"]["balanced_accuracy"] == 1.0
    assert all(row["gate"] == "diagnostic_only" for row in report["frozen_probes"].values())
    assert report["acceptance"]["passed"] is False
    assert "online_target_masked_recovery" in report["acceptance"]["failed_checks"]


def _semantic_fixture() -> tuple[list[dict], list[dict], dict]:
    cards = [
        {
            "card_id": str(index),
            "card_category": "POKEMON",
            "card_type": "POKEMON",
            "card_tags": [],
            "pokemon_type": "G",
            "printed_hp": 100,
            "hp_applicability": "POKEMON",
            "rule_flags": [],
        }
        for index in range(1, 1268)
    ]
    by_id = {card["card_id"]: card for card in cards}
    by_id["1180"].update(
        {
            "card_category": "TRAINER",
            "card_type": "TOOL",
            "card_tags": ["TECHNICAL_MACHINE"],
            "pokemon_type": None,
        }
    )
    fossil_ids = ("1099", "1136", "1138", "1150", "1151")
    for card_id in fossil_ids:
        by_id[card_id].update(
            {
                "card_type": "ITEM",
                "card_category": "TRAINER",
                "card_tags": ["FOSSIL"],
                "pokemon_type": None,
                "printed_hp": 60,
                "hp_applicability": "PLAYABLE_AS_POKEMON",
            }
        )
    for card in cards[:29]:
        card["rule_flags"].append("ACE_SPEC")
    for card in cards[:30]:
        card["rule_flags"].append("MEGA_POKEMON_EX")
    for card in cards[:121]:
        card["rule_flags"].append("POKEMON_EX")
    for card in cards[:32]:
        card["rule_flags"].append("TERA")

    details: list[dict] = []

    def add(card_id: str, detail_type: str, *, attack_id: int | None = None, move_name: str = "x", **extra: object) -> None:
        details.append(
            {
                "card_id": card_id,
                "card_index": int(card_id) - 1,
                "source_row_index": len(details),
                "local_detail_index": 0,
                "detail_type": detail_type,
                "attack_id": attack_id,
                "move_name": move_name,
                **extra,
            }
        )

    add("1180", "CARD_EFFECT", move_name="Technical Machine rule")
    for attack_id in range(1, 1557):
        card_id = "1"
        move_name = f"attack-{attack_id}"
        extra: dict[str, object] = {}
        if attack_id == 1556:
            card_id, move_name = "1180", "Geobuster"
            extra = {"energy_costs": {"F": 4}, "base_damage": 350}
        elif attack_id == 1408:
            card_id, move_name = "979", "Orichalcum Fang"
        elif attack_id == 1409:
            card_id, move_name = "979", "Impact Blow"
        add(card_id, "ATTACK", attack_id=attack_id, move_name=move_name, **extra)
    for _ in range(218):
        add("1", "ABILITY")
    for card_id in fossil_ids:
        add(card_id, "CARD_EFFECT")
        add(card_id, "CARD_EFFECT")
    for _ in range(229):
        add("1", "CARD_EFFECT")
    local_counts: dict[str, int] = {}
    for global_index, row in enumerate(details):
        row["global_detail_index"] = global_index
        local = local_counts.get(row["card_id"], 0)
        row["local_detail_index"] = local
        local_counts[row["card_id"]] = local + 1
    manifest = {
        "schema_version": "static_card_v2",
        "source_sha256": evaluator.EXPECTED_SOURCE_SHA256,
        "source_row_count": 2022,
        "card_count": 1267,
        "detail_count": 2014,
        "detail_type_counts": {"ABILITY": 218, "ATTACK": 1556, "CARD_EFFECT": 240},
        "unresolved_count": 0,
        "known_effect_text_corrections": [
            {
                "correction_id": "a",
                "reason": "english_csv_contains_japanese_attack_effect_text",
                "card_id": 480,
                "attack_id": 678,
                "before_hash": "before",
                "after_hash": "after",
            },
            {
                "correction_id": "b",
                "reason": "english_csv_contains_japanese_attack_effect_text",
                "card_id": 481,
                "attack_id": 680,
                "before_hash": "before",
                "after_hash": "after",
            },
        ],
    }
    assert len(details) == 2014
    return cards, details, manifest


def test_semantic_contract_rejects_consistently_aligned_but_wrong_cache() -> None:
    cards, details, manifest = _semantic_fixture()
    assert _semantic_contract(cards, details, manifest)["passed"] is True
    corrupted = [dict(row) for row in details]
    geobuster = next(row for row in corrupted if row.get("move_name") == "Geobuster")
    geobuster["base_damage"] = 35
    result = _semantic_contract(cards, corrupted, manifest)
    assert result["passed"] is False
    assert result["checks"]["core_memory_contract"] is False


def test_online_gate_requires_trained_improvement_over_untrained_and_label_baseline() -> None:
    train_labels = [0, 1] * 30
    target = np.asarray([0, 1] * 10, dtype=np.int64)
    strong_logits = np.stack((1 - target, target), axis=1) * 10.0
    weak_logits = np.zeros((len(target), 2), dtype=np.float64)
    trained = {
        split: {"target": target, "prediction": strong_logits}
        for split in ("validation", "test")
    }
    untrained = {
        split: {"target": target, "prediction": weak_logits}
        for split in ("validation", "test")
    }
    assert _classification_recovery_report("field", trained, untrained, train_labels)["gate"] == "pass"
    assert _classification_recovery_report("field", untrained, untrained, train_labels)["gate"] == "fail"

    regression_target = np.linspace(0.0, 10.0, 20)
    trained_regression = {
        split: {"target": regression_target, "prediction": regression_target + 0.1}
        for split in ("validation", "test")
    }
    untrained_regression = {
        split: {"target": regression_target, "prediction": np.zeros_like(regression_target)}
        for split in ("validation", "test")
    }
    assert _regression_recovery_report(
        "value", trained_regression, untrained_regression, list(np.linspace(0.0, 10.0, 60))
    )["gate"] == "pass"


@pytest.mark.parametrize("spelling", ["--artifacts-dir", "--artifact-dir"])
def test_cli_accepts_documented_artifact_directory_aliases(spelling: str) -> None:
    args = build_argument_parser().parse_args(
        [
            "--cache-dir",
            "cache",
            spelling,
            "artifacts",
            "--split-manifest",
            "split.json",
            "--output-dir",
            "evaluation",
        ]
    )
    assert args.artifact_dir == Path("artifacts")
