from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from data.dynamic_card_dataset import DynamicCardTrainingBatch
from data.replay_dataset import ReplayDatasetSummary
from data.state_schema import AREA_IDS, CardInstanceState, collate_card_dynamic
from models.dynamic_card_auxiliary import DynamicCardAuxiliaryOutput
from models.static_card_adapter import StaticCardEmbeddingAdapter
from training.train_dynamic_card_fusion import (
    DynamicCardTrainingModel,
    DynamicModelOutput,
    ReplayCollection,
    checkpoint_payload,
    compute_losses,
    energy_counterfactual_diagnostic,
    instance_state_counterfactual_diagnostic,
    load_checkpoint,
    set_seed,
    split_metadata,
    validate_static_adapter_alignment,
)


def _config() -> dict:
    return json.loads(open("configs/dynamic_card_fusion.json", encoding="utf-8").read())


def _batch() -> DynamicCardTrainingBatch:
    instance = CardInstanceState(
        21,
        1,
        0,
        0,
        AREA_IDS["ACTIVE"],
        "active",
        0,
        is_pokemon=True,
        hp=100,
        max_hp=120,
        copy_count=1,
    )
    dynamic = collate_card_dynamic([instance])
    return DynamicCardTrainingBatch(
        card_dynamic_batch=dynamic,
        sample_indices=torch.tensor([0]),
        static_known_mask=torch.ones(1),
        detail_exists_mask=torch.ones(1),
        energy_resolved_mask=torch.ones(1),
        detail_valid_mask=torch.ones(1, 2),
        attack_ids=torch.tensor([[1, 2]]),
        attack_costs=torch.zeros(1, 2, 12),
        attack_detail_mask=torch.ones(1, 2),
        payable_targets=torch.tensor([[1.0, 0.0]]),
        energy_remaining_targets=torch.zeros(1, 2, 12),
        payment_supervision_mask=torch.tensor([[1.0, 0.0]]),
        hp_targets=torch.tensor([[5.0 / 6.0, 1.0 / 6.0]]),
        hp_mask=torch.ones(1),
        zone_targets=dynamic.zone_ids,
        role_targets=dynamic.field_role_ids,
    )


def _output(unsupervised_value: float) -> DynamicModelOutput:
    auxiliary = DynamicCardAuxiliaryOutput(
        payable_logits=torch.tensor([[0.5, unsupervised_value]]),
        energy_remaining=torch.cat(
            [torch.zeros(1, 1, 12), torch.full((1, 1, 12), unsupervised_value)], dim=1
        ),
        hp_state=torch.tensor([[0.8, 0.2]]),
        zone_logits=torch.zeros(1, 13),
        role_logits=torch.zeros(1, 5),
    )
    return DynamicModelOutput(
        instance_tokens=torch.zeros(1, 128),
        attention_weights=torch.zeros(1, 4, 2),
        static_detail_tokens=torch.zeros(1, 2, 128),
        static_detail_mask=torch.ones(1, 2),
        auxiliary=auxiliary,
    )


@pytest.mark.parametrize("unsupervised_value", [1000.0, float("nan"), float("inf")])
def test_unresolved_attack_does_not_change_payment_losses(unsupervised_value: float) -> None:
    batch = _batch()
    weights = _config()["training"]["loss_weights"]
    baseline = compute_losses(_output(0.0), batch, weights)
    changed = compute_losses(_output(unsupervised_value), batch, weights)
    assert torch.allclose(baseline.payable, changed.payable)
    assert torch.allclose(baseline.energy_remaining, changed.energy_remaining)
    assert torch.isfinite(changed.total)


def test_split_metadata_rejects_episode_leakage() -> None:
    def collection(episode: int, date: str) -> ReplayCollection:
        sample = SimpleNamespace(episode_id=episode, replay_id=None, source_date=date)
        return ReplayCollection([sample], ReplayDatasetSummary(replay_count=1, sample_count=1))

    with pytest.raises(ValueError, match="leakage"):
        split_metadata({
            "train": collection(1, "2026-07-08"),
            "validation": collection(1, "2026-07-10"),
            "test": collection(2, "2026-07-11"),
        })
    result = split_metadata({
        "train": collection(1, "2026-07-08"),
        "validation": collection(2, "2026-07-10"),
        "test": collection(3, "2026-07-11"),
    })
    assert not any(result["episode_overlap"].values())


def test_dynamic_checkpoint_contains_required_state_and_reloads(tmp_path) -> None:
    config = _config()
    set_seed(23)
    weight = torch.randn(2, 128)
    details = torch.randn(2, 7, 128)
    masks = torch.ones(2, 7)
    types = torch.ones(2, 7, dtype=torch.long)
    adapter = StaticCardEmbeddingAdapter(
        weight.clone(), {"21": 0, "22": 1}, freeze=True,
        detail_tokens=details.clone(), detail_mask=masks.clone(), detail_type_ids=types.clone(),
    )
    model = DynamicCardTrainingModel(adapter, config)
    optimizer = torch.optim.AdamW([parameter for parameter in model.parameters() if parameter.requires_grad])
    best_state = {
        "dynamic_instance_encoder": model.dynamic_instance_encoder.state_dict(),
        "card_instance_fusion": model.card_instance_fusion.state_dict(),
        "auxiliary_heads": model.auxiliary_heads.state_dict(),
    }
    payload = checkpoint_payload(
        model,
        optimizer,
        config,
        {"version": "static"},
        {"splits": {}},
        2,
        7,
        {},
        1.0,
        {},
        best_state,
    )
    required = {
        "dynamic_instance_encoder",
        "card_instance_fusion",
        "auxiliary_heads",
        "optimizer_state",
        "training_config",
        "static_artifact_version",
        "replay_split_metadata",
    }
    assert required <= payload.keys()
    path = tmp_path / "checkpoint.pt"
    torch.save(payload, path)

    adapter_two = StaticCardEmbeddingAdapter(
        weight.clone(), {"21": 0, "22": 1}, freeze=True,
        detail_tokens=details.clone(), detail_mask=masks.clone(), detail_type_ids=types.clone(),
    )
    reloaded = DynamicCardTrainingModel(adapter_two, config)
    load_checkpoint(path, reloaded)
    for left, right in zip(model.dynamic_instance_encoder.parameters(), reloaded.dynamic_instance_encoder.parameters()):
        assert torch.equal(left, right)


def test_fixed_seed_reproduces_dynamic_initialization() -> None:
    config = _config()
    static_weight = torch.zeros(1, 128)

    def build() -> DynamicCardTrainingModel:
        adapter = StaticCardEmbeddingAdapter(
            static_weight.clone(), {"21": 0}, freeze=True,
            detail_tokens=torch.zeros(1, 7, 128),
            detail_mask=torch.ones(1, 7),
            detail_type_ids=torch.ones(1, 7, dtype=torch.long),
        )
        return DynamicCardTrainingModel(adapter, config)

    set_seed(101)
    first = build()
    set_seed(101)
    second = build()
    assert all(torch.equal(left, right) for left, right in zip(first.parameters(), second.parameters()))


def test_static_artifact_alignment_excludes_invalid_catalog_slots_and_is_numeric() -> None:
    config = _config()
    config["model"]["max_details"] = 3
    adapter = StaticCardEmbeddingAdapter(
        torch.zeros(1, 128),
        {"21": 0},
        freeze=True,
        detail_tokens=torch.zeros(1, 3, 128),
        detail_mask=torch.ones(1, 3),
        detail_type_ids=torch.tensor([[1, 2, 3]]),
    )
    catalog = SimpleNamespace(
        static_catalog=SimpleNamespace(
            card_id_to_index={21: 0},
            details_by_card_id={21: [
                {"detail_index": 0, "detail_type": "attack", "attack_id": None},
                {"detail_index": 1, "detail_type": "ability"},
                {"detail_index": 2, "detail_type": "special_effect"},
            ]},
        ),
        invalid_detail_slots=lambda card_id: {0},
    )
    result = validate_static_adapter_alignment(adapter, catalog, config)
    assert result["shape_valid"]
    assert result["catalog_invalid_slots_excluded"] == 1
    assert result["card_coverage"] == 1.0
    assert result["detail_slot_coverage"] == 1.0
    assert result["coverage"] == 1.0
    assert result["mismatch_examples"] == []


def test_static_artifact_alignment_reports_type_order_and_nonfinite_tokens() -> None:
    config = _config()
    config["model"]["max_details"] = 2
    tokens = torch.zeros(1, 2, 128)
    tokens[0, 1, 0] = float("nan")
    adapter = StaticCardEmbeddingAdapter(
        torch.zeros(1, 128),
        {"21": 0},
        freeze=True,
        detail_tokens=tokens,
        detail_mask=torch.ones(1, 2),
        detail_type_ids=torch.tensor([[2, 1]]),
    )
    catalog = SimpleNamespace(
        static_catalog=SimpleNamespace(
            card_id_to_index={21: 0},
            details_by_card_id={21: [
                {"detail_index": 0, "detail_type": "attack", "attack_id": 1},
                {"detail_index": 1, "detail_type": "ability"},
            ]},
        ),
        invalid_detail_slots=lambda card_id: set(),
    )
    result = validate_static_adapter_alignment(adapter, catalog, config)
    assert result["card_coverage"] == 0.0
    assert result["detail_slot_coverage"] < 1.0
    assert result["nonfinite_valid_token_count"] == 1
    assert result["mismatch_examples"][0]["kind"] == "card_alignment_mismatch"


def test_energy_counterfactual_prefers_and_summarizes_supervised_candidates() -> None:
    config = _config()
    batch = _batch()
    batch.attack_costs[0, 0, 0] = 1.0
    batch.attack_costs[0, 1, 0] = 1.0
    adapter = StaticCardEmbeddingAdapter(
        torch.zeros(1, 128),
        {"21": 0},
        freeze=True,
        detail_tokens=torch.zeros(1, 2, 128),
        detail_mask=torch.ones(1, 2),
        detail_type_ids=torch.ones(1, 2, dtype=torch.long),
    )
    model = DynamicCardTrainingModel(adapter, config).eval()
    result = energy_counterfactual_diagnostic(
        model, batch, torch.device("cpu"), strict=False, max_candidates=2
    )
    assert result["evaluated_candidate_count"] == 2
    assert result["supervised_candidate_count"] == 1
    assert result["best_supervised_candidate"]["detail_index"] == 0
    assert result["payable_positive_count"] == 1
    assert isinstance(result["max_probability_delta"], float)
    assert isinstance(result["mean_probability_delta"], float)


def test_trained_model_instance_state_counterfactuals_change_same_card_token() -> None:
    config = _config()
    batch = _batch()
    batch.attack_costs[0, 0, 0] = 1.0
    adapter = StaticCardEmbeddingAdapter(
        torch.zeros(1, 128),
        {"21": 0},
        freeze=True,
        detail_tokens=torch.zeros(1, 2, 128),
        detail_mask=torch.ones(1, 2),
        detail_type_ids=torch.ones(1, 2, dtype=torch.long),
    )
    model = DynamicCardTrainingModel(adapter, config).eval()
    result = instance_state_counterfactual_diagnostic(
        model, batch, torch.device("cpu"), strict=True, minimum_token_l2=0.0
    )
    assert result["success"]
    assert result["same_card_id_preserved"]
    assert result["hp_token_l2"] > 0
    assert result["energy_token_l2"] > 0
    assert result["zone_token_l2"] > 0
