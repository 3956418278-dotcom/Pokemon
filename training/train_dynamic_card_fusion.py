from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import resource
import time
import traceback
from collections import Counter
from dataclasses import dataclass, field, fields as dataclass_fields, is_dataclass
from functools import partial
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data.dynamic_card_dataset import (
    AttackCostCatalog,
    DynamicCardTrainingBatch,
    collate_dynamic_card_samples,
)
from data.replay_dataset import ReplayDatasetSummary, ReplayDecisionDataset, ReplayDecisionSample
from data.state_schema import FIELD_ROLE_IDS, ZONE_IDS
from models.card_instance_fusion import CardInstanceFusion, CardInstanceFusionOutput
from models.dynamic_card_auxiliary import DynamicCardAuxiliaryHeads, DynamicCardAuxiliaryOutput
from models.dynamic_instance_encoder import DynamicInstanceEncoder
from models.static_card_adapter import StaticCardEmbeddingAdapter
from scripts.audit_replay_features import build_report, load_known_static_ids


@dataclass
class ReplayCollection:
    samples: list[ReplayDecisionSample]
    summary: ReplayDatasetSummary
    replay_paths: list[Path] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> ReplayDecisionSample:
        return self.samples[index]


@dataclass
class DynamicModelOutput:
    instance_tokens: torch.Tensor
    attention_weights: torch.Tensor
    static_detail_tokens: torch.Tensor
    static_detail_mask: torch.Tensor
    auxiliary: DynamicCardAuxiliaryOutput


@dataclass
class LossBundle:
    total: torch.Tensor
    payable: torch.Tensor
    energy_remaining: torch.Tensor
    hp_state: torch.Tensor
    zone_role: torch.Tensor

    def task(self, name: str) -> torch.Tensor:
        return getattr(self, name)


class DynamicCardTrainingModel(nn.Module):
    def __init__(self, static_adapter: StaticCardEmbeddingAdapter, config: dict[str, Any]) -> None:
        super().__init__()
        model_config = config["model"]
        self.static_adapter = static_adapter
        for parameter in self.static_adapter.parameters():
            parameter.requires_grad_(False)
        self.dynamic_instance_encoder = DynamicInstanceEncoder(
            output_dim=int(model_config["dynamic_dim"]),
            dropout=float(model_config["dropout"]),
        )
        self.card_instance_fusion = CardInstanceFusion(
            static_dim=int(model_config["static_dim"]),
            dynamic_dim=int(model_config["dynamic_dim"]),
            output_dim=int(model_config["output_dim"]),
            num_heads=int(model_config["attention_heads"]),
            dropout=float(model_config["dropout"]),
        )
        self.auxiliary_heads = DynamicCardAuxiliaryHeads(token_dim=int(model_config["output_dim"]))

    def train(self, mode: bool = True) -> "DynamicCardTrainingModel":
        super().train(mode)
        self.static_adapter.eval()
        return self

    def forward(self, batch: DynamicCardTrainingBatch, return_attention: bool = False) -> DynamicModelOutput:
        dynamic_batch = batch.dynamic_batch
        static = self.static_adapter.forward_features(dynamic_batch.card_ids)
        if static.detail_tokens is None or static.detail_mask is None:
            raise RuntimeError("dynamic card training requires exported detail token artifacts")
        if static.detail_tokens.shape[1] != batch.attack_detail_mask.shape[1]:
            raise ValueError(
                "static detail width and training label width differ: "
                f"{static.detail_tokens.shape[1]} != {batch.attack_detail_mask.shape[1]}"
            )
        if batch.detail_valid_mask.shape != static.detail_mask.shape:
            raise ValueError(
                "catalog detail-valid mask and static detail mask differ: "
                f"{tuple(batch.detail_valid_mask.shape)} != {tuple(static.detail_mask.shape)}"
            )
        effective_detail_mask = static.detail_mask.float() * (batch.detail_valid_mask > 0).float()
        dynamic_batch.static_known_mask = static.known_mask.float()
        effective_detail_exists = (effective_detail_mask > 0).any(dim=1).float()
        dynamic_batch.detail_exists_mask = effective_detail_exists
        batch.detail_exists_mask = effective_detail_exists
        dynamic_repr = self.dynamic_instance_encoder(dynamic_batch)
        fusion = self.card_instance_fusion(
            static.summary,
            dynamic_repr,
            static.detail_tokens,
            effective_detail_mask,
            static.detail_type_ids,
            return_attention=True,
        )
        if not isinstance(fusion, CardInstanceFusionOutput):
            raise TypeError("CardInstanceFusion did not return diagnostic output")
        auxiliary = self.auxiliary_heads(
            fusion.card_instance_token,
            static.detail_tokens,
            effective_detail_mask,
            static.detail_type_ids,
        )
        attention = fusion.attention_weights if return_attention else fusion.attention_weights.detach()
        return DynamicModelOutput(
            instance_tokens=fusion.card_instance_token,
            attention_weights=attention,
            static_detail_tokens=static.detail_tokens,
            static_detail_mask=effective_detail_mask,
            auxiliary=auxiliary,
        )


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(values.dtype)
    while mask.dim() < values.dim():
        mask = mask.unsqueeze(-1)
    expanded = mask.expand_as(values)
    denominator = expanded.sum()
    safe_values = torch.where(expanded > 0, values, torch.zeros_like(values))
    if denominator.item() == 0:
        return safe_values.sum()
    return safe_values.sum() / denominator


def compute_losses(
    output: DynamicModelOutput,
    batch: DynamicCardTrainingBatch,
    loss_weights: dict[str, float],
) -> LossBundle:
    payment_valid = batch.payment_supervision_mask > 0
    payable_logits = torch.where(
        payment_valid,
        output.auxiliary.payable_logits,
        torch.zeros_like(output.auxiliary.payable_logits),
    )
    payable_targets = torch.where(
        payment_valid,
        batch.payable_targets,
        torch.zeros_like(batch.payable_targets),
    )
    payable_raw = F.binary_cross_entropy_with_logits(
        payable_logits,
        payable_targets,
        reduction="none",
    )
    payable = _masked_mean(payable_raw, batch.payment_supervision_mask)
    energy_valid = payment_valid.unsqueeze(-1)
    energy_prediction = torch.where(
        energy_valid,
        output.auxiliary.energy_remaining,
        torch.zeros_like(output.auxiliary.energy_remaining),
    )
    energy_targets = torch.where(
        energy_valid,
        batch.energy_remaining_targets,
        torch.zeros_like(batch.energy_remaining_targets),
    )
    energy_raw = F.smooth_l1_loss(
        energy_prediction,
        energy_targets,
        reduction="none",
    )
    energy_remaining = _masked_mean(energy_raw, batch.payment_supervision_mask)
    hp_raw = F.smooth_l1_loss(output.auxiliary.hp_state, batch.hp_targets, reduction="none")
    hp_state = _masked_mean(hp_raw, batch.hp_mask)
    if batch.instance_count:
        zone = F.cross_entropy(output.auxiliary.zone_logits, batch.zone_targets)
        role = F.cross_entropy(output.auxiliary.role_logits, batch.role_targets)
        zone_role = 0.5 * (zone + role)
    else:
        zone_role = output.instance_tokens.sum() * 0.0
    total = (
        float(loss_weights["payable"]) * payable
        + float(loss_weights["energy_remaining"]) * energy_remaining
        + float(loss_weights["hp_state"]) * hp_state
        + float(loss_weights["zone_role"]) * zone_role
    )
    return LossBundle(total, payable, energy_remaining, hp_state, zone_role)


def cuda_runtime_info() -> dict[str, Any]:
    available = bool(torch.cuda.is_available())
    info: dict[str, Any] = {
        "available": available,
        "usable": False,
        "device_name": None,
        "device_capability": None,
        "compiled_architectures": list(torch.cuda.get_arch_list()) if available else [],
        "fallback_reason": None,
    }
    if not available:
        info["fallback_reason"] = "torch.cuda.is_available() is false"
        return info
    try:
        major, minor = torch.cuda.get_device_capability(0)
        architecture = f"sm_{major}{minor}"
        compiled = set(info["compiled_architectures"])
        info["device_name"] = torch.cuda.get_device_name(0)
        info["device_capability"] = architecture
        info["usable"] = architecture in compiled
        if not info["usable"]:
            info["fallback_reason"] = (
                f"device architecture {architecture} is absent from this PyTorch build: "
                f"{sorted(compiled)}"
            )
    except (AssertionError, RuntimeError) as exc:
        info["fallback_reason"] = f"CUDA runtime probe failed: {type(exc).__name__}: {exc}"
    return info


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if cuda_runtime_info()["usable"]:
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def _merge_summary(target: ReplayDatasetSummary, source: ReplayDatasetSummary) -> None:
    target.replay_count += source.replay_count
    target.sample_count += source.sample_count
    target.skipped_no_select += source.skipped_no_select
    target.parser_errors.extend(source.parser_errors)
    target.max_instances = max(target.max_instances, source.max_instances)
    target.max_options = max(target.max_options, source.max_options)
    target.max_events = max(target.max_events, source.max_events)
    target.max_token_estimate = max(target.max_token_estimate, source.max_token_estimate)


def load_replay_collection(paths: Sequence[Path], max_decisions: int) -> ReplayCollection:
    if not paths:
        raise ValueError("at least one replay path is required")
    base = max_decisions // len(paths)
    remainder = max_decisions % len(paths)
    samples: list[ReplayDecisionSample] = []
    summary = ReplayDatasetSummary()
    replay_paths: list[Path] = []
    for index, path in enumerate(paths):
        limit = base + int(index < remainder)
        if limit <= 0:
            continue
        dataset = ReplayDecisionDataset.from_paths([path], max_samples=limit)
        if not dataset.samples:
            raise RuntimeError(f"no decision samples were parsed from {path}")
        samples.extend(dataset.samples)
        replay_paths.extend(dataset.replay_paths)
        _merge_summary(summary, dataset.summary)
    summary.sample_count = len(samples)
    return ReplayCollection(samples=samples[:max_decisions], summary=summary, replay_paths=replay_paths)


def combine_collections(collections: Iterable[ReplayCollection]) -> ReplayCollection:
    samples: list[ReplayDecisionSample] = []
    paths: list[Path] = []
    summary = ReplayDatasetSummary()
    for collection in collections:
        samples.extend(collection.samples)
        paths.extend(collection.replay_paths)
        _merge_summary(summary, collection.summary)
    summary.sample_count = len(samples)
    return ReplayCollection(samples, summary, paths)


def split_metadata(collections: dict[str, ReplayCollection]) -> dict[str, Any]:
    result: dict[str, Any] = {"schema_version": "replay_episode_date_split_v1", "splits": {}}
    episode_sets: dict[str, set[str]] = {}
    for split, collection in collections.items():
        missing_provenance = [
            index
            for index, sample in enumerate(collection.samples)
            if sample.episode_id is None and sample.replay_id is None and not getattr(sample, "source_path", None)
        ]
        unknown_dates = [index for index, sample in enumerate(collection.samples) if not sample.source_date]
        if missing_provenance:
            raise ValueError(f"{split} has samples without episode/replay/source path: {missing_provenance[:20]}")
        if unknown_dates:
            raise ValueError(f"{split} has samples without a source date: {unknown_dates[:20]}")
        episodes = {
            str(
                sample.episode_id
                if sample.episode_id is not None
                else sample.replay_id
                if sample.replay_id is not None
                else getattr(sample, "source_path", None)
            )
            for sample in collection.samples
        }
        episode_sets[split] = episodes
        result["splits"][split] = {
            "decision_samples": len(collection.samples),
            "processed_replays": collection.summary.replay_count,
            "episode_count": len(episodes),
            "dates": dict(Counter(sample.source_date or "unknown" for sample in collection.samples)),
            "parser_error_count": len(collection.summary.parser_errors),
        }
    overlaps: dict[str, list[str]] = {}
    names = list(collections)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            shared = sorted(episode_sets[left] & episode_sets[right])
            overlaps[f"{left}__{right}"] = shared
    result["episode_overlap"] = overlaps
    if any(overlaps.values()):
        raise ValueError(f"episode leakage detected across replay splits: {overlaps}")
    dates = {
        split: sorted({sample.source_date for sample in collection.samples if sample.source_date})
        for split, collection in collections.items()
    }
    if max(dates["train"]) >= min(dates["validation"]) or max(dates["validation"]) >= min(dates["test"]):
        raise ValueError(f"replay dates are not a strict temporal split: {dates}")
    result["temporal_order_verified"] = True
    return result


def make_loader(
    collection: ReplayCollection,
    catalog: AttackCostCatalog,
    config: dict[str, Any],
    *,
    shuffle: bool,
    seed_offset: int = 0,
) -> DataLoader:
    generator = torch.Generator().manual_seed(int(config["seed"]) + int(seed_offset))
    return DataLoader(
        collection.samples,
        batch_size=int(config["training"]["decision_batch_size"]),
        shuffle=shuffle,
        num_workers=int(config["training"]["num_workers"]),
        collate_fn=partial(
            collate_dynamic_card_samples,
            catalog=catalog,
            max_details=int(config["model"]["max_details"]),
        ),
        generator=generator,
    )


def build_model(static_artifact_dir: Path, config: dict[str, Any], device: torch.device) -> DynamicCardTrainingModel:
    adapter = StaticCardEmbeddingAdapter.from_artifacts(static_artifact_dir, freeze=True)
    model = DynamicCardTrainingModel(adapter, config).to(device)
    return model


DETAIL_TYPE_TO_ID = {"attack": 1, "ability": 2, "special_effect": 3}


def validate_static_adapter_alignment(
    adapter: StaticCardEmbeddingAdapter,
    catalog: AttackCostCatalog,
    config: dict[str, Any],
    *,
    mismatch_limit: int = 100,
) -> dict[str, Any]:
    """Validate the exported static tensors against catalog metadata and exclusions."""

    expected_card_count = len(catalog.static_catalog.card_id_to_index)
    expected_width = int(config["model"]["max_details"])
    expected_dim = int(config["model"]["static_dim"])
    summary = adapter.embedding.weight.detach()[1:]
    detail_tokens = adapter.detail_tokens.detach()[1:] if adapter.detail_tokens is not None else None
    detail_mask = adapter.detail_mask.detach()[1:] if adapter.detail_mask is not None else None
    detail_type_ids = adapter.detail_type_ids.detach()[1:] if adapter.detail_type_ids is not None else None
    shapes = {
        "card_summary": list(summary.shape),
        "detail_tokens": list(detail_tokens.shape) if detail_tokens is not None else None,
        "detail_mask": list(detail_mask.shape) if detail_mask is not None else None,
        "detail_type_ids": list(detail_type_ids.shape) if detail_type_ids is not None else None,
    }
    expected_shapes = {
        "card_summary": [expected_card_count, expected_dim],
        "detail_tokens": [expected_card_count, expected_width, expected_dim],
        "detail_mask": [expected_card_count, expected_width],
        "detail_type_ids": [expected_card_count, expected_width],
    }
    shape_checks = {
        name: shapes[name] == expected
        for name, expected in expected_shapes.items()
    }
    mismatch_examples: list[dict[str, Any]] = []

    def mismatch(payload: dict[str, Any]) -> None:
        if len(mismatch_examples) < mismatch_limit:
            mismatch_examples.append(payload)

    for name, valid in shape_checks.items():
        if not valid:
            mismatch({"kind": "shape_mismatch", "tensor": name, "actual": shapes[name], "expected": expected_shapes[name]})
    if not all(shape_checks.values()) or detail_tokens is None or detail_mask is None or detail_type_ids is None:
        return {
            "schema_version": "static_artifact_detail_alignment_v1",
            "shapes": shapes,
            "expected_shapes": expected_shapes,
            "shape_checks": shape_checks,
            "shape_valid": False,
            "checked_card_count": expected_card_count,
            "aligned_card_count": 0,
            "expected_valid_detail_slots": 0,
            "actual_valid_detail_slots": 0,
            "aligned_detail_slots": 0,
            "catalog_invalid_slots_excluded": 0,
            "nonfinite_valid_token_count": 0,
            "card_coverage": 0.0,
            "detail_slot_coverage": 0.0,
            "coverage": 0.0,
            "mismatch_count": len(mismatch_examples),
            "mismatch_examples": mismatch_examples,
        }

    adapter_mapping = {int(card_id): int(index) for card_id, index in adapter.card_id_to_index.items()}
    aligned_cards = 0
    expected_slot_total = 0
    actual_slot_total = 0
    aligned_slot_total = 0
    excluded_slot_total = 0
    nonfinite_valid_tokens = 0
    for card_id, catalog_index in sorted(catalog.static_catalog.card_id_to_index.items(), key=lambda item: item[1]):
        card_mismatches: list[str] = []
        adapter_index = adapter_mapping.get(int(card_id))
        if adapter_index != int(catalog_index):
            card_mismatches.append("card_index")
        row_index = int(catalog_index)
        metadata = catalog.static_catalog.details_by_card_id.get(int(card_id), [])
        invalid_slots = set(catalog.invalid_detail_slots(card_id))
        excluded_slot_total += len(invalid_slots)
        expected_counts = Counter()
        attack_ordinal = 0
        for detail in metadata:
            detail_type = str(detail.get("detail_type", "")).strip().lower()
            if detail_type == "attack":
                physical_slot = attack_ordinal
                attack_ordinal += 1
                if physical_slot in invalid_slots:
                    continue
            if detail_type in DETAIL_TYPE_TO_ID:
                expected_counts[detail_type] += 1
        expected_sequence = [
            type_id
            for detail_type, type_id in DETAIL_TYPE_TO_ID.items()
            for _ in range(int(expected_counts[detail_type]))
        ]
        raw_positions = torch.nonzero(detail_mask[row_index] > 0, as_tuple=False).flatten().tolist()
        effective_positions = [position for position in raw_positions if position not in invalid_slots]
        actual_sequence = [int(detail_type_ids[row_index, position].item()) for position in effective_positions]
        finite_by_position = [
            bool(torch.isfinite(detail_tokens[row_index, position]).all().item())
            for position in effective_positions
        ]
        nonfinite_valid_tokens += sum(not valid for valid in finite_by_position)
        expected_slot_total += len(expected_sequence)
        actual_slot_total += len(actual_sequence)
        aligned_slot_total += sum(
            actual_type == expected_type and finite
            for actual_type, expected_type, finite in zip(actual_sequence, expected_sequence, finite_by_position)
        )
        if actual_sequence != expected_sequence:
            card_mismatches.append("detail_type_count_or_group_order")
        if not all(finite_by_position):
            card_mismatches.append("nonfinite_valid_detail_token")
        if not bool(torch.isfinite(summary[row_index]).all().item()):
            card_mismatches.append("nonfinite_card_summary")
        if card_mismatches:
            mismatch({
                "kind": "card_alignment_mismatch",
                "card_id": int(card_id),
                "catalog_index": int(catalog_index),
                "adapter_index": adapter_index,
                "reasons": card_mismatches,
                "invalid_slots_excluded": sorted(invalid_slots),
                "expected_type_counts": dict(expected_counts),
                "expected_type_sequence": expected_sequence,
                "actual_type_sequence": actual_sequence,
                "effective_positions": effective_positions,
            })
        else:
            aligned_cards += 1

    card_coverage = aligned_cards / max(expected_card_count, 1)
    detail_slot_coverage = aligned_slot_total / max(expected_slot_total, actual_slot_total, 1)
    return {
        "schema_version": "static_artifact_detail_alignment_v1",
        "shapes": shapes,
        "expected_shapes": expected_shapes,
        "shape_checks": shape_checks,
        "shape_valid": True,
        "checked_card_count": expected_card_count,
        "aligned_card_count": aligned_cards,
        "expected_valid_detail_slots": expected_slot_total,
        "actual_valid_detail_slots": actual_slot_total,
        "aligned_detail_slots": aligned_slot_total,
        "catalog_invalid_slots_excluded": excluded_slot_total,
        "nonfinite_valid_token_count": nonfinite_valid_tokens,
        "card_coverage": card_coverage,
        "detail_slot_coverage": detail_slot_coverage,
        "coverage": min(card_coverage, detail_slot_coverage),
        "mismatch_count": expected_card_count - aligned_cards,
        "mismatch_examples": mismatch_examples,
    }


def validate_static_artifact_alignment(
    static_artifact_dir: Path,
    catalog: AttackCostCatalog,
    config: dict[str, Any],
) -> dict[str, Any]:
    adapter = StaticCardEmbeddingAdapter.from_artifacts(static_artifact_dir, freeze=True)
    return validate_static_adapter_alignment(adapter, catalog, config)


def _metric_sums(
    output: DynamicModelOutput,
    batch: DynamicCardTrainingBatch,
    losses: LossBundle,
) -> dict[str, float]:
    payment_mask = batch.payment_supervision_mask > 0
    payment_count = int(payment_mask.sum().item())
    hp_mask = batch.hp_mask > 0
    hp_count = int(hp_mask.sum().item())
    instances = batch.instance_count
    payable_predictions = output.auxiliary.payable_logits >= 0
    payable_targets = batch.payable_targets >= 0.5
    payable_correct = int(((payable_predictions == payable_targets) & payment_mask).sum().item())
    energy_abs = (
        (output.auxiliary.energy_remaining - batch.energy_remaining_targets).abs()
        * payment_mask.unsqueeze(-1)
    ).sum().item()
    hp_abs = (
        (output.auxiliary.hp_state - batch.hp_targets).abs() * hp_mask.unsqueeze(-1)
    ).sum().item()
    zone_correct = int((output.auxiliary.zone_logits.argmax(-1) == batch.zone_targets).sum().item())
    role_correct = int((output.auxiliary.role_logits.argmax(-1) == batch.role_targets).sum().item())
    return {
        "batches": 1.0,
        "total_loss_sum": float(losses.total.detach().item()),
        "payable_loss_sum": float(losses.payable.detach().item()) * max(payment_count, 1),
        "energy_loss_sum": float(losses.energy_remaining.detach().item()) * max(payment_count * 12, 1),
        "hp_loss_sum": float(losses.hp_state.detach().item()) * max(hp_count * 2, 1),
        "zone_role_loss_sum": float(losses.zone_role.detach().item()) * max(instances, 1),
        "payment_count": float(payment_count),
        "payable_correct": float(payable_correct),
        "energy_absolute_error": float(energy_abs),
        "energy_component_count": float(payment_count * 12),
        "hp_absolute_error": float(hp_abs),
        "hp_component_count": float(hp_count * 2),
        "instances": float(instances),
        "zone_correct": float(zone_correct),
        "role_correct": float(role_correct),
        "attack_details": float(batch.attack_detail_mask.sum().item()),
        "unresolved_attack_details": float(
            (batch.attack_detail_mask * (1.0 - batch.payment_supervision_mask)).sum().item()
        ),
        "visible_instances": float(batch.dynamic_batch.visibility_mask.sum().item()),
        "static_known_instances": float(batch.static_known_mask.sum().item()),
        "detail_bearing_instances": float(batch.detail_exists_mask.sum().item()),
        "energy_resolved_instances": float(batch.energy_resolved_mask.sum().item()),
        "hp_supervised_instances": float(batch.hp_mask.sum().item()),
    }


def _add_metrics(total: dict[str, float], update: dict[str, float]) -> None:
    for key, value in update.items():
        total[key] = total.get(key, 0.0) + value


def _finalize_metrics(raw: dict[str, float]) -> dict[str, float | int]:
    def safe(numerator: str, denominator: str) -> float:
        return raw.get(numerator, 0.0) / max(raw.get(denominator, 0.0), 1.0)

    return {
        "loss": safe("total_loss_sum", "batches"),
        "payable_loss": safe("payable_loss_sum", "payment_count"),
        "energy_remaining_loss": safe("energy_loss_sum", "energy_component_count"),
        "hp_state_loss": safe("hp_loss_sum", "hp_component_count"),
        "zone_role_loss": safe("zone_role_loss_sum", "instances"),
        "payable_accuracy": safe("payable_correct", "payment_count"),
        "energy_remaining_mae": safe("energy_absolute_error", "energy_component_count"),
        "hp_state_mae": safe("hp_absolute_error", "hp_component_count"),
        "zone_accuracy": safe("zone_correct", "instances"),
        "role_accuracy": safe("role_correct", "instances"),
        "instance_count": int(raw.get("instances", 0.0)),
        "payment_supervision_count": int(raw.get("payment_count", 0.0)),
        "attack_detail_count": int(raw.get("attack_details", 0.0)),
        "unresolved_attack_detail_count": int(raw.get("unresolved_attack_details", 0.0)),
        "unresolved_ratio": safe("unresolved_attack_details", "attack_details"),
        "visible_instance_ratio": safe("visible_instances", "instances"),
        "static_known_ratio": safe("static_known_instances", "instances"),
        "detail_exists_ratio": safe("detail_bearing_instances", "instances"),
        "energy_resolved_ratio": safe("energy_resolved_instances", "instances"),
        "hp_supervision_ratio": safe("hp_supervised_instances", "instances"),
    }


def run_epoch(
    model: DynamicCardTrainingModel,
    loader: DataLoader,
    device: torch.device,
    loss_weights: dict[str, float],
    optimizer: torch.optim.Optimizer | None = None,
    gradient_clip_norm: float = 1.0,
) -> dict[str, float | int]:
    training = optimizer is not None
    model.train(training)
    raw: dict[str, float] = {}
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for cpu_batch in loader:
            batch = cpu_batch.to(device)
            if not batch.instance_count:
                continue
            if training:
                optimizer.zero_grad(set_to_none=True)
            output = model(batch)
            losses = compute_losses(output, batch, loss_weights)
            if not torch.isfinite(losses.total):
                raise FloatingPointError("non-finite dynamic card training loss")
            if training:
                losses.total.backward()
                nn.utils.clip_grad_norm_(
                    [parameter for parameter in model.parameters() if parameter.requires_grad],
                    gradient_clip_norm,
                )
                optimizer.step()
            _add_metrics(raw, _metric_sums(output, batch, losses))
    if not raw:
        raise RuntimeError("the replay loader produced no non-empty dynamic card batches")
    return _finalize_metrics(raw)


def _component_gradient_norm(module: nn.Module) -> float:
    total = 0.0
    for parameter in module.parameters():
        if parameter.grad is not None:
            total += float(parameter.grad.detach().square().sum().item())
    return math.sqrt(total)


def gradient_audit(
    model: DynamicCardTrainingModel,
    batch: DynamicCardTrainingBatch,
    loss_weights: dict[str, float],
) -> dict[str, Any]:
    model.train()
    result: dict[str, Any] = {}
    for task in ("payable", "energy_remaining", "hp_state", "zone_role"):
        model.zero_grad(set_to_none=True)
        output = model(batch)
        losses = compute_losses(output, batch, loss_weights)
        task_loss = losses.task(task)
        task_loss.backward()
        row = {
            "loss": float(task_loss.detach().item()),
            "dynamic_instance_encoder": _component_gradient_norm(model.dynamic_instance_encoder),
            "card_instance_fusion": _component_gradient_norm(model.card_instance_fusion),
            "auxiliary_heads": _component_gradient_norm(model.auxiliary_heads),
        }
        row["has_effective_gradient"] = all(row[name] > 0 for name in (
            "dynamic_instance_encoder",
            "card_instance_fusion",
            "auxiliary_heads",
        ))
        result[task] = row
    if not all(row["has_effective_gradient"] for row in result.values()):
        raise RuntimeError(f"one or more auxiliary tasks had no effective gradient: {result}")
    return result


def select_tiny_samples(
    samples: Sequence[ReplayDecisionSample],
    catalog: AttackCostCatalog,
    config: dict[str, Any],
) -> list[ReplayDecisionSample]:
    target = int(config["tiny_overfit"]["decision_samples"])
    supervised: list[ReplayDecisionSample] = []
    fallback: list[ReplayDecisionSample] = []
    for sample in samples[: min(len(samples), 512)]:
        batch = collate_dynamic_card_samples([sample], catalog, int(config["model"]["max_details"]))
        if batch.payment_supervision_mask.sum().item() > 0 and batch.hp_mask.sum().item() > 0:
            supervised.append(sample)
        else:
            fallback.append(sample)
        if len(supervised) >= target:
            break
    if not supervised:
        raise RuntimeError("could not find a real replay decision with resolved attack supervision for tiny overfit")
    selected = supervised[:target]
    if len(selected) < target:
        selected.extend(fallback[: target - len(selected)])
    return selected


def tiny_overfit(
    static_artifact_dir: Path,
    samples: Sequence[ReplayDecisionSample],
    catalog: AttackCostCatalog,
    config: dict[str, Any],
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any]]:
    selected = select_tiny_samples(samples, catalog, config)
    batch = collate_dynamic_card_samples(selected, catalog, int(config["model"]["max_details"])).to(device)
    model = build_model(static_artifact_dir, config, device)
    gradient_result = gradient_audit(model, batch, config["training"]["loss_weights"])
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(config["tiny_overfit"]["learning_rate"]),
    )

    model.eval()
    with torch.no_grad():
        initial_output = model(batch)
        initial_losses = compute_losses(initial_output, batch, config["training"]["loss_weights"])
        initial = {name: float(getattr(initial_losses, name).item()) for name in (
            "total", "payable", "energy_remaining", "hp_state", "zone_role"
        )}
    history: list[dict[str, float | int]] = []
    model.train()
    steps = int(config["tiny_overfit"]["steps"])
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        output = model(batch)
        losses = compute_losses(output, batch, config["training"]["loss_weights"])
        losses.total.backward()
        nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 5.0)
        optimizer.step()
        if step == 0 or (step + 1) % 25 == 0 or step + 1 == steps:
            history.append({"step": step + 1, "loss": float(losses.total.detach().item())})
    model.eval()
    with torch.no_grad():
        final_output = model(batch)
        final_losses = compute_losses(final_output, batch, config["training"]["loss_weights"])
        final = {name: float(getattr(final_losses, name).item()) for name in (
            "total", "payable", "energy_remaining", "hp_state", "zone_role"
        )}
    ratio = final["total"] / max(initial["total"], 1e-12)
    required = float(config["tiny_overfit"]["required_loss_ratio"])
    task_ratios = {
        name: final[name] / max(initial[name], 1e-12)
        for name in ("payable", "energy_remaining", "hp_state", "zone_role")
    }
    required_task_ratios = config["tiny_overfit"].get("required_task_loss_ratios", {})
    task_success = {
        name: task_ratios[name] <= float(required_task_ratios.get(name, 1.0))
        for name in task_ratios
    }
    result = {
        "decision_samples": len(selected),
        "instance_count": batch.instance_count,
        "payment_supervision_count": int(batch.payment_supervision_mask.sum().item()),
        "initial_losses": initial,
        "final_losses": final,
        "loss_ratio": ratio,
        "required_loss_ratio": required,
        "task_loss_ratios": task_ratios,
        "required_task_loss_ratios": required_task_ratios,
        "task_success": task_success,
        "success": ratio <= required and all(task_success.values()),
        "history": history,
    }
    if not result["success"]:
        raise RuntimeError(f"tiny-batch overfit gate failed: {result}")
    return result, gradient_result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_versions(
    static_dir: Path,
    card_records: Path,
    detail_metadata: Path,
) -> dict[str, Any]:
    paths = []
    if static_dir.exists():
        paths.extend(p for p in static_dir.iterdir() if p.is_file())
    if card_records.exists():
        paths.append(card_records)
    if detail_metadata.exists():
        paths.append(detail_metadata)
    files = {
        path.name: {"path": str(path), "sha256": _sha256(path), "size_bytes": path.stat().st_size}
        for path in paths
    }
    source_manifest = Path(__file__).resolve().parents[1] / "source_manifest.json"
    return {
        "schema_version": "dynamic_artifact_versions_v1",
        "static_files": files,
        "dynamic_source_manifest": (
            json.loads(source_manifest.read_text(encoding="utf-8")) if source_manifest.exists() else None
        ),
    }


def checkpoint_payload(
    model: DynamicCardTrainingModel,
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
    artifact_metadata: dict[str, Any],
    replay_metadata: dict[str, Any],
    epoch: int,
    global_step: int,
    metrics: dict[str, Any],
    best_validation_loss: float,
    best_metrics: dict[str, Any],
    best_model_state: dict[str, dict[str, torch.Tensor]],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": "dynamic_card_fusion_checkpoint_v1",
        "dynamic_instance_encoder": model.dynamic_instance_encoder.state_dict(),
        "card_instance_fusion": model.card_instance_fusion.state_dict(),
        "auxiliary_heads": model.auxiliary_heads.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "training_config": config,
        "static_artifact_version": artifact_metadata,
        "replay_split_metadata": replay_metadata,
        "epoch": epoch,
        "global_step": global_step,
        "metrics": metrics,
        "best_validation_loss": best_validation_loss,
        "best_metrics": best_metrics,
        "best_model_state": best_model_state,
        "python_random_state": random.getstate(),
        "torch_rng_state": torch.get_rng_state(),
    }
    if cuda_runtime_info()["usable"]:
        payload["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    return payload


def _torch_load(path: Path, device: torch.device | str) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def load_checkpoint(
    path: Path,
    model: DynamicCardTrainingModel,
    optimizer: torch.optim.Optimizer | None = None,
    expected_config: dict[str, Any] | None = None,
    expected_artifacts: dict[str, Any] | None = None,
    expected_replay_split: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = _torch_load(path, next(model.parameters()).device)
    if payload.get("schema_version") != "dynamic_card_fusion_checkpoint_v1":
        raise ValueError(f"unsupported checkpoint schema: {payload.get('schema_version')}")
    if expected_config is not None:
        for key in ("schema_version", "model"):
            if payload["training_config"].get(key) != expected_config.get(key):
                raise ValueError(f"checkpoint {key} is incompatible with the requested run")
    if expected_artifacts is not None:
        recorded = {
            name: row.get("sha256")
            for name, row in payload["static_artifact_version"].get("static_files", {}).items()
        }
        current = {
            name: row.get("sha256")
            for name, row in expected_artifacts.get("static_files", {}).items()
        }
        if recorded != current:
            raise ValueError("checkpoint static artifact hashes do not match the current artifacts")
    if expected_replay_split is not None and payload.get("replay_split_metadata") != expected_replay_split:
        raise ValueError("checkpoint replay split metadata does not match the current split")
    model.dynamic_instance_encoder.load_state_dict(payload["dynamic_instance_encoder"])
    model.card_instance_fusion.load_state_dict(payload["card_instance_fusion"])
    model.auxiliary_heads.load_state_dict(payload["auxiliary_heads"])
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer_state"])
    return payload


def _model_state(model: DynamicCardTrainingModel) -> dict[str, dict[str, torch.Tensor]]:
    return {
        "dynamic_instance_encoder": copy.deepcopy(model.dynamic_instance_encoder.state_dict()),
        "card_instance_fusion": copy.deepcopy(model.card_instance_fusion.state_dict()),
        "auxiliary_heads": copy.deepcopy(model.auxiliary_heads.state_dict()),
    }


def _restore_model_state(model: DynamicCardTrainingModel, state: dict[str, dict[str, torch.Tensor]]) -> None:
    model.dynamic_instance_encoder.load_state_dict(state["dynamic_instance_encoder"])
    model.card_instance_fusion.load_state_dict(state["card_instance_fusion"])
    model.auxiliary_heads.load_state_dict(state["auxiliary_heads"])


def _restore_rng_state(payload: dict[str, Any]) -> None:
    if "python_random_state" in payload:
        random.setstate(payload["python_random_state"])
    if "torch_rng_state" in payload:
        torch.set_rng_state(payload["torch_rng_state"].cpu())
    if cuda_runtime_info()["usable"] and "cuda_rng_state_all" in payload:
        torch.cuda.set_rng_state_all([state.cpu() for state in payload["cuda_rng_state_all"]])


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def train_formal(
    static_artifact_dir: Path,
    collections: dict[str, ReplayCollection],
    catalog: AttackCostCatalog,
    config: dict[str, Any],
    artifact_metadata: dict[str, Any],
    replay_metadata: dict[str, Any],
    output_dir: Path,
    device: torch.device,
    resume: Path | None,
) -> tuple[Path, Path, dict[str, Any], DynamicCardTrainingModel]:
    model = build_model(static_artifact_dir, config, device)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    start_epoch = 0
    global_step = 0
    best_loss = float("inf")
    best_metrics: dict[str, Any] = {}
    best_model_state: dict[str, dict[str, torch.Tensor]] | None = None
    resumed: dict[str, Any] | None = None
    if resume is not None:
        resumed = load_checkpoint(
            resume,
            model,
            optimizer,
            expected_config=config,
            expected_artifacts=artifact_metadata,
            expected_replay_split=replay_metadata,
        )
        start_epoch = int(resumed["epoch"]) + 1
        global_step = int(resumed.get("global_step", 0))
        best_loss = float(resumed.get("best_validation_loss", float("inf")))
        best_metrics = copy.deepcopy(resumed.get("best_metrics", {}))
        best_model_state = copy.deepcopy(resumed.get("best_model_state"))
        _restore_rng_state(resumed)

    validation_loader = make_loader(collections["validation"], catalog, config, shuffle=False)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_path = checkpoint_dir / "dynamic_card_fusion_best.pt"
    last_path = checkpoint_dir / "dynamic_card_fusion_last.pt"
    metrics_path = output_dir / "logs" / "training_metrics.jsonl"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    if resume is None:
        metrics_path.write_text("", encoding="utf-8")
    else:
        append_jsonl(metrics_path, {"event": "resume", "checkpoint": str(resume), "start_epoch": start_epoch})
        if best_model_state is not None and resumed is not None:
            best_payload = copy.deepcopy(resumed)
            best_payload.update(best_model_state)
            torch.save(best_payload, best_path)
        torch.save(resumed, last_path)

    for epoch in range(start_epoch, int(config["training"]["epochs"])):
        train_loader = make_loader(collections["train"], catalog, config, shuffle=True, seed_offset=epoch)
        train_metrics = run_epoch(
            model,
            train_loader,
            device,
            config["training"]["loss_weights"],
            optimizer,
            float(config["training"]["gradient_clip_norm"]),
        )
        validation_metrics = run_epoch(
            model,
            validation_loader,
            device,
            config["training"]["loss_weights"],
        )
        global_step += len(train_loader)
        row = {
            "epoch": epoch,
            "global_step": global_step,
            "train": train_metrics,
            "validation": validation_metrics,
        }
        append_jsonl(metrics_path, row)
        is_best = float(validation_metrics["loss"]) < best_loss
        if is_best:
            best_loss = float(validation_metrics["loss"])
            best_metrics = copy.deepcopy(row)
            best_model_state = _model_state(model)
        if best_model_state is None:
            raise RuntimeError("best model state was not initialized")
        payload = checkpoint_payload(
            model,
            optimizer,
            config,
            artifact_metadata,
            replay_metadata,
            epoch,
            global_step,
            row,
            best_loss,
            best_metrics,
            best_model_state,
        )
        torch.save(payload, last_path)
        if is_best:
            torch.save(payload, best_path)
    if not best_path.exists() or not last_path.exists():
        raise RuntimeError("formal training did not produce both best and last checkpoints")
    load_checkpoint(best_path, model, expected_config=config, expected_artifacts=artifact_metadata)
    return best_path, last_path, best_metrics, model


def checkpoint_reload_consistency(
    static_artifact_dir: Path,
    checkpoint: Path,
    model: DynamicCardTrainingModel,
    batch: DynamicCardTrainingBatch,
    config: dict[str, Any],
    device: torch.device,
    artifact_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    model.eval()
    with torch.no_grad():
        expected = model(batch.to(device)).instance_tokens.detach().cpu()
    reloaded = build_model(static_artifact_dir, config, device)
    load_checkpoint(
        checkpoint,
        reloaded,
        expected_config=config,
        expected_artifacts=artifact_metadata,
    )
    reloaded.eval()
    with torch.no_grad():
        actual = reloaded(batch.to(device)).instance_tokens.detach().cpu()
    max_abs = float((expected - actual).abs().max().item()) if expected.numel() else 0.0
    result = {"max_absolute_difference": max_abs, "tolerance": 1e-6, "success": max_abs <= 1e-6}
    if not result["success"]:
        raise RuntimeError(f"checkpoint reload consistency failed: {result}")
    return result


def diagnostic_examples(
    model: DynamicCardTrainingModel,
    batch: DynamicCardTrainingBatch,
    device: torch.device,
    limit: int = 32,
) -> list[dict[str, Any]]:
    model.eval()
    gpu_batch = batch.to(device)
    with torch.no_grad():
        output = model(gpu_batch, return_attention=True)
    examples: list[dict[str, Any]] = []
    positions = torch.nonzero(gpu_batch.attack_detail_mask > 0, as_tuple=False)
    for row, detail in positions[:limit].tolist():
        examples.append(
            {
                "card_id": int(gpu_batch.dynamic_batch.card_ids[row].item()),
                "serial": int(gpu_batch.dynamic_batch.serials[row].item()),
                "detail_index": int(detail),
                "attack_id": int(gpu_batch.attack_ids[row, detail].item()),
                "supervised": bool(gpu_batch.payment_supervision_mask[row, detail].item()),
                "payable_target": float(gpu_batch.payable_targets[row, detail].item()),
                "payable_probability": float(torch.sigmoid(output.auxiliary.payable_logits[row, detail]).item()),
                "remaining_target": gpu_batch.energy_remaining_targets[row, detail].tolist(),
                "remaining_prediction": output.auxiliary.energy_remaining[row, detail].tolist(),
                "attention_by_head": output.attention_weights[row, :, :].tolist(),
            }
        )
    return examples


def energy_counterfactual_diagnostic(
    model: DynamicCardTrainingModel,
    batch: DynamicCardTrainingBatch,
    device: torch.device,
    strict: bool = True,
    minimum_probability_delta: float = 0.0,
    max_candidates: int = 32,
) -> dict[str, Any]:
    candidate_tensor = torch.nonzero(
        (batch.attack_detail_mask > 0) & (batch.attack_costs.sum(dim=-1) > 0),
        as_tuple=False,
    )
    if not candidate_tensor.numel():
        result = {
            "success": False,
            "skipped": True,
            "reason": "no non-zero attack cost was available",
            "candidate_count": 0,
            "evaluated_candidate_count": 0,
            "supervised_candidate_count": 0,
            "examples": [],
        }
        if strict:
            raise RuntimeError(result["reason"])
        return result
    candidates = [tuple(int(value) for value in position) for position in candidate_tensor.tolist()]
    candidates.sort(
        key=lambda position: (
            -int(batch.payment_supervision_mask[position[0], position[1]].item() > 0),
            position[0],
            position[1],
        )
    )
    candidates = candidates[: max(int(max_candidates), 1)]
    model.eval()
    examples: list[dict[str, Any]] = []
    for row, detail in candidates:
        low = copy.deepcopy(batch)
        high = copy.deepcopy(batch)
        low.dynamic_batch.energy_counts[row].zero_()
        low.dynamic_batch.energy_valid_mask[row] = 1.0
        low.dynamic_batch.energy_resolved_mask[row] = 1.0
        low.dynamic_batch.numerical_features[row, 6] = 0.0
        low.dynamic_batch.numerical_mask[row, 6] = 1.0

        cost = high.attack_costs[row, detail].long()
        high_energy = torch.zeros_like(high.dynamic_batch.energy_counts[row])
        high_energy[1:10] = cost[1:10].to(high_energy.dtype)
        high_energy[10:] = cost[10:].to(high_energy.dtype)
        high_energy[1] += cost[0].to(high_energy.dtype)
        high.dynamic_batch.energy_counts[row] = high_energy
        high.dynamic_batch.energy_valid_mask[row] = 1.0
        high.dynamic_batch.energy_resolved_mask[row] = 1.0
        high.dynamic_batch.numerical_features[row, 6] = high_energy.sum()
        high.dynamic_batch.numerical_mask[row, 6] = 1.0

        with torch.no_grad():
            low_output = model(low.to(device), return_attention=True)
            high_output = model(high.to(device), return_attention=True)
        low_probability = float(torch.sigmoid(low_output.auxiliary.payable_logits[row, detail]).item())
        high_probability = float(torch.sigmoid(high_output.auxiliary.payable_logits[row, detail]).item())
        supervised = bool(batch.payment_supervision_mask[row, detail].item() > 0)
        examples.append({
            "card_id": int(batch.dynamic_batch.card_ids[row].item()),
            "serial": int(batch.dynamic_batch.serials[row].item()),
            "detail_index": int(detail),
            "attack_id": int(batch.attack_ids[row, detail].item()),
            "attack_cost": batch.attack_costs[row, detail].tolist(),
            "supervised": supervised,
            "payable_target": float(batch.payable_targets[row, detail].item()) if supervised else None,
            "low_energy_payable_probability": low_probability,
            "sufficient_energy_payable_probability": high_probability,
            "probability_delta": high_probability - low_probability,
            "instance_token_l2_distance": float(torch.linalg.vector_norm(
                high_output.instance_tokens[row] - low_output.instance_tokens[row]
            ).item()),
        })

    supervised_examples = [example for example in examples if example["supervised"]]
    best_supervised = (
        max(supervised_examples, key=lambda example: example["probability_delta"])
        if supervised_examples
        else None
    )
    deltas = [float(example["probability_delta"]) for example in examples]
    positive_labels = sum(float(example["payable_target"] or 0.0) >= 0.5 for example in supervised_examples)
    negative_labels = len(supervised_examples) - positive_labels
    gate_success = bool(
        best_supervised is not None
        and float(best_supervised["probability_delta"]) >= minimum_probability_delta
        and float(best_supervised["instance_token_l2_distance"]) > 0.0
    )
    result = {
        "skipped": False,
        "candidate_count": int(candidate_tensor.shape[0]),
        "evaluated_candidate_count": len(examples),
        "supervised_candidate_count": len(supervised_examples),
        "payable_positive_count": positive_labels,
        "payable_negative_count": negative_labels,
        "payable_positive_coverage": positive_labels / max(len(supervised_examples), 1),
        "payable_negative_coverage": negative_labels / max(len(supervised_examples), 1),
        "positive_delta_count": sum(delta > 0 for delta in deltas),
        "negative_delta_count": sum(delta < 0 for delta in deltas),
        "positive_delta_coverage": sum(delta > 0 for delta in deltas) / max(len(deltas), 1),
        "max_probability_delta": max(deltas),
        "mean_probability_delta": sum(deltas) / max(len(deltas), 1),
        "best_supervised_candidate": best_supervised,
        "probability_delta": best_supervised["probability_delta"] if best_supervised else None,
        "instance_token_l2_distance": (
            best_supervised["instance_token_l2_distance"] if best_supervised else None
        ),
        "minimum_probability_delta": minimum_probability_delta,
        "success": gate_success,
        "examples": examples,
    }
    if strict and not result["success"]:
        raise RuntimeError(f"energy counterfactual diagnostic failed: {result}")
    return result


def instance_state_counterfactual_diagnostic(
    model: DynamicCardTrainingModel,
    batch: DynamicCardTrainingBatch,
    device: torch.device,
    *,
    strict: bool = True,
    minimum_token_l2: float = 0.0,
) -> dict[str, Any]:
    payment_rows = batch.payment_supervision_mask.sum(dim=1) > 0
    hp_rows = batch.hp_mask > 0
    common_rows = torch.nonzero(payment_rows & hp_rows, as_tuple=False).flatten()
    candidate_rows = common_rows if common_rows.numel() else torch.nonzero(hp_rows, as_tuple=False).flatten()
    if not candidate_rows.numel():
        result = {"success": False, "skipped": True, "reason": "no HP-valid card instance was available"}
        if strict:
            raise RuntimeError(result["reason"])
        return result
    row = int(candidate_rows[0].item())

    def token_distance(left: DynamicCardTrainingBatch, right: DynamicCardTrainingBatch) -> float:
        model.eval()
        with torch.no_grad():
            left_token = model(left.to(device)).instance_tokens[row]
            right_token = model(right.to(device)).instance_tokens[row]
        return float(torch.linalg.vector_norm(right_token - left_token).item())

    healthy = copy.deepcopy(batch)
    damaged = copy.deepcopy(batch)
    max_hp = max(float(batch.dynamic_batch.numerical_features[row, 1].item()), 1.0)
    healthy.dynamic_batch.numerical_features[row, 0:5] = torch.tensor(
        [max_hp, max_hp, 0.0, 1.0, 0.0], dtype=healthy.dynamic_batch.numerical_features.dtype
    )
    damaged_hp = max_hp * 0.25
    damaged.dynamic_batch.numerical_features[row, 0:5] = torch.tensor(
        [damaged_hp, max_hp, max_hp - damaged_hp, 0.25, 0.75],
        dtype=damaged.dynamic_batch.numerical_features.dtype,
    )
    healthy.dynamic_batch.numerical_mask[row, 0:5] = 1.0
    damaged.dynamic_batch.numerical_mask[row, 0:5] = 1.0

    no_energy = copy.deepcopy(batch)
    with_energy = copy.deepcopy(batch)
    no_energy.dynamic_batch.energy_counts[row].zero_()
    no_energy.dynamic_batch.numerical_features[row, 6] = 0.0
    attack_positions = torch.nonzero(
        (batch.payment_supervision_mask[row] > 0) & (batch.attack_costs[row].sum(dim=-1) > 0),
        as_tuple=False,
    ).flatten()
    high_energy = torch.zeros_like(with_energy.dynamic_batch.energy_counts[row])
    if attack_positions.numel():
        cost = batch.attack_costs[row, int(attack_positions[0].item())].long()
        high_energy[1:10] = cost[1:10].to(high_energy.dtype)
        high_energy[10:] = cost[10:].to(high_energy.dtype)
        high_energy[1] += cost[0].to(high_energy.dtype)
    else:
        high_energy[1] = 1.0
    with_energy.dynamic_batch.energy_counts[row] = high_energy
    with_energy.dynamic_batch.numerical_features[row, 6] = high_energy.sum()
    for variant in (no_energy, with_energy):
        variant.dynamic_batch.energy_valid_mask[row] = 1.0
        variant.dynamic_batch.energy_resolved_mask[row] = 1.0
        variant.dynamic_batch.numerical_mask[row, 6] = 1.0

    active = copy.deepcopy(batch)
    bench = copy.deepcopy(batch)
    active.dynamic_batch.zone_ids[row] = ZONE_IDS["ACTIVE"]
    active.dynamic_batch.field_role_ids[row] = FIELD_ROLE_IDS["ACTIVE"]
    bench.dynamic_batch.zone_ids[row] = ZONE_IDS["BENCH"]
    bench.dynamic_batch.field_role_ids[row] = FIELD_ROLE_IDS["BENCH"]

    distances = {
        "hp_token_l2": token_distance(healthy, damaged),
        "energy_token_l2": token_distance(no_energy, with_energy),
        "zone_token_l2": token_distance(active, bench),
    }
    success = all(value > minimum_token_l2 for value in distances.values())
    result = {
        "success": success,
        "skipped": False,
        "card_id": int(batch.dynamic_batch.card_ids[row].item()),
        "serial": int(batch.dynamic_batch.serials[row].item()),
        "row": row,
        "same_card_id_preserved": True,
        "minimum_token_l2": minimum_token_l2,
        **distances,
    }
    if strict and not success:
        raise RuntimeError(f"instance-state counterfactual diagnostic failed: {result}")
    return result


def cpu_benchmark(
    static_artifact_dir: Path,
    checkpoint: Path,
    batch: DynamicCardTrainingBatch,
    config: dict[str, Any],
) -> dict[str, Any]:
    device = torch.device("cpu")
    model = build_model(static_artifact_dir, config, device)
    load_checkpoint(checkpoint, model, expected_config=config)
    model.eval()
    batch = batch.to(device)
    with torch.no_grad():
        static_features = model.static_adapter.forward_features(batch.dynamic_batch.card_ids)
    valid_detail_tokens = (
        int((static_features.detail_mask > 0).sum().item())
        if static_features.detail_mask is not None
        else 0
    )
    warmup = int(config["benchmark"]["warmup_iterations"])
    iterations = int(config["benchmark"]["iterations"])
    old_threads = torch.get_num_threads()
    torch.set_num_threads(min(2, old_threads))
    before_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    timings: list[float] = []
    try:
        with torch.no_grad():
            for _ in range(warmup):
                model(batch)
            for _ in range(iterations):
                start = time.perf_counter()
                model(batch)
                timings.append((time.perf_counter() - start) * 1000.0)
    finally:
        torch.set_num_threads(old_threads)
    after_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    ordered = sorted(timings)

    def timing_percentile(q: float) -> float:
        return ordered[min(len(ordered) - 1, round((len(ordered) - 1) * q))]

    detail_bearing_instances = int(batch.detail_exists_mask.sum().item())
    padded_detail_slots = int(batch.attack_detail_mask.shape[1] * batch.instance_count)
    parameter_bytes = sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())
    buffer_bytes = sum(buffer.numel() * buffer.element_size() for buffer in model.buffers())

    def tensor_bytes(value: Any) -> int:
        if isinstance(value, torch.Tensor):
            return value.numel() * value.element_size()
        if is_dataclass(value):
            return sum(tensor_bytes(getattr(value, item.name)) for item in dataclass_fields(value))
        if isinstance(value, dict):
            return sum(tensor_bytes(item) for item in value.values())
        if isinstance(value, (list, tuple)):
            return sum(tensor_bytes(item) for item in value)
        return 0

    batch_bytes = tensor_bytes(batch)
    return {
        "device": "cpu",
        "torch_threads": min(2, old_threads),
        "iterations": iterations,
        "warmup_iterations": warmup,
        "decision_samples": int(batch.sample_indices.max().item() + 1) if batch.sample_indices.numel() else 0,
        "instance_count": batch.instance_count,
        "card_instance_token_count": batch.instance_count,
        "static_detail_token_count": valid_detail_tokens,
        "total_model_token_count": batch.instance_count + valid_detail_tokens,
        "instances_per_second": batch.instance_count * 1000.0 / max(sum(timings) / len(timings), 1e-9),
        "detail_bearing_instance_count": detail_bearing_instances,
        "padded_detail_token_slots": padded_detail_slots,
        "latency_ms": {
            "mean": sum(timings) / len(timings),
            "p50": timing_percentile(0.50),
            "p95": timing_percentile(0.95),
            "max": max(timings),
        },
        "memory": {
            "model_parameter_bytes": parameter_bytes,
            "model_buffer_bytes": buffer_bytes,
            "batch_tensor_bytes": batch_bytes,
            "estimated_model_and_batch_bytes": parameter_bytes + buffer_bytes + batch_bytes,
            "process_peak_rss_before_kib": before_rss,
            "process_peak_rss_after_kib": after_rss,
            "process_peak_rss_delta_kib": max(after_rss - before_rss, 0),
            "process_rss_note": "ru_maxrss is the whole-process historical peak; estimated bytes isolate model buffers and batch",
        },
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def evaluate_quality_gates(
    config: dict[str, Any],
    audit: dict[str, Any],
    validation_metrics: dict[str, Any],
    counterfactual: dict[str, Any],
    instance_counterfactual: dict[str, Any],
) -> dict[str, Any]:
    gates = config.get("gates", {})
    checks = {
        "parser_errors": (
            not bool(gates.get("require_zero_parser_errors", True))
            or int(audit.get("parser_error_count", 0)) == 0
        ),
        "card_id_coverage": (
            audit.get("static_lookup", {}).get("coverage") is not None
            and float(audit["static_lookup"]["coverage"])
            >= float(gates.get("minimum_card_id_coverage", 0.99))
        ),
        "attack_cost_coverage": (
            float(audit.get("detail_alignment", {}).get("attack_cost_coverage", 0.0))
            >= float(gates.get("minimum_attack_cost_coverage", 0.99))
        ),
        "static_artifact_detail_alignment": (
            audit.get("detail_alignment", {}).get("coverage") is not None
            and float(audit["detail_alignment"]["coverage"])
            >= float(gates.get("minimum_detail_alignment_coverage", 0.99))
        ),
        "finite_validation_metrics": all(
            math.isfinite(float(value))
            for value in validation_metrics.values()
            if isinstance(value, (int, float))
        ),
        "energy_counterfactual": (
            not bool(gates.get("require_energy_counterfactual", True))
            or bool(counterfactual.get("success", False))
        ),
        "instance_state_counterfactual": (
            not bool(gates.get("require_instance_state_counterfactual", True))
            or bool(instance_counterfactual.get("success", False))
        ),
    }
    result = {"checks": checks, "success": all(checks.values())}
    if not result["success"]:
        raise RuntimeError(f"quality gates failed: {result}")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train dynamic card-instance fusion on real replay decisions.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--static-artifact-dir", type=Path, required=True)
    parser.add_argument("--card-records", type=Path, required=True)
    parser.add_argument("--detail-metadata", type=Path, required=True)
    parser.add_argument("--train-replay-dir", type=Path, action="append", required=True)
    parser.add_argument("--validation-replay-dir", type=Path, action="append", required=True)
    parser.add_argument("--test-replay-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/dynamic_card_fusion"))
    parser.add_argument("--resume", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    completed_stage = "configuration"
    try:
        config = json.loads(args.config.read_text(encoding="utf-8"))
        set_seed(int(config["seed"]))
        cuda_info = cuda_runtime_info()
        runtime_config = config.get("runtime", {})
        if bool(runtime_config.get("require_cuda", False)) and not cuda_info["usable"]:
            raise RuntimeError(
                "this run requires a CUDA device supported by the installed PyTorch build: "
                f"{cuda_info['fallback_reason']}"
            )
        required_device_name = str(runtime_config.get("required_device_name_contains", "")).strip()
        actual_device_name = str(cuda_info.get("device_name") or "")
        if required_device_name and required_device_name.lower() not in actual_device_name.lower():
            raise RuntimeError(
                f"this run requires a CUDA device containing {required_device_name!r}, "
                f"but Kaggle assigned {actual_device_name!r}"
            )
        device = torch.device("cuda" if cuda_info["usable"] else "cpu")
        catalog = AttackCostCatalog.from_files(
            args.card_records,
            args.detail_metadata,
            args.static_artifact_dir / "card_id_to_index.json",
        )
        if catalog.max_details > int(config["model"]["max_details"]):
            raise ValueError(
                f"catalog needs {catalog.max_details} detail slots but config only provides "
                f"{config['model']['max_details']}"
            )

        completed_stage = "replay_loading"
        collections = {
            "train": load_replay_collection(args.train_replay_dir, int(config["data"]["max_train_decisions"])),
            "validation": load_replay_collection(
                args.validation_replay_dir, int(config["data"]["max_validation_decisions"])
            ),
            "test": load_replay_collection(args.test_replay_dir, int(config["data"]["max_test_decisions"])),
        }
        replay_metadata = split_metadata(collections)
        artifact_metadata = artifact_versions(args.static_artifact_dir, args.card_records, args.detail_metadata)
        run_config = {
            "config": config,
            "arguments": {
                "static_artifact_dir": str(args.static_artifact_dir),
                "card_records": str(args.card_records),
                "detail_metadata": str(args.detail_metadata),
                "train_replay_dirs": [str(path) for path in args.train_replay_dir],
                "validation_replay_dirs": [str(path) for path in args.validation_replay_dir],
                "test_replay_dirs": [str(path) for path in args.test_replay_dir],
                "resume": str(args.resume) if args.resume else None,
            },
            "runtime": {
                "device": str(device),
                "torch_version": torch.__version__,
                "cuda": cuda_info,
            },
        }
        _write_json(args.output_dir / "metadata" / "run_config.json", run_config)
        _write_json(args.output_dir / "metadata" / "replay_split.json", replay_metadata)
        _write_json(args.output_dir / "metadata" / "artifact_versions.json", artifact_metadata)

        completed_stage = "replay_audit"
        combined = combine_collections(collections.values())
        known_ids = load_known_static_ids(args.static_artifact_dir / "card_id_to_index.json")
        audit = build_report(combined, known_ids, catalog)
        static_alignment = validate_static_artifact_alignment(args.static_artifact_dir, catalog, config)
        audit.setdefault("detail_alignment", {})["replay_metadata_coverage"] = audit.get(
            "detail_alignment", {}
        ).get("coverage")
        audit["detail_alignment"].update({
            "static_artifact": static_alignment,
            "card_coverage": static_alignment["card_coverage"],
            "detail_slot_coverage": static_alignment["detail_slot_coverage"],
            "coverage": static_alignment["coverage"],
            "mismatch_examples": static_alignment["mismatch_examples"],
        })
        audit["split_summaries"] = replay_metadata["splits"]
        _write_json(args.output_dir / "audit" / "replay_feature_audit.json", audit)

        completed_stage = "smoke_and_tiny_overfit"
        tiny_result, gradient_result = tiny_overfit(
            args.static_artifact_dir,
            collections["train"].samples,
            catalog,
            config,
            device,
        )
        _write_json(args.output_dir / "evaluation" / "tiny_overfit.json", tiny_result)
        _write_json(args.output_dir / "evaluation" / "gradient_audit.json", gradient_result)

        completed_stage = "formal_training"
        best_path, last_path, best_metrics, model = train_formal(
            args.static_artifact_dir,
            collections,
            catalog,
            config,
            artifact_metadata,
            replay_metadata,
            args.output_dir,
            device,
            args.resume,
        )
        validation_loader = make_loader(collections["validation"], catalog, config, shuffle=False)
        test_loader = make_loader(collections["test"], catalog, config, shuffle=False)
        validation_metrics = run_epoch(model, validation_loader, device, config["training"]["loss_weights"])
        test_metrics = run_epoch(model, test_loader, device, config["training"]["loss_weights"])

        completed_stage = "reload_and_benchmark"
        fixed_samples = select_tiny_samples(collections["validation"].samples, catalog, config)
        fixed_batch = collate_dynamic_card_samples(
            fixed_samples, catalog, int(config["model"]["max_details"])
        )
        reload_result = checkpoint_reload_consistency(
            args.static_artifact_dir,
            best_path,
            model,
            fixed_batch,
            config,
            device,
            artifact_metadata,
        )
        diagnostics = diagnostic_examples(model, fixed_batch, device)
        counterfactual = energy_counterfactual_diagnostic(
            model,
            fixed_batch,
            device,
            strict=bool(config.get("gates", {}).get("require_energy_counterfactual", True)),
            minimum_probability_delta=float(
                config.get("gates", {}).get("minimum_energy_probability_delta", 0.0)
            ),
            max_candidates=int(config.get("gates", {}).get("counterfactual_max_candidates", 32)),
        )
        instance_counterfactual = instance_state_counterfactual_diagnostic(
            model,
            fixed_batch,
            device,
            strict=bool(config.get("gates", {}).get("require_instance_state_counterfactual", True)),
            minimum_token_l2=float(
                config.get("gates", {}).get("minimum_instance_counterfactual_token_l2", 0.0)
            ),
        )
        benchmark = cpu_benchmark(args.static_artifact_dir, best_path, fixed_batch, config)
        quality_gates = evaluate_quality_gates(
            config, audit, validation_metrics, counterfactual, instance_counterfactual
        )
        evaluation = {
            "best_epoch": best_metrics,
            "validation": validation_metrics,
            "test": test_metrics,
            "checkpoint_reload": reload_result,
            "tiny_overfit": tiny_result,
            "gradient_audit": gradient_result,
            "energy_counterfactual": counterfactual,
            "instance_state_counterfactual": instance_counterfactual,
            "quality_gates": quality_gates,
        }
        _write_json(args.output_dir / "evaluation" / "validation_metrics.json", evaluation)
        _write_json(
            args.output_dir / "evaluation" / "diagnostic_examples.json",
            {
                "attack_examples": diagnostics,
                "energy_counterfactual": counterfactual,
                "instance_state_counterfactual": instance_counterfactual,
            },
        )
        _write_json(args.output_dir / "benchmark" / "benchmark.json", benchmark)
        _write_json(args.output_dir / "evaluation" / "quality_gates.json", quality_gates)

        summary = {
            "success": True,
            "run_mode": config.get("run_mode", "formal"),
            "completed_stage": "complete",
            "device": str(device),
            "data": replay_metadata["splits"],
            "parser_error_count": audit["parser_error_count"],
            "card_id_coverage": audit["static_lookup"]["coverage"],
            "detail_alignment_coverage": audit["detail_alignment"]["coverage"],
            "payment_candidate_detail_unresolved_ratio": audit["energy_resolution"].get(
                "detail_unresolved_ratio"
            ),
            "special_energy_unresolved_detail_ratio": audit["energy_resolution"].get(
                "special_energy_unresolved_detail_ratio"
            ),
            "all_attack_payment_unsupervised_ratio": audit.get(
                "training_payment_supervision", {}
            ).get("unsupervised_ratio"),
            "tiny_overfit": {
                "success": tiny_result["success"],
                "loss_ratio": tiny_result["loss_ratio"],
            },
            "validation_metrics": validation_metrics,
            "test_metrics": test_metrics,
            "best_checkpoint": str(best_path),
            "last_checkpoint": str(last_path),
            "checkpoint_reload": reload_result,
            "energy_counterfactual": counterfactual,
            "instance_state_counterfactual": instance_counterfactual,
            "quality_gates": quality_gates,
            "benchmark": {
                "instance_count": benchmark["instance_count"],
                "mean_latency_ms": benchmark["latency_ms"]["mean"],
                "peak_rss_delta_kib": benchmark["memory"]["process_peak_rss_delta_kib"],
            },
            "remaining_issues": [],
        }
        _write_json(args.output_dir / "run_summary.json", summary)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    except BaseException as exc:
        failure = {
            "success": False,
            "completed_stage": completed_stage,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        _write_json(args.output_dir / "run_summary.json", failure)
        raise


if __name__ == "__main__":
    main()
