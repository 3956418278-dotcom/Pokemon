from __future__ import annotations

"""Leakage-audited evaluation for static CardEncoder v2.

Frozen artifact probes are retained as diagnostic input-retention reports only.
Acceptance instead uses the source checkpoint online: each structured target is
masked before inference and the trained heads must outperform both a same-seed
untrained model and a train-label baseline.  Ownership is evaluated with an
owner summary that excludes the candidate and a pre-fusion candidate token.

The catalog vocabulary and numeric normalization used by CardEncoder were
built from the complete, already-known card catalog.  That transductive schema
fact is reported rather than hidden; examples, labels and pairs remain
strictly split-local here.
"""

import argparse
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.card_dataset import CardDataset, collate_cards
from data.card_preprocessing import EXPECTED_SOURCE_SHA256, MASK_TOKEN
from training.pretrain_card_encoder import StaticPretrainingModel, clone_batch, move_batch
from training.validate_card_artifacts import validate_artifact_directory


EVALUATION_SCHEMA_VERSION = "static_card_embedding_evaluation_v2"
SUPPORTED_SPLIT_SCHEMA = "static_card_split_v2"
SUPPORTED_CACHE_SCHEMA = "static_card_v2"
SUPPORTED_ARTIFACT_SCHEMA = "static_card_artifacts_v2"
SUPPORTED_MODEL_VERSION = "card_encoder_v2"
RIDGE_ALPHAS = (1.0e-3, 1.0e-2, 1.0e-1, 1.0, 10.0, 100.0)
FIXED_BINARY_RIDGE_ALPHA = 1.0
ENERGY_ORDER = ("C", "G", "R", "W", "L", "P", "F", "D", "M", "N", "Y", "A")
FROZEN_DIAGNOSTICS = {
    "card_category",
    "pokemon_stage",
    "pokemon_type",
    "printed_hp",
    "retreat",
    "rule_flags",
    "detail_type",
    "attack_cost",
    "damage_base",
    "damage_mode",
    "same_species",
    "direct_evolution",
    "ownership",
}
ONLINE_REQUIRED_GATES = (
    "pokemon_stage",
    "pokemon_type",
    "printed_hp",
    "retreat",
    "rule_flags",
    "attack_cost",
    "damage_base",
    "damage_mode",
    "ownership",
)
REQUIRED_PROBES = (
    "card_category",
    "pokemon_stage",
    "pokemon_type",
    "printed_hp",
    "retreat",
    "rule_flags",
    "detail_type",
    "attack_cost",
    "damage_base",
    "damage_mode",
    "same_species",
    "direct_evolution",
    "ownership",
)


class EvaluationError(ValueError):
    """Raised when an honest evaluation cannot be constructed."""


def _load_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _load_tensor(path: Path) -> torch.Tensor:
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        value = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch < 2.1
        value = torch.load(path, map_location="cpu")
    if not isinstance(value, torch.Tensor):
        raise EvaluationError(f"expected tensor in {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _counts(values: Sequence[Any]) -> dict[str, int]:
    return dict(sorted((str(key), int(value)) for key, value in Counter(values).items()))


def _finite_number(value: Any) -> bool:
    return not isinstance(value, float) or math.isfinite(value)


def _all_finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_all_finite(item) for item in value)
    return _finite_number(value)


def _resolve_checkpoint(
    checkpoint_value: str,
    *,
    artifact_dir: Path,
    split_manifest_path: Path,
) -> Path | None:
    raw = Path(checkpoint_value)
    candidates = [raw]
    if not raw.is_absolute():
        candidates.extend((artifact_dir / raw, split_manifest_path.parent / raw))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _validate_split_manifest(
    manifest: dict[str, Any],
    cards: list[dict[str, Any]],
) -> dict[str, list[int]]:
    if manifest.get("schema_version") != SUPPORTED_SPLIT_SCHEMA:
        raise EvaluationError(f"split manifest must be {SUPPORTED_SPLIT_SCHEMA}")
    if manifest.get("mode") != "card_id":
        raise EvaluationError("only card_id split manifests are supported")
    split_keys = {
        "train": ("train_indices", "train_card_ids"),
        "validation": ("validation_indices", "validation_card_ids"),
        "test": ("test_indices", "test_card_ids"),
    }
    splits: dict[str, list[int]] = {}
    seen: set[int] = set()
    for name, (indices_key, ids_key) in split_keys.items():
        indices = [int(value) for value in manifest.get(indices_key, [])]
        if not indices:
            raise EvaluationError(f"{name} split is empty")
        if len(indices) != len(set(indices)):
            raise EvaluationError(f"{name} split contains duplicate indices")
        if any(index < 0 or index >= len(cards) for index in indices):
            raise EvaluationError(f"{name} split contains an out-of-range card index")
        overlap = seen.intersection(indices)
        if overlap:
            raise EvaluationError(f"card indices occur in multiple splits: {sorted(overlap)[:10]}")
        declared_ids = [str(value) for value in manifest.get(ids_key, [])]
        actual_ids = [str(cards[index]["card_id"]) for index in indices]
        if declared_ids != actual_ids:
            raise EvaluationError(f"{name} card IDs do not match its ordered indices")
        seen.update(indices)
        splits[name] = indices
    if seen != set(range(len(cards))):
        missing = sorted(set(range(len(cards))) - seen)
        raise EvaluationError(f"split manifest is not a complete card partition; missing {missing[:10]}")
    return splits


def _checkpoint_lineage(
    artifact_manifest: dict[str, Any],
    split_manifest_path: Path,
    artifact_dir: Path,
    expected_checkpoint_stage: str | None,
) -> dict[str, Any]:
    record = artifact_manifest.get("checkpoint")
    if not isinstance(record, dict) or not record.get("path") or not record.get("sha256"):
        raise EvaluationError("artifact manifest has no checkpoint lineage")
    path = _resolve_checkpoint(
        str(record["path"]),
        artifact_dir=artifact_dir,
        split_manifest_path=split_manifest_path,
    )
    if path is None:
        if expected_checkpoint_stage is not None:
            raise EvaluationError(
                "the source checkpoint is unavailable, so split-selection lineage cannot be verified"
            )
        return {
            "path": str(record["path"]),
            "sha256": str(record["sha256"]),
            "stage": None,
            "verified": False,
        }
    actual_hash = _sha256(path)
    if actual_hash != str(record["sha256"]):
        raise EvaluationError("artifact checkpoint SHA-256 does not match the source checkpoint")
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise EvaluationError("source checkpoint is not a mapping")
    stage = checkpoint.get("stage")
    if expected_checkpoint_stage is not None and stage != expected_checkpoint_stage:
        raise EvaluationError(
            f"evaluation requires checkpoint stage {expected_checkpoint_stage!r}, got {stage!r}"
        )
    expected_split_hash = checkpoint.get("lineage", {}).get("split_manifest_sha256")
    actual_split_hash = _sha256(split_manifest_path)
    if expected_split_hash is None:
        raise EvaluationError("checkpoint has no split_manifest_sha256 lineage")
    if str(expected_split_hash) != actual_split_hash:
        raise EvaluationError("checkpoint split manifest lineage does not match the evaluator split")
    return {
        "path": str(path),
        "sha256": actual_hash,
        "stage": stage,
        "verified": True,
        "split_manifest_sha256": actual_split_hash,
    }


def _semantic_contract(
    cards: list[dict[str, Any]],
    details: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    """Check the audited catalog facts which are independent of model loss.

    Alignment can only prove that an export faithfully reproduces its cache.
    These checks prevent a consistently-wrong cache and export from passing.
    """

    by_card: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in details:
        by_card[str(row.get("card_id"))].append(row)
    card_by_id = {str(card.get("card_id")): card for card in cards}
    detail_counts = _counts([row.get("detail_type") for row in details])
    source_rows = [int(row.get("source_row_index", -1)) for row in details]
    attacks = [row for row in details if row.get("detail_type") == "ATTACK"]
    attack_ids = [row.get("attack_id") for row in attacks]

    fossils = [card for card in cards if "FOSSIL" in (card.get("card_tags") or [])]
    fossil_ok = bool(len(fossils) == 5)
    if fossil_ok:
        fossil_ok = all(
            card.get("card_type") == "ITEM"
            and card.get("printed_hp") == 60
            and card.get("hp_applicability") == "PLAYABLE_AS_POKEMON"
            and len(by_card[str(card["card_id"])]) == 2
            and all(row.get("detail_type") == "CARD_EFFECT" for row in by_card[str(card["card_id"])])
            for card in fossils
        )

    core = card_by_id.get("1180", {})
    core_rows = by_card.get("1180", [])
    geobuster = next((row for row in core_rows if row.get("move_name") == "Geobuster"), None)
    core_ok = bool(
        core.get("card_type") == "TOOL"
        and "TECHNICAL_MACHINE" in (core.get("card_tags") or [])
        and [row.get("detail_type") for row in core_rows] == ["CARD_EFFECT", "ATTACK"]
        and geobuster is not None
        and geobuster.get("attack_id") == 1556
        and geobuster.get("energy_costs") == {"F": 4}
        and geobuster.get("base_damage") == 350
    )

    koraidon = [
        (row.get("move_name"), row.get("attack_id"))
        for row in by_card.get("979", [])
        if row.get("detail_type") == "ATTACK"
    ]
    koraidon_ok = koraidon == [("Orichalcum Fang", 1408), ("Impact Blow", 1409)]
    corrections = manifest.get("known_effect_text_corrections") or []
    correction_pairs = [(row.get("card_id"), row.get("attack_id")) for row in corrections]
    correction_ok = correction_pairs == [(480, 678), (481, 680)] and all(
        row.get("correction_id")
        and row.get("reason") == "english_csv_contains_japanese_attack_effect_text"
        and row.get("before_hash")
        and row.get("after_hash")
        for row in corrections
    )
    non_pokemon = [card for card in cards if card.get("card_category") != "POKEMON"]
    non_pokemon_attacks = [
        (str(row.get("card_id")), row.get("move_name"), row.get("attack_id"))
        for row in details
        if card_by_id.get(str(row.get("card_id")), {}).get("card_category") != "POKEMON"
        and row.get("detail_type") == "ATTACK"
    ]
    non_pokemon_abilities = [
        row
        for row in details
        if card_by_id.get(str(row.get("card_id")), {}).get("card_category") != "POKEMON"
        and row.get("detail_type") == "ABILITY"
    ]
    allowed_rule_flags = {"ACE_SPEC", "MEGA_POKEMON_EX", "POKEMON_EX", "TERA"}
    rule_counts = Counter(flag for card in cards for flag in (card.get("rule_flags") or []))
    rule_rows_valid = all(
        len(card.get("rule_flags") or []) == len(set(card.get("rule_flags") or []))
        and set(card.get("rule_flags") or []) <= allowed_rule_flags
        for card in cards
    )

    checks = {
        "source_sha256": manifest.get("source_sha256") == EXPECTED_SOURCE_SHA256,
        "card_count": len(cards) == 1267 and int(manifest.get("card_count", -1)) == 1267,
        "detail_count": len(details) == 2014 and int(manifest.get("detail_count", -1)) == 2014,
        "detail_type_counts": detail_counts == {"ABILITY": 218, "ATTACK": 1556, "CARD_EFFECT": 240}
        and manifest.get("detail_type_counts") == {"ABILITY": 218, "ATTACK": 1556, "CARD_EFFECT": 240},
        "unresolved_zero": int(manifest.get("unresolved_count", -1)) == 0,
        "source_detail_rows_unique": len(source_rows) == 2014
        and len(set(source_rows)) == 2014
        and min(source_rows, default=-1) >= 0
        and max(source_rows, default=0) < int(manifest.get("source_row_count", 0)),
        "attack_ids_complete_unique": len(attacks) == 1556
        and all(value is not None for value in attack_ids)
        and len(set(int(value) for value in attack_ids if value is not None)) == 1556,
        "fossil_contract": fossil_ok,
        "core_memory_contract": core_ok,
        "koraidon_mapping": koraidon_ok,
        "known_text_corrections": correction_ok,
        "non_pokemon_has_no_pokemon_type": all(card.get("pokemon_type") is None for card in non_pokemon),
        "non_pokemon_detail_exceptions": not non_pokemon_abilities
        and non_pokemon_attacks == [("1180", "Geobuster", 1556)],
        "canonical_rule_flags": rule_rows_valid
        and dict(rule_counts) == {
            "ACE_SPEC": 29,
            "MEGA_POKEMON_EX": 30,
            "POKEMON_EX": 121,
            "TERA": 32,
        },
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "limitations": [
            "The cache does not retain the complete raw CSV rows, so non-empty source-row coverage is proved by the pinned source hash, exact counts, unique source_row_index values, and preprocessing tests rather than reconstructed here.",
            "Semantic contracts cover the audited failure modes and known exceptions; they are not a substitute for reviewing a future source-corpus hash change.",
        ],
    }


def _fit_standardizer(x_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if x_train.ndim != 2 or not len(x_train):
        raise EvaluationError("cannot fit a standardizer without 2-D training examples")
    mean = x_train.mean(axis=0, dtype=np.float64)
    std = x_train.std(axis=0, dtype=np.float64)
    std = np.where(std < 1.0e-8, 1.0, std)
    return mean, std


def _standardize(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (x.astype(np.float64, copy=False) - mean) / std


def _ridge_fit(x: np.ndarray, y: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    if x.ndim != 2 or y.shape[0] != x.shape[0]:
        raise EvaluationError("ridge input rows are not aligned")
    target = y.astype(np.float64, copy=False)
    if target.ndim == 1:
        target = target[:, None]
    y_mean = target.mean(axis=0)
    centered = target - y_mean
    gram = x.T @ x
    regularized = gram + float(alpha) * np.eye(gram.shape[0], dtype=np.float64)
    try:
        weights = np.linalg.solve(regularized, x.T @ centered)
    except np.linalg.LinAlgError:
        weights = np.linalg.pinv(regularized) @ (x.T @ centered)
    return weights, y_mean


def _ridge_predict(x: np.ndarray, model: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    weights, intercept = model
    return x @ weights + intercept


def _accuracy(y_true: Sequence[Any], y_pred: Sequence[Any]) -> float:
    if not y_true:
        return 0.0
    return float(sum(left == right for left, right in zip(y_true, y_pred)) / len(y_true))


def _balanced_accuracy(y_true: Sequence[Any], y_pred: Sequence[Any]) -> float:
    grouped: dict[Any, list[bool]] = defaultdict(list)
    for truth, prediction in zip(y_true, y_pred):
        grouped[truth].append(truth == prediction)
    if not grouped:
        return 0.0
    return float(np.mean([np.mean(values) for values in grouped.values()]))


def _binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    truth = y_true.astype(np.int64)
    prediction = y_pred.astype(np.int64)
    tp = int(np.sum((truth == 1) & (prediction == 1)))
    fp = int(np.sum((truth == 0) & (prediction == 1)))
    fn = int(np.sum((truth == 1) & (prediction == 0)))
    accuracy = float(np.mean(truth == prediction)) if len(truth) else 0.0
    recalls = []
    for label in (0, 1):
        mask = truth == label
        if mask.any():
            recalls.append(float(np.mean(prediction[mask] == label)))
    balanced = float(np.mean(recalls)) if recalls else 0.0
    denominator = 2 * tp + fp + fn
    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced,
        "f1": float((2 * tp) / denominator) if denominator else 0.0,
    }


def _confidence_status(
    split_labels: dict[str, Sequence[Any]],
    *,
    classification: bool,
) -> tuple[str, list[str]]:
    reasons: list[str] = []
    for name in ("train", "validation", "test"):
        count = len(split_labels[name])
        if count == 0:
            return "unsupported", [f"{name} has no applicable examples"]
        if count < (50 if name == "train" else 20):
            reasons.append(f"{name} has only {count} examples")
    if classification:
        train_classes = set(split_labels["train"])
        if len(train_classes) < 2:
            return "unsupported", ["training labels contain fewer than two classes"]
        for name, labels in split_labels.items():
            counts = Counter(labels)
            rare = sorted(str(label) for label, count in counts.items() if count < 5)
            if rare:
                reasons.append(f"{name} classes with support <5: {rare}")
        unseen = sorted(set(split_labels["test"]) - train_classes, key=str)
        if unseen:
            reasons.append(f"test classes absent from train: {[str(value) for value in unseen]}")
    return ("low_confidence" if reasons else "supported"), reasons


def _probe_shell(
    name: str,
    split_labels: dict[str, Sequence[Any]],
    *,
    task: str,
    diagnostic_only: bool = False,
) -> dict[str, Any]:
    classification = task in {"classification", "binary", "multilabel"}
    status, reasons = _confidence_status(
        split_labels,
        classification=classification and task != "multilabel",
    )
    return {
        "name": name,
        "task": task,
        "status": status,
        "confidence_reasons": reasons,
        "diagnostic_only": bool(diagnostic_only),
        "support": {
            split: {
                "examples": len(labels),
                **({"class_counts": _counts(labels)} if classification and task != "multilabel" else {}),
            }
            for split, labels in split_labels.items()
        },
        "fit_protocol": {
            "embedding_state": "frozen",
            "standardizer_fit": "train_only",
            "probe_fit": "train_only",
            "model_selection": "validation_only",
            "test_evaluations": 1,
        },
    }


def _categorical_probe(
    name: str,
    features: dict[str, np.ndarray],
    labels: dict[str, list[str]],
    *,
    diagnostic_only: bool = False,
) -> dict[str, Any]:
    report = _probe_shell(
        name,
        labels,
        task="classification",
        diagnostic_only=diagnostic_only,
    )
    if report["status"] == "unsupported":
        report.update({"selection": None, "baseline": None, "test": None, "gate": "not_applicable"})
        return report
    classes = sorted(set(labels["train"]))
    class_to_id = {label: index for index, label in enumerate(classes)}
    mean, std = _fit_standardizer(features["train"])
    x = {split: _standardize(value, mean, std) for split, value in features.items()}
    y_train = np.zeros((len(labels["train"]), len(classes)), dtype=np.float64)
    y_train[np.arange(len(labels["train"])), [class_to_id[value] for value in labels["train"]]] = 1.0
    candidates: list[dict[str, float]] = []
    models: dict[float, tuple[np.ndarray, np.ndarray]] = {}
    for alpha in RIDGE_ALPHAS:
        model = _ridge_fit(x["train"], y_train, alpha)
        models[alpha] = model
        predictions = [classes[index] for index in np.argmax(_ridge_predict(x["validation"], model), axis=1)]
        score = _balanced_accuracy(labels["validation"], predictions)
        candidates.append({"alpha": alpha, "validation_balanced_accuracy": score})
    selected = max(candidates, key=lambda row: (row["validation_balanced_accuracy"], -row["alpha"]))
    model = models[float(selected["alpha"])]
    test_predictions = [classes[index] for index in np.argmax(_ridge_predict(x["test"], model), axis=1)]
    majority = Counter(labels["train"]).most_common(1)[0][0]
    baseline_predictions = [majority] * len(labels["test"])
    test_metrics = {
        "accuracy": _accuracy(labels["test"], test_predictions),
        "balanced_accuracy": _balanced_accuracy(labels["test"], test_predictions),
        "unseen_test_examples": int(sum(value not in class_to_id for value in labels["test"])),
    }
    baseline = {
        "strategy": "train_majority_class",
        "class": majority,
        "test_accuracy": _accuracy(labels["test"], baseline_predictions),
        "test_balanced_accuracy": _balanced_accuracy(labels["test"], baseline_predictions),
    }
    passes = test_metrics["balanced_accuracy"] + 1.0e-12 >= baseline["test_balanced_accuracy"]
    report.update(
        {
            "classes_fit_from_train": classes,
            "selection": {
                "criterion": "validation_balanced_accuracy",
                "candidates": candidates,
                "selected_alpha": selected["alpha"],
            },
            "baseline": baseline,
            "test": test_metrics,
            "gate": _gate(report, passes),
        }
    )
    return report


def _regression_probe(
    name: str,
    features: dict[str, np.ndarray],
    labels: dict[str, list[float]],
) -> dict[str, Any]:
    report = _probe_shell(name, labels, task="regression")
    if report["status"] == "unsupported":
        report.update({"selection": None, "baseline": None, "test": None, "gate": "not_applicable"})
        return report
    mean, std = _fit_standardizer(features["train"])
    x = {split: _standardize(value, mean, std) for split, value in features.items()}
    y = {split: np.asarray(value, dtype=np.float64) for split, value in labels.items()}
    candidates: list[dict[str, float]] = []
    models: dict[float, tuple[np.ndarray, np.ndarray]] = {}
    for alpha in RIDGE_ALPHAS:
        model = _ridge_fit(x["train"], y["train"], alpha)
        models[alpha] = model
        prediction = _ridge_predict(x["validation"], model).reshape(-1)
        candidates.append({"alpha": alpha, "validation_mae": float(np.mean(np.abs(prediction - y["validation"])))})
    selected = min(candidates, key=lambda row: (row["validation_mae"], -row["alpha"]))
    prediction = _ridge_predict(x["test"], models[float(selected["alpha"])]).reshape(-1)
    baseline_value = float(np.median(y["train"]))
    mae = float(np.mean(np.abs(prediction - y["test"])))
    rmse = float(np.sqrt(np.mean(np.square(prediction - y["test"]))))
    baseline_mae = float(np.mean(np.abs(y["test"] - baseline_value)))
    report.update(
        {
            "selection": {
                "criterion": "validation_mae",
                "candidates": candidates,
                "selected_alpha": selected["alpha"],
            },
            "baseline": {
                "strategy": "train_median",
                "value": baseline_value,
                "test_mae": baseline_mae,
            },
            "test": {"mae": mae, "rmse": rmse},
            "gate": _gate(report, mae <= baseline_mae + 1.0e-12),
        }
    )
    return report


def _threshold_candidates(scores: np.ndarray) -> list[float]:
    unique = np.unique(scores.astype(np.float64))
    if not len(unique):
        return [0.5]
    if len(unique) > 200:
        unique = np.quantile(unique, np.linspace(0.0, 1.0, 200))
    epsilon = max(1.0, float(np.max(np.abs(unique)))) * 1.0e-9
    middle = ((unique[:-1] + unique[1:]) / 2.0).tolist()
    return [float(unique[0] - epsilon), *[float(value) for value in middle], float(unique[-1] + epsilon)]


def _binary_probe(
    name: str,
    features: dict[str, np.ndarray],
    labels: dict[str, list[int]],
    *,
    limitations: list[str] | None = None,
    return_test_predictions: bool = False,
) -> dict[str, Any]:
    report = _probe_shell(name, labels, task="binary")
    if limitations:
        report["limitations"] = list(limitations)
    if report["status"] == "unsupported":
        report.update({"selection": None, "baseline": None, "test": None, "gate": "not_applicable"})
        return report
    mean, std = _fit_standardizer(features["train"])
    x = {split: _standardize(value, mean, std) for split, value in features.items()}
    y = {split: np.asarray(value, dtype=np.int64) for split, value in labels.items()}
    model = _ridge_fit(x["train"], y["train"], FIXED_BINARY_RIDGE_ALPHA)
    validation_scores = _ridge_predict(x["validation"], model).reshape(-1)
    candidates = []
    for threshold in _threshold_candidates(validation_scores):
        metrics = _binary_metrics(y["validation"], validation_scores >= threshold)
        candidates.append({"threshold": threshold, "validation_balanced_accuracy": metrics["balanced_accuracy"]})
    selected = max(candidates, key=lambda row: (row["validation_balanced_accuracy"], -abs(row["threshold"] - 0.5)))
    test_scores = _ridge_predict(x["test"], model).reshape(-1)
    test_prediction = (test_scores >= float(selected["threshold"])).astype(np.int64)
    test_metrics = _binary_metrics(y["test"], test_prediction)
    majority = int(np.mean(y["train"]) >= 0.5)
    baseline_metrics = _binary_metrics(y["test"], np.full_like(y["test"], majority))
    report.update(
        {
            "selection": {
                "fixed_ridge_alpha": FIXED_BINARY_RIDGE_ALPHA,
                "criterion": "validation_balanced_accuracy",
                "candidate_count": len(candidates),
                "selected_threshold": selected["threshold"],
                "selected_validation_balanced_accuracy": selected["validation_balanced_accuracy"],
            },
            "baseline": {"strategy": "train_majority_class", "class": majority, **baseline_metrics},
            "test": test_metrics,
            "gate": _gate(report, test_metrics["balanced_accuracy"] + 1.0e-12 >= baseline_metrics["balanced_accuracy"]),
        }
    )
    if return_test_predictions:
        report["_test_prediction"] = test_prediction.tolist()
        report["_baseline_test_prediction"] = np.full_like(y["test"], majority).tolist()
    return report


def _gate(report: dict[str, Any], passes: bool) -> str:
    if report["diagnostic_only"]:
        return "diagnostic_only"
    if report["status"] != "supported":
        return "not_applicable"
    return "pass" if passes else "fail"


def _mark_frozen_diagnostic(report: dict[str, Any]) -> dict[str, Any]:
    """Frozen probes measure input retention, never training semantics."""

    report["diagnostic_only"] = True
    report["gate"] = "diagnostic_only"
    report.setdefault("limitations", []).append(
        "The probed target is an original CardEncoder input or the frozen artifact contains target-context; this result measures representation retention, not target-masked learning."
    )
    for child in (report.get("per_label") or {}).values():
        child["diagnostic_only"] = True
        child["gate"] = "diagnostic_only"
        child.setdefault("limitations", []).append(
            "This per-label result belongs to a frozen input-retention diagnostic and cannot gate training semantics."
        )
    return report


def _multioutput_regression_probe(
    name: str,
    features: dict[str, np.ndarray],
    labels: dict[str, np.ndarray],
    output_names: Sequence[str],
) -> dict[str, Any]:
    shell_labels = {split: list(range(len(value))) for split, value in labels.items()}
    report = _probe_shell(name, shell_labels, task="regression")
    if report["status"] == "unsupported":
        report.update({"selection": None, "baseline": None, "test": None, "gate": "not_applicable"})
        return report
    mean, std = _fit_standardizer(features["train"])
    x = {split: _standardize(value, mean, std) for split, value in features.items()}
    y = {split: value.astype(np.float64, copy=False) for split, value in labels.items()}
    candidates: list[dict[str, float]] = []
    models: dict[float, tuple[np.ndarray, np.ndarray]] = {}
    for alpha in RIDGE_ALPHAS:
        model = _ridge_fit(x["train"], y["train"], alpha)
        models[alpha] = model
        prediction = _ridge_predict(x["validation"], model)
        candidates.append({"alpha": alpha, "validation_mae": float(np.mean(np.abs(prediction - y["validation"])))})
    selected = min(candidates, key=lambda row: (row["validation_mae"], -row["alpha"]))
    raw = _ridge_predict(x["test"], models[float(selected["alpha"])])
    rounded = np.maximum(0.0, np.rint(raw))
    baseline_vector = np.median(y["train"], axis=0)
    mae = float(np.mean(np.abs(raw - y["test"])))
    baseline_mae = float(np.mean(np.abs(y["test"] - baseline_vector)))
    per_output = {
        output_names[index]: {
            "train_nonzero": int(np.sum(y["train"][:, index] > 0)),
            "test_mae": float(np.mean(np.abs(raw[:, index] - y["test"][:, index]))),
            "status": "unsupported" if np.all(y["train"][:, index] == y["train"][0, index]) else (
                "low_confidence" if np.sum(y["train"][:, index] > 0) < 5 else "supported"
            ),
        }
        for index in range(len(output_names))
    }
    report.update(
        {
            "outputs": per_output,
            "selection": {
                "criterion": "validation_mae",
                "candidates": candidates,
                "selected_alpha": selected["alpha"],
            },
            "baseline": {
                "strategy": "train_coordinate_median",
                "vector": [float(value) for value in baseline_vector],
                "test_mae": baseline_mae,
            },
            "test": {
                "mae": mae,
                "exact_vector_accuracy_after_rounding": float(np.mean(np.all(rounded == y["test"], axis=1))),
            },
            "gate": _gate(report, mae <= baseline_mae + 1.0e-12),
        }
    )
    return report


def _multilabel_probe(
    name: str,
    features: dict[str, np.ndarray],
    label_sets: dict[str, list[set[str]]],
    vocabulary: Sequence[str],
) -> dict[str, Any]:
    shell = {split: list(range(len(rows))) for split, rows in label_sets.items()}
    report = _probe_shell(name, shell, task="multilabel")
    report["label_schema"] = {
        "source": "complete_known_catalog_transductive_schema",
        "labels": list(vocabulary),
    }
    if not vocabulary or report["status"] == "unsupported":
        report.update({"selection": None, "baseline": None, "test": None, "gate": "not_applicable"})
        return report
    predictions: dict[str, list[np.ndarray]] = {split: [] for split in label_sets}
    baselines: dict[str, list[np.ndarray]] = {split: [] for split in label_sets}
    label_reports: dict[str, Any] = {}
    for label in vocabulary:
        binary = {
            split: [int(label in values) for values in rows]
            for split, rows in label_sets.items()
        }
        child = _binary_probe(
            f"{name}:{label}",
            features,
            binary,
            return_test_predictions=True,
        )
        label_reports[label] = child
        if child["test"] is None:
            predictions["test"].append(np.zeros(len(binary["test"]), dtype=np.int64))
            baselines["test"].append(np.zeros(len(binary["test"]), dtype=np.int64))
            continue
        predictions["test"].append(np.asarray(child.pop("_test_prediction"), dtype=np.int64))
        baselines["test"].append(
            np.asarray(child.pop("_baseline_test_prediction"), dtype=np.int64)
        )
    truth = np.asarray([[int(label in row) for label in vocabulary] for row in label_sets["test"]], dtype=np.int64)
    prediction = np.stack(predictions["test"], axis=1)
    baseline_prediction = np.stack(baselines["test"], axis=1)

    def multilabel_metrics(value: np.ndarray) -> dict[str, float]:
        tp = int(np.sum((truth == 1) & (value == 1)))
        fp = int(np.sum((truth == 0) & (value == 1)))
        fn = int(np.sum((truth == 1) & (value == 0)))
        denominator = 2 * tp + fp + fn
        return {
            "micro_f1": float(2 * tp / denominator) if denominator else 1.0,
            "exact_match_accuracy": float(np.mean(np.all(truth == value, axis=1))),
        }

    test_metrics = multilabel_metrics(prediction)
    baseline_metrics = multilabel_metrics(baseline_prediction)
    supported_children = [row for row in label_reports.values() if row["status"] == "supported"]
    if not supported_children:
        report["status"] = "low_confidence"
        report["confidence_reasons"].append("no rule flag has full supported confidence")
    report.update(
        {
            "per_label": label_reports,
            "selection": {"method": "per-label validation threshold; fixed ridge alpha"},
            "baseline": {"strategy": "per-label train majority", **baseline_metrics},
            "test": test_metrics,
            "gate": _gate(report, test_metrics["micro_f1"] + 1.0e-12 >= baseline_metrics["micro_f1"]),
        }
    )
    return report


def _select_card_rows(
    cards: list[dict[str, Any]],
    card_embeddings: np.ndarray,
    splits: dict[str, list[int]],
    value: Callable[[dict[str, Any]], Any],
    applicable: Callable[[dict[str, Any]], bool],
) -> tuple[dict[str, np.ndarray], dict[str, list[Any]]]:
    features: dict[str, np.ndarray] = {}
    labels: dict[str, list[Any]] = {}
    for split, indices in splits.items():
        selected = [index for index in indices if applicable(cards[index])]
        features[split] = card_embeddings[selected]
        labels[split] = [value(cards[index]) for index in selected]
    return features, labels


def _select_detail_rows(
    details: list[dict[str, Any]],
    detail_embeddings: np.ndarray,
    splits: dict[str, list[int]],
    value: Callable[[dict[str, Any]], Any],
    applicable: Callable[[dict[str, Any]], bool],
) -> tuple[dict[str, np.ndarray], dict[str, list[Any]], dict[str, list[int]]]:
    features: dict[str, np.ndarray] = {}
    labels: dict[str, list[Any]] = {}
    row_indices: dict[str, list[int]] = {}
    for split, card_indices in splits.items():
        owners = set(card_indices)
        selected = [
            index
            for index, detail in enumerate(details)
            if int(detail["card_index"]) in owners and applicable(detail)
        ]
        features[split] = detail_embeddings[selected]
        labels[split] = [value(details[index]) for index in selected]
        row_indices[split] = selected
    return features, labels, row_indices


def _pair_features(left: np.ndarray, right: np.ndarray, *, directed: bool) -> np.ndarray:
    if directed:
        return np.concatenate((left, right, right - left, left * right), axis=-1)
    return np.concatenate((np.abs(left - right), left * right), axis=-1)


def _balanced_negative_pairs(
    positives: Sequence[tuple[int, int]],
    candidates: Sequence[int],
    *,
    seed: int,
    directed: bool,
    max_pairs: int,
) -> tuple[list[tuple[int, int]], list[int]]:
    rng = random.Random(seed)
    positive_set = set(positives)
    positive_canonical = (
        positive_set if directed else {tuple(sorted(value)) for value in positive_set}
    )
    selected_positive = list(positives)
    if len(selected_positive) > max_pairs:
        selected_positive = rng.sample(selected_positive, max_pairs)
    negative: set[tuple[int, int]] = set()
    attempts = 0
    target = len(selected_positive)
    while len(negative) < target and attempts < max(1000, target * 100):
        attempts += 1
        left, right = rng.choice(candidates), rng.choice(candidates)
        pair = (left, right)
        canonical = pair if directed else tuple(sorted(pair))
        if left == right or canonical in positive_canonical or canonical in negative:
            continue
        negative.add(canonical)
    pairs = selected_positive + list(sorted(negative))
    labels = [1] * len(selected_positive) + [0] * len(negative)
    return pairs, labels


def _same_species_examples(
    cards: list[dict[str, Any]],
    embeddings: np.ndarray,
    indices: list[int],
    *,
    seed: int,
    max_pairs: int,
) -> tuple[np.ndarray, list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    candidates = [index for index in indices if cards[index].get("species")]
    for index in candidates:
        groups[str(cards[index]["species"])].append(index)
    positives = sorted(
        (values[left], values[right])
        for values in groups.values()
        for left in range(len(values))
        for right in range(left + 1, len(values))
    )
    pairs, labels = _balanced_negative_pairs(
        positives,
        candidates,
        seed=seed,
        directed=False,
        max_pairs=max_pairs,
    )
    if not pairs:
        return np.empty((0, embeddings.shape[1] * 2), dtype=np.float64), []
    left = embeddings[[pair[0] for pair in pairs]]
    right = embeddings[[pair[1] for pair in pairs]]
    return _pair_features(left, right, directed=False), labels


def _direct_evolution_examples(
    cards: list[dict[str, Any]],
    embeddings: np.ndarray,
    indices: list[int],
    *,
    seed: int,
    max_pairs: int,
) -> tuple[np.ndarray, list[int]]:
    allowed = set(indices)
    names: dict[str, list[int]] = defaultdict(list)
    for index in indices:
        names[str(cards[index]["card_name"])].append(index)
    positives: set[tuple[int, int]] = set()
    for child in indices:
        parent_name = cards[child].get("evolves_from_card_name")
        if parent_name:
            positives.update((parent, child) for parent in names.get(str(parent_name), []) if parent in allowed)
    for parent in indices:
        child_name = cards[parent].get("evolves_to_card_name")
        if child_name:
            positives.update((parent, child) for child in names.get(str(child_name), []) if child in allowed)
    candidates = [
        index
        for index in indices
        if cards[index].get("card_category") == "POKEMON" or "FOSSIL" in (cards[index].get("card_tags") or [])
    ]
    pairs, labels = _balanced_negative_pairs(
        sorted(positives),
        candidates,
        seed=seed,
        directed=True,
        max_pairs=max_pairs,
    )
    if not pairs:
        return np.empty((0, embeddings.shape[1] * 4), dtype=np.float64), []
    left = embeddings[[pair[0] for pair in pairs]]
    right = embeddings[[pair[1] for pair in pairs]]
    return _pair_features(left, right, directed=True), labels


def _ownership_examples(
    details: list[dict[str, Any]],
    card_embeddings: np.ndarray,
    detail_embeddings: np.ndarray,
    indices: list[int],
    *,
    seed: int,
    max_pairs: int,
) -> tuple[np.ndarray, list[int]]:
    allowed = set(indices)
    by_type: dict[str, list[int]] = defaultdict(list)
    rows = [row for row, detail in enumerate(details) if int(detail["card_index"]) in allowed]
    for row in rows:
        by_type[str(details[row]["detail_type"])].append(row)
    rng = random.Random(seed)
    if len(rows) > max_pairs:
        rows = rng.sample(rows, max_pairs)
    owner_rows: list[int] = []
    candidate_rows: list[int] = []
    labels: list[int] = []
    for row in rows:
        owner = int(details[row]["card_index"])
        owner_rows.append(owner)
        candidate_rows.append(row)
        labels.append(1)
        alternatives = [
            candidate
            for candidate in by_type[str(details[row]["detail_type"])]
            if int(details[candidate]["card_index"]) != owner
        ]
        if alternatives:
            owner_rows.append(owner)
            candidate_rows.append(rng.choice(alternatives))
            labels.append(0)
    if not labels:
        width = card_embeddings.shape[1] + detail_embeddings.shape[1] * 3
        return np.empty((0, width), dtype=np.float64), []
    owner = card_embeddings[owner_rows]
    candidate = detail_embeddings[candidate_rows]
    if owner.shape[1] != candidate.shape[1]:
        features = np.concatenate((owner, candidate), axis=-1)
    else:
        features = np.concatenate((owner, candidate, np.abs(owner - candidate), owner * candidate), axis=-1)
    return features, labels


def _checkpoint_payload(path: Path) -> dict[str, Any]:
    try:
        value = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    if not isinstance(value, dict):
        raise EvaluationError("source checkpoint is not a mapping")
    required = {"config", "schema", "model_state", "training_schema_version", "data_schema_version"}
    missing = sorted(required - set(value))
    if missing:
        raise EvaluationError(f"source checkpoint is missing online-evaluation fields: {missing}")
    return value


def _vocab_index(schema: dict[str, Any], field: str, token: str) -> int:
    vocab = schema.get("vocab", {}).get(field)
    if not isinstance(vocab, dict) or token not in vocab:
        raise EvaluationError(f"checkpoint schema {field!r} vocab has no {token!r}")
    return int(vocab[token])


def _item_for_global_index(dataset: CardDataset, index: int) -> dict[str, Any]:
    start, end = dataset.detail_offsets[index : index + 2]
    card = dataset.cards[index]
    return {
        "index": index,
        "card_id": str(card["card_id"]),
        "card": card,
        "record": card,
        "details": dataset.details[start:end],
        "schema": dataset.schema,
    }


@torch.no_grad()
def _collect_structured_recovery(
    model: StaticPretrainingModel,
    dataset: CardDataset,
    indices: list[int],
    *,
    device: torch.device,
    batch_size: int,
) -> dict[str, dict[str, np.ndarray]]:
    """Run deterministic target-masked inference for structured fields."""

    subset = dataset.subset(indices)
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda rows: collate_cards(rows, dataset.schema),
    )
    schema = dataset.schema
    pokemon_id = _vocab_index(schema, "card_category", "POKEMON")
    stage_mask_id = _vocab_index(schema, "stage", MASK_TOKEN)
    type_mask_id = _vocab_index(schema, "pokemon_type", MASK_TOKEN)
    damage_mode_mask_id = _vocab_index(schema, "damage_mode", MASK_TOKEN)
    hp_stats = schema["normalization"]["printed_hp"]
    retreat_stats = schema["normalization"]["retreat"]
    damage_stats = schema["normalization"]["attack_base_damage"]
    tag_alias_indices = [
        int(index)
        for tag, index in schema.get("card_tag_vocab", {}).items()
        if str(tag) == "TERA" or str(tag).startswith("TERA_TYPE_")
    ]
    collected: dict[str, dict[str, list[np.ndarray]]] = {
        name: {"target": [], "prediction": []}
        for name in (
            "pokemon_stage",
            "pokemon_type",
            "printed_hp",
            "retreat",
            "rule_flags",
            "attack_cost",
            "damage_base",
            "damage_mode",
        )
    }

    def record(name: str, target: torch.Tensor, prediction: torch.Tensor) -> None:
        collected[name]["target"].append(target.detach().cpu().numpy())
        collected[name]["prediction"].append(prediction.detach().cpu().numpy())

    model.eval()
    for raw in loader:
        pokemon = raw["card_category_ids"] == pokemon_id

        masked = clone_batch(raw)
        masked["stage_ids"][pokemon] = stage_mask_id
        batch = move_batch(masked, device)
        output = model.encoder(batch, return_details=True)
        prediction = model.card_heads(output.card_summary)["stage"]
        record("pokemon_stage", raw["stage_ids"][pokemon], prediction[pokemon.to(device)])

        masked = clone_batch(raw)
        masked["pokemon_type_ids"][pokemon] = type_mask_id
        batch = move_batch(masked, device)
        output = model.encoder(batch, return_details=True)
        prediction = model.card_heads(output.card_summary)["pokemon_type"]
        record("pokemon_type", raw["pokemon_type_ids"][pokemon], prediction[pokemon.to(device)])

        for name, value_key, raw_key, applicability_key, stats in (
            ("printed_hp", "printed_hp", "printed_hp_raw", "printed_hp_mask", hp_stats),
            ("retreat", "retreat", "retreat_raw", "retreat_mask", retreat_stats),
        ):
            valid = raw[applicability_key] > 0
            masked = clone_batch(raw)
            masked[value_key][valid] = 0.0
            masked[applicability_key][valid] = 0.0
            batch = move_batch(masked, device)
            output = model.encoder(batch, return_details=True)
            normalized = model.card_heads(output.card_summary)[name][valid.to(device)]
            prediction = normalized * float(stats["std"]) + float(stats["mean"])
            record(name, raw[raw_key][valid], prediction)

        masked = clone_batch(raw)
        masked["rule_flag_multihot"].zero_()
        if tag_alias_indices:
            masked["card_tag_multihot"][:, tag_alias_indices] = 0.0
        batch = move_batch(masked, device)
        output = model.encoder(batch, return_details=True)
        record("rule_flags", raw["rule_flag_multihot"], model.card_heads(output.card_summary)["rule_flags"])

        attack = raw["attack_energy_mask"] > 0
        masked = clone_batch(raw)
        masked["attack_energy_counts"][attack] = 0.0
        batch = move_batch(masked, device)
        output = model.encoder(batch, return_details=True)
        detail_prediction = model.detail_heads(output.detail_tokens)
        record("attack_cost", raw["attack_energy_counts"][attack], detail_prediction["energy_counts"][attack.to(device)])

        attack = raw["attack_energy_mask"] > 0
        damage = raw["attack_damage_mask"] > 0
        masked = clone_batch(raw)
        masked["attack_base_damage"][attack] = 0.0
        masked["attack_damage_mask"][attack] = 0.0
        masked["attack_damage_mode"][attack] = damage_mode_mask_id
        batch = move_batch(masked, device)
        output = model.encoder(batch, return_details=True)
        detail_prediction = model.detail_heads(output.detail_tokens)
        normalized = detail_prediction["base_damage"][damage.to(device)]
        raw_damage = normalized * float(damage_stats["std"]) + float(damage_stats["mean"])
        record("damage_base", raw["attack_base_damage_raw"][damage], raw_damage)
        record("damage_mode", raw["attack_damage_mode"][attack], detail_prediction["damage_mode"][attack.to(device)])

    return {
        name: {
            key: np.concatenate(rows, axis=0) if rows else np.empty((0,), dtype=np.float64)
            for key, rows in values.items()
        }
        for name, values in collected.items()
    }


@torch.no_grad()
def _collect_online_ownership(
    model: StaticPretrainingModel,
    dataset: CardDataset,
    indices: list[int],
    *,
    device: torch.device,
    batch_size: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """Evaluate ownership without exposing the candidate detail to its owner."""

    allowed = set(indices)
    detail_rows = [
        row for row, detail in enumerate(dataset.details) if int(detail["card_index"]) in allowed
    ]
    if not detail_rows:
        return {"target": np.empty((0,), dtype=np.int64), "score": np.empty((0,), dtype=np.float64)}

    candidate_tokens: dict[int, torch.Tensor] = {}
    subset = dataset.subset(indices)
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda rows: collate_cards(rows, dataset.schema),
    )
    model.eval()
    for raw in loader:
        output = model.encoder(move_batch(raw, device), return_details=True)
        for row in range(raw["detail_mask"].shape[0]):
            for slot in torch.nonzero(raw["detail_mask"][row] > 0, as_tuple=False).flatten().tolist():
                global_index = int(raw["detail_global_indices"][row, slot])
                candidate_tokens[global_index] = output.pre_fusion_detail_tokens[row, slot].detach()

    owner_summaries: dict[int, torch.Tensor] = {}
    for start in range(0, len(detail_rows), batch_size):
        selected = detail_rows[start : start + batch_size]
        items = [_item_for_global_index(dataset, int(dataset.details[row]["card_index"])) for row in selected]
        raw = collate_cards(items, dataset.schema)
        for batch_row, detail_row in enumerate(selected):
            slot = int(dataset.details[detail_row]["local_detail_index"])
            raw["detail_mask"][batch_row, slot] = 0.0
            raw["detail_text_mask"][batch_row, slot] = 0.0
        output = model.encoder(move_batch(raw, device), return_details=True)
        for batch_row, detail_row in enumerate(selected):
            owner_summaries[detail_row] = output.card_summary[batch_row].detach()

    by_signature: dict[tuple[str, str], list[int]] = defaultdict(list)
    by_type: dict[str, list[int]] = defaultdict(list)
    for row in detail_rows:
        detail = dataset.details[row]
        detail_type = str(detail["detail_type"])
        subtype = str(detail.get("detail_subtype") or "")
        by_signature[(detail_type, subtype)].append(row)
        by_type[detail_type].append(row)
    rng = random.Random(seed)
    owner_batch: list[torch.Tensor] = []
    candidate_batch: list[torch.Tensor] = []
    labels: list[int] = []
    for row in detail_rows:
        owner = int(dataset.details[row]["card_index"])
        detail_type = str(dataset.details[row]["detail_type"])
        subtype = str(dataset.details[row].get("detail_subtype") or "")
        alternatives = [
            candidate
            for candidate in by_signature[(detail_type, subtype)]
            if int(dataset.details[candidate]["card_index"]) != owner
        ]
        if not alternatives:
            alternatives = [
                candidate
                for candidate in by_type[detail_type]
                if int(dataset.details[candidate]["card_index"]) != owner
            ]
        if not alternatives:
            continue
        owner_batch.extend((owner_summaries[row], owner_summaries[row]))
        candidate_batch.extend((candidate_tokens[row], candidate_tokens[rng.choice(alternatives)]))
        labels.extend((1, 0))
    if not labels:
        return {"target": np.empty((0,), dtype=np.int64), "score": np.empty((0,), dtype=np.float64)}
    logits = model.ownership_head(torch.stack(owner_batch), torch.stack(candidate_batch))
    return {
        "target": np.asarray(labels, dtype=np.int64),
        "score": logits.detach().cpu().numpy().astype(np.float64),
    }


def _classification_recovery_report(
    name: str,
    trained: dict[str, dict[str, np.ndarray]],
    untrained: dict[str, dict[str, np.ndarray]],
    train_labels: Sequence[int],
) -> dict[str, Any]:
    majority = Counter(int(value) for value in train_labels).most_common(1)[0][0]
    rows: dict[str, Any] = {}
    for split in ("validation", "test"):
        target = trained[split]["target"].astype(np.int64).tolist()
        trained_prediction = np.argmax(trained[split]["prediction"], axis=-1).astype(np.int64).tolist()
        untrained_prediction = np.argmax(untrained[split]["prediction"], axis=-1).astype(np.int64).tolist()
        rows[split] = {
            "support": len(target),
            "trained_balanced_accuracy": _balanced_accuracy(target, trained_prediction),
            "untrained_balanced_accuracy": _balanced_accuracy(target, untrained_prediction),
            "label_baseline_balanced_accuracy": _balanced_accuracy(target, [majority] * len(target)),
        }
    reference = max(rows["test"]["untrained_balanced_accuracy"], rows["test"]["label_baseline_balanced_accuracy"])
    passed = rows["test"]["trained_balanced_accuracy"] >= reference + 0.02
    return {
        "name": name,
        "task": "online_target_masked_classification",
        "leakage_control": "target input is replaced by the schema MASK token before encoder inference",
        "selection": "no probe fit; checkpoint head argmax; test evaluated once",
        **rows,
        "required_margin": 0.02,
        "gate": "pass" if passed else "fail",
    }


def _regression_recovery_report(
    name: str,
    trained: dict[str, dict[str, np.ndarray]],
    untrained: dict[str, dict[str, np.ndarray]],
    train_labels: Sequence[float],
) -> dict[str, Any]:
    median = float(np.median(np.asarray(train_labels, dtype=np.float64)))
    rows: dict[str, Any] = {}
    for split in ("validation", "test"):
        target = trained[split]["target"].astype(np.float64).reshape(-1)
        prediction = trained[split]["prediction"].astype(np.float64).reshape(-1)
        initial = untrained[split]["prediction"].astype(np.float64).reshape(-1)
        rows[split] = {
            "support": int(target.size),
            "trained_mae": float(np.mean(np.abs(prediction - target))),
            "untrained_mae": float(np.mean(np.abs(initial - target))),
            "label_baseline_mae": float(np.mean(np.abs(median - target))),
        }
    reference = min(rows["test"]["untrained_mae"], rows["test"]["label_baseline_mae"])
    passed = rows["test"]["trained_mae"] <= reference * 0.95
    return {
        "name": name,
        "task": "online_target_masked_regression",
        "leakage_control": "target value and its presence mask are cleared before encoder inference",
        "selection": "no probe fit; checkpoint regression head; test evaluated once",
        **rows,
        "required_relative_mae": 0.95,
        "gate": "pass" if passed else "fail",
    }


def _multioutput_recovery_report(
    name: str,
    trained: dict[str, dict[str, np.ndarray]],
    untrained: dict[str, dict[str, np.ndarray]],
    train_labels: np.ndarray,
) -> dict[str, Any]:
    active = np.any(train_labels != 0, axis=0)
    if not bool(active.any()):
        return {"name": name, "task": "online_target_masked_multioutput", "status": "unsupported", "gate": "fail"}
    baseline = np.median(train_labels[:, active], axis=0)
    rows: dict[str, Any] = {}
    for split in ("validation", "test"):
        target = trained[split]["target"][:, active].astype(np.float64)
        prediction = trained[split]["prediction"][:, active].astype(np.float64)
        initial = untrained[split]["prediction"][:, active].astype(np.float64)
        rows[split] = {
            "support": int(target.shape[0]),
            "active_output_count": int(active.sum()),
            "trained_mae": float(np.mean(np.abs(prediction - target))),
            "untrained_mae": float(np.mean(np.abs(initial - target))),
            "label_baseline_mae": float(np.mean(np.abs(target - baseline))),
        }
    reference = min(rows["test"]["untrained_mae"], rows["test"]["label_baseline_mae"])
    passed = rows["test"]["trained_mae"] <= reference * 0.95
    return {
        "name": name,
        "task": "online_target_masked_multioutput_regression",
        "leakage_control": "the complete attack energy-count vector is zeroed before encoder inference",
        "selection": "active outputs and label baseline derived from train only; test evaluated once",
        **rows,
        "required_relative_mae": 0.95,
        "gate": "pass" if passed else "fail",
    }


def _thresholded_online_binary(
    trained_validation: np.ndarray,
    trained_test: np.ndarray,
    untrained_validation: np.ndarray,
    untrained_test: np.ndarray,
    validation_target: np.ndarray,
    test_target: np.ndarray,
) -> dict[str, Any]:
    def select(scores: np.ndarray) -> tuple[float, float]:
        candidates = [
            (threshold, _binary_metrics(validation_target, scores >= threshold)["balanced_accuracy"])
            for threshold in _threshold_candidates(scores)
        ]
        return max(candidates, key=lambda row: (row[1], -abs(row[0] - 0.5)))

    trained_threshold, trained_val = select(trained_validation)
    untrained_threshold, untrained_val = select(untrained_validation)
    trained_test_metrics = _binary_metrics(test_target, trained_test >= trained_threshold)
    untrained_test_metrics = _binary_metrics(test_target, untrained_test >= untrained_threshold)
    return {
        "trained_validation_balanced_accuracy": trained_val,
        "untrained_validation_balanced_accuracy": untrained_val,
        "trained_threshold_selected_on_validation": trained_threshold,
        "untrained_threshold_selected_on_validation": untrained_threshold,
        "trained_test": trained_test_metrics,
        "untrained_test": untrained_test_metrics,
    }


def _rule_recovery_report(
    trained: dict[str, dict[str, np.ndarray]],
    untrained: dict[str, dict[str, np.ndarray]],
    train_labels: np.ndarray,
    vocabulary: Sequence[str],
) -> dict[str, Any]:
    per_label: dict[str, Any] = {}
    supported: list[dict[str, Any]] = []
    for column, label in enumerate(vocabulary):
        validation_target = trained["validation"]["target"][:, column].astype(np.int64)
        test_target = trained["test"]["target"][:, column].astype(np.int64)
        support = {
            "train_positive": int(train_labels[:, column].sum()),
            "validation_positive": int(validation_target.sum()),
            "test_positive": int(test_target.sum()),
        }
        row = {"support": support}
        if min(support.values()) < 5:
            row.update({"status": "low_confidence", "gate": "not_applicable"})
        else:
            metrics = _thresholded_online_binary(
                trained["validation"]["prediction"][:, column],
                trained["test"]["prediction"][:, column],
                untrained["validation"]["prediction"][:, column],
                untrained["test"]["prediction"][:, column],
                validation_target,
                test_target,
            )
            reference = max(0.5, metrics["untrained_test"]["balanced_accuracy"])
            passed = metrics["trained_test"]["balanced_accuracy"] >= reference + 0.02
            row.update({"status": "supported", **metrics, "required_margin": 0.02, "gate": "pass" if passed else "fail"})
            supported.append(row)
        per_label[str(label)] = row
    passed = bool(supported) and all(row["gate"] == "pass" for row in supported)
    return {
        "name": "rule_flags",
        "task": "online_target_masked_multilabel",
        "leakage_control": "rule multihot and deterministic TERA/TERA_TYPE card-tag aliases are cleared",
        "selection": "binary thresholds selected independently on validation; test evaluated once",
        "per_label": per_label,
        "gate": "pass" if passed else "fail",
    }


def _ownership_recovery_report(
    trained: dict[str, dict[str, np.ndarray]],
    untrained: dict[str, dict[str, np.ndarray]],
) -> dict[str, Any]:
    metrics = _thresholded_online_binary(
        trained["validation"]["score"],
        trained["test"]["score"],
        untrained["validation"]["score"],
        untrained["test"]["score"],
        trained["validation"]["target"],
        trained["test"]["target"],
    )
    reference = max(0.5, metrics["untrained_test"]["balanced_accuracy"])
    passed = metrics["trained_test"]["balanced_accuracy"] >= reference + 0.02
    return {
        "name": "ownership",
        "task": "online_leave_one_detail_out",
        "leakage_control": "owner summary excludes the candidate; candidate is pre-fusion; negative is another card with matching type and preferably subtype",
        "selection": "threshold selected on validation; test evaluated once",
        "validation_support": int(trained["validation"]["target"].size),
        "test_support": int(trained["test"]["target"].size),
        **metrics,
        "required_margin": 0.02,
        "gate": "pass" if passed else "fail",
    }


def _train_targets(dataset: CardDataset, train_indices: list[int]) -> dict[str, Any]:
    cards = dataset.cards
    details = dataset.details
    offsets = dataset.detail_offsets
    pokemon = [cards[index] for index in train_indices if cards[index].get("card_category") == "POKEMON"]
    train_details = [row for index in train_indices for row in details[offsets[index] : offsets[index + 1]]]
    attacks = [row for row in train_details if row.get("detail_type") == "ATTACK"]
    damage = [row for row in attacks if row.get("base_damage") is not None]
    schema = dataset.schema
    return {
        "pokemon_stage": [_vocab_index(schema, "stage", str(card["stage"])) for card in pokemon],
        "pokemon_type": [_vocab_index(schema, "pokemon_type", str(card["pokemon_type"])) for card in pokemon],
        "printed_hp": [float(cards[index]["printed_hp"]) for index in train_indices if cards[index].get("printed_hp") is not None],
        "retreat": [float(cards[index]["retreat"]) for index in train_indices if cards[index].get("retreat") is not None],
        "rule_flags": np.asarray(
            [[float(flag in (card.get("rule_flags") or [])) for flag in schema["rule_flag_vocab"]] for card in (cards[index] for index in train_indices)],
            dtype=np.float64,
        ),
        "attack_cost": np.asarray(
            [[float(row.get("energy_costs", {}).get(symbol, 0)) for symbol in schema["energy_types"]] for row in attacks],
            dtype=np.float64,
        ),
        "damage_base": [float(row["base_damage"]) for row in damage],
        "damage_mode": [_vocab_index(schema, "damage_mode", str(row["damage_mode"])) for row in attacks],
    }


def _evaluate_online_checkpoint(
    checkpoint_path: Path,
    cache_dir: Path,
    splits: dict[str, list[int]],
    *,
    seed: int,
    batch_size: int = 64,
) -> dict[str, Any]:
    checkpoint = _checkpoint_payload(checkpoint_path)
    dataset = CardDataset.from_cache(cache_dir)
    schema = checkpoint["schema"]
    if schema != dataset.schema:
        raise EvaluationError("checkpoint feature schema does not match the evaluated cache")
    model_config = checkpoint["config"].get("model")
    if not isinstance(model_config, dict):
        raise EvaluationError("checkpoint config has no model mapping")
    initialization_seed = int(checkpoint["config"].get("seed", seed))
    devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(initialization_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(initialization_seed)
        untrained = StaticPretrainingModel(schema, model_config)
    trained = StaticPretrainingModel(schema, model_config)
    trained.load_state_dict(checkpoint["model_state"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trained = trained.to(device).eval()
    untrained = untrained.to(device).eval()

    trained_outputs: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    untrained_outputs: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for split in ("validation", "test"):
        trained_outputs[split] = _collect_structured_recovery(
            trained, dataset, splits[split], device=device, batch_size=batch_size
        )
        untrained_outputs[split] = _collect_structured_recovery(
            untrained, dataset, splits[split], device=device, batch_size=batch_size
        )
    by_task_trained = {
        task: {split: trained_outputs[split][task] for split in ("validation", "test")}
        for task in trained_outputs["validation"]
    }
    by_task_untrained = {
        task: {split: untrained_outputs[split][task] for split in ("validation", "test")}
        for task in untrained_outputs["validation"]
    }
    structured_arrays = [
        value
        for model_outputs in (by_task_trained, by_task_untrained)
        for split_outputs in model_outputs.values()
        for task_outputs in split_outputs.values()
        for value in task_outputs.values()
    ]
    if not all(bool(np.isfinite(value).all()) for value in structured_arrays):
        gates = {
            name: {"name": name, "task": "online_target_masked", "gate": "fail", "reason": "non_finite_model_output"}
            for name in ONLINE_REQUIRED_GATES
        }
        return {
            "protocol": {
                "checkpoint_heads": True,
                "target_masked_online_inference": True,
                "initialization_baseline_seed": initialization_seed,
                "device": str(device),
                "online_outputs_finite": False,
                "test_evaluations": 1,
            },
            "gates": gates,
            "required_gates": list(ONLINE_REQUIRED_GATES),
            "failed_gates": list(ONLINE_REQUIRED_GATES),
            "passed": False,
            "relations": {"status": "diagnostic_only"},
        }
    targets = _train_targets(dataset, splits["train"])
    gates: dict[str, Any] = {
        "pokemon_stage": _classification_recovery_report(
            "pokemon_stage", by_task_trained["pokemon_stage"], by_task_untrained["pokemon_stage"], targets["pokemon_stage"]
        ),
        "pokemon_type": _classification_recovery_report(
            "pokemon_type", by_task_trained["pokemon_type"], by_task_untrained["pokemon_type"], targets["pokemon_type"]
        ),
        "printed_hp": _regression_recovery_report(
            "printed_hp", by_task_trained["printed_hp"], by_task_untrained["printed_hp"], targets["printed_hp"]
        ),
        "retreat": _regression_recovery_report(
            "retreat", by_task_trained["retreat"], by_task_untrained["retreat"], targets["retreat"]
        ),
        "rule_flags": _rule_recovery_report(
            by_task_trained["rule_flags"], by_task_untrained["rule_flags"], targets["rule_flags"], list(schema["rule_flag_vocab"])
        ),
        "attack_cost": _multioutput_recovery_report(
            "attack_cost", by_task_trained["attack_cost"], by_task_untrained["attack_cost"], targets["attack_cost"]
        ),
        "damage_base": _regression_recovery_report(
            "damage_base", by_task_trained["damage_base"], by_task_untrained["damage_base"], targets["damage_base"]
        ),
        "damage_mode": _classification_recovery_report(
            "damage_mode", by_task_trained["damage_mode"], by_task_untrained["damage_mode"], targets["damage_mode"]
        ),
    }
    joint_damage_control = (
        "base damage, damage presence, and damage-mode id are jointly cleared before encoder inference"
    )
    gates["damage_base"]["leakage_control"] = joint_damage_control
    gates["damage_mode"]["leakage_control"] = joint_damage_control
    ownership_trained: dict[str, dict[str, np.ndarray]] = {}
    ownership_untrained: dict[str, dict[str, np.ndarray]] = {}
    for split_number, split in enumerate(("validation", "test")):
        ownership_trained[split] = _collect_online_ownership(
            trained,
            dataset,
            splits[split],
            device=device,
            batch_size=batch_size,
            seed=seed + 5000 + split_number,
        )
        ownership_untrained[split] = _collect_online_ownership(
            untrained,
            dataset,
            splits[split],
            device=device,
            batch_size=batch_size,
            seed=seed + 5000 + split_number,
        )
    ownership_finite = all(
        bool(np.isfinite(value).all())
        for outputs in (ownership_trained, ownership_untrained)
        for split_outputs in outputs.values()
        for value in split_outputs.values()
    )
    if ownership_finite:
        gates["ownership"] = _ownership_recovery_report(ownership_trained, ownership_untrained)
    else:
        gates["ownership"] = {
            "name": "ownership",
            "task": "online_leave_one_detail_out",
            "gate": "fail",
            "reason": "non_finite_model_output",
        }
    failed = sorted(name for name, row in gates.items() if row.get("gate") != "pass")
    return {
        "protocol": {
            "checkpoint_heads": True,
            "target_masked_online_inference": True,
            "initialization_baseline_seed": initialization_seed,
            "device": str(device),
            "online_outputs_finite": ownership_finite,
            "test_evaluations": 1,
        },
        "gates": gates,
        "required_gates": list(ONLINE_REQUIRED_GATES),
        "failed_gates": failed,
        "passed": not failed and set(gates) == set(ONLINE_REQUIRED_GATES),
        "relations": {
            "status": "diagnostic_only",
            "reason": "Frozen pair probes cannot remove identity aliases. A future online relation gate must mask card_name/species/previous/evolves_from/evolves_to and use category/stage-matched hard negatives.",
        },
    }


def evaluate_card_embeddings(
    *,
    cache_dir: Path,
    artifact_dir: Path,
    split_manifest_path: Path,
    output_dir: Path | None = None,
    seed: int = 20260713,
    max_pairs_per_split: int = 4096,
    expected_checkpoint_stage: str | None = "split_selection_best",
) -> dict[str, Any]:
    """Evaluate one frozen v2 export under a pre-declared card-ID split."""

    cache_dir = Path(cache_dir)
    artifact_dir = Path(artifact_dir)
    split_manifest_path = Path(split_manifest_path)
    cards: list[dict[str, Any]] = _load_json(cache_dir / "cards.json")
    details: list[dict[str, Any]] = _load_json(cache_dir / "details.json")
    preprocess_manifest: dict[str, Any] = _load_json(cache_dir / "preprocess_manifest.json")
    artifact_manifest: dict[str, Any] = _load_json(artifact_dir / "artifact_manifest.json")
    if preprocess_manifest.get("schema_version") != SUPPORTED_CACHE_SCHEMA:
        raise EvaluationError(f"cache must declare schema_version {SUPPORTED_CACHE_SCHEMA}")
    if artifact_manifest.get("schema_version") != SUPPORTED_ARTIFACT_SCHEMA:
        raise EvaluationError(f"artifacts must declare schema_version {SUPPORTED_ARTIFACT_SCHEMA}")
    if artifact_manifest.get("source_schema_version") != SUPPORTED_CACHE_SCHEMA:
        raise EvaluationError("artifact source schema does not identify the static v2 cache")
    if artifact_manifest.get("model_version") != SUPPORTED_MODEL_VERSION:
        raise EvaluationError(f"artifact model_version must be {SUPPORTED_MODEL_VERSION}")
    embedding_dim = int(artifact_manifest.get("card_embedding_dim", -1))
    alignment = validate_artifact_directory(
        cache_dir,
        artifact_dir,
        expected_embedding_dim=embedding_dim,
    )
    split_manifest: dict[str, Any] = _load_json(split_manifest_path)
    splits = _validate_split_manifest(split_manifest, cards)
    checkpoint = _checkpoint_lineage(
        artifact_manifest,
        split_manifest_path,
        artifact_dir,
        expected_checkpoint_stage,
    )
    card_tensor = _load_tensor(artifact_dir / "card_embeddings.pt").detach().cpu()
    detail_tensor = _load_tensor(artifact_dir / "detail_embeddings.pt").detach().cpu()
    if card_tensor.requires_grad or detail_tensor.requires_grad:
        raise EvaluationError("exported embedding tensors must not require gradients")
    if not bool(torch.isfinite(card_tensor).all()) or not bool(torch.isfinite(detail_tensor).all()):
        raise EvaluationError("embedding input contains non-finite values")
    card_embeddings = card_tensor.to(torch.float64).numpy()
    detail_embeddings = detail_tensor.to(torch.float64).numpy()

    probes: dict[str, dict[str, Any]] = {}
    features, labels = _select_card_rows(
        cards,
        card_embeddings,
        splits,
        lambda row: str(row["card_category"]),
        lambda _row: True,
    )
    probes["card_category"] = _categorical_probe(
        "card_category", features, labels, diagnostic_only=True
    )
    for probe_name, field in (("pokemon_stage", "stage"), ("pokemon_type", "pokemon_type")):
        features, labels = _select_card_rows(
            cards,
            card_embeddings,
            splits,
            lambda row, field=field: str(row[field]),
            lambda row, field=field: row.get("card_category") == "POKEMON" and row.get(field) is not None,
        )
        probes[probe_name] = _categorical_probe(probe_name, features, labels)
    for probe_name, field in (("printed_hp", "printed_hp"), ("retreat", "retreat")):
        features, labels = _select_card_rows(
            cards,
            card_embeddings,
            splits,
            lambda row, field=field: float(row[field]),
            lambda row, field=field: row.get(field) is not None,
        )
        probes[probe_name] = _regression_probe(probe_name, features, labels)

    card_features, card_rule_sets = _select_card_rows(
        cards,
        card_embeddings,
        splits,
        lambda row: set(str(value) for value in (row.get("rule_flags") or [])),
        lambda _row: True,
    )
    rule_vocabulary = sorted({flag for card in cards for flag in (card.get("rule_flags") or [])})
    probes["rule_flags"] = _multilabel_probe(
        "rule_flags", card_features, card_rule_sets, rule_vocabulary
    )

    features, labels, _rows = _select_detail_rows(
        details,
        detail_embeddings,
        splits,
        lambda row: str(row["detail_type"]),
        lambda _row: True,
    )
    probes["detail_type"] = _categorical_probe(
        "detail_type", features, labels, diagnostic_only=True
    )
    attack_features, _attack_labels, attack_rows = _select_detail_rows(
        details,
        detail_embeddings,
        splits,
        lambda _row: 0,
        lambda row: row.get("detail_type") == "ATTACK",
    )
    attack_costs = {
        split: np.asarray(
            [
                [float(details[index].get("energy_costs", {}).get(symbol, 0)) for symbol in ENERGY_ORDER]
                for index in attack_rows[split]
            ],
            dtype=np.float64,
        )
        for split in splits
    }
    probes["attack_cost"] = _multioutput_regression_probe(
        "attack_cost", attack_features, attack_costs, ENERGY_ORDER
    )
    damage_features, damage_labels, _ = _select_detail_rows(
        details,
        detail_embeddings,
        splits,
        lambda row: float(row["base_damage"]),
        lambda row: row.get("detail_type") == "ATTACK" and row.get("base_damage") is not None,
    )
    probes["damage_base"] = _regression_probe("damage_base", damage_features, damage_labels)
    mode_features, mode_labels, _ = _select_detail_rows(
        details,
        detail_embeddings,
        splits,
        lambda row: str(row["damage_mode"]),
        lambda row: row.get("detail_type") == "ATTACK",
    )
    probes["damage_mode"] = _categorical_probe("damage_mode", mode_features, mode_labels)

    pair_builders = {
        "same_species": _same_species_examples,
        "direct_evolution": _direct_evolution_examples,
    }
    pair_limitations = {
        "same_species": [
            "Frozen pair probe; identity aliases cannot be re-masked after export.",
            "This does not replace the online masked same-species pretraining-head evaluation.",
        ],
        "direct_evolution": [
            "Frozen pair probe; evolution/name identity fields cannot be re-masked after export.",
            "This does not replace the online masked directed-relation head evaluation.",
        ],
    }
    for offset, (name, builder) in enumerate(pair_builders.items()):
        pair_features: dict[str, np.ndarray] = {}
        pair_labels: dict[str, list[int]] = {}
        for split_index, (split, indices) in enumerate(splits.items()):
            pair_features[split], pair_labels[split] = builder(
                cards,
                card_embeddings,
                indices,
                seed=seed + offset * 100 + split_index,
                max_pairs=max_pairs_per_split,
            )
        probes[name] = _binary_probe(
            name,
            pair_features,
            pair_labels,
            limitations=pair_limitations[name],
        )

    ownership_features: dict[str, np.ndarray] = {}
    ownership_labels: dict[str, list[int]] = {}
    for split_index, (split, indices) in enumerate(splits.items()):
        ownership_features[split], ownership_labels[split] = _ownership_examples(
            details,
            card_embeddings,
            detail_embeddings,
            indices,
            seed=seed + 1000 + split_index,
            max_pairs=max_pairs_per_split,
        )
    probes["ownership"] = _binary_probe(
        "ownership",
        ownership_features,
        ownership_labels,
        limitations=[
            "Frozen artifact pair probe: exported card summaries already saw their own details.",
            "Exported detail tokens are contextualized, not pre-fusion candidates.",
            "Same-type wrong-detail negatives are split-local, but true leave-one-detail-out requires online checkpoint evaluation.",
        ],
    )

    for name, probe in probes.items():
        if name not in FROZEN_DIAGNOSTICS:
            raise EvaluationError(f"frozen probe {name!r} has no diagnostic-only declaration")
        _mark_frozen_diagnostic(probe)

    missing = sorted(set(REQUIRED_PROBES) - set(probes))
    semantic_contract = _semantic_contract(cards, details, preprocess_manifest)
    checkpoint_path = Path(str(checkpoint["path"]))
    online_recovery = _evaluate_online_checkpoint(
        checkpoint_path,
        cache_dir,
        splits,
        seed=seed,
    )
    report: dict[str, Any] = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "success": False,
        "protocol": {
            "embedding_state": "frozen",
            "training_fit_scope": "train_only",
            "standardization_scope": "train_only",
            "validation_scope": "ridge_alpha_or_binary_threshold_selection_only",
            "test_scope": "single_final_evaluation_after_selection",
            "detail_split_rule": "owner_card_split",
            "pair_split_rule": "both_cards_in_same_split",
            "ridge_alphas": list(RIDGE_ALPHAS),
            "binary_fixed_ridge_alpha": FIXED_BINARY_RIDGE_ALPHA,
            "seed": int(seed),
            "max_positive_pairs_per_split": int(max_pairs_per_split),
            "frozen_probe_role": "diagnostic_only_input_retention",
            "acceptance_model_signal": "checkpoint_online_target_masked_recovery",
        },
        "transductive_schema": {
            "declared_by_split_manifest": bool(split_manifest.get("transductive_catalog_schema")),
            "note": split_manifest.get("transductive_note"),
            "scope": "known-catalog vocabularies and CardEncoder numeric normalization only",
            "probe_parameter_fit_uses_validation_or_test_examples": False,
            "probe_selection_uses_validation": True,
            "probe_selection_uses_test": False,
        },
        "lineage": {
            "cache_dir": str(cache_dir),
            "artifact_dir": str(artifact_dir),
            "artifact_schema_version": artifact_manifest.get("schema_version"),
            "split_manifest": str(split_manifest_path),
            "split_manifest_sha256": _sha256(split_manifest_path),
            "checkpoint": checkpoint,
        },
        "data_integrity": {
            **alignment,
            "split_partition_complete_and_disjoint": True,
            "finite_card_embeddings": True,
            "finite_detail_embeddings": True,
            "card_split_counts": {split: len(indices) for split, indices in splits.items()},
            "detail_split_counts": {
                split: sum(int(detail["card_index"]) in set(indices) for detail in details)
                for split, indices in splits.items()
            },
        },
        "semantic_contract": semantic_contract,
        "frozen_probes": probes,
        # Compatibility alias. Every row is explicitly diagnostic_only.
        "probes": probes,
        "online_recovery": online_recovery,
    }
    finite_before_acceptance = _all_finite(report)
    hard_checks = {
        "artifact_alignment": bool(
            alignment["offsets_match_source"]
            and alignment["type_ids_match_source"]
            and alignment["metadata_rows_match_source"]
            and alignment["manifest_hashes_valid"]
        ),
        "split_partition": True,
        "selection_checkpoint_lineage": bool(checkpoint["verified"]),
        "transductive_schema_declared": bool(split_manifest.get("transductive_catalog_schema")),
        "finite_inputs_and_metrics": finite_before_acceptance,
        "all_required_probes_present": not missing,
        "all_frozen_probes_diagnostic_only": all(
            probe.get("diagnostic_only") is True and probe.get("gate") == "diagnostic_only"
            for probe in probes.values()
        ),
        "semantic_catalog_contract": bool(semantic_contract["passed"]),
        "online_required_gates_present": set(online_recovery.get("gates", {})) == set(ONLINE_REQUIRED_GATES),
        "online_target_masked_recovery": bool(online_recovery.get("passed")),
    }
    passed = all(hard_checks.values())
    report["acceptance"] = {
        "passed": passed,
        "hard_checks": hard_checks,
        "failed_checks": [name for name, value in hard_checks.items() if not value],
        "failed_probe_gates": list(online_recovery.get("failed_gates", [])),
        "diagnostic_only_probes": sorted(FROZEN_DIAGNOSTICS),
        "low_confidence_probes": sorted(name for name, probe in probes.items() if probe["status"] == "low_confidence"),
        "unsupported_probes": sorted(name for name, probe in probes.items() if probe["status"] == "unsupported"),
        "gate_policy": (
            "Every frozen-artifact probe is diagnostic-only because the target or target-context was an encoder input. "
            "Acceptance requires the independent semantic catalog contract and checkpoint-online target-masked heads "
            "to outperform both a same-seed untrained model and a train-label baseline by the declared margin."
        ),
    }
    report["success"] = passed
    if not _all_finite(report):
        raise EvaluationError("evaluation report contains NaN or infinity")
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "evaluation.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    return report


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate frozen static CardEncoder v2 artifacts")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--artifacts-dir", "--artifact-dir", dest="artifact_dir", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--max-pairs-per-split", type=int, default=4096)
    parser.add_argument(
        "--expected-checkpoint-stage",
        default="split_selection_best",
        help="Use 'none' only to skip the stage-name equality check; a verifiable checkpoint remains required for online gates.",
    )
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    expected_stage = None if args.expected_checkpoint_stage.casefold() == "none" else args.expected_checkpoint_stage
    report = evaluate_card_embeddings(
        cache_dir=args.cache_dir,
        artifact_dir=args.artifact_dir,
        split_manifest_path=args.split_manifest,
        output_dir=args.output_dir,
        seed=args.seed,
        max_pairs_per_split=args.max_pairs_per_split,
        expected_checkpoint_stage=expected_stage,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
