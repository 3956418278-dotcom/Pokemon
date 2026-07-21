from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
import math
import random
from pathlib import Path
from typing import Any

import torch
import yaml

from decision_agent_v1.models.value_head import ValueHead


def _move_batch(batch: dict[str, Any], device: str) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def classification_metrics(predictions: list[int], targets: list[int], classes: int = 3) -> dict[str, Any]:
    confusion = [[0 for _ in range(classes)] for _ in range(classes)]
    for target, prediction in zip(targets, predictions):
        confusion[target][prediction] += 1
    f1_values = []
    for label in range(classes):
        true_positive = confusion[label][label]
        false_positive = sum(confusion[row][label] for row in range(classes) if row != label)
        false_negative = sum(confusion[label][column] for column in range(classes) if column != label)
        precision = true_positive / max(true_positive + false_positive, 1)
        recall = true_positive / max(true_positive + false_negative, 1)
        f1_values.append(2 * precision * recall / max(precision + recall, 1e-12))
    return {
        "win_draw_loss_accuracy": sum(p == t for p, t in zip(predictions, targets)) / max(len(targets), 1),
        "macro_f1": sum(f1_values) / classes,
        "confusion_matrix": confusion,
    }


@torch.no_grad()
def batch_metrics(model: Any, outputs: dict[str, torch.Tensor], batch: dict[str, Any]) -> dict[str, Any]:
    predictions = outputs["value_logits"].argmax(dim=-1).tolist()
    targets = batch["value_target"].tolist()
    values = ValueHead.scalar(outputs["value_logits"]).tolist()
    metrics = classification_metrics(predictions, targets)
    exact = 0
    set_exact = 0
    equivalence_exact = 0
    first = 0
    single_correct = 0
    single_count = 0
    supervised = 0
    option_count_correct = 0
    by_mode = Counter()
    by_context = Counter()
    by_outcome: dict[str, list[float]] = defaultdict(list)
    by_turn: dict[str, list[float]] = defaultdict(list)
    by_agent: dict[str, list[float]] = defaultdict(list)
    for row, target in enumerate(batch["target_sequences"]):
        if bool(batch["policy_sample_mask"][row]) and target:
            supervised += 1
            decoded = model.multiselect_decoder.decode(
                outputs["policy_logits"][row : row + 1],
                outputs["option_embeddings"][row : row + 1],
                batch["option_mask"][row : row + 1],
                len(target),
            )[0]
            first += int(decoded[:1] == list(target[:1]))
            exact += int(decoded == list(target))
            set_exact += int(sorted(decoded) == sorted(target))
            decoded_groups = [
                int(batch["option_equivalence_group"][row, index]) for index in decoded
            ]
            target_groups = list(batch["target_equivalence_groups"][row])
            mode_name = batch["selection_modes"][row].value
            if mode_name == "UNORDERED_UNIQUE_SUBSET":
                equivalence_exact += int(sorted(decoded_groups) == sorted(target_groups))
            else:
                equivalence_exact += int(decoded_groups == target_groups)
            if len(target) == 1:
                single_count += 1
                single_correct += int(decoded == list(target))
            option_count_correct += int(len(decoded) == len(target))
            by_mode[batch["selection_modes"][row].value] += 1
            by_context[str(batch["metadata"][row]["select_context"])] += 1
        outcome_name = ("LOSS", "DRAW", "WIN")[targets[row]]
        by_outcome[outcome_name].append(values[row])
        turn = int(batch["metadata"][row]["turn"])
        bucket = f"{turn // 5 * 5}-{turn // 5 * 5 + 4}"
        by_turn[bucket].append(values[row])
        by_agent[str(batch["metadata"][row]["agent_index"])].append(values[row])
    metrics.update(
        {
            "first_choice_accuracy": first / max(supervised, 1),
            "single_select_accuracy": single_correct / max(single_count, 1),
            "sequence_exact_match": exact / max(supervised, 1),
            "set_exact_match": set_exact / max(supervised, 1),
            "equivalence_aware_accuracy": equivalence_exact / max(supervised, 1),
            "option_count_accuracy": option_count_correct / max(supervised, 1),
            "samples_by_selection_mode": dict(by_mode),
            "samples_by_select_context": dict(by_context),
            "unknown_semantics_count": int((~batch["policy_sample_mask"]).sum()),
            "value_mean": sum(values) / max(len(values), 1),
            "value_std": float(torch.tensor(values).std(unbiased=False)) if values else 0.0,
            "value_by_terminal_outcome": {key: sum(rows) / len(rows) for key, rows in by_outcome.items()},
            "value_by_turn_bucket": {key: sum(rows) / len(rows) for key, rows in by_turn.items()},
            "value_by_agent": {key: sum(rows) / len(rows) for key, rows in by_agent.items()},
        }
    )
    return metrics


def expected_calibration_error(
    probabilities: list[list[float]], targets: list[int], bins: int = 10
) -> float:
    if not probabilities:
        return 0.0
    total = len(targets)
    result = 0.0
    for lower_index in range(bins):
        lower = lower_index / bins
        upper = (lower_index + 1) / bins
        members = []
        for index, row in enumerate(probabilities):
            confidence = max(row)
            if lower <= confidence < upper or (lower_index == bins - 1 and confidence == 1.0):
                members.append(index)
        if not members:
            continue
        confidence = sum(max(probabilities[index]) for index in members) / len(members)
        accuracy = sum(
            max(range(3), key=lambda label: probabilities[index][label]) == targets[index]
            for index in members
        ) / len(members)
        result += len(members) / total * abs(accuracy - confidence)
    return result


def value_metrics_from_probabilities(
    probabilities: list[list[float]], targets: list[int], turns: list[int]
) -> dict[str, Any]:
    predictions = [max(range(3), key=lambda label: row[label]) for row in probabilities]
    base = classification_metrics(predictions, targets)
    log_loss = -sum(math.log(max(row[target], 1e-12)) for row, target in zip(probabilities, targets)) / max(len(targets), 1)
    brier = sum(
        sum((row[label] - float(label == target)) ** 2 for label in range(3))
        for row, target in zip(probabilities, targets)
    ) / max(len(targets), 1)
    values = [row[2] - row[0] for row in probabilities]
    by_outcome: dict[str, list[float]] = defaultdict(list)
    by_turn: dict[str, list[int]] = defaultdict(list)
    for index, (target, turn) in enumerate(zip(targets, turns)):
        by_outcome[("LOSS", "DRAW", "WIN")[target]].append(values[index])
        by_turn[f"{turn // 5 * 5}-{turn // 5 * 5 + 4}"].append(index)
    turn_metrics = {}
    for bucket, indices in sorted(by_turn.items()):
        bucket_probs = [probabilities[index] for index in indices]
        bucket_targets = [targets[index] for index in indices]
        bucket_predictions = [predictions[index] for index in indices]
        turn_metrics[bucket] = {
            "count": len(indices),
            "accuracy": sum(a == b for a, b in zip(bucket_predictions, bucket_targets)) / len(indices),
            "expected_calibration_error": expected_calibration_error(bucket_probs, bucket_targets),
        }
    return {
        **base,
        "value_log_loss": log_loss,
        "brier_score": brier,
        "expected_calibration_error": expected_calibration_error(probabilities, targets),
        "value_mean": sum(values) / max(len(values), 1),
        "value_std": float(torch.tensor(values).std(unbiased=False)) if values else 0.0,
        "value_by_terminal_outcome": {
            key: {
                "count": len(rows),
                "mean": sum(rows) / len(rows),
                "std": float(torch.tensor(rows).std(unbiased=False)),
            }
            for key, rows in by_outcome.items()
        },
        "turn_bucket_metrics": turn_metrics,
    }


def fit_training_baselines(
    corpus: Any,
) -> tuple[list[int], dict[tuple[int, int], list[int]]]:
    """Fit all train-only baselines in one cache pass."""

    class_counts = [0, 0, 0]
    counts: dict[tuple[int, int], Counter[int]] = defaultdict(Counter)
    for batch in corpus.iter_batches("train", 4096):
        for label in batch["value_target"].tolist():
            class_counts[int(label)] += 1
        for row, metadata in enumerate(batch["metadata"]):
            if not metadata["policy_supervision"]:
                continue
            key = (int(metadata["select_type"]), int(metadata["select_context"]))
            for index in metadata["target_sequence"]:
                counts[key][int(batch["option_type"][row, index])] += 1
    frequency = {
        key: [option_type for option_type, _ in counter.most_common()]
        for key, counter in counts.items()
    }
    return class_counts, frequency


def fit_policy_frequency_baseline(corpus: Any) -> dict[tuple[int, int], list[int]]:
    return fit_training_baselines(corpus)[1]


def _prediction_count(metadata: dict[str, Any]) -> int:
    if metadata["selection_mode"] == "SINGLE":
        return 1
    return max(0, int(metadata["min_count"]))


def _baseline_positions(
    name: str,
    batch: dict[str, Any],
    row: int,
    count: int,
    frequency: dict[tuple[int, int], list[int]],
    rng: random.Random,
) -> list[int]:
    valid = torch.nonzero(batch["option_mask"][row], as_tuple=False).flatten().tolist()
    count = min(count, len(valid))
    if name == "random_legal":
        return rng.sample(valid, count)
    if name == "deterministic_legal":
        return sorted(valid, key=lambda index: int(batch["original_option_index"][row, index]))[:count]
    if name == "frequency_context":
        metadata = batch["metadata"][row]
        ranking = frequency.get((int(metadata["select_type"]), int(metadata["select_context"])), [])
        rank = {option_type: index for index, option_type in enumerate(ranking)}
        return sorted(
            valid,
            key=lambda index: (
                rank.get(int(batch["option_type"][row, index]), len(rank)),
                int(batch["original_option_index"][row, index]),
            ),
        )[:count]
    raise ValueError(name)


def evaluate_model(
    model: Any,
    corpus: Any,
    split: str,
    batch_size: int,
    device: str,
    *,
    frequency_baseline: dict[tuple[int, int], list[int]] | None = None,
    class_prior: list[float] | None = None,
    seed: int = 20260718,
) -> dict[str, Any]:
    model.eval()
    frequency = frequency_baseline or {}
    rng = random.Random(seed)
    probabilities: list[list[float]] = []
    targets: list[int] = []
    turns: list[int] = []
    policy_totals = {
        name: Counter()
        for name in ("trained_model", "random_legal", "frequency_context", "deterministic_legal")
    }
    context_totals: dict[str, Counter[str]] = defaultdict(Counter)
    option_count_totals: dict[str, Counter[str]] = defaultdict(Counter)
    policy_loss_sum = 0.0
    value_loss_sum = 0.0
    decision_count = 0
    from decision_agent_v1.training.losses import joint_policy_value_loss

    with torch.no_grad():
        for cpu_batch in corpus.iter_batches(split, batch_size):
            batch = _move_batch(cpu_batch, device)
            outputs = model(batch)
            losses = joint_policy_value_loss(model.multiselect_decoder, outputs, batch)
            policy_loss_sum += float(losses["policy_loss"]) * len(batch["metadata"])
            value_loss_sum += float(losses["value_loss"]) * len(batch["metadata"])
            probs = outputs["value_logits"].softmax(dim=-1).cpu().tolist()
            probabilities.extend(probs)
            targets.extend(batch["value_target"].cpu().tolist())
            turns.extend(int(row["turn"]) for row in batch["metadata"])
            decision_count += len(batch["metadata"])
            prediction_counts = [
                _prediction_count(metadata) if metadata["policy_supervision"] else 0
                for metadata in batch["metadata"]
            ]
            max_prediction_count = max(prediction_counts, default=0)
            model_sequences = (
                model.multiselect_decoder.decode(
                    outputs["policy_logits"],
                    outputs["option_embeddings"],
                    batch["option_mask"],
                    max_prediction_count,
                )
                if max_prediction_count
                else [[] for _ in batch["metadata"]]
            )
            for row, metadata in enumerate(batch["metadata"]):
                if not metadata["policy_supervision"]:
                    continue
                count = _prediction_count(metadata)
                predictions = {
                    "trained_model": model_sequences[row][:count]
                }
                for name in ("random_legal", "frequency_context", "deterministic_legal"):
                    predictions[name] = _baseline_positions(
                        name, cpu_batch, row, count, frequency, rng
                    )
                target = list(metadata["target_sequence"])
                target_groups = list(metadata["target_equivalence_groups"])
                for name, prediction in predictions.items():
                    prediction_groups = [
                        int(cpu_batch["option_equivalence_group"][row, index]) for index in prediction
                    ]
                    unordered = metadata["selection_mode"] == "UNORDERED_UNIQUE_SUBSET"
                    equivalent = (
                        sorted(prediction_groups) == sorted(target_groups)
                        if unordered
                        else prediction_groups == target_groups
                    )
                    totals = policy_totals[name]
                    totals["count"] += 1
                    totals["first_correct"] += int(prediction[:1] == target[:1])
                    totals["sequence_exact"] += int(prediction == target)
                    totals["set_exact"] += int(sorted(prediction) == sorted(target))
                    totals["equivalence_exact"] += int(equivalent)
                    totals["count_correct"] += int(len(prediction) == len(target))
                    if len(target) == 1:
                        totals["single_count"] += 1
                        totals["single_correct"] += int(equivalent)
                trained_correct = int(
                    (
                        sorted([int(cpu_batch["option_equivalence_group"][row, index]) for index in predictions["trained_model"]])
                        == sorted(target_groups)
                    )
                    if metadata["selection_mode"] == "UNORDERED_UNIQUE_SUBSET"
                    else [int(cpu_batch["option_equivalence_group"][row, index]) for index in predictions["trained_model"]]
                    == target_groups
                )
                context_totals[str(metadata["select_context"])]["count"] += 1
                context_totals[str(metadata["select_context"])]["correct"] += trained_correct
                option_count_totals[str(int(cpu_batch["option_mask"][row].sum()))]["count"] += 1
                option_count_totals[str(int(cpu_batch["option_mask"][row].sum()))]["correct"] += trained_correct

    def finalize(counter: Counter[str]) -> dict[str, float | int]:
        count = counter["count"]
        return {
            "sample_count": count,
            "equivalence_aware_accuracy": counter["equivalence_exact"] / max(count, 1),
            "single_select_accuracy": counter["single_correct"] / max(counter["single_count"], 1),
            "first_choice_accuracy": counter["first_correct"] / max(count, 1),
            "sequence_exact_match": counter["sequence_exact"] / max(count, 1),
            "set_exact_match": counter["set_exact"] / max(count, 1),
            "option_count_accuracy": counter["count_correct"] / max(count, 1),
        }

    value = value_metrics_from_probabilities(probabilities, targets, turns)
    prior = class_prior or [1 / 3] * 3
    prior_probs = [list(prior) for _ in targets]
    result = {
        "split": split,
        "decision_count": decision_count,
        "policy_loss": policy_loss_sum / max(decision_count, 1),
        "value_loss": value_loss_sum / max(decision_count, 1),
        "policy": {name: finalize(counter) for name, counter in policy_totals.items()},
        "value": value,
        "value_class_prior_baseline": value_metrics_from_probabilities(prior_probs, targets, turns),
        "policy_by_select_context": {
            key: {"count": rows["count"], "equivalence_aware_accuracy": rows["correct"] / rows["count"]}
            for key, rows in sorted(context_totals.items(), key=lambda item: int(item[0]))
        },
        "policy_by_option_count": {
            key: {"count": rows["count"], "equivalence_aware_accuracy": rows["correct"] / rows["count"]}
            for key, rows in sorted(option_count_totals.items(), key=lambda item: int(item[0]))
        },
    }
    return result


def main() -> None:
    from decision_agent_v1.data.cache import CachedDecisionCorpus, cache_identity
    from decision_agent_v1.models.policy_value_model import PolicyValueModel

    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=root / "decision_agent_v1/configs/policy_value_v1.yaml")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    corpus = CachedDecisionCorpus(args.cache_dir, cache_identity(root, config))
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    for key, manifest_key in (
        ("data_schema_hash", "schema_hash"),
        ("action_contract_hash", "action_contract_hash"),
        ("card_vocabulary_hash", "card_vocabulary_hash"),
    ):
        if checkpoint.get(key) != corpus.manifest[manifest_key]:
            raise RuntimeError(f"checkpoint/cache mismatch: {key}")
    model = PolicyValueModel(**checkpoint["model_config"]).to(args.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    counts, frequency = fit_training_baselines(corpus)
    prior = [count / max(sum(counts), 1) for count in counts]
    metrics = evaluate_model(
        model,
        corpus,
        args.split,
        int(config["training"]["batch_size"]),
        args.device,
        frequency_baseline=frequency,
        class_prior=prior,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
