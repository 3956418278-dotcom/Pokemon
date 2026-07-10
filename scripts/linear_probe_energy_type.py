#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


ENERGY_LABELS = {"C", "G", "R", "W", "L", "P", "F", "D", "M", "N", "Y", "A", "TEAM_ROCKET"}


def load_data(embeddings_path: Path, records_path: Path, mode: str) -> tuple[np.ndarray, np.ndarray, list[str], list[dict]]:
    embeddings = np.load(embeddings_path).astype(np.float32)
    records = json.loads(records_path.read_text(encoding="utf-8"))
    labels = []
    indices = []
    kept_records = []
    for idx, record in enumerate(records):
        card_type = record.get("card_type")
        if mode == "pokemon" and card_type != "POKEMON":
            continue
        if mode == "pokemon_energy" and card_type not in {"POKEMON", "BASIC_ENERGY", "SPECIAL_ENERGY"}:
            continue
        label = record.get("pokemon_type")
        if card_type in {"BASIC_ENERGY", "SPECIAL_ENERGY"} and record.get("provided_energy_types"):
            label = record["provided_energy_types"][0]
        if label not in ENERGY_LABELS:
            continue
        indices.append(idx)
        labels.append(str(label))
        kept_records.append(record)
    label_names = sorted(set(labels))
    label_to_index = {label: i for i, label in enumerate(label_names)}
    y = np.array([label_to_index[label] for label in labels], dtype=np.int64)
    return embeddings[np.array(indices)], y, label_names, kept_records


def stratified_split(y: np.ndarray, val_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_parts = []
    val_parts = []
    for cls in sorted(set(y.tolist())):
        cls_idx = np.where(y == cls)[0]
        rng.shuffle(cls_idx)
        val_count = max(1, int(round(len(cls_idx) * val_ratio))) if len(cls_idx) > 1 else 0
        val_parts.append(cls_idx[:val_count])
        train_parts.append(cls_idx[val_count:])
    train = np.concatenate(train_parts)
    val = np.concatenate(val_parts)
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


def train_probe(
    x: np.ndarray,
    y: np.ndarray,
    classes: int,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    epochs: int,
    lr: float,
    weight_decay: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    mean = x[train_idx].mean(axis=0, keepdims=True)
    std = x[train_idx].std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    x_norm = (x - mean) / std
    x_aug = np.concatenate([x_norm, np.ones((x_norm.shape[0], 1), dtype=np.float32)], axis=1)
    rng = np.random.default_rng(seed)
    w = rng.normal(0.0, 0.01, size=(x_aug.shape[1], classes)).astype(np.float32)

    y_train = y[train_idx]
    onehot = np.eye(classes, dtype=np.float32)[y_train]
    for _epoch in range(epochs):
        logits = x_aug[train_idx] @ w
        probs = softmax(logits)
        grad = x_aug[train_idx].T @ (probs - onehot) / len(train_idx)
        grad[:-1] += weight_decay * w[:-1]
        w -= lr * grad

    def accuracy(indices: np.ndarray) -> float:
        pred = (x_aug[indices] @ w).argmax(axis=1)
        return float((pred == y[indices]).mean())

    metrics = {
        "train_accuracy": accuracy(train_idx),
        "validation_accuracy": accuracy(val_idx),
    }
    return w, x_aug, metrics


def confusion(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]) -> list[list[int]]:
    mat = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        mat[int(t), int(p)] += 1
    return mat.tolist()


def run_mode(args: argparse.Namespace, mode: str) -> dict:
    x, y, labels, records = load_data(args.embeddings, args.records, mode)
    train_idx, val_idx = stratified_split(y, args.val_ratio, args.seed)
    counts = Counter(labels[int(v)] for v in y)
    majority = max(counts.values()) / len(y)
    w, x_aug, metrics = train_probe(
        x,
        y,
        len(labels),
        train_idx,
        val_idx,
        args.epochs,
        args.lr,
        args.weight_decay,
        args.seed,
    )
    val_pred = (x_aug[val_idx] @ w).argmax(axis=1)
    examples = []
    for local_idx, pred in zip(val_idx[:20], val_pred[:20]):
        record = records[int(local_idx)]
        examples.append(
            {
                "card_id": record["card_id"],
                "name": record["name"],
                "true": labels[int(y[local_idx])],
                "pred": labels[int(pred)],
            }
        )
    return {
        "mode": mode,
        "samples": int(len(y)),
        "train_samples": int(len(train_idx)),
        "validation_samples": int(len(val_idx)),
        "labels": labels,
        "class_counts": dict(sorted(counts.items())),
        "majority_baseline": majority,
        **metrics,
        "confusion_matrix": confusion(y[val_idx], val_pred, labels),
        "validation_examples": examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", type=Path, default=Path("outputs/card_pretrain/artifacts/card_embeddings.npy"))
    parser.add_argument("--records", type=Path, default=Path("outputs/card_pretrain/artifacts/card_data/card_records.json"))
    parser.add_argument("--epochs", type=int, default=1200)
    parser.add_argument("--lr", type=float, default=0.15)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260710)
    args = parser.parse_args()
    results = [run_mode(args, "pokemon"), run_mode(args, "pokemon_energy")]
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
