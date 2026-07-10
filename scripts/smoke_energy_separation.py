from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.card_dataset import CardDataset, collate_cards, energy_type_label
from models.card_encoder import CardEncoder
from models.card_pretrain_heads import MaskedFieldHeads, energy_separation_loss, masked_field_loss
from training.pretrain_card_encoder import mask_batch


def linear_probe(x: np.ndarray, y: list[int], seed: int, epochs: int, lr: float) -> tuple[float, float, float]:
    labels = np.asarray(y, dtype=np.int64)
    classes = sorted(set(labels.tolist()))
    remap = {value: index for index, value in enumerate(classes)}
    yy = np.asarray([remap[int(value)] for value in labels], dtype=np.int64)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(yy))
    val_count = max(1, int(len(yy) * 0.25))
    val_idx = order[:val_count]
    train_idx = order[val_count:]

    mean = x[train_idx].mean(axis=0, keepdims=True)
    std = x[train_idx].std(axis=0, keepdims=True) + 1e-6
    z = (x - mean) / std
    w = np.zeros((z.shape[1], len(classes)), dtype=np.float64)
    b = np.zeros((len(classes),), dtype=np.float64)
    for _ in range(epochs):
        logits = z[train_idx] @ w + b
        logits -= logits.max(axis=1, keepdims=True)
        probs = np.exp(logits)
        probs /= probs.sum(axis=1, keepdims=True)
        probs[np.arange(len(train_idx)), yy[train_idx]] -= 1.0
        w -= lr * ((z[train_idx].T @ probs / len(train_idx)) + 1e-4 * w)
        b -= lr * probs.mean(axis=0)

    train_acc = float(((z[train_idx] @ w + b).argmax(axis=1) == yy[train_idx]).mean())
    val_acc = float(((z[val_idx] @ w + b).argmax(axis=1) == yy[val_idx]).mean())
    majority = float(np.bincount(yy[train_idx]).max() / len(train_idx))
    return train_acc, val_acc, majority


def embeddings_for_indices(encoder: CardEncoder, dataset: CardDataset, indices: list[int], batch_size: int) -> np.ndarray:
    rows = []
    encoder.eval()
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            items = [{"index": i, "card_id": dataset.records[i]["card_id"], "record": dataset.records[i]} for i in batch_indices]
            rows.append(encoder(collate_cards(items, dataset.schema)).cpu().numpy())
    return np.concatenate(rows, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, default=Path("outputs/card_pretrain/artifacts/card_data"))
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/card_pretrain/checkpoints/card_encoder_best.pt"))
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--only-energy-cards", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset = CardDataset.from_cache(args.cache_dir, rebuild=False)
    encoder = CardEncoder(dataset.schema, embedding_dim=128)
    if args.checkpoint.exists():
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        missing, unexpected = encoder.load_state_dict(checkpoint["encoder"], strict=False)
        print("loaded_checkpoint", str(args.checkpoint), "missing", len(missing), "unexpected", len(unexpected))

    labels = []
    labeled_indices = []
    for index, record in enumerate(dataset.records):
        if args.only_energy_cards and record.get("card_type") not in {"BASIC_ENERGY", "SPECIAL_ENERGY"}:
            continue
        label = energy_type_label(record)
        if label is not None:
            labeled_indices.append(index)
            labels.append(dataset.schema["vocab"]["energy_type_target"].get(str(label), 0))
    counts = {
        token: labels.count(index)
        for token, index in dataset.schema["vocab"]["energy_type_target"].items()
        if labels.count(index)
    }
    print("records", len(dataset.records))
    print("energy_labeled_cards", len(labeled_indices))
    print("energy_label_counts", counts)

    before = embeddings_for_indices(encoder, dataset, labeled_indices, 128)
    print("probe_before_train_val_majority", linear_probe(before, labels, args.seed, 700, 0.15))

    heads = MaskedFieldHeads(dataset.schema, 128)
    optimizer = torch.optim.AdamW(list(encoder.parameters()) + list(heads.parameters()), lr=1e-3, weight_decay=0.01)
    non_labeled = [i for i in range(len(dataset.records)) if i not in set(labeled_indices)]
    subset_indices = labeled_indices + random.sample(non_labeled, min(256, len(non_labeled)))
    random.shuffle(subset_indices)
    subset = dataset.subset(subset_indices)
    loader = DataLoader(subset, batch_size=96, shuffle=True, collate_fn=lambda items: collate_cards(items, dataset.schema))
    encoder.train()
    heads.train()
    for step, batch in enumerate(loader):
        if step >= args.steps:
            break
        embedding = encoder(mask_batch(batch, 0.15))
        pred = heads(embedding)
        mask_loss, metrics = masked_field_loss(pred, batch)
        sep_loss = energy_separation_loss(embedding, batch["energy_type_target"], margin=0.25)
        loss = mask_loss + 0.5 * sep_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(heads.parameters()), 1.0)
        optimizer.step()
        print(
            "step",
            step,
            "loss",
            round(float(loss.detach()), 4),
            "energy_sep",
            round(float(sep_loss.detach()), 4),
            "energy_acc",
            round(float(metrics.get("mask_energy_type_acc", -1.0)), 4),
        )

    after = embeddings_for_indices(encoder, dataset, labeled_indices, 128)
    print("probe_after_train_val_majority", linear_probe(after, labels, args.seed, 700, 0.15))


if __name__ == "__main__":
    main()
