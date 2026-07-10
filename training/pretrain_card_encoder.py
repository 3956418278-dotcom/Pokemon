from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data.card_dataset import CardDataset, collate_cards, split_indices
from data.card_preprocessing import DEFAULT_CACHE_DIR, write_card_cache
from models.card_encoder import CardEncoder
from models.card_pretrain_heads import (
    AttackOwnerHead,
    MaskedFieldHeads,
    RelationHead,
    binary_metrics,
    energy_separation_loss,
    info_nce_loss,
    masked_field_loss,
)


def load_config(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        import yaml

        return yaml.safe_load(text)
    except Exception:
        if path.suffix.lower() == ".json":
            return json.loads(text)
        return parse_simple_yaml(text)


def parse_simple_yaml(text: str) -> dict[str, Any]:
    def parse_value(raw: str) -> Any:
        value = raw.strip()
        if value.lower() in {"true", "false"}:
            return value.lower() == "true"
        if value.lower() in {"null", "none"}:
            return None
        try:
            return int(value)
        except ValueError:
            pass
        try:
            return float(value)
        except ValueError:
            return value.strip("\"'")

    config: dict[str, Any] = {}
    current: dict[str, Any] | None = None
    for line in text.splitlines():
        stripped = line.split("#", 1)[0].rstrip()
        if not stripped:
            continue
        if not line.startswith(" ") and stripped.endswith(":"):
            key = stripped[:-1]
            config[key] = {}
            current = config[key]
            continue
        if not line.startswith(" "):
            key, value = stripped.split(":", 1)
            config[key.strip()] = parse_value(value)
            current = None
            continue
        if current is not None and ":" in stripped:
            key, value = stripped.split(":", 1)
            current[key.strip()] = parse_value(value)
    return config


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def mask_batch(batch: dict[str, Any], mask_probability: float) -> dict[str, Any]:
    masked = dict(batch)
    if mask_probability <= 0:
        return masked
    single = batch["single_cats"].clone()
    mask = torch.rand(single.shape, device=single.device) < mask_probability
    single[mask] = 0
    numeric = batch["numeric"].clone()
    numeric_mask = batch["numeric_mask"].clone()
    nmask = torch.rand(numeric.shape, device=numeric.device) < mask_probability
    numeric[nmask] = 0.0
    numeric_mask[nmask] = 0.0
    text_hashes = batch["text_hashes"].clone()
    tmask = torch.rand(text_hashes.shape, device=text_hashes.device) < (mask_probability * 0.25)
    text_hashes[tmask] = 0
    masked["single_cats"] = single
    masked["numeric"] = numeric
    masked["numeric_mask"] = numeric_mask
    masked["text_hashes"] = text_hashes
    return masked


def build_relation_batch(
    dataset: CardDataset,
    relation_pairs: dict[str, list[tuple[int, int]]],
    batch_size: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], torch.Tensor, torch.Tensor, list[str]]:
    relation_names = ["evolves_to", "same_evolution_chain", "same_name"]
    left_indices = []
    right_indices = []
    labels = []
    relation_type = []
    label_names = []
    card_count = len(dataset.records)
    per_relation = max(1, batch_size // len(relation_names))
    positive_sets = {name: set(pairs) for name, pairs in relation_pairs.items()}
    for rel_id, name in enumerate(relation_names):
        positives = relation_pairs.get(name) or []
        for pair in random.sample(positives, min(len(positives), per_relation)):
            left_indices.append(pair[0])
            right_indices.append(pair[1])
            labels.append(1.0)
            relation_type.append(rel_id)
            label_names.append(name)
        needed = per_relation
        attempts = 0
        while needed > 0 and attempts < per_relation * 20:
            attempts += 1
            pair = (random.randrange(card_count), random.randrange(card_count))
            if pair[0] == pair[1] or pair in positive_sets.get(name, set()):
                continue
            left_indices.append(pair[0])
            right_indices.append(pair[1])
            labels.append(0.0)
            relation_type.append(rel_id)
            label_names.append(name)
            needed -= 1
    if not left_indices:
        left_indices = [0]
        right_indices = [0]
        labels = [0.0]
        relation_type = [0]
        label_names = ["evolves_to"]
    left_items = [{"index": index, "card_id": dataset.records[index]["card_id"], "record": dataset.records[index]} for index in left_indices]
    right_items = [{"index": index, "card_id": dataset.records[index]["card_id"], "record": dataset.records[index]} for index in right_indices]
    return (
        left_items,
        right_items,
        torch.tensor(relation_type, dtype=torch.long),
        torch.tensor(labels, dtype=torch.float32),
        label_names,
    )


def attack_feature_from_record(record: dict[str, Any], attack_index: int) -> list[float]:
    damages = record.get("attack_damage") or []
    costs = record.get("attack_energy_costs") or []
    texts = record.get("attack_texts") or []
    damage = float(damages[attack_index] or 0.0) if attack_index < len(damages) else 0.0
    cost = costs[attack_index] if attack_index < len(costs) else {}
    total_cost = float(sum(int(v) for v in cost.values()))
    energy_slots = ["COLORLESS", "GRASS", "FIRE", "WATER", "LIGHTNING", "PSYCHIC", "FIGHTING", "DARKNESS", "METAL", "DRAGON", "RAINBOW", "TEAM_ROCKET"]
    energy_counts = [float(cost.get(slot, 0) + cost.get(slot[:1], 0)) for slot in energy_slots]
    text_len = float(len(texts[attack_index])) if attack_index < len(texts) else 0.0
    return [damage / 300.0, total_cost / 6.0, text_len / 500.0, 1.0 if text_len > 0 else 0.0, *[value / 6.0 for value in energy_counts]]


def build_attack_owner_batch(
    dataset: CardDataset,
    batch_size: int,
) -> tuple[list[dict[str, Any]], torch.Tensor, torch.Tensor]:
    positives: list[tuple[int, int]] = []
    for index, record in enumerate(dataset.records):
        for attack_index, _name in enumerate(record.get("attack_names") or []):
            positives.append((index, attack_index))
    if not positives:
        item = {"index": 0, "card_id": dataset.records[0]["card_id"], "record": dataset.records[0]}
        return [item], torch.zeros(1, 16), torch.zeros(1)

    selected = random.sample(positives, min(len(positives), max(1, batch_size // 2)))
    items: list[dict[str, Any]] = []
    features: list[list[float]] = []
    labels: list[float] = []
    card_count = len(dataset.records)
    for card_index, attack_index in selected:
        record = dataset.records[card_index]
        feature = attack_feature_from_record(record, attack_index)
        items.append({"index": card_index, "card_id": record["card_id"], "record": record})
        features.append(feature)
        labels.append(1.0)
        negative_index = random.randrange(card_count)
        attempts = 0
        while negative_index == card_index and attempts < 20:
            negative_index = random.randrange(card_count)
            attempts += 1
        neg_record = dataset.records[negative_index]
        items.append({"index": negative_index, "card_id": neg_record["card_id"], "record": neg_record})
        features.append(feature)
        labels.append(0.0)
    return items, torch.tensor(features, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32)


def average_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    totals: dict[str, list[float]] = {}
    for row in rows:
        for key, value in row.items():
            if isinstance(value, (float, int)) and math.isfinite(float(value)):
                totals.setdefault(key, []).append(float(value))
    return {key: sum(values) / len(values) for key, values in totals.items() if values}


def run_epoch(
    dataset: CardDataset,
    loader: DataLoader,
    encoder: CardEncoder,
    mask_heads: MaskedFieldHeads,
    relation_head: RelationHead,
    attack_owner_head: AttackOwnerHead,
    optimizer: torch.optim.Optimizer | None,
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, float]:
    train = optimizer is not None
    encoder.train(train)
    mask_heads.train(train)
    relation_head.train(train)
    attack_owner_head.train(train)
    relation_pairs = dataset.relation_samples()
    metric_rows = []
    weights = config["loss_weights"]
    for batch in loader:
        batch = move_batch(batch, device)
        masked = mask_batch(batch, float(config["tasks"].get("mask_probability", 0.15)))
        embedding = encoder(masked)
        pred = mask_heads(embedding)
        mask_loss, mask_metrics = masked_field_loss(pred, batch)
        text_emb = encoder.text_embedding(batch)
        struct_emb = encoder.structure_embedding(batch)
        contrastive_loss, contrast_metrics = info_nce_loss(text_emb, struct_emb, float(config["tasks"].get("temperature", 0.07)))

        left_items, right_items, rel_type, labels, label_names = build_relation_batch(
            dataset, relation_pairs, int(config["training"].get("relation_batch_size", 96))
        )
        left_batch = move_batch(collate_cards(left_items, dataset.schema), device)
        right_batch = move_batch(collate_cards(right_items, dataset.schema), device)
        left = encoder(left_batch)
        right = encoder(right_batch)
        rel_type = rel_type.to(device)
        labels = labels.to(device)
        rel_logits = relation_head(left, right, rel_type)
        relation_loss = F.binary_cross_entropy_with_logits(rel_logits, labels)
        rel_metrics = binary_metrics(rel_logits.detach(), labels.detach(), "relation")

        attack_items, attack_features, attack_labels = build_attack_owner_batch(
            dataset,
            int(config["training"].get("attack_relation_batch_size", config["training"].get("relation_batch_size", 96))),
        )
        attack_batch = move_batch(collate_cards(attack_items, dataset.schema), device)
        attack_features = attack_features.to(device)
        attack_labels = attack_labels.to(device)
        attack_card_embeddings = encoder(attack_batch)
        attack_logits = attack_owner_head(attack_card_embeddings, attack_features)
        attack_loss = F.binary_cross_entropy_with_logits(attack_logits, attack_labels)
        attack_metrics = binary_metrics(attack_logits.detach(), attack_labels.detach(), "attack_owner")
        energy_loss = energy_separation_loss(
            embedding,
            batch["energy_type_target"],
            margin=float(config["tasks"].get("energy_separation_margin", 0.25)),
        )

        total = (
            float(weights.get("mask", 1.0)) * mask_loss
            + float(weights.get("contrastive", 1.0)) * contrastive_loss
            + float(weights.get("relation", 1.0)) * (relation_loss + attack_loss)
            + float(weights.get("energy", 0.25)) * energy_loss
        )
        if train:
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters())
                + list(mask_heads.parameters())
                + list(relation_head.parameters())
                + list(attack_owner_head.parameters()),
                1.0,
            )
            optimizer.step()
        metric_rows.append(
            {
                "total_loss": float(total.detach().cpu()),
                "mask_loss": float(mask_loss.detach().cpu()),
                "contrastive_loss": float(contrastive_loss.detach().cpu()),
                "relation_loss": float(relation_loss.detach().cpu()),
                "attack_owner_loss": float(attack_loss.detach().cpu()),
                "energy_separation_loss": float(energy_loss.detach().cpu()),
                **mask_metrics,
                **contrast_metrics,
                **rel_metrics,
                **attack_metrics,
            }
        )
    return average_metrics(metric_rows)


def save_checkpoint(
    path: Path,
    encoder: CardEncoder,
    mask_heads: MaskedFieldHeads,
    relation_head: RelationHead,
    attack_owner_head: AttackOwnerHead,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    config: dict[str, Any],
    schema: dict[str, Any],
    metrics: dict[str, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "encoder": encoder.state_dict(),
            "mask_heads": mask_heads.state_dict(),
            "relation_head": relation_head.state_dict(),
            "attack_owner_head": attack_owner_head.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "config": config,
            "schema": schema,
            "metrics": metrics,
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/card_pretrain.yaml"))
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--resume", type=Path, default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    set_seed(int(config["seed"]))

    cache_dir = Path(config["data"].get("cache_dir", DEFAULT_CACHE_DIR))
    if args.rebuild_cache:
        summary = write_card_cache(cache_dir)
        print("card preprocessing summary:", json.dumps(summary, indent=2, ensure_ascii=False))
    dataset = CardDataset.from_cache(cache_dir, rebuild=args.rebuild_cache)
    train_idx, val_idx = split_indices(
        len(dataset),
        float(config["data"].get("validation_ratio", 0.15)),
        int(config["seed"]),
        str(config["data"].get("split_mode", "card_id")),
    )
    print(f"split_mode={config['data'].get('split_mode', 'card_id')} train={len(train_idx)} val={len(val_idx)}")
    print("records:", len(dataset.records))

    train_ds = dataset.subset(train_idx)
    val_ds = dataset.subset(val_idx)
    collate = lambda items: collate_cards(items, dataset.schema)
    train_loader = DataLoader(train_ds, batch_size=int(config["training"]["batch_size"]), shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=int(config["training"]["batch_size"]), shuffle=False, collate_fn=collate)

    device_name = str(config["training"].get("device", "auto"))
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    encoder = CardEncoder(
        dataset.schema,
        embedding_dim=int(config["model"].get("embedding_dim", 128)),
        freeze_text_encoder=bool(config["model"].get("freeze_text_encoder", False)),
    ).to(device)
    mask_heads = MaskedFieldHeads(dataset.schema, int(config["model"].get("embedding_dim", 128))).to(device)
    relation_head = RelationHead(int(config["model"].get("embedding_dim", 128))).to(device)
    attack_owner_head = AttackOwnerHead(int(config["model"].get("embedding_dim", 128))).to(device)
    optimizer = torch.optim.AdamW(
        list(encoder.parameters())
        + list(mask_heads.parameters())
        + list(relation_head.parameters())
        + list(attack_owner_head.parameters()),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"].get("weight_decay", 0.01)),
    )

    start_epoch = 0
    if args.resume is not None and args.resume.exists():
        checkpoint = torch.load(args.resume, map_location=device)
        encoder.load_state_dict(checkpoint["encoder"])
        mask_heads.load_state_dict(checkpoint["mask_heads"])
        relation_head.load_state_dict(checkpoint["relation_head"])
        if "attack_owner_head" in checkpoint:
            attack_owner_head.load_state_dict(checkpoint["attack_owner_head"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1

    checkpoint_dir = Path(config["training"]["checkpoint_dir"])
    log_dir = Path(config["training"]["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    for epoch in range(start_epoch, int(config["training"]["epochs"])):
        train_metrics = run_epoch(dataset, train_loader, encoder, mask_heads, relation_head, attack_owner_head, optimizer, config, device)
        with torch.no_grad():
            val_metrics = run_epoch(dataset, val_loader, encoder, mask_heads, relation_head, attack_owner_head, None, config, device)
        row = {"epoch": epoch, "train": train_metrics, "validation": val_metrics}
        print(json.dumps(row, indent=2))
        (log_dir / "card_pretrain_metrics.jsonl").open("a", encoding="utf-8").write(json.dumps(row) + "\n")
        save_checkpoint(checkpoint_dir / "card_encoder_last.pt", encoder, mask_heads, relation_head, attack_owner_head, optimizer, epoch, config, dataset.schema, val_metrics)
        if val_metrics.get("total_loss", float("inf")) < best_val:
            best_val = val_metrics["total_loss"]
            save_checkpoint(checkpoint_dir / "card_encoder_best.pt", encoder, mask_heads, relation_head, attack_owner_head, optimizer, epoch, config, dataset.schema, val_metrics)


if __name__ == "__main__":
    main()
