from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from .card_preprocessing import DEFAULT_CACHE_DIR, NULL_TOKEN, UNK_TOKEN, ENERGY_TYPES, load_or_create_records, stable_hash


SINGLE_CATEGORICAL_FIELDS = [
    "card_type",
    "subtype",
    "pokemon_type",
    "stage",
    "weakness_type",
    "resistance_type",
    "trainer_type",
]
MULTI_CATEGORICAL_FIELDS = ["rule_flags", "provided_energy_types"]
NUMERIC_FIELDS = [
    "hp",
    "retreat_cost",
    "weakness_value",
    "resistance_value",
    "attack_count",
    "ability_count",
    "max_attack_damage",
    "mean_attack_damage",
    "max_attack_energy_cost",
    "mean_attack_energy_cost",
    "text_length",
    *[f"energy_cost_{energy}" for energy in ENERGY_TYPES],
]


def card_text(record: dict[str, Any]) -> str:
    parts = [
        record.get("name") or "",
        record.get("card_type") or "",
        record.get("subtype") or "",
        " ".join(record.get("ability_texts") or []),
        " ".join(record.get("attack_names") or []),
        " ".join(record.get("attack_texts") or []),
        record.get("full_effect_text") or "",
    ]
    return "\n".join(part for part in parts if part)


def build_vocab(records: list[dict[str, Any]]) -> dict[str, Any]:
    vocab: dict[str, Any] = {"single": {}, "multi": {}, "energy_types": ENERGY_TYPES}
    for field in SINGLE_CATEGORICAL_FIELDS:
        values = {NULL_TOKEN, UNK_TOKEN}
        values.update(str(record.get(field)) for record in records if record.get(field) is not None)
        vocab["single"][field] = {value: index for index, value in enumerate(sorted(values))}
    for field in MULTI_CATEGORICAL_FIELDS + ["attack_energy_types"]:
        values = {UNK_TOKEN}
        for record in records:
            if field == "attack_energy_types":
                for cost in record.get("attack_energy_costs") or []:
                    values.update(str(key) for key in cost)
            else:
                values.update(str(value) for value in record.get(field) or [])
        vocab["multi"][field] = {value: index for index, value in enumerate(sorted(values))}
    energy_values = {UNK_TOKEN}
    for record in records:
        label = energy_type_label(record)
        if label is not None:
            energy_values.add(str(label))
    vocab["energy_type_target"] = {value: index for index, value in enumerate(sorted(energy_values))}
    return vocab


def energy_type_label(record: dict[str, Any]) -> str | None:
    if record.get("card_type") == "POKEMON" and record.get("pokemon_type") is not None:
        return str(record["pokemon_type"])
    if record.get("card_type") in {"BASIC_ENERGY", "SPECIAL_ENERGY"} and record.get("provided_energy_types"):
        return str(record["provided_energy_types"][0])
    return None


def numeric_values(record: dict[str, Any]) -> dict[str, float | None]:
    damages = [value for value in record.get("attack_damage") or [] if value is not None]
    total_costs = [sum(int(v) for v in cost.values()) for cost in record.get("attack_energy_costs") or []]
    energy_counts = Counter()
    for cost in record.get("attack_energy_costs") or []:
        for key, value in cost.items():
            energy_counts[str(key)] += int(value)
    text_len = len(card_text(record))
    return {
        "hp": record.get("hp"),
        "retreat_cost": record.get("retreat_cost"),
        "weakness_value": record.get("weakness_value"),
        "resistance_value": record.get("resistance_value"),
        "attack_count": len(record.get("attack_names") or []),
        "ability_count": len(record.get("ability_texts") or []),
        "max_attack_damage": max(damages) if damages else None,
        "mean_attack_damage": sum(damages) / len(damages) if damages else None,
        "max_attack_energy_cost": max(total_costs) if total_costs else None,
        "mean_attack_energy_cost": sum(total_costs) / len(total_costs) if total_costs else None,
        "text_length": text_len,
        **{f"energy_cost_{energy}": float(energy_counts.get(energy, 0)) for energy in ENERGY_TYPES},
    }


def build_normalization(records: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {}
    for field in NUMERIC_FIELDS:
        values = []
        for record in records:
            value = numeric_values(record).get(field)
            if value is not None:
                values.append(float(value))
        if not values:
            stats[field] = {"mean": 0.0, "std": 1.0}
            continue
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / max(1, len(values))
        stats[field] = {"mean": mean, "std": max(variance**0.5, 1.0)}
    return stats


def make_feature_schema(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "vocab": build_vocab(records),
        "normalization": build_normalization(records),
        "single_fields": SINGLE_CATEGORICAL_FIELDS,
        "multi_fields": MULTI_CATEGORICAL_FIELDS,
        "numeric_fields": NUMERIC_FIELDS,
        "text_hash_dim": 2048,
    }


class CardDataset(Dataset):
    def __init__(
        self,
        records: list[dict[str, Any]],
        card_id_to_index: dict[str, int],
        schema: dict[str, Any],
        indices: list[int] | None = None,
    ) -> None:
        self.records = records
        self.card_id_to_index = card_id_to_index
        self.schema = schema
        self.indices = indices or list(range(len(records)))

    @classmethod
    def from_cache(cls, cache_dir: Path = DEFAULT_CACHE_DIR, rebuild: bool = False) -> "CardDataset":
        records, mapping, _summary = load_or_create_records(cache_dir, rebuild=rebuild)
        schema_path = cache_dir / "card_feature_schema.json"
        if schema_path.exists() and not rebuild:
            schema = json.loads(schema_path.read_text(encoding="utf-8"))
            if "energy_type_target" not in schema.get("vocab", {}):
                schema = make_feature_schema(records)
                schema_path.write_text(json.dumps(schema, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        else:
            schema = make_feature_schema(records)
            schema_path.write_text(json.dumps(schema, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return cls(records, mapping, schema)

    def subset(self, indices: list[int]) -> "CardDataset":
        return CardDataset(self.records, self.card_id_to_index, self.schema, indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, Any]:
        index = self.indices[item]
        record = self.records[index]
        return {"index": index, "card_id": record["card_id"], "record": record}

    def relation_samples(self) -> dict[str, list[tuple[int, int]]]:
        name_to_indices: dict[str, list[int]] = defaultdict(list)
        evolves_pairs: list[tuple[int, int]] = []
        for index, record in enumerate(self.records):
            name_to_indices[record["name"]].append(index)
        name_first_index = {name: values[0] for name, values in name_to_indices.items()}
        for index, record in enumerate(self.records):
            parent = record.get("evolves_from")
            if parent and parent in name_first_index:
                evolves_pairs.append((name_first_index[parent], index))
        same_chain = set(evolves_pairs)
        for parent, child in evolves_pairs:
            for parent2, child2 in evolves_pairs:
                if child == parent2:
                    same_chain.add((parent, child2))
        same_name = []
        for values in name_to_indices.values():
            if len(values) > 1:
                for left in values:
                    for right in values:
                        if left != right:
                            same_name.append((left, right))
        return {
            "evolves_to": evolves_pairs,
            "same_evolution_chain": sorted(same_chain),
            "same_name": same_name,
        }


def encode_record(record: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
    single = []
    for field in schema["single_fields"]:
        value = record.get(field)
        token = str(value) if value is not None else NULL_TOKEN
        vocab = schema["vocab"]["single"][field]
        single.append(vocab.get(token, vocab[UNK_TOKEN]))

    multi_values = []
    multi_offsets = []
    for field in schema["multi_fields"]:
        vocab = schema["vocab"]["multi"][field]
        values = [vocab.get(str(value), vocab[UNK_TOKEN]) for value in record.get(field) or []]
        multi_offsets.append(len(multi_values))
        multi_values.extend(values or [vocab[UNK_TOKEN]])
    attack_vocab = schema["vocab"]["multi"]["attack_energy_types"]
    attack_energy_values = []
    for cost in record.get("attack_energy_costs") or []:
        attack_energy_values.extend([attack_vocab.get(str(key), attack_vocab[UNK_TOKEN])] * int(value) for key, value in cost.items())
    flat_attack_energy = []
    for value in attack_energy_values:
        if isinstance(value, list):
            flat_attack_energy.extend(value)
        else:
            flat_attack_energy.append(value)
    multi_offsets.append(len(multi_values))
    multi_values.extend(flat_attack_energy or [attack_vocab[UNK_TOKEN]])

    raw_numeric = numeric_values(record)
    numeric = []
    numeric_mask = []
    for field in schema["numeric_fields"]:
        value = raw_numeric.get(field)
        stats = schema["normalization"][field]
        if value is None:
            numeric.append(0.0)
            numeric_mask.append(0.0)
        else:
            numeric.append((float(value) - stats["mean"]) / stats["std"])
            numeric_mask.append(1.0)

    text = card_text(record)
    hash_dim = int(schema.get("text_hash_dim", 2048))
    text_hashes = [stable_hash(token.lower(), hash_dim) for token in re_tokenize(text)]
    return {
        "single": single,
        "multi_values": multi_values,
        "multi_offsets": multi_offsets,
        "numeric": numeric,
        "numeric_mask": numeric_mask,
        "text_hashes": text_hashes or [0],
        "text": text,
    }


def re_tokenize(text: str) -> list[str]:
    return [token for token in torch_text_split(text) if token]


def torch_text_split(text: str) -> list[str]:
    return [part for part in __import__("re").split(r"[^A-Za-z0-9{}']+", text) if part]


def collate_cards(items: list[dict[str, Any]], schema: dict[str, Any] | None = None) -> dict[str, torch.Tensor | list[str]]:
    if not items:
        raise ValueError("Cannot collate an empty batch")
    schema = schema or items[0].get("schema")
    if schema is None:
        raise ValueError("schema is required")
    encoded = [encode_record(item["record"], schema) for item in items]
    max_multi = max(len(row["multi_values"]) for row in encoded)
    max_offsets = max(len(row["multi_offsets"]) for row in encoded)
    max_text = max(len(row["text_hashes"]) for row in encoded)
    batch = {
        "card_index": torch.tensor([item["index"] for item in items], dtype=torch.long),
        "single_cats": torch.tensor([row["single"] for row in encoded], dtype=torch.long),
        "multi_values": torch.tensor([row["multi_values"] + [0] * (max_multi - len(row["multi_values"])) for row in encoded], dtype=torch.long),
        "multi_mask": torch.tensor([[1.0] * len(row["multi_values"]) + [0.0] * (max_multi - len(row["multi_values"])) for row in encoded], dtype=torch.float32),
        "multi_offsets": torch.tensor([row["multi_offsets"] + [0] * (max_offsets - len(row["multi_offsets"])) for row in encoded], dtype=torch.long),
        "numeric": torch.tensor([row["numeric"] for row in encoded], dtype=torch.float32),
        "numeric_mask": torch.tensor([row["numeric_mask"] for row in encoded], dtype=torch.float32),
        "text_hashes": torch.tensor([row["text_hashes"] + [0] * (max_text - len(row["text_hashes"])) for row in encoded], dtype=torch.long),
        "text_mask": torch.tensor([[1.0] * len(row["text_hashes"]) + [0.0] * (max_text - len(row["text_hashes"])) for row in encoded], dtype=torch.float32),
        "card_type_target": torch.tensor([schema["vocab"]["single"]["card_type"].get(str(item["record"].get("card_type")), 1) for item in items], dtype=torch.long),
        "pokemon_type_target": torch.tensor([schema["vocab"]["single"]["pokemon_type"].get(str(item["record"].get("pokemon_type")), 0) if item["record"].get("pokemon_type") is not None else -100 for item in items], dtype=torch.long),
        "stage_target": torch.tensor([schema["vocab"]["single"]["stage"].get(str(item["record"].get("stage")), 0) if item["record"].get("stage") is not None else -100 for item in items], dtype=torch.long),
        "trainer_type_target": torch.tensor([schema["vocab"]["single"]["trainer_type"].get(str(item["record"].get("trainer_type")), 0) if item["record"].get("trainer_type") is not None else -100 for item in items], dtype=torch.long),
        "energy_type_target": torch.tensor([schema["vocab"]["energy_type_target"].get(str(energy_type_label(item["record"])), 0) if energy_type_label(item["record"]) is not None else -100 for item in items], dtype=torch.long),
        "hp_target": torch.tensor([float(numeric_values(item["record"]).get("hp") or 0.0) for item in items], dtype=torch.float32),
        "retreat_target": torch.tensor([int(numeric_values(item["record"]).get("retreat_cost") or 0) for item in items], dtype=torch.long).clamp(0, 5),
        "damage_target": torch.tensor([float(numeric_values(item["record"]).get("max_attack_damage") or 0.0) for item in items], dtype=torch.float32),
        "energy_cost_target": torch.tensor([float(numeric_values(item["record"]).get("max_attack_energy_cost") or 0.0) for item in items], dtype=torch.float32),
        "card_ids": [item["card_id"] for item in items],
        "texts": [row["text"] for row in encoded],
    }
    return batch


def split_indices(count: int, val_ratio: float, seed: int, mode: str) -> tuple[list[int], list[int]]:
    rng = random.Random(seed)
    indices = list(range(count))
    rng.shuffle(indices)
    val_count = max(1, int(count * val_ratio))
    if mode not in {"sample", "card_id"}:
        raise ValueError("split mode must be 'sample' or 'card_id'")
    return indices[val_count:], indices[:val_count]
