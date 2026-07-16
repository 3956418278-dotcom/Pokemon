from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from .card_preprocessing import DEFAULT_CACHE_DIR, NULL_TOKEN, UNK_TOKEN, MASK_TOKEN, ENERGY_TYPES, load_or_create_corpus, stable_hash


SINGLE_CATEGORICAL_FIELDS = [
    "category",
    "card_type",
    "subtype",
    "name",
    "pokemon_type",
    "stage",
    "hp_applicability",
    "evolves_from",
    "weakness_type",
    "resistance_type",
    "trainer_type",
]
MULTI_CATEGORICAL_FIELDS = ["rule_flags", "card_tags", "provided_energy_types", "evolves_to"]
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
    # Rule text belongs to independently indexed details. Duplicating it here
    # leaks ownership and makes leave-one-detail-out supervision meaningless.
    return ""


def build_vocab(records: list[dict[str, Any]]) -> dict[str, Any]:
    vocab: dict[str, Any] = {"single": {}, "multi": {}, "energy_types": ENERGY_TYPES}
    for field in SINGLE_CATEGORICAL_FIELDS:
        values = {NULL_TOKEN, UNK_TOKEN, MASK_TOKEN}
        values.update(str(record.get(field)) for record in records if record.get(field) is not None)
        vocab["single"][field] = {value: index for index, value in enumerate(sorted(values))}
    for field in MULTI_CATEGORICAL_FIELDS + ["attack_energy_types"]:
        values = {UNK_TOKEN, MASK_TOKEN}
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
    vocab["detail_type"] = {value: index for index, value in enumerate([NULL_TOKEN, UNK_TOKEN, MASK_TOKEN, "ATTACK", "ABILITY", "CARD_EFFECT"])}
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


def make_feature_schema(records: list[dict[str, Any]], details: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    details = details or []
    detail_subtypes = sorted({str(x.get("detail_subtype")) for x in details})
    damage_modes = sorted({str(x.get("damage_mode")) for x in details})
    damage_values = [float(x["damage_base"]) for x in details if x.get("damage_base") is not None]
    damage_mean = sum(damage_values) / len(damage_values) if damage_values else 0.0
    damage_std = max((sum((x - damage_mean) ** 2 for x in damage_values) / max(1, len(damage_values))) ** 0.5, 1.0)
    return {
        "schema_version": 3,
        "vocab": build_vocab(records),
        "normalization": build_normalization(records),
        "single_fields": SINGLE_CATEGORICAL_FIELDS,
        "multi_fields": MULTI_CATEGORICAL_FIELDS,
        "numeric_fields": NUMERIC_FIELDS,
        "text_hash_dim": 2048,
        "detail_subtype_vocab": {value: index for index, value in enumerate([NULL_TOKEN, UNK_TOKEN, MASK_TOKEN, *detail_subtypes])},
        "damage_mode_vocab": {value: index for index, value in enumerate([NULL_TOKEN, UNK_TOKEN, MASK_TOKEN, *damage_modes])},
        "detail_count": len(details),
        "detail_damage_normalization": {"mean": damage_mean, "std": damage_std},
    }


class CardDataset(Dataset):
    def __init__(
        self,
        records: list[dict[str, Any]],
        card_id_to_index: dict[str, int],
        schema: dict[str, Any],
        indices: list[int] | None = None,
        details: list[dict[str, Any]] | None = None,
        detail_offsets: list[int] | None = None,
        manifest: dict[str, Any] | None = None,
    ) -> None:
        self.records = records
        self.card_id_to_index = card_id_to_index
        self.schema = schema
        self.indices = indices or list(range(len(records)))
        self.details = details or []
        self.detail_offsets = detail_offsets or [0] * (len(records) + 1)
        self.manifest = manifest or {}

    @classmethod
    def from_cache(cls, cache_dir: Path = DEFAULT_CACHE_DIR, rebuild: bool = False) -> "CardDataset":
        records, details, offsets, mapping, manifest = load_or_create_corpus(cache_dir, rebuild=rebuild)
        schema_path = cache_dir / "card_feature_schema.json"
        if schema_path.exists() and not rebuild:
            schema = json.loads(schema_path.read_text(encoding="utf-8"))
            if schema.get("schema_version") != 3 or "energy_type_target" not in schema.get("vocab", {}) or schema.get("detail_count") != len(details):
                schema = make_feature_schema(records, details)
                schema_path.write_text(json.dumps(schema, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        else:
            schema = make_feature_schema(records, details)
            schema_path.write_text(json.dumps(schema, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return cls(records, mapping, schema, details=details, detail_offsets=offsets, manifest=manifest)

    def subset(self, indices: list[int]) -> "CardDataset":
        return CardDataset(self.records, self.card_id_to_index, self.schema, indices, self.details, self.detail_offsets, self.manifest)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, Any]:
        index = self.indices[item]
        record = self.records[index]
        start, end = self.detail_offsets[index], self.detail_offsets[index + 1]
        return {"index": index, "card_id": record["card_id"], "record": record, "details": self.details[start:end]}

    def relation_samples(self) -> dict[str, list[tuple[int, int]]]:
        name_to_indices: dict[str, list[int]] = defaultdict(list)
        evolves_pairs: list[tuple[int, int]] = []
        for index in self.indices:
            record = self.records[index]
            name_to_indices[record["name"]].append(index)
        name_first_index = {name: values[0] for name, values in name_to_indices.items()}
        for index in self.indices:
            record = self.records[index]
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
    return __import__("re").findall(r"\{[^}]+\}|[^\W_]+(?:['’][^\W_]+)*|\d+|[^\w\s]", text.casefold(), flags=__import__("re").UNICODE)


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
    detail_rows: list[list[dict[str, Any]]] = []
    for item in items:
        details = list(item.get("details") or [])
        if not details:
            details = [{"detail_type": NULL_TOKEN, "detail_subtype": NULL_TOKEN, "text": "", "energy_counts": [0] * 12, "damage_base": None, "damage_mode": NULL_TOKEN, "attack_id": None, "detail_index": -1, "_padding": True}]
        detail_rows.append(details)
    max_details = max(len(row) for row in detail_rows)
    max_detail_tokens = max(1, max(len(re_tokenize(str(detail.get("text") or ""))) for row in detail_rows for detail in row))
    detail_hashes, detail_token_masks, detail_masks = [], [], []
    detail_type_ids, detail_subtype_ids, detail_energy_counts = [], [], []
    detail_damage, detail_damage_mask, detail_damage_modes, detail_attack_ids, detail_indices = [], [], [], [], []
    hash_dim = int(schema.get("text_hash_dim", 2048))
    for row in detail_rows:
        encoded_row, mask_row, type_row, subtype_row, energy_row = [], [], [], [], []
        damage_row, damage_mask_row, mode_row, attack_row, index_row = [], [], [], [], []
        for detail in row:
            text = str(detail.get("text") or "")
            values = [stable_hash(token.lower(), hash_dim) for token in re_tokenize(text)] or [0]
            encoded_row.append(values + [0] * (max_detail_tokens - len(values)))
            mask_row.append([1.0] * len(values) + [0.0] * (max_detail_tokens - len(values)))
            type_row.append(schema["vocab"]["detail_type"].get(str(detail.get("detail_type")), 1))
            subtype_row.append(schema["detail_subtype_vocab"].get(str(detail.get("detail_subtype")), 1))
            energy_row.append([float(x) for x in detail.get("energy_counts", [0] * 12)])
            raw_damage = detail.get("damage_base")
            stats = schema["detail_damage_normalization"]
            damage_row.append(0.0 if raw_damage is None else (float(raw_damage) - stats["mean"]) / stats["std"])
            damage_mask_row.append(0.0 if raw_damage is None else 1.0)
            mode_row.append(schema["damage_mode_vocab"].get(str(detail.get("damage_mode")), 1))
            attack_row.append(int(detail.get("attack_id") or 0))
            index_row.append(int(detail.get("detail_index", -1)))
        padding = max_details - len(row)
        encoded_row.extend([[0] * max_detail_tokens for _ in range(padding)])
        mask_row.extend([[0.0] * max_detail_tokens for _ in range(padding)])
        type_row.extend([0] * padding); subtype_row.extend([0] * padding); energy_row.extend([[0.0] * 12 for _ in range(padding)])
        damage_row.extend([0.0] * padding); damage_mask_row.extend([0.0] * padding); mode_row.extend([0] * padding)
        attack_row.extend([0] * padding); index_row.extend([-1] * padding)
        detail_hashes.append(encoded_row)
        detail_token_masks.append(mask_row)
        detail_masks.append([0.0 if detail.get("_padding") else 1.0 for detail in row] + [0.0] * padding)
        detail_type_ids.append(type_row); detail_subtype_ids.append(subtype_row); detail_energy_counts.append(energy_row)
        detail_damage.append(damage_row); detail_damage_mask.append(damage_mask_row); detail_damage_modes.append(mode_row)
        detail_attack_ids.append(attack_row); detail_indices.append(index_row)
    def normalized_target(item: dict[str, Any], field: str) -> float:
        value = numeric_values(item["record"]).get(field)
        if value is None:
            return 0.0
        stats = schema["normalization"][field]
        return (float(value) - stats["mean"]) / stats["std"]

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
        "category_target": torch.tensor([schema["vocab"]["single"]["category"].get(str(item["record"].get("category")), 1) for item in items], dtype=torch.long),
        "pokemon_type_target": torch.tensor([schema["vocab"]["single"]["pokemon_type"].get(str(item["record"].get("pokemon_type")), 0) if item["record"].get("pokemon_type") is not None else -100 for item in items], dtype=torch.long),
        "stage_target": torch.tensor([schema["vocab"]["single"]["stage"].get(str(item["record"].get("stage")), 0) if item["record"].get("stage") is not None else -100 for item in items], dtype=torch.long),
        "trainer_type_target": torch.tensor([schema["vocab"]["single"]["trainer_type"].get(str(item["record"].get("trainer_type")), 0) if item["record"].get("trainer_type") is not None else -100 for item in items], dtype=torch.long),
        "energy_type_target": torch.tensor([schema["vocab"]["energy_type_target"].get(str(energy_type_label(item["record"])), 0) if energy_type_label(item["record"]) is not None else -100 for item in items], dtype=torch.long),
        "hp_target": torch.tensor([normalized_target(item, "hp") for item in items], dtype=torch.float32),
        "retreat_target": torch.tensor([int(numeric_values(item["record"]).get("retreat_cost") or 0) for item in items], dtype=torch.long).clamp(0, 5),
        "damage_target": torch.tensor([normalized_target(item, "max_attack_damage") for item in items], dtype=torch.float32),
        "energy_cost_target": torch.tensor([max(0.0, float(numeric_values(item["record"]).get("max_attack_energy_cost") or 0.0)) for item in items], dtype=torch.float32),
        "card_ids": [item["card_id"] for item in items],
        "texts": [row["text"] for row in encoded],
        "detail_text_hashes": torch.tensor(detail_hashes, dtype=torch.long),
        "detail_text_token_mask": torch.tensor(detail_token_masks, dtype=torch.float32),
        "detail_mask": torch.tensor(detail_masks, dtype=torch.bool),
        "detail_type_ids": torch.tensor(detail_type_ids, dtype=torch.long),
        "detail_subtype_ids": torch.tensor(detail_subtype_ids, dtype=torch.long),
        "attack_energy_counts": torch.tensor(detail_energy_counts, dtype=torch.float32),
        "detail_damage": torch.tensor(detail_damage, dtype=torch.float32),
        "detail_damage_mask": torch.tensor(detail_damage_mask, dtype=torch.float32),
        "detail_damage_mode_ids": torch.tensor(detail_damage_modes, dtype=torch.long),
        "detail_attack_ids": torch.tensor(detail_attack_ids, dtype=torch.long),
        "detail_indices": torch.tensor(detail_indices, dtype=torch.long),
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


def split_record_indices(records: list[dict[str, Any]], val_ratio: float, seed: int, mode: str) -> tuple[list[int], list[int]]:
    """Group related printings when evaluating semantic generalization."""
    if mode == "sample":
        return split_indices(len(records), val_ratio, seed, mode)
    if mode not in {"card_id", "name", "evolution_chain"}:
        raise ValueError("split mode must be sample, card_id, name, or evolution_chain")
    groups: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        if mode == "card_id":
            key = str(record.get("card_id", index))
        elif mode == "name":
            key = str(record.get("name") or record.get("card_id", index)).lower()
        else:
            key = str(record.get("evolves_from") or record.get("name") or record.get("card_id", index)).lower()
        groups[key].append(index)
    keys = list(groups)
    random.Random(seed).shuffle(keys)
    target = max(1, int(len(records) * val_ratio))
    validation, training = [], []
    for key in keys:
        (validation if len(validation) < target else training).extend(groups[key])
    return training, validation
