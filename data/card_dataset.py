from __future__ import annotations

import json
import random
from collections import defaultdict
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
BASE_NUMERIC_FIELDS = [
    "hp",
    "retreat_cost",
    "weakness_value",
    "resistance_value",
    "attack_count",
    "ability_count",
    "special_effect_count",
]
OPTIONAL_SUMMARY_NUMERIC_FIELDS = ["max_attack_damage", "min_attack_energy_cost", "max_attack_energy_cost"]
NUMERIC_FIELDS = BASE_NUMERIC_FIELDS
DAMAGE_MODES = ["fixed", "plus", "times", "variable", "none", "unknown"]
EFFECT_SOURCE_TYPES = ["trainer", "item", "supporter", "stadium", "tool", "special_energy", "rule_box", "other", "ability"]
DETAIL_TYPE_IDS = {"padding": 0, "attack": 1, "ability": 2, "special_effect": 3}


def card_text(record: dict[str, Any]) -> str:
    parts = [
        record.get("name") or "",
        record.get("card_type") or "",
        record.get("subtype") or "",
        " ".join(record.get("ability_texts") or []),
        " ".join(record.get("attack_names") or []),
        " ".join(record.get("attack_texts") or []),
        " ".join(effect.get("effect_text", "") for effect in record.get("special_effects") or []),
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
    vocab["damage_mode"] = {value: index for index, value in enumerate(DAMAGE_MODES)}
    vocab["effect_source_type"] = {value: index for index, value in enumerate(EFFECT_SOURCE_TYPES)}
    vocab["detail_type"] = DETAIL_TYPE_IDS
    return vocab


def energy_type_label(record: dict[str, Any]) -> str | None:
    if record.get("card_type") == "POKEMON" and record.get("pokemon_type") is not None:
        return str(record["pokemon_type"])
    if record.get("card_type") in {"BASIC_ENERGY", "SPECIAL_ENERGY"} and record.get("provided_energy_types"):
        return str(record["provided_energy_types"][0])
    return None


def numeric_values(record: dict[str, Any]) -> dict[str, float | None]:
    attacks = record.get("attacks") or []
    damages = [attack.get("damage_value") for attack in attacks if attack.get("damage_value") is not None]
    total_costs = [sum(int(v) for v in (attack.get("energy_costs") or {}).values()) for attack in attacks]
    if not attacks:
        damages = [value for value in record.get("attack_damage") or [] if value is not None]
        total_costs = [sum(int(v) for v in cost.values()) for cost in record.get("attack_energy_costs") or []]
    return {
        "hp": record.get("hp"),
        "retreat_cost": record.get("retreat_cost"),
        "weakness_value": record.get("weakness_value"),
        "resistance_value": record.get("resistance_value"),
        "attack_count": len(attacks or record.get("attack_names") or []),
        "ability_count": len(record.get("abilities") or record.get("ability_texts") or []),
        "special_effect_count": len(record.get("special_effects") or []),
        "max_attack_damage": max(damages) if damages else None,
        "min_attack_energy_cost": min(total_costs) if total_costs else None,
        "max_attack_energy_cost": max(total_costs) if total_costs else None,
    }


def build_normalization(records: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {}
    for field in BASE_NUMERIC_FIELDS + OPTIONAL_SUMMARY_NUMERIC_FIELDS:
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


def build_detail_normalization(records: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    damage_values = []
    total_costs = []
    for record in records:
        for attack in normalized_attacks(record):
            if attack.get("damage_value") is not None:
                damage_values.append(float(attack["damage_value"]))
            total_costs.append(float(sum(int(v) for v in (attack.get("energy_costs") or {}).values())))

    def stats(values: list[float]) -> dict[str, float]:
        if not values:
            return {"mean": 0.0, "std": 1.0}
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / max(1, len(values))
        return {"mean": mean, "std": max(variance**0.5, 1.0)}

    return {
        "attack_damage": stats(damage_values),
        "attack_total_energy_cost": stats(total_costs),
    }


def make_feature_schema(records: list[dict[str, Any]], include_optional_summary_numeric: bool = False) -> dict[str, Any]:
    numeric_fields = list(BASE_NUMERIC_FIELDS)
    if include_optional_summary_numeric:
        numeric_fields.extend(OPTIONAL_SUMMARY_NUMERIC_FIELDS)
    return {
        "vocab": build_vocab(records),
        "normalization": build_normalization(records),
        "detail_normalization": build_detail_normalization(records),
        "single_fields": SINGLE_CATEGORICAL_FIELDS,
        "multi_fields": MULTI_CATEGORICAL_FIELDS,
        "numeric_fields": numeric_fields,
        "optional_summary_numeric_fields": OPTIONAL_SUMMARY_NUMERIC_FIELDS,
        "include_optional_summary_numeric": include_optional_summary_numeric,
        "text_hash_dim": 2048,
        "schema_version": "static_detail_v1",
    }


def normalized_attacks(record: dict[str, Any]) -> list[dict[str, Any]]:
    attacks = record.get("attacks") or []
    if attacks:
        return attacks
    rows = []
    names = record.get("attack_names") or []
    texts = record.get("attack_texts") or []
    damages = record.get("attack_damage") or []
    costs = record.get("attack_energy_costs") or []
    ids = record.get("attack_ids") or []
    for index, name in enumerate(names):
        damage = damages[index] if index < len(damages) else None
        rows.append(
            {
                "attack_id": str(ids[index]) if index < len(ids) else None,
                "name": name or "",
                "effect_text": texts[index] if index < len(texts) else "",
                "damage_raw": str(damage) if damage is not None else None,
                "damage_value": float(damage) if damage is not None else None,
                "damage_mode": "fixed" if damage is not None else "none",
                "energy_costs": costs[index] if index < len(costs) else {},
            }
        )
    return rows


def normalized_abilities(record: dict[str, Any]) -> list[dict[str, Any]]:
    abilities = record.get("abilities") or []
    if abilities:
        return abilities
    return [{"name": "", "effect_text": text or ""} for text in record.get("ability_texts") or []]


def normalized_effects(record: dict[str, Any]) -> list[dict[str, Any]]:
    return record.get("special_effects") or []


def token_hashes(text: str, schema: dict[str, Any]) -> list[int]:
    hash_dim = int(schema.get("text_hash_dim", 2048))
    return [stable_hash(token.lower(), hash_dim) for token in re_tokenize(text)] or [0]


def energy_count_vector(costs: dict[str, int]) -> list[float]:
    return [float(costs.get(energy, 0)) for energy in ENERGY_TYPES]


def provided_energy_vector(record: dict[str, Any]) -> list[float]:
    values = set(str(value) for value in record.get("provided_energy_types") or [])
    return [1.0 if energy in values else 0.0 for energy in ENERGY_TYPES]


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
            if schema.get("schema_version") != "static_detail_v1" or "energy_type_target" not in schema.get("vocab", {}):
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
    text_hashes = token_hashes(text, schema)
    detail_norm = schema.get("detail_normalization", {})
    damage_stats = detail_norm.get("attack_damage", {"mean": 0.0, "std": 1.0})
    cost_stats = detail_norm.get("attack_total_energy_cost", {"mean": 0.0, "std": 1.0})
    damage_vocab = schema["vocab"]["damage_mode"]
    source_vocab = schema["vocab"]["effect_source_type"]
    attacks = []
    for attack in normalized_attacks(record):
        costs = {str(key): int(value) for key, value in (attack.get("energy_costs") or {}).items()}
        total_cost = float(sum(costs.values()))
        damage_value = attack.get("damage_value")
        damage_mask = 1.0 if damage_value is not None else 0.0
        normalized_damage = 0.0
        if damage_value is not None:
            normalized_damage = (float(damage_value) - damage_stats["mean"]) / damage_stats["std"]
        attacks.append(
            {
                "attack_id": attack.get("attack_id"),
                "name": attack.get("name") or "",
                "name_hashes": token_hashes(attack.get("name") or "", schema),
                "effect_text": attack.get("effect_text") or "",
                "effect_hashes": token_hashes(attack.get("effect_text") or "", schema),
                "energy_counts": energy_count_vector(costs),
                "total_energy_cost": (total_cost - cost_stats["mean"]) / cost_stats["std"],
                "raw_total_energy_cost": total_cost,
                "damage": normalized_damage,
                "raw_damage": float(damage_value or 0.0),
                "damage_mask": damage_mask,
                "damage_mode": damage_vocab.get(str(attack.get("damage_mode") or "unknown"), damage_vocab["unknown"]),
            }
        )
    abilities = [
        {
            "name": ability.get("name") or "",
            "name_hashes": token_hashes(ability.get("name") or "", schema),
            "effect_text": ability.get("effect_text") or "",
            "effect_hashes": token_hashes(ability.get("effect_text") or "", schema),
            "source_type": source_vocab.get("ability", source_vocab["other"]),
        }
        for ability in normalized_abilities(record)
    ]
    effects = [
        {
            "source_type_name": effect.get("source_type") or "other",
            "source_type": source_vocab.get(str(effect.get("source_type") or "other"), source_vocab["other"]),
            "name": effect.get("name") or "",
            "name_hashes": token_hashes(effect.get("name") or "", schema),
            "effect_text": effect.get("effect_text") or "",
            "effect_hashes": token_hashes(effect.get("effect_text") or "", schema),
        }
        for effect in normalized_effects(record)
    ]
    return {
        "single": single,
        "multi_values": multi_values,
        "multi_offsets": multi_offsets,
        "numeric": numeric,
        "numeric_mask": numeric_mask,
        "text_hashes": text_hashes,
        "text": text,
        "attacks": attacks,
        "abilities": abilities,
        "effects": effects,
        "provided_energy_vector": provided_energy_vector(record),
        "provided_energy_amount": float(len(record.get("provided_energy_types") or [])),
        "is_energy": float(record.get("card_type") in {"BASIC_ENERGY", "SPECIAL_ENERGY"}),
        "is_basic_energy": float(record.get("card_type") == "BASIC_ENERGY"),
        "is_special_energy": float(record.get("card_type") == "SPECIAL_ENERGY"),
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
    max_attacks = max(1, max(len(row["attacks"]) for row in encoded))
    max_abilities = max(1, max(len(row["abilities"]) for row in encoded))
    max_effects = max(1, max(len(row["effects"]) for row in encoded))
    max_attack_name = max(1, max((len(attack["name_hashes"]) for row in encoded for attack in row["attacks"]), default=1))
    max_attack_effect = max(1, max((len(attack["effect_hashes"]) for row in encoded for attack in row["attacks"]), default=1))
    max_ability_name = max(1, max((len(ability["name_hashes"]) for row in encoded for ability in row["abilities"]), default=1))
    max_ability_effect = max(1, max((len(ability["effect_hashes"]) for row in encoded for ability in row["abilities"]), default=1))
    max_effect_name = max(1, max((len(effect["name_hashes"]) for row in encoded for effect in row["effects"]), default=1))
    max_effect_text = max(1, max((len(effect["effect_hashes"]) for row in encoded for effect in row["effects"]), default=1))

    def pad_hashes(values: list[int], length: int) -> list[int]:
        return values + [0] * (length - len(values))

    def hash_mask(values: list[int], length: int) -> list[float]:
        return [1.0] * len(values) + [0.0] * (length - len(values))

    def pad_rows(rows: list[dict[str, Any]], length: int) -> list[dict[str, Any]]:
        return rows + [{} for _ in range(length - len(rows))]

    attack_rows = [pad_rows(row["attacks"], max_attacks) for row in encoded]
    ability_rows = [pad_rows(row["abilities"], max_abilities) for row in encoded]
    effect_rows = [pad_rows(row["effects"], max_effects) for row in encoded]
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
        "attack_name_hashes": torch.tensor(
            [[pad_hashes(attack.get("name_hashes", [0]), max_attack_name) for attack in rows] for rows in attack_rows],
            dtype=torch.long,
        ),
        "attack_name_mask": torch.tensor(
            [[hash_mask(attack.get("name_hashes", [0]), max_attack_name) for attack in rows] for rows in attack_rows],
            dtype=torch.float32,
        ),
        "attack_effect_hashes": torch.tensor(
            [[pad_hashes(attack.get("effect_hashes", [0]), max_attack_effect) for attack in rows] for rows in attack_rows],
            dtype=torch.long,
        ),
        "attack_effect_mask": torch.tensor(
            [[hash_mask(attack.get("effect_hashes", [0]), max_attack_effect) for attack in rows] for rows in attack_rows],
            dtype=torch.float32,
        ),
        "attack_energy_counts": torch.tensor(
            [[attack.get("energy_counts", [0.0] * len(ENERGY_TYPES)) for attack in rows] for rows in attack_rows],
            dtype=torch.float32,
        ),
        "attack_total_energy_cost": torch.tensor(
            [[float(attack.get("total_energy_cost", 0.0)) for attack in rows] for rows in attack_rows],
            dtype=torch.float32,
        ),
        "attack_raw_total_energy_cost": torch.tensor(
            [[float(attack.get("raw_total_energy_cost", 0.0)) for attack in rows] for rows in attack_rows],
            dtype=torch.float32,
        ),
        "attack_damage": torch.tensor([[float(attack.get("damage", 0.0)) for attack in rows] for rows in attack_rows], dtype=torch.float32),
        "attack_raw_damage": torch.tensor([[float(attack.get("raw_damage", 0.0)) for attack in rows] for rows in attack_rows], dtype=torch.float32),
        "attack_damage_mask": torch.tensor([[float(attack.get("damage_mask", 0.0)) for attack in rows] for rows in attack_rows], dtype=torch.float32),
        "attack_damage_mode": torch.tensor([[int(attack.get("damage_mode", 0)) for attack in rows] for rows in attack_rows], dtype=torch.long),
        "attack_mask": torch.tensor(
            [[1.0 if attack else 0.0 for attack in rows] for rows in attack_rows],
            dtype=torch.float32,
        ),
        "ability_name_hashes": torch.tensor(
            [[pad_hashes(ability.get("name_hashes", [0]), max_ability_name) for ability in rows] for rows in ability_rows],
            dtype=torch.long,
        ),
        "ability_name_mask": torch.tensor(
            [[hash_mask(ability.get("name_hashes", [0]), max_ability_name) for ability in rows] for rows in ability_rows],
            dtype=torch.float32,
        ),
        "ability_effect_hashes": torch.tensor(
            [[pad_hashes(ability.get("effect_hashes", [0]), max_ability_effect) for ability in rows] for rows in ability_rows],
            dtype=torch.long,
        ),
        "ability_effect_mask": torch.tensor(
            [[hash_mask(ability.get("effect_hashes", [0]), max_ability_effect) for ability in rows] for rows in ability_rows],
            dtype=torch.float32,
        ),
        "ability_mask": torch.tensor([[1.0 if ability else 0.0 for ability in rows] for rows in ability_rows], dtype=torch.float32),
        "effect_name_hashes": torch.tensor(
            [[pad_hashes(effect.get("name_hashes", [0]), max_effect_name) for effect in rows] for rows in effect_rows],
            dtype=torch.long,
        ),
        "effect_name_mask": torch.tensor(
            [[hash_mask(effect.get("name_hashes", [0]), max_effect_name) for effect in rows] for rows in effect_rows],
            dtype=torch.float32,
        ),
        "effect_text_hashes": torch.tensor(
            [[pad_hashes(effect.get("effect_hashes", [0]), max_effect_text) for effect in rows] for rows in effect_rows],
            dtype=torch.long,
        ),
        "effect_text_mask": torch.tensor(
            [[hash_mask(effect.get("effect_hashes", [0]), max_effect_text) for effect in rows] for rows in effect_rows],
            dtype=torch.float32,
        ),
        "effect_source_type": torch.tensor([[int(effect.get("source_type", 0)) for effect in rows] for rows in effect_rows], dtype=torch.long),
        "effect_mask": torch.tensor([[1.0 if effect else 0.0 for effect in rows] for rows in effect_rows], dtype=torch.float32),
        "provided_energy_multihot": torch.tensor([row["provided_energy_vector"] for row in encoded], dtype=torch.float32),
        "provided_energy_amount": torch.tensor([row["provided_energy_amount"] for row in encoded], dtype=torch.float32),
        "is_energy": torch.tensor([row["is_energy"] for row in encoded], dtype=torch.float32),
        "is_basic_energy": torch.tensor([row["is_basic_energy"] for row in encoded], dtype=torch.float32),
        "is_special_energy": torch.tensor([row["is_special_energy"] for row in encoded], dtype=torch.float32),
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
        "detail_metadata": [
            {
                "card_id": item["card_id"],
                "attacks": [
                    {
                        "attack_id": attack.get("attack_id"),
                        "attack_name": attack.get("name"),
                    }
                    for attack in row["attacks"]
                ],
                "abilities": [
                    {
                        "ability_index": index,
                        "ability_name": ability.get("name"),
                    }
                    for index, ability in enumerate(row["abilities"])
                ],
                "effects": [
                    {
                        "effect_index": index,
                        "effect_source": effect.get("source_type_name"),
                        "effect_name": effect.get("name"),
                    }
                    for index, effect in enumerate(row["effects"])
                ],
            }
            for item, row in zip(items, encoded)
        ],
    }
    numeric_fields = list(schema["numeric_fields"])
    for field, mask_name in [
        ("attack_count", "attack_mask"),
        ("ability_count", "ability_mask"),
        ("special_effect_count", "effect_mask"),
    ]:
        if field in numeric_fields:
            index = numeric_fields.index(field)
            stats = schema["normalization"][field]
            count = batch[mask_name].sum(dim=-1)
            batch["numeric"][:, index] = (count - float(stats["mean"])) / float(stats["std"])
            batch["numeric_mask"][:, index] = 1.0
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
    if mode == "sample":
        return split_indices(len(records), val_ratio, seed, mode)
    if mode != "card_id":
        raise ValueError("split mode must be 'sample' or 'card_id'")
    rng = random.Random(seed)
    groups: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        groups[str(record.get("card_id", index))].append(index)
    group_items = list(groups.items())
    rng.shuffle(group_items)
    target_val = max(1, int(len(records) * val_ratio))
    val_indices: list[int] = []
    train_indices: list[int] = []
    for _card_id, indices in group_items:
        if len(val_indices) < target_val:
            val_indices.extend(indices)
        else:
            train_indices.extend(indices)
    return sorted(train_indices), sorted(val_indices)
