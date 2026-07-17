from __future__ import annotations

import json
import random
import re
import unicodedata
from pathlib import Path
from typing import Any, Iterable

import torch
from torch.utils.data import Dataset

from .card_preprocessing import DEFAULT_CACHE_DIR, ENERGY_TYPES, load_or_create_corpus


PAD_TOKEN = "<PAD>"
NULL_TOKEN = "<NULL>"
UNK_TOKEN = "<UNK>"
MASK_TOKEN = "<MASK>"
SPECIAL_TOKENS = [PAD_TOKEN, NULL_TOKEN, UNK_TOKEN]
DETAIL_SPECIAL_TOKENS = [*SPECIAL_TOKENS, MASK_TOKEN]
SCHEMA_VERSION = 8
TOKEN_PATTERN = re.compile(r"\{[^}]+\}|[^\W_]+(?:['’][^\W_]+)*|\d+|[^\w\s]", re.UNICODE)


def _normalized(value: Any) -> str:
    if value is None:
        return NULL_TOKEN
    text = unicodedata.normalize("NFC", str(value)).replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return NULL_TOKEN if not text or text.casefold() in {"n/a", "nan", "none", "-", "—"} else text


def _vocab(values: Iterable[Any], *, include_mask: bool = False) -> dict[str, int]:
    special_tokens = DETAIL_SPECIAL_TOKENS if include_mask else SPECIAL_TOKENS
    normalized = {_normalized(value) for value in values}
    normalized.difference_update(DETAIL_SPECIAL_TOKENS)
    ordered = sorted(normalized, key=lambda value: (value.casefold(), value))
    return {value: index for index, value in enumerate([*special_tokens, *ordered])}


def _id(vocab: dict[str, int], value: Any) -> int:
    return vocab.get(_normalized(value), vocab[UNK_TOKEN])


def _tokens(text: Any) -> list[str]:
    value = unicodedata.normalize("NFKC", str(text or "")).casefold()
    return TOKEN_PATTERN.findall(value)


def _stats(values: Iterable[Any]) -> dict[str, float]:
    present = [float(value) for value in values if value is not None]
    if not present:
        return {"mean": 0.0, "std": 1.0}
    mean = sum(present) / len(present)
    variance = sum((value - mean) ** 2 for value in present) / len(present)
    return {"mean": mean, "std": max(variance**0.5, 1.0)}


def _normalize_number(value: Any, stats: dict[str, float]) -> float:
    return 0.0 if value is None else (float(value) - stats["mean"]) / stats["std"]


def make_feature_schema(records: list[dict[str, Any]], details: list[dict[str, Any]]) -> dict[str, Any]:
    detail_names = [detail.get("detail_name") for detail in details]
    text_tokens = [token for detail in details for token in _tokens(detail.get("text"))]
    name_values = [
        value
        for record in records
        for value in [record.get("name"), record.get("evolves_from"), *(record.get("evolves_to") or [])]
    ]
    schema = {
        "schema_version": SCHEMA_VERSION,
        "non_detail_card_fields": [
            "name_id", "card_type_id", "stage_id", "rule_id", "category_id", "type_id",
            "weakness_type_id", "resistance_type_id", "hp_normalized", "hp_mask",
            "retreat_normalized", "retreat_mask", "evolves_from_name_id", "evolves_from_mask",
            "evolves_to_name_ids",
        ],
        "non_detail_batch_fields": [
            "name_ids", "card_type_ids", "stage_ids", "rule_ids", "category_ids", "type_ids",
            "weakness_type_ids", "resistance_type_ids", "hp_normalized", "hp_mask",
            "retreat_normalized", "retreat_mask", "evolves_from_name_ids", "evolves_from_mask",
            "evolves_to_name_ids", "evolves_to_mask",
        ],
        "attack_energy_types": list(ENERGY_TYPES),
        "name_vocab": _vocab(name_values),
        "card_type_vocab": _vocab(record.get("card_type") for record in records),
        "stage_vocab": _vocab(record.get("stage") for record in records),
        "rule_vocab": _vocab(record.get("rule") for record in records),
        "category_vocab": _vocab(record.get("category") for record in records),
        "type_vocab": _vocab(
            value
            for record in records
            for value in [record.get("type"), record.get("weakness_type"), record.get("resistance_type")]
        ),
        "numeric_scaling": {
            "mode": "fixed_divisors",
            "hp_divisor": 100.0,
            "retreat_divisor": 4.0,
        },
        "detail_type_vocab": _vocab((detail.get("detail_type") for detail in details), include_mask=True),
        "detail_subtype_vocab": _vocab((detail.get("detail_subtype") for detail in details), include_mask=True),
        "detail_name_vocab": _vocab(detail_names, include_mask=True),
        "detail_text_vocab": _vocab(text_tokens, include_mask=True),
        "damage_mode_vocab": _vocab((detail.get("damage_mode") for detail in details), include_mask=True),
        "detail_normalization": {
            "attack_damage": _stats(
                detail.get("damage_base")
                for detail in details
                if detail.get("detail_type") == "ATTACK"
            ),
        },
        "card_count": len(records),
        "detail_count": len(details),
    }
    return schema


def _schema_valid(schema: dict[str, Any], records: list[dict[str, Any]], details: list[dict[str, Any]]) -> bool:
    required = {
        "name_vocab", "card_type_vocab", "stage_vocab", "rule_vocab", "category_vocab", "type_vocab",
        "numeric_scaling", "detail_text_vocab", "detail_normalization",
    }
    return (
        schema.get("schema_version") == SCHEMA_VERSION
        and schema.get("card_count") == len(records)
        and schema.get("detail_count") == len(details)
        and schema.get("numeric_scaling", {}).get("mode") == "fixed_divisors"
        and required <= schema.keys()
    )


def encode_card(record: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
    evolves_from = record.get("evolves_from")
    evolves_to = list(record.get("evolves_to") or [])
    hp = record.get("hp")
    retreat = record.get("retreat_cost")
    type_value = record.get("type")
    weakness_type = record.get("weakness_type")
    resistance_type = record.get("resistance_type")
    scaling = schema["numeric_scaling"]
    return {
        "name_id": _id(schema["name_vocab"], record.get("name")),
        "card_type_id": _id(schema["card_type_vocab"], record.get("card_type")),
        "stage_id": _id(schema["stage_vocab"], record.get("stage")),
        "rule_id": _id(schema["rule_vocab"], record.get("rule")),
        "category_id": _id(schema["category_vocab"], record.get("category")),
        "type_id": _id(schema["type_vocab"], type_value),
        "weakness_type_id": _id(schema["type_vocab"], weakness_type),
        "resistance_type_id": _id(schema["type_vocab"], resistance_type),
        "hp_normalized": 0.0 if hp is None else float(hp) / scaling["hp_divisor"],
        "hp_mask": float(hp is not None),
        "retreat_normalized": 0.0 if retreat is None else float(retreat) / scaling["retreat_divisor"],
        "retreat_mask": float(retreat is not None),
        "evolves_from_name_id": _id(schema["name_vocab"], evolves_from),
        "evolves_from_mask": float(evolves_from is not None),
        "evolves_to_name_ids": [_id(schema["name_vocab"], value) for value in evolves_to],
    }


def encode_detail(detail: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
    detail_type = str(detail.get("detail_type") or NULL_TOKEN)
    detail_name = _normalized(detail.get("detail_name"))
    is_attack = detail_type == "ATTACK"
    damage = detail.get("damage_base") if is_attack else None
    text_ids = [_id(schema["detail_text_vocab"], token) for token in _tokens(detail.get("text"))]
    return {
        "detail_id": int(detail.get("detail_id", detail.get("detail_index", -1))),
        "detail_type_id": _id(schema["detail_type_vocab"], detail_type),
        "detail_subtype_id": _id(schema["detail_subtype_vocab"], detail.get("detail_subtype")),
        "detail_name_id": _id(schema["detail_name_vocab"], detail_name),
        "detail_text_ids": text_ids,
        "attack_energy_counts": [float(value) for value in (detail.get("energy_counts") or [0] * 12)[:12]] if is_attack else [0.0] * 12,
        "attack_energy_mask": float(is_attack),
        "damage_raw": float(damage or 0),
        "damage_raw_text": detail.get("damage_raw"),
        "damage_normalized": _normalize_number(damage, schema["detail_normalization"]["attack_damage"]),
        "damage_mode_id": _id(schema["damage_mode_vocab"], detail.get("damage_mode") if is_attack else None),
        "damage_mask": float(is_attack and damage is not None),
        "attack_id": int(detail.get("attack_id") or 0),
        "source_row": int(detail.get("source_row", -1)),
    }


class CardDataset(Dataset):
    def __init__(
        self,
        records: list[dict[str, Any]],
        details: list[dict[str, Any]],
        detail_offsets: list[int],
        card_id_to_index: dict[str, int],
        schema: dict[str, Any],
        indices: list[int] | None = None,
        manifest: dict[str, Any] | None = None,
    ) -> None:
        self.records = records
        self.details = details
        self.detail_offsets = detail_offsets
        self.card_id_to_index = {str(key): int(value) for key, value in card_id_to_index.items()}
        self.schema = schema
        self.indices = list(range(len(records))) if indices is None else list(indices)
        self.manifest = manifest or {}
        if len(detail_offsets) != len(records) + 1:
            raise ValueError("detail_offsets must contain one boundary per card plus the final boundary")
        for index, record in enumerate(records):
            start, end = detail_offsets[index : index + 2]
            expected_ids = list(range(start, end))
            if record.get("detail_ids") != expected_ids:
                raise ValueError(f"card {record.get('card_id')} has inconsistent detail_ids")
            if any(details[detail_id].get("detail_id") != detail_id for detail_id in expected_ids):
                raise ValueError(f"card {record.get('card_id')} points to an invalid detail_id")

    @classmethod
    def from_cache(cls, cache_dir: Path = DEFAULT_CACHE_DIR, rebuild: bool = False) -> "CardDataset":
        records, details, offsets, mapping, manifest = load_or_create_corpus(cache_dir, rebuild=rebuild)
        schema_path = cache_dir / "card_feature_schema.json"
        schema = json.loads(schema_path.read_text(encoding="utf-8")) if schema_path.exists() and not rebuild else {}
        if not _schema_valid(schema, records, details):
            schema = make_feature_schema(records, details)
            schema_path.write_text(json.dumps(schema, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return cls(records, details, offsets, mapping, schema, manifest=manifest)

    def subset(self, indices: list[int]) -> "CardDataset":
        return CardDataset(
            self.records, self.details, self.detail_offsets, self.card_id_to_index,
            self.schema, indices=indices, manifest=self.manifest,
        )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, Any]:
        index = self.indices[item]
        record = self.records[index]
        return {
            "index": index,
            "card_id": str(record["card_id"]),
            "record": record,
            "card": encode_card(record, self.schema),
            "details": [encode_detail(self.details[detail_id], self.schema) for detail_id in record["detail_ids"]],
        }


def collate_cards(items: list[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        raise ValueError("cannot collate an empty card batch")
    cards = [item["card"] for item in items]
    max_evolves = max(1, max(len(card["evolves_to_name_ids"]) for card in cards))
    max_details = max(1, max(len(item["details"]) for item in items))
    max_text = max(1, max((len(detail["detail_text_ids"]) for item in items for detail in item["details"]), default=0))

    batch: dict[str, Any] = {}
    scalar_long = {
        "name_ids": "name_id", "card_type_ids": "card_type_id", "stage_ids": "stage_id",
        "rule_ids": "rule_id", "category_ids": "category_id", "type_ids": "type_id",
        "weakness_type_ids": "weakness_type_id", "resistance_type_ids": "resistance_type_id",
        "evolves_from_name_ids": "evolves_from_name_id",
    }
    scalar_float = [
        "hp_normalized", "hp_mask", "retreat_normalized", "retreat_mask", "evolves_from_mask",
    ]
    for output_name, field in scalar_long.items():
        batch[output_name] = torch.tensor([card[field] for card in cards], dtype=torch.long)
    for field in scalar_float:
        batch[field] = torch.tensor([card[field] for card in cards], dtype=torch.float32)
    batch["evolves_to_name_ids"] = torch.tensor(
        [
            card["evolves_to_name_ids"] + [0] * (max_evolves - len(card["evolves_to_name_ids"]))
            for card in cards
        ],
        dtype=torch.long,
    )
    batch["evolves_to_mask"] = torch.tensor(
        [
            [1.0] * len(card["evolves_to_name_ids"])
            + [0.0] * (max_evolves - len(card["evolves_to_name_ids"]))
            for card in cards
        ],
        dtype=torch.float32,
    )

    detail_fields: dict[str, list[Any]] = {name: [] for name in [
        "detail_ids", "detail_type_ids", "detail_subtype_ids", "detail_name_ids",
        "detail_mask", "detail_text_ids", "detail_text_mask", "attack_energy_counts", "attack_energy_mask",
        "damage_raw", "damage_normalized", "damage_mode_ids", "damage_mask", "attack_ids",
    ]}
    damage_raw_texts: list[list[str | None]] = []
    for item in items:
        rows = item["details"]
        padding = max_details - len(rows)
        detail_fields["detail_ids"].append([row["detail_id"] for row in rows] + [-1] * padding)
        for source, target in [
            ("detail_type_id", "detail_type_ids"), ("detail_subtype_id", "detail_subtype_ids"),
            ("detail_name_id", "detail_name_ids"),
            ("damage_mode_id", "damage_mode_ids"), ("attack_id", "attack_ids"),
        ]:
            detail_fields[target].append([row[source] for row in rows] + [0] * padding)
        for source in ["attack_energy_mask", "damage_raw", "damage_normalized", "damage_mask"]:
            detail_fields[source].append([row[source] for row in rows] + [0.0] * padding)
        detail_fields["detail_mask"].append([1.0] * len(rows) + [0.0] * padding)
        detail_fields["attack_energy_counts"].append([row["attack_energy_counts"] for row in rows] + [[0.0] * 12 for _ in range(padding)])
        text_rows = [row["detail_text_ids"] + [0] * (max_text - len(row["detail_text_ids"])) for row in rows]
        text_masks = [[1.0] * len(row["detail_text_ids"]) + [0.0] * (max_text - len(row["detail_text_ids"])) for row in rows]
        detail_fields["detail_text_ids"].append(text_rows + [[0] * max_text for _ in range(padding)])
        detail_fields["detail_text_mask"].append(text_masks + [[0.0] * max_text for _ in range(padding)])
        damage_raw_texts.append([row["damage_raw_text"] for row in rows] + [None] * padding)
    long_fields = {
        "detail_ids", "detail_type_ids", "detail_subtype_ids", "detail_name_ids", "detail_text_ids",
        "damage_mode_ids", "attack_ids",
    }
    for field, values in detail_fields.items():
        batch[field] = torch.tensor(values, dtype=torch.long if field in long_fields else torch.float32)
    batch["detail_mask"] = batch["detail_mask"].bool()
    batch["damage_raw_texts"] = damage_raw_texts
    return batch


def split_train_validation_test(
    dataset: CardDataset,
    validation_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 0,
) -> tuple[CardDataset, CardDataset, CardDataset]:
    if validation_ratio < 0 or test_ratio < 0 or validation_ratio + test_ratio >= 1:
        raise ValueError("validation_ratio and test_ratio must be non-negative and sum to less than one")
    indices = list(dataset.indices)
    random.Random(seed).shuffle(indices)
    test_count = int(round(len(indices) * test_ratio))
    validation_count = int(round(len(indices) * validation_ratio))
    test = indices[:test_count]
    validation = indices[test_count : test_count + validation_count]
    train = indices[test_count + validation_count :]
    return dataset.subset(train), dataset.subset(validation), dataset.subset(test)
