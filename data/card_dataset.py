from __future__ import annotations

import hashlib
import json
import random
import re
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import torch
from torch.utils.data import Dataset

from .card_preprocessing import (
    MASK_TOKEN,
    NULL_TOKEN,
    PAD_TOKEN,
    SCHEMA_VERSION,
    UNK_TOKEN,
    load_or_create_corpus,
)


DETAIL_TYPES = ("ATTACK", "ABILITY", "CARD_EFFECT")
DAMAGE_MODES = ("NONE", "FIXED", "MULTIPLY", "MINUS", "PLUS")

V3_CARD_FIELD_SLOTS = (
    "card_name",
    "card_kind",
    "printed_class",
    "rule",
    "rule_family",
    "category_family",
    "category_qualifier",
    "evolves_from",
    "evolves_to",
    "hp",
    "pokemon_type",
    "energy_printed_type",
    "weakness",
    "resistance",
    "retreat",
)
V3_ENERGY_SYMBOLS = ("C", "G", "R", "W", "L", "P", "F", "D", "M", "N", "Y", "A", "TEAM_ROCKET")
V3_ATTACK_ENERGY_SYMBOLS = V3_ENERGY_SYMBOLS[:-1]
V3_REFERENCE_FIELDS = (*V3_CARD_FIELD_SLOTS, "owner", "zone", "number", "energy_type", "detail_name")
V3_FIELD_RECOVERY_GROUPS = (
    ("printed_class", "card_kind"),
    ("rule", "rule_family"),
    ("category_family", "category_qualifier"),
    ("evolves_from", "evolves_to"),
    ("hp",),
    ("pokemon_type",),
    ("energy_printed_type",),
    ("weakness",),
    ("resistance",),
    ("retreat",),
)
V3_TEXT_TOKEN_KINDS = (PAD_TOKEN, "TEXT", "STRUCTURE_REFERENCE")
V3_MAX_TEXT_SUBWORD_TOKENS = 256
V3_SENTENCEPIECE_CONFIG: dict[str, Any] = {
    "model_type": "unigram",
    "vocab_size": 1024,
    "hard_vocab_limit": False,
    "character_coverage": 1.0,
    "byte_fallback": False,
    "pad_id": 0,
    "unk_id": 1,
    "bos_id": -1,
    "eos_id": -1,
    "user_defined_symbols": ["[MASK_TEXT]"],
    "input_sentence_size": 0,
    "shuffle_input_sentence": False,
}


def text_tokens(text: str) -> list[str]:
    """Compatibility tokenizer for parser-only tests; formal runs use SentencePiece."""

    return [
        token.casefold()
        for token in re.findall(r"\{[^{}]+\}|[A-Za-z0-9]+(?:[\u2019'][A-Za-z0-9]+)?|[^\s]", text or "")
    ]


def _categorical_vocab(values: Iterable[Any]) -> dict[str, int]:
    ordered = [NULL_TOKEN, UNK_TOKEN, MASK_TOKEN]
    ordered.extend(sorted({str(value) for value in values if value is not None and str(value) not in ordered}))
    return {value: index for index, value in enumerate(ordered)}


def _detail_vocab(values: Iterable[Any]) -> dict[str, int]:
    ordered = [PAD_TOKEN, MASK_TOKEN, UNK_TOKEN, NULL_TOKEN]
    ordered.extend(sorted({str(value) for value in values if value is not None and str(value) not in ordered}))
    return {value: index for index, value in enumerate(ordered)}


def _first_value(record: dict[str, Any], *keys: str) -> Any | None:
    for key in keys:
        value = record.get(key)
        if value is not None and value != "":
            return value
    return None


def _evolves_to_names(card: dict[str, Any]) -> list[str]:
    values = card.get("evolves_to_card_names")
    if isinstance(values, list):
        return [str(value) for value in values if value]
    value = _first_value(card, "evolves_to_card_name")
    return [str(value)] if value is not None else []


def _card_value(card: dict[str, Any], field: str) -> str:
    if field == "card_name":
        return str(_first_value(card, "card_name_normalized", "card_name", "name") or NULL_TOKEN)
    if field == "card_kind":
        return str(_first_value(card, "card_kind", "card_category") or NULL_TOKEN)
    if field == "printed_class":
        return str(_first_value(card, "printed_class") or NULL_TOKEN)
    if field == "rule":
        return str(_first_value(card, "rule") or "NONE")
    if field == "rule_family":
        return str(_first_value(card, "rule_family") or "NONE")
    if field == "category_family":
        return str(_first_value(card, "category_family") or "NONE")
    if field == "category_qualifier":
        return str(_first_value(card, "category_qualifier") or "NONE")
    if field == "evolves_from":
        return str(_first_value(card, "evolves_from_card_name", "evolves_from") or NULL_TOKEN)
    if field == "evolves_to":
        names = _evolves_to_names(card)
        return "|".join(names) if names else NULL_TOKEN
    if field == "hp":
        value = _first_value(card, "printed_hp", "hp")
        return str(int(value)) if value is not None else NULL_TOKEN
    if field == "pokemon_type":
        return str(_first_value(card, "pokemon_type") or NULL_TOKEN)
    if field == "energy_printed_type":
        counts = card.get("provided_energy_counts") or {}
        parts = [f"{symbol}:{int(counts.get(symbol, 0))}" for symbol in V3_ENERGY_SYMBOLS]
        return "|".join(parts)
    if field == "weakness":
        return str(_first_value(card, "weakness_type") or NULL_TOKEN)
    if field == "resistance":
        return str(_first_value(card, "resistance_type") or NULL_TOKEN)
    if field == "retreat":
        value = _first_value(card, "retreat", "retreat_cost")
        return str(int(value)) if value is not None else NULL_TOKEN
    raise KeyError(f"unknown static-v3 card field {field!r}")


def _field_applicability(card: dict[str, Any], field: str) -> bool:
    explicit = card.get("field_applicability")
    if isinstance(explicit, dict) and field in explicit:
        return bool(explicit[field])
    if field in {
        "card_name",
        "card_kind",
        "printed_class",
        "rule",
        "rule_family",
        "category_family",
        "category_qualifier",
    }:
        return True
    kind = str(_first_value(card, "card_kind", "card_category") or "")
    if field in {"evolves_from", "evolves_to"}:
        return kind == "POKEMON" or bool(_first_value(card, "evolves_from_card_name")) or bool(_evolves_to_names(card))
    if field == "hp":
        return _first_value(card, "printed_hp", "hp") is not None
    if field == "energy_printed_type":
        return kind == "ENERGY"
    if field in {"pokemon_type", "weakness", "resistance", "retreat"}:
        return kind == "POKEMON"
    return False


def _model_stream(detail: dict[str, Any]) -> list[dict[str, Any]]:
    stream = detail.get("model_text_tokens")
    if not isinstance(stream, list):
        raise ValueError("static-v3 details require canonical model_text_tokens")
    if any(not isinstance(value, dict) for value in stream):
        raise ValueError("model_text_tokens entries must be objects")
    return list(stream)


def _reference_map(detail: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {
        int(reference["reference_id"]): reference
        for reference in detail.get("text_references") or []
        if isinstance(reference, dict) and reference.get("reference_id") is not None
    }


def _reference_field_value(reference: dict[str, Any]) -> tuple[str | None, str | None]:
    payload = reference.get("payload") if isinstance(reference.get("payload"), dict) else {}
    field = str(payload.get("field_name") or "")
    if field not in V3_REFERENCE_FIELDS:
        return None, None
    value = payload.get("field_value")
    return field, str(value) if value is not None else NULL_TOKEN


def collect_sentencepiece_corpus(details: Iterable[dict[str, Any]]) -> list[str]:
    corpus: list[str] = []
    for detail in details:
        for token in _model_stream(detail):
            kind = str(token.get("token_kind") or "").upper()
            if kind == "TEXT":
                text = str(token.get("text") or "")
                if text:
                    corpus.append(text)
            elif kind != "STRUCTURE_REFERENCE":
                raise ValueError(f"unknown model text token kind {kind!r}")
    if not corpus:
        raise ValueError("static-v3 SentencePiece corpus is empty")
    return corpus


def _frozen_sentencepiece_config(config: dict[str, Any] | None) -> dict[str, Any]:
    frozen = dict(V3_SENTENCEPIECE_CONFIG)
    frozen["user_defined_symbols"] = list(V3_SENTENCEPIECE_CONFIG["user_defined_symbols"])
    if config is not None and config != frozen:
        raise ValueError("static-v3 SentencePiece configuration is frozen")
    return frozen


def train_sentencepiece(
    details: Iterable[dict[str, Any]],
    model_path: Path,
    *,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    try:
        import sentencepiece as sentencepiece
    except ImportError as exc:  # pragma: no cover - supplied by the formal kernel
        raise RuntimeError("sentencepiece is required by static-v3") from exc
    model_path = Path(model_path)
    if model_path.suffix != ".model":
        raise ValueError("SentencePiece output must end in .model")
    corpus = collect_sentencepiece_corpus(details)
    frozen = _frozen_sentencepiece_config(config)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_prefix = str(model_path.with_suffix(""))
    sentencepiece.SentencePieceTrainer.train(
        sentence_iterator=iter(corpus),
        model_prefix=model_prefix,
        **frozen,
    )
    if not model_path.is_file():
        raise RuntimeError(f"SentencePiece did not create {model_path}")
    model_bytes = model_path.read_bytes()
    corpus_bytes = "\n".join(corpus).encode("utf-8")
    config_bytes = json.dumps(frozen, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "tokenizer_type": "sentencepiece",
        "model_path": str(model_path),
        "model_sha256": hashlib.sha256(model_bytes).hexdigest(),
        "model_size_bytes": len(model_bytes),
        "corpus_sha256": hashlib.sha256(corpus_bytes).hexdigest(),
        "corpus_segment_count": len(corpus),
        "config": frozen,
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
    }


@lru_cache(maxsize=8)
def _load_sentencepiece_cached(model_path: str, expected_sha256: str) -> Any:
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"SentencePiece model does not exist: {path}")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if expected_sha256 and actual != expected_sha256:
        raise ValueError(f"SentencePiece hash mismatch: expected {expected_sha256}, got {actual}")
    try:
        import sentencepiece as sentencepiece
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("sentencepiece is required by static-v3") from exc
    processor = sentencepiece.SentencePieceProcessor()
    if not processor.load(str(path)):
        raise ValueError(f"could not load SentencePiece model {path}")
    return processor


def load_sentencepiece(model_path: Path, expected_sha256: str) -> Any:
    return _load_sentencepiece_cached(str(Path(model_path).resolve()), str(expected_sha256))


def _text_ids(schema: dict[str, Any], text: str) -> list[int]:
    contract = schema.get("tokenizer_contract") or {}
    if contract.get("tokenizer_type") == "sentencepiece":
        processor = load_sentencepiece(Path(str(contract["model_path"])), str(contract["model_sha256"]))
        return [int(value) for value in processor.encode(text, out_type=int)]
    if contract.get("tokenizer_type") != "lexical_compatibility_v0":
        raise ValueError(f"unsupported tokenizer contract {contract!r}")
    vocab = schema["text_vocab"]
    return [int(vocab.get(token, schema["text_unk_id"])) for token in text_tokens(text)]


def _mixed_text_length(detail: dict[str, Any], schema: dict[str, Any]) -> int:
    length = 0
    for token in _model_stream(detail):
        kind = str(token.get("token_kind") or "").upper()
        if kind == "TEXT":
            length += len(_text_ids(schema, str(token.get("text") or "")))
        elif kind == "STRUCTURE_REFERENCE":
            length += 1
        else:
            raise ValueError(f"unknown model text token kind {kind!r}")
    return length


def bind_sentencepiece(
    schema: dict[str, Any],
    details: Iterable[dict[str, Any]],
    tokenizer_contract: dict[str, Any],
) -> dict[str, Any]:
    if schema.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"expected schema {SCHEMA_VERSION}")
    frozen = _frozen_sentencepiece_config(tokenizer_contract.get("config"))
    processor = load_sentencepiece(Path(str(tokenizer_contract["model_path"])), str(tokenizer_contract["model_sha256"]))
    bound = dict(schema)
    bound["vocab"] = dict(schema["vocab"])
    bound["vocab_sizes"] = dict(schema["vocab_sizes"])
    text_vocab = {processor.id_to_piece(index): index for index in range(processor.get_piece_size())}
    bound["vocab"]["text"] = text_vocab
    bound["vocab_sizes"]["text"] = len(text_vocab)
    bound["text_vocab"] = text_vocab
    bound["text_vocab_size"] = len(text_vocab)
    bound["text_pad_id"] = int(processor.pad_id())
    bound["text_unk_id"] = int(processor.unk_id())
    bound["text_mask_id"] = int(processor.piece_to_id("[MASK_TEXT]"))
    bound["tokenizer_contract"] = {**tokenizer_contract, "config": frozen}
    maximum = 0
    for detail in details:
        length = _mixed_text_length(detail, bound)
        if length > V3_MAX_TEXT_SUBWORD_TOKENS:
            raise ValueError(
                f"card={detail.get('card_id')} detail={detail.get('global_detail_index')} "
                f"text_length={length} exceeds {V3_MAX_TEXT_SUBWORD_TOKENS}; truncation is forbidden"
            )
        maximum = max(maximum, length)
    bound["observed_max_text_subword_tokens"] = maximum
    bound["formal_tokenizer_ready"] = True
    return bound


def make_feature_schema(
    cards: list[dict[str, Any]],
    details: list[dict[str, Any]] | None = None,
    *,
    tokenizer_contract: dict[str, Any] | None = None,
    reference_type_enum: Iterable[str] | None = None,
) -> dict[str, Any]:
    details = details or []
    field_to_value_group = {field: field for field in V3_CARD_FIELD_SLOTS}
    for field in ("card_name", "evolves_from"):
        field_to_value_group[field] = "card_name"
    values_by_group: dict[str, list[str]] = defaultdict(list)
    for card in cards:
        for field in V3_CARD_FIELD_SLOTS:
            values_by_group[field_to_value_group[field]].append(_card_value(card, field))
        values_by_group["card_name"].extend(_evolves_to_names(card))
    value_vocabs = {group: _categorical_vocab(values) for group, values in values_by_group.items()}

    reference_types = [str(value) for value in (reference_type_enum or [])]
    reference_values: dict[str, list[str]] = defaultdict(list)
    lexical_tokens: set[str] = set()
    for detail in details:
        references = _reference_map(detail)
        for token in _model_stream(detail):
            kind = str(token.get("token_kind") or "").upper()
            if kind == "TEXT":
                lexical_tokens.update(text_tokens(str(token.get("text") or "")))
            elif kind == "STRUCTURE_REFERENCE":
                reference = references.get(int(token.get("reference_id", -1)), token)
                reference_type = reference.get("reference_type") or token.get("reference_type")
                if reference_type is None:
                    raise ValueError("structure reference is missing reference_type")
                reference_types.append(str(reference_type))
                field, value = _reference_field_value(reference)
                if field is not None and value is not None:
                    reference_values[field].append(value)
            else:
                raise ValueError(f"unknown model text token kind {kind!r}")

    reference_value_vocabs: dict[str, dict[str, int]] = {}
    for field in V3_REFERENCE_FIELDS:
        values = list(reference_values.get(field, []))
        if field in V3_CARD_FIELD_SLOTS:
            values.extend(value_vocabs[field_to_value_group[field]].keys())
        reference_value_vocabs[field] = _detail_vocab(values)

    text_vocab = {PAD_TOKEN: 0, "[MASK_TEXT]": 1, UNK_TOKEN: 2}
    for token in sorted(lexical_tokens):
        if token not in text_vocab:
            text_vocab[token] = len(text_vocab)
    detail_type_vocab = _detail_vocab(DETAIL_TYPES)
    damage_mode_vocab = _detail_vocab(DAMAGE_MODES)
    damage_value_vocab = _detail_vocab(
        detail.get("damage_value") for detail in details if detail.get("detail_type") == "ATTACK"
    )
    reference_type_vocab = _detail_vocab(reference_types)
    fingerprint_values = sorted({str(detail.get("detail_fingerprint")) for detail in details if detail.get("detail_fingerprint")})
    detail_fingerprint_vocab = {value: index + 1 for index, value in enumerate(fingerprint_values)}
    profile_max = max(
        [0] + [int(value) for card in cards for value in (card.get("provided_energy_counts") or {}).values()]
    )
    attack_max = max(
        [0]
        + [
            int((detail.get("cost_counts") or {}).get(symbol, 0))
            for detail in details
            if detail.get("detail_type") == "ATTACK"
            for symbol in V3_ATTACK_ENERGY_SYMBOLS
        ]
    )
    card_type_vocab = _categorical_vocab(card.get("card_type") for card in cards)
    vocab: dict[str, Any] = {
        **{field: value_vocabs[field_to_value_group[field]] for field in V3_CARD_FIELD_SLOTS},
        "detail_type": detail_type_vocab,
        "damage_mode": damage_mode_vocab,
        "damage_value": damage_value_vocab,
        "reference_type": reference_type_vocab,
        "text": text_vocab,
        "card_type": card_type_vocab,
    }
    tokenizer = dict(tokenizer_contract or {"tokenizer_type": "lexical_compatibility_v0"})
    schema = {
        "schema_version": SCHEMA_VERSION,
        "card_field_slots": list(V3_CARD_FIELD_SLOTS),
        "field_recovery_groups": [list(group) for group in V3_FIELD_RECOVERY_GROUPS],
        "field_to_value_group": field_to_value_group,
        "value_vocabs": value_vocabs,
        "vocab": vocab,
        "vocab_sizes": {name: len(values) for name, values in vocab.items()},
        "reference_fields": list(V3_REFERENCE_FIELDS),
        "reference_field_ids": {field: index + 1 for index, field in enumerate(V3_REFERENCE_FIELDS)},
        "reference_value_vocabs": reference_value_vocabs,
        "reference_type_vocab": reference_type_vocab,
        "text_token_kind_vocab": {value: index for index, value in enumerate(V3_TEXT_TOKEN_KINDS)},
        "text_vocab": text_vocab,
        "text_vocab_size": len(text_vocab),
        "text_pad_id": text_vocab[PAD_TOKEN],
        "text_mask_id": text_vocab["[MASK_TEXT]"],
        "text_unk_id": text_vocab[UNK_TOKEN],
        "tokenizer_contract": tokenizer,
        "max_text_subword_tokens": V3_MAX_TEXT_SUBWORD_TOKENS,
        "energy_symbols": list(V3_ENERGY_SYMBOLS),
        "attack_energy_symbols": list(V3_ATTACK_ENERGY_SYMBOLS),
        "profile_energy_count_max": profile_max,
        "profile_energy_count_mask_id": profile_max + 1,
        "profile_energy_count_vocab_size": profile_max + 2,
        "card_numeric_scales": {
            "hp": max(1, max((int(card.get("printed_hp") or 0) for card in cards), default=1)),
            "retreat": max(1, max((int(card.get("retreat") or 0) for card in cards), default=1)),
        },
        "attack_energy_count_max": attack_max,
        "attack_energy_count_vocab_size": attack_max + 1,
        "detail_fingerprint_vocab": detail_fingerprint_vocab,
        "max_details": max((int(card["detail_end"]) - int(card["detail_start"]) for card in cards), default=0),
        "model_input_batch_fields": [
            "card_field_value_ids",
            "card_field_applicability_mask",
            "card_kind_route_ids",
            "card_numeric_values",
            "evolves_to_name_ids",
            "evolves_to_name_mask",
            "provided_energy_count_ids",
            "detail_type_ids",
            "attack_energy_count_ids",
            "attack_damage_value_ids",
            "attack_damage_mode_ids",
            "detail_text_ids",
            "detail_text_token_kind_ids",
            "detail_text_token_mask",
            "detail_structure_reference_type_ids",
            "detail_structure_reference_field_ids",
            "detail_structure_reference_value_ids",
            "detail_mask",
            "same_card_detail_reference_matrix",
        ],
        "non_model_fields": [
            "reference_id",
            "name_raw",
            "name_normalized",
            "name_id",
            "previous_species",
            "rule_flags",
            "card_tags",
            "hp_applicability",
            "provided_energy_types",
        ],
    }
    if tokenizer.get("tokenizer_type") == "sentencepiece":
        schema = bind_sentencepiece(schema, details, tokenizer)
    else:
        maximum = max((_mixed_text_length(detail, schema) for detail in details), default=0)
        if maximum > V3_MAX_TEXT_SUBWORD_TOKENS:
            raise ValueError(f"compatibility token length {maximum} exceeds {V3_MAX_TEXT_SUBWORD_TOKENS}")
        schema["observed_max_text_subword_tokens"] = maximum
    return schema


class CardDataset(Dataset):
    def __init__(
        self,
        cards: list[dict[str, Any]],
        details: list[dict[str, Any]],
        detail_offsets: list[int],
        card_id_to_index: dict[str, int],
        schema: dict[str, Any],
        indices: list[int] | None = None,
    ) -> None:
        self.cards = cards
        self.records = cards
        self.details = details
        self.detail_offsets = detail_offsets
        self.card_id_to_index = card_id_to_index
        self.schema = schema
        self.indices = list(range(len(cards))) if indices is None else list(indices)

    @classmethod
    def from_cache(cls, cache_dir: Path, rebuild: bool = False) -> "CardDataset":
        cards, details, offsets, mapping, manifest = load_or_create_corpus(Path(cache_dir), rebuild=rebuild)
        if manifest.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"expected {SCHEMA_VERSION} cache at {cache_dir}")
        schema = make_feature_schema(
            cards,
            details,
            tokenizer_contract=manifest.get("tokenizer_contract"),
            reference_type_enum=(manifest.get("feature_schema") or {}).get("allowed_reference_types"),
        )
        return cls(cards, details, offsets, mapping, schema)

    def subset(self, indices: list[int]) -> "CardDataset":
        return CardDataset(
            self.cards,
            self.details,
            self.detail_offsets,
            self.card_id_to_index,
            self.schema,
            indices,
        )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, Any]:
        index = self.indices[item]
        start, end = self.detail_offsets[index : index + 2]
        card = self.cards[index]
        return {
            "index": index,
            "card_id": card["card_id"],
            "card": card,
            "record": card,
            "details": self.details[start:end],
            "schema": self.schema,
        }

    def relation_samples(self) -> dict[str, list[tuple[int, int]]]:
        """Return only same-name and printed evolution relations allowed by v3."""

        allowed = set(self.indices)
        name_to_indices: dict[str, list[int]] = defaultdict(list)
        for index in self.indices:
            name_to_indices[str(self.cards[index]["card_name_normalized"])].append(index)
        same_name = sorted(
            (left, right)
            for indices in name_to_indices.values()
            for left in indices
            for right in indices
            if left != right
        )
        evolves_to: set[tuple[int, int]] = set()
        for child in self.indices:
            parent_name = self.cards[child].get("evolves_from_card_name")
            for parent in name_to_indices.get(str(parent_name), []):
                if parent in allowed:
                    evolves_to.add((parent, child))
        for parent in self.indices:
            for child_name in _evolves_to_names(self.cards[parent]):
                for child in name_to_indices.get(child_name, []):
                    if child in allowed:
                        evolves_to.add((parent, child))
        return {"same_name": same_name, "evolves_to": sorted(evolves_to)}


def _value_id(schema: dict[str, Any], field: str, value: Any) -> int:
    group = schema["field_to_value_group"][field]
    vocab = schema["value_vocabs"][group]
    token = NULL_TOKEN if value is None else str(value)
    return int(vocab.get(token, vocab[UNK_TOKEN]))


def _detail_value_id(schema: dict[str, Any], field: str, value: Any) -> int:
    vocab = schema["vocab"][field]
    token = NULL_TOKEN if value is None else str(value)
    return int(vocab.get(token, vocab[UNK_TOKEN]))


def _encode_detail_stream(detail: dict[str, Any], schema: dict[str, Any]) -> list[tuple[str, Any]]:
    references = _reference_map(detail)
    encoded: list[tuple[str, Any]] = []
    for token in _model_stream(detail):
        kind = str(token.get("token_kind") or "").upper()
        if kind == "TEXT":
            encoded.extend(("TEXT", token_id) for token_id in _text_ids(schema, str(token.get("text") or "")))
        elif kind == "STRUCTURE_REFERENCE":
            reference_id = int(token.get("reference_id", -1))
            reference = references.get(reference_id)
            if reference is None:
                raise ValueError(f"detail {detail.get('global_detail_index')} has an unresolved reference token")
            encoded.append(("STRUCTURE_REFERENCE", reference))
        else:
            raise ValueError(f"unknown model text token kind {kind!r}")
    if len(encoded) > V3_MAX_TEXT_SUBWORD_TOKENS:
        raise ValueError(
            f"card={detail.get('card_id')} detail={detail.get('global_detail_index')} "
            f"text_length={len(encoded)} exceeds {V3_MAX_TEXT_SUBWORD_TOKENS}; truncation is forbidden"
        )
    return encoded


def collate_cards(items: list[dict[str, Any]], schema: dict[str, Any] | None = None) -> dict[str, Any]:
    if not items:
        raise ValueError("cannot collate an empty card batch")
    schema = schema or items[0].get("schema")
    if not isinstance(schema, dict) or schema.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"collate_cards requires schema {SCHEMA_VERSION}")
    if tuple(schema["card_field_slots"]) != V3_CARD_FIELD_SLOTS:
        raise ValueError("card field slot order changed")
    if tuple(schema["reference_fields"]) != V3_REFERENCE_FIELDS:
        raise ValueError("reference field order changed")

    cards = [item.get("card", item["record"]) for item in items]
    details_by_card = [list(item.get("details") or []) for item in items]
    batch_size = len(items)
    max_details = max(1, max(len(details) for details in details_by_card))
    max_evolves_to = max(1, max(len(_evolves_to_names(card)) for card in cards))

    field_value_ids = torch.empty((batch_size, len(V3_CARD_FIELD_SLOTS)), dtype=torch.long)
    field_applicability = torch.zeros_like(field_value_ids, dtype=torch.bool)
    for batch_index, card in enumerate(cards):
        for field_index, field in enumerate(V3_CARD_FIELD_SLOTS):
            field_value_ids[batch_index, field_index] = _value_id(schema, field, _card_value(card, field))
            field_applicability[batch_index, field_index] = _field_applicability(card, field)

    field_index = {field: index for index, field in enumerate(V3_CARD_FIELD_SLOTS)}
    card_numeric_values = torch.zeros((batch_size, 2), dtype=torch.float32)
    numeric_scales = schema["card_numeric_scales"]
    for batch_index, card in enumerate(cards):
        hp = _first_value(card, "printed_hp", "hp")
        retreat = _first_value(card, "retreat", "retreat_cost")
        if hp is not None:
            card_numeric_values[batch_index, 0] = float(hp) / float(numeric_scales["hp"])
        if retreat is not None:
            card_numeric_values[batch_index, 1] = float(retreat) / float(numeric_scales["retreat"])

    card_name_vocab = schema["value_vocabs"]["card_name"]
    evolves_to_name_ids = torch.zeros((batch_size, max_evolves_to), dtype=torch.long)
    evolves_to_name_mask = torch.zeros((batch_size, max_evolves_to), dtype=torch.bool)
    for batch_index, card in enumerate(cards):
        for position, name in enumerate(_evolves_to_names(card)):
            evolves_to_name_ids[batch_index, position] = int(card_name_vocab.get(name, card_name_vocab[UNK_TOKEN]))
            evolves_to_name_mask[batch_index, position] = True

    profile_max = int(schema["profile_energy_count_max"])
    provided_energy_count_ids = torch.zeros((batch_size, len(V3_ENERGY_SYMBOLS)), dtype=torch.long)
    for batch_index, card in enumerate(cards):
        counts = card.get("provided_energy_counts") or {}
        for symbol_index, symbol in enumerate(V3_ENERGY_SYMBOLS):
            count = int(counts.get(symbol, 0))
            if count < 0 or count > profile_max:
                raise ValueError(f"printed energy count {symbol}={count} is outside the frozen range")
            provided_energy_count_ids[batch_index, symbol_index] = count

    streams = [[_encode_detail_stream(detail, schema) for detail in details] for details in details_by_card]
    max_text_tokens = max(1, max((len(stream) for card_streams in streams for stream in card_streams), default=0))
    detail_shape = (batch_size, max_details)
    text_shape = (batch_size, max_details, max_text_tokens)
    detail_mask = torch.zeros(detail_shape, dtype=torch.bool)
    detail_type_ids = torch.zeros(detail_shape, dtype=torch.long)
    attack_mask = torch.zeros(detail_shape, dtype=torch.bool)
    attack_energy_count_ids = torch.zeros(
        (batch_size, max_details, len(V3_ATTACK_ENERGY_SYMBOLS)), dtype=torch.long
    )
    attack_damage_value_ids = torch.zeros(detail_shape, dtype=torch.long)
    attack_damage_mode_ids = torch.zeros(detail_shape, dtype=torch.long)
    detail_text_ids = torch.full(text_shape, int(schema["text_pad_id"]), dtype=torch.long)
    detail_text_token_kind_ids = torch.zeros(text_shape, dtype=torch.long)
    detail_text_token_mask = torch.zeros(text_shape, dtype=torch.bool)
    detail_plain_text_mask = torch.zeros(text_shape, dtype=torch.bool)
    detail_structure_reference_mask = torch.zeros(text_shape, dtype=torch.bool)
    detail_structure_reference_type_ids = torch.zeros(text_shape, dtype=torch.long)
    detail_structure_reference_field_ids = torch.zeros(text_shape, dtype=torch.long)
    detail_structure_reference_value_ids = torch.zeros(text_shape, dtype=torch.long)
    same_card_matrix = torch.zeros((batch_size, max_details, max_details), dtype=torch.bool)
    detail_fingerprint_ids = torch.zeros(detail_shape, dtype=torch.long)
    detail_global_indices = torch.full(detail_shape, -1, dtype=torch.long)
    detail_local_indices = torch.full(detail_shape, -1, dtype=torch.long)
    detail_metadata: list[dict[str, Any]] = []

    kind_vocab = schema["text_token_kind_vocab"]
    reference_type_vocab = schema["reference_type_vocab"]
    reference_field_ids = schema["reference_field_ids"]
    fingerprint_vocab = schema["detail_fingerprint_vocab"]
    attack_max = int(schema["attack_energy_count_max"])
    for batch_index, (card, details, card_streams) in enumerate(zip(cards, details_by_card, streams)):
        global_to_local = {
            int(detail.get("global_detail_index", local)): local for local, detail in enumerate(details)
        }
        metadata_rows: list[dict[str, Any]] = []
        for local_index, (detail, stream) in enumerate(zip(details, card_streams)):
            detail_mask[batch_index, local_index] = True
            detail_type = str(detail["detail_type"])
            detail_type_ids[batch_index, local_index] = _detail_value_id(schema, "detail_type", detail_type)
            global_index = int(detail.get("global_detail_index", local_index))
            detail_global_indices[batch_index, local_index] = global_index
            detail_local_indices[batch_index, local_index] = int(detail.get("local_detail_index", local_index))
            fingerprint = str(detail["detail_fingerprint"])
            detail_fingerprint_ids[batch_index, local_index] = int(fingerprint_vocab[fingerprint])

            if detail_type == "ATTACK":
                attack_mask[batch_index, local_index] = True
                costs = detail.get("cost_counts") or {}
                for symbol_index, symbol in enumerate(V3_ATTACK_ENERGY_SYMBOLS):
                    count = int(costs.get(symbol, 0))
                    if count < 0 or count > attack_max:
                        raise ValueError(f"attack energy count {symbol}={count} is outside the frozen range")
                    attack_energy_count_ids[batch_index, local_index, symbol_index] = count
                attack_damage_value_ids[batch_index, local_index] = _detail_value_id(
                    schema, "damage_value", detail.get("damage_value")
                )
                attack_damage_mode_ids[batch_index, local_index] = _detail_value_id(
                    schema, "damage_mode", detail.get("damage_mode")
                )

            for token_index, (kind, value) in enumerate(stream):
                detail_text_token_mask[batch_index, local_index, token_index] = True
                detail_text_token_kind_ids[batch_index, local_index, token_index] = int(kind_vocab[kind])
                if kind == "TEXT":
                    detail_text_ids[batch_index, local_index, token_index] = int(value)
                    detail_plain_text_mask[batch_index, local_index, token_index] = True
                else:
                    reference = value
                    reference_type = str(reference["reference_type"])
                    detail_structure_reference_mask[batch_index, local_index, token_index] = True
                    detail_structure_reference_type_ids[batch_index, local_index, token_index] = int(
                        reference_type_vocab.get(reference_type, reference_type_vocab[UNK_TOKEN])
                    )
                    field, field_value = _reference_field_value(reference)
                    if field is not None and field_value is not None:
                        field_id = int(reference_field_ids[field])
                        value_vocab = schema["reference_value_vocabs"][field]
                        detail_structure_reference_field_ids[batch_index, local_index, token_index] = field_id
                        detail_structure_reference_value_ids[batch_index, local_index, token_index] = int(
                            value_vocab.get(field_value, value_vocab[UNK_TOKEN])
                        )
                    payload = reference.get("payload") if isinstance(reference.get("payload"), dict) else {}
                    target_global = payload.get("target_global_detail_index")
                    if (
                        reference_type == "SAME_CARD_DETAIL_REF"
                        and target_global is not None
                        and int(target_global) in global_to_local
                    ):
                        same_card_matrix[batch_index, local_index, global_to_local[int(target_global)]] = True

            metadata_rows.append(
                {
                    "global_detail_index": global_index,
                    "local_detail_index": int(detail.get("local_detail_index", local_index)),
                    "detail_type": detail_type,
                    "name_raw": detail.get("name_raw"),
                    "name_normalized": detail.get("name_normalized"),
                    "name_id": detail.get("name_id"),
                    "detail_fingerprint": fingerprint,
                    "text_references": detail.get("text_references") or [],
                }
            )
        detail_metadata.append({"card_id": str(card["card_id"]), "details": metadata_rows})

    card_type_vocab = schema["vocab"]["card_type"]
    card_type_ids = torch.tensor(
        [int(card_type_vocab.get(str(card.get("card_type")), card_type_vocab[UNK_TOKEN])) for card in cards],
        dtype=torch.long,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "card_index": torch.tensor([int(item["index"]) for item in items], dtype=torch.long),
        "card_ids": [str(item["card_id"]) for item in items],
        "card_type_ids": card_type_ids,
        "card_field_value_ids": field_value_ids,
        "card_field_applicability_mask": field_applicability,
        "card_kind_route_ids": field_value_ids[:, field_index["card_kind"]].clone(),
        "card_numeric_values": card_numeric_values,
        "card_name_ids": field_value_ids[:, field_index["card_name"]],
        "evolves_from_name_ids": field_value_ids[:, field_index["evolves_from"]],
        "evolves_to_name_ids": evolves_to_name_ids,
        "evolves_to_name_mask": evolves_to_name_mask,
        "provided_energy_count_ids": provided_energy_count_ids,
        "detail_counts": detail_mask.sum(dim=1).long(),
        "detail_mask": detail_mask,
        "detail_type_ids": detail_type_ids,
        "attack_mask": attack_mask,
        "attack_energy_count_ids": attack_energy_count_ids,
        "attack_damage_value_ids": attack_damage_value_ids,
        "attack_damage_mode_ids": attack_damage_mode_ids,
        "detail_text_ids": detail_text_ids,
        "detail_text_token_kind_ids": detail_text_token_kind_ids,
        "detail_text_token_mask": detail_text_token_mask,
        "detail_plain_text_mask": detail_plain_text_mask,
        "detail_structure_reference_mask": detail_structure_reference_mask,
        "detail_structure_reference_type_ids": detail_structure_reference_type_ids,
        "detail_structure_reference_field_ids": detail_structure_reference_field_ids,
        "detail_structure_reference_value_ids": detail_structure_reference_value_ids,
        "same_card_detail_reference_matrix": same_card_matrix,
        "detail_fingerprint_ids": detail_fingerprint_ids,
        "detail_global_indices": detail_global_indices,
        "detail_local_indices": detail_local_indices,
        "detail_metadata": detail_metadata,
    }


def split_indices(count: int, val_ratio: float, seed: int, mode: str) -> tuple[list[int], list[int]]:
    if mode not in {"sample", "card_id"}:
        raise ValueError("split mode must be sample or card_id")
    indices = list(range(count))
    random.Random(seed).shuffle(indices)
    validation_count = max(1, int(count * val_ratio))
    return sorted(indices[validation_count:]), sorted(indices[:validation_count])


def split_record_indices(
    records: list[dict[str, Any]],
    val_ratio: float,
    seed: int,
    mode: str,
) -> tuple[list[int], list[int]]:
    if mode == "sample":
        return split_indices(len(records), val_ratio, seed, mode)
    if mode != "card_id":
        raise ValueError("split mode must be sample or card_id")
    groups: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        groups[str(record.get("card_id", index))].append(index)
    group_values = list(groups.values())
    random.Random(seed).shuffle(group_values)
    target = max(1, int(len(records) * val_ratio))
    validation: list[int] = []
    train: list[int] = []
    for indices in group_values:
        (validation if len(validation) < target else train).extend(indices)
    return sorted(train), sorted(validation)
