from __future__ import annotations

import csv
import hashlib
import importlib
import io
import json
import os
import re
import shutil
import sys
import tempfile
import unicodedata
import zipfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
LEGACY_V2_CACHE_DIR = ROOT / "artifacts" / "card_data_v2"
DEFAULT_CACHE_DIR = ROOT / "artifacts" / "card_data_v3"
TEXT_REFERENCE_OVERRIDES_PATH = ROOT / "configs" / "text_reference_overrides.jsonl"
SCHEMA_VERSION = "static_card_v3"
LEGACY_V2_SCHEMA_VERSION = "static_card_v2"
TEXT_REFERENCE_OVERRIDE_SCHEMA_VERSION = "static_card_text_reference_override_v1"

NULL_TOKEN = "<NULL>"
UNK_TOKEN = "<UNK>"
MASK_TOKEN = "<MASK>"
PAD_TOKEN = "<PAD>"

ENERGY_TYPES = ["C", "G", "R", "W", "L", "P", "F", "D", "M", "N", "Y", "A"]
PROVIDED_ENERGY_TYPES = [*ENERGY_TYPES, "TEAM_ROCKET"]
V3_REFERENCE_FIELDS = [
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
    "owner",
    "zone",
    "number",
    "energy_type",
    "detail_name",
]
FIELD_SELECTOR_REFERENCE_TYPES = {
    f"{field.upper()}_SELECTOR": field for field in V3_REFERENCE_FIELDS
}
RELATION_REFERENCE_TYPES = {
    "SELF_DETAIL_REF",
    "SAME_CARD_DETAIL_REF",
    "CROSS_CARD_ATTACK_REF",
    "CROSS_CARD_ABILITY_REF",
    "CROSS_CARD_CARD_EFFECT_REF",
    "EXACT_CARD_NAME_REF",
    "SAME_NAME_ATTACK_SELECTOR",
    "NAME_FRAGMENT_SELECTOR",
}
ALLOWED_REFERENCE_TYPES = set(FIELD_SELECTOR_REFERENCE_TYPES) | RELATION_REFERENCE_TYPES
ENERGY_ALIASES = {
    "0": "C",
    "1": "G",
    "2": "R",
    "3": "W",
    "4": "L",
    "5": "P",
    "6": "F",
    "7": "D",
    "8": "M",
    "9": "N",
    "10": "Y",
    "11": "A",
    "COLORLESS": "C",
    "GRASS": "G",
    "FIRE": "R",
    "WATER": "W",
    "LIGHTNING": "L",
    "PSYCHIC": "P",
    "FIGHTING": "F",
    "DARKNESS": "D",
    "METAL": "M",
    "DRAGON": "N",
    "FAIRY": "Y",
    "ANY": "A",
    "RAINBOW": "A",
    "TEAM ROCKET": "TEAM_ROCKET",
    "TEAM_ROCKET": "TEAM_ROCKET",
    "竜": "N",
}

SOURCE_COLUMNS = [
    "Card ID",
    "Card Name",
    "Expansion",
    "Collection No.",
    "Stage (Pokémon)/Type (Energy and Trainer)",
    "Rule",
    "Category",
    "Previous stage",
    "HP",
    "Type",
    "Weakness",
    "Resistance (Type)",
    "Retreat",
    "Move Name",
    "Cost",
    "Damage",
    "Effect Explanation",
]
CARD_LEVEL_COLUMNS = SOURCE_COLUMNS[1:13]
DETAIL_COLUMNS = ["Move Name", "Cost", "Damage", "Effect Explanation"]

KIND_SCHEMA: dict[str, tuple[str, str, str | None, str | None, str | None]] = {
    "Basic Pokémon": ("POKEMON", "POKEMON", "BASIC", None, None),
    "Stage 1 Pokémon": ("POKEMON", "POKEMON", "STAGE1", None, None),
    "Stage 2 Pokémon": ("POKEMON", "POKEMON", "STAGE2", None, None),
    "Item": ("TRAINER", "ITEM", None, "ITEM", None),
    "Pokémon Tool": ("TRAINER", "TOOL", None, "TOOL", None),
    "Supporter": ("TRAINER", "SUPPORTER", None, "SUPPORTER", None),
    "Stadium": ("TRAINER", "STADIUM", None, "STADIUM", None),
    "Basic Energy": ("ENERGY", "BASIC_ENERGY", None, None, "BASIC"),
    "Special Energy": ("ENERGY", "SPECIAL_ENERGY", None, None, "SPECIAL"),
}
PRINTED_CLASS_BY_KIND = {
    "Basic Pokémon": "BASIC_POKEMON",
    "Stage 1 Pokémon": "STAGE1_POKEMON",
    "Stage 2 Pokémon": "STAGE2_POKEMON",
    "Item": "ITEM",
    "Pokémon Tool": "POKEMON_TOOL",
    "Supporter": "SUPPORTER",
    "Stadium": "STADIUM",
    "Basic Energy": "BASIC_ENERGY",
    "Special Energy": "SPECIAL_ENERGY",
}
CG_CARD_TYPES = {
    "POKEMON": 0,
    "ITEM": 1,
    "TOOL": 2,
    "SUPPORTER": 3,
    "STADIUM": 4,
    "BASIC_ENERGY": 5,
    "SPECIAL_ENERGY": 6,
}

EXPECTED_SOURCE_SHA256 = "a0ea63cf7adcb65d35436ce0eb390de6e2e35654a7c67c065a45f4abaa00f373"
EXPECTED_COUNTS = {
    "source_rows": 2022,
    "cards": 1267,
    "details": 2014,
    "ATTACK": 1556,
    "ABILITY": 223,
    "CARD_EFFECT": 235,
    "tools": 27,
    "fossils": 5,
    "tera_effects": 32,
}

# Card 979's CG table accidentally shifted the displayed attack names/text by the
# preceding Tera rule.  Its structural fields and engine ids still align as below.
KNOWN_ATTACK_ID_BINDINGS: dict[int, dict[str, int]] = {
    979: {"Orichalcum Fang": 1408, "Impact Blow": 1409},
}
# These two rows are Japanese text accidentally embedded in the English CSV.
# The source file hash pins the exact bad input; CG supplies the audited English
# wording.  Any new text mismatch remains fatal.
KNOWN_EFFECT_TEXT_CORRECTIONS = {
    (480, 678): "cg_english_effect_card480_attack678_v1",
    (481, 680): "cg_english_effect_card481_attack680_v1",
}


class CardPreprocessingError(ValueError):
    """Raised when source semantics cannot be proven without guessing."""


@dataclass(frozen=True)
class CardRecord:
    card_id: str
    card_name: str
    card_name_normalized: str
    card_name_id: int
    name: str
    card_kind: str
    card_kind_id: int
    card_category: str
    card_type: str
    card_type_id: int
    printed_class: str
    printed_class_id: int
    category_family: str
    category_family_id: int
    category_qualifier: str | None
    category_qualifier_id: int
    rule: str | None
    rule_id: int
    rule_family: str | None
    rule_family_id: int
    card_tags: list[str]
    stage: str | None
    trainer_subtype: str | None
    energy_subtype: str | None
    previous_species: str | None
    evolves_from_card_name: str | None
    evolves_to_card_name: str | None
    evolves_from_name_id: int
    evolves_to_card_names: list[str]
    evolves_to_name_ids: list[int]
    pokemon_type: str | None
    provided_energy_counts: dict[str, int]
    printed_hp: int | None
    hp_applicability: str
    retreat: int | None
    weakness_type: str | None
    resistance_type: str | None
    rule_flags: list[str]
    detail_start: int
    detail_end: int
    expansion: str
    collection_no: str
    source_kind: str
    source_category: str | None
    # Compatibility aliases for callers transitioning from static_detail_v1.
    subtype: str | None
    trainer_type: str | None
    hp: int | None
    retreat_cost: int | None
    evolves_from: str | None
    provided_energy_types: list[str]


@dataclass(frozen=True)
class DetailRecord:
    global_detail_index: int
    local_detail_index: int
    card_index: int
    card_id: str
    source_row: int
    detail_type: str
    name_raw: str
    name_normalized: str
    name_id: int
    effect_text_raw: str
    effect_text: str
    effect_text_override: dict[str, Any] | None
    model_text_tokens: list[dict[str, Any]]
    text_references: list[dict[str, Any]]
    detail_fingerprint: str
    cost_raw: str
    cost_mode: str
    cost_counts: dict[str, int]
    damage_raw: str
    damage_value: int | None
    energy_costs: dict[str, int]
    base_damage: int | None
    damage_mode: str
    attack_id: int | None


def stable_hash(text: str, modulo: int) -> int:
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % modulo


def normalize_missing(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text or text.lower() in {"n/a", "nan", "none", "-", "—"}:
        return ""
    return text


def normalize_space(value: Any) -> str:
    return re.sub(r"\s+", " ", normalize_missing(value)).strip()


def normalize_unicode_text(value: Any) -> str:
    """Return the canonical, human-readable Unicode form used by v3 metadata."""

    normalized = unicodedata.normalize("NFKC", normalize_missing(value))
    normalized = normalized.translate(str.maketrans({"’": "'", "‘": "'", "ʼ": "'"}))
    return re.sub(r"\s+", " ", normalized).strip()


def normalize_detail_name(value: Any) -> str:
    name = normalize_unicode_text(value)
    name = re.sub(r"^\[Ability\]\s*", "", name, flags=re.IGNORECASE)
    if name.casefold() == "[tera]":
        return "Tera"
    return name


def _build_metadata_vocab(values: Iterable[str | None]) -> tuple[dict[str, int], list[str]]:
    normalized: set[str] = set()
    for value in values:
        if value is None:
            continue
        text = unicodedata.normalize("NFKC", str(value))
        text = text.translate(str.maketrans({"’": "'", "‘": "'", "ʼ": "'"}))
        text = re.sub(r"\s+", " ", text).strip()
        if text:
            normalized.add(text)
    ordered = [NULL_TOKEN, UNK_TOKEN, *sorted(normalized)]
    return {value: index for index, value in enumerate(ordered)}, ordered


def parse_int(value: Any) -> int | None:
    text = normalize_missing(value)
    if not text:
        return None
    if not re.fullmatch(r"-?\d+", text):
        raise CardPreprocessingError(f"expected an integer, got {text!r}")
    return int(text)


def parse_damage(value: Any) -> int | None:
    text = normalize_missing(value)
    if not text:
        return None
    match = re.fullmatch(r"-?(\d+)(?:\+|[xX×])?", text)
    if match is None:
        raise CardPreprocessingError(f"unrecognized damage value {text!r}")
    return int(match.group(1))


def parse_damage_mode(value: Any) -> str:
    text = normalize_missing(value)
    if not text:
        return "NONE"
    if re.fullmatch(r"-\d+", text):
        return "MINUS"
    if re.fullmatch(r"\d+\+", text):
        return "PLUS"
    if re.fullmatch(r"\d+[xX×]", text):
        return "MULTIPLY"
    if re.fullmatch(r"\d+", text):
        return "FIXED"
    raise CardPreprocessingError(f"unrecognized damage mode {text!r}")


def normalize_energy_symbol(symbol: Any) -> str:
    clean = normalize_space(symbol).upper().replace("-", "_")
    normalized = ENERGY_ALIASES.get(clean, clean)
    if normalized not in PROVIDED_ENERGY_TYPES:
        raise CardPreprocessingError(f"unknown energy symbol {symbol!r}")
    return normalized


def _parse_symbol_counts(value: Any, *, allow_bullets: bool, allow_no_cost: bool) -> dict[str, int]:
    text = normalize_missing(value)
    if not text:
        return {}
    if allow_no_cost and text.casefold() == "no cost":
        return {}
    counts: Counter[str] = Counter()
    consumed = [False] * len(text)
    for match in re.finditer(r"\{([^{}]+)\}", text):
        counts[normalize_energy_symbol(match.group(1))] += 1
        for index in range(match.start(), match.end()):
            consumed[index] = True
    if text == "竜":
        return {"N": 1}
    for index, char in enumerate(text):
        if char == "●" and allow_bullets:
            counts["C"] += 1
            consumed[index] = True
        elif char.isspace():
            consumed[index] = True
    remainder = "".join(char for index, char in enumerate(text) if not consumed[index])
    if remainder:
        raise CardPreprocessingError(f"unparsed energy expression {text!r}; remainder={remainder!r}")
    return dict(sorted(counts.items()))


def parse_energy_symbols(value: Any) -> list[str]:
    counts = _parse_symbol_counts(value, allow_bullets=False, allow_no_cost=False)
    return [symbol for symbol, count in counts.items() for _ in range(count)]


def energy_cost_dict(value: Any) -> dict[str, int]:
    return _parse_symbol_counts(value, allow_bullets=True, allow_no_cost=True)


def normalize_energy_costs(costs: dict[str, int]) -> dict[str, int]:
    normalized: Counter[str] = Counter()
    for key, value in costs.items():
        symbol = normalize_energy_symbol(key)
        count = int(value)
        if count < 0:
            raise CardPreprocessingError(f"negative energy count for {key!r}")
        normalized[symbol] += count
    return dict(sorted(normalized.items()))


def split_flags(value: Any) -> list[str]:
    text = normalize_missing(value)
    if not text:
        return []
    return [part.strip() for part in re.split(r"[,;/]+", text) if part.strip()]


def card_type_from_kind(kind: str) -> str:
    try:
        return KIND_SCHEMA[kind][1]
    except KeyError as exc:
        raise CardPreprocessingError(f"unknown source card kind {kind!r}") from exc


def stage_from_kind(kind: str) -> str | None:
    try:
        return KIND_SCHEMA[kind][2]
    except KeyError as exc:
        raise CardPreprocessingError(f"unknown source card kind {kind!r}") from exc


def trainer_type_from_card_type(card_type: str) -> str | None:
    return card_type if card_type in {"ITEM", "TOOL", "SUPPORTER", "STADIUM"} else None


def effect_source_type(card_type: str, rule_flag: str | None = None) -> str:
    if rule_flag:
        return "rule_box"
    return {
        "ITEM": "item",
        "TOOL": "tool",
        "SUPPORTER": "supporter",
        "STADIUM": "stadium",
        "SPECIAL_ENERGY": "special_energy",
    }.get(card_type, "other")


def normalize_card_tags(raw_category: Any) -> list[str]:
    value = normalize_missing(raw_category)
    if not value:
        return []
    direct = {
        "Ancient": ["ANCIENT"],
        "Future": ["FUTURE"],
        "Fossil": ["FOSSIL"],
        "Technical Machine": ["TECHNICAL_MACHINE"],
    }
    if value in direct:
        return direct[value]
    tera = re.fullmatch(r"Tera\(([^)]+)\)", value)
    if tera:
        tera_type = re.sub(r"[^A-Z0-9]+", "_", tera.group(1).upper()).strip("_")
        return ["TERA", f"TERA_TYPE_{tera_type}"]
    owner = re.fullmatch(r"Trainer's Pokémon[（(]([^）)]+)[）)]", value)
    if owner:
        owner_name = re.sub(r"[^A-Z0-9]+", "_", owner.group(1).upper()).strip("_")
        return ["TRAINERS_POKEMON", f"TRAINER_OWNER_{owner_name}"]
    raise CardPreprocessingError(f"unknown Category tag {value!r}")


def normalize_rule_flags(raw_rule: Any, card_tags: Iterable[str]) -> list[str]:
    mapping = {
        "Pokémon ex": "POKEMON_EX",
        "Mega Pokémon ex": "MEGA_POKEMON_EX",
        "ACE SPEC": "ACE_SPEC",
    }
    result: set[str] = set()
    for flag in split_flags(raw_rule):
        try:
            result.add(mapping[flag])
        except KeyError as exc:
            raise CardPreprocessingError(f"unknown Rule flag {flag!r}") from exc
    if "TERA" in set(card_tags):
        result.add("TERA")
    return sorted(result)


def normalize_card_kind(raw_kind: Any) -> str:
    kind = normalize_unicode_text(raw_kind)
    if kind not in KIND_SCHEMA:
        raise CardPreprocessingError(f"unknown source card kind {kind!r}")
    return KIND_SCHEMA[kind][0]


def normalize_rule_and_family(raw_rule: Any) -> tuple[str | None, str | None]:
    value = normalize_unicode_text(raw_rule)
    if not value:
        return "NONE", "NONE"
    mapping = {
        "Pokémon ex": ("POKEMON_EX", "POKEMON_EX"),
        "Mega Pokémon ex": ("MEGA_POKEMON_EX", "POKEMON_EX"),
        "ACE SPEC": ("ACE_SPEC", "ACE_SPEC"),
    }
    try:
        return mapping[value]
    except KeyError as exc:
        raise CardPreprocessingError(f"unknown Rule value {value!r}") from exc


def category_family_and_qualifier(
    card_category: str,
    card_tags: Iterable[str],
) -> tuple[str, str | None]:
    tags = list(card_tags)
    if "FOSSIL" in tags:
        return "FOSSIL", "NONE"
    if "TECHNICAL_MACHINE" in tags:
        return "TECHNICAL_MACHINE", "NONE"
    if "TERA" in tags:
        return "TERA", next(tag.removeprefix("TERA_TYPE_") for tag in tags if tag.startswith("TERA_TYPE_"))
    if "TRAINERS_POKEMON" in tags:
        return "TRAINERS_POKEMON", next(
            tag.removeprefix("TRAINER_OWNER_") for tag in tags if tag.startswith("TRAINER_OWNER_")
        )
    if "ANCIENT" in tags:
        return "ANCIENT", "NONE"
    if "FUTURE" in tags:
        return "FUTURE", "NONE"
    if tags:
        raise CardPreprocessingError(f"unmapped category tags {tags!r}")
    return "NONE", "NONE"


def canonical_species_name(card_name: str) -> str:
    name = normalize_space(card_name)
    name = re.sub(r"^.+?[\u2019']s\s+", "", name)
    name = re.sub(r"^Mega\s+", "", name)
    name = re.sub(r"\s+ex$", "", name)
    if not name:
        raise CardPreprocessingError(f"could not derive species from {card_name!r}")
    return name


def read_csv_from_zip(zip_path: Path, member: str) -> tuple[list[dict[str, str]], bytes]:
    with zipfile.ZipFile(zip_path) as zf:
        payload = zf.read(member)
    text = io.StringIO(payload.decode("utf-8-sig"), newline="")
    return list(csv.DictReader(text)), payload


def find_card_csv() -> Path | None:
    candidates = [
        ROOT / "EN_Card_Data.csv",
        ROOT / "kaggle_extract" / "EN_Card_Data.csv",
        Path("/kaggle/input/pokemon-tcg-ai-battle/EN_Card_Data.csv"),
        Path("/kaggle/input/competitions/pokemon-tcg-ai-battle/EN_Card_Data.csv"),
    ]
    for path in candidates:
        if path.exists():
            return path
    if Path("/kaggle/input").exists():
        matches = sorted(Path("/kaggle/input").rglob("EN_Card_Data.csv"))
        if matches:
            return matches[0]
    return None


def load_csv_rows() -> tuple[list[dict[str, str]], str]:
    rows, source, _sha256 = load_csv_rows_with_hash()
    return rows, source


def load_csv_rows_with_hash() -> tuple[list[dict[str, str]], str, str]:
    csv_path = find_card_csv()
    if csv_path is not None:
        payload = csv_path.read_bytes()
        rows = list(csv.DictReader(io.StringIO(payload.decode("utf-8-sig"), newline="")))
        source = str(csv_path)
    else:
        zip_path = ROOT / "pokemon-tcg-ai-battle.zip"
        if not zip_path.exists():
            raise FileNotFoundError("EN_Card_Data.csv was not found in project, Kaggle input, or competition zip")
        rows, payload = read_csv_from_zip(zip_path, "EN_Card_Data.csv")
        source = f"{zip_path}::EN_Card_Data.csv"
    if not rows or list(rows[0]) != SOURCE_COLUMNS:
        actual = list(rows[0]) if rows else []
        raise CardPreprocessingError(f"source columns changed: {actual!r}")
    return rows, source, hashlib.sha256(payload).hexdigest()


def enum_name(value: Any) -> str | None:
    if value is None:
        return None
    if hasattr(value, "name"):
        return str(value.name)
    text = str(value)
    return text if text and text != "None" else None


def load_cg_data(*, required: bool = True) -> tuple[dict[int, Any], dict[int, Any]]:
    candidate_parents = [ROOT / "outputs", ROOT / "kaggle_cg_runtime_dataset"]
    if Path("/kaggle/input").exists():
        for cg_init in sorted(Path("/kaggle/input").rglob("cg/__init__.py")):
            candidate_parents.append(cg_init.parents[1])
        for cg_init in sorted(Path("/kaggle/input").rglob("sample_submission/sample_submission/cg/__init__.py")):
            candidate_parents.append(cg_init.parents[1])
    for cg_parent in candidate_parents:
        if cg_parent.exists() and str(cg_parent) not in sys.path:
            sys.path.insert(0, str(cg_parent))
    try:
        api = importlib.import_module("cg.api")
        cards = {int(card.cardId): card for card in api.all_card_data()}
        attacks = {int(attack.attackId): attack for attack in api.all_attack()}
    except Exception as exc:
        if required:
            raise CardPreprocessingError("CG data is required for strict attack-id reconciliation") from exc
        return {}, {}
    if required and (not cards or not attacks):
        raise CardPreprocessingError("CG returned empty card or attack tables")
    return cards, attacks


def _single_energy_symbol(value: Any, field: str, *, required: bool) -> str | None:
    counts = _parse_symbol_counts(value, allow_bullets=False, allow_no_cost=False)
    total = sum(counts.values())
    if total == 0 and not required:
        return None
    if total != 1:
        raise CardPreprocessingError(f"{field} expected one energy symbol, got {value!r}")
    return next(iter(counts))


def _cg_energy_cost(attack: Any) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for value in getattr(attack, "energies", []) or []:
        counts[normalize_energy_symbol(enum_name(value))] += 1
    return dict(sorted(counts.items()))


def _validate_and_bind_cg(
    cards: list[dict[str, Any]],
    details: list[dict[str, Any]],
    cg_cards: dict[int, Any],
    cg_attacks: dict[int, Any],
) -> None:
    source_ids = {int(card["card_id"]) for card in cards}
    if source_ids != set(cg_cards):
        raise CardPreprocessingError(
            f"CSV/CG card ids differ: csv_only={sorted(source_ids - set(cg_cards))[:10]}, "
            f"cg_only={sorted(set(cg_cards) - source_ids)[:10]}"
        )
    attack_details_by_card: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for detail in details:
        if detail["detail_type"] == "ATTACK":
            attack_details_by_card[int(detail["card_id"])].append(detail)
    bound_ids: set[int] = set()
    for card in cards:
        card_id = int(card["card_id"])
        cg_card = cg_cards[card_id]
        if int(getattr(cg_card, "cardType")) != CG_CARD_TYPES[card["card_type"]]:
            raise CardPreprocessingError(f"card {card_id} CSV/CG card type mismatch")
        if normalize_space(card["card_name"]) != normalize_space(getattr(cg_card, "name", "")):
            raise CardPreprocessingError(f"card {card_id} CSV/CG name mismatch")
        if card["printed_hp"] is not None and int(getattr(cg_card, "hp", 0) or 0) != int(card["printed_hp"]):
            raise CardPreprocessingError(f"card {card_id} CSV/CG HP mismatch")
        if card["retreat"] is not None and int(getattr(cg_card, "retreatCost", 0) or 0) != int(card["retreat"]):
            raise CardPreprocessingError(f"card {card_id} CSV/CG Retreat mismatch")
        if normalize_space(card.get("evolves_from_card_name")) != normalize_space(getattr(cg_card, "evolvesFrom", None)):
            raise CardPreprocessingError(f"card {card_id} CSV/CG evolution source mismatch")
        if card["card_category"] == "POKEMON":
            if card["pokemon_type"] != normalize_energy_symbol(enum_name(getattr(cg_card, "energyType", None))):
                raise CardPreprocessingError(f"card {card_id} CSV/CG Pokémon type mismatch")
            for source_field, cg_field in [("weakness_type", "weakness"), ("resistance_type", "resistance")]:
                cg_value = getattr(cg_card, cg_field, None)
                normalized_cg = normalize_energy_symbol(enum_name(cg_value)) if cg_value is not None else None
                if card[source_field] != normalized_cg:
                    raise CardPreprocessingError(f"card {card_id} CSV/CG {source_field} mismatch")
            expected_stage = {
                "BASIC": (True, False, False),
                "STAGE1": (False, True, False),
                "STAGE2": (False, False, True),
            }[card["stage"]]
            actual_stage = tuple(bool(getattr(cg_card, field, False)) for field in ("basic", "stage1", "stage2"))
            if actual_stage != expected_stage:
                raise CardPreprocessingError(f"card {card_id} CSV/CG stage mismatch")
        flag_checks = {
            "POKEMON_EX": "ex",
            "MEGA_POKEMON_EX": "megaEx",
            "TERA": "tera",
            "ACE_SPEC": "aceSpec",
        }
        for source_flag, cg_field in flag_checks.items():
            if (source_flag in card["rule_flags"]) != bool(getattr(cg_card, cg_field, False)):
                raise CardPreprocessingError(f"card {card_id} CSV/CG flag mismatch for {source_flag}")
        source_attacks = attack_details_by_card.get(card_id, [])
        cg_ids = [int(value) for value in (getattr(cg_card, "attacks", []) or [])]
        if len(source_attacks) != len(cg_ids):
            raise CardPreprocessingError(
                f"card {card_id} attack count mismatch: CSV={len(source_attacks)} CG={len(cg_ids)}"
            )
        explicit = KNOWN_ATTACK_ID_BINDINGS.get(card_id)
        if explicit is not None:
            if set(explicit) != {detail["name_raw"] for detail in source_attacks} or set(explicit.values()) != set(cg_ids):
                raise CardPreprocessingError(f"card {card_id} known attack-id resolution no longer matches source")
            pairs = [(detail, explicit[detail["name_raw"]]) for detail in source_attacks]
        else:
            pairs = list(zip(source_attacks, cg_ids))
        for detail, attack_id in pairs:
            if attack_id not in cg_attacks:
                raise CardPreprocessingError(f"card {card_id} references missing CG attack {attack_id}")
            attack = cg_attacks[attack_id]
            if explicit is None and normalize_space(detail["name_raw"]) != normalize_space(getattr(attack, "name", "")):
                raise CardPreprocessingError(
                    f"card {card_id} attack name mismatch for {attack_id}: "
                    f"CSV={detail['name_raw']!r}, CG={getattr(attack, 'name', None)!r}"
                )
            cg_cost = _cg_energy_cost(attack)
            if detail["cost_counts"] != cg_cost:
                raise CardPreprocessingError(
                    f"card {card_id} attack {attack_id} cost mismatch: CSV={detail['cost_counts']}, CG={cg_cost}"
                )
            cg_damage = int(getattr(attack, "damage", 0) or 0)
            normalized_cg_damage = abs(cg_damage) if detail["damage_mode"] == "MINUS" else cg_damage
            if (
                detail["damage_mode"] in {"FIXED", "PLUS", "MINUS"}
                and detail["damage_value"] != normalized_cg_damage
            ):
                raise CardPreprocessingError(
                    f"card {card_id} attack {attack_id} damage mismatch: "
                    f"CSV={detail['damage_value']}, CG={cg_damage}"
                )
            if detail["damage_mode"] == "NONE" and cg_damage != 0:
                raise CardPreprocessingError(f"card {card_id} effect-only attack {attack_id} has CG damage {cg_damage}")
            source_effect = normalize_unicode_text(detail["effect_text"])
            cg_effect = normalize_unicode_text(getattr(attack, "text", ""))
            if source_effect != cg_effect:
                if explicit is not None:
                    # Card 979's CG name/text fields are the known shifted fields;
                    # its CSV rule and attack text remain canonical.
                    pass
                elif (card_id, attack_id) in KNOWN_EFFECT_TEXT_CORRECTIONS:
                    detail["effect_text"] = normalize_unicode_text(getattr(attack, "text", ""))
                    detail["effect_text_override"] = {
                        "override_id": KNOWN_EFFECT_TEXT_CORRECTIONS[(card_id, attack_id)],
                        "reason": "english_csv_contains_japanese_attack_effect_text",
                        "source": "cg.api.attack.text",
                        "attack_id": attack_id,
                    }
                else:
                    raise CardPreprocessingError(
                        f"card {card_id} attack {attack_id} effect text mismatch outside known corrections"
                    )
            if attack_id in bound_ids:
                raise CardPreprocessingError(f"CG attack id {attack_id} was bound more than once")
            bound_ids.add(attack_id)
            detail["attack_id"] = attack_id
    if bound_ids != set(cg_attacks):
        raise CardPreprocessingError(
            f"not every CG attack was bound: bound={len(bound_ids)} cg={len(cg_attacks)}"
        )


def _serialized_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def _compact_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def load_text_reference_overrides(
    path: Path = TEXT_REFERENCE_OVERRIDES_PATH,
) -> tuple[list[dict[str, Any]], str | None]:
    path = Path(path)
    if not path.exists():
        return [], None
    payload = path.read_bytes()
    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for line_number, raw_line in enumerate(payload.decode("utf-8").splitlines(), start=1):
        if not raw_line.strip():
            continue
        try:
            record = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise CardPreprocessingError(f"invalid text-reference override JSON on line {line_number}") from exc
        required = {
            "schema_version",
            "override_id",
            "source_card_id",
            "source_row",
            "effect_text_sha256",
            "model_source_span",
            "action",
            "reference_type",
            "payload",
            "reason",
            "source",
        }
        if set(record) != required:
            raise CardPreprocessingError(
                f"text-reference override {record.get('override_id', line_number)!r} fields changed: "
                f"missing={sorted(required - set(record))}, extra={sorted(set(record) - required)}"
            )
        if record["schema_version"] != TEXT_REFERENCE_OVERRIDE_SCHEMA_VERSION:
            raise CardPreprocessingError(f"override line {line_number} has unknown schema_version")
        override_id = str(record["override_id"])
        if not override_id or override_id in seen_ids:
            raise CardPreprocessingError(f"duplicate or empty override_id {override_id!r}")
        seen_ids.add(override_id)
        if record["action"] not in {"FORCE_PLAIN_TEXT", "FORCE_REFERENCE"}:
            raise CardPreprocessingError(f"override {override_id!r} has unknown action {record['action']!r}")
        span = record["model_source_span"]
        if not (
            isinstance(span, list)
            and len(span) == 2
            and all(isinstance(value, int) and not isinstance(value, bool) for value in span)
            and 0 <= span[0] < span[1]
        ):
            raise CardPreprocessingError(f"override {override_id!r} has invalid model_source_span")
        reference_type = record["reference_type"]
        if record["action"] == "FORCE_REFERENCE" and reference_type not in ALLOWED_REFERENCE_TYPES:
            raise CardPreprocessingError(f"override {override_id!r} has unknown reference_type {reference_type!r}")
        if record["action"] == "FORCE_PLAIN_TEXT" and reference_type is not None:
            raise CardPreprocessingError(f"FORCE_PLAIN_TEXT override {override_id!r} must use null reference_type")
        if not isinstance(record["payload"], dict):
            raise CardPreprocessingError(f"override {override_id!r} payload must be an object")
        records.append(record)
    records.sort(key=lambda row: (int(row["source_card_id"]), int(row["source_row"]), str(row["override_id"])))
    return records, hashlib.sha256(payload).hexdigest()


def _validate_reference_payload(reference_type: str, payload: dict[str, Any]) -> None:
    if reference_type not in ALLOWED_REFERENCE_TYPES:
        raise CardPreprocessingError(f"unknown structure reference type {reference_type!r}")
    if reference_type in FIELD_SELECTOR_REFERENCE_TYPES:
        expected = {"field_name", "field_value"}
        if set(payload) != expected or payload["field_name"] != FIELD_SELECTOR_REFERENCE_TYPES[reference_type]:
            raise CardPreprocessingError(f"invalid selector payload for {reference_type}: {payload!r}")
        return
    if reference_type in {"SELF_DETAIL_REF", "SAME_CARD_DETAIL_REF"}:
        expected = {"target_global_detail_index", "target_detail_name_id"}
    elif reference_type == "EXACT_CARD_NAME_REF":
        expected = {"target_card_name_id", "matching_target_card_ids", "resolution_status"}
    elif reference_type == "NAME_FRAGMENT_SELECTOR":
        expected = {"fragment", "matching_target_card_ids", "resolution_status"}
    else:
        expected = {
            "target_card_name_id",
            "target_detail_name_id",
            "target_detail_type",
            "matching_target_card_ids",
            "matching_target_global_detail_indices",
            "resolution_status",
        }
    if set(payload) != expected:
        raise CardPreprocessingError(
            f"invalid relation payload for {reference_type}: missing={sorted(expected - set(payload))}, "
            f"extra={sorted(set(payload) - expected)}"
        )


def _candidate(
    *,
    start: int,
    end: int,
    reference_type: str,
    payload: dict[str, Any],
    priority: int,
    emitted_order: int,
    origin: str,
) -> dict[str, Any]:
    _validate_reference_payload(reference_type, payload)
    return {
        "start": start,
        "end": end,
        "reference_type": reference_type,
        "payload": payload,
        "priority": priority,
        "emitted_order": emitted_order,
        "origin": origin,
    }


def _cross_reference_payload(
    targets: list[dict[str, Any]],
    *,
    card_name_id: int = 0,
    detail_name_id: int = 0,
    detail_type: str | None = None,
) -> dict[str, Any]:
    card_ids = sorted({target["card_id"] for target in targets}, key=int)
    detail_indices = sorted(
        target["global_detail_index"] for target in targets if "global_detail_index" in target
    )
    return {
        "target_card_name_id": int(card_name_id),
        "target_detail_name_id": int(detail_name_id),
        "target_detail_type": detail_type,
        "matching_target_card_ids": card_ids,
        "matching_target_global_detail_indices": detail_indices,
        "resolution_status": "RESOLVED_UNIQUE" if len(targets) == 1 else "RESOLVED_MULTIPLE",
    }


def _raw_span(effect_text_raw: str, effect_text: str, start: int, end: int) -> list[int] | None:
    fragment = effect_text[start:end]
    if not fragment:
        return None
    matches = list(re.finditer(re.escape(fragment), effect_text_raw, flags=re.IGNORECASE))
    if len(matches) != 1:
        return None
    return [matches[0].start(), matches[0].end()]


def _reference_fingerprint(detail: dict[str, Any]) -> str:
    model_stream = [
        {"token_kind": "TEXT", "text": token["text"]}
        if token["token_kind"] == "TEXT"
        else {"token_kind": "STRUCTURE_REFERENCE", "reference_type": token["reference_type"]}
        for token in detail["model_text_tokens"]
    ]
    semantic = {
        "detail_type": detail["detail_type"],
        "cost_mode": detail["cost_mode"],
        "cost_counts": detail["cost_counts"],
        "damage_value": detail["damage_value"],
        "damage_mode": detail["damage_mode"],
        "model_text_tokens": model_stream,
    }
    return hashlib.sha256(_compact_json(semantic).encode("utf-8")).hexdigest()


def _finalize_v3_metadata_and_references(
    cards: list[dict[str, Any]],
    details: list[dict[str, Any]],
    *,
    overrides_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], str | None, list[str]]:
    evolves_to_by_name: dict[str, set[str]] = defaultdict(set)
    for card in cards:
        child_name = normalize_unicode_text(card["card_name"])
        parent_name = normalize_unicode_text(card.get("evolves_from_card_name"))
        if parent_name:
            evolves_to_by_name[parent_name].add(child_name)
        explicit_target = normalize_unicode_text(card.get("evolves_to_card_name"))
        if explicit_target:
            evolves_to_by_name[normalize_unicode_text(card["card_name"])].add(explicit_target)

    card_name_values: list[str | None] = []
    for card in cards:
        card_name_values.extend(
            [card["card_name"], card.get("evolves_from_card_name"), card.get("evolves_to_card_name")]
        )
        card_name_values.extend(evolves_to_by_name.get(normalize_unicode_text(card["card_name"]), set()))
    card_name_to_id, card_name_vocab = _build_metadata_vocab(card_name_values)
    detail_name_to_id, detail_name_vocab = _build_metadata_vocab(
        detail["name_normalized"] for detail in details
    )
    card_kind_to_id, card_kind_vocab = _build_metadata_vocab(card["card_kind"] for card in cards)
    card_type_to_id, card_type_vocab = _build_metadata_vocab(card["card_type"] for card in cards)
    printed_class_to_id, printed_class_vocab = _build_metadata_vocab(
        card["printed_class"] for card in cards
    )
    category_family_to_id, category_family_vocab = _build_metadata_vocab(
        card["category_family"] for card in cards
    )
    category_qualifier_to_id, category_qualifier_vocab = _build_metadata_vocab(
        card["category_qualifier"] for card in cards
    )
    rule_to_id, rule_vocab = _build_metadata_vocab(card["rule"] for card in cards)
    rule_family_to_id, rule_family_vocab = _build_metadata_vocab(card["rule_family"] for card in cards)

    for card in cards:
        normalized_name = normalize_unicode_text(card["card_name"])
        target_names = sorted(evolves_to_by_name.get(normalized_name, set()))
        card["card_name_normalized"] = normalized_name
        card["card_name_id"] = card_name_to_id[normalized_name]
        card["card_kind_id"] = card_kind_to_id[card["card_kind"]]
        card["card_type_id"] = card_type_to_id[card["card_type"]]
        card["printed_class_id"] = printed_class_to_id[card["printed_class"]]
        card["category_family_id"] = category_family_to_id[card["category_family"]]
        card["category_qualifier_id"] = category_qualifier_to_id[card["category_qualifier"]]
        card["rule_id"] = rule_to_id[card["rule"]]
        card["rule_family_id"] = rule_family_to_id[card["rule_family"]]
        parent_name = normalize_unicode_text(card.get("evolves_from_card_name"))
        card["evolves_from_name_id"] = card_name_to_id[parent_name] if parent_name else 0
        card["evolves_to_card_names"] = target_names
        card["evolves_to_name_ids"] = [card_name_to_id[target] for target in target_names]

    detail_by_index = {detail["global_detail_index"]: detail for detail in details}
    details_by_card: dict[str, list[dict[str, Any]]] = defaultdict(list)
    details_by_name: dict[str, list[dict[str, Any]]] = defaultdict(list)
    cards_by_name: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for card in cards:
        cards_by_name[card["card_name_normalized"].casefold()].append(card)
    for detail in details:
        detail["name_id"] = (
            detail_name_to_id.get(detail["name_normalized"], 1)
            if detail["name_normalized"]
            else 0
        )
        details_by_card[detail["card_id"]].append(detail)
        if detail["name_normalized"]:
            details_by_name[detail["name_normalized"].casefold()].append(detail)

    overrides, overrides_sha256 = load_text_reference_overrides(overrides_path)
    overrides_by_source: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for override in overrides:
        overrides_by_source[(str(override["source_card_id"]), int(override["source_row"]))].append(override)

    detail_names = sorted(
        {detail["name_normalized"] for detail in details if len(detail["name_normalized"]) >= 3},
        key=lambda value: (-len(value), value),
    )
    detail_name_pattern = (
        re.compile(r"(?<!\w)(?:" + "|".join(re.escape(value) for value in detail_names) + r")(?!\w)", re.IGNORECASE)
        if detail_names
        else None
    )
    card_names = sorted(
        {card["card_name_normalized"] for card in cards if len(card["card_name_normalized"]) >= 3},
        key=lambda value: (-len(value), value),
    )
    card_name_variants = [
        (
            re.escape(value[:-3]) + r"(?:\s+ex|\s+\{ex\})"
            if value.casefold().endswith(" ex")
            else re.escape(value)
        )
        for value in card_names
    ]
    card_name_pattern = (
        re.compile(r"(?<!\w)(?:" + "|".join(card_name_variants) + r")(?!\w)")
        if card_names
        else None
    )

    unresolved_audit: list[dict[str, Any]] = []
    selected_by_detail: dict[int, list[dict[str, Any]]] = {}
    emitted_order = 0
    zone_phrases = {
        "Active Spot": "ACTIVE_SPOT",
        "Bench": "BENCH",
        "discard pile": "DISCARD_PILE",
        "Lost Zone": "LOST_ZONE",
        "Prize cards": "PRIZE",
        "Prize card": "PRIZE",
    }
    fixed_field_phrases = [
        ("Mega Evolution Pokémon ex", "RULE_SELECTOR", "rule", "MEGA_POKEMON_EX"),
        ("Mega Pokémon ex", "RULE_SELECTOR", "rule", "MEGA_POKEMON_EX"),
        ("Trainer's Pokémon", "CATEGORY_FAMILY_SELECTOR", "category_family", "TRAINERS_POKEMON"),
        ("Evolution Pokémon", "CARD_KIND_SELECTOR", "card_kind", "EVOLUTION_POKEMON"),
        ("Stage 2 Pokémon", "PRINTED_CLASS_SELECTOR", "printed_class", "STAGE2_POKEMON"),
        ("Stage 1 Pokémon", "PRINTED_CLASS_SELECTOR", "printed_class", "STAGE1_POKEMON"),
        ("Basic Pokémon", "PRINTED_CLASS_SELECTOR", "printed_class", "BASIC_POKEMON"),
        ("Pokémon Tool", "PRINTED_CLASS_SELECTOR", "printed_class", "POKEMON_TOOL"),
        ("Special Energy", "PRINTED_CLASS_SELECTOR", "printed_class", "SPECIAL_ENERGY"),
        ("Basic Energy", "PRINTED_CLASS_SELECTOR", "printed_class", "BASIC_ENERGY"),
        ("Pokémon ex", "RULE_FAMILY_SELECTOR", "rule_family", "POKEMON_EX"),
        ("ACE SPEC", "RULE_SELECTOR", "rule", "ACE_SPEC"),
        ("Rule Box", "RULE_FAMILY_SELECTOR", "rule_family", "RULE_BOX"),
        ("Supporter", "PRINTED_CLASS_SELECTOR", "printed_class", "SUPPORTER"),
        ("Stadium", "PRINTED_CLASS_SELECTOR", "printed_class", "STADIUM"),
        ("Item", "PRINTED_CLASS_SELECTOR", "printed_class", "ITEM"),
        ("Tool", "PRINTED_CLASS_SELECTOR", "printed_class", "POKEMON_TOOL"),
        ("Trainer", "CARD_KIND_SELECTOR", "card_kind", "TRAINER"),
        ("Energy", "CARD_KIND_SELECTOR", "card_kind", "ENERGY"),
        ("Pokémon", "CARD_KIND_SELECTOR", "card_kind", "POKEMON"),
        ("Ancient", "CATEGORY_FAMILY_SELECTOR", "category_family", "ANCIENT"),
        ("Future", "CATEGORY_FAMILY_SELECTOR", "category_family", "FUTURE"),
        ("Tera", "CATEGORY_FAMILY_SELECTOR", "category_family", "TERA"),
    ]
    fixed_field_phrases.sort(key=lambda row: (-len(row[0]), row[0]))
    for detail in details:
        text = detail["effect_text"]
        candidates: list[dict[str, Any]] = []

        for phrase, reference_type, field_name, field_value in fixed_field_phrases:
            for match in re.finditer(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", text, re.IGNORECASE):
                candidates.append(
                    _candidate(
                        start=match.start(), end=match.end(), reference_type=reference_type,
                        payload={"field_name": field_name, "field_value": field_value}, priority=950,
                        emitted_order=emitted_order, origin="fixed_card_field_phrase",
                    )
                )
                emitted_order += 1

        for match in re.finditer(r"\{([^{}]+)\}", text):
            raw_value = match.group(1)
            normalized = ENERGY_ALIASES.get(raw_value.upper().replace("-", "_"))
            if normalized in PROVIDED_ENERGY_TYPES:
                candidates.append(
                    _candidate(
                        start=match.start(), end=match.end(), reference_type="ENERGY_TYPE_SELECTOR",
                        payload={"field_name": "energy_type", "field_value": normalized}, priority=900,
                        emitted_order=emitted_order, origin="fixed_energy_symbol",
                    )
                )
                emitted_order += 1
            elif raw_value.casefold() in {"ex", "v"}:
                candidates.append(
                    _candidate(
                        start=match.start(), end=match.end(), reference_type="RULE_FAMILY_SELECTOR",
                        payload={"field_name": "rule_family", "field_value": raw_value.upper()}, priority=900,
                        emitted_order=emitted_order, origin="fixed_rule_symbol",
                    )
                )
                emitted_order += 1

        for phrase, value in zone_phrases.items():
            for match in re.finditer(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", text, re.IGNORECASE):
                candidates.append(
                    _candidate(
                        start=match.start(), end=match.end(), reference_type="ZONE_SELECTOR",
                        payload={"field_name": "zone", "field_value": value}, priority=850,
                        emitted_order=emitted_order, origin="fixed_zone_phrase",
                    )
                )
                emitted_order += 1

        if detail_name_pattern is not None:
            for match in detail_name_pattern.finditer(text):
                normalized_match = normalize_detail_name(match.group(0)).casefold()
                targets = details_by_name.get(normalized_match, [])
                same_card = [target for target in targets if target["card_id"] == detail["card_id"]]
                suffix = text[match.end() : match.end() + 24]
                context = text[max(0, match.start() - 120) : min(len(text), match.end() + 32)]
                same_name_selector = bool(
                    re.match(r"\s+attack\b", suffix, re.IGNORECASE)
                    and re.search(r"(?:each|any|for each).{0,100}(?:has|have|with)\s+(?:the\s+)?", context, re.IGNORECASE)
                )
                if same_name_selector:
                    attack_targets = [target for target in targets if target["detail_type"] == "ATTACK"]
                    if attack_targets:
                        card_name_ids = {cards[target["card_index"]]["card_name_id"] for target in attack_targets}
                        payload = _cross_reference_payload(
                            attack_targets,
                            card_name_id=next(iter(card_name_ids)) if len(card_name_ids) == 1 else 0,
                            detail_name_id=attack_targets[0]["name_id"],
                            detail_type="ATTACK",
                        )
                        candidates.append(
                            _candidate(
                                start=match.start(), end=match.end(), reference_type="SAME_NAME_ATTACK_SELECTOR",
                                payload=payload, priority=760, emitted_order=emitted_order,
                                origin="catalog_same_name_attack_phrase",
                            )
                        )
                        emitted_order += 1
                    continue
                if same_card:
                    if len(same_card) == 1:
                        target = same_card[0]
                        reference_type = (
                            "SELF_DETAIL_REF"
                            if target["global_detail_index"] == detail["global_detail_index"]
                            else "SAME_CARD_DETAIL_REF"
                        )
                        candidates.append(
                            _candidate(
                                start=match.start(), end=match.end(), reference_type=reference_type,
                                payload={
                                    "target_global_detail_index": target["global_detail_index"],
                                    "target_detail_name_id": target["name_id"],
                                },
                                priority=800, emitted_order=emitted_order, origin="catalog_same_card_detail_name",
                            )
                        )
                        emitted_order += 1
                    else:
                        unresolved_audit.append({
                            "card_id": detail["card_id"], "source_row": detail["source_row"],
                            "source_global_detail_index": detail["global_detail_index"],
                            "model_source_span": [match.start(), match.end()], "candidate_text": match.group(0),
                            "reason": "ambiguous_same_card_detail_name",
                        })
                    continue
                suffix_type = None
                if re.match(r"\s+attack\b", suffix, re.IGNORECASE):
                    suffix_type = "ATTACK"
                elif re.match(r"\s+abilit(?:y|ies)\b", suffix, re.IGNORECASE):
                    suffix_type = "ABILITY"
                if suffix_type is None:
                    continue
                if any(target["detail_type"] == suffix_type for target in targets):
                    unresolved_audit.append({
                        "card_id": detail["card_id"], "source_row": detail["source_row"],
                        "source_global_detail_index": detail["global_detail_index"],
                        "model_source_span": [match.start(), match.end()], "candidate_text": match.group(0),
                        "reason": "cross_card_detail_name_without_proven_card_name",
                    })

        for fragment_match in re.finditer(
            r"[\"“](?P<fragment>[^\"”]{2,80})[\"”]\s+in\s+(?:its|their|the)\s+name",
            text,
            re.IGNORECASE,
        ):
            fragment = normalize_unicode_text(fragment_match.group("fragment"))
            matching_cards = [
                card for card in cards if fragment.casefold() in card["card_name_normalized"].casefold()
            ]
            if matching_cards:
                matching_card_ids = sorted({card["card_id"] for card in matching_cards}, key=int)
                payload = {
                    "fragment": fragment,
                    "matching_target_card_ids": matching_card_ids,
                    "resolution_status": (
                        "RESOLVED_UNIQUE" if len(matching_card_ids) == 1 else "RESOLVED_MULTIPLE"
                    ),
                }
                candidates.append(
                    _candidate(
                        start=fragment_match.start("fragment"), end=fragment_match.end("fragment"),
                        reference_type="NAME_FRAGMENT_SELECTOR", payload=payload, priority=720,
                        emitted_order=emitted_order, origin="quoted_catalog_name_fragment",
                    )
                )
                emitted_order += 1

        if card_name_pattern is not None:
            card_mentions = list(card_name_pattern.finditer(text))
            for card_match in card_mentions:
                normalized_mention = normalize_unicode_text(card_match.group(0))
                normalized_mention = re.sub(r"\{ex\}$", "ex", normalized_mention, flags=re.IGNORECASE)
                matching_cards = cards_by_name.get(normalized_mention.casefold(), [])
                if not matching_cards or detail_name_pattern is None:
                    continue
                for name_match in detail_name_pattern.finditer(
                    text,
                    card_match.end(),
                    min(len(text), card_match.end() + 160),
                ):
                    suffix = text[name_match.end() : name_match.end() + 24]
                    if re.match(r"\s+attack\b", suffix, re.IGNORECASE):
                        target_type = "ATTACK"
                    elif re.match(r"\s+abilit(?:y|ies)\b", suffix, re.IGNORECASE):
                        target_type = "ABILITY"
                    elif re.match(r"\s+(?:effect|rule)\b", suffix, re.IGNORECASE):
                        target_type = "CARD_EFFECT"
                    else:
                        continue
                    normalized_detail_name = normalize_detail_name(name_match.group(0)).casefold()
                    matching_card_ids = {card["card_id"] for card in matching_cards}
                    targets = [
                        target
                        for target in details_by_name.get(normalized_detail_name, [])
                        if target["card_id"] in matching_card_ids and target["detail_type"] == target_type
                    ]
                    if not targets:
                        unresolved_audit.append({
                            "card_id": detail["card_id"], "source_row": detail["source_row"],
                            "source_global_detail_index": detail["global_detail_index"],
                            "model_source_span": [card_match.start(), name_match.end()],
                            "candidate_text": text[card_match.start() : name_match.end()],
                            "reason": "card_and_detail_phrase_failed_catalog_ownership",
                        })
                        continue
                    reference_type = {
                        "ATTACK": "CROSS_CARD_ATTACK_REF",
                        "ABILITY": "CROSS_CARD_ABILITY_REF",
                        "CARD_EFFECT": "CROSS_CARD_CARD_EFFECT_REF",
                    }[target_type]
                    candidates.append(
                        _candidate(
                            start=card_match.start(), end=name_match.end(), reference_type=reference_type,
                            payload=_cross_reference_payload(
                                targets,
                                card_name_id=matching_cards[0]["card_name_id"],
                                detail_name_id=targets[0]["name_id"], detail_type=target_type,
                            ),
                            priority=820, emitted_order=emitted_order,
                            origin="catalog_card_name_detail_name_type_phrase",
                        )
                    )
                    emitted_order += 1

            for match in card_mentions:
                normalized_match = normalize_unicode_text(match.group(0))
                normalized_match = re.sub(r"\{ex\}$", "ex", normalized_match, flags=re.IGNORECASE).casefold()
                matching_cards = cards_by_name.get(normalized_match, [])
                if not matching_cards:
                    continue
                before = text[max(0, match.start() - 70) : match.start()]
                after = text[match.end() : match.end() + 45]
                proven_context = bool(
                    re.search(r"(?:search your deck for|if|when|while)\s+(?:an?\s+|up to \d+\s+)?$", before, re.IGNORECASE)
                    or re.match(r"\s+(?:is|are|was|were|card\b|in\s+your|on\s+your)", after, re.IGNORECASE)
                )
                if not proven_context:
                    continue
                matching_card_ids = sorted({card["card_id"] for card in matching_cards}, key=int)
                candidates.append(
                    _candidate(
                        start=match.start(), end=match.end(), reference_type="EXACT_CARD_NAME_REF",
                        payload={
                            "target_card_name_id": matching_cards[0]["card_name_id"],
                            "matching_target_card_ids": matching_card_ids,
                            "resolution_status": (
                                "RESOLVED_UNIQUE" if len(matching_card_ids) == 1 else "RESOLVED_MULTIPLE"
                            ),
                        },
                        priority=700, emitted_order=emitted_order, origin="catalog_exact_card_name_phrase",
                    )
                )
                emitted_order += 1

        source_overrides = overrides_by_source.pop((detail["card_id"], detail["source_row"]), [])
        plain_spans: list[tuple[int, int]] = []
        for override in source_overrides:
            expected_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if str(override["effect_text_sha256"]).lower() != expected_hash:
                raise CardPreprocessingError(f"override {override['override_id']!r} effect_text_sha256 drifted")
            start, end = override["model_source_span"]
            if end > len(text):
                raise CardPreprocessingError(f"override {override['override_id']!r} span exceeds canonical effect text")
            if override["action"] == "FORCE_PLAIN_TEXT":
                plain_spans.append((start, end))
            else:
                _validate_reference_payload(override["reference_type"], override["payload"])
                candidates.append(
                    _candidate(
                        start=start, end=end, reference_type=override["reference_type"],
                        payload=override["payload"], priority=1000, emitted_order=emitted_order,
                        origin=f"override:{override['override_id']}",
                    )
                )
                emitted_order += 1
        candidates = [
            candidate
            for candidate in candidates
            if not any(candidate["start"] < end and start < candidate["end"] for start, end in plain_spans)
        ]
        ranked = sorted(
            candidates,
            key=lambda row: (-row["priority"], -(row["end"] - row["start"]), row["start"], row["emitted_order"]),
        )
        selected: list[dict[str, Any]] = []
        for candidate in ranked:
            if any(candidate["start"] < row["end"] and row["start"] < candidate["end"] for row in selected):
                continue
            selected.append(candidate)
        selected_by_detail[detail["global_detail_index"]] = sorted(
            selected, key=lambda row: (row["start"], row["emitted_order"])
        )

    if overrides_by_source:
        missing = sorted(overrides_by_source)[:10]
        raise CardPreprocessingError(f"text-reference overrides point to missing card/source rows: {missing}")

    reference_order = sorted(
        (
            (detail_by_index[global_index]["card_index"], global_index, candidate["start"], candidate["emitted_order"], candidate)
            for global_index, candidates in selected_by_detail.items()
            for candidate in candidates
        ),
        key=lambda row: row[:4],
    )
    for reference_id, (_card_index, global_index, _span_start, _order, candidate) in enumerate(reference_order):
        candidate["reference_id"] = reference_id
        candidate["source_global_detail_index"] = global_index

    for detail in details:
        selected = selected_by_detail[detail["global_detail_index"]]
        references: list[dict[str, Any]] = []
        model_tokens: list[dict[str, Any]] = []
        cursor = 0
        for candidate in selected:
            if cursor < candidate["start"]:
                model_tokens.append({"token_kind": "TEXT", "text": detail["effect_text"][cursor : candidate["start"]]})
            reference = {
                "reference_id": candidate["reference_id"],
                "reference_type": candidate["reference_type"],
                "source_global_detail_index": detail["global_detail_index"],
                "model_source_span": [candidate["start"], candidate["end"]],
                "raw_source_span": _raw_span(
                    detail["effect_text_raw"], detail["effect_text"], candidate["start"], candidate["end"]
                ),
                "payload": candidate["payload"],
            }
            references.append(reference)
            model_tokens.append({
                "token_kind": "STRUCTURE_REFERENCE",
                "reference_type": candidate["reference_type"],
                "reference_id": candidate["reference_id"],
            })
            cursor = candidate["end"]
        if cursor < len(detail["effect_text"]):
            model_tokens.append({"token_kind": "TEXT", "text": detail["effect_text"][cursor:]})
        if not model_tokens and detail["effect_text"]:
            model_tokens = [{"token_kind": "TEXT", "text": detail["effect_text"]}]
        detail["text_references"] = references
        detail["model_text_tokens"] = model_tokens
        detail["detail_fingerprint"] = _reference_fingerprint(detail)

    schema = {
        "schema_version": SCHEMA_VERSION,
        "metadata_id_policy": {
            "reserved": {NULL_TOKEN: 0, UNK_TOKEN: 1},
            "normalization": "Unicode NFKC, whitespace collapsed, detail [Ability] prefix removed",
            "ordering": "normalized Unicode strings sorted by Python code-point order from id 2",
            "model_boundary": (
                "card_name_id is a card identity input without exact-name recovery; "
                "detail name ids and numeric reference ids are relation/audit metadata only"
            ),
        },
        "field_vocabs": {
            "card_name": card_name_vocab,
            "detail_name": detail_name_vocab,
            "card_kind": card_kind_vocab,
            "card_type": card_type_vocab,
            "printed_class": printed_class_vocab,
            "category_family": category_family_vocab,
            "category_qualifier": category_qualifier_vocab,
            "rule": rule_vocab,
            "rule_family": rule_family_vocab,
        },
        "allowed_reference_types": sorted(ALLOWED_REFERENCE_TYPES),
        "reference_fields": V3_REFERENCE_FIELDS,
        "model_text_token_kinds": ["TEXT", "STRUCTURE_REFERENCE"],
        "canonical_detail_fields": [
            "global_detail_index", "local_detail_index", "card_index", "card_id", "source_row",
            "detail_type", "name_raw", "name_normalized", "name_id", "effect_text_raw", "effect_text",
            "effect_text_override",
            "model_text_tokens", "text_references", "detail_fingerprint", "cost_raw", "cost_mode",
            "cost_counts", "damage_raw", "damage_value", "damage_mode", "attack_id",
        ],
        "compatibility_detail_fields": ["energy_costs", "base_damage"],
        "non_model_metadata_fields": [
            "card_id", "name_raw", "name_normalized", "name_id", "source_row", "attack_id",
            "reference_id", "model_source_span", "raw_source_span", "effect_text_override",
            "energy_costs", "base_damage",
        ],
    }
    return schema, unresolved_audit, overrides_sha256, [str(row["override_id"]) for row in overrides]


def build_corpus(
    *,
    rows: list[dict[str, str]] | None = None,
    source: str | None = None,
    source_sha256: str | None = None,
    cg_cards: dict[int, Any] | None = None,
    cg_attacks: dict[int, Any] | None = None,
    text_reference_overrides_path: Path = TEXT_REFERENCE_OVERRIDES_PATH,
    enforce_expected: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[int], dict[str, Any]]:
    if rows is None:
        rows, loaded_source, loaded_sha = load_csv_rows_with_hash()
        source = source or loaded_source
        source_sha256 = source_sha256 or loaded_sha
    if cg_cards is None or cg_attacks is None:
        cg_cards, cg_attacks = load_cg_data(required=True)
    source = source or "<provided rows>"
    source_sha256 = source_sha256 or hashlib.sha256(
        json.dumps(rows, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()

    grouped: dict[str, list[tuple[int, dict[str, str]]]] = defaultdict(list)
    for source_row, row in enumerate(rows):
        card_id = normalize_missing(row.get("Card ID"))
        if not card_id or not card_id.isdigit():
            raise CardPreprocessingError(f"source row {source_row} has invalid Card ID {card_id!r}")
        grouped[card_id].append((source_row, row))

    sorted_ids = sorted(grouped, key=int)
    first_rows: dict[str, dict[str, str]] = {}
    for card_id in sorted_ids:
        card_rows = grouped[card_id]
        first = card_rows[0][1]
        first_rows[card_id] = first
        for column in CARD_LEVEL_COLUMNS:
            values = {normalize_missing(row.get(column)) for _index, row in card_rows}
            if len(values) != 1:
                raise CardPreprocessingError(f"card {card_id} has inconsistent {column}: {sorted(values)!r}")

    name_to_category: dict[str, str] = {}
    for card_id, first in first_rows.items():
        name = normalize_missing(first.get("Card Name"))
        kind = normalize_missing(first.get("Stage (Pokémon)/Type (Energy and Trainer)"))
        if not name:
            raise CardPreprocessingError(f"card {card_id} has no Card Name")
        if kind not in KIND_SCHEMA:
            raise CardPreprocessingError(f"card {card_id} has unknown kind {kind!r}")
        category = KIND_SCHEMA[kind][0]
        previous = name_to_category.setdefault(name, category)
        if previous != category:
            raise CardPreprocessingError(f"card name {name!r} appears in multiple categories")

    cards: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    offsets = [0]
    unresolved: list[dict[str, Any]] = []
    for card_index, card_id in enumerate(sorted_ids):
        card_rows = grouped[card_id]
        first = card_rows[0][1]
        name = normalize_missing(first.get("Card Name"))
        kind = normalize_missing(first.get("Stage (Pokémon)/Type (Energy and Trainer)"))
        card_category, card_type, stage, trainer_subtype, energy_subtype = KIND_SCHEMA[kind]
        card_tags = normalize_card_tags(first.get("Category"))
        rule_flags = normalize_rule_flags(first.get("Rule"), card_tags)
        card_kind = normalize_card_kind(kind)
        rule, rule_family = normalize_rule_and_family(first.get("Rule"))
        category_family, category_qualifier = category_family_and_qualifier(card_category, card_tags)
        printed_class = PRINTED_CLASS_BY_KIND[kind]
        raw_previous_field = normalize_missing(first.get("Previous stage")) or None
        raw_previous = raw_previous_field if card_category == "POKEMON" else None
        evolves_to_card_name = None
        if card_category != "POKEMON" and raw_previous_field is not None:
            fossil_target = re.fullmatch(r"Evolve to (.+)", raw_previous_field)
            if "FOSSIL" not in card_tags or fossil_target is None:
                raise CardPreprocessingError(
                    f"non-Pokémon card {card_id} has unsupported Previous stage value {raw_previous_field!r}"
                )
            evolves_to_card_name = normalize_space(fossil_target.group(1))
            if evolves_to_card_name not in name_to_category:
                raise CardPreprocessingError(
                    f"Fossil card {card_id} references unknown evolution target {evolves_to_card_name!r}"
                )
        if stage in {"STAGE1", "STAGE2"} and raw_previous is None:
            raise CardPreprocessingError(f"evolution card {card_id} has no Previous stage")
        if stage == "BASIC" and raw_previous is not None:
            raise CardPreprocessingError(f"basic Pokémon {card_id} unexpectedly has Previous stage")
        if raw_previous is not None and raw_previous not in name_to_category:
            raise CardPreprocessingError(f"card {card_id} references unknown Previous stage {raw_previous!r}")
        previous_species = (
            canonical_species_name(raw_previous)
            if raw_previous is not None and name_to_category[raw_previous] == "POKEMON"
            else None
        )

        type_value = first.get("Type")
        if card_category == "POKEMON":
            pokemon_type = _single_energy_symbol(type_value, "Type", required=True)
            provided_energy_counts: dict[str, int] = {}
        elif card_category == "ENERGY":
            pokemon_type = None
            provided_energy_counts = _parse_symbol_counts(type_value, allow_bullets=False, allow_no_cost=False)
            if not provided_energy_counts:
                raise CardPreprocessingError(f"energy card {card_id} has no provided-energy signature")
        else:
            pokemon_type = None
            provided_energy_counts = {}
            if normalize_missing(type_value):
                raise CardPreprocessingError(f"trainer card {card_id} unexpectedly has Type={type_value!r}")

        raw_hp = parse_int(first.get("HP"))
        if card_category == "POKEMON":
            if raw_hp is None or raw_hp <= 0:
                raise CardPreprocessingError(f"Pokémon card {card_id} has invalid HP {raw_hp!r}")
            printed_hp, hp_applicability = raw_hp, "POKEMON"
        elif "FOSSIL" in card_tags:
            if card_type != "ITEM" or raw_hp != 60:
                raise CardPreprocessingError(f"Fossil card {card_id} must be a 60-HP Item")
            printed_hp, hp_applicability = raw_hp, "PLAYABLE_AS_POKEMON"
        else:
            if raw_hp is not None:
                raise CardPreprocessingError(f"non-Pokémon card {card_id} unexpectedly has HP={raw_hp}")
            printed_hp, hp_applicability = None, "NOT_APPLICABLE"

        retreat = parse_int(first.get("Retreat"))
        if card_category == "POKEMON":
            retreat = 0 if retreat is None else retreat
            if retreat < 0:
                raise CardPreprocessingError(f"card {card_id} has negative Retreat")
        elif retreat is not None:
            raise CardPreprocessingError(f"non-Pokémon card {card_id} unexpectedly has Retreat={retreat}")
        weakness = _single_energy_symbol(first.get("Weakness"), "Weakness", required=False)
        resistance = _single_energy_symbol(first.get("Resistance (Type)"), "Resistance", required=False)
        if card_category != "POKEMON" and (weakness is not None or resistance is not None):
            raise CardPreprocessingError(f"non-Pokémon card {card_id} has Weakness/Resistance")

        card_start = len(details)
        for local_detail_index, (source_row, row) in enumerate(
            (entry for entry in card_rows if any(normalize_missing(entry[1].get(column)) for column in DETAIL_COLUMNS))
        ):
            name_value = row.get("Move Name")
            name_raw = "" if name_value is None else str(name_value)
            name_normalized = normalize_detail_name(name_raw)
            cost_value = row.get("Cost")
            cost_raw = "" if cost_value is None else str(cost_value)
            cost_normalized = normalize_missing(cost_raw)
            damage_value_raw = row.get("Damage")
            damage_raw = "" if damage_value_raw is None else str(damage_value_raw)
            damage_normalized = normalize_missing(damage_raw)
            raw_effect_value = row.get("Effect Explanation")
            effect_text_raw = "" if raw_effect_value is None else str(raw_effect_value)
            effect_text = normalize_unicode_text(raw_effect_value)
            is_ability = normalize_unicode_text(name_raw).startswith("[Ability]")
            if is_ability:
                if card_category != "POKEMON" and "FOSSIL" not in card_tags:
                    unresolved.append({
                        "card_id": card_id,
                        "source_row": source_row,
                        "reason": "ability_on_non_pokemon_non_fossil",
                    })
                    continue
                if cost_normalized or damage_normalized:
                    unresolved.append({
                        "card_id": card_id,
                        "source_row": source_row,
                        "reason": "ability_has_attack_cost_or_damage",
                    })
                    continue
                detail_type = "ABILITY"
                cost_mode, cost_counts, damage_value, damage_mode = "NOT_APPLICABLE", {}, None, "NONE"
            elif cost_normalized or damage_normalized:
                detail_type = "ATTACK"
                if not name_normalized or not cost_normalized:
                    unresolved.append({"card_id": card_id, "source_row": source_row, "reason": "attack_missing_name_or_cost"})
                    continue
                if card_category != "POKEMON" and "TECHNICAL_MACHINE" not in card_tags:
                    unresolved.append({"card_id": card_id, "source_row": source_row, "reason": "non_pokemon_attack_without_tm_tag"})
                    continue
                cost_counts = energy_cost_dict(cost_normalized)
                cost_mode = "EXPLICIT_ZERO" if cost_normalized.casefold() == "no cost" else "COUNTS"
                damage_value = parse_damage(damage_normalized)
                damage_mode = parse_damage_mode(damage_normalized)
            elif effect_text:
                detail_type = "CARD_EFFECT"
                cost_mode, cost_counts, damage_value, damage_mode = "NOT_APPLICABLE", {}, None, "NONE"
                if card_category == "POKEMON" and name_normalized != "Tera":
                    unresolved.append({"card_id": card_id, "source_row": source_row, "reason": "unknown_pokemon_rule_row"})
                    continue
            else:
                unresolved.append({"card_id": card_id, "source_row": source_row, "reason": "unclassifiable_detail"})
                continue
            details.append(
                asdict(
                    DetailRecord(
                        global_detail_index=len(details),
                        local_detail_index=local_detail_index,
                        card_index=card_index,
                        card_id=card_id,
                        source_row=source_row,
                        detail_type=detail_type,
                        name_raw=name_raw,
                        name_normalized=name_normalized,
                        name_id=-1,
                        effect_text_raw=effect_text_raw,
                        effect_text=effect_text,
                        effect_text_override=None,
                        model_text_tokens=[],
                        text_references=[],
                        detail_fingerprint="",
                        cost_raw=cost_raw,
                        cost_mode=cost_mode,
                        cost_counts=cost_counts,
                        damage_raw=damage_raw,
                        damage_value=damage_value,
                        energy_costs=cost_counts,
                        base_damage=damage_value,
                        damage_mode=damage_mode,
                        attack_id=None,
                    )
                )
            )
        # A source row with no detail fields is valid only for a Basic Energy card.
        empty_rows = [
            source_row
            for source_row, row in card_rows
            if not any(normalize_missing(row.get(column)) for column in DETAIL_COLUMNS)
        ]
        if empty_rows and not (card_type == "BASIC_ENERGY" and len(card_rows) == 1):
            raise CardPreprocessingError(f"card {card_id} has unexpected empty detail rows {empty_rows}")

        card_end = len(details)
        provider_list = [symbol for symbol, count in provided_energy_counts.items() for _ in range(count)]
        cards.append(
            asdict(
                CardRecord(
                    card_id=card_id,
                    card_name=name,
                    card_name_normalized=normalize_unicode_text(name),
                    card_name_id=-1,
                    name=name,
                    card_kind=card_kind,
                    card_kind_id=-1,
                    card_category=card_category,
                    card_type=card_type,
                    card_type_id=-1,
                    printed_class=printed_class,
                    printed_class_id=-1,
                    category_family=category_family,
                    category_family_id=-1,
                    category_qualifier=category_qualifier,
                    category_qualifier_id=-1,
                    rule=rule,
                    rule_id=-1,
                    rule_family=rule_family,
                    rule_family_id=-1,
                    card_tags=card_tags,
                    stage=stage,
                    trainer_subtype=trainer_subtype,
                    energy_subtype=energy_subtype,
                    previous_species=previous_species,
                    evolves_from_card_name=raw_previous,
                    evolves_to_card_name=evolves_to_card_name,
                    evolves_from_name_id=-1,
                    evolves_to_card_names=[],
                    evolves_to_name_ids=[],
                    pokemon_type=pokemon_type,
                    provided_energy_counts=provided_energy_counts,
                    printed_hp=printed_hp,
                    hp_applicability=hp_applicability,
                    retreat=retreat,
                    weakness_type=weakness,
                    resistance_type=resistance,
                    rule_flags=rule_flags,
                    detail_start=card_start,
                    detail_end=card_end,
                    expansion=normalize_missing(first.get("Expansion")),
                    collection_no=normalize_missing(first.get("Collection No.")),
                    source_kind=kind,
                    source_category=normalize_missing(first.get("Category")) or None,
                    subtype=kind,
                    trainer_type=trainer_subtype,
                    hp=printed_hp,
                    retreat_cost=retreat,
                    evolves_from=raw_previous,
                    provided_energy_types=provider_list,
                )
            )
        )
        offsets.append(card_end)

    if unresolved:
        raise CardPreprocessingError(
            f"unresolved detail rows ({len(unresolved)}): {json.dumps(unresolved[:20], ensure_ascii=False)}"
        )
    _validate_and_bind_cg(cards, details, cg_cards, cg_attacks)

    feature_schema, unresolved_text_references, overrides_sha256, applied_override_ids = (
        _finalize_v3_metadata_and_references(
            cards,
            details,
            overrides_path=Path(text_reference_overrides_path),
        )
    )

    card_id_to_index = {card["card_id"]: index for index, card in enumerate(cards)}
    detail_counts = Counter(detail["detail_type"] for detail in details)
    reference_counts = Counter(
        reference["reference_type"]
        for detail in details
        for reference in detail["text_references"]
    )
    evolution_edges = sorted(
        {
            (
                card["evolves_from_name_id"],
                card["card_name_id"],
                "PRINTED_EVOLVES_FROM",
            )
            for card in cards
            if card["evolves_from_name_id"] != 0
        }
        | {
            (card["card_name_id"], target_id, "PRINTED_EVOLVES_TO")
            for card in cards
            for target_id in card["evolves_to_name_ids"]
        }
    )
    mapping_sha256 = hashlib.sha256(_serialized_json_bytes(card_id_to_index)).hexdigest()
    corpus_fingerprint = hashlib.sha256(
        _compact_json(
            {
                "schema_version": SCHEMA_VERSION,
                "cards": cards,
                "details": details,
                "detail_offsets": offsets,
                "card_id_to_index": card_id_to_index,
            }
        ).encode("utf-8")
    ).hexdigest()
    try:
        override_source = str(Path(text_reference_overrides_path).resolve().relative_to(ROOT.resolve()))
    except ValueError:
        override_source = str(text_reference_overrides_path)
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "source": source,
        "source_sha256": source_sha256,
        "source_columns": SOURCE_COLUMNS,
        "source_row_count": len(rows),
        "card_count": len(cards),
        "detail_count": len(details),
        "detail_type_counts": dict(sorted(detail_counts.items())),
        "text_reference_count": sum(reference_counts.values()),
        "text_reference_type_counts": dict(sorted(reference_counts.items())),
        "unresolved_text_reference_count": len(unresolved_text_references),
        "unresolved_text_references": unresolved_text_references,
        "card_category_counts": dict(sorted(Counter(card["card_category"] for card in cards).items())),
        "card_type_counts": dict(sorted(Counter(card["card_type"] for card in cards).items())),
        "cg_cards_loaded": len(cg_cards),
        "cg_attacks_loaded": len(cg_attacks),
        "unresolved_count": 0,
        "known_attack_id_resolutions": {str(key): value for key, value in KNOWN_ATTACK_ID_BINDINGS.items()},
        "known_effect_text_corrections": [
            {
                "correction_id": KNOWN_EFFECT_TEXT_CORRECTIONS[(int(detail["card_id"]), int(detail["attack_id"]))],
                "reason": "english_csv_contains_japanese_attack_effect_text",
                "card_id": int(detail["card_id"]),
                "attack_id": int(detail["attack_id"]),
                "source_row": int(detail["source_row"]),
                "before_hash": hashlib.sha256(detail["effect_text_raw"].encode("utf-8")).hexdigest(),
                "after_hash": hashlib.sha256(detail["effect_text"].encode("utf-8")).hexdigest(),
            }
            for detail in details
            if (int(detail["card_id"]), int(detail["attack_id"] or -1)) in KNOWN_EFFECT_TEXT_CORRECTIONS
        ],
        "previous_species_compatibility": {
            "model_input": False,
            "relation_source": False,
            "note": "legacy metadata only; no current-card species field or species id is generated",
        },
        "detail_order": "numeric_card_id_then_source_csv_row",
        "source_row_semantics": "zero_based_logical_csv_data_record; header excluded",
        "detail_offsets_last": offsets[-1],
        "mapping_role": "PREPROCESS_CARD_RECORD_ROW",
        "feature_schema": feature_schema,
        "evolves_to_derivation": {
            "derived": True,
            "rule": "inverse of normalized printed evolves_from plus explicit Fossil Evolve to target",
            "ordering": "normalized Unicode target name order",
        },
        "evolution_edges": [
            {"source_name_id": source_id, "target_name_id": target_id, "relation_type": relation_type}
            for source_id, target_id, relation_type in evolution_edges
        ],
        "text_reference_overrides": {
            "path": override_source,
            "sha256": overrides_sha256,
            "applied_override_ids": applied_override_ids,
        },
        "mappings": {
            "card_id_to_index": {
                "role": "PREPROCESS_CARD_RECORD_ROW",
                "path": "card_id_to_index.json",
                "sha256": mapping_sha256,
            }
        },
        "corpus_fingerprint": corpus_fingerprint,
    }
    _validate_corpus(cards, details, offsets, card_id_to_index, manifest, enforce_expected=enforce_expected)
    return cards, details, offsets, manifest


def _validate_corpus(
    cards: list[dict[str, Any]],
    details: list[dict[str, Any]],
    offsets: list[int],
    card_id_to_index: dict[str, int],
    manifest: dict[str, Any],
    *,
    enforce_expected: bool,
) -> None:
    if len(card_id_to_index) != len(cards) or set(card_id_to_index.values()) != set(range(len(cards))):
        raise CardPreprocessingError("card_id_to_index is not a contiguous bijection")
    if len(offsets) != len(cards) + 1 or offsets[0] != 0 or offsets[-1] != len(details):
        raise CardPreprocessingError("detail_offsets shape or endpoint is invalid")
    if any(left > right for left, right in zip(offsets, offsets[1:])):
        raise CardPreprocessingError("detail_offsets are not monotonic")
    for card_index, card in enumerate(cards):
        if (card["detail_start"], card["detail_end"]) != (offsets[card_index], offsets[card_index + 1]):
            raise CardPreprocessingError(f"card {card['card_id']} detail range disagrees with offsets")
        for local_index, detail in enumerate(details[offsets[card_index] : offsets[card_index + 1]]):
            if detail["global_detail_index"] != offsets[card_index] + local_index:
                raise CardPreprocessingError("global detail indices are not contiguous")
            if detail["local_detail_index"] != local_index or detail["card_index"] != card_index:
                raise CardPreprocessingError(f"detail ownership mismatch for card {card['card_id']}")
            if detail["card_id"] != card["card_id"]:
                raise CardPreprocessingError("detail card_id disagrees with its offset owner")
            if detail["source_row"] < 0:
                raise CardPreprocessingError("source_row must be a zero-based logical record index")
            if detail["name_normalized"]:
                if detail["name_id"] < 2:
                    raise CardPreprocessingError("named detail is missing its normalized metadata id")
            elif detail["name_id"] != 0:
                raise CardPreprocessingError("unnamed detail must use the metadata null id")
            forbidden_fields = {
                "source_row_index", "source_line_number", "source_effect_text", "detail_subtype", "move_name"
            }
            if forbidden_fields & set(detail):
                raise CardPreprocessingError(f"legacy detail fields leaked into v3: {sorted(forbidden_fields & set(detail))}")
            if detail["energy_costs"] != detail["cost_counts"] or detail["base_damage"] != detail["damage_value"]:
                raise CardPreprocessingError("compatibility detail numerics drifted from v3 canonical fields")
            if detail["cost_mode"] not in {"EXPLICIT_ZERO", "NOT_APPLICABLE", "COUNTS"}:
                raise CardPreprocessingError(f"unknown cost mode {detail['cost_mode']!r}")
            if detail["damage_mode"] not in {"NONE", "FIXED", "PLUS", "MINUS", "MULTIPLY"}:
                raise CardPreprocessingError(f"unknown damage mode {detail['damage_mode']!r}")
            if detail["detail_type"] == "ATTACK" and detail["cost_mode"] == "NOT_APPLICABLE":
                raise CardPreprocessingError("attack detail cannot have NOT_APPLICABLE cost")
            if detail["detail_type"] != "ATTACK" and detail["cost_mode"] != "NOT_APPLICABLE":
                raise CardPreprocessingError("non-attack detail must have NOT_APPLICABLE cost")
            if any(
                not isinstance(count, int) or isinstance(count, bool) or count < 0
                for count in detail["cost_counts"].values()
            ):
                raise CardPreprocessingError("energy counts must be non-negative integers")
            previous_end = 0
            for reference in detail["text_references"]:
                if reference["source_global_detail_index"] != detail["global_detail_index"]:
                    raise CardPreprocessingError("text reference source detail index drifted")
                reference_type = reference["reference_type"]
                _validate_reference_payload(reference_type, reference["payload"])
                start, end = reference["model_source_span"]
                if not (previous_end <= start < end <= len(detail["effect_text"])):
                    raise CardPreprocessingError("text reference spans overlap or exceed canonical effect text")
                previous_end = end
            if detail["detail_fingerprint"] != _reference_fingerprint(detail):
                raise CardPreprocessingError("detail fingerprint does not match model-visible semantics")
            token_reference_ids = [
                token["reference_id"]
                for token in detail["model_text_tokens"]
                if token.get("token_kind") == "STRUCTURE_REFERENCE"
            ]
            if token_reference_ids != [reference["reference_id"] for reference in detail["text_references"]]:
                raise CardPreprocessingError("model text reference tokens do not align with text_references")
            if any(token.get("token_kind") not in {"TEXT", "STRUCTURE_REFERENCE"} for token in detail["model_text_tokens"]):
                raise CardPreprocessingError("unknown model_text_tokens token kind")
        if any(
            card[field] < 2
            for field in (
                "card_name_id", "card_kind_id", "card_type_id", "printed_class_id",
                "category_family_id", "category_qualifier_id", "rule_id", "rule_family_id",
            )
        ):
            raise CardPreprocessingError(f"card {card['card_id']} has an invalid metadata id")
        expected_printed_class = PRINTED_CLASS_BY_KIND[card["source_kind"]]
        if card["printed_class"] != expected_printed_class:
            raise CardPreprocessingError(f"card {card['card_id']} printed class drifted from source kind")
        if len(card["evolves_to_card_names"]) != len(card["evolves_to_name_ids"]):
            raise CardPreprocessingError(f"card {card['card_id']} evolution target ids are misaligned")
    source_rows = [detail["source_row"] for detail in details]
    if len(source_rows) != len(set(source_rows)):
        raise CardPreprocessingError("a source row generated more than one detail")
    reference_ids = [
        reference["reference_id"]
        for detail in details
        for reference in detail["text_references"]
    ]
    if reference_ids != list(range(len(reference_ids))):
        raise CardPreprocessingError("reference ids are not contiguous in canonical detail/span order")
    mapping_declaration = manifest.get("mappings", {}).get("card_id_to_index", {})
    if mapping_declaration.get("sha256") != hashlib.sha256(_serialized_json_bytes(card_id_to_index)).hexdigest():
        raise CardPreprocessingError("card_id_to_index mapping hash does not match canonical serialization")
    if manifest.get("mapping_role") != "PREPROCESS_CARD_RECORD_ROW":
        raise CardPreprocessingError("preprocess mapping role is missing or ambiguous")
    expected_fingerprint = hashlib.sha256(
        _compact_json(
            {
                "schema_version": SCHEMA_VERSION,
                "cards": cards,
                "details": details,
                "detail_offsets": offsets,
                "card_id_to_index": card_id_to_index,
            }
        ).encode("utf-8")
    ).hexdigest()
    if manifest.get("corpus_fingerprint") != expected_fingerprint:
        raise CardPreprocessingError("corpus fingerprint does not match canonical records")
    if not enforce_expected:
        return
    actual = {
        "source_rows": manifest["source_row_count"],
        "cards": len(cards),
        "details": len(details),
        **Counter(detail["detail_type"] for detail in details),
        "tools": sum(card["card_type"] == "TOOL" for card in cards),
        "fossils": sum("FOSSIL" in card["card_tags"] for card in cards),
        "tera_effects": sum(detail["name_raw"] in {"[Tera]", "Tera"} for detail in details),
    }
    if manifest["source_sha256"] != EXPECTED_SOURCE_SHA256:
        raise CardPreprocessingError(
            f"source SHA-256 changed: {manifest['source_sha256']} != {EXPECTED_SOURCE_SHA256}"
        )
    mismatches = {key: (actual.get(key), expected) for key, expected in EXPECTED_COUNTS.items() if actual.get(key) != expected}
    if mismatches:
        raise CardPreprocessingError(f"full-corpus invariants changed: {mismatches}")
    core = [detail for detail in details if detail["card_id"] == "1180" and detail["detail_type"] == "ATTACK"]
    if len(core) != 1 or core[0]["name_raw"] != "Geobuster" or core[0]["attack_id"] != 1556:
        raise CardPreprocessingError("Core Memory / Geobuster semantic invariant failed")
    fossils = [card for card in cards if "FOSSIL" in card["card_tags"]]
    if any(
        card["printed_hp"] != 60
        or card["hp_applicability"] != "PLAYABLE_AS_POKEMON"
        or card["printed_class"] != "ITEM"
        or card["category_family"] != "FOSSIL"
        for card in fossils
    ):
        raise CardPreprocessingError("Fossil HP applicability invariant failed")
    fossil_abilities = [
        detail
        for detail in details
        if detail["card_id"] in {card["card_id"] for card in fossils}
        and detail["name_raw"].startswith("[Ability]")
    ]
    if len(fossil_abilities) != 5 or any(detail["detail_type"] != "ABILITY" for detail in fossil_abilities):
        raise CardPreprocessingError("Fossil ability invariant failed")


def build_records() -> tuple[list[CardRecord], dict[str, Any]]:
    cards, _details, _offsets, manifest = build_corpus()
    return [CardRecord(**card) for card in cards], manifest


def summarize_records(records: list[CardRecord] | list[dict[str, Any]]) -> dict[str, Any]:
    rows = [asdict(record) if isinstance(record, CardRecord) else record for record in records]
    return {
        "schema_version": SCHEMA_VERSION,
        "card_count": len(rows),
        "card_category_counts": dict(sorted(Counter(row["card_category"] for row in rows).items())),
        "card_type_counts": dict(sorted(Counter(row["card_type"] for row in rows).items())),
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_bytes(_serialized_json_bytes(value))


def _guard_v3_cache_write(cache_dir: Path) -> None:
    if cache_dir.resolve() == LEGACY_V2_CACHE_DIR.resolve():
        raise CardPreprocessingError(f"v2 cache is read-only and cannot be rebuilt: {cache_dir}")
    manifest_path = cache_dir / "preprocess_manifest.json"
    if manifest_path.exists():
        try:
            existing_schema = json.loads(manifest_path.read_text(encoding="utf-8")).get("schema_version")
        except (json.JSONDecodeError, OSError) as exc:
            raise CardPreprocessingError(f"refusing to overwrite unreadable cache manifest {manifest_path}") from exc
        if existing_schema != SCHEMA_VERSION:
            raise CardPreprocessingError(
                f"refusing to overwrite {existing_schema!r} cache with {SCHEMA_VERSION!r}: {cache_dir}"
            )


def write_card_cache(cache_dir: Path = DEFAULT_CACHE_DIR) -> dict[str, Any]:
    cache_dir = Path(cache_dir)
    _guard_v3_cache_write(cache_dir)
    cards, details, offsets, manifest = build_corpus()
    mapping = {card["card_id"]: index for index, card in enumerate(cards)}
    values = {
        "cards.json": cards,
        "details.json": details,
        "detail_offsets.json": offsets,
        "card_id_to_index.json": mapping,
        "preprocess_manifest.json": manifest,
    }
    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(
        tempfile.mkdtemp(prefix=f".{cache_dir.name}.staging-", dir=str(cache_dir.parent))
    )
    try:
        for filename, value in values.items():
            _write_json(temporary_dir / filename, value)
        cache_dir.mkdir(parents=True, exist_ok=True)
        unexpected = {path.name for path in cache_dir.iterdir()} - set(values)
        if unexpected:
            raise CardPreprocessingError(
                f"refusing to mix v3 canonical cache with unexpected files: {sorted(unexpected)}"
            )
        for filename in values:
            os.replace(temporary_dir / filename, cache_dir / filename)
    finally:
        shutil.rmtree(temporary_dir, ignore_errors=True)
    return manifest


def load_or_create_corpus(
    cache_dir: Path = DEFAULT_CACHE_DIR,
    rebuild: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[int], dict[str, int], dict[str, Any]]:
    cache_dir = Path(cache_dir)
    paths = {
        "cards": cache_dir / "cards.json",
        "details": cache_dir / "details.json",
        "offsets": cache_dir / "detail_offsets.json",
        "mapping": cache_dir / "card_id_to_index.json",
        "manifest": cache_dir / "preprocess_manifest.json",
    }
    if rebuild or not all(path.exists() for path in paths.values()):
        write_card_cache(cache_dir)
    cards = json.loads(paths["cards"].read_text(encoding="utf-8"))
    details = json.loads(paths["details"].read_text(encoding="utf-8"))
    offsets = [int(value) for value in json.loads(paths["offsets"].read_text(encoding="utf-8"))]
    mapping = {str(key): int(value) for key, value in json.loads(paths["mapping"].read_text(encoding="utf-8")).items()}
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise CardPreprocessingError(
            f"cache {cache_dir} has schema {manifest.get('schema_version')!r}, expected {SCHEMA_VERSION!r}"
        )
    _validate_corpus(cards, details, offsets, mapping, manifest, enforce_expected=True)
    return cards, details, offsets, mapping, manifest


def load_or_create_records(
    cache_dir: Path = DEFAULT_CACHE_DIR,
    rebuild: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, int], dict[str, Any]]:
    cards, _details, _offsets, mapping, manifest = load_or_create_corpus(cache_dir, rebuild=rebuild)
    return cards, mapping, manifest
