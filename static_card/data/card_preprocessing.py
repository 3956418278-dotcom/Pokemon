from __future__ import annotations

import csv
import hashlib
import importlib
import json
import re
import sys
import unicodedata
import zipfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE_DIR = ROOT / "static_card" / "artifacts" / "card_data"
CORPUS_SCHEMA_VERSION = 6
ENERGY_TYPES = ["C", "G", "R", "W", "L", "P", "F", "D", "M", "N", "Y", "A"]
CG_ENERGY_TYPES = {0: "C", 1: "G", 2: "R", 3: "W", 4: "L", 5: "P", 6: "F", 7: "D", 8: "M", 9: "N", 10: "Y"}
CG_CARD_TYPES = {0: "POKEMON", 1: "ITEM", 2: "TOOL", 3: "SUPPORTER", 4: "STADIUM", 5: "BASIC_ENERGY", 6: "SPECIAL_ENERGY"}
SOURCE_COLUMNS = [
    "Card ID", "Card Name", "Expansion", "Collection No.", "Stage (Pokémon)/Type (Energy and Trainer)",
    "Rule", "Category", "Previous stage", "HP", "Type", "Weakness", "Resistance (Type)", "Retreat",
    "Move Name", "Cost", "Damage", "Effect Explanation",
]
CARD_LEVEL_COLUMNS = SOURCE_COLUMNS[:13]
CARD_TYPE_STAGE_MAP: dict[str, tuple[str, str | None]] = {
    "Basic Pokémon": ("POKEMON", "BASIC"),
    "Stage 1 Pokémon": ("POKEMON", "STAGE1"),
    "Stage 2 Pokémon": ("POKEMON", "STAGE2"),
    "Item": ("ITEM", None),
    "Pokémon Tool": ("TOOL", None),
    "Supporter": ("SUPPORTER", None),
    "Stadium": ("STADIUM", None),
    "Basic Energy": ("BASIC_ENERGY", None),
    "Special Energy": ("SPECIAL_ENERGY", None),
}
RULE_MAP: dict[str, str] = {
    "Pokémon ex": "POKEMON_EX",
    "Mega Pokémon ex": "MEGA_POKEMON_EX",
    "ACE SPEC": "ACE_SPEC",
}
TYPE_MAP = {
    **{f"{{{energy_type}}}": energy_type for energy_type in CG_ENERGY_TYPES.values() if energy_type != "N"},
    "{N}": "DRAGON",
    "竜": "DRAGON",
}


@dataclass
class DetailRecord:
    detail_id: int
    detail_index: int
    card_id: int
    source_row: int
    source_line: int
    detail_type: str
    detail_subtype: str
    move_name: str
    detail_name: str
    text: str
    source_text: str
    attack_id: int | None
    energy_counts: list[int]
    damage_raw: str | None
    damage_base: int | None
    damage_mode: str
    corrections: list[str] = field(default_factory=list)
    source_fields: dict[str, str] = field(default_factory=dict)


@dataclass
class CardRecord:
    card_id: int
    name: str
    card_type: str
    stage: str | None
    rule: str | None
    category: str | None
    type: str | None
    hp: int | None
    weakness_type: str | None
    resistance_type: str | None
    retreat_cost: int | None
    evolves_from: str | None
    evolves_to: list[str]
    detail_ids: list[int]
    expansion: str
    collection_no: str
    source_fields: dict[str, str] = field(default_factory=dict)


def normalize_missing(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if not text or text.lower() in {"n/a", "nan", "none", "-", "—"} else text


def normalized_text(value: Any) -> str:
    text = unicodedata.normalize("NFC", normalize_missing(value)).replace("\xa0", " ")
    return re.sub(r"\s+", " ", text).strip()


def parse_int(value: Any) -> int | None:
    match = re.search(r"-?\d+", normalize_missing(value))
    return int(match.group(0)) if match else None


def parse_damage_fields(value: Any) -> tuple[int | None, str]:
    text = normalize_missing(value)
    if not text:
        return None, "NONE"
    numbers = [int(x) for x in re.findall(r"\d+", text)]
    base = numbers[0] if numbers else None
    if "×" in text or "x" in text.lower():
        return base, "MULTIPLIER"
    if "+" in text:
        return base, "PLUS"
    if "-" in text and base is not None:
        return base, "MINUS"
    return base, "FIXED" if base is not None else "VARIABLE"


def parse_energy_symbols(value: Any) -> list[str]:
    text = normalize_missing(value)
    symbols = [x.strip().upper() for x in re.findall(r"\{([^}]+)\}", text) if x.strip()]
    symbols.extend(["C"] * text.count("●"))
    return symbols


def energy_counts(value: Any) -> list[int]:
    counts = Counter(parse_energy_symbols(value))
    unknown = sorted(set(counts) - set(ENERGY_TYPES))
    if unknown:
        raise ValueError(f"unknown energy symbols: {unknown} in {value!r}")
    return [int(counts.get(name, 0)) for name in ENERGY_TYPES]


def cg_energy_counts(values: list[Any]) -> list[int]:
    result = [0] * len(ENERGY_TYPES)
    for value in values or []:
        key = CG_ENERGY_TYPES.get(int(value))
        if key is None:
            raise ValueError(f"unknown CG energy enum: {value!r}")
        result[ENERGY_TYPES.index(key)] += 1
    return result


def card_type_and_stage(kind: Any) -> tuple[str, str | None]:
    normalized = normalized_text(kind)
    try:
        return CARD_TYPE_STAGE_MAP[normalized]
    except KeyError as exc:
        raise ValueError(f"unknown E-column value: {normalized!r}") from exc


def normalized_rule(value: Any) -> str | None:
    normalized = normalized_text(value)
    if not normalized:
        return None
    try:
        return RULE_MAP[normalized]
    except KeyError as exc:
        raise ValueError(f"unknown F-column value: {normalized!r}") from exc


def normalized_type(value: Any) -> str:
    normalized = normalized_text(value)
    try:
        return TYPE_MAP[normalized]
    except KeyError as exc:
        raise ValueError(f"unknown Type-column value: {normalized!r}") from exc


def normalized_energy_symbol(value: Any) -> str | None:
    normalized = normalized_text(value)
    if not normalized:
        return None
    symbols = parse_energy_symbols(normalized)
    if len(symbols) != 1 or symbols[0] not in ENERGY_TYPES:
        raise ValueError(f"expected one energy type symbol, got {value!r}")
    return symbols[0]


def read_csv_from_zip(zip_path: Path, member: str) -> list[dict[str, str]]:
    with zipfile.ZipFile(zip_path) as zf:
        return list(csv.DictReader(zf.read(member).decode("utf-8-sig").splitlines()))


def find_card_csv() -> Path | None:
    candidates = [ROOT / "EN_Card_Data.csv", ROOT / "kaggle" / "kaggle_extract" / "EN_Card_Data.csv", Path("/kaggle/input/pokemon-tcg-ai-battle/EN_Card_Data.csv"), Path("/kaggle/input/competitions/pokemon-tcg-ai-battle/EN_Card_Data.csv")]
    for path in candidates:
        if path.exists():
            return path
    return next(iter(sorted(Path("/kaggle/input").rglob("EN_Card_Data.csv"))), None) if Path("/kaggle/input").exists() else None


def load_csv_rows() -> tuple[list[dict[str, str]], str, str]:
    path = find_card_csv()
    if path is not None:
        raw = path.read_bytes()
        rows = list(csv.DictReader(raw.decode("utf-8-sig").splitlines()))
        source = str(path)
    else:
        archive = ROOT / "pokemon-tcg-ai-battle.zip"
        if not archive.exists():
            raise FileNotFoundError("EN_Card_Data.csv not found")
        rows = read_csv_from_zip(archive, "EN_Card_Data.csv")
        raw = zipfile.ZipFile(archive).read("EN_Card_Data.csv")
        source = f"{archive}::EN_Card_Data.csv"
    if list(rows[0]) != SOURCE_COLUMNS:
        raise ValueError(f"CSV columns changed: {list(rows[0])}")
    return rows, source, hashlib.sha256(raw).hexdigest()


def load_cg_data(required: bool = True) -> tuple[dict[int, Any], dict[int, Any]]:
    import_roots = [ROOT]

    local_runtime = ROOT / "kaggle" / "datasets" / "cg_runtime"
    if (local_runtime / "cg" / "__init__.py").exists():
        import_roots.append(local_runtime)

    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        for cg_package in sorted(kaggle_input.glob("*/cg/__init__.py")):
            import_roots.append(cg_package.parent.parent)

    for path in import_roots:
        path_text = str(path)
        if path_text not in sys.path:
            sys.path.insert(0, path_text)

    try:
        api = importlib.import_module("cg.api")
        cards = {int(x.cardId): x for x in api.all_card_data()}
        attacks = {int(x.attackId): x for x in api.all_attack()}
    except Exception as exc:
        if required:
            searched = "\n".join(f"- {path}" for path in import_roots)
            raise RuntimeError(
                "cg runtime is required for strict card preprocessing.\n"
                f"Searched import roots:\n{searched}"
            ) from exc
        return {}, {}

    if required and (not cards or not attacks):
        raise RuntimeError("cg runtime returned empty card or attack tables")

    return cards, attacks


def _detail_type(row: dict[str, str]) -> str | None:
    fields = [normalize_missing(row[x]) for x in ("Move Name", "Cost", "Damage", "Effect Explanation")]
    if not any(fields):
        return None
    name = normalize_missing(row["Move Name"])
    if name.startswith("[Ability]"):
        return "ABILITY"
    if normalize_missing(row["Category"]) == "Fossil":
        return "CARD_EFFECT"
    if normalize_missing(row["Cost"]) or normalize_missing(row["Damage"]):
        return "ATTACK"
    return "CARD_EFFECT"


def _detail_subtype(row: dict[str, str], detail_type: str) -> str:
    kind, tag = normalize_missing(row["Stage (Pokémon)/Type (Energy and Trainer)"]), normalize_missing(row["Category"])
    if detail_type == "ATTACK":
        return "TOOL_ATTACK" if "Tool" in kind else "POKEMON_ATTACK"
    if detail_type == "ABILITY":
        return "FOSSIL_ABILITY" if tag == "Fossil" else "POKEMON_ABILITY"
    if tag == "Fossil":
        return "FOSSIL_EFFECT"
    if tag == "Technical Machine":
        return "TECHNICAL_MACHINE_EFFECT"
    card_type, _stage = card_type_and_stage(kind)
    if card_type == "SPECIAL_ENERGY":
        return "SPECIAL_ENERGY_EFFECT"
    return f"{card_type}_EFFECT"


def _norm_identity(text: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", normalized_text(text).casefold())


def build_corpus() -> tuple[list[CardRecord], list[DetailRecord], list[int], dict[str, Any]]:
    rows, source, source_sha = load_csv_rows()
    cg_cards, cg_attacks = load_cg_data(required=True)
    grouped: dict[int, list[tuple[int, dict[str, str]]]] = defaultdict(list)
    for source_row, row in enumerate(rows):
        card_id = parse_int(row["Card ID"])
        if card_id is None:
            raise ValueError(f"source row {source_row} has no Card ID")
        grouped[card_id].append((source_row, row))
    if len(grouped) != len(cg_cards):
        raise ValueError(f"CSV/CG card count mismatch: {len(grouped)} != {len(cg_cards)}")

    details: list[DetailRecord] = []
    offsets = [0]
    corrections: list[dict[str, Any]] = []
    pending_cards: list[dict[str, Any]] = []
    fossil_targets_by_name: dict[str, set[str]] = defaultdict(set)

    for card_id in sorted(grouped):
        entries = grouped[card_id]
        first = entries[0][1]
        for _, row in entries[1:]:
            for column in CARD_LEVEL_COLUMNS:
                if row[column] != first[column]:
                    raise ValueError(f"card {card_id} has inconsistent {column}")
        cg_card = cg_cards.get(card_id)
        if cg_card is None:
            raise ValueError(f"card {card_id} missing from CG")
        kind = normalized_text(first["Stage (Pokémon)/Type (Energy and Trainer)"])
        card_type, stage = card_type_and_stage(kind)
        expected_cg_type = CG_CARD_TYPES[int(cg_card.cardType)]
        if card_type != expected_cg_type:
            raise ValueError(f"card {card_id} type mismatch: CSV={card_type} CG={expected_cg_type}")
        if _norm_identity(first["Card Name"]) != _norm_identity(cg_card.name):
            raise ValueError(f"card {card_id} name mismatch")

        attack_rows = [(source_row, row) for source_row, row in entries if _detail_type(row) == "ATTACK"]
        attack_ids = [int(x) for x in cg_card.attacks or []]
        if len(attack_rows) != len(attack_ids):
            raise ValueError(f"card {card_id} attack count mismatch: {len(attack_rows)} != {len(attack_ids)}")
        attack_binding = {source_row: aid for (source_row, _), aid in zip(attack_rows, attack_ids)}

        detail_start = len(details)
        for source_row, row in entries:
            detail_type = _detail_type(row)
            if detail_type is None:
                continue
            source_text = normalize_missing(row["Effect Explanation"])
            text = source_text
            move_name = normalize_missing(row["Move Name"])
            if detail_type == "ABILITY":
                detail_name = re.sub(
                    r"^\[Ability\]\s*",
                    "",
                    move_name,
                    flags=re.IGNORECASE,
                ).strip()
            else:
                detail_name = move_name
            attack_id = attack_binding.get(source_row)
            counts = [0] * len(ENERGY_TYPES)
            damage_base, damage_mode = parse_damage_fields(row["Damage"])
            row_corrections: list[str] = []
            if detail_type == "ATTACK":
                attack = cg_attacks.get(int(attack_id))
                if attack is None:
                    raise ValueError(f"card {card_id} missing attack {attack_id}")
                if card_id != 979 and _norm_identity(move_name) != _norm_identity(attack.name):
                    raise ValueError(f"card {card_id} attack name mismatch: {move_name} != {attack.name}")
                counts = cg_energy_counts(list(attack.energies or []))
                csv_counts = energy_counts(row["Cost"])
                if counts != csv_counts:
                    raise ValueError(f"card {card_id} attack {attack_id} cost mismatch: {csv_counts} != {counts}")
                if damage_mode in {"FIXED", "NONE"} and (damage_base if damage_base is not None else 0) != int(attack.damage):
                    raise ValueError(f"card {card_id} attack {attack_id} damage mismatch: {damage_base} != {attack.damage}")
                if card_id in {480, 481}:
                    text = normalize_missing(attack.text)
                    row_corrections.append("CG_ENGLISH_TEXT_REPLACEMENT")
                if card_id == 979:
                    row_corrections.append("FIXED_ATTACK_ID_BINDING_979")
            elif detail_type == "ABILITY":
                if card_id == 481:
                    matching = [skill for skill in (cg_card.skills or []) if _norm_identity(skill.name) == _norm_identity(detail_name)]
                    if len(matching) != 1:
                        raise ValueError(f"card {card_id} ability CG binding failed: {move_name}")
                    text = normalize_missing(matching[0].text)
                    row_corrections.append("CG_ENGLISH_TEXT_REPLACEMENT")
            detail = DetailRecord(
                detail_id=len(details), detail_index=len(details), card_id=card_id,
                source_row=source_row, source_line=source_row + 2,
                detail_type=detail_type, detail_subtype=_detail_subtype(row, detail_type), move_name=move_name,
                detail_name=detail_name, text=text, source_text=source_text, attack_id=attack_id, energy_counts=counts,
                damage_raw=normalize_missing(row["Damage"]) or None, damage_base=damage_base,
                damage_mode=damage_mode, corrections=row_corrections,
                source_fields={column: row[column] for column in SOURCE_COLUMNS[13:]},
            )
            details.append(detail)
            if row_corrections:
                corrections.append({"card_id": card_id, "source_line": source_row + 2, "corrections": row_corrections, "source_text_sha256": hashlib.sha256(source_text.encode()).hexdigest(), "final_text_sha256": hashlib.sha256(text.encode()).hexdigest()})

        rule = normalized_rule(first["Rule"])
        cg_rule_checks = {
            "POKEMON_EX": bool(cg_card.ex),
            "MEGA_POKEMON_EX": bool(cg_card.megaEx),
            "ACE_SPEC": bool(cg_card.aceSpec),
        }
        if rule is not None and not cg_rule_checks[rule]:
            raise ValueError(f"card {card_id} rule mismatch: CSV={rule} CG={cg_rule_checks}")
        if rule is None and any(cg_rule_checks.values()):
            raise ValueError(f"card {card_id} has CG rule flags but no CSV Rule: {cg_rule_checks}")

        category = normalized_text(first["Category"]) or None
        type_value = normalized_text(first["Type"])
        if card_type == "POKEMON":
            card_type_value = normalized_type(type_value)
            expected_pokemon_type = CG_ENERGY_TYPES.get(int(cg_card.energyType))
            if expected_pokemon_type == "N":
                expected_pokemon_type = "DRAGON"
            if card_type_value != expected_pokemon_type:
                raise ValueError(
                    f"card {card_id} Pokémon type mismatch: CSV={card_type_value} CG={expected_pokemon_type}"
                )
        elif card_type == "BASIC_ENERGY":
            card_type_value = normalized_type(type_value)
        else:
            card_type_value = None

        hp = parse_int(first["HP"])
        weakness_type = normalized_energy_symbol(first["Weakness"])
        resistance_type = normalized_energy_symbol(first["Resistance (Type)"])
        retreat_cost = (parse_int(first["Retreat"]) or 0) if card_type == "POKEMON" else None

        previous_stage = normalized_text(first["Previous stage"])
        evolves_from = previous_stage or None if card_type == "POKEMON" else None
        if category == "Fossil":
            match = re.fullmatch(r"Evolve to\s+(.+)", previous_stage, flags=re.IGNORECASE)
            if match is None:
                raise ValueError(f"fossil card {card_id} has invalid Previous stage: {previous_stage!r}")
            fossil_targets_by_name[normalized_text(first["Card Name"])].add(normalized_text(match.group(1)))

        detail_ids = list(range(detail_start, len(details)))
        pending_cards.append({
            "card_id": card_id,
            "name": normalized_text(first["Card Name"]),
            "card_type": card_type,
            "stage": stage,
            "rule": rule,
            "category": category,
            "type": card_type_value,
            "hp": hp,
            "weakness_type": weakness_type,
            "resistance_type": resistance_type,
            "retreat_cost": retreat_cost,
            "evolves_from": evolves_from,
            "detail_ids": detail_ids,
            "expansion": normalized_text(first["Expansion"]),
            "collection_no": normalized_text(first["Collection No."]),
            "source_fields": {column: first[column] for column in CARD_LEVEL_COLUMNS},
        })
        offsets.append(len(details))

    reverse_evolutions: dict[str, set[str]] = defaultdict(set)
    for card in pending_cards:
        if card["card_type"] == "POKEMON" and card["evolves_from"]:
            reverse_evolutions[card["evolves_from"]].add(card["name"])
    for fossil_name, explicit_targets in fossil_targets_by_name.items():
        reverse_targets = reverse_evolutions.get(fossil_name, set())
        if explicit_targets != reverse_targets:
            raise ValueError(
                f"fossil evolution mismatch for {fossil_name}: explicit={sorted(explicit_targets)} "
                f"reverse={sorted(reverse_targets)}"
            )

    cards = [
        CardRecord(
            **card,
            evolves_to=sorted(reverse_evolutions.get(card["name"], set()) | fossil_targets_by_name.get(card["name"], set())),
        )
        for card in pending_cards
    ]

    for card in cards:
        if card.card_type != "SPECIAL_ENERGY":
            continue
        if card.type is not None or not card.detail_ids:
            raise ValueError(f"special energy card {card.card_id} has invalid type/detail routing")
        if any(
            details[detail_id].detail_type != "CARD_EFFECT"
            or details[detail_id].detail_subtype != "SPECIAL_ENERGY_EFFECT"
            for detail_id in card.detail_ids
        ):
            raise ValueError(f"special energy card {card.card_id} has invalid detail classification")

    type_counts = Counter(x.detail_type for x in details)
    expected = {"ATTACK": 1556, "ABILITY": 223, "CARD_EFFECT": 235}
    if len(rows) != 2022 or len(cards) != 1267 or len(details) != 2014 or dict(type_counts) != expected or len(offsets) != 1268:
        raise ValueError(f"canonical corpus invariant failed: rows={len(rows)} cards={len(cards)} details={len(details)} types={dict(type_counts)} offsets={len(offsets)}")
    manifest = {
        "schema_version": CORPUS_SCHEMA_VERSION,
        "corpus_schema_version": CORPUS_SCHEMA_VERSION,
        "card_record_schema_version": CORPUS_SCHEMA_VERSION,
        "source": source, "source_sha256": source_sha, "source_rows": len(rows),
        "card_count": len(cards), "detail_count": len(details), "detail_type_counts": dict(type_counts),
        "cg_cards_loaded": len(cg_cards), "cg_attacks_loaded": len(cg_attacks), "corrections": corrections,
        "card_record_fields": [field.name for field in fields(CardRecord)],
        "detail_record_fields": [field.name for field in fields(DetailRecord)],
        "metadata_only_fields": ["card_id", "expansion", "collection_no", "source_fields"],
        "type_values": sorted({
            value
            for card in cards
            for value in (card.type, card.weakness_type, card.resistance_type)
            if value is not None
        }),
        "special_energy_detail_contract": {
            "card_type": "SPECIAL_ENERGY",
            "type": None,
            "detail_type": "CARD_EFFECT",
            "detail_subtype": "SPECIAL_ENERGY_EFFECT",
        },
        "source_column_contract": {
            "Card ID": "stable identity and replay mapping",
            "Card Name": "full card name",
            "Expansion": "metadata",
            "Collection No.": "metadata",
            "Stage (Pokémon)/Type (Energy and Trainer)": "card_type + stage",
            "Rule": "single rule",
            "Category": "complete category string",
            "Previous stage": "evolves_from / evolves_to",
            "HP": "raw card HP",
            "Type": "shared type for Pokémon and Basic Energy; audit-only for Special Energy",
            "Weakness": "weakness type only",
            "Resistance (Type)": "resistance type only",
            "Retreat": "Pokémon retreat cost",
            "Move Name": "independent DetailRecord",
            "Cost": "independent DetailRecord",
            "Damage": "independent DetailRecord",
            "Effect Explanation": "independent DetailRecord",
        },
    }
    return cards, details, offsets, manifest


def build_records() -> tuple[list[CardRecord], dict[str, Any]]:
    cards, _details, _offsets, manifest = build_corpus()
    return cards, manifest


def summarize_records(records: list[CardRecord]) -> dict[str, Any]:
    count = len(records)
    nullable = [
        "stage", "rule", "category", "type", "hp", "weakness_type", "resistance_type",
        "retreat_cost", "evolves_from",
    ]
    name_lengths = [len(record.name) for record in records]
    return {
        "card_count": count,
        "card_type_counts": dict(Counter(record.card_type for record in records)),
        "type_value_counts": dict(Counter(record.type for record in records if record.type is not None)),
        "missing_field_ratio": {
            field_name: round(sum(getattr(record, field_name) is None for record in records) / count, 4)
            for field_name in nullable
        },
        "name_length": {
            "min": min(name_lengths),
            "mean": round(mean(name_lengths), 2),
            "max": max(name_lengths),
        },
    }


def write_card_cache(cache_dir: Path = DEFAULT_CACHE_DIR) -> dict[str, Any]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cards, details, offsets, manifest = build_corpus()
    mapping = {card.card_id: index for index, card in enumerate(cards)}
    summary = {**summarize_records(cards), **manifest}
    files = {
        "card_records.json": [asdict(x) for x in cards], "cards.json": [asdict(x) for x in cards],
        "details.json": [asdict(x) for x in details], "detail_offsets.json": offsets,
        "card_id_to_index.json": mapping, "card_preprocess_summary.json": summary, "preprocess_manifest.json": manifest,
    }
    for name, payload in files.items():
        (cache_dir / name).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def load_or_create_corpus(cache_dir: Path = DEFAULT_CACHE_DIR, rebuild: bool = False):
    required = [cache_dir / x for x in ("cards.json", "details.json", "detail_offsets.json", "card_id_to_index.json", "preprocess_manifest.json")]
    if not rebuild and all(x.exists() for x in required):
        try:
            cached_manifest = json.loads(required[4].read_text(encoding="utf-8"))
            rebuild = cached_manifest.get("schema_version") != CORPUS_SCHEMA_VERSION
        except (OSError, ValueError, TypeError):
            rebuild = True
    if rebuild or not all(x.exists() for x in required):
        write_card_cache(cache_dir)
    cards = json.loads(required[0].read_text(encoding="utf-8"))
    details = json.loads(required[1].read_text(encoding="utf-8"))
    offsets = json.loads(required[2].read_text(encoding="utf-8"))
    mapping = json.loads(required[3].read_text(encoding="utf-8"))
    manifest = json.loads(required[4].read_text(encoding="utf-8"))
    if len(offsets) != len(cards) + 1 or offsets[-1] != len(details):
        raise ValueError("card cache has invalid detail offsets")
    for index, card in enumerate(cards):
        expected_ids = list(range(offsets[index], offsets[index + 1]))
        if card.get("detail_ids") != expected_ids:
            raise ValueError(f"card {card.get('card_id')} detail_ids do not match detail_offsets")
        if any(details[detail_id].get("detail_id") != detail_id for detail_id in expected_ids):
            raise ValueError(f"card {card.get('card_id')} points to an invalid detail_id")
        if any(details[detail_id].get("card_id") != card.get("card_id") for detail_id in expected_ids):
            raise ValueError(f"card {card.get('card_id')} detail_ids point to another card")
    return cards, details, offsets, mapping, manifest


def load_or_create_records(cache_dir: Path = DEFAULT_CACHE_DIR, rebuild: bool = False):
    cards, _details, _offsets, mapping, manifest = load_or_create_corpus(cache_dir, rebuild)
    summary_path = cache_dir / "card_preprocess_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else manifest
    return cards, {str(k): int(v) for k, v in mapping.items()}, summary

def main() -> None:
    summary = write_card_cache()

    print(
        json.dumps(
            {
                "output_dir": str(DEFAULT_CACHE_DIR),
                "source": summary["source"],
                "card_count": summary["card_count"],
                "detail_count": summary["detail_count"],
                "detail_type_counts": summary["detail_type_counts"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
