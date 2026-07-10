from __future__ import annotations

import csv
import hashlib
import importlib
import json
import re
import sys
import zipfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE_DIR = ROOT / "artifacts" / "card_data"
ENERGY_TYPES = ["C", "G", "R", "W", "L", "P", "F", "D", "M", "N", "Y", "A"]
NULL_TOKEN = "<NULL>"
UNK_TOKEN = "<UNK>"


@dataclass
class CardRecord:
    card_id: str
    name: str
    card_type: str
    subtype: str | None
    pokemon_type: str | None
    stage: str | None
    hp: int | None
    retreat_cost: int | None
    weakness_type: str | None
    weakness_value: float | None
    resistance_type: str | None
    resistance_value: float | None
    evolves_from: str | None
    rule_flags: list[str]
    ability_texts: list[str]
    attack_texts: list[str]
    attack_names: list[str]
    attack_damage: list[int | None]
    attack_energy_costs: list[dict[str, int]]
    trainer_type: str | None
    provided_energy_types: list[str]
    full_effect_text: str
    attack_ids: list[int]


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


def parse_int(value: Any) -> int | None:
    text = normalize_missing(value)
    if not text:
        return None
    match = re.search(r"-?\d+", text)
    return int(match.group(0)) if match else None


def parse_damage(value: Any) -> int | None:
    text = normalize_missing(value)
    if not text:
        return None
    numbers = [int(x) for x in re.findall(r"\d+", text)]
    if not numbers:
        return None
    if "x" in text.lower():
        return max(numbers)
    return numbers[0]


def parse_energy_symbols(value: Any) -> list[str]:
    text = normalize_missing(value)
    if not text:
        return []
    symbols = re.findall(r"\{([^}]+)\}", text)
    normalized = []
    for symbol in symbols:
        clean = symbol.strip().upper()
        if clean:
            normalized.append(clean)
    return normalized


def energy_cost_dict(value: Any) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for symbol in parse_energy_symbols(value):
        counts[symbol] += 1
    return dict(sorted(counts.items()))


def split_flags(value: Any) -> list[str]:
    text = normalize_missing(value)
    if not text:
        return []
    parts = re.split(r"[,;/]+", text)
    return sorted({part.strip() for part in parts if part.strip()})


def card_type_from_kind(kind: str) -> str:
    lower = kind.lower()
    if "pok" in lower:
        return "POKEMON"
    if "basic energy" in lower:
        return "BASIC_ENERGY"
    if "special energy" in lower:
        return "SPECIAL_ENERGY"
    if "supporter" in lower:
        return "SUPPORTER"
    if "stadium" in lower:
        return "STADIUM"
    if "tool" in lower:
        return "TOOL"
    if "item" in lower:
        return "ITEM"
    return kind.upper().replace(" ", "_") if kind else "UNKNOWN"


def stage_from_kind(kind: str) -> str | None:
    lower = kind.lower()
    if "basic" in lower and "energy" not in lower:
        return "BASIC"
    if "stage 1" in lower or "stage1" in lower:
        return "STAGE1"
    if "stage 2" in lower or "stage2" in lower:
        return "STAGE2"
    return None


def trainer_type_from_card_type(card_type: str) -> str | None:
    return card_type if card_type in {"ITEM", "TOOL", "SUPPORTER", "STADIUM"} else None


def read_csv_from_zip(zip_path: Path, member: str) -> list[dict[str, str]]:
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(member) as handle:
            text = handle.read().decode("utf-8-sig").splitlines()
    return list(csv.DictReader(text))


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
    csv_path = find_card_csv()
    if csv_path is not None:
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle)), str(csv_path)
    zip_path = ROOT / "pokemon-tcg-ai-battle.zip"
    if zip_path.exists():
        return read_csv_from_zip(zip_path, "EN_Card_Data.csv"), str(zip_path) + "::EN_Card_Data.csv"
    raise FileNotFoundError("EN_Card_Data.csv not found in project, Kaggle input, or pokemon-tcg-ai-battle.zip")


def enum_name(value: Any) -> str | None:
    if value is None:
        return None
    if hasattr(value, "name"):
        return str(value.name)
    text = str(value)
    return text if text and text != "None" else None


def load_cg_data() -> tuple[dict[int, Any], dict[int, Any]]:
    candidate_parents = [ROOT / "outputs"]
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
        return cards, attacks
    except Exception:
        return {}, {}


def build_records() -> tuple[list[CardRecord], dict[str, Any]]:
    rows, source = load_csv_rows()
    cg_cards, cg_attacks = load_cg_data()
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("Card ID", "")).strip()].append(row)

    records: list[CardRecord] = []
    consistency_warnings: list[str] = []
    for card_id in sorted(grouped, key=lambda x: int(x) if x.isdigit() else x):
        card_rows = grouped[card_id]
        first = card_rows[0]
        kind = normalize_missing(first.get("Stage (Pokémon)/Type (Energy and Trainer)"))
        card_type = card_type_from_kind(kind)
        card_int = int(card_id)
        cg_card = cg_cards.get(card_int)

        attack_names: list[str] = []
        attack_texts: list[str] = []
        attack_damage: list[int | None] = []
        attack_energy_costs: list[dict[str, int]] = []
        attack_ids: list[int] = []

        for row in card_rows:
            move_name = normalize_missing(row.get("Move Name"))
            cost = normalize_missing(row.get("Cost"))
            damage = normalize_missing(row.get("Damage"))
            effect = normalize_missing(row.get("Effect Explanation"))
            if move_name or cost or damage:
                attack_names.append(move_name)
                attack_texts.append(effect)
                attack_damage.append(parse_damage(damage))
                attack_energy_costs.append(energy_cost_dict(cost))

        ability_texts: list[str] = []
        if cg_card is not None:
            for skill in getattr(cg_card, "skills", []) or []:
                text = normalize_missing(getattr(skill, "text", ""))
                name = normalize_missing(getattr(skill, "name", ""))
                if text or name:
                    ability_texts.append(" ".join(part for part in [name, text] if part))
            attack_ids = [int(x) for x in getattr(cg_card, "attacks", []) or []]
            if attack_ids:
                attack_names = []
                attack_texts = []
                attack_damage = []
                attack_energy_costs = []
                for attack_id in attack_ids:
                    attack = cg_attacks.get(attack_id)
                    if attack is None:
                        consistency_warnings.append(f"card {card_id} references missing attack {attack_id}")
                        continue
                    attack_names.append(normalize_missing(getattr(attack, "name", "")))
                    attack_texts.append(normalize_missing(getattr(attack, "text", "")))
                    attack_damage.append(parse_damage(getattr(attack, "damage", None)))
                    costs = Counter(enum_name(v) or str(v) for v in getattr(attack, "energies", []) or [])
                    attack_energy_costs.append(dict(sorted(costs.items())))

        provided_energy_types = parse_energy_symbols(first.get("Type"))
        if card_type == "POKEMON" and provided_energy_types:
            pokemon_type = provided_energy_types[0]
        elif cg_card is not None:
            pokemon_type = enum_name(getattr(cg_card, "energyType", None))
        else:
            pokemon_type = None

        effects = [normalize_missing(row.get("Effect Explanation")) for row in card_rows]
        full_effect_text = "\n".join(text for text in effects if text)
        record = CardRecord(
            card_id=card_id,
            name=normalize_missing(first.get("Card Name")) or card_id,
            card_type=card_type,
            subtype=kind or None,
            pokemon_type=pokemon_type,
            stage=stage_from_kind(kind),
            hp=parse_int(first.get("HP")) if cg_card is None else (int(getattr(cg_card, "hp", 0)) or None),
            retreat_cost=parse_int(first.get("Retreat"))
            if cg_card is None
            else (int(getattr(cg_card, "retreatCost", 0)) if card_type == "POKEMON" else None),
            weakness_type=parse_energy_symbols(first.get("Weakness"))[0] if parse_energy_symbols(first.get("Weakness")) else None,
            weakness_value=2.0 if normalize_missing(first.get("Weakness")) else None,
            resistance_type=parse_energy_symbols(first.get("Resistance (Type)"))[0]
            if parse_energy_symbols(first.get("Resistance (Type)"))
            else None,
            resistance_value=-30.0 if normalize_missing(first.get("Resistance (Type)")) else None,
            evolves_from=normalize_missing(first.get("Previous stage")) or getattr(cg_card, "evolvesFrom", None),
            rule_flags=sorted(set(split_flags(first.get("Rule")) + ([flag for flag in ["ex", "megaEx", "tera", "aceSpec"] if bool(getattr(cg_card, flag, False))] if cg_card is not None else []))),
            ability_texts=ability_texts,
            attack_texts=attack_texts,
            attack_names=attack_names,
            attack_damage=attack_damage,
            attack_energy_costs=attack_energy_costs,
            trainer_type=trainer_type_from_card_type(card_type),
            provided_energy_types=provided_energy_types if "ENERGY" in card_type else [],
            full_effect_text=full_effect_text,
            attack_ids=attack_ids,
        )
        records.append(record)

    metadata = {
        "source": source,
        "cg_cards_loaded": len(cg_cards),
        "cg_attacks_loaded": len(cg_attacks),
        "consistency_warnings": consistency_warnings[:100],
    }
    return records, metadata


def summarize_records(records: list[CardRecord]) -> dict[str, Any]:
    count = len(records)
    type_counts = Counter(record.card_type for record in records)
    nullable_fields = ["pokemon_type", "stage", "hp", "retreat_cost", "weakness_type", "resistance_type", "evolves_from"]
    missing = {
        field: round(sum(1 for record in records if getattr(record, field) is None) / count, 4) if count else 0.0
        for field in nullable_fields
    }
    text_lengths = [
        len(" ".join([record.name, record.full_effect_text, *record.ability_texts, *record.attack_names, *record.attack_texts]))
        for record in records
    ]
    return {
        "card_count": count,
        "type_counts": dict(sorted(type_counts.items())),
        "missing_field_ratio": missing,
        "text_length": {
            "min": min(text_lengths) if text_lengths else 0,
            "mean": round(mean(text_lengths), 2) if text_lengths else 0,
            "max": max(text_lengths) if text_lengths else 0,
        },
    }


def write_card_cache(cache_dir: Path = DEFAULT_CACHE_DIR) -> dict[str, Any]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    records, source_metadata = build_records()
    card_id_to_index = {record.card_id: index for index, record in enumerate(records)}
    name_to_ids: dict[str, list[str]] = defaultdict(list)
    for record in records:
        name_to_ids[record.name].append(record.card_id)

    summary = summarize_records(records)
    summary.update(source_metadata)
    payload = [asdict(record) for record in records]
    (cache_dir / "card_records.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (cache_dir / "card_id_to_index.json").write_text(
        json.dumps(card_id_to_index, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (cache_dir / "card_preprocess_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary


def load_or_create_records(cache_dir: Path = DEFAULT_CACHE_DIR, rebuild: bool = False) -> tuple[list[dict[str, Any]], dict[str, int], dict[str, Any]]:
    records_path = cache_dir / "card_records.json"
    mapping_path = cache_dir / "card_id_to_index.json"
    summary_path = cache_dir / "card_preprocess_summary.json"
    if rebuild or not records_path.exists() or not mapping_path.exists():
        write_card_cache(cache_dir)
    records = json.loads(records_path.read_text(encoding="utf-8"))
    card_id_to_index = {str(k): int(v) for k, v in json.loads(mapping_path.read_text(encoding="utf-8")).items()}
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else summarize_records([CardRecord(**r) for r in records])
    return records, card_id_to_index, summary
