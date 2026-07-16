from __future__ import annotations

import csv
import hashlib
import importlib
import json
import re
import sys
import zipfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE_DIR = ROOT / "artifacts" / "card_data"
ENERGY_TYPES = ["C", "G", "R", "W", "L", "P", "F", "D", "M", "N", "Y", "A"]
CG_ENERGY_TYPES = {0: "C", 1: "G", 2: "R", 3: "W", 4: "L", 5: "P", 6: "F", 7: "D", 8: "M", 9: "N", 10: "Y"}
CG_CARD_TYPES = {0: "POKEMON", 1: "ITEM", 2: "TOOL", 3: "SUPPORTER", 4: "STADIUM", 5: "BASIC_ENERGY", 6: "SPECIAL_ENERGY"}
NULL_TOKEN = "<NULL>"
UNK_TOKEN = "<UNK>"
MASK_TOKEN = "<MASK>"
SOURCE_COLUMNS = [
    "Card ID", "Card Name", "Expansion", "Collection No.", "Stage (Pokémon)/Type (Energy and Trainer)",
    "Rule", "Category", "Previous stage", "HP", "Type", "Weakness", "Resistance (Type)", "Retreat",
    "Move Name", "Cost", "Damage", "Effect Explanation",
]
CARD_LEVEL_COLUMNS = SOURCE_COLUMNS[:13]


@dataclass
class DetailRecord:
    detail_index: int
    card_id: str
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
    card_id: str
    name: str
    expansion: str
    collection_no: str
    category: str
    card_type: str
    subtype: str | None
    card_tags: list[str]
    pokemon_type: str | None
    stage: str | None
    hp: int | None
    hp_applicability: str
    retreat_cost: int | None
    weakness_type: str | None
    weakness_value: float | None
    resistance_type: str | None
    resistance_value: float | None
    evolves_from: str | None
    evolves_to: list[str]
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
    detail_start: int
    detail_end: int
    source_fields: dict[str, str] = field(default_factory=dict)


def stable_hash(text: str, modulo: int) -> int:
    return int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16) % modulo


def normalize_missing(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if not text or text.lower() in {"n/a", "nan", "none", "-", "—"} else text


def normalized_text(value: Any) -> str:
    return re.sub(r"\s+", " ", normalize_missing(value).replace("\xa0", " ")).strip()


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


def parse_damage(value: Any) -> int | None:
    return parse_damage_fields(value)[0]


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


def energy_cost_dict(value: Any) -> dict[str, int]:
    return {name: count for name, count in zip(ENERGY_TYPES, energy_counts(value)) if count}


def cg_energy_counts(values: list[Any]) -> list[int]:
    result = [0] * len(ENERGY_TYPES)
    for value in values or []:
        key = CG_ENERGY_TYPES.get(int(value))
        if key is None:
            raise ValueError(f"unknown CG energy enum: {value!r}")
        result[ENERGY_TYPES.index(key)] += 1
    return result


def split_flags(value: Any) -> list[str]:
    text = normalize_missing(value)
    return sorted({x.strip() for x in re.split(r"[,;/]+", text) if x.strip()}) if text else []


def normalized_card_tags(value: Any) -> list[str]:
    text = normalize_missing(value)
    if not text:
        return []
    if text.startswith("Tera(") and text.endswith(")"):
        return ["TERA", f"TERA_{text[5:-1].upper()}"]
    if text.startswith("Trainer's Pokémon"):
        owner = re.search(r"[（(]([^）)]+)[）)]", text)
        owner_tag = re.sub(r"[^A-Z0-9]+", "_", owner.group(1).upper()).strip("_") if owner else "UNKNOWN"
        return ["TRAINERS_POKEMON", f"OWNER_{owner_tag}"]
    return [re.sub(r"[^A-Z0-9]+", "_", text.upper()).strip("_")]


def card_type_from_kind(kind: str) -> str:
    lower = kind.lower()
    if "basic energy" in lower:
        return "BASIC_ENERGY"
    if "special energy" in lower:
        return "SPECIAL_ENERGY"
    if "tool" in lower:
        return "TOOL"
    if "supporter" in lower:
        return "SUPPORTER"
    if "stadium" in lower:
        return "STADIUM"
    if "item" in lower:
        return "ITEM"
    if "pok" in lower:
        return "POKEMON"
    return kind.upper().replace(" ", "_") if kind else "UNKNOWN"


def broad_category(card_type: str) -> str:
    if card_type == "POKEMON":
        return "POKEMON"
    if card_type in {"ITEM", "TOOL", "SUPPORTER", "STADIUM"}:
        return "TRAINER"
    if card_type in {"BASIC_ENERGY", "SPECIAL_ENERGY"}:
        return "ENERGY"
    return "UNKNOWN"


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
        return list(csv.DictReader(zf.read(member).decode("utf-8-sig").splitlines()))


def find_card_csv() -> Path | None:
    candidates = [ROOT / "EN_Card_Data.csv", ROOT / "kaggle_extract" / "EN_Card_Data.csv", Path("/kaggle/input/pokemon-tcg-ai-battle/EN_Card_Data.csv"), Path("/kaggle/input/competitions/pokemon-tcg-ai-battle/EN_Card_Data.csv")]
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

    local_runtime = ROOT / "kaggle_cg_runtime_dataset"
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
    if normalize_missing(row["Category"]) == "Fossil":
        return "CARD_EFFECT"
    name = normalize_missing(row["Move Name"])
    if name.startswith("[Ability]"):
        return "ABILITY"
    if normalize_missing(row["Cost"]) or normalize_missing(row["Damage"]):
        return "ATTACK"
    return "CARD_EFFECT"


def _detail_subtype(row: dict[str, str], detail_type: str) -> str:
    kind, tag = normalize_missing(row["Stage (Pokémon)/Type (Energy and Trainer)"]), normalize_missing(row["Category"])
    if detail_type == "ATTACK":
        return "TOOL_ATTACK" if "Tool" in kind else "POKEMON_ATTACK"
    if detail_type == "ABILITY":
        return "POKEMON_ABILITY"
    if tag == "Fossil":
        return "FOSSIL_EFFECT"
    if tag == "Technical Machine":
        return "TECHNICAL_MACHINE_EFFECT"
    return f"{card_type_from_kind(kind)}_EFFECT"


def _norm_identity(text: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", normalized_text(text).casefold())


def build_corpus() -> tuple[list[CardRecord], list[DetailRecord], list[int], dict[str, Any]]:
    rows, source, source_sha = load_csv_rows()
    cg_cards, cg_attacks = load_cg_data(required=True)
    grouped: dict[str, list[tuple[int, dict[str, str]]]] = defaultdict(list)
    for source_row, row in enumerate(rows):
        grouped[normalize_missing(row["Card ID"])].append((source_row, row))
    if len(grouped) != len(cg_cards):
        raise ValueError(f"CSV/CG card count mismatch: {len(grouped)} != {len(cg_cards)}")

    cards: list[CardRecord] = []
    details: list[DetailRecord] = []
    offsets = [0]
    corrections = []
    evolves_to: dict[str, set[str]] = defaultdict(set)
    for _, entries in grouped.items():
        first = entries[0][1]
        parent = normalize_missing(first["Previous stage"])
        if parent:
            evolves_to[parent].add(normalize_missing(first["Card Name"]))

    for card_id in sorted(grouped, key=int):
        entries = grouped[card_id]
        first = entries[0][1]
        for _, row in entries[1:]:
            for column in CARD_LEVEL_COLUMNS:
                if normalized_text(row[column]) != normalized_text(first[column]):
                    raise ValueError(f"card {card_id} has inconsistent {column}")
        cid = int(card_id)
        cg_card = cg_cards.get(cid)
        if cg_card is None:
            raise ValueError(f"card {card_id} missing from CG")
        kind = normalize_missing(first["Stage (Pokémon)/Type (Energy and Trainer)"])
        card_type = card_type_from_kind(kind)
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

        attack_names, attack_texts, attack_damage, attack_costs = [], [], [], []
        ability_texts = []
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
           ##########################
            attack_id = attack_binding.get(source_row)
            counts = [0] * len(ENERGY_TYPES)
            damage_base, damage_mode = parse_damage_fields(row["Damage"])
            row_corrections: list[str] = []
            if detail_type == "ATTACK":
                attack = cg_attacks.get(int(attack_id))
                if attack is None:
                    raise ValueError(f"card {card_id} missing attack {attack_id}")
                if cid != 979 and _norm_identity(move_name) != _norm_identity(attack.name):
                    raise ValueError(f"card {card_id} attack name mismatch: {move_name} != {attack.name}")
                counts = cg_energy_counts(list(attack.energies or []))
                csv_counts = energy_counts(row["Cost"])
                if counts != csv_counts:
                    raise ValueError(f"card {card_id} attack {attack_id} cost mismatch: {csv_counts} != {counts}")
                if damage_mode in {"FIXED", "NONE"} and (damage_base if damage_base is not None else 0) != int(attack.damage):
                    raise ValueError(f"card {card_id} attack {attack_id} damage mismatch: {damage_base} != {attack.damage}")
                if cid in {480, 481}:
                    text = normalize_missing(attack.text)
                    row_corrections.append("CG_ENGLISH_TEXT_REPLACEMENT")
                if cid == 979:
                    row_corrections.append("FIXED_ATTACK_ID_BINDING_979")
                attack_names.append(move_name)
                attack_texts.append(text)
                attack_damage.append(damage_base)
                attack_costs.append({name: count for name, count in zip(ENERGY_TYPES, counts) if count})
            elif detail_type == "ABILITY":
                if cid == 481:
                    
                    matching = [skill for skill in (cg_card.skills or []) if _norm_identity(skill.name) == _norm_identity(source_ability_name)]
                    if len(matching) != 1:
                        raise ValueError(f"card {card_id} ability CG binding failed: {move_name}")
                    text = normalize_missing(matching[0].text)
                    row_corrections.append("CG_ENGLISH_TEXT_REPLACEMENT")
                ability_texts.append(text)
            detail = DetailRecord(
                detail_index=len(details), card_id=card_id, source_row=source_row, source_line=source_row + 2,
                detail_type=detail_type, detail_subtype=_detail_subtype(row, detail_type), move_name=move_name,
                detail_name=detail_name ,text=text, source_text=source_text, attack_id=attack_id, energy_counts=counts,
                damage_raw=normalize_missing(row["Damage"]) or None, damage_base=damage_base,
                damage_mode=damage_mode, corrections=row_corrections,
                source_fields={column: row[column] for column in SOURCE_COLUMNS[13:]},
            )
            details.append(detail)
            if row_corrections:
                corrections.append({"card_id": card_id, "source_line": source_row + 2, "corrections": row_corrections, "source_text_sha256": hashlib.sha256(source_text.encode()).hexdigest(), "final_text_sha256": hashlib.sha256(text.encode()).hexdigest()})

        category_value = normalize_missing(first["Category"])
        tags = normalized_card_tags(category_value)
        rule_flags = []
        for flag, enabled in (("POKEMON_EX", cg_card.ex), ("MEGA_POKEMON_EX", cg_card.megaEx), ("TERA", cg_card.tera), ("ACE_SPEC", cg_card.aceSpec)):
            if enabled:
                rule_flags.append(flag)
        rule_flags = sorted(set(rule_flags))
        source_rule = normalize_missing(first["Rule"])
        expected_source_flag = {"Pokémon ex": "POKEMON_EX", "Mega Pokémon ex": "MEGA_POKEMON_EX", "ACE SPEC": "ACE_SPEC"}.get(source_rule)
        if source_rule and expected_source_flag not in rule_flags:
            raise ValueError(f"card {card_id} rule mismatch: {source_rule} vs {rule_flags}")
        provided = parse_energy_symbols(first["Type"]) if "ENERGY" in card_type else []
        pokemon_type = CG_ENERGY_TYPES.get(int(cg_card.energyType)) if card_type == "POKEMON" else None
        hp = parse_int(first["HP"])
        if hp is None and card_type in {"ITEM", "TOOL"} and int(cg_card.hp) > 0:
            hp = int(cg_card.hp)
        hp_applicability = "PLAYABLE_AS_POKEMON" if category_value == "Fossil" else ("POKEMON" if card_type == "POKEMON" else "NOT_APPLICABLE")
        effects = [detail.text for detail in details[offsets[-1]:] if detail.text]
        cards.append(CardRecord(
            card_id=card_id, name=normalize_missing(first["Card Name"]), expansion=normalize_missing(first["Expansion"]),
            collection_no=normalize_missing(first["Collection No."]), category=broad_category(card_type), card_type=card_type,
            subtype=kind or None, card_tags=tags, pokemon_type=pokemon_type, stage=stage_from_kind(kind), hp=hp,
            hp_applicability=hp_applicability, retreat_cost=parse_int(first["Retreat"]),
            weakness_type=(parse_energy_symbols(first["Weakness"]) or [None])[0], weakness_value=2.0 if normalize_missing(first["Weakness"]) else None,
            resistance_type=(parse_energy_symbols(first["Resistance (Type)"]) or [None])[0], resistance_value=-30.0 if normalize_missing(first["Resistance (Type)"]) else None,
            evolves_from=normalize_missing(first["Previous stage"]) or None, evolves_to=sorted(evolves_to.get(normalize_missing(first["Card Name"]), set())),
            rule_flags=rule_flags, ability_texts=ability_texts, attack_texts=attack_texts, attack_names=attack_names,
            attack_damage=attack_damage, attack_energy_costs=attack_costs, trainer_type=trainer_type_from_card_type(card_type),
            provided_energy_types=provided, full_effect_text="\n".join(effects), attack_ids=attack_ids,
            detail_start=offsets[-1], detail_end=len(details),
            source_fields={column: first[column] for column in CARD_LEVEL_COLUMNS},
        ))
        offsets.append(len(details))

    type_counts = Counter(x.detail_type for x in details)
    expected = {"ATTACK": 1556, "ABILITY": 218, "CARD_EFFECT": 240}
    if len(rows) != 2022 or len(cards) != 1267 or len(details) != 2014 or dict(type_counts) != expected or len(offsets) != 1268:
        raise ValueError(f"canonical corpus invariant failed: rows={len(rows)} cards={len(cards)} details={len(details)} types={dict(type_counts)} offsets={len(offsets)}")
    manifest = {
        "schema_version": 3, "source": source, "source_sha256": source_sha, "source_rows": len(rows),
        "card_count": len(cards), "detail_count": len(details), "detail_type_counts": dict(type_counts),
        "cg_cards_loaded": len(cg_cards), "cg_attacks_loaded": len(cg_attacks), "corrections": corrections,
        "source_column_contract": {
            "Card ID": "card identity, ordering, mapping; metadata only", "Card Name": "card identity feature and metadata",
            "Expansion": "metadata", "Collection No.": "metadata", "Stage (Pokémon)/Type (Energy and Trainer)": "category/stage/subtype",
            "Rule": "rule flags", "Category": "card tags", "Previous stage": "evolves_from/evolves_to",
            "HP": "raw/normalized/presence/applicability", "Type": "pokemon type or provided energy",
            "Weakness": "weakness type", "Resistance (Type)": "resistance type", "Retreat": "raw/normalized/presence",
            "Move Name": "detail name feature, detail classification, CG alignment, and source metadata", "Cost": "12-dimensional energy counts",
            "Damage": "raw/base/mode/presence", "Effect Explanation": "detail text",
        },
    }
    return cards, details, offsets, manifest


def build_records() -> tuple[list[CardRecord], dict[str, Any]]:
    cards, _details, _offsets, manifest = build_corpus()
    return cards, manifest


def summarize_records(records: list[CardRecord]) -> dict[str, Any]:
    count = len(records)
    nullable = ["pokemon_type", "stage", "hp", "retreat_cost", "weakness_type", "resistance_type", "evolves_from"]
    lengths = [len(" ".join([x.name, x.full_effect_text, *x.ability_texts, *x.attack_names, *x.attack_texts])) for x in records]
    return {"card_count": count, "type_counts": dict(Counter(x.card_type for x in records)), "missing_field_ratio": {field: round(sum(getattr(x, field) is None for x in records) / count, 4) for field in nullable}, "text_length": {"min": min(lengths), "mean": round(mean(lengths), 2), "max": max(lengths)}}


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
    if rebuild or not all(x.exists() for x in required):
        write_card_cache(cache_dir)
    return (
        json.loads(required[0].read_text(encoding="utf-8")), json.loads(required[1].read_text(encoding="utf-8")),
        json.loads(required[2].read_text(encoding="utf-8")), json.loads(required[3].read_text(encoding="utf-8")),
        json.loads(required[4].read_text(encoding="utf-8")),
    )


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