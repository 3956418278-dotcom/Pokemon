#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DECK_DIR = ROOT / "decks"
DECKS_MD = DECK_DIR / "source_decks.txt"
OUTPUT_JSON = DECK_DIR / "baseline_decks.json"
OUTPUT_MD = DECK_DIR / "baseline_decks.md"


SECTION_NAMES = {"Pokémon", "Pokemon", "Trainer", "Energy"}
ENERGY_NAME_TO_ENGINE = {
    "grass energy": "Basic {G} Energy",
    "fire energy": "Basic {R} Energy",
    "water energy": "Basic {W} Energy",
    "lightning energy": "Basic {L} Energy",
    "psychic energy": "Basic {P} Energy",
    "fighting energy": "Basic {F} Energy",
    "darkness energy": "Basic {D} Energy",
    "metal energy": "Basic {M} Energy",
}
MISSING_ENERGY_REPLACEMENTS = {
    "growing grass energy": "Basic {G} Energy",
    "telepathic psychic energy": "Basic {P} Energy",
}
TRAINER_FILLER_NAMES = [
    "Night Stretcher",
    "Poké Pad",
    "Ultra Ball",
    "Buddy-Buddy Poffin",
    "Switch",
    "Lillie's Determination",
    "Boss’s Orders",
]


def normalize_name(value: str) -> str:
    value = unicodedata.normalize("NFKC", value)
    value = value.replace("’", "'").replace("`", "'").replace("´", "'")
    value = value.replace("é", "e").replace("É", "E")
    value = value.lower()
    value = re.sub(r"[\s\-_:]+", " ", value)
    value = re.sub(r"[^a-z0-9 '{}.&]", "", value)
    return value.strip()


def parse_decks(text: str) -> list[dict[str, Any]]:
    decks: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    section: str | None = None

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        section_match = re.match(r"^(Pokémon|Pokemon|Trainer|Energy):\s*(\d+)\s*$", line)
        if section_match:
            section = "Pokemon" if section_match.group(1) in {"Pokémon", "Pokemon"} else section_match.group(1)
            if current is None:
                raise ValueError(f"Section before deck name: {line}")
            current["section_totals"][section] = int(section_match.group(2))
            continue

        card_match = re.match(r"^(\d+)\s+(.+?)\s+([A-Z0-9]{2,5})\s+([A-Za-z0-9]+)\s*$", line)
        if card_match and current is not None and section is not None:
            count = int(card_match.group(1))
            current["cards"].append(
                {
                    "section": section,
                    "count": count,
                    "name": card_match.group(2).strip(),
                    "set": card_match.group(3),
                    "number": card_match.group(4),
                }
            )
            continue

        current = {"name": line, "section_totals": {}, "cards": []}
        decks.append(current)
        section = None

    return decks


def load_engine_cards() -> list[Any]:
    cg_parent = ROOT / "outputs/cg_runtime"
    if not (cg_parent / "cg" / "api.py").exists():
        raise FileNotFoundError(
            "outputs/cg_runtime/cg/api.py not found; build or download the cg runtime first"
        )
    sys.path.insert(0, str(cg_parent))
    from cg.api import all_card_data

    return all_card_data()


def card_type_name(card: Any) -> str:
    from cg.api import CardType

    return CardType(card.cardType).name


def build_name_index(cards: list[Any]) -> dict[str, list[Any]]:
    by_name: dict[str, list[Any]] = defaultdict(list)
    for card in cards:
        by_name[normalize_name(card.name)].append(card)
    return by_name


def match_card(deck_card: dict[str, Any], by_name: dict[str, list[Any]]) -> list[Any]:
    raw_name = deck_card["name"]
    normalized = normalize_name(raw_name)
    mapped_energy = ENERGY_NAME_TO_ENGINE.get(normalized)
    if mapped_energy is not None:
        normalized = normalize_name(mapped_energy)
    return by_name.get(normalized, [])


def first_card_by_name(name: str, by_name: dict[str, list[Any]]) -> Any:
    matches = by_name.get(normalize_name(name), [])
    if not matches:
        raise KeyError(f"Card not found in engine pool: {name}")
    return matches[0]


def replacement_for_missing(
    missing_card: dict[str, Any],
    deck_name_counts: Counter[str],
    by_name: dict[str, list[Any]],
    existing_trainer_names: list[str],
) -> dict[str, Any] | None:
    normalized = normalize_name(missing_card["name"])
    if missing_card["section"] == "Energy" and normalized in MISSING_ENERGY_REPLACEMENTS:
        replacement_name = MISSING_ENERGY_REPLACEMENTS[normalized]
        card = first_card_by_name(replacement_name, by_name)
        deck_name_counts[card.name] += missing_card["count"]
        return {
            "original_name": missing_card["name"],
            "original_set": missing_card["set"],
            "original_number": missing_card["number"],
            "count": missing_card["count"],
            "replacement_kind": "energy_same_type",
            "card_id": int(card.cardId),
            "engine_name": card.name,
            "card_type": card_type_name(card),
            "note": "Missing special Energy replaced with same-type Basic Energy.",
        }

    if missing_card["section"] == "Trainer":
        remaining = int(missing_card["count"])
        replacements = []
        filler_names = list(dict.fromkeys(existing_trainer_names + TRAINER_FILLER_NAMES))
        for filler_name in filler_names:
            card = first_card_by_name(filler_name, by_name)
            available = max(0, 4 - deck_name_counts[card.name])
            if available <= 0:
                continue
            take = min(remaining, available)
            deck_name_counts[card.name] += take
            remaining -= take
            replacements.append(
                {
                    "original_name": missing_card["name"],
                    "original_set": missing_card["set"],
                    "original_number": missing_card["number"],
                    "count": take,
                    "replacement_kind": "trainer_placeholder",
                    "card_id": int(card.cardId),
                    "engine_name": card.name,
                    "card_type": card_type_name(card),
                    "note": "Missing Trainer replaced by available Trainer filler; deck-local Trainers are preferred and this is not a card-equivalent match.",
                }
            )
            if remaining == 0:
                return {"split": replacements}
        return None

    return None


def main() -> None:
    decks = parse_decks(DECKS_MD.read_text(encoding="utf-8"))
    engine_cards = load_engine_cards()
    by_name = build_name_index(engine_cards)

    output: dict[str, Any] = {
        "source": str(DECKS_MD.name),
        "card_pool": {
            "source": "cg.api.all_card_data",
            "unique_cards": len(engine_cards),
        },
        "decks": [],
    }

    for deck in decks:
        present_cards = []
        missing_cards = []
        replacement_cards = []
        expanded_ids: list[int] = []
        patched_expanded_ids: list[int] = []
        present_count = 0
        missing_count = 0
        replaced_count = 0
        unreplaced_count = 0
        deck_name_counts: Counter[str] = Counter()
        existing_trainer_names: list[str] = []

        for deck_card in deck["cards"]:
            matches = match_card(deck_card, by_name)
            record = dict(deck_card)
            if matches:
                chosen = matches[0]
                record.update(
                    {
                        "exists": True,
                        "card_id": int(chosen.cardId),
                        "engine_name": chosen.name,
                        "card_type": card_type_name(chosen),
                        "basic": bool(chosen.basic),
                        "stage1": bool(chosen.stage1),
                        "stage2": bool(chosen.stage2),
                    }
                )
                present_cards.append(record)
                expanded_ids.extend([int(chosen.cardId)] * int(deck_card["count"]))
                patched_expanded_ids.extend([int(chosen.cardId)] * int(deck_card["count"]))
                deck_name_counts[chosen.name] += int(deck_card["count"])
                if record["section"] == "Trainer":
                    existing_trainer_names.append(chosen.name)
                present_count += int(deck_card["count"])
            else:
                record["exists"] = False
                missing_cards.append(record)
                missing_count += int(deck_card["count"])

        for missing_card in missing_cards:
            replacement = replacement_for_missing(
                missing_card,
                deck_name_counts,
                by_name,
                existing_trainer_names,
            )
            if replacement is None:
                unreplaced_count += int(missing_card["count"])
                continue
            replacements = replacement.get("split", [replacement])
            for repl in replacements:
                replacement_cards.append(repl)
                patched_expanded_ids.extend([int(repl["card_id"])] * int(repl["count"]))
                replaced_count += int(repl["count"])

        type_counts = Counter()
        for card in present_cards:
            type_counts[card["card_type"]] += card["count"]
        patched_type_counts = Counter(type_counts)
        for card in replacement_cards:
            patched_type_counts[card["card_type"]] += card["count"]

        output["decks"].append(
            {
                "name": deck["name"],
                "declared_total": sum(deck["section_totals"].values()),
                "present_total": present_count,
                "missing_total": missing_count,
                "replaced_total": replaced_count,
                "unreplaced_total": unreplaced_count,
                "section_totals": deck["section_totals"],
                "present_type_counts": dict(sorted(type_counts.items())),
                "patched_type_counts": dict(sorted(patched_type_counts.items())),
                "present_cards": present_cards,
                "missing_cards": missing_cards,
                "replacement_cards": replacement_cards,
                "deck_ids_if_all_present": expanded_ids if missing_count == 0 and len(expanded_ids) == 60 else [],
                "patched_deck_ids": patched_expanded_ids
                if unreplaced_count == 0 and len(patched_expanded_ids) == 60
                else [],
            }
        )

    OUTPUT_JSON.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# Baseline decks card-pool match",
        "",
        f"Source: `decks/{DECKS_MD.name}`",
        f"Card pool: `cg.api.all_card_data()` ({len(engine_cards)} unique cards)",
        "",
    ]
    for deck in output["decks"]:
        lines.extend(
            [
                f"## {deck['name']}",
                "",
                f"- Declared cards: {deck['declared_total']}",
                f"- Present in competition pool: {deck['present_total']}",
                f"- Missing from competition pool: {deck['missing_total']}",
                f"- Replaced for patched baseline: {deck['replaced_total']}",
                f"- Unreplaced: {deck['unreplaced_total']}",
                f"- Present type counts: {deck['present_type_counts']}",
                f"- Patched type counts: {deck['patched_type_counts']}",
                "",
                "| Count | Card | Set | No. | Match | Type |",
                "|---:|---|---|---|---|---|",
            ]
        )
        for card in deck["present_cards"]:
            lines.append(
                f"| {card['count']} | {card['name']} | {card['set']} | {card['number']} | "
                f"{card['card_id']} {card['engine_name']} | {card['card_type']} |"
            )
        for card in deck["missing_cards"]:
            lines.append(
                f"| {card['count']} | {card['name']} | {card['set']} | {card['number']} | MISSING |  |"
            )
        if deck["replacement_cards"]:
            lines.extend(
                [
                    "",
                    "Replacement notes:",
                    "",
                    "| Count | Missing card | Replacement | Kind | Note |",
                    "|---:|---|---|---|---|",
                ]
            )
            for card in deck["replacement_cards"]:
                kind = (
                    "TRAINER_FILL"
                    if card["replacement_kind"] == "trainer_placeholder"
                    else "ENERGY_SAME_TYPE"
                )
                lines.append(
                    f"| {card['count']} | {card['original_name']} | "
                    f"{card['card_id']} {card['engine_name']} | {kind} | {card['note']} |"
                )
        lines.append("")

    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUTPUT_JSON}")
    print(f"Wrote {OUTPUT_MD}")


if __name__ == "__main__":
    main()
