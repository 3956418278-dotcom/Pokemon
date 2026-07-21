from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

from data.game_memory import GameMemoryState


def load_deck_csv(path: str | Path) -> list[int]:
    """Load a submitted deck without assuming row order or exposing opponent data."""

    result: list[int] = []
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        normalized = {str(key).strip().lower().replace(" ", "_"): value for key, value in row.items()}
        card_id = normalized.get("card_id") or normalized.get("cardid") or normalized.get("id")
        count = normalized.get("count") or normalized.get("quantity") or "1"
        if card_id not in (None, ""):
            result.extend([int(card_id)] * int(count))
    if len(result) != 60:
        raise ValueError(f"deck.csv must describe exactly 60 cards, found {len(result)}")
    return result


@dataclass
class LiveStateUpgradeContext:
    self_deck: list[int]
    memory: GameMemoryState = field(default_factory=GameMemoryState)

    @classmethod
    def from_deck_csv(cls, path: str | Path) -> "LiveStateUpgradeContext":
        return cls(self_deck=load_deck_csv(path))
