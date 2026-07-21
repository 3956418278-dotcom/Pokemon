from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from .config import TargetDeckConfig


@dataclass(frozen=True)
class DeckCardToken:
    """One physical deck copy: rule identity plus explicit copy identity."""

    card_id: int
    card_index: int
    copy_ordinal: int
    copies_in_deck: int


@dataclass(frozen=True)
class TargetDeck:
    name: str
    card_ids: tuple[int, ...]
    card_tokens: tuple[DeckCardToken, ...]
    sha256: str


def _deck_hash(card_ids: list[int]) -> str:
    contents = "".join(f"{card_id}\n" for card_id in card_ids).encode("utf-8")
    return hashlib.sha256(contents).hexdigest()


def load_target_deck(config: TargetDeckConfig, root: str | Path) -> TargetDeck:
    root = Path(root)
    decks = json.loads((root / config.source).read_text(encoding="utf-8"))["decks"]
    selected = decks[config.index]
    if selected["name"] != config.expected_name:
        raise ValueError(
            f"deck index {config.index} is {selected['name']!r}, expected {config.expected_name!r}"
        )
    if int(selected.get("replaced_total", 0)) != 0:
        raise ValueError("scored target deck must not contain patched card replacements")
    card_ids = [int(value) for value in selected["patched_deck_ids"]]
    if len(card_ids) != 60:
        raise ValueError(f"target deck contains {len(card_ids)} cards, expected 60")
    digest = _deck_hash(card_ids)
    if digest != config.expected_deck_sha256:
        raise ValueError(f"target deck hash changed: expected {config.expected_deck_sha256}, got {digest}")

    raw_vocab = json.loads((root / config.card_vocab_path).read_text(encoding="utf-8"))
    vocab = {int(card_id): int(index) + 2 for card_id, index in raw_vocab.items()}
    totals = Counter(card_ids)
    seen: defaultdict[int, int] = defaultdict(int)
    tokens = []
    for card_id in card_ids:
        if card_id not in vocab:
            raise ValueError(f"card ID {card_id} is absent from the configured vocabulary")
        tokens.append(
            DeckCardToken(
                card_id=card_id,
                card_index=vocab[card_id],
                copy_ordinal=seen[card_id],
                copies_in_deck=totals[card_id],
            )
        )
        seen[card_id] += 1
    return TargetDeck(selected["name"], tuple(card_ids), tuple(tokens), digest)
