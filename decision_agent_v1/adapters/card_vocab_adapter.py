from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CardVocabulary:
    card_id_to_index: dict[int, int]
    index_to_card_id: tuple[int | None, ...]
    pad_index: int = 0
    unk_index: int = 1

    @classmethod
    def from_json(cls, path: str | Path) -> "CardVocabulary":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("card vocabulary must be a JSON object")
        indexed = sorted(
            ((int(card_id), int(index)) for card_id, index in raw.items()),
            key=lambda item: item[1],
        )
        if [index for _, index in indexed] != list(range(len(indexed))):
            raise ValueError("card vocabulary indices must be contiguous from zero")
        mapping = {card_id: source_index + 2 for card_id, source_index in indexed}
        return cls(mapping, (None, None, *(card_id for card_id, _ in indexed)))

    @classmethod
    def from_card_ids(cls, card_ids: list[int]) -> "CardVocabulary":
        unique = sorted(set(int(card_id) for card_id in card_ids))
        return cls(
            {card_id: offset + 2 for offset, card_id in enumerate(unique)},
            (None, None, *unique),
        )

    def encode(self, card_id: int | None) -> int:
        if card_id is None:
            return self.unk_index
        return self.card_id_to_index.get(int(card_id), self.unk_index)

    def decode(self, index: int) -> int | None:
        if 0 <= index < len(self.index_to_card_id):
            return self.index_to_card_id[index]
        return None

    def __len__(self) -> int:
        return len(self.index_to_card_id)
