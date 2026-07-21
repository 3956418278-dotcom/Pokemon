from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def _hash_json(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class DeckPrior:
    """Train-split-only, explainable deck-template posterior."""

    card_ids: tuple[int, ...]
    template_fingerprints: tuple[str, ...]
    template_counts: np.ndarray
    prior: np.ndarray
    card_kinds: tuple[str, ...]
    source_dates: tuple[str, ...]
    template_hash: str

    @classmethod
    def load(cls, path: str | Path) -> "DeckPrior":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            card_ids=tuple(int(x) for x in raw["card_ids"]),
            template_fingerprints=tuple(row["fingerprint"] for row in raw["templates"]),
            template_counts=np.asarray([row["counts"] for row in raw["templates"]], dtype=np.float32),
            prior=np.asarray([row["prior"] for row in raw["templates"]], dtype=np.float64),
            card_kinds=tuple(raw["card_kinds"]),
            source_dates=tuple(raw["source_dates"]),
            template_hash=str(raw["template_hash"]),
        )

    @property
    def template_count(self) -> int:
        return len(self.template_fingerprints)

    @property
    def card_to_column(self) -> dict[int, int]:
        return {card_id: index for index, card_id in enumerate(self.card_ids)}

    def vector(self, counts: dict[int, int]) -> np.ndarray:
        result = np.zeros(len(self.card_ids), dtype=np.float32)
        columns = self.card_to_column
        for card_id, count in counts.items():
            column = columns.get(int(card_id))
            if column is not None:
                result[column] = float(count)
        return result

    def label_for_deck(self, counts: dict[int, int]) -> int:
        truth = self.vector(counts)
        distance = np.abs(self.template_counts - truth[None, :]).sum(axis=1)
        return int(distance.argmin())

    def posterior(self, public_counts: dict[int, int]) -> np.ndarray:
        revealed = self.vector(public_counts)
        shortage = np.maximum(revealed[None, :] - self.template_counts, 0.0).sum(axis=1)
        log_weight = np.log(np.maximum(self.prior, 1e-12)) - 8.0 * shortage
        log_weight -= float(log_weight.max())
        weight = np.exp(log_weight)
        return weight / max(float(weight.sum()), 1e-12)

    def expected_remaining(self, public_counts: dict[int, int], posterior: np.ndarray) -> np.ndarray:
        revealed = self.vector(public_counts)
        remaining = np.maximum(self.template_counts - revealed[None, :], 0.0)
        return posterior @ remaining

    def category_totals(self, expected: np.ndarray) -> tuple[float, float, float]:
        totals = {"POKEMON": 0.0, "TRAINER": 0.0, "ENERGY": 0.0}
        for value, kind in zip(expected, self.card_kinds):
            totals[kind] = totals.get(kind, 0.0) + float(value)
        return totals["POKEMON"], totals["TRAINER"], totals["ENERGY"]


def _kind(card_type: str | None) -> str:
    value = str(card_type or "").upper()
    if "ENERGY" in value:
        return "ENERGY"
    if "POKEMON" in value or value in {"BASIC", "STAGE_1", "STAGE_2"}:
        return "POKEMON"
    return "TRAINER"


def build_deck_prior(
    rows: Iterable[dict[str, Any]],
    card_records: Iterable[dict[str, Any]],
    source_dates: Iterable[str],
    *,
    max_templates: int = 64,
) -> tuple[dict[str, Any], np.ndarray]:
    dates = tuple(sorted(str(value) for value in source_dates))
    allowed = set(dates)
    valid = [
        row for row in rows
        if str(row.get("source_date")) in allowed
        and int(row.get("deck_size", sum(row.get("card_counts", {}).values()))) == 60
    ]
    if not valid:
        raise ValueError("no complete train-split decks for prior")
    frequency = Counter(str(row["deck_fingerprint"]) for row in valid)
    representative: dict[str, dict[int, int]] = {}
    for row in valid:
        fingerprint = str(row["deck_fingerprint"])
        representative.setdefault(
            fingerprint,
            {int(card_id): int(count) for card_id, count in row["card_counts"].items()},
        )
    fingerprints = [value for value, _ in frequency.most_common(max_templates)]
    card_ids = sorted({card_id for value in fingerprints for card_id in representative[value]})
    card_to_column = {card_id: index for index, card_id in enumerate(card_ids)}
    kind_by_id = {int(row["card_id"]): _kind(row.get("card_type")) for row in card_records}
    total = sum(frequency[value] for value in fingerprints)
    templates = []
    matrix = np.zeros((len(fingerprints), len(card_ids)), dtype=np.float32)
    for template_index, fingerprint in enumerate(fingerprints):
        counts = representative[fingerprint]
        for card_id, count in counts.items():
            if card_id in card_to_column:
                matrix[template_index, card_to_column[card_id]] = count
        templates.append(
            {
                "template_id": template_index,
                "fingerprint": fingerprint,
                "observation_count": frequency[fingerprint],
                "prior": frequency[fingerprint] / total,
                "card_counts": {str(k): v for k, v in sorted(counts.items())},
                "counts": matrix[template_index].astype(int).tolist(),
            }
        )
    payload: dict[str, Any] = {
        "schema_version": "opponent_deck_templates_v1",
        "source_dates": list(dates),
        "source_split": "train",
        "complete_deck_observations": len(valid),
        "unique_fingerprints": len(frequency),
        "template_selection": f"top_{max_templates}_exact_fingerprints_by_train_frequency",
        "card_ids": card_ids,
        "card_kinds": [kind_by_id.get(card_id, "TRAINER") for card_id in card_ids],
        "templates": templates,
    }
    payload["template_hash"] = _hash_json(payload)
    binary = (matrix > 0).astype(np.float32)
    return payload, binary.T @ binary


def normalized_entropy(probabilities: np.ndarray) -> float:
    if len(probabilities) <= 1:
        return 0.0
    entropy = -float(np.sum(probabilities * np.log(np.maximum(probabilities, 1e-12))))
    return entropy / math.log(len(probabilities))
