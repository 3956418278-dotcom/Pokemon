"""Deterministic flat vectors built directly from ``EN_Card_Data.csv``.

The production CSV contains up to three rows for one Card ID because attacks,
abilities, and card effects occupy separate rows.  ``CardRegistry`` preserves
that source order, encodes every cell into a fixed-width row vector, then
flattens the rows and zero-pads cards with fewer rows.

This module deliberately uses only the Python standard library.  In
particular, it never uses Python's process-randomized ``hash()`` function.
"""

from __future__ import annotations

import csv
import hashlib
import io
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SOURCE_COLUMNS = (
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
)

EXPECTED_SOURCE_SHA256 = "a0ea63cf7adcb65d35436ce0eb390de6e2e35654a7c67c065a45f4abaa00f373"
EXPECTED_ROW_COUNT = 2022
EXPECTED_CARD_COUNT = 1267
MAX_ROWS_PER_CARD = 3

# A separate hash space per CSV column prevents equal text in two semantically
# different columns from becoming the same feature.  One whole-cell hash is
# intentionally used instead of a learned vocabulary so the layout is stable
# at Kaggle inference time and also works for synthetic/previously unseen text.
HASH_BUCKETS_PER_COLUMN = 32
HASH_ALGORITHM = "sha256"

# Raw text is hashed for every column, including these numeric columns.  The
# two additional features preserve useful magnitude and explicit parse
# validity while the hash retains suffixes such as ``120+`` versus ``120×``.
NUMERIC_SCALES = {
    "Card ID": 2048.0,
    "Collection No.": 256.0,
    "HP": 400.0,
    "Retreat": 8.0,
    "Damage": 400.0,
}


class CardRegistryError(ValueError):
    """Raised when a CSV cannot satisfy the registry's deterministic contract."""


def _canonical_card_id(value: Any) -> str:
    text = "" if value is None else str(value).strip()
    if re.fullmatch(r"[+-]?\d+", text):
        return str(int(text))
    return text


def _numeric_prefix(value: str) -> int | None:
    """Return a leading integer while accepting damage suffixes such as +/×."""

    match = re.match(r"^\s*([+-]?\d+)", value)
    if match is None:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _scaled_numeric(value: str, scale: float) -> tuple[float, float]:
    parsed = _numeric_prefix(value)
    if parsed is None:
        return 0.0, 0.0
    try:
        scaled = float(parsed) / scale
    except (OverflowError, ValueError):
        return 0.0, 0.0
    if not math.isfinite(scaled):
        return 0.0, 0.0
    return scaled, 1.0


def _column_layout() -> tuple[list[dict[str, Any]], int, tuple[str, ...]]:
    layout: list[dict[str, Any]] = []
    feature_names: list[str] = []
    cursor = 0
    for column in SOURCE_COLUMNS:
        start = cursor
        hash_start = cursor
        hash_end = hash_start + HASH_BUCKETS_PER_COLUMN
        feature_names.extend(
            f"{column}::hash_{bucket:02d}"
            for bucket in range(HASH_BUCKETS_PER_COLUMN)
        )
        cursor = hash_end

        numeric_value_offset: int | None = None
        numeric_valid_offset: int | None = None
        if column in NUMERIC_SCALES:
            numeric_value_offset = cursor
            numeric_valid_offset = cursor + 1
            feature_names.extend(
                (f"{column}::numeric_value", f"{column}::numeric_valid")
            )
            cursor += 2

        layout.append(
            {
                "column": column,
                "start": start,
                "end": cursor,
                "hash_start": hash_start,
                "hash_end": hash_end,
                "numeric_value_offset": numeric_value_offset,
                "numeric_valid_offset": numeric_valid_offset,
                "numeric_scale": NUMERIC_SCALES.get(column),
            }
        )
    return layout, cursor, tuple(feature_names)


def _hash_bucket(column: str, value: str) -> tuple[int, float]:
    payload = f"{column}\x1f{value}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    bucket = int.from_bytes(digest[:8], "big") % HASH_BUCKETS_PER_COLUMN
    sign = 1.0 if digest[8] & 1 else -1.0
    return bucket, sign


def _encode_row(
    row: dict[str, str],
    layout: list[dict[str, Any]],
    row_vector_dim: int,
) -> tuple[float, ...]:
    vector = [0.0] * row_vector_dim
    for spec in layout:
        column = str(spec["column"])
        value = row.get(column, "")
        value = "" if value is None else str(value)
        bucket, sign = _hash_bucket(column, value)
        vector[int(spec["hash_start"]) + bucket] = sign

        numeric_value_offset = spec["numeric_value_offset"]
        if numeric_value_offset is not None:
            numeric, valid = _scaled_numeric(value, float(spec["numeric_scale"]))
            vector[int(numeric_value_offset)] = numeric
            vector[int(spec["numeric_valid_offset"])] = valid

    if not all(math.isfinite(value) for value in vector):
        raise CardRegistryError("row encoding produced a non-finite value")
    return tuple(vector)


@dataclass
class CardRegistry:
    """Mapping from canonical Card ID strings to fixed flat CSV vectors."""

    card_vector_dim: int
    row_vector_dim: int
    source_sha256: str
    row_count: int
    card_count: int
    max_rows_per_card: int
    feature_names: tuple[str, ...]
    vectors: dict[str, tuple[float, ...]]
    _row_feature_names: tuple[str, ...]
    _column_offsets: tuple[dict[str, Any], ...]

    @classmethod
    def from_csv(cls, path: Path, strict: bool = True) -> "CardRegistry":
        """Load a registry while preserving the source order of rows per card.

        ``strict=True`` pins the production competition file.  ``strict=False``
        retains the exact same 17-column encoding contract but permits small
        synthetic datasets for tests and smoke checks.
        """

        source_path = Path(path)
        payload = source_path.read_bytes()
        source_sha256 = hashlib.sha256(payload).hexdigest()
        try:
            text = payload.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise CardRegistryError(f"CSV is not valid UTF-8: {source_path}") from exc

        reader = csv.DictReader(io.StringIO(text, newline=""))
        actual_columns = tuple(reader.fieldnames or ())
        if actual_columns != SOURCE_COLUMNS:
            raise CardRegistryError(
                "CSV columns changed or are out of order: "
                f"expected={list(SOURCE_COLUMNS)!r}, actual={list(actual_columns)!r}"
            )

        grouped_rows: dict[str, list[dict[str, str]]] = {}
        row_count = 0
        for csv_line, raw_row in enumerate(reader, start=2):
            if None in raw_row:
                raise CardRegistryError(f"CSV line {csv_line} has extra fields")
            row = {
                column: "" if raw_row.get(column) is None else str(raw_row[column])
                for column in SOURCE_COLUMNS
            }
            card_id = _canonical_card_id(row["Card ID"])
            if not card_id:
                raise CardRegistryError(f"CSV line {csv_line} has an empty Card ID")
            grouped_rows.setdefault(card_id, []).append(row)
            row_count += 1

        card_count = len(grouped_rows)
        max_rows_per_card = max(
            (len(card_rows) for card_rows in grouped_rows.values()),
            default=0,
        )
        if max_rows_per_card > MAX_ROWS_PER_CARD:
            raise CardRegistryError(
                f"a Card ID has {max_rows_per_card} rows; maximum supported is "
                f"{MAX_ROWS_PER_CARD}"
            )

        if strict:
            mismatches: list[str] = []
            if source_sha256 != EXPECTED_SOURCE_SHA256:
                mismatches.append(
                    f"sha256={source_sha256} (expected {EXPECTED_SOURCE_SHA256})"
                )
            if row_count != EXPECTED_ROW_COUNT:
                mismatches.append(
                    f"row_count={row_count} (expected {EXPECTED_ROW_COUNT})"
                )
            if card_count != EXPECTED_CARD_COUNT:
                mismatches.append(
                    f"card_count={card_count} (expected {EXPECTED_CARD_COUNT})"
                )
            if max_rows_per_card != MAX_ROWS_PER_CARD:
                mismatches.append(
                    "max_rows_per_card="
                    f"{max_rows_per_card} (expected {MAX_ROWS_PER_CARD})"
                )
            if mismatches:
                raise CardRegistryError(
                    "strict EN_Card_Data.csv validation failed: " + "; ".join(mismatches)
                )

        layout, row_vector_dim, row_feature_names = _column_layout()
        card_vector_dim = row_vector_dim * max_rows_per_card
        feature_names = tuple(
            f"row_{row_slot}::{feature_name}"
            for row_slot in range(max_rows_per_card)
            for feature_name in row_feature_names
        )

        vectors: dict[str, tuple[float, ...]] = {}
        for card_id, card_rows in grouped_rows.items():
            flat: list[float] = []
            for row in card_rows:
                flat.extend(_encode_row(row, layout, row_vector_dim))
            flat.extend(
                [0.0]
                * ((max_rows_per_card - len(card_rows)) * row_vector_dim)
            )
            vector = tuple(flat)
            if len(vector) != card_vector_dim:
                raise CardRegistryError(
                    f"internal vector width mismatch for Card ID {card_id}"
                )
            if not all(math.isfinite(value) for value in vector):
                raise CardRegistryError(
                    f"non-finite vector value for Card ID {card_id}"
                )
            vectors[card_id] = vector

        return cls(
            card_vector_dim=card_vector_dim,
            row_vector_dim=row_vector_dim,
            source_sha256=source_sha256,
            row_count=row_count,
            card_count=card_count,
            max_rows_per_card=max_rows_per_card,
            feature_names=feature_names,
            vectors=vectors,
            _row_feature_names=row_feature_names,
            _column_offsets=tuple(dict(spec) for spec in layout),
        )

    def vector(self, card_id: Any) -> tuple[float, ...]:
        """Return a card vector, or an all-zero vector for an unknown Card ID."""

        return self.vectors.get(
            _canonical_card_id(card_id),
            (0.0,) * self.card_vector_dim,
        )

    def schema(self) -> dict[str, Any]:
        """Return the complete, JSON-serializable feature contract."""

        column_offsets = {
            str(spec["column"]): {
                key: value
                for key, value in spec.items()
                if key != "column"
            }
            for spec in self._column_offsets
        }
        row_offsets = [
            {
                "row_slot": row_slot,
                "start": row_slot * self.row_vector_dim,
                "end": (row_slot + 1) * self.row_vector_dim,
            }
            for row_slot in range(self.max_rows_per_card)
        ]
        return {
            "schema_version": "csv_long_vector_v1",
            "column_order": list(SOURCE_COLUMNS),
            "columns": list(SOURCE_COLUMNS),
            "encoding": {
                "type": "per_column_whole_cell_feature_hashing_with_numeric_features",
                "hash_algorithm": HASH_ALGORITHM,
                "hash_buckets_per_column": HASH_BUCKETS_PER_COLUMN,
                "hash_input": "<column>\\x1f<exact CSV cell text>",
                "signed_hash": True,
                "empty_cell_is_hashed": True,
                "numeric_columns": {
                    column: {
                        "scale": scale,
                        "parser": "leading_signed_integer",
                        "adds_validity_feature": True,
                    }
                    for column, scale in NUMERIC_SCALES.items()
                },
                "padding": "all-zero row vectors appended after source rows",
                "row_order": "original CSV order within Card ID",
            },
            "column_offsets": column_offsets,
            "row_offsets": row_offsets,
            "dimensions": {
                "row_vector_dim": self.row_vector_dim,
                "card_vector_dim": self.card_vector_dim,
                "row_slots": self.max_rows_per_card,
            },
            "row_vector_dim": self.row_vector_dim,
            "card_vector_dim": self.card_vector_dim,
            "source_sha256": self.source_sha256,
            "counts": {
                "rows": self.row_count,
                "cards": self.card_count,
                "max_rows_per_card": self.max_rows_per_card,
            },
            "row_count": self.row_count,
            "card_count": self.card_count,
            "max_rows_per_card": self.max_rows_per_card,
            "row_feature_names": list(self._row_feature_names),
            "feature_names": list(self.feature_names),
        }


__all__ = [
    "CardRegistry",
    "CardRegistryError",
    "EXPECTED_SOURCE_SHA256",
    "SOURCE_COLUMNS",
]
