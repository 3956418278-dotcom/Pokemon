from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import pytest

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from csv_long_vector import (
    SOURCE_COLUMNS,
    CardRegistry,
    CardRegistryError,
)


def _row(card_id: str, name: str, move: str, damage: str) -> dict[str, str]:
    row = {column: "" for column in SOURCE_COLUMNS}
    row.update(
        {
            "Card ID": card_id,
            "Card Name": name,
            "Expansion": "TST",
            "Collection No.": card_id,
            "Stage (Pokémon)/Type (Energy and Trainer)": "Basic Pokémon",
            "Rule": "n/a",
            "Category": "Test category",
            "Previous stage": "n/a",
            "HP": "120",
            "Type": "{G}",
            "Weakness": "{R}",
            "Resistance (Type)": "n/a",
            "Retreat": "2",
            "Move Name": move,
            "Cost": "{G}{C}",
            "Damage": damage,
            "Effect Explanation": f"Effect for {move}",
        }
    )
    return row


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SOURCE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def test_flattening_preserves_row_order_and_zero_pads(tmp_path: Path) -> None:
    csv_path = tmp_path / "cards.csv"
    _write_csv(
        csv_path,
        [
            _row("1", "Alpha", "First Move", "20"),
            _row("1", "Alpha", "Second Move", "40+"),
            _row("2", "Beta", "Only Move", "30×"),
        ],
    )

    registry = CardRegistry.from_csv(csv_path, strict=False)

    assert registry.row_count == 3
    assert registry.card_count == 2
    assert registry.max_rows_per_card == 2
    assert registry.card_vector_dim == registry.row_vector_dim * 2
    assert len(registry.feature_names) == registry.card_vector_dim

    alpha = registry.vector(1)
    alpha_first = alpha[: registry.row_vector_dim]
    alpha_second = alpha[registry.row_vector_dim :]
    assert alpha_first != alpha_second
    assert any(value != 0.0 for value in alpha_first)
    assert any(value != 0.0 for value in alpha_second)

    beta = registry.vector("2")
    assert any(value != 0.0 for value in beta[: registry.row_vector_dim])
    assert beta[registry.row_vector_dim :] == (0.0,) * registry.row_vector_dim


def test_unknown_determinism_finiteness_and_json_schema(tmp_path: Path) -> None:
    csv_path = tmp_path / "cards.csv"
    _write_csv(
        csv_path,
        [
            _row("001", "Alpha", "First Move", "20"),
            _row("001", "Alpha", "Second Move", "40+"),
            _row("2", "Beta", "Only Move", "n/a"),
        ],
    )

    first = CardRegistry.from_csv(csv_path, strict=False)
    second = CardRegistry.from_csv(csv_path, strict=False)

    assert first.vectors == second.vectors
    assert first.feature_names == second.feature_names
    assert first.schema() == second.schema()
    assert first.vector("001") == first.vector(1)
    assert first.vector("not-a-card") == (0.0,) * first.card_vector_dim
    assert all(
        math.isfinite(value)
        for vector in first.vectors.values()
        for value in vector
    )

    schema = first.schema()
    json.dumps(schema)
    assert schema["column_order"] == list(SOURCE_COLUMNS)
    assert list(schema["column_offsets"]) == list(SOURCE_COLUMNS)
    assert schema["dimensions"] == {
        "row_vector_dim": first.row_vector_dim,
        "card_vector_dim": first.card_vector_dim,
        "row_slots": first.max_rows_per_card,
    }
    assert schema["counts"] == {
        "rows": 3,
        "cards": 2,
        "max_rows_per_card": 2,
    }
    assert schema["source_sha256"] == first.source_sha256
    assert len(schema["feature_names"]) == first.card_vector_dim

    # Every one of the 17 source columns owns a disjoint hash segment and
    # contributes to an encoded, non-padding row.
    encoded_row = first.vector(1)[: first.row_vector_dim]
    for column in SOURCE_COLUMNS:
        offsets = schema["column_offsets"][column]
        assert any(
            value != 0.0
            for value in encoded_row[offsets["hash_start"] : offsets["hash_end"]]
        )


def test_reversing_source_rows_reverses_flattened_row_slots(tmp_path: Path) -> None:
    first_path = tmp_path / "first.csv"
    reversed_path = tmp_path / "reversed.csv"
    first_row = _row("7", "Order", "First", "10")
    second_row = _row("7", "Order", "Second", "20")
    _write_csv(first_path, [first_row, second_row])
    _write_csv(reversed_path, [second_row, first_row])

    normal = CardRegistry.from_csv(first_path, strict=False).vector(7)
    reversed_vector = CardRegistry.from_csv(reversed_path, strict=False).vector(7)
    row_dim = CardRegistry.from_csv(first_path, strict=False).row_vector_dim

    assert normal[:row_dim] == reversed_vector[row_dim:]
    assert normal[row_dim:] == reversed_vector[:row_dim]


def test_strict_mode_rejects_a_synthetic_source(tmp_path: Path) -> None:
    csv_path = tmp_path / "cards.csv"
    _write_csv(csv_path, [_row("1", "Alpha", "Move", "10")])

    with pytest.raises(CardRegistryError, match="strict EN_Card_Data.csv validation failed"):
        CardRegistry.from_csv(csv_path)
