from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path


SCRIPT_PATH = Path("scripts/normalize_replay_statistics.py")
SPEC = importlib.util.spec_from_file_location("normalize_replay_statistics", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
normalize_statistics = MODULE.normalize_statistics


def _write_partition(
    root: Path,
    date: str,
    deck_count: int,
    cards: list[tuple[int, str, int, int]],
    pairs: list[tuple[int, str, int, str, int]],
) -> None:
    directory = root / date
    directory.mkdir(parents=True)
    (directory / "summary.json").write_text(
        json.dumps({"valid_complete_deck_count": deck_count}), encoding="utf-8"
    )
    with (directory / "card_frequency.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "card_id", "card_name", "card_kind", "deck_presence_count",
                "deck_presence_frequency", "total_copy_count", "copy_share",
                "mean_copies_when_present",
            ],
        )
        writer.writeheader()
        for card_id, name, presence, copies in cards:
            writer.writerow(
                {
                    "card_id": card_id,
                    "card_name": name,
                    "card_kind": "fixture",
                    "deck_presence_count": presence,
                    "deck_presence_frequency": presence / deck_count,
                    "total_copy_count": copies,
                    "copy_share": copies / (60 * deck_count),
                    "mean_copies_when_present": copies / presence,
                }
            )
    with (directory / "card_pair_frequency.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "card_id_a", "card_name_a", "card_id_b", "card_name_b",
                "deck_cooccurrence_count", "deck_cooccurrence_frequency",
                "p_b_given_a", "p_a_given_b", "lift",
            ],
        )
        writer.writeheader()
        for card_a, name_a, card_b, name_b, count in pairs:
            writer.writerow(
                {
                    "card_id_a": card_a,
                    "card_name_a": name_a,
                    "card_id_b": card_b,
                    "card_name_b": name_b,
                    "deck_cooccurrence_count": count,
                }
            )


def test_normalization_aggregates_days_and_filters_rare_pairs(tmp_path: Path) -> None:
    source = tmp_path / "statistics"
    _write_partition(
        source,
        "2026-07-01",
        100,
        [(1, "one", 80, 240), (2, "two", 50, 100), (3, "rare", 1, 1)],
        [(1, "one", 2, "two", 45), (1, "one", 3, "rare", 1)],
    )
    _write_partition(
        source,
        "2026-07-02",
        100,
        [(1, "one", 70, 210), (2, "two", 40, 80), (3, "rare", 1, 1)],
        [(1, "one", 2, "two", 35), (2, "two", 3, "rare", 1)],
    )
    fixture = source / "fixture"
    fixture.mkdir()

    output = tmp_path / "normalized"
    summary = normalize_statistics(
        source, output, min_card_count=5, min_pair_count=5, prior_strength=10
    )

    assert summary["source_dates"] == ["2026-07-01", "2026-07-02"]
    assert summary["total_valid_complete_decks"] == 200
    assert summary["retained_pair_count"] == 1
    assert summary["dropped_low_support_pair_count"] == 2

    with (output / "normalized_card_statistics.csv").open(encoding="utf-8-sig") as handle:
        cards = {int(row["card_id"]): row for row in csv.DictReader(handle)}
    assert int(cards[1]["deck_presence_count"]) == 150
    assert cards[1]["meets_min_support"] == "True"
    assert cards[3]["meets_min_support"] == "False"
    assert 0.0 <= float(cards[1]["normalized_popularity_score"]) <= 1.0
    assert float(cards[1]["normalized_popularity_score"]) > float(
        cards[3]["normalized_popularity_score"]
    )

    with (output / "normalized_card_pair_statistics.csv").open(encoding="utf-8-sig") as handle:
        pairs = list(csv.DictReader(handle))
    assert [(int(row["card_id_a"]), int(row["card_id_b"])) for row in pairs] == [(1, 2)]
    assert int(pairs[0]["deck_cooccurrence_count"]) == 80
    assert 0.0 <= float(pairs[0]["normalized_affinity_score"]) <= 1.0


def test_normalization_rejects_pair_count_above_marginal(tmp_path: Path) -> None:
    source = tmp_path / "statistics"
    _write_partition(
        source,
        "2026-07-01",
        10,
        [(1, "one", 2, 2), (2, "two", 3, 3)],
        [(1, "one", 2, "two", 3)],
    )
    try:
        normalize_statistics(source, tmp_path / "out", min_card_count=1, min_pair_count=1)
    except ValueError as error:
        assert "occurs more often" in str(error)
    else:
        raise AssertionError("invalid pair marginal was accepted")
