from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable


DATE_DIRECTORY = re.compile(r"^\d{4}-\d{2}-\d{2}$")
CARD_OUTPUT_FIELDS = [
    "card_id",
    "card_name",
    "card_kind",
    "deck_presence_count",
    "raw_presence_frequency",
    "smoothed_presence_frequency",
    "total_copy_count",
    "raw_copy_share",
    "mean_copies_when_present",
    "meets_min_support",
    "popularity_percentile",
    "support_weight",
    "normalized_popularity_score",
]
PAIR_OUTPUT_FIELDS = [
    "card_id_a",
    "card_name_a",
    "card_id_b",
    "card_name_b",
    "deck_cooccurrence_count",
    "raw_cooccurrence_frequency",
    "expected_cooccurrence_count",
    "smoothed_cooccurrence_frequency",
    "smoothed_p_b_given_a",
    "smoothed_p_a_given_b",
    "smoothed_lift",
    "normalized_pmi",
    "support_weight",
    "shrunk_normalized_pmi",
    "normalized_affinity_score",
    "normalized_positive_association",
]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _mean_rank_percentiles(values: list[float]) -> list[float]:
    """Return tie-aware percentile ranks in [0, 1]."""
    if len(values) <= 1:
        return [1.0] * len(values)
    ordered = sorted(range(len(values)), key=values.__getitem__)
    result = [0.0] * len(values)
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and values[ordered[end]] == values[ordered[start]]:
            end += 1
        mean_rank = (start + end - 1) / 2
        percentile = mean_rank / (len(values) - 1)
        for index in ordered[start:end]:
            result[index] = percentile
        start = end
    return result


def discover_daily_statistics(input_dir: Path) -> list[Path]:
    days = [
        path
        for path in input_dir.iterdir()
        if path.is_dir()
        and DATE_DIRECTORY.fullmatch(path.name)
        and (path / "summary.json").is_file()
        and (path / "card_frequency.csv").is_file()
        and (path / "card_pair_frequency.csv").is_file()
    ]
    if not days:
        raise ValueError(f"no complete YYYY-MM-DD statistic directories found under {input_dir}")
    return sorted(days)


def normalize_statistics(
    input_dir: Path,
    output_dir: Path,
    *,
    min_card_count: int = 20,
    min_pair_count: int = 20,
    prior_strength: float = 100.0,
) -> dict[str, Any]:
    if min_card_count < 1 or min_pair_count < 1:
        raise ValueError("minimum support counts must be positive")
    if prior_strength <= 0:
        raise ValueError("prior_strength must be positive")

    daily_dirs = discover_daily_statistics(input_dir)
    total_decks = 0
    cards: dict[int, dict[str, Any]] = {}
    pairs: dict[tuple[int, int], dict[str, Any]] = {}

    for daily_dir in daily_dirs:
        summary = json.loads((daily_dir / "summary.json").read_text(encoding="utf-8"))
        day_decks = int(summary["valid_complete_deck_count"])
        if day_decks <= 0:
            continue
        total_decks += day_decks

        seen_cards: set[int] = set()
        for row in _read_csv(daily_dir / "card_frequency.csv"):
            card_id = int(row["card_id"])
            if card_id in seen_cards:
                raise ValueError(f"duplicate card_id {card_id} in {daily_dir}")
            seen_cards.add(card_id)
            presence = int(row["deck_presence_count"])
            if not 0 <= presence <= day_decks:
                raise ValueError(f"invalid presence count for card {card_id} in {daily_dir}")
            aggregate = cards.setdefault(
                card_id,
                {
                    "card_id": card_id,
                    "card_name": row["card_name"],
                    "card_kind": row["card_kind"],
                    "deck_presence_count": 0,
                    "total_copy_count": 0,
                },
            )
            aggregate["deck_presence_count"] += presence
            aggregate["total_copy_count"] += int(row["total_copy_count"])

        seen_pairs: set[tuple[int, int]] = set()
        for row in _read_csv(daily_dir / "card_pair_frequency.csv"):
            card_id_a, card_id_b = sorted((int(row["card_id_a"]), int(row["card_id_b"])))
            key = (card_id_a, card_id_b)
            if key in seen_pairs:
                raise ValueError(f"duplicate card pair {key} in {daily_dir}")
            seen_pairs.add(key)
            count = int(row["deck_cooccurrence_count"])
            if not 0 <= count <= day_decks:
                raise ValueError(f"invalid cooccurrence count for pair {key} in {daily_dir}")
            name_a = row["card_name_a"] if int(row["card_id_a"]) == card_id_a else row["card_name_b"]
            name_b = row["card_name_b"] if int(row["card_id_b"]) == card_id_b else row["card_name_a"]
            aggregate = pairs.setdefault(
                key,
                {
                    "card_id_a": card_id_a,
                    "card_name_a": name_a,
                    "card_id_b": card_id_b,
                    "card_name_b": name_b,
                    "deck_cooccurrence_count": 0,
                },
            )
            aggregate["deck_cooccurrence_count"] += count

    if total_decks <= 0:
        raise ValueError("the selected daily statistics contain no valid complete decks")

    # Jeffreys' Beta(1/2, 1/2) prior prevents exact 0/1 probabilities.
    card_rows: list[dict[str, Any]] = []
    smoothed_presence: dict[int, float] = {}
    for card_id in sorted(cards):
        aggregate = cards[card_id]
        presence = aggregate["deck_presence_count"]
        copies = aggregate["total_copy_count"]
        smoothed = (presence + 0.5) / (total_decks + 1.0)
        smoothed_presence[card_id] = smoothed
        support_weight = presence / (presence + prior_strength)
        card_rows.append(
            {
                **aggregate,
                "raw_presence_frequency": presence / total_decks,
                "smoothed_presence_frequency": smoothed,
                "raw_copy_share": copies / (60 * total_decks),
                "mean_copies_when_present": copies / presence if presence else 0.0,
                "meets_min_support": presence >= min_card_count,
                "popularity_percentile": 0.0,
                "support_weight": support_weight,
                "normalized_popularity_score": 0.0,
            }
        )

    percentiles = _mean_rank_percentiles([row["smoothed_presence_frequency"] for row in card_rows])
    for row, percentile in zip(card_rows, percentiles):
        row["popularity_percentile"] = percentile
        row["normalized_popularity_score"] = percentile * row["support_weight"]

    pair_rows: list[dict[str, Any]] = []
    dropped_pair_count = 0
    for key in sorted(pairs):
        aggregate = pairs[key]
        card_id_a, card_id_b = key
        count_a = cards.get(card_id_a, {}).get("deck_presence_count", 0)
        count_b = cards.get(card_id_b, {}).get("deck_presence_count", 0)
        count_ab = aggregate["deck_cooccurrence_count"]
        if count_ab > min(count_a, count_b):
            raise ValueError(f"pair {key} occurs more often than one of its cards")
        if count_ab < min_pair_count or min(count_a, count_b) < min_card_count:
            dropped_pair_count += 1
            continue

        p_a = smoothed_presence[card_id_a]
        p_b = smoothed_presence[card_id_b]
        expected_probability = p_a * p_b
        expected_count = total_decks * expected_probability
        # The independence expectation is the prior mean. It shrinks unstable rare-pair
        # lift toward 1 without erasing a well-supported association.
        joint = (count_ab + prior_strength * expected_probability) / (total_decks + prior_strength)
        conditional_b_given_a = (count_ab + prior_strength * p_b) / (count_a + prior_strength)
        conditional_a_given_b = (count_ab + prior_strength * p_a) / (count_b + prior_strength)
        lift = joint / expected_probability
        pmi = math.log(lift)
        npmi = max(-1.0, min(1.0, pmi / -math.log(joint)))
        support_weight = count_ab / (count_ab + prior_strength)
        shrunk_npmi = npmi * support_weight
        pair_rows.append(
            {
                **aggregate,
                "raw_cooccurrence_frequency": count_ab / total_decks,
                "expected_cooccurrence_count": expected_count,
                "smoothed_cooccurrence_frequency": joint,
                "smoothed_p_b_given_a": conditional_b_given_a,
                "smoothed_p_a_given_b": conditional_a_given_b,
                "smoothed_lift": lift,
                "normalized_pmi": npmi,
                "support_weight": support_weight,
                "shrunk_normalized_pmi": shrunk_npmi,
                "normalized_affinity_score": (shrunk_npmi + 1.0) / 2.0,
                "normalized_positive_association": max(0.0, shrunk_npmi),
            }
        )

    pair_rows.sort(
        key=lambda row: (
            -row["normalized_positive_association"],
            -row["deck_cooccurrence_count"],
            row["card_id_a"],
            row["card_id_b"],
        )
    )
    _write_csv(output_dir / "normalized_card_statistics.csv", CARD_OUTPUT_FIELDS, card_rows)
    _write_csv(output_dir / "normalized_card_pair_statistics.csv", PAIR_OUTPUT_FIELDS, pair_rows)
    result = {
        "source_statistics_directory": str(input_dir),
        "source_dates": [path.name for path in daily_dirs],
        "source_date_count": len(daily_dirs),
        "total_valid_complete_decks": total_decks,
        "card_count": len(card_rows),
        "low_support_card_count": sum(
            not row["meets_min_support"] for row in card_rows
        ),
        "retained_pair_count": len(pair_rows),
        "dropped_low_support_pair_count": dropped_pair_count,
        "min_card_count": min_card_count,
        "min_pair_count": min_pair_count,
        "prior_strength": prior_strength,
        "normalization": {
            "card_probability": "Jeffreys Beta(0.5, 0.5) posterior mean",
            "card_score": "tie-aware popularity percentile multiplied by support weight",
            "pair_prior": "independence expectation with configurable prior strength",
            "pair_score": "normalized PMI multiplied by support weight",
        },
    }
    (output_dir / "normalization_summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate, denoise, and normalize daily replay card/pair statistics."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("outputs/replay_extract/statistics"),
        help="Directory containing YYYY-MM-DD statistic partitions.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/replay_extract/statistics_normalized"),
    )
    parser.add_argument("--min-card-count", type=int, default=20)
    parser.add_argument("--min-pair-count", type=int, default=20)
    parser.add_argument(
        "--prior-strength",
        type=float,
        default=100.0,
        help="Equivalent deck count of the smoothing prior; larger values shrink rare observations more.",
    )
    args = parser.parse_args()
    result = normalize_statistics(
        args.input_dir,
        args.output_dir,
        min_card_count=args.min_card_count,
        min_pair_count=args.min_pair_count,
        prior_strength=args.prior_strength,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
