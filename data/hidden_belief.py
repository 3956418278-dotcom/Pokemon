from __future__ import annotations

import math

from .decision_schema import ResidualMarginals


def build_residual_marginals(
    expected_hidden_by_card: dict[int, float],
    exact_hidden_by_card_zone: dict[int, dict[str, float]],
    hidden_zone_counts: dict[str, float],
) -> ResidualMarginals:
    """Subtract fixed exact hidden copies before constructing the IPF problem."""

    cards = set(expected_hidden_by_card) | set(exact_hidden_by_card_zone)
    unresolved_by_card = {
        card_id: float(expected_hidden_by_card.get(card_id, 0.0))
        - sum(float(value) for value in exact_hidden_by_card_zone.get(card_id, {}).values())
        for card_id in sorted(cards)
    }
    unresolved_by_zone = {
        zone: float(total)
        - sum(
            float(per_zone.get(zone, 0.0))
            for per_zone in exact_hidden_by_card_zone.values()
        )
        for zone, total in hidden_zone_counts.items()
    }
    result = ResidualMarginals(unresolved_by_card, unresolved_by_zone)
    result.validate()
    return result


def residual_ipf_expected_counts(
    marginals: ResidualMarginals,
    logits: dict[int, dict[str, float]],
    *,
    max_iterations: int = 200,
    tolerance: float = 1e-6,
) -> dict[int, dict[str, float]]:
    """Project unresolved-copy scores onto residual card and zone marginals.

    The result contains expected counts only. It is deliberately not exposed as
    a presence probability or a calibrated posterior.
    """

    marginals.validate(tolerance=max(tolerance, 1e-8))
    cards = list(marginals.unresolved_by_card)
    zones = list(marginals.unresolved_by_zone)
    matrix = {
        card_id: {
            zone: math.exp(max(-30.0, min(30.0, float(logits.get(card_id, {}).get(zone, 0.0)))))
            for zone in zones
        }
        for card_id in cards
    }
    if not cards or not zones:
        return matrix

    for _ in range(max_iterations):
        for card_id in cards:
            target = max(0.0, float(marginals.unresolved_by_card[card_id]))
            total = sum(matrix[card_id].values())
            scale = target / total if total > 0.0 else 0.0
            for zone in zones:
                matrix[card_id][zone] *= scale
        for zone in zones:
            target = max(0.0, float(marginals.unresolved_by_zone[zone]))
            total = sum(matrix[card_id][zone] for card_id in cards)
            scale = target / total if total > 0.0 else 0.0
            for card_id in cards:
                matrix[card_id][zone] *= scale

        row_error = max(
            abs(sum(matrix[card_id].values()) - marginals.unresolved_by_card[card_id])
            for card_id in cards
        )
        column_error = max(
            abs(sum(matrix[card_id][zone] for card_id in cards) - marginals.unresolved_by_zone[zone])
            for zone in zones
        )
        if max(row_error, column_error) <= tolerance:
            return matrix
    raise RuntimeError("residual IPF did not converge within max_iterations")


def unresolved_zone_entropy(
    expected_zone_counts: dict[int, dict[str, float]],
    unresolved_by_card: dict[int, float],
) -> dict[int, float]:
    """Entropy of each unresolved copy's zone allocation, never presence entropy."""

    result: dict[int, float] = {}
    for card_id, total in unresolved_by_card.items():
        if total <= 0.0:
            result[card_id] = 0.0
            continue
        entropy = 0.0
        for expected in expected_zone_counts.get(card_id, {}).values():
            probability = max(0.0, float(expected)) / float(total)
            if probability > 0.0:
                entropy -= probability * math.log(probability)
        result[card_id] = entropy
    return result
