from __future__ import annotations

from collections.abc import Mapping


# Only public, decision-relevant scalars remain. Hand/deck/discard size signals
# are deliberately absent; card identity and physical-copy features are tokens.
GLOBAL_FEATURE_NAMES = (
    "turn",
    "turn_action_count",
    "is_first_player",
    "relative_current_player",
    "self_prize_count",
    "opponent_prize_count",
    "self_active_count",
    "self_bench_count",
    "opponent_active_count",
    "opponent_bench_count",
    "energy_attached",
    "supporter_played",
    "stadium_played",
    "retreated",
    "select_type",
    "select_context",
    "remain_damage_counter",
    "remain_energy_cost",
    "min_count",
    "max_count",
)

FORBIDDEN_COUNT_FEATURES = frozenset(
    {
        "self_deck_count",
        "opponent_deck_count",
        "self_hand_count",
        "opponent_hand_count",
        "self_discard_count",
        "opponent_discard_count",
    }
)


def compact_global_features(values: Mapping[str, float | int | bool]) -> tuple[float, ...]:
    """Build the fixed global vector and reject accidental count leakage."""

    leaked = FORBIDDEN_COUNT_FEATURES.intersection(values)
    if leaked:
        raise ValueError(f"forbidden non-field count features supplied: {sorted(leaked)}")
    missing = [name for name in GLOBAL_FEATURE_NAMES if name not in values]
    if missing:
        raise ValueError(f"missing compact global features: {missing}")
    return tuple(float(values[name]) for name in GLOBAL_FEATURE_NAMES)
