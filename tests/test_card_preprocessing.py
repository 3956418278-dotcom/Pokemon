from __future__ import annotations

from data.card_preprocessing import energy_cost_dict, parse_damage, parse_energy_symbols


def test_card_preprocessing_helpers() -> None:
    assert parse_energy_symbols("{G}{C}{C}") == ["G", "C", "C"]
    assert energy_cost_dict("{G}{C}{C}") == {"C": 2, "G": 1}
    assert parse_damage("120+") == 120

