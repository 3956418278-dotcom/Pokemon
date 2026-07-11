from __future__ import annotations

from data.card_preprocessing import energy_cost_dict, normalize_energy_symbol, parse_damage, parse_damage_mode, parse_energy_symbols


def test_card_preprocessing_helpers() -> None:
    assert parse_energy_symbols("{G}{C}{C}") == ["G", "C", "C"]
    assert energy_cost_dict("{G}{C}{C}") == {"C": 2, "G": 1}
    assert parse_damage("120+") == 120
    assert parse_damage_mode("120+") == "plus"
    assert parse_damage_mode("30x") == "times"
    assert normalize_energy_symbol("0") == "C"
    assert normalize_energy_symbol("7") == "D"
    assert normalize_energy_symbol("TEAM ROCKET") == "A"
