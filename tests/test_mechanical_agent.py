from __future__ import annotations

from competition_selfplay.mechanical_agent import (
    AREA_ACTIVE,
    AREA_BENCH,
    AREA_DECK,
    AREA_HAND,
    ATTACK_KANGASKHAN,
    ATTACK_RAGING_DRAW,
    ATTACK_TEAL,
    CIPHERMANIAC,
    BOSSES_ORDERS,
    CONTEXT_ACTIVATE,
    CONTEXT_DISCARD,
    CONTEXT_MAIN,
    CONTEXT_SETUP_BENCH,
    CONTEXT_TO_DECK,
    CONTEXT_TO_HAND,
    CRISPIN,
    CHIEN_PAO,
    ENERGY_SWITCH,
    FIGHTING_ENERGY,
    GLASS_TRUMPET,
    GRASS_ENERGY,
    LATIAS,
    MEGA_KANGASKHAN,
    MEOWTH,
    NIGHT_STRETCHER,
    PSYCHIC_ENERGY,
    MechanicalAgent,
    OPTION_ABILITY,
    OPTION_ATTACH,
    OPTION_ATTACK,
    OPTION_CARD,
    OPTION_END,
    OPTION_NO,
    OPTION_PLAY,
    OPTION_RETREAT,
    OPTION_YES,
    TEAL_OGERPON,
    ULTRA_BALL,
    WATER_ENERGY,
    is_legal_action,
)


def _pokemon(
    card_id: int,
    *,
    hp: int = 210,
    max_hp: int | None = None,
    energies: tuple[int, ...] = (),
    serial: int | None = None,
) -> dict:
    return {
        "id": card_id,
        "hp": hp,
        "maxHp": max_hp if max_hp is not None else hp,
        "serial": serial,
        "energies": list(energies),
        "energyCards": [{"id": energy, "serial": index} for index, energy in enumerate(energies)],
    }


def _observation(
    *,
    deck_count: int = 30,
    hand: list[dict] | None = None,
    active: dict | None = None,
    bench: list[dict] | None = None,
    opponent_active: dict | None = None,
    opponent_bench: list[dict] | None = None,
    context: int = CONTEXT_MAIN,
    options: list[dict] | None = None,
    effect: int | None = None,
    context_card: int | None = None,
    minimum: int = 1,
    maximum: int = 1,
    deck: list[dict] | None = None,
    discard: list[dict] | None = None,
    stadium: dict | None = None,
) -> dict:
    return {
        "current": {
            "yourIndex": 0,
            "players": [
                {
                    "deckCount": deck_count,
                    "hand": hand or [],
                    "discard": discard or [],
                    "active": [active] if active else [],
                    "bench": bench or [],
                },
                {
                    "deckCount": 30,
                    "hand": None,
                    "active": [opponent_active] if opponent_active else [],
                    "bench": opponent_bench or [],
                },
            ],
            "stadium": [stadium] if stadium else [],
        },
        "select": {
            "type": 0,
            "context": context,
            "minCount": minimum,
            "maxCount": maximum,
            "effect": {"id": effect} if effect else None,
            "contextCard": {"id": context_card} if context_card else None,
            "deck": deck,
            "option": options or [],
        },
    }


def test_low_deck_declines_teal_dance_and_allows_end() -> None:
    observation = _observation(
        deck_count=6,
        hand=[{"id": GRASS_ENERGY}],
        active=_pokemon(TEAL_OGERPON, energies=(1, 1)),
        context=CONTEXT_ACTIVATE,
        context_card=TEAL_OGERPON,
        options=[{"type": OPTION_YES}, {"type": OPTION_NO}],
    )
    action = MechanicalAgent().act(observation)
    assert action == [1]
    assert is_legal_action(observation, action)


def test_low_deck_does_not_play_deck_search() -> None:
    observation = _observation(
        deck_count=6,
        hand=[{"id": ULTRA_BALL}],
        active=_pokemon(TEAL_OGERPON),
        options=[{"type": OPTION_PLAY, "index": 0}, {"type": OPTION_END}],
    )
    assert MechanicalAgent().act(observation) == [1]


def test_first_deck_boundary_blocks_search_but_allows_teal_draw_one() -> None:
    search = _observation(
        deck_count=9,
        hand=[{"id": ULTRA_BALL}],
        active=_pokemon(MEGA_KANGASKHAN, hp=300),
        options=[{"type": OPTION_PLAY, "index": 0}, {"type": OPTION_END}],
    )
    assert MechanicalAgent().act(search) == [1]

    teal_draw = _observation(
        deck_count=9,
        hand=[{"id": GRASS_ENERGY}],
        active=_pokemon(MEGA_KANGASKHAN, hp=300, serial=1),
        bench=[_pokemon(TEAL_OGERPON, serial=2)],
        options=[
            {"type": OPTION_ABILITY, "area": AREA_BENCH, "index": 0},
            {"type": OPTION_END},
        ],
    )
    assert MechanicalAgent().act(teal_draw) == [0]


def test_draw_count_uses_largest_legal_amount_at_each_deck_boundary() -> None:
    options = [
        {"type": 0, "number": 0},
        {"type": 0, "number": 1},
        {"type": 0, "number": 3},
        {"type": 0, "number": 6},
    ]
    assert MechanicalAgent().act(_observation(
        deck_count=11,
        context=38,
        options=options,
    )) == [3]
    assert MechanicalAgent().act(_observation(
        deck_count=9,
        context=38,
        options=options,
    )) == [1]
    assert MechanicalAgent().act(_observation(
        deck_count=6,
        context=38,
        options=options,
    )) == [0]


def test_multi_card_pokemon_draw_obeys_cutoff_and_precedes_attack_when_allowed() -> None:
    options = [
        {"type": OPTION_ABILITY, "area": AREA_ACTIVE, "index": 0},
        {"type": OPTION_ATTACK, "attackId": ATTACK_KANGASKHAN},
    ]
    above_cutoff = _observation(
        deck_count=11,
        active=_pokemon(MEGA_KANGASKHAN, hp=300, energies=(1, 1, 1), serial=1),
        opponent_active=_pokemon(999, hp=210, serial=2),
        options=options,
    )
    assert MechanicalAgent().act(above_cutoff) == [0]

    at_cutoff = _observation(
        deck_count=10,
        active=_pokemon(MEGA_KANGASKHAN, hp=300, energies=(1, 1, 1), serial=1),
        opponent_active=_pokemon(999, hp=210, serial=2),
        options=options,
    )
    assert MechanicalAgent().act(at_cutoff) == [1]


def test_single_card_pokemon_draw_precedes_attack_between_seven_and_nine() -> None:
    observation = _observation(
        deck_count=9,
        hand=[{"id": GRASS_ENERGY, "serial": 30}],
        active=_pokemon(TEAL_OGERPON, energies=(1, 1, 1), serial=10),
        opponent_active=_pokemon(999, hp=210, energies=(1, 1, 1), serial=20),
        options=[
            {"type": OPTION_ABILITY, "area": AREA_ACTIVE, "index": 0},
            {"type": OPTION_ATTACK, "attackId": ATTACK_TEAL},
        ],
    )
    assert MechanicalAgent().act(observation) == [0]


def test_attack_waits_for_nonterminal_core_placement_even_when_it_can_knock_out() -> None:
    observation = _observation(
        hand=[{"id": LATIAS, "serial": 30}],
        active=_pokemon(TEAL_OGERPON, energies=(1, 1, 1), serial=10),
        opponent_active=_pokemon(999, hp=100, serial=20),
        options=[
            {"type": OPTION_PLAY, "index": 0},
            {"type": OPTION_ATTACK, "attackId": ATTACK_TEAL},
        ],
    )
    assert MechanicalAgent().act(observation) == [0]


def test_burst_roar_is_hard_blocked_near_deckout() -> None:
    observation = _observation(
        deck_count=9,
        active=_pokemon(63, hp=240, serial=1),
        options=[
            {"type": OPTION_ATTACK, "attackId": ATTACK_RAGING_DRAW},
            {"type": OPTION_END},
        ],
    )
    assert MechanicalAgent().act(observation) == [1]


def test_boss_is_used_for_energy_rich_target_when_teal_can_attack() -> None:
    observation = _observation(
        hand=[{"id": BOSSES_ORDERS}],
        active=_pokemon(TEAL_OGERPON, energies=(1, 1, 1)),
        opponent_active=_pokemon(999, hp=300),
        opponent_bench=[_pokemon(998, hp=210, energies=(1, 1, 1))],
        options=[{"type": OPTION_PLAY, "index": 0}, {"type": OPTION_END}],
    )
    assert MechanicalAgent().act(observation) == [0]


def test_crispin_puts_grass_in_hand_and_prefers_water_as_second_energy() -> None:
    agent = MechanicalAgent()
    deck_cards = [{"id": GRASS_ENERGY}, {"id": WATER_ENERGY}]
    to_hand = _observation(
        active=_pokemon(TEAL_OGERPON),
        bench=[_pokemon(108)],
        context=CONTEXT_TO_HAND,
        effect=CRISPIN,
        deck=deck_cards,
        options=[
            {"type": OPTION_CARD, "area": AREA_DECK, "index": 0, "playerIndex": 0},
            {"type": OPTION_CARD, "area": AREA_DECK, "index": 1, "playerIndex": 0},
        ],
    )
    assert agent.act(to_hand) == [0]


def test_forced_bench_trim_discards_chien_then_used_meowth() -> None:
    bench = [_pokemon(TEAL_OGERPON), _pokemon(LATIAS), _pokemon(MEOWTH, hp=170), _pokemon(CHIEN_PAO, hp=120)]
    observation = _observation(
        active=_pokemon(63, hp=240),
        bench=bench,
        context=CONTEXT_DISCARD,
        options=[
            {"type": OPTION_CARD, "area": AREA_BENCH, "index": index, "playerIndex": 0}
            for index in range(len(bench))
        ],
        minimum=2,
        maximum=2,
    )
    assert MechanicalAgent().act(observation) == [3, 2]


def test_return_to_deck_uses_inverse_importance() -> None:
    bench = [_pokemon(TEAL_OGERPON, energies=(1, 1, 1)), _pokemon(CHIEN_PAO, hp=120)]
    observation = _observation(
        active=_pokemon(63, hp=240),
        bench=bench,
        context=9,
        options=[
            {"type": OPTION_CARD, "area": AREA_BENCH, "index": 0, "playerIndex": 0},
            {"type": OPTION_CARD, "area": AREA_BENCH, "index": 1, "playerIndex": 0},
        ],
    )
    assert MechanicalAgent().act(observation) == [1]


def test_ciphermaniac_to_deck_context_uses_forward_acquisition_order() -> None:
    deck_cards = [{"id": GRASS_ENERGY}, {"id": ULTRA_BALL}, {"id": CHIEN_PAO}]
    observation = _observation(
        active=_pokemon(MEGA_KANGASKHAN, hp=300, serial=1),
        bench=[_pokemon(TEAL_OGERPON, energies=(GRASS_ENERGY,), serial=2)],
        context=CONTEXT_TO_DECK,
        effect=CIPHERMANIAC,
        deck=deck_cards,
        options=[
            {"type": OPTION_CARD, "area": AREA_DECK, "index": index, "playerIndex": 0}
            for index in range(len(deck_cards))
        ],
        minimum=2,
        maximum=2,
    )
    assert MechanicalAgent().act(observation) == [0, 1]


def test_ciphermaniac_advances_need_between_two_selected_cards() -> None:
    deck_cards = [
        {"id": TEAL_OGERPON, "serial": 1},
        {"id": TEAL_OGERPON, "serial": 2},
        {"id": GRASS_ENERGY, "serial": 3},
        {"id": LATIAS, "serial": 4},
    ]
    observation = _observation(
        active=_pokemon(MEGA_KANGASKHAN, hp=300, serial=10),
        context=CONTEXT_TO_DECK,
        effect=CIPHERMANIAC,
        deck=deck_cards,
        options=[
            {"type": OPTION_CARD, "area": AREA_DECK, "index": index, "playerIndex": 0}
            for index in range(len(deck_cards))
        ],
        minimum=2,
        maximum=2,
    )
    assert MechanicalAgent().act(observation) == [0, 2]


def test_bench_trim_protects_one_healthy_latias_copy() -> None:
    bench = [
        _pokemon(LATIAS, hp=40, max_hp=210, serial=1),
        _pokemon(LATIAS, hp=210, max_hp=210, serial=2),
        _pokemon(CHIEN_PAO, hp=120, serial=3),
    ]
    observation = _observation(
        active=_pokemon(MEGA_KANGASKHAN, hp=300, serial=4),
        bench=bench,
        context=CONTEXT_DISCARD,
        options=[
            {"type": OPTION_CARD, "area": AREA_BENCH, "index": index, "playerIndex": 0}
            for index in range(len(bench))
        ],
        minimum=2,
        maximum=2,
    )
    assert MechanicalAgent().act(observation) == [2, 0]


def test_main_phase_does_not_play_excess_teal_or_latias_copies() -> None:
    for extra_id in (TEAL_OGERPON, LATIAS):
        observation = _observation(
            hand=[{"id": extra_id, "serial": 30}],
            active=_pokemon(MEGA_KANGASKHAN, hp=300, serial=1),
            bench=[
                _pokemon(TEAL_OGERPON, serial=2),
                _pokemon(TEAL_OGERPON, serial=3),
                _pokemon(LATIAS, serial=4),
            ],
            options=[{"type": OPTION_PLAY, "index": 0}, {"type": OPTION_END}],
        )
        assert MechanicalAgent().act(observation) == [1]


def test_setup_bench_caps_core_copy_counts() -> None:
    hand = [
        {"id": TEAL_OGERPON, "serial": 1},
        {"id": TEAL_OGERPON, "serial": 2},
        {"id": TEAL_OGERPON, "serial": 3},
        {"id": LATIAS, "serial": 4},
        {"id": LATIAS, "serial": 5},
        {"id": MEGA_KANGASKHAN, "serial": 6},
    ]
    observation = _observation(
        hand=hand,
        active=_pokemon(MEGA_KANGASKHAN, hp=300, serial=10),
        context=CONTEXT_SETUP_BENCH,
        options=[
            {"type": OPTION_CARD, "area": AREA_HAND, "index": index, "playerIndex": 0}
            for index in range(len(hand))
        ],
        minimum=0,
        maximum=5,
    )
    selected_ids = [hand[index]["id"] for index in MechanicalAgent().act(observation)]
    assert selected_ids.count(TEAL_OGERPON) == 2
    assert selected_ids.count(LATIAS) == 1
    assert MEGA_KANGASKHAN in selected_ids


def test_energy_shuffle_uses_inverse_energy_importance() -> None:
    active = _pokemon(108, energies=(GRASS_ENERGY, WATER_ENERGY))
    observation = _observation(
        active=active,
        context=32,
        options=[
            {
                "type": 6,
                "area": AREA_ACTIVE,
                "index": 0,
                "playerIndex": 0,
                "energyIndex": 0,
            },
            {
                "type": 6,
                "area": AREA_ACTIVE,
                "index": 0,
                "playerIndex": 0,
                "energyIndex": 1,
            },
        ],
    )
    assert MechanicalAgent().act(observation) == [1]


def test_core_active_without_attack_must_retreat_before_end() -> None:
    observation = _observation(
        active=_pokemon(TEAL_OGERPON, hp=20, max_hp=210, energies=(6,), serial=10),
        bench=[
            _pokemon(LATIAS, serial=11),
            _pokemon(MEGA_KANGASKHAN, hp=300, serial=12),
        ],
        options=[{"type": OPTION_RETREAT}, {"type": OPTION_END}],
    )
    assert MechanicalAgent().act(observation) == [0]


def test_switch_after_core_retreat_chooses_non_core_high_hp_pivot() -> None:
    bench = [
        _pokemon(LATIAS, serial=11),
        _pokemon(MEGA_KANGASKHAN, hp=300, serial=12),
    ]
    observation = _observation(
        active=_pokemon(TEAL_OGERPON, hp=20, max_hp=210, energies=(6,), serial=10),
        bench=bench,
        context=3,
        options=[
            {"type": OPTION_CARD, "area": AREA_BENCH, "index": 0, "playerIndex": 0},
            {"type": OPTION_CARD, "area": AREA_BENCH, "index": 1, "playerIndex": 0},
        ],
    )
    assert MechanicalAgent().act(observation) == [1]


def test_first_teal_gets_third_grass_before_second_teal() -> None:
    primary = _pokemon(TEAL_OGERPON, energies=(1, 1), serial=10)
    secondary = _pokemon(TEAL_OGERPON, serial=11)
    observation = _observation(
        hand=[{"id": GRASS_ENERGY, "serial": 30}],
        active=_pokemon(MEGA_KANGASKHAN, hp=300, serial=12),
        bench=[primary, secondary],
        options=[
            {
                "type": 8,
                "index": 0,
                "inPlayArea": AREA_BENCH,
                "inPlayIndex": 0,
            },
            {
                "type": 8,
                "index": 0,
                "inPlayArea": AREA_BENCH,
                "inPlayIndex": 1,
            },
        ],
    )
    assert MechanicalAgent().act(observation) == [0]


def test_teal_dance_keeps_first_teal_concentration_order() -> None:
    primary = _pokemon(TEAL_OGERPON, energies=(1, 1), serial=10)
    secondary = _pokemon(TEAL_OGERPON, serial=11)
    observation = _observation(
        hand=[{"id": GRASS_ENERGY, "serial": 30}],
        active=_pokemon(MEGA_KANGASKHAN, hp=300, serial=12),
        bench=[primary, secondary],
        options=[
            {"type": OPTION_ABILITY, "area": AREA_BENCH, "index": 0},
            {"type": OPTION_ABILITY, "area": AREA_BENCH, "index": 1},
            {"type": OPTION_END},
        ],
    )
    assert MechanicalAgent().act(observation) == [0]


def test_primary_teal_keeps_energy_until_it_covers_each_opposing_pokemon() -> None:
    primary = _pokemon(TEAL_OGERPON, energies=(1, 1, 1), serial=10)
    secondary = _pokemon(TEAL_OGERPON, serial=11)
    observation = _observation(
        hand=[{"id": GRASS_ENERGY, "serial": 30}],
        active=_pokemon(MEGA_KANGASKHAN, hp=300, serial=12),
        bench=[primary, secondary, _pokemon(LATIAS, serial=13)],
        opponent_active=_pokemon(998, hp=210, energies=(1,) * 10, serial=20),
        opponent_bench=[_pokemon(999, hp=330, serial=21)],
        options=[
            {
                "type": OPTION_ATTACH,
                "index": 0,
                "inPlayArea": AREA_BENCH,
                "inPlayIndex": 0,
            },
            {
                "type": OPTION_ATTACH,
                "index": 0,
                "inPlayArea": AREA_BENCH,
                "inPlayIndex": 1,
            },
        ],
    )
    # The Active target is already covered because it carries 10 Energy, but
    # the 330 HP zero-Energy Bench target is not.  Energy stays concentrated
    # on the first 2 instead of beginning the second 2.
    assert MechanicalAgent().act(observation) == [0]


def test_search_keeps_grass_above_latias_until_primary_teal_covers_field() -> None:
    deck_cards = [{"id": GRASS_ENERGY}, {"id": LATIAS}]
    observation = _observation(
        active=_pokemon(MEGA_KANGASKHAN, hp=300, serial=12),
        bench=[_pokemon(TEAL_OGERPON, energies=(1, 1, 1), serial=10)],
        opponent_active=_pokemon(999, hp=330, serial=20),
        context=CONTEXT_TO_HAND,
        effect=ULTRA_BALL,
        deck=deck_cards,
        options=[
            {"type": OPTION_CARD, "area": AREA_DECK, "index": 0, "playerIndex": 0},
            {"type": OPTION_CARD, "area": AREA_DECK, "index": 1, "playerIndex": 0},
        ],
        minimum=0,
        maximum=1,
    )
    assert MechanicalAgent().act(observation) == [0]


def test_second_teal_dance_waits_when_primary_already_used_but_not_ready() -> None:
    primary = _pokemon(TEAL_OGERPON, energies=(GRASS_ENERGY,), serial=10)
    secondary = _pokemon(TEAL_OGERPON, serial=11)
    observation = _observation(
        hand=[{"id": GRASS_ENERGY, "serial": 30}],
        active=_pokemon(MEGA_KANGASKHAN, hp=300, serial=12),
        bench=[primary, secondary],
        options=[
            {"type": OPTION_ABILITY, "area": AREA_BENCH, "index": 1},
            {"type": OPTION_END},
        ],
    )
    assert MechanicalAgent().act(observation) == [1]


def test_non_grass_does_not_feed_critically_damaged_first_teal() -> None:
    primary = _pokemon(
        TEAL_OGERPON,
        hp=50,
        max_hp=210,
        energies=(1, 1, 1, 4),
        serial=10,
    )
    secondary = _pokemon(TEAL_OGERPON, serial=11)
    observation = _observation(
        hand=[{"id": 6, "serial": 30}],
        active=primary,
        bench=[secondary, _pokemon(LATIAS, serial=12)],
        opponent_active=_pokemon(999, hp=330, max_hp=330, energies=(6,), serial=99),
        options=[
            {"type": 8, "index": 0, "inPlayArea": AREA_ACTIVE, "inPlayIndex": 0},
            {"type": 8, "index": 0, "inPlayArea": AREA_BENCH, "inPlayIndex": 0},
        ],
    )
    assert MechanicalAgent().act(observation) == [1]


def test_ultra_ball_preserves_meowth_when_it_can_fetch_needed_crispin() -> None:
    hand = [
        {"id": CIPHERMANIAC, "serial": 1},
        {"id": NIGHT_STRETCHER, "serial": 2},
        {"id": ENERGY_SWITCH, "serial": 3},
        {"id": MEOWTH, "serial": 4},
        {"id": 1205, "serial": 5},
        {"id": 140, "serial": 6},
    ]
    observation = _observation(
        deck_count=15,
        hand=hand,
        discard=[{"id": GRASS_ENERGY, "serial": 40}],
        active=_pokemon(TEAL_OGERPON, energies=(1,), serial=10),
        bench=[_pokemon(MEGA_KANGASKHAN, hp=300, serial=11)],
        context=CONTEXT_DISCARD,
        effect=ULTRA_BALL,
        options=[
            {"type": OPTION_CARD, "area": AREA_HAND, "index": index, "playerIndex": 0}
            for index in range(len(hand))
        ],
        minimum=2,
        maximum=2,
    )
    action = MechanicalAgent().act(observation)
    selected = {hand[index]["id"] for index in action}
    assert selected == {CIPHERMANIAC, ENERGY_SWITCH}
    assert NIGHT_STRETCHER not in selected
    assert MEOWTH not in selected


def test_meowth_searches_needed_energy_supporter_despite_unrelated_supporter_in_hand() -> None:
    observation = _observation(
        hand=[
            {"id": MEOWTH, "serial": 1},
            {"id": BOSSES_ORDERS, "serial": 2},
        ],
        active=_pokemon(MEGA_KANGASKHAN, hp=300, energies=(1, 1, 1), serial=10),
        bench=[_pokemon(TEAL_OGERPON, energies=(GRASS_ENERGY,), serial=11)],
        opponent_active=_pokemon(999, hp=210, serial=20),
        options=[
            {"type": OPTION_PLAY, "index": 0},
            {"type": OPTION_ATTACK, "attackId": ATTACK_KANGASKHAN},
        ],
    )
    assert MechanicalAgent().act(observation) == [0]


def test_inverse_order_keeps_latias_over_second_teal_during_primary_power_gap() -> None:
    hand = [
        {"id": LATIAS, "serial": 1},
        {"id": TEAL_OGERPON, "serial": 2},
    ]
    observation = _observation(
        hand=hand,
        active=_pokemon(MEGA_KANGASKHAN, hp=300, serial=10),
        bench=[_pokemon(TEAL_OGERPON, energies=(1, 1, 1), serial=11)],
        opponent_active=_pokemon(999, hp=330, serial=20),
        context=CONTEXT_DISCARD,
        effect=ULTRA_BALL,
        options=[
            {"type": OPTION_CARD, "area": AREA_HAND, "index": 0, "playerIndex": 0},
            {"type": OPTION_CARD, "area": AREA_HAND, "index": 1, "playerIndex": 0},
        ],
        minimum=1,
        maximum=1,
    )
    assert MechanicalAgent().act(observation) == [1]


def test_search_order_follows_first_teal_grass_latias_second_teal() -> None:
    deck_cards = [
        {"id": TEAL_OGERPON},
        {"id": GRASS_ENERGY},
        {"id": LATIAS},
    ]
    options = [
        {"type": OPTION_CARD, "area": AREA_DECK, "index": index, "playerIndex": 0}
        for index in range(len(deck_cards))
    ]

    missing_first = _observation(
        active=_pokemon(MEGA_KANGASKHAN, hp=300, serial=1),
        context=CONTEXT_TO_HAND,
        effect=ULTRA_BALL,
        deck=deck_cards,
        options=options,
        minimum=0,
        maximum=1,
    )
    assert MechanicalAgent().act(missing_first) == [0]

    missing_grass = _observation(
        active=_pokemon(MEGA_KANGASKHAN, hp=300, serial=1),
        bench=[_pokemon(TEAL_OGERPON, energies=(GRASS_ENERGY,), serial=2)],
        context=CONTEXT_TO_HAND,
        effect=ULTRA_BALL,
        deck=deck_cards,
        options=options,
        minimum=0,
        maximum=1,
    )
    assert MechanicalAgent().act(missing_grass) == [1]

    missing_latias = _observation(
        active=_pokemon(MEGA_KANGASKHAN, hp=300, serial=1),
        bench=[_pokemon(TEAL_OGERPON, energies=(1, 1, 1), serial=2)],
        context=CONTEXT_TO_HAND,
        effect=ULTRA_BALL,
        deck=deck_cards,
        options=options,
        minimum=0,
        maximum=1,
    )
    assert MechanicalAgent().act(missing_latias) == [2]

    missing_second = _observation(
        active=_pokemon(MEGA_KANGASKHAN, hp=300, serial=1),
        bench=[
            _pokemon(TEAL_OGERPON, energies=(1, 1, 1), serial=2),
            _pokemon(LATIAS, serial=3),
        ],
        context=CONTEXT_TO_HAND,
        effect=ULTRA_BALL,
        deck=deck_cards,
        options=options,
        minimum=0,
        maximum=1,
    )
    assert MechanicalAgent().act(missing_second) == [0]


def test_episode_zero_turn_two_does_not_discard_grass_or_needed_high_hp_body() -> None:
    hand = [
        {"id": GLASS_TRUMPET, "serial": 43},
        {"id": FIGHTING_ENERGY, "serial": 58},
        {"id": MEGA_KANGASKHAN, "serial": 3},
        {"id": GRASS_ENERGY, "serial": 51},
        {"id": PSYCHIC_ENERGY, "serial": 61},
    ]
    observation = _observation(
        hand=hand,
        active=_pokemon(MEGA_KANGASKHAN, hp=300, serial=5),
        bench=[_pokemon(TEAL_OGERPON, energies=(GRASS_ENERGY, WATER_ENERGY), serial=10)],
        context=CONTEXT_DISCARD,
        effect=ULTRA_BALL,
        options=[
            {"type": OPTION_CARD, "area": AREA_HAND, "index": index, "playerIndex": 0}
            for index in range(len(hand))
        ],
        minimum=2,
        maximum=2,
    )
    selected = {hand[index]["id"] for index in MechanicalAgent().act(observation)}
    assert selected == {GLASS_TRUMPET, PSYCHIC_ENERGY}
    assert GRASS_ENERGY not in selected
    assert MEGA_KANGASKHAN not in selected
