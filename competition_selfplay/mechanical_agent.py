"""Deterministic policy for the fixed Raging Bolt Ogerpon deck.

The policy consumes the competition's raw public observation.  This is
intentional: attack IDs and effect-chain context are not retained by the V1
model adapter, while a mechanical policy must distinguish those choices.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Any


# Card IDs -----------------------------------------------------------------
GRASS_ENERGY = 1
WATER_ENERGY = 3
LIGHTNING_ENERGY = 4
PSYCHIC_ENERGY = 5
FIGHTING_ENERGY = 6

RAGING_BOLT = 63
IRON_LEAVES = 75
TEAL_OGERPON = 96
WELLSPRING_OGERPON = 108
FEZANDIPITI = 140
LATIAS = 184
CHIEN_PAO = 209
LILLIES_CLEFAIRY = 272
MEGA_KANGASKHAN = 756
PASSIMIAN = 978
MEOWTH = 1071

UNFAIR_STAMP = 1080
NIGHT_STRETCHER = 1097
GLASS_TRUMPET = 1098
ENERGY_SWITCH = 1116
ULTRA_BALL = 1121
BOSSES_ORDERS = 1182
CIPHERMANIAC = 1188
CRISPIN = 1198
CYRANO = 1205
LILLIES_DETERMINATION = 1227
AREA_ZERO = 1250

POKEMON_IDS = frozenset(
    {
        RAGING_BOLT,
        IRON_LEAVES,
        TEAL_OGERPON,
        WELLSPRING_OGERPON,
        FEZANDIPITI,
        LATIAS,
        CHIEN_PAO,
        LILLIES_CLEFAIRY,
        MEGA_KANGASKHAN,
        PASSIMIAN,
        MEOWTH,
    }
)
ENERGY_IDS = frozenset(
    {GRASS_ENERGY, WATER_ENERGY, LIGHTNING_ENERGY, PSYCHIC_ENERGY, FIGHTING_ENERGY}
)
DECK_SEARCH_IDS = frozenset({MEOWTH, ULTRA_BALL, CIPHERMANIAC, CRISPIN, CYRANO})
DRAW_COUNTS = {
    TEAL_OGERPON: 1,
    MEGA_KANGASKHAN: 2,
    FEZANDIPITI: 3,
    UNFAIR_STAMP: 5,
    RAGING_BOLT: 6,
    LILLIES_DETERMINATION: 8,
}
DRAW_TRAINER_IDS = frozenset({UNFAIR_STAMP, LILLIES_DETERMINATION})

# Simulator enums ----------------------------------------------------------
AREA_DECK = 1
AREA_HAND = 2
AREA_DISCARD = 3
AREA_ACTIVE = 4
AREA_BENCH = 5
AREA_STADIUM = 7

OPTION_NUMBER = 0
OPTION_YES = 1
OPTION_NO = 2
OPTION_CARD = 3
OPTION_ENERGY_CARD = 5
OPTION_ENERGY = 6
OPTION_PLAY = 7
OPTION_ATTACH = 8
OPTION_ABILITY = 10
OPTION_RETREAT = 12
OPTION_ATTACK = 13
OPTION_END = 14

CONTEXT_MAIN = 0
CONTEXT_SETUP_ACTIVE = 1
CONTEXT_SETUP_BENCH = 2
CONTEXT_SWITCH = 3
CONTEXT_TO_ACTIVE = 4
CONTEXT_TO_BENCH = 5
CONTEXT_TO_FIELD = 6
CONTEXT_TO_HAND = 7
CONTEXT_DISCARD = 8
CONTEXT_TO_DECK = 9
CONTEXT_TO_DECK_BOTTOM = 10
CONTEXT_NOT_MOVE = 12
CONTEXT_DAMAGE = 15
CONTEXT_ATTACH_FROM = 21
CONTEXT_ATTACH_TO = 22
CONTEXT_EFFECT_TARGET = 25
CONTEXT_DISCARD_ENERGY_CARD = 26
CONTEXT_SWITCH_ENERGY_CARD = 28
CONTEXT_DISCARD_ENERGY = 30
CONTEXT_TO_DECK_ENERGY = 32
CONTEXT_SWITCH_ENERGY = 33
CONTEXT_ATTACK = 35
CONTEXT_DRAW_COUNT = 38
CONTEXT_IS_FIRST = 41
CONTEXT_MULLIGAN = 42
CONTEXT_ACTIVATE = 43

ATTACK_RAGING_DRAW = 71
ATTACK_RAGING_DAMAGE = 72
ATTACK_IRON_LEAVES = 89
ATTACK_TEAL = 120
ATTACK_WELLSPRING_SOB = 135
ATTACK_WELLSPRING_PUMP = 136
ATTACK_FEZANDIPITI = 183
ATTACK_LATIAS = 243
ATTACK_CHIEN_PAO = 281
ATTACK_CLEFAIRY = 371
ATTACK_KANGASKHAN = 1092
ATTACK_PASSIMIAN = 1407
ATTACK_MEOWTH = 1546

HP = {
    RAGING_BOLT: 240,
    IRON_LEAVES: 220,
    TEAL_OGERPON: 210,
    WELLSPRING_OGERPON: 210,
    FEZANDIPITI: 210,
    LATIAS: 210,
    CHIEN_PAO: 120,
    LILLIES_CLEFAIRY: 190,
    MEGA_KANGASKHAN: 300,
    PASSIMIAN: 110,
    MEOWTH: 170,
}


def _as_int(value: Any, default: int = -1) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _card_id(card: Any) -> int | None:
    if isinstance(card, dict):
        value = _as_int(card.get("id"), -1)
        return value if value >= 0 else None
    return None


def _serial(card: Any) -> int:
    if isinstance(card, dict):
        return _as_int(card.get("serial"), -1)
    return -1


def _energy_count(card: dict[str, Any] | None) -> int:
    return len((card or {}).get("energyCards") or (card or {}).get("energies") or [])


def _grass_count(card: dict[str, Any] | None) -> int:
    return sum(_as_int(value) == 1 for value in ((card or {}).get("energies") or []))


def _water_count(card: dict[str, Any] | None) -> int:
    return sum(_as_int(value) == 3 for value in ((card or {}).get("energies") or []))


@dataclass(frozen=True)
class PolicySnapshot:
    player_index: int
    deck_count: int
    self_hand: tuple[dict[str, Any], ...]
    hand_ids: tuple[int, ...]
    self_discard: tuple[dict[str, Any], ...]
    self_active: dict[str, Any] | None
    self_bench: tuple[dict[str, Any], ...]
    opponent_active: dict[str, Any] | None
    opponent_bench: tuple[dict[str, Any], ...]
    stadium: dict[str, Any] | None
    prize_count: int
    opponent_prize_count: int
    energy_attached: bool
    retreated: bool

    @property
    def self_field(self) -> tuple[dict[str, Any], ...]:
        return ((self.self_active,) if self.self_active is not None else ()) + self.self_bench

    @property
    def opponent_field(self) -> tuple[dict[str, Any], ...]:
        return ((self.opponent_active,) if self.opponent_active is not None else ()) + self.opponent_bench

    def field_count(self, card_id: int) -> int:
        return sum(_card_id(card) == card_id for card in self.self_field)

    def available_count(self, card_id: int) -> int:
        return self.field_count(card_id) + self.hand_ids.count(card_id)

    @property
    def has_latias(self) -> bool:
        return self.field_count(LATIAS) > 0

    @property
    def has_tera(self) -> bool:
        return any(_card_id(card) in {TEAL_OGERPON, WELLSPRING_OGERPON} for card in self.self_field)


class MechanicalAgent:
    """Rules-first agent for one fixed deck.

    All selections are made from the legal option list.  Unsupported contexts
    fall back to a stable legal choice and are counted for later audit.
    """

    def __init__(self) -> None:
        self.statistics: Counter[str] = Counter()
        self.primary_teal_serial: int | None = None
        self.secondary_teal_serial: int | None = None

    def reset(self) -> None:
        self.statistics.clear()
        self.primary_teal_serial = None
        self.secondary_teal_serial = None

    def act(self, observation: dict[str, Any]) -> list[int]:
        select = observation.get("select") or {}
        options = list(select.get("option") or [])
        if not options:
            return []
        snapshot = self._snapshot(observation)
        self._refresh_teal_roles(snapshot)
        context = _as_int(select.get("context"), -1)
        if context == CONTEXT_MAIN:
            return [self._best_main_option(observation, snapshot, options)]
        return self._effect_selection(observation, snapshot, options)

    def _snapshot(self, observation: dict[str, Any]) -> PolicySnapshot:
        current = observation.get("current") or {}
        players = current.get("players") or [{}, {}]
        player_index = _as_int(current.get("yourIndex"), 0)
        if player_index not in (0, 1):
            player_index = 0
        own = players[player_index] if player_index < len(players) else {}
        opponent_index = 1 - player_index
        opponent = players[opponent_index] if opponent_index < len(players) else {}
        hand = tuple(
            card_id for card in (own.get("hand") or []) if (card_id := _card_id(card)) is not None
        )
        own_active = next(iter(own.get("active") or []), None)
        opponent_active = next(iter(opponent.get("active") or []), None)
        stadium = next(iter(current.get("stadium") or []), None)
        return PolicySnapshot(
            player_index=player_index,
            deck_count=_as_int(own.get("deckCount"), 0),
            self_hand=tuple(
                card for card in (own.get("hand") or []) if isinstance(card, dict)
            ),
            hand_ids=hand,
            self_discard=tuple(
                card for card in (own.get("discard") or []) if isinstance(card, dict)
            ),
            self_active=own_active,
            self_bench=tuple(card for card in (own.get("bench") or []) if isinstance(card, dict)),
            opponent_active=opponent_active,
            opponent_bench=tuple(
                card for card in (opponent.get("bench") or []) if isinstance(card, dict)
            ),
            stadium=stadium,
            prize_count=len(own.get("prize") or []),
            opponent_prize_count=len(opponent.get("prize") or []),
            energy_attached=bool(current.get("energyAttached", False)),
            retreated=bool(current.get("retreated", False)),
        )

    def _refresh_teal_roles(self, snapshot: PolicySnapshot) -> None:
        """Keep stable first/second Teal identities for the current game.

        The submitted agent instance survives across calls and is reset on the
        deck request, so serial identity is the reliable meaning of "first 2".
        If it leaves play, the surviving second copy is promoted.
        """

        teals = [card for card in snapshot.self_field if _card_id(card) == TEAL_OGERPON]
        serials = [_serial(card) for card in teals if _serial(card) >= 0]
        if self.primary_teal_serial not in serials:
            if self.secondary_teal_serial in serials:
                self.primary_teal_serial = self.secondary_teal_serial
            else:
                self.primary_teal_serial = serials[0] if serials else None
            self.secondary_teal_serial = None
        if self.secondary_teal_serial not in serials or self.secondary_teal_serial == self.primary_teal_serial:
            self.secondary_teal_serial = next(
                (serial for serial in serials if serial != self.primary_teal_serial), None
            )

    def _teal_role(self, card: dict[str, Any] | None) -> int:
        serial = _serial(card)
        if serial >= 0 and serial == self.primary_teal_serial:
            return 1
        if serial >= 0 and serial == self.secondary_teal_serial:
            return 2
        return 3

    # Main phase -----------------------------------------------------------
    def _best_main_option(
        self,
        observation: dict[str, Any],
        snapshot: PolicySnapshot,
        options: list[dict[str, Any]],
    ) -> int:
        scored = [
            (self._main_priority(observation, snapshot, options, option), -index, index)
            for index, option in enumerate(options)
        ]
        return max(scored)[2]

    def _main_priority(
        self,
        observation: dict[str, Any],
        snapshot: PolicySnapshot,
        options: list[dict[str, Any]],
        option: dict[str, Any],
    ) -> tuple[int, float]:
        """Strict decision layers; numeric scores only break ties inside a layer."""

        option_type = _as_int(option.get("type"), -1)
        base = self._main_score(observation, snapshot, option)
        attack_options = [
            item for item in options if _as_int(item.get("type"), -1) == OPTION_ATTACK
        ]

        forced_core_retreat = (
            option_type == OPTION_RETREAT
            and self._must_vacate_active(snapshot, attack_options)
        )
        if (
            option_type in {OPTION_PLAY, OPTION_ABILITY, OPTION_RETREAT, OPTION_ATTACK}
            and base < 0
            and not forced_core_retreat
        ):
            return (-200, base)

        if option_type == OPTION_PLAY:
            card_id = self._hand_option_card_id(observation, snapshot, option)
            if card_id == BOSSES_ORDERS and self._boss_target(snapshot) is not None:
                return (990, base)
            if card_id == IRON_LEAVES and self._iron_leaves_immediate_knockout(snapshot):
                return (985, base)
            if card_id == AREA_ZERO and snapshot.has_tera:
                return (925, base)
            needed_core_play = (
                (card_id == TEAL_OGERPON and snapshot.field_count(TEAL_OGERPON) < 2)
                or (card_id == LATIAS and snapshot.field_count(LATIAS) < 1)
                or (
                    card_id == WELLSPRING_OGERPON
                    and snapshot.field_count(WELLSPRING_OGERPON) < 1
                )
            )
            if needed_core_play:
                return (920, base)

        if option_type == OPTION_ATTACH:
            source = self._hand_option_card_id(observation, snapshot, option)
            target = self._in_play_target(snapshot, option)
            if self._attachment_enables_knockout(source, target, snapshot):
                return (980, base)
            target_id = _card_id(target)
            if target_id == TEAL_OGERPON:
                role = self._teal_role(target)
                primary = self._primary_teal(snapshot)
                primary_covers_field = self._teal_covers_opponent_field(primary, snapshot)
                if role == 1 and source == GRASS_ENERGY and _grass_count(target) < 3:
                    return (975, base)
                if (
                    role == 1
                    and self._teal_needs_damage_energy(target, snapshot)
                    and not self._teal_is_endangered(target)
                ):
                    return (970, base)
                if (
                    role == 2
                    and primary_covers_field
                    and source == GRASS_ENERGY
                    and _grass_count(target) < 3
                ):
                    return (965, base)
            if target_id == WELLSPRING_OGERPON and (
                (source == WATER_ENERGY and _water_count(target) == 0)
                or _energy_count(target) < 3
            ):
                return (955, base)
            if (
                target_id == TEAL_OGERPON
                and self._teal_needs_damage_energy(target, snapshot)
                and not self._teal_is_endangered(target)
            ):
                return (930, base)

        if option_type == OPTION_ABILITY:
            card = self._field_option_card(snapshot, option)
            if _card_id(card) == TEAL_OGERPON and GRASS_ENERGY in snapshot.hand_ids:
                role = self._teal_role(card)
                primary = self._primary_teal(snapshot)
                if role == 1 and _grass_count(card) < 3:
                    return (975, base)
                if role == 1 and self._teal_needs_damage_energy(card, snapshot):
                    return (970, base)
                if (
                    role == 2
                    and self._teal_covers_opponent_field(primary, snapshot)
                    and _grass_count(card) < 3
                ):
                    return (965, base)
                return (940, base)
            if _card_id(card) in {MEGA_KANGASKHAN, FEZANDIPITI}:
                return (940, base)

        if option_type == OPTION_RETREAT and self._must_vacate_active(snapshot, attack_options):
            active = snapshot.self_active
            active_hp = _as_int((active or {}).get("hp"), 0)
            active_max = max(1, _as_int((active or {}).get("maxHp"), active_hp))
            urgent = active_hp * 3 <= active_max
            return (950 if urgent else 905, base)

        if option_type == OPTION_END:
            return (-100 if self._must_vacate_active(snapshot, attack_options) else 0, base)

        if option_type == OPTION_ATTACH:
            return (910, base)
        if option_type == OPTION_ABILITY:
            return (885, base)
        if option_type == OPTION_PLAY:
            return (880, base)
        if option_type == OPTION_RETREAT:
            return (300, base)
        if option_type == OPTION_ATTACK:
            # Attacking terminates the turn.  It is selected only after every
            # currently useful play, attachment, Ability, and retreat has been
            # exhausted; attack score chooses between terminal alternatives.
            return (200, base)
        return (100, base)

    def _must_vacate_active(
        self,
        snapshot: PolicySnapshot,
        attack_options: list[dict[str, Any]],
    ) -> bool:
        active_id = _card_id(snapshot.self_active)
        if active_id == LATIAS:
            return not self._latias_should_tank(snapshot)
        if active_id in {TEAL_OGERPON, WELLSPRING_OGERPON}:
            # An attack ends the turn and is the explicitly allowed exception.
            return not attack_options
        return False

    def _main_score(
        self,
        observation: dict[str, Any],
        snapshot: PolicySnapshot,
        option: dict[str, Any],
    ) -> float:
        option_type = _as_int(option.get("type"), -1)
        if option_type == OPTION_END:
            return 0.0
        if option_type == OPTION_PLAY:
            card_id = self._hand_option_card_id(observation, snapshot, option)
            return self._play_score(card_id, snapshot)
        if option_type == OPTION_ATTACH:
            source = self._hand_option_card_id(observation, snapshot, option)
            target = self._in_play_target(snapshot, option)
            return 750.0 + self._energy_target_score(source, target, snapshot)
        if option_type == OPTION_ABILITY:
            card = self._field_option_card(snapshot, option)
            return self._ability_score(_card_id(card), snapshot, card)
        if option_type == OPTION_RETREAT:
            return self._retreat_score(snapshot)
        if option_type == OPTION_ATTACK:
            return self._attack_score(_as_int(option.get("attackId"), -1), snapshot)
        return -100.0

    def _play_score(self, card_id: int | None, snapshot: PolicySnapshot) -> float:
        if card_id is None:
            return -1000.0
        if not self._deck_search_allowed(snapshot) and card_id in DECK_SEARCH_IDS:
            # A Basic may still be placed as a body; its optional search is
            # declined by the activation gate below.
            if card_id not in POKEMON_IDS:
                return -10000.0
        if card_id in DRAW_TRAINER_IDS and not self._voluntary_draw_allowed(
            snapshot, DRAW_COUNTS[card_id]
        ):
            return -10000.0
        if card_id in POKEMON_IDS:
            return self._pokemon_play_score(card_id, snapshot)
        if card_id == AREA_ZERO:
            return 1450.0 if snapshot.has_tera else -500.0
        if card_id == BOSSES_ORDERS:
            target = self._boss_target(snapshot)
            if target is not None and self._teal_can_attack(snapshot):
                return 5000.0 + _energy_count(target) * 100.0
            return -500.0
        if card_id == CRISPIN:
            return 1350.0 if self._deck_search_allowed(snapshot) and self._needs_energy(snapshot) else -500.0
        if card_id == CYRANO:
            return (
                1325.0
                if self._deck_search_allowed(snapshot)
                and (self._missing_core(snapshot) or self._high_hp_slots_missing(snapshot) > 0)
                else -500.0
            )
        if card_id == ULTRA_BALL:
            return (
                1275.0
                if self._deck_search_allowed(snapshot)
                and (self._missing_core(snapshot) or self._high_hp_slots_missing(snapshot) > 0)
                else -400.0
            )
        if card_id == LILLIES_DETERMINATION:
            draw_target = 8 if snapshot.prize_count == 6 else 6
            return (
                1500.0 + (draw_target - len(snapshot.hand_ids)) * 20.0
                if self._voluntary_draw_allowed(snapshot, draw_target)
                and len(snapshot.hand_ids) <= draw_target
                else -600.0
            )
        if card_id == UNFAIR_STAMP:
            return (
                1450.0
                if self._voluntary_draw_allowed(snapshot, DRAW_COUNTS[UNFAIR_STAMP])
                else -10000.0
            )
        if card_id == CIPHERMANIAC:
            return 850.0 if self._deck_search_allowed(snapshot) else -10000.0
        if card_id == ENERGY_SWITCH:
            return 900.0 if self._energy_switch_is_useful(snapshot) else -200.0
        if card_id == GLASS_TRUMPET:
            return 800.0 if snapshot.has_tera and self._has_colorless_bench(snapshot) else -300.0
        if card_id == NIGHT_STRETCHER:
            return 775.0
        return -100.0

    def _pokemon_play_score(self, card_id: int, snapshot: PolicySnapshot) -> float:
        count = snapshot.field_count(card_id)
        if card_id == TEAL_OGERPON:
            return 1900.0 if count == 0 else 1500.0 if count == 1 else -1000.0
        if card_id == LATIAS:
            return 1700.0 if count == 0 else -1000.0
        if card_id == WELLSPRING_OGERPON:
            if count > 0:
                return -1000.0
            return 1550.0 if WATER_ENERGY in snapshot.hand_ids else -500.0
        if card_id == MEOWTH:
            needed_supporter = self._needed_supporter_id(snapshot)
            return (
                1250.0
                if self._deck_search_allowed(snapshot)
                and needed_supporter is not None
                and needed_supporter not in snapshot.hand_ids
                else -1000.0
            )
        if card_id == IRON_LEAVES:
            if self._iron_leaves_immediate_knockout(snapshot):
                return 2200.0
            return 725.0 if self._high_hp_slots_missing(snapshot) > 0 else -500.0
        if card_id == FEZANDIPITI:
            return 1000.0 if self._high_hp_slots_missing(snapshot) > 0 else -500.0
        if card_id == MEGA_KANGASKHAN:
            return 875.0 if self._high_hp_slots_missing(snapshot) > 0 else -500.0
        if card_id == RAGING_BOLT:
            return 700.0 if self._high_hp_slots_missing(snapshot) > 0 else -500.0
        if card_id == LILLIES_CLEFAIRY:
            tactical = any(_card_id(card) == RAGING_BOLT for card in snapshot.opponent_field)
            return 700.0 + 30.0 * len(snapshot.opponent_bench) if tactical else -500.0
        if card_id == PASSIMIAN:
            return 650.0 if len(snapshot.self_field) >= 5 else -500.0
        if card_id == CHIEN_PAO:
            if snapshot.stadium and _card_id(snapshot.stadium) == AREA_ZERO:
                return -900.0
            return 600.0 if snapshot.stadium is not None else -1000.0
        return 100.0

    def _ability_score(
        self,
        card_id: int | None,
        snapshot: PolicySnapshot,
        card: dict[str, Any] | None,
    ) -> float:
        if card_id == TEAL_OGERPON:
            if (
                not self._voluntary_draw_allowed(snapshot, DRAW_COUNTS[TEAL_OGERPON])
                or GRASS_ENERGY not in snapshot.hand_ids
            ):
                return -10000.0
            role = self._teal_role(card)
            primary = self._primary_teal(snapshot)
            primary_covers_field = self._teal_covers_opponent_field(primary, snapshot)
            if role == 1 and _grass_count(card) < 3:
                return 5000.0 + _grass_count(card) * 100.0
            if role == 1 and self._teal_needs_damage_energy(card, snapshot):
                return 4800.0 + _energy_count(card) * 100.0
            if role == 2 and primary_covers_field and _grass_count(card) < 3:
                return 4500.0 + _grass_count(card) * 100.0
            if role != 1 and not primary_covers_field:
                # Teal Dance can attach only to its owner. Keep Grass in hand
                # for the first Teal until it can Knock Out every current
                # opposing Pokémon after that Pokémon becomes Active.
                return -10000.0
            return 1200.0 - role * 50.0
        if card_id == MEGA_KANGASKHAN:
            return (
                1200.0
                if self._voluntary_draw_allowed(snapshot, DRAW_COUNTS[MEGA_KANGASKHAN])
                else -10000.0
            )
        if card_id == FEZANDIPITI:
            return (
                1300.0
                if self._voluntary_draw_allowed(snapshot, DRAW_COUNTS[FEZANDIPITI])
                else -10000.0
            )
        if card_id == MEOWTH:
            return 1200.0 if self._deck_search_allowed(snapshot) else -10000.0
        if card_id == IRON_LEAVES:
            return 1700.0 if self._iron_leaves_immediate_knockout(snapshot) else 350.0
        if card_id == CHIEN_PAO:
            return 900.0 if self._should_remove_stadium(snapshot) else -1000.0
        return 250.0

    def _retreat_score(self, snapshot: PolicySnapshot) -> float:
        active = snapshot.self_active
        if active is None or not snapshot.self_bench:
            return -1000.0
        active_id = _card_id(active)
        best_bench = max(snapshot.self_bench, key=lambda card: self._combat_value(card, snapshot))
        best_value = self._combat_value(best_bench, snapshot)
        active_value = self._combat_value(active, snapshot)
        if active_id == LATIAS and self._latias_should_tank(snapshot):
            return -500.0
        if active_id in {LATIAS, TEAL_OGERPON, WELLSPRING_OGERPON} and best_value >= active_value:
            return 1250.0 + best_value - active_value
        if best_value > active_value + 100.0:
            return 1100.0 + best_value - active_value
        return -150.0

    # Effect chains --------------------------------------------------------
    def _effect_selection(
        self,
        observation: dict[str, Any],
        snapshot: PolicySnapshot,
        options: list[dict[str, Any]],
    ) -> list[int]:
        select = observation.get("select") or {}
        context = _as_int(select.get("context"), -1)
        effect_id = _card_id(select.get("effect"))
        context_card_id = _card_id(select.get("contextCard"))
        minimum = max(0, _as_int(select.get("minCount"), 0))
        maximum = max(minimum, _as_int(select.get("maxCount"), minimum))

        if context in {CONTEXT_IS_FIRST, CONTEXT_MULLIGAN, CONTEXT_ACTIVATE}:
            return [self._yes_no_choice(context, effect_id, context_card_id, snapshot, options)]
        if context == CONTEXT_SETUP_ACTIVE:
            return [self._best_by_score(options, lambda option: self._setup_active_score(
                self._option_card_id(observation, snapshot, option)
            ))]
        if context == CONTEXT_SETUP_BENCH:
            return self._choose_setup_bench(observation, snapshot, options, minimum, maximum)
        if effect_id == CYRANO and context == CONTEXT_TO_HAND:
            return self._choose_cyrano(observation, snapshot, options, minimum, maximum)
        if effect_id == CRISPIN and context == CONTEXT_TO_HAND:
            return [self._best_by_score(options, lambda option: 1000.0 if self._option_card_id(
                observation, snapshot, option
            ) == GRASS_ENERGY else self._energy_value(self._option_card_id(observation, snapshot, option)))]
        if effect_id == CRISPIN and context == CONTEXT_ATTACH_TO:
            return [self._best_by_score(options, lambda option: self._crispin_second_energy_score(
                self._option_card_id(observation, snapshot, option), snapshot
            ))]
        if effect_id in {TEAL_OGERPON, GLASS_TRUMPET} and context == CONTEXT_ATTACH_TO:
            return [self._best_by_score(options, lambda option: self._energy_value(
                self._option_card_id(observation, snapshot, option)
            ))]
        if effect_id in {ULTRA_BALL, MEOWTH, NIGHT_STRETCHER, CIPHERMANIAC} and context == CONTEXT_TO_HAND:
            return self._choose_ranked(
                observation,
                snapshot,
                options,
                minimum,
                maximum,
                lambda option: self._to_hand_score(
                    effect_id, self._option_card_id(observation, snapshot, option), snapshot
                ),
            )
        if context in {CONTEXT_SWITCH, CONTEXT_TO_ACTIVE}:
            return [self._best_by_score(options, lambda option: self._switch_target_score(
                self._option_card(observation, snapshot, option), snapshot,
                opponent=_as_int(option.get("playerIndex"), snapshot.player_index) != snapshot.player_index,
            ))]
        if context in {CONTEXT_ATTACH_FROM, CONTEXT_EFFECT_TARGET}:
            return self._choose_ranked(
                observation,
                snapshot,
                options,
                minimum,
                maximum,
                lambda option: self._energy_target_score(
                    context_card_id,
                    self._option_card(observation, snapshot, option),
                    snapshot,
                ),
            )
        if context == CONTEXT_DISCARD:
            return self._choose_ranked(
                observation,
                snapshot,
                options,
                minimum,
                maximum,
                lambda option: self._discard_score(
                    self._option_card(observation, snapshot, option),
                    snapshot,
                    field=_as_int(option.get("area"), -1) == AREA_BENCH,
                ),
            )
        if context == CONTEXT_SWITCH_ENERGY_CARD:
            return self._choose_ranked(
                observation,
                snapshot,
                options,
                minimum,
                maximum,
                lambda option: self._energy_release_score(
                    observation, snapshot, option, moving=True
                ),
            )
        if context == CONTEXT_DAMAGE:
            return self._choose_ranked(
                observation,
                snapshot,
                options,
                minimum,
                maximum,
                lambda option: self._damage_target_score(
                    self._option_card(observation, snapshot, option)
                ),
            )
        if context in {CONTEXT_DISCARD_ENERGY_CARD, CONTEXT_DISCARD_ENERGY}:
            return self._choose_energy_discard(observation, snapshot, options, minimum, maximum)
        if context in {CONTEXT_TO_DECK_ENERGY, CONTEXT_SWITCH_ENERGY}:
            return self._choose_ranked(
                observation,
                snapshot,
                options,
                minimum,
                maximum,
                lambda option: self._energy_release_score(
                    observation, snapshot, option, moving=False
                ),
            )
        if context == CONTEXT_DRAW_COUNT:
            allowed = 0 if snapshot.deck_count <= 6 else 1 if snapshot.deck_count <= 10 else 99
            return [self._best_by_score(options, lambda option: (
                _as_int(option.get("number"), 0)
                if _as_int(option.get("number"), 0) <= allowed
                else -10000.0 - _as_int(option.get("number"), 0)
            ))]
        if context == CONTEXT_ATTACK:
            return [self._best_by_score(options, lambda option: self._attack_score(
                _as_int(option.get("attackId"), -1), snapshot
            ))]
        if context in {CONTEXT_TO_HAND, CONTEXT_TO_BENCH, CONTEXT_TO_FIELD}:
            return self._choose_ranked(
                observation,
                snapshot,
                options,
                minimum,
                maximum,
                lambda option: self._generic_card_score(
                    self._option_card_id(observation, snapshot, option), snapshot
                ),
            )
        if effect_id == CIPHERMANIAC and context == CONTEXT_TO_DECK:
            # Ciphermaniac selects cards from the deck to place on top. It is
            # acquisition, despite the simulator context name, so use the
            # forward order rather than the disposal inverse.
            return self._choose_sequential_acquisition(
                observation, snapshot, options, minimum, maximum
            )
        if context in {CONTEXT_TO_DECK, CONTEXT_TO_DECK_BOTTOM, CONTEXT_NOT_MOVE}:
            return self._choose_ranked(
                observation,
                snapshot,
                options,
                minimum,
                maximum,
                lambda option: self._inverse_importance_key(
                    self._option_card(observation, snapshot, option), snapshot
                ),
            )

        self.statistics[f"fallback:{_as_int(select.get('type'), -1)}:{context}:{effect_id}"] += 1
        return list(range(min(minimum if maximum > 1 else 1, maximum, len(options))))

    # Choice scoring -------------------------------------------------------
    def _yes_no_choice(
        self,
        context: int,
        effect_id: int | None,
        context_card_id: int | None,
        snapshot: PolicySnapshot,
        options: list[dict[str, Any]],
    ) -> int:
        choose_yes = True
        if context == CONTEXT_IS_FIRST:
            choose_yes = False
        elif context == CONTEXT_MULLIGAN:
            choose_yes = True
        elif context == CONTEXT_ACTIVATE:
            card_id = context_card_id or effect_id
            if card_id in DRAW_COUNTS:
                choose_yes = self._voluntary_draw_allowed(snapshot, DRAW_COUNTS[card_id])
                if card_id == TEAL_OGERPON:
                    choose_yes = choose_yes and GRASS_ENERGY in snapshot.hand_ids
            elif card_id == MEOWTH:
                choose_yes = self._deck_search_allowed(snapshot)
            elif card_id == CHIEN_PAO:
                choose_yes = self._should_remove_stadium(snapshot)
            elif card_id == IRON_LEAVES:
                choose_yes = self._iron_leaves_immediate_knockout(snapshot)
        wanted = OPTION_YES if choose_yes else OPTION_NO
        for index, option in enumerate(options):
            if _as_int(option.get("type"), -1) == wanted:
                return index
        return 0

    def _setup_active_score(self, card_id: int | None) -> float:
        # Keep the protected core on the Bench where possible.  Mega is not the
        # default tank because it yields three Prize cards when Knocked Out.
        return {
            RAGING_BOLT: 1000.0,
            IRON_LEAVES: 950.0,
            FEZANDIPITI: 900.0,
            MEGA_KANGASKHAN: 850.0,
            PASSIMIAN: 800.0,
            CHIEN_PAO: 750.0,
            LILLIES_CLEFAIRY: 700.0,
            MEOWTH: 650.0,
            LATIAS: 500.0,
            WELLSPRING_OGERPON: 450.0,
            TEAL_OGERPON: 400.0,
        }.get(card_id, 0.0)

    def _choose_setup_bench(
        self,
        observation: dict[str, Any],
        snapshot: PolicySnapshot,
        options: list[dict[str, Any]],
        minimum: int,
        maximum: int,
    ) -> list[int]:
        ranked = sorted(
            range(len(options)),
            key=lambda index: (
                self._generic_card_score(
                    self._option_card_id(observation, snapshot, options[index]), snapshot
                ),
                -index,
            ),
            reverse=True,
        )
        selected: list[int] = []
        counts = Counter(_card_id(card) for card in snapshot.self_field)
        caps = {TEAL_OGERPON: 2, LATIAS: 1, WELLSPRING_OGERPON: 1}
        for index in ranked:
            if len(selected) >= maximum:
                break
            card_id = self._option_card_id(observation, snapshot, options[index])
            if card_id in caps and counts[card_id] >= caps[card_id]:
                continue
            selected.append(index)
            counts[card_id] += 1
        for index in ranked:
            if len(selected) >= minimum:
                break
            if index not in selected:
                selected.append(index)
        return selected

    def _choose_cyrano(
        self,
        observation: dict[str, Any],
        snapshot: PolicySnapshot,
        options: list[dict[str, Any]],
        minimum: int,
        maximum: int,
    ) -> list[int]:
        return self._choose_sequential_acquisition(
            observation, snapshot, options, minimum, maximum
        )

    def _choose_sequential_acquisition(
        self,
        observation: dict[str, Any],
        snapshot: PolicySnapshot,
        options: list[dict[str, Any]],
        minimum: int,
        maximum: int,
    ) -> list[int]:
        """Choose multiple cards while advancing the formation need per pick."""

        desired = self._desired_acquisition_ids(snapshot)
        selected: list[int] = []
        used: set[int] = set()
        for wanted in desired:
            for index, option in enumerate(options):
                if index in used:
                    continue
                if self._option_card_id(observation, snapshot, option) == wanted:
                    selected.append(index)
                    used.add(index)
                    break
            if len(selected) >= maximum:
                break

        owned_teal = snapshot.available_count(TEAL_OGERPON) + sum(
            self._option_card_id(observation, snapshot, options[index]) == TEAL_OGERPON
            for index in selected
        )
        owned_latias = snapshot.available_count(LATIAS) + sum(
            self._option_card_id(observation, snapshot, options[index]) == LATIAS
            for index in selected
        )
        owned_wellspring = snapshot.available_count(WELLSPRING_OGERPON) + sum(
            self._option_card_id(observation, snapshot, options[index]) == WELLSPRING_OGERPON
            for index in selected
        )
        ranked = sorted(
            (index for index in range(len(options)) if index not in used),
            key=lambda index: (
                self._generic_card_score(
                    self._option_card_id(observation, snapshot, options[index]), snapshot
                ),
                -index,
            ),
            reverse=True,
        )
        for index in ranked:
            if len(selected) >= maximum:
                break
            card_id = self._option_card_id(observation, snapshot, options[index])
            if card_id == TEAL_OGERPON and owned_teal >= 2:
                continue
            if card_id == LATIAS and owned_latias >= 1:
                continue
            if card_id == WELLSPRING_OGERPON and owned_wellspring >= 1:
                continue
            selected.append(index)
            owned_teal += card_id == TEAL_OGERPON
            owned_latias += card_id == LATIAS
            owned_wellspring += card_id == WELLSPRING_OGERPON
        for index in range(len(options)):
            if len(selected) >= minimum:
                break
            if index not in selected:
                selected.append(index)
        return selected

    def _desired_acquisition_ids(self, snapshot: PolicySnapshot) -> list[int]:
        desired: list[int] = []
        if (
            WATER_ENERGY in snapshot.hand_ids
            and snapshot.available_count(WELLSPRING_OGERPON) < 1
        ):
            desired.append(WELLSPRING_OGERPON)

        teal_owned = snapshot.available_count(TEAL_OGERPON)
        if teal_owned < 1:
            desired.append(TEAL_OGERPON)
            teal_owned += 1

        grass_in_hand = snapshot.hand_ids.count(GRASS_ENERGY)
        energy_in_hand = sum(card_id in ENERGY_IDS for card_id in snapshot.hand_ids)
        primary = self._primary_teal(snapshot)
        primary_grass_gap = max(0, 3 - _grass_count(primary))
        primary_power_gap = max(
            0,
            self._teal_required_energy_count(primary, snapshot) - _energy_count(primary),
        )
        primary_gap = max(
            max(0, primary_grass_gap - grass_in_hand),
            max(0, primary_power_gap - energy_in_hand),
        )
        desired.extend([GRASS_ENERGY] * primary_gap)
        grass_used_for_primary = min(grass_in_hand, max(primary_grass_gap, primary_power_gap))
        grass_remaining = grass_in_hand - grass_used_for_primary

        if snapshot.available_count(LATIAS) < 1:
            desired.append(LATIAS)
        if teal_owned < 2:
            desired.append(TEAL_OGERPON)

        secondary = next(
            (
                card
                for card in snapshot.self_field
                if _card_id(card) == TEAL_OGERPON
                and _serial(card) == self.secondary_teal_serial
            ),
            None,
        )
        secondary_gap = max(0, 3 - _grass_count(secondary))
        grass_used_for_secondary = min(grass_remaining, secondary_gap)
        desired.extend([GRASS_ENERGY] * (secondary_gap - grass_used_for_secondary))
        return desired

    def _to_hand_score(
        self, effect_id: int | None, card_id: int | None, snapshot: PolicySnapshot
    ) -> float:
        del effect_id
        return self._generic_card_score(card_id, snapshot)

    def _generic_card_score(self, card_id: int | None, snapshot: PolicySnapshot) -> float:
        if card_id is None:
            return 0.0
        return self._card_importance({"id": card_id}, snapshot)

    def _discard_score(
        self, card: dict[str, Any] | None, snapshot: PolicySnapshot, *, field: bool
    ) -> tuple[int, ...]:
        del field
        return self._inverse_importance_key(card, snapshot)

    def _inverse_importance_key(
        self, card: dict[str, Any] | None, snapshot: PolicySnapshot
    ) -> tuple[int, ...]:
        return tuple(-value for value in self._card_importance_key(card, snapshot))

    def _card_importance_key(
        self, card: dict[str, Any] | None, snapshot: PolicySnapshot
    ) -> tuple[int, int, int, int]:
        """One state-dependent ordering shared by all card-removal choices.

        The ordering branches only on explicitly named key-card/resource
        conditions; it never computes an aggregate board-potential score.
        Every card nevertheless receives a key, so discard/shuffle decisions
        are inverse sorts instead of unrelated local weights.
        """

        card_id = _card_id(card)
        if card_id is None:
            return (0, 0, 0, 0)
        discard_ids = Counter(_card_id(item) for item in snapshot.self_discard)
        investment = _energy_count(card) * 100
        hp = max(0, _as_int(card.get("hp"), 0)) if isinstance(card, dict) else 0
        serial_tie = -max(0, _serial(card))

        phase = self._formation_phase(snapshot)
        search_allowed = snapshot.deck_count >= 10
        body_slots_missing = self._high_hp_slots_missing(snapshot)
        wellspring_route = (
            WATER_ENERGY in snapshot.hand_ids
            and snapshot.available_count(WELLSPRING_OGERPON) == 0
        )
        need = 0

        # Every one of the fixed deck's 27 card IDs has an explicit branch.
        # `need` expresses the current key-card gap; `tier` is the within-state
        # order. No broad Trainer/Supporter/Pokemon bucket participates here.
        if card_id == TEAL_OGERPON:
            role = self._teal_role(card)
            if role == 3:
                present = snapshot.field_count(TEAL_OGERPON)
                role = 1 if present == 0 else 2 if present == 1 else 3
            need = 6 if phase in {"first_teal", "second_teal"} else 0
            tier = {1: 100, 2: 90, 3: 54}[role]
        elif card_id == GRASS_ENERGY:
            need = 6 if phase in {"primary_grass", "primary_power", "second_grass"} else 0
            tier = 95
        elif card_id == LATIAS:
            need = 6 if phase == "latias" else 0
            tier = 94 if self._is_protected_latias(card, snapshot) else 52
        elif card_id == WELLSPRING_OGERPON:
            need = 7 if wellspring_route else 0
            tier = 94 if WATER_ENERGY in snapshot.hand_ids or _water_count(card) else 62
        elif card_id == WATER_ENERGY:
            need = 6 if snapshot.available_count(WELLSPRING_OGERPON) else 4
            tier = 90 if snapshot.available_count(WELLSPRING_OGERPON) else 70
        elif card_id == CRISPIN:
            need = 5 if search_allowed and phase in {"primary_grass", "primary_power", "second_grass"} else 0
            tier = 84 if search_allowed and self._needs_energy(snapshot) else 30
        elif card_id == CYRANO:
            need = 5 if search_allowed and phase in {"first_teal", "latias", "second_teal"} else 0
            tier = 80 if search_allowed and self._missing_core(snapshot) else 24
        elif card_id == ULTRA_BALL:
            need = 4 if search_allowed and phase in {"first_teal", "latias", "second_teal", "bodies"} else 0
            tier = 76 if search_allowed and (self._missing_core(snapshot) or body_slots_missing) else 22
        elif card_id == MEOWTH:
            if isinstance(card, dict) and "hp" in card:
                tier = 5  # Its on-play Supporter search has already been spent.
            else:
                needed_supporter = self._needed_supporter_id(snapshot)
                supporter_route = (
                    needed_supporter is not None
                    and needed_supporter not in snapshot.hand_ids
                )
                need = 3 if search_allowed and supporter_route else 0
                tier = 68 if search_allowed and supporter_route else 18
        elif card_id == ENERGY_SWITCH:
            need = 3 if self._energy_switch_is_useful(snapshot) else 0
            tier = 86 if self._energy_switch_is_useful(snapshot) else 66
        elif card_id == NIGHT_STRETCHER:
            phase_target = {
                "first_teal": TEAL_OGERPON,
                "primary_grass": GRASS_ENERGY,
                "primary_power": GRASS_ENERGY,
                "latias": LATIAS,
                "second_teal": TEAL_OGERPON,
                "second_grass": GRASS_ENERGY,
            }.get(phase)
            recovers_key = phase_target is not None and discard_ids[phase_target] > 0
            need = 5 if recovers_key else 0
            tier = 89 if recovers_key else 64
        elif card_id == GLASS_TRUMPET:
            useful = snapshot.has_tera and self._has_colorless_bench(snapshot) and any(
                _card_id(item) in ENERGY_IDS for item in snapshot.self_discard
            )
            need = 2 if useful and phase in {"primary_grass", "primary_power", "second_grass"} else 0
            tier = 82 if useful else 54
        elif card_id == AREA_ZERO:
            active_area_zero = _card_id(snapshot.stadium) == AREA_ZERO
            need = 3 if snapshot.has_tera and not active_area_zero else 0
            tier = 72 if snapshot.has_tera and (not active_area_zero or len(snapshot.self_bench) > 5) else 45
        elif card_id == BOSSES_ORDERS:
            tactical = self._boss_target(snapshot) is not None and self._teal_can_attack(snapshot)
            need = 6 if tactical else 0
            tier = 93 if tactical else 28
        elif card_id == CIPHERMANIAC:
            tier = 38 if search_allowed else 12
        elif card_id == LILLIES_DETERMINATION:
            draw_target = 8 if snapshot.prize_count == 6 else 6
            tier = 42 if snapshot.deck_count > 10 and len(snapshot.hand_ids) <= draw_target else 14
        elif card_id == UNFAIR_STAMP:
            tier = 48 if snapshot.deck_count > 10 else 12
        elif card_id == FEZANDIPITI:
            need = 2 if body_slots_missing else 0
            tier = 78 if snapshot.deck_count > 10 else 64
        elif card_id == MEGA_KANGASKHAN:
            need = 2 if body_slots_missing else 0
            tier = 76 if body_slots_missing else 58
        elif card_id == IRON_LEAVES:
            tactical = self._iron_leaves_immediate_knockout(snapshot)
            need = 6 if tactical else 2 if body_slots_missing else 0
            tier = 92 if tactical else 74
        elif card_id == RAGING_BOLT:
            need = 2 if body_slots_missing else 0
            tier = 76
        elif card_id == LILLIES_CLEFAIRY:
            tier = 60
        elif card_id == PASSIMIAN:
            tier = 46
        elif card_id == LIGHTNING_ENERGY:
            need = 2 if snapshot.available_count(RAGING_BOLT) else 0
            tier = 74 if snapshot.available_count(RAGING_BOLT) else 56
        elif card_id == FIGHTING_ENERGY:
            need = 2 if snapshot.available_count(RAGING_BOLT) else 0
            tier = 73 if snapshot.available_count(RAGING_BOLT) else 55
        elif card_id == PSYCHIC_ENERGY:
            need = 2 if snapshot.available_count(LATIAS) else 0
            tier = 64 if snapshot.available_count(LATIAS) or snapshot.available_count(LILLIES_CLEFAIRY) else 52
        elif card_id == CHIEN_PAO:
            tier = 0  # One-use Stadium-removal tool remains first disposal.
        else:
            raise AssertionError(f"fixed deck card {card_id} has no explicit importance rule")
        if card_id in POKEMON_IDS and card_id != CHIEN_PAO:
            # Attached key Energy changes the current card ordering; this is
            # why a powered generic body is not discarded before an empty
            # draw/search engine.
            grass = _grass_count(card)
            water = _water_count(card)
            other = max(0, _energy_count(card) - grass - water)
            tier += min(30, grass * 20 + water * 10 + other * 5)
        return (need * 100 + tier, investment, hp, serial_tie)

    def _card_importance(
        self, card: dict[str, Any] | None, snapshot: PolicySnapshot
    ) -> float:
        tier, investment, hp, _serial_value = self._card_importance_key(card, snapshot)
        return float(tier * 10000 + investment + hp)

    def _energy_donor_score(
        self, parent: dict[str, Any] | None, snapshot: PolicySnapshot
    ) -> tuple[int, ...]:
        if parent is None:
            return (-1000,)
        parent_id = _card_id(parent)
        score = list(self._inverse_importance_key(parent, snapshot))
        if parent_id == TEAL_OGERPON and _grass_count(parent) <= 3:
            score[0] -= 1000
        if parent_id == WELLSPRING_OGERPON and _energy_count(parent) <= 3:
            score[0] -= 900
        return tuple(score)

    def _switch_target_score(
        self, card: dict[str, Any] | None, snapshot: PolicySnapshot, *, opponent: bool
    ) -> float:
        if card is None:
            return -1000.0
        if opponent:
            return _energy_count(card) * 1000.0 - _as_int(card.get("hp"), 0)
        card_id = _card_id(card)
        if card_id == IRON_LEAVES and self._iron_leaves_immediate_knockout(snapshot):
            return 10000.0
        if card_id == TEAL_OGERPON and _grass_count(card) >= 3:
            damage = 30 + 30 * (_energy_count(card) + _energy_count(snapshot.opponent_active))
            if damage >= _as_int((snapshot.opponent_active or {}).get("hp"), 9999):
                return 9000.0
        if card_id == WELLSPRING_OGERPON and _water_count(card) > 0 and _energy_count(card) >= 3:
            return 8500.0
        # The default pivot must keep 1/2/3 on the Bench. Remaining HP is the
        # decisive state; Energy investment is only a tie-breaker.
        core_penalty = 6000.0 if card_id in {LATIAS, TEAL_OGERPON, WELLSPRING_OGERPON} else 0.0
        prize_penalty = 250.0 if card_id == MEGA_KANGASKHAN else 0.0
        return (
            _as_int(card.get("hp"), HP.get(card_id or -1, 0)) * 10.0
            + _energy_count(card) * 20.0
            - core_penalty
            - prize_penalty
        )

    def _damage_target_score(self, card: dict[str, Any] | None) -> float:
        if card is None:
            return -1000.0
        return _energy_count(card) * 1000.0 - _as_int(card.get("hp"), 0)

    def _energy_target_score(
        self,
        energy_card_id: int | None,
        target: dict[str, Any] | None,
        snapshot: PolicySnapshot,
    ) -> float:
        if target is None:
            return -1000.0
        target_id = _card_id(target)
        energies = _energy_count(target)
        if self._attachment_enables_knockout(energy_card_id, target, snapshot):
            return 12000.0
        if target_id == TEAL_OGERPON:
            role = self._teal_role(target)
            primary = self._primary_teal(snapshot)
            primary_covers_field = self._teal_covers_opponent_field(primary, snapshot)
            endangered_penalty = 5000.0 if self._teal_is_endangered(target) else 0.0
            if energy_card_id == GRASS_ENERGY:
                if role == 1 and _grass_count(target) < 3:
                    return 9000.0 + _grass_count(target) * 100.0 - endangered_penalty
                if role == 1 and self._teal_needs_damage_energy(target, snapshot):
                    return 8800.0 + _energy_count(target) * 100.0 - endangered_penalty
                if role == 2 and primary_covers_field and _grass_count(target) < 3:
                    return 8500.0 + _grass_count(target) * 100.0 - endangered_penalty
                if _grass_count(target) < 3:
                    return 6000.0 - role * 100.0 - endangered_penalty
            if role == 1 and self._teal_needs_damage_energy(target, snapshot):
                return 8700.0 + energies * 50.0 - endangered_penalty
            # Non-Grass does not advance the mandatory GGG attack cost.
            return 6500.0 - role * 100.0 - endangered_penalty + self._energy_value(energy_card_id) / 10.0
        if target_id == WELLSPRING_OGERPON:
            if energy_card_id == WATER_ENERGY and _water_count(target) == 0:
                return 8800.0
            if _water_count(target) > 0 and energies < 3:
                return 6200.0 + max(0, 3 - energies) * 100.0
            return 2600.0 + self._energy_value(energy_card_id) / 10.0
        if target_id == IRON_LEAVES and energy_card_id == GRASS_ENERGY:
            return 5800.0
        if target_id == RAGING_BOLT and energy_card_id in {LIGHTNING_ENERGY, FIGHTING_ENERGY}:
            missing_lightning = not any(
                _as_int(value) == LIGHTNING_ENERGY for value in (target.get("energies") or [])
            )
            missing_fighting = not any(
                _as_int(value) == FIGHTING_ENERGY for value in (target.get("energies") or [])
            )
            if (energy_card_id == LIGHTNING_ENERGY and missing_lightning) or (
                energy_card_id == FIGHTING_ENERGY and missing_fighting
            ):
                return 6100.0
            return 3200.0
        if not snapshot.has_latias:
            return 800.0 + max(0, 3 - energies) * 100.0
        return 1500.0 + HP.get(target_id or -1, 0) + self._energy_value(energy_card_id) / 10.0

    def _primary_teal(self, snapshot: PolicySnapshot) -> dict[str, Any] | None:
        return next(
            (
                card
                for card in snapshot.self_field
                if _card_id(card) == TEAL_OGERPON
                and _serial(card) == self.primary_teal_serial
            ),
            None,
        )

    def _teal_is_endangered(self, card: dict[str, Any] | None) -> bool:
        hp = _as_int((card or {}).get("hp"), 0)
        max_hp = max(1, _as_int((card or {}).get("maxHp"), HP[TEAL_OGERPON]))
        return hp * 3 <= max_hp

    def _teal_needs_damage_energy(
        self, card: dict[str, Any] | None, snapshot: PolicySnapshot
    ) -> bool:
        return (
            card is not None
            and _grass_count(card) >= 3
            and not self._teal_covers_opponent_field(card, snapshot)
        )

    def _teal_covers_opponent_field(
        self, card: dict[str, Any] | None, snapshot: PolicySnapshot
    ) -> bool:
        if card is None or _grass_count(card) < 3:
            return False
        return all(
            30 + 30 * (_energy_count(card) + _energy_count(target))
            >= _as_int(target.get("hp"), 0)
            for target in snapshot.opponent_field
        )

    def _teal_required_energy_count(
        self, card: dict[str, Any] | None, snapshot: PolicySnapshot
    ) -> int:
        required = 3
        for target in snapshot.opponent_field:
            hp = max(0, _as_int(target.get("hp"), 0))
            target_energy = _energy_count(target)
            required = max(required, math.ceil(max(0, hp - 30) / 30.0) - target_energy)
        return max(3, required)

    def _attachment_enables_knockout(
        self,
        energy_card_id: int | None,
        target: dict[str, Any] | None,
        snapshot: PolicySnapshot,
    ) -> bool:
        if target is None or snapshot.opponent_active is None:
            return False
        can_be_active = target is snapshot.self_active or (
            snapshot.has_latias and not snapshot.retreated and target in snapshot.self_bench
        )
        if not can_be_active:
            return False
        target_id = _card_id(target)
        opponent_hp = _as_int(snapshot.opponent_active.get("hp"), 9999)
        total_after = _energy_count(target) + 1
        grass_after = _grass_count(target) + (energy_card_id == GRASS_ENERGY)
        water_after = _water_count(target) + (energy_card_id == WATER_ENERGY)
        energies_after = list(target.get("energies") or []) + [energy_card_id]
        if target_id == TEAL_OGERPON:
            damage = 30 + 30 * (total_after + _energy_count(snapshot.opponent_active))
            return grass_after >= 3 and damage >= opponent_hp
        if target_id == WELLSPRING_OGERPON:
            return (water_after >= 1 and total_after >= 3 and opponent_hp <= 100) or (
                total_after >= 1 and opponent_hp <= 20
            )
        if target_id == IRON_LEAVES:
            return grass_after >= 2 and total_after >= 3 and opponent_hp <= 180
        if target_id == RAGING_BOLT:
            has_lightning = LIGHTNING_ENERGY in energies_after
            has_fighting = FIGHTING_ENERGY in energies_after
            board_energy = 1 + sum(_energy_count(card) for card in snapshot.self_field)
            return has_lightning and has_fighting and board_energy * 70 >= opponent_hp
        if target_id == LATIAS:
            psychic_after = sum(value == PSYCHIC_ENERGY for value in energies_after)
            return psychic_after >= 2 and total_after >= 3 and opponent_hp <= 200
        return False

    def _crispin_second_energy_score(
        self, card_id: int | None, snapshot: PolicySnapshot
    ) -> float:
        if card_id == WATER_ENERGY and (
            snapshot.field_count(WELLSPRING_OGERPON) > 0 or WELLSPRING_OGERPON in snapshot.hand_ids
        ):
            return 1000.0
        return self._energy_value(card_id)

    def _energy_release_score(
        self,
        observation: dict[str, Any],
        snapshot: PolicySnapshot,
        option: dict[str, Any],
        *,
        moving: bool,
    ) -> tuple[int, ...]:
        energy = self._option_card(observation, snapshot, option)
        parent = self._option_parent_card(observation, snapshot, option)
        energy_id = _card_id(energy)
        parent_id = _card_id(parent)
        if moving:
            incomplete_teal = any(
                _card_id(card) == TEAL_OGERPON and _grass_count(card) < 3
                for card in snapshot.self_field
            )
            wanted_move = 1 if energy_id == GRASS_ENERGY and incomplete_teal else 0
            return (wanted_move, *self._inverse_importance_key(parent, snapshot))

        marginal = {
            GRASS_ENERGY: 95,
            WATER_ENERGY: 80,
            PSYCHIC_ENERGY: 60,
            LIGHTNING_ENERGY: 58,
            FIGHTING_ENERGY: 58,
        }.get(energy_id, 40)
        if parent_id == TEAL_OGERPON and energy_id == GRASS_ENERGY and _grass_count(parent) <= 3:
            marginal = 120
        if parent_id == WELLSPRING_OGERPON and (
            energy_id == WATER_ENERGY or _energy_count(parent) <= 3
        ):
            marginal = max(marginal, 94 if energy_id == WATER_ENERGY else 90)
        if parent_id == RAGING_BOLT and energy_id in {LIGHTNING_ENERGY, FIGHTING_ENERGY}:
            marginal = 110
        parent_inverse = self._inverse_importance_key(parent, snapshot)
        return (-marginal, *parent_inverse)

    # Battle evaluation ----------------------------------------------------
    def _attack_score(self, attack_id: int, snapshot: PolicySnapshot) -> float:
        if attack_id == ATTACK_RAGING_DRAW:
            return (
                7500.0
                if self._voluntary_draw_allowed(snapshot, DRAW_COUNTS[RAGING_BOLT])
                and len(snapshot.hand_ids) <= 2
                else -10000.0
            )
        damage = self._attack_damage(attack_id, snapshot)
        opponent_hp = _as_int((snapshot.opponent_active or {}).get("hp"), 9999)
        knockout = damage >= opponent_hp
        score = 1000.0 + damage
        if knockout:
            score += 5000.0
        if attack_id == ATTACK_WELLSPRING_PUMP and snapshot.opponent_bench:
            score += max(_energy_count(card) for card in snapshot.opponent_bench) * 150.0
        if attack_id == ATTACK_RAGING_DAMAGE:
            score -= 50.0 * sum(_energy_count(card) for card in snapshot.self_field)
        return score

    def _attack_damage(self, attack_id: int, snapshot: PolicySnapshot) -> int:
        self_energy = _energy_count(snapshot.self_active)
        opponent_energy = _energy_count(snapshot.opponent_active)
        if attack_id == ATTACK_TEAL:
            return 30 + 30 * (self_energy + opponent_energy)
        if attack_id == ATTACK_RAGING_DAMAGE:
            return 70 * sum(_energy_count(card) for card in snapshot.self_field)
        if attack_id == ATTACK_CLEFAIRY:
            return 20 + 20 * (len(snapshot.self_bench) + len(snapshot.opponent_bench))
        if attack_id == ATTACK_PASSIMIAN:
            return 20 * len(snapshot.self_field)
        return {
            ATTACK_IRON_LEAVES: 180,
            ATTACK_WELLSPRING_SOB: 20,
            ATTACK_WELLSPRING_PUMP: 100,
            ATTACK_FEZANDIPITI: 100,
            ATTACK_LATIAS: 200,
            ATTACK_CHIEN_PAO: 120,
            ATTACK_KANGASKHAN: 200,
            ATTACK_MEOWTH: 60,
        }.get(attack_id, 0)

    def _combat_value(self, card: dict[str, Any], snapshot: PolicySnapshot) -> float:
        card_id = _card_id(card)
        hp = _as_int(card.get("hp"), HP.get(card_id or -1, 0))
        energies = _energy_count(card)
        value = hp + energies * 80.0
        if card_id == TEAL_OGERPON:
            value += 300.0 + 30.0 * (_energy_count(card) + _energy_count(snapshot.opponent_active))
        elif card_id == WELLSPRING_OGERPON and _water_count(card) > 0 and energies >= 3:
            value += 500.0
        elif card_id == IRON_LEAVES and _grass_count(card) >= 2 and energies >= 3:
            value += 400.0
        elif card_id == RAGING_BOLT:
            value += 70.0 * sum(_energy_count(item) for item in snapshot.self_field)
        elif card_id == MEGA_KANGASKHAN:
            value -= 150.0  # Three-Prize liability.
        return value

    def _teal_can_attack(self, snapshot: PolicySnapshot) -> bool:
        return any(_card_id(card) == TEAL_OGERPON and _grass_count(card) >= 3 for card in snapshot.self_field)

    def _boss_target(self, snapshot: PolicySnapshot) -> dict[str, Any] | None:
        candidates = [card for card in snapshot.opponent_bench if _energy_count(card) > 2]
        return max(candidates, key=_energy_count) if candidates else None

    def _iron_leaves_immediate_knockout(self, snapshot: PolicySnapshot) -> bool:
        opponent_hp = _as_int((snapshot.opponent_active or {}).get("hp"), 9999)
        grass = sum(_grass_count(card) for card in snapshot.self_field)
        total = sum(_energy_count(card) for card in snapshot.self_field)
        return opponent_hp <= 180 and grass >= 2 and total >= 3

    def _latias_should_tank(self, snapshot: PolicySnapshot) -> bool:
        if snapshot.self_active is None or _card_id(snapshot.self_active) != LATIAS:
            return False
        latias_hp = _as_int(snapshot.self_active.get("hp"), 0)
        # Exact opponent dynamic damage is card-specific.  Attached Energy is a
        # conservative public threat proxy; only keep Latias active when every
        # alternative has less remaining HP under that threat.
        threat = 30 + 70 * _energy_count(snapshot.opponent_active)
        alternatives_survive = any(_as_int(card.get("hp"), 0) > threat for card in snapshot.self_bench)
        return latias_hp > threat and not alternatives_survive

    # Resource policy ------------------------------------------------------
    @staticmethod
    def _deck_search_allowed(snapshot: PolicySnapshot) -> bool:
        return snapshot.deck_count >= 10

    @staticmethod
    def _voluntary_draw_allowed(snapshot: PolicySnapshot, draw_count: int) -> bool:
        if draw_count <= 0:
            return True
        if snapshot.deck_count <= 6:
            return False
        if snapshot.deck_count <= 10:
            return draw_count == 1
        return True

    def _formation_phase(self, snapshot: PolicySnapshot) -> str:
        """Return the first unmet item in the user's explicit setup order."""

        teal_available = snapshot.available_count(TEAL_OGERPON)
        if teal_available < 1:
            return "first_teal"

        hand_grass = snapshot.hand_ids.count(GRASS_ENERGY)
        primary = self._primary_teal(snapshot)
        primary_grass = _grass_count(primary)
        if primary_grass + hand_grass < 3:
            return "primary_grass"
        hand_energy = sum(card_id in ENERGY_IDS for card_id in snapshot.hand_ids)
        if (
            primary is not None
            and _energy_count(primary) + hand_energy
            < self._teal_required_energy_count(primary, snapshot)
        ):
            return "primary_power"

        if snapshot.available_count(LATIAS) < 1:
            return "latias"
        if teal_available < 2:
            return "second_teal"

        secondary = next(
            (
                card
                for card in snapshot.self_field
                if _card_id(card) == TEAL_OGERPON
                and _serial(card) == self.secondary_teal_serial
            ),
            None,
        )
        if _grass_count(secondary) + hand_grass < 3:
            return "second_grass"
        if self._high_hp_slots_missing(snapshot) > 0:
            return "bodies"
        return "complete"

    def _high_hp_slots_missing(self, snapshot: PolicySnapshot) -> int:
        occupied = sum(
            _card_id(card) not in {LATIAS, TEAL_OGERPON, WELLSPRING_OGERPON}
            and _as_int(card.get("maxHp"), HP.get(_card_id(card) or -1, 0)) >= 210
            for card in snapshot.self_field
        )
        return max(0, 2 - occupied)

    def _is_protected_latias(
        self, card: dict[str, Any] | None, snapshot: PolicySnapshot
    ) -> bool:
        candidates = [
            item
            for item in (*snapshot.self_field, *snapshot.self_hand)
            if _card_id(item) == LATIAS
        ]
        if not candidates:
            # A deck-search candidate is the needed copy when none is owned.
            return True
        protected = max(
            candidates,
            key=lambda item: (
                _as_int(item.get("hp"), HP[LATIAS]),
                -max(0, _serial(item)),
            ),
        )
        card_serial = _serial(card)
        return card is protected or (
            card_serial >= 0 and card_serial == _serial(protected)
        )

    def _needs_energy(self, snapshot: PolicySnapshot) -> bool:
        for card in snapshot.self_field:
            if _card_id(card) == TEAL_OGERPON and (
                _grass_count(card) < 3
                or (
                    self._teal_role(card) == 1
                    and self._teal_needs_damage_energy(card, snapshot)
                )
            ):
                return True
            if _card_id(card) == WELLSPRING_OGERPON and (
                _water_count(card) == 0 or _energy_count(card) < 3
            ):
                return True
        return snapshot.field_count(TEAL_OGERPON) > 0

    def _energy_switch_is_useful(self, snapshot: PolicySnapshot) -> bool:
        donors = [card for card in snapshot.self_field if _energy_count(card) > 0]
        targets = [
            card
            for card in snapshot.self_field
            if _card_id(card) in {TEAL_OGERPON, WELLSPRING_OGERPON}
            and (
                _grass_count(card) < 3
                or _energy_count(card) < 3
                or (
                    _card_id(card) == TEAL_OGERPON
                    and self._teal_role(card) == 1
                    and self._teal_needs_damage_energy(card, snapshot)
                )
            )
        ]
        return bool(donors and targets and any(donor not in targets for donor in donors))

    def _energy_value(self, card_id: int | None) -> float:
        return {
            GRASS_ENERGY: 500.0,
            WATER_ENERGY: 400.0,
            PSYCHIC_ENERGY: 250.0,
            LIGHTNING_ENERGY: 225.0,
            FIGHTING_ENERGY: 225.0,
        }.get(card_id, 100.0)

    def _choose_energy_discard(
        self,
        observation: dict[str, Any],
        snapshot: PolicySnapshot,
        options: list[dict[str, Any]],
        minimum: int,
        maximum: int,
    ) -> list[int]:
        opponent_hp = _as_int((snapshot.opponent_active or {}).get("hp"), 0)
        needed = max(minimum, min(maximum, math.ceil(opponent_hp / 70.0)))
        return self._choose_ranked(
            observation,
            snapshot,
            options,
            needed,
            needed,
            lambda option: self._energy_release_score(
                observation, snapshot, option, moving=False
            ),
        )

    # Observation resolution ---------------------------------------------
    def _hand_option_card_id(
        self,
        observation: dict[str, Any],
        snapshot: PolicySnapshot,
        option: dict[str, Any],
    ) -> int | None:
        return _card_id(self._zone_card(observation, snapshot, AREA_HAND, _as_int(option.get("index"), -1), snapshot.player_index))

    def _field_option_card(
        self, snapshot: PolicySnapshot, option: dict[str, Any]
    ) -> dict[str, Any] | None:
        area = _as_int(option.get("area"), -1)
        index = _as_int(option.get("index"), -1)
        return self._field_card(snapshot, area, index, opponent=False)

    def _in_play_target(
        self, snapshot: PolicySnapshot, option: dict[str, Any]
    ) -> dict[str, Any] | None:
        return self._field_card(
            snapshot,
            _as_int(option.get("inPlayArea"), -1),
            _as_int(option.get("inPlayIndex"), -1),
            opponent=False,
        )

    def _option_card_id(
        self,
        observation: dict[str, Any],
        snapshot: PolicySnapshot,
        option: dict[str, Any],
    ) -> int | None:
        direct = _as_int(option.get("cardId"), -1)
        if direct >= 0:
            return direct
        return _card_id(self._option_card(observation, snapshot, option))

    def _option_card(
        self,
        observation: dict[str, Any],
        snapshot: PolicySnapshot,
        option: dict[str, Any],
    ) -> dict[str, Any] | None:
        area = _as_int(option.get("area"), -1)
        index = _as_int(option.get("index"), -1)
        absolute_player = _as_int(option.get("playerIndex"), snapshot.player_index)
        parent = self._zone_card(observation, snapshot, area, index, absolute_player)
        if _as_int(option.get("type"), -1) in {OPTION_ENERGY_CARD, OPTION_ENERGY}:
            energy_index = _as_int(option.get("energyIndex"), -1)
            energy_cards = (parent or {}).get("energyCards") or []
            if 0 <= energy_index < len(energy_cards) and isinstance(energy_cards[energy_index], dict):
                return energy_cards[energy_index]
        return parent

    def _option_parent_card(
        self,
        observation: dict[str, Any],
        snapshot: PolicySnapshot,
        option: dict[str, Any],
    ) -> dict[str, Any] | None:
        return self._zone_card(
            observation,
            snapshot,
            _as_int(option.get("area"), -1),
            _as_int(option.get("index"), -1),
            _as_int(option.get("playerIndex"), snapshot.player_index),
        )

    def _zone_card(
        self,
        observation: dict[str, Any],
        snapshot: PolicySnapshot,
        area: int,
        index: int,
        absolute_player: int,
    ) -> dict[str, Any] | None:
        select = observation.get("select") or {}
        if area == AREA_DECK:
            cards = select.get("deck") or []
            if 0 <= index < len(cards) and isinstance(cards[index], dict):
                return cards[index]
            # Search options use original deck indices while ``select.deck``
            # can be compact. Match option position as a final fallback.
            for option_index, option in enumerate(select.get("option") or []):
                if _as_int(option.get("area"), -1) == area and _as_int(option.get("index"), -1) == index:
                    if 0 <= option_index < len(cards) and isinstance(cards[option_index], dict):
                        return cards[option_index]
            return None
        current = observation.get("current") or {}
        players = current.get("players") or [{}, {}]
        if not 0 <= absolute_player < len(players):
            return None
        player = players[absolute_player]
        zone_name = {
            AREA_HAND: "hand",
            AREA_DISCARD: "discard",
            AREA_ACTIVE: "active",
            AREA_BENCH: "bench",
        }.get(area)
        if zone_name is None:
            return None
        cards = player.get(zone_name) or []
        return cards[index] if 0 <= index < len(cards) and isinstance(cards[index], dict) else None

    def _field_card(
        self, snapshot: PolicySnapshot, area: int, index: int, *, opponent: bool
    ) -> dict[str, Any] | None:
        active = snapshot.opponent_active if opponent else snapshot.self_active
        bench = snapshot.opponent_bench if opponent else snapshot.self_bench
        if area == AREA_ACTIVE:
            return active if index in (0, -1) else None
        if area == AREA_BENCH and 0 <= index < len(bench):
            return bench[index]
        return None

    # Generic legal selection --------------------------------------------
    def _choose_ranked(
        self,
        observation: dict[str, Any],
        snapshot: PolicySnapshot,
        options: list[dict[str, Any]],
        minimum: int,
        maximum: int,
        scorer: Any,
    ) -> list[int]:
        del observation, snapshot
        ranked = sorted(range(len(options)), key=lambda index: (scorer(options[index]), -index), reverse=True)
        count = min(maximum, len(ranked))
        if minimum == 0:
            def is_positive(value: Any) -> bool:
                return value[0] > 0 if isinstance(value, tuple) and value else value > 0

            positive = [index for index in ranked if is_positive(scorer(options[index]))]
            count = min(count, len(positive))
            ranked = positive
        count = max(minimum, count)
        return ranked[: min(count, len(ranked))]

    @staticmethod
    def _best_by_score(options: list[dict[str, Any]], scorer: Any) -> int:
        return max(range(len(options)), key=lambda index: (scorer(options[index]), -index))

    def _missing_core(self, snapshot: PolicySnapshot) -> bool:
        return snapshot.available_count(TEAL_OGERPON) < 2 or snapshot.available_count(LATIAS) < 1

    def _needed_supporter_id(self, snapshot: PolicySnapshot) -> int | None:
        if self._boss_target(snapshot) is not None and self._teal_can_attack(snapshot):
            return BOSSES_ORDERS
        phase = self._formation_phase(snapshot)
        if phase in {"primary_grass", "primary_power", "second_grass"}:
            return CRISPIN
        if phase in {"first_teal", "latias", "second_teal", "bodies"}:
            return CYRANO
        draw_target = 8 if snapshot.prize_count == 6 else 6
        if (
            self._voluntary_draw_allowed(snapshot, draw_target)
            and len(snapshot.hand_ids) <= draw_target
        ):
            return LILLIES_DETERMINATION
        return None

    def _has_colorless_bench(self, snapshot: PolicySnapshot) -> bool:
        return any(_card_id(card) in {MEGA_KANGASKHAN, MEOWTH} for card in snapshot.self_bench)

    def _should_remove_stadium(self, snapshot: PolicySnapshot) -> bool:
        return (
            snapshot.stadium is not None
            and _card_id(snapshot.stadium) != AREA_ZERO
            and len(snapshot.self_bench) <= 5
        )

def is_legal_action(observation: dict[str, Any], action: object) -> bool:
    """Cheap shape gate used by smoke tests and future submission wrapping."""

    if not isinstance(action, list) or not all(type(index) is int for index in action):
        return False
    select = observation.get("select") or {}
    options = select.get("option") or []
    minimum = max(0, _as_int(select.get("minCount"), 0))
    maximum = max(minimum, _as_int(select.get("maxCount"), minimum))
    return (
        minimum <= len(action) <= maximum
        and len(action) == len(set(action))
        and all(0 <= index < len(options) for index in action)
    )
