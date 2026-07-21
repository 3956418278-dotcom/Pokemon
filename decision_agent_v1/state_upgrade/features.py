from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

from data.game_memory import GameMemoryState

from decision_agent_v1.adapters.card_vocab_adapter import CardVocabulary
from decision_agent_v1.contracts.schemas import DecisionSampleV1

from .deck_prior import DeckPrior, normalized_entropy


RECENT_EVENT_COUNT = 16
MAX_LEDGER_CARDS = 32
MAX_SELF_DECK_CARDS = 32
MAX_BELIEF_CARDS = 32
BELIEF_TOP_K = 8
LEDGER_SUMMARY_DIM = 92
LEDGER_CARD_NUMERIC_DIM = 20
SELF_DECK_SUMMARY_DIM = 11


@dataclass(frozen=True)
class StateUpgradeFeatures:
    ledger_summary: tuple[float, ...]
    ledger_card_index: tuple[int, ...]
    ledger_card_numeric: tuple[tuple[float, ...], ...]
    recent_event_type: tuple[int, ...]
    recent_event_player: tuple[int, ...]
    recent_event_card_index: tuple[int, ...]
    recent_event_source: tuple[int, ...]
    recent_event_target: tuple[int, ...]
    recent_event_numeric: tuple[tuple[float, ...], ...]
    self_deck_summary: tuple[float, ...]
    self_deck_card_index: tuple[int, ...]
    self_deck_card_numeric: tuple[tuple[float, ...], ...]
    belief_summary: tuple[float, ...]
    belief_template_index: tuple[int, ...]
    belief_template_probability: tuple[float, ...]
    belief_card_index: tuple[int, ...]
    belief_card_expected: tuple[float, ...]
    archetype_target: int
    next_public_target: int
    next_public_mask: bool


def _self_known_outside_hidden(memory: GameMemoryState, your_index: int) -> Counter[int]:
    result: Counter[int] = Counter()
    for item in memory.serials.values():
        if item.player_index == your_index and item.card_id is not None and item.current_area not in {1, 6, None}:
            result[int(item.card_id)] += 1
    return result


def _opponent_public_counts(memory: GameMemoryState, your_index: int) -> Counter[int]:
    result: Counter[int] = Counter()
    opponent = 1 - your_index
    for item in memory.serials.values():
        if item.player_index == opponent and item.card_id is not None:
            result[int(item.card_id)] += 1
    return result


def _ledger_summary(
    memory: GameMemoryState, sample: DecisionSampleV1
) -> tuple[float, ...]:
    """Sanitized summary derived from the one authoritative GameMemory.

    A serial's last public zone is historical evidence, not a claim about its
    current hidden position. Exact-zone counts therefore include only serials
    still visible in the current observation.
    """

    rows: list[float] = []
    your_index = sample.agent_index
    for relative_player in (0, 1):
        player_index = your_index if relative_player == 0 else 1 - your_index
        memories = [row for row in memory.serials.values() if row.player_index == player_index]
        visible = [row for row in memories if row.currently_visible]
        public = memory.public_counts_by_player.get(player_index, {})
        rows.extend(
            (
                len(memories) / 60.0,
                sum(row.played for row in memories) / 60.0,
                sum(row.attached for row in memories) / 60.0,
                sum(row.evolved for row in memories) / 20.0,
                sum(row.attacked for row in memories) / 20.0,
                sum(row.damaged for row in memories) / 20.0,
                sum(row.discarded_public for row in memories) / 60.0,
                sum(row.current_area == 2 for row in visible) / 20.0,
                sum(row.current_area == 3 for row in visible) / 60.0,
                sum(row.current_area in {4, 5} for row in visible) / 6.0,
                sum(row.current_area in {8, 9, 10} for row in visible) / 20.0,
                sum(row.current_area == 7 for row in visible),
                min(sum(row.seen_count for row in memories), 100) / 100.0,
                min(sum(row.moved_count for row in memories), 100) / 100.0,
                public.get("deck", 0) / 60.0,
                public.get("hand", 0) / 20.0,
                public.get("prize", 0) / 6.0,
                public.get("bench", 0) / 5.0,
                public.get("active", 0),
                float(relative_player),
            )
        )
    rows.extend(memory.cumulative_public_event_counts.get(index, 0) / 100.0 for index in range(24))
    rows.extend(memory.current_turn_public_event_counts.get(index, 0) / 20.0 for index in range(24))
    rows.extend(
        (
            float(sample.global_state.supporter_played),
            float(sample.global_state.stadium_played),
            float(sample.global_state.retreated),
            float(sample.global_state.energy_attached),
        )
    )
    if len(rows) != LEDGER_SUMMARY_DIM:
        raise AssertionError(f"ledger summary width changed: {len(rows)}")
    return tuple(rows)


def _ledger_card_numeric(
    memory: GameMemoryState,
    your_index: int,
    owner_relative: int,
    card_id: int,
    known_serial_count: int,
    ambiguous_serial_count: int,
    visible_observation_count: int,
    movement_event_count: int,
    first_known_turn: int | None,
    last_known_turn: int | None,
) -> tuple[float, ...]:
    player_index = your_index if owner_relative == 0 else 1 - your_index
    visible = [
        row for row in memory.serials.values()
        if row.player_index == player_index
        and row.card_id == card_id
        and row.currently_visible
    ]
    zones = [sum(row.current_area == area for row in visible) / 4.0 for area in range(1, 13)]
    result = (
        float(owner_relative),
        known_serial_count / 4.0,
        ambiguous_serial_count / 4.0,
        visible_observation_count / 32.0,
        movement_event_count / 16.0,
        (first_known_turn or 0) / 100.0,
        (last_known_turn or 0) / 100.0,
        *zones,
        float(bool(visible)),
    )
    if len(result) != LEDGER_CARD_NUMERIC_DIM:
        raise AssertionError("ledger Card ID token width changed")
    return result


def build_state_upgrade_features(
    sample: DecisionSampleV1,
    memory: GameMemoryState,
    self_deck: list[int],
    opponent_deck: list[int],
    prior: DeckPrior,
    vocabulary: CardVocabulary,
    *,
    next_public_card_id: int | None,
    recent_event_decision_ages: tuple[int, ...] | None = None,
) -> StateUpgradeFeatures:
    your_index = sample.agent_index
    records = sorted(
        memory.card_id_memory_records(your_index),
        key=lambda row: (-row.known_serial_count, row.owner_relative, row.card_id),
    )[:MAX_LEDGER_CARDS]
    events = memory.recent_events[-RECENT_EVENT_COUNT:]
    decision_ages = recent_event_decision_ages or tuple(row.observation_age for row in events)
    if len(decision_ages) != len(events):
        raise ValueError("recent event decision ages must align with event tokens")
    initial = Counter(int(card_id) for card_id in self_deck)
    known_out = _self_known_outside_hidden(memory, your_index)
    remaining = Counter({card_id: max(count - known_out[card_id], 0) for card_id, count in initial.items()})
    self_ids = sorted(initial, key=lambda card_id: (-remaining[card_id], card_id))[:MAX_SELF_DECK_CARDS]
    kind_by_card = dict(zip(prior.card_ids, prior.card_kinds))
    remaining_by_kind = {"POKEMON": 0, "TRAINER": 0, "ENERGY": 0}
    for card_id, count in remaining.items():
        kind = kind_by_card.get(card_id, "TRAINER")
        remaining_by_kind[kind] = remaining_by_kind.get(kind, 0) + count
    hidden = memory.anonymous_hidden_pools_record(your_index)
    public = _opponent_public_counts(memory, your_index)
    posterior = prior.posterior(dict(public))
    expected = prior.expected_remaining(dict(public), posterior)
    top_templates = np.argsort(-posterior)[:BELIEF_TOP_K]
    top_cards = [int(index) for index in np.argsort(-expected)[:MAX_BELIEF_CARDS] if expected[index] > 0]
    pokemon, trainer, energy = prior.category_totals(expected)
    opponent_hidden = hidden.opponent_unknown_hand_count + hidden.opponent_unknown_deck_count + hidden.opponent_unknown_prize_count
    return StateUpgradeFeatures(
        ledger_summary=_ledger_summary(memory, sample),
        ledger_card_index=tuple(vocabulary.encode(row.card_id) for row in records),
        ledger_card_numeric=tuple(
            _ledger_card_numeric(
                memory, your_index, row.owner_relative, row.card_id,
                row.known_serial_count, row.ambiguous_serial_count,
                row.visible_observation_count, row.movement_event_count,
                row.first_known_turn, row.last_known_turn,
            )
            for row in records
        ),
        recent_event_type=tuple(int(row.event_type) + 1 for row in events),
        recent_event_player=tuple(int(row.actor_relative) + 1 if row.actor_relative in (0, 1) else 0 for row in events),
        recent_event_card_index=tuple(vocabulary.encode(row.card_id if row.identity_visible else None) for row in events),
        recent_event_source=tuple(int(row.from_area or 0) for row in events),
        recent_event_target=tuple(int(row.to_area or 0) for row in events),
        recent_event_numeric=tuple((min(row.turn_age, 32) / 32.0, min(decision_age, 64) / 64.0, min(row.batch_position, 32) / 32.0, min(max(row.observed_at_turn_action_count, 0), 32) / 32.0, float(row.identity_visible), float(row.is_reverse)) for row, decision_age in zip(events, decision_ages)),
        self_deck_summary=(len(self_deck) / 60.0, sum(known_out.values()) / 60.0, sum(remaining.values()) / 60.0, hidden.self_unknown_deck_count / 60.0, hidden.self_unknown_prize_count / 6.0, len(initial) / 32.0, len(remaining) / 32.0, remaining_by_kind["POKEMON"] / 60.0, remaining_by_kind["TRAINER"] / 60.0, remaining_by_kind["ENERGY"] / 60.0, float(sum(remaining.values()) == len(self_deck) - sum(known_out.values()))),
        self_deck_card_index=tuple(vocabulary.encode(card_id) for card_id in self_ids),
        self_deck_card_numeric=tuple((initial[card_id] / 4.0, known_out[card_id] / 4.0, remaining[card_id] / 4.0) for card_id in self_ids),
        belief_summary=(normalized_entropy(posterior), sum(public.values()) / 60.0, opponent_hidden / 60.0, pokemon / 60.0, trainer / 60.0, energy / 60.0, float(posterior[top_templates[0]]) if len(top_templates) else 0.0, len(public) / 32.0),
        belief_template_index=tuple(int(index) + 1 for index in top_templates),
        belief_template_probability=tuple(float(posterior[index]) for index in top_templates),
        belief_card_index=tuple(vocabulary.encode(prior.card_ids[index]) for index in top_cards),
        belief_card_expected=tuple(float(expected[index] / 4.0) for index in top_cards),
        archetype_target=prior.label_for_deck(dict(Counter(int(x) for x in opponent_deck))),
        next_public_target=vocabulary.encode(next_public_card_id) if next_public_card_id is not None else 0,
        next_public_mask=next_public_card_id is not None,
    )


def next_public_cards(rows: list[tuple[DecisionSampleV1, GameMemoryState]]) -> dict[tuple[int, int], int | None]:
    result: dict[tuple[int, int], int | None] = {}
    by_agent: dict[int, list[tuple[DecisionSampleV1, GameMemoryState]]] = {0: [], 1: []}
    for sample, memory in rows:
        by_agent[sample.agent_index].append((sample, memory))
    for agent, stream in by_agent.items():
        stream.sort(key=lambda value: value[0].decision_index)
        public_sets = [set(_opponent_public_counts(memory, agent)) for _, memory in stream]
        for index, (sample, _) in enumerate(stream):
            target = None
            for future in public_sets[index + 1:]:
                novel = sorted(future - public_sets[index])
                if novel:
                    target = novel[0]
                    break
            result[(agent, sample.decision_index)] = target
    return result


def _event_key(row: object) -> tuple[object, ...]:
    return tuple(
        getattr(row, name)
        for name in (
            "observed_at_turn", "observed_at_turn_action_count", "batch_position",
            "event_type", "player_index", "card_id", "serial", "from_area", "to_area",
            "target_card_id", "target_serial", "attack_id", "value", "is_reverse",
        )
    )


def event_decision_ages(
    rows: list[tuple[DecisionSampleV1, GameMemoryState]],
) -> dict[tuple[int, int], tuple[int, ...]]:
    """Exact age in agent decisions for each retained public event token."""

    result: dict[tuple[int, int], tuple[int, ...]] = {}
    by_agent: dict[int, list[tuple[DecisionSampleV1, GameMemoryState]]] = {0: [], 1: []}
    for sample, memory in rows:
        by_agent[sample.agent_index].append((sample, memory))
    for agent, stream in by_agent.items():
        stream.sort(key=lambda value: value[0].decision_index)
        first_decision: dict[tuple[object, ...], int] = {}
        for sample, memory in stream:
            events = memory.recent_events[-RECENT_EVENT_COUNT:]
            for event in events:
                first_decision.setdefault(_event_key(event), sample.decision_index)
            result[(agent, sample.decision_index)] = tuple(
                max(0, sample.decision_index - first_decision[_event_key(event)])
                for event in events
            )
    return result
