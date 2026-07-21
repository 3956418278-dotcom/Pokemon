from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable

from data.legal_options import policy_mask_decision
from data.replay_dataset import ReplayDecisionDataset, ReplayDecisionSample

from decision_agent_v1.contracts.action_contract import ActionSemanticsContract
from decision_agent_v1.contracts.schemas import DecisionSampleV1, SelectionMode, TerminalOutcome

from .observation_adapter import ObservationAdapter


@dataclass(frozen=True)
class ReplayAdapterReport:
    source_samples: int
    adapted_samples: int
    episodes: int
    missing_terminal_groups: int
    outcome_counts: dict[str, int]
    selection_mode_counts: dict[str, int]
    unknown_combinations: dict[str, int]


def _outcome(reward: float) -> TerminalOutcome:
    if reward > 0:
        return TerminalOutcome.WIN
    if reward < 0:
        return TerminalOutcome.LOSS
    return TerminalOutcome.DRAW


def _canonical_option_key(sample: DecisionSampleV1, index: int) -> tuple[object, ...]:
    option = sample.options[index]
    return (
        option.option_type,
        option.relative_player,
        option.area,
        option.position_index,
        option.card_id if option.card_id is not None else -1,
        option.serial if option.serial is not None else -1,
        option.energy_type,
        option.damage_value,
        option.original_option_index,
    )


def adapt_replay_dataset(
    dataset: ReplayDecisionDataset,
    observation_adapter: ObservationAdapter,
    contract: ActionSemanticsContract,
    *,
    max_decisions: int | None = None,
) -> tuple[list[DecisionSampleV1], ReplayAdapterReport]:
    """Adapt the existing dataset without traversing Replay JSON a second time."""

    grouped: dict[tuple[str, int], list[ReplayDecisionSample]] = defaultdict(list)
    for sample in dataset.samples:
        grouped[(sample.replay_key, sample.agent_index)].append(sample)

    outcomes: dict[tuple[str, int], TerminalOutcome] = {}
    for key, rows in grouped.items():
        terminal = [row for row in rows if str(row.status).upper() == "DONE"]
        if terminal:
            outcomes[key] = _outcome(terminal[-1].reward)
    # Some Replay streams end one side with an empty terminal action, which the
    # existing behavior-cloning dataset intentionally does not materialize as a
    # decision row.  Pokémon TCG is two-player and zero-sum, so complete the
    # missing perspective from the audited terminal label on the other stream.
    by_replay: dict[str, dict[int, TerminalOutcome]] = defaultdict(dict)
    for (replay_key, agent_index), outcome in outcomes.items():
        by_replay[replay_key][agent_index] = outcome
    opposite = {
        TerminalOutcome.WIN: TerminalOutcome.LOSS,
        TerminalOutcome.LOSS: TerminalOutcome.WIN,
        TerminalOutcome.DRAW: TerminalOutcome.DRAW,
    }
    for replay_key, agent_outcomes in by_replay.items():
        if len(agent_outcomes) == 1:
            known_agent, known_outcome = next(iter(agent_outcomes.items()))
            other_key = (replay_key, 1 - known_agent)
            if other_key in grouped:
                outcomes[other_key] = opposite[known_outcome]

    result: list[DecisionSampleV1] = []
    unknown = Counter()
    mode_counts = Counter()
    outcome_counts = Counter()
    for key in sorted(grouped):
        if key not in outcomes:
            continue
        rows = sorted(grouped[key], key=lambda row: (row.step_index, row.action_step_index or -1))
        episode_count = len(rows)
        for decision_index, raw in enumerate(rows):
            if max_decisions is not None and len(result) >= max_decisions:
                break
            if raw.action_target is None:
                continue
            cards = observation_adapter.cards(raw.parsed)
            global_state = observation_adapter.global_state(raw.parsed)
            history = observation_adapter.history(raw.parsed, raw.memory_after)
            groups = tuple(int(value) for value in raw.action_target.equivalence_class_ids)
            options = observation_adapter.options(raw.parsed, cards, groups)
            option_types = {option.option_type for option in options}
            mode = contract.mode_for(
                raw.select_type,
                raw.select_context,
                option_types,
                global_state.max_count,
                raw.action_semantics.value,
            )
            policy_applies, mask_reason = policy_mask_decision(
                raw.observation.get("select") or {}, raw.action_target
            )
            provisional = DecisionSampleV1(
                episode_id=raw.replay_key,
                source_date=raw.source_date,
                agent_index=raw.agent_index,
                decision_index=decision_index,
                step=raw.step_index,
                turn=global_state.turn,
                turn_action_count=global_state.turn_action_count,
                cards=cards,
                global_state=global_state,
                history=history,
                options=options,
                selected_option_indices=tuple(int(value) for value in raw.action),
                selected_equivalence_groups=tuple(
                    groups[index] for index in raw.action if 0 <= index < len(groups)
                ),
                min_count=global_state.min_count,
                max_count=global_state.max_count,
                selection_mode=mode,
                terminal_outcome=outcomes[key],
                episode_decision_count=episode_count,
                policy_supervision=bool(policy_applies and mode is not SelectionMode.UNKNOWN),
                policy_mask_reason=(mask_reason if mode is not SelectionMode.UNKNOWN else "UNKNOWN_ACTION_SEMANTICS"),
            )
            if mode is SelectionMode.UNORDERED_UNIQUE_SUBSET:
                canonical = tuple(
                    sorted(provisional.selected_option_indices, key=lambda index: _canonical_option_key(provisional, index))
                )
                provisional = DecisionSampleV1(
                    **{
                        **provisional.__dict__,
                        "selected_option_indices": canonical,
                        "selected_equivalence_groups": tuple(groups[index] for index in canonical),
                    }
                )
            if mode is SelectionMode.UNKNOWN:
                key_name = f"{raw.select_type}:{raw.select_context}:{','.join(map(str, sorted(option_types)))}"
                unknown[key_name] += 1
            result.append(provisional)
            mode_counts[mode.value] += 1
            outcome_counts[outcomes[key].name] += 1
        if max_decisions is not None and len(result) >= max_decisions:
            break
    return result, ReplayAdapterReport(
        source_samples=len(dataset.samples),
        adapted_samples=len(result),
        episodes=len({sample.episode_id for sample in result}),
        missing_terminal_groups=sum(key not in outcomes for key in grouped),
        outcome_counts=dict(outcome_counts),
        selection_mode_counts=dict(mode_counts),
        unknown_combinations=dict(unknown),
    )


def split_complete_episodes(
    samples: Iterable[DecisionSampleV1],
    train_dates: set[str],
    validation_dates: set[str],
    test_dates: set[str],
) -> dict[str, list[DecisionSampleV1]]:
    splits = {"train": [], "validation": [], "test": []}
    episode_split: dict[str, str] = {}
    for sample in samples:
        if sample.source_date in train_dates:
            split = "train"
        elif sample.source_date in validation_dates:
            split = "validation"
        elif sample.source_date in test_dates:
            split = "test"
        else:
            continue
        previous = episode_split.setdefault(sample.episode_id, split)
        if previous != split:
            raise ValueError(f"episode {sample.episode_id} crosses date splits")
        splits[split].append(sample)
    return splits
