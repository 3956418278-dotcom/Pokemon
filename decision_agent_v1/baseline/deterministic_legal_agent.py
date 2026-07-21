from __future__ import annotations

from collections import Counter

from decision_agent_v1.adapters.simulator_adapter import LiveDecisionView
from decision_agent_v1.contracts.schemas import OptionView, SelectionMode


def _stable_key(option: OptionView) -> tuple[object, ...]:
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


class DeterministicLegalAgent:
    """A tactics-free adapter/contract baseline that only returns legal indices."""

    def __init__(self) -> None:
        self.statistics: Counter[str] = Counter()

    def select_positions(self, view: LiveDecisionView) -> list[int]:
        if not view.options:
            return []
        ordered = sorted(range(len(view.options)), key=lambda index: _stable_key(view.options[index]))
        if view.selection_mode is SelectionMode.SINGLE:
            count = 1
        else:
            count = max(0, view.global_state.min_count)
        count = min(count, max(0, view.global_state.max_count), len(ordered))
        if view.selection_mode is SelectionMode.UNKNOWN:
            self.statistics[
                f"unknown:{view.global_state.select_type}:{view.global_state.select_context}"
            ] += 1
            ordered = list(range(len(view.options)))
        selected = ordered[:count]
        if view.selection_mode is SelectionMode.UNORDERED_UNIQUE_SUBSET:
            selected.sort(key=lambda index: _stable_key(view.options[index]))
        if len(set(selected)) != len(selected):
            raise AssertionError("deterministic legal baseline selected a duplicate option")
        if not (view.global_state.min_count <= len(selected) <= view.global_state.max_count):
            raise AssertionError(
                "deterministic legal baseline violated minCount/maxCount: "
                f"type={view.global_state.select_type}, "
                f"context={view.global_state.select_context}, "
                f"mode={view.selection_mode.value}, "
                f"min={view.global_state.min_count}, max={view.global_state.max_count}, "
                f"options={len(view.options)}, selected={len(selected)}"
            )
        return selected

    def act(self, view: LiveDecisionView) -> list[int]:
        return [view.options[index].original_option_index for index in self.select_positions(view)]
