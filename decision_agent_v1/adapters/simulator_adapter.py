from __future__ import annotations

from dataclasses import dataclass

from data.game_memory import GameMemoryState
from data.observation_parser import parse_observation

from decision_agent_v1.contracts.action_contract import ActionSemanticsContract
from decision_agent_v1.contracts.schemas import (
    CardInstanceView,
    GlobalStateView,
    HistoryView,
    OptionView,
    SelectionMode,
)

from .observation_adapter import ObservationAdapter


@dataclass(frozen=True)
class LiveDecisionView:
    cards: tuple[CardInstanceView, ...]
    global_state: GlobalStateView
    history: HistoryView
    options: tuple[OptionView, ...]
    selection_mode: SelectionMode


class SimulatorAdapter:
    def __init__(self, observation_adapter: ObservationAdapter, contract: ActionSemanticsContract) -> None:
        self.observation_adapter = observation_adapter
        self.contract = contract
        self.memory_by_agent: dict[int, GameMemoryState] = {}

    def reset(self) -> None:
        self.memory_by_agent.clear()

    def adapt(self, observation: dict[str, object]) -> LiveDecisionView:
        parsed = parse_observation(observation)
        agent_index = parsed.global_snapshot.your_index
        memory = self.memory_by_agent.setdefault(agent_index, GameMemoryState())
        memory.update_from_parsed(parsed)
        cards = self.observation_adapter.cards(parsed)
        groups = tuple(range(len(parsed.select_options)))
        options = self.observation_adapter.options(parsed, cards, groups)
        option_types = {option.option_type for option in options}
        mode = self.contract.mode_for(
            parsed.global_snapshot.select_type,
            parsed.global_snapshot.select_context,
            option_types,
            parsed.global_snapshot.select_max_count,
        )
        return LiveDecisionView(
            cards=cards,
            global_state=self.observation_adapter.global_state(parsed),
            history=self.observation_adapter.history(parsed, memory),
            options=options,
            selection_mode=mode,
        )

    @staticmethod
    def restore_original_indices(view: LiveDecisionView, positions: list[int]) -> list[int]:
        return [view.options[position].original_option_index for position in positions]
