from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from torch.utils.data import Dataset

from decision_agent_v1.contracts.schemas import DecisionSampleV1, SelectionMode


class DecisionDatasetV1(Dataset[DecisionSampleV1]):
    def __init__(
        self,
        samples: Sequence[DecisionSampleV1],
        *,
        policy_modes: set[SelectionMode] | None = None,
        max_episodes: int | None = None,
        max_decisions: int | None = None,
    ) -> None:
        allowed_episodes: set[str] = set()
        selected = []
        for sample in samples:
            if sample.episode_id not in allowed_episodes:
                if max_episodes is not None and len(allowed_episodes) >= max_episodes:
                    continue
                allowed_episodes.add(sample.episode_id)
            if policy_modes is not None and sample.selection_mode not in policy_modes:
                sample = DecisionSampleV1(
                    **{
                        **sample.__dict__,
                        "policy_supervision": False,
                        "policy_mask_reason": "FILTERED_SELECTION_MODE",
                    }
                )
            selected.append(sample)
            if max_decisions is not None and len(selected) >= max_decisions:
                break
        self.samples = selected

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> DecisionSampleV1:
        return self.samples[index]

    def statistics(self) -> dict[str, dict[str, int]]:
        return {
            "selection_mode": dict(Counter(item.selection_mode.value for item in self.samples)),
            "select_context": dict(Counter(str(item.global_state.select_context) for item in self.samples)),
            "terminal_outcome": dict(Counter(item.terminal_outcome.name for item in self.samples)),
        }
