from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .game_memory import GameMemoryState
from .observation_parser import parse_observation
from .state_schema import ParsedObservation


@dataclass
class ReplayDecisionSample:
    replay_id: str | None
    episode_id: int | None
    step_index: int
    agent_index: int
    observation: dict[str, Any]
    action: list[int]
    reward: float
    status: str | None
    parsed: ParsedObservation
    memory_before: GameMemoryState
    memory_after: GameMemoryState
    option_count: int
    legal_option_types: list[int]
    select_type: int
    select_context: int
    done: bool = False


@dataclass
class ReplayDatasetSummary:
    replay_count: int = 0
    sample_count: int = 0
    skipped_no_select: int = 0
    parser_errors: list[dict[str, Any]] = field(default_factory=list)
    max_instances: int = 0
    max_options: int = 0
    max_events: int = 0
    max_token_estimate: int = 0


def _int(value: Any, default: int = -1) -> int:
    if value is None:
        return default
    if hasattr(value, "value"):
        return int(value.value)
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _action_list(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, list):
        return [_int(item, 0) for item in value]
    return [_int(value, 0)]


def _option_type(option: dict[str, Any]) -> int:
    return _int(option.get("type"), -1)


def iter_replay_paths(paths: Iterable[Path]) -> list[Path]:
    replay_paths: list[Path] = []
    for path in paths:
        if path.is_dir():
            replay_paths.extend(sorted(path.rglob("*replay.json")))
            replay_paths.extend(sorted(path.rglob("*.json")))
            replay_paths.extend(sorted(path.rglob("*.jsonl")))
        elif path.suffix.lower() in {".json", ".jsonl"}:
            replay_paths.append(path)
    return sorted(dict.fromkeys(replay_paths))


class ReplayDecisionDataset:
    """Decision-point dataset backed by Kaggle replay JSON observations.

    Each sample is one agent perspective at one step. Game length is intentionally
    not fixed; batching should pad/mask per sample after encoding.
    """

    def __init__(
        self,
        replay_paths: list[Path],
        include_no_select: bool = False,
        controlled_agents: set[int] | None = None,
        max_samples: int | None = None,
    ) -> None:
        self.replay_paths = replay_paths
        self.include_no_select = include_no_select
        self.controlled_agents = controlled_agents
        self.samples: list[ReplayDecisionSample] = []
        self.summary = ReplayDatasetSummary(replay_count=len(replay_paths))
        self._load(max_samples=max_samples)

    @classmethod
    def from_paths(
        cls,
        paths: Iterable[str | Path],
        include_no_select: bool = False,
        controlled_agents: set[int] | None = None,
        max_samples: int | None = None,
    ) -> "ReplayDecisionDataset":
        replay_paths = iter_replay_paths([Path(path) for path in paths])
        return cls(
            replay_paths,
            include_no_select=include_no_select,
            controlled_agents=controlled_agents,
            max_samples=max_samples,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> ReplayDecisionSample:
        return self.samples[index]

    def _load(self, max_samples: int | None) -> None:
        for replay_path in self.replay_paths:
            if replay_path.suffix.lower() == ".jsonl":
                for replay in self._load_jsonl(replay_path):
                    self._append_replay(replay, replay_path, max_samples)
                    if max_samples is not None and len(self.samples) >= max_samples:
                        return
            else:
                replay = json.loads(replay_path.read_text(encoding="utf-8"))
                self._append_replay(replay, replay_path, max_samples)
                if max_samples is not None and len(self.samples) >= max_samples:
                    return
        self.summary.sample_count = len(self.samples)

    def _load_jsonl(self, replay_path: Path) -> Iterable[dict[str, Any]]:
        for line in replay_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                yield json.loads(line)

    def _append_replay(self, replay: dict[str, Any], replay_path: Path, max_samples: int | None) -> None:
        memories: dict[int, GameMemoryState] = {}
        replay_id = replay.get("id")
        episode_id = replay.get("info", {}).get("EpisodeId")
        for step_index, step in enumerate(replay.get("steps", []) or []):
            for agent_index, agent_step in enumerate(step or []):
                if self.controlled_agents is not None and agent_index not in self.controlled_agents:
                    continue
                observation = (agent_step or {}).get("observation")
                if observation is None:
                    continue
                if observation.get("select") is None and not self.include_no_select:
                    self.summary.skipped_no_select += 1
                    continue
                try:
                    parsed = parse_observation(observation)
                except Exception as exc:
                    self.summary.parser_errors.append(
                        {
                            "replay": str(replay_path),
                            "step": step_index,
                            "agent": agent_index,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    continue
                memory = memories.setdefault(agent_index, GameMemoryState())
                memory_before = copy.deepcopy(memory)
                memory.update_from_parsed(parsed)
                memory_after = copy.deepcopy(memory)
                option_types = [_option_type(option) for option in parsed.select_options]
                token_estimate = (
                    1
                    + len(parsed.card_instances)
                    + 1
                    + 1
                    + 2
                    + min(len(memory_after.recent_events), memory_after.max_recent_events)
                )
                self.summary.max_instances = max(self.summary.max_instances, len(parsed.card_instances))
                self.summary.max_options = max(self.summary.max_options, len(parsed.select_options))
                self.summary.max_events = max(self.summary.max_events, len(parsed.events))
                self.summary.max_token_estimate = max(self.summary.max_token_estimate, token_estimate)
                self.samples.append(
                    ReplayDecisionSample(
                        replay_id=str(replay_id) if replay_id is not None else None,
                        episode_id=_int(episode_id, -1) if episode_id is not None else None,
                        step_index=step_index,
                        agent_index=agent_index,
                        observation=observation,
                        action=_action_list((agent_step or {}).get("action")),
                        reward=float((agent_step or {}).get("reward") or 0.0),
                        status=(agent_step or {}).get("status"),
                        parsed=parsed,
                        memory_before=memory_before,
                        memory_after=memory_after,
                        option_count=len(parsed.select_options),
                        legal_option_types=option_types,
                        select_type=parsed.global_snapshot.select_type,
                        select_context=parsed.global_snapshot.select_context,
                        done=str((agent_step or {}).get("status", "")).upper() == "DONE",
                    )
                )
                if max_samples is not None and len(self.samples) >= max_samples:
                    self.summary.sample_count = len(self.samples)
                    return
        self.summary.sample_count = len(self.samples)

    def to_index_rows(self) -> list[dict[str, Any]]:
        rows = []
        for index, sample in enumerate(self.samples):
            rows.append(
                {
                    "index": index,
                    "replay_id": sample.replay_id,
                    "episode_id": sample.episode_id,
                    "step_index": sample.step_index,
                    "agent_index": sample.agent_index,
                    "action": sample.action,
                    "reward": sample.reward,
                    "status": sample.status,
                    "option_count": sample.option_count,
                    "select_type": sample.select_type,
                    "select_context": sample.select_context,
                    "instance_count": len(sample.parsed.card_instances),
                    "event_count": len(sample.parsed.events),
                    "recent_event_count": len(sample.memory_after.recent_events),
                }
            )
        return rows


def collate_replay_decisions(samples: list[ReplayDecisionSample]) -> dict[str, Any]:
    return {
        "observations": [sample.observation for sample in samples],
        "actions": [sample.action for sample in samples],
        "rewards": [sample.reward for sample in samples],
        "parsed": [sample.parsed for sample in samples],
        "memory_before": [sample.memory_before for sample in samples],
        "memory_after": [sample.memory_after for sample in samples],
        "metadata": [
            {
                "replay_id": sample.replay_id,
                "episode_id": sample.episode_id,
                "step_index": sample.step_index,
                "agent_index": sample.agent_index,
                "option_count": sample.option_count,
                "select_type": sample.select_type,
                "select_context": sample.select_context,
            }
            for sample in samples
        ],
    }
