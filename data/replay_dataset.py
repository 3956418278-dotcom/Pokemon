from __future__ import annotations

import copy
import gzip
import hashlib
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Iterable, Iterator

from .game_memory import GameMemoryState
from .decision_schema import (
    ActionSemantics,
    DecisionContextRecord,
    DecisionKey,
    FieldState,
    LegalActionTarget,
    MatchContextRecord,
    OptionalField,
    ReplayDecisionContract,
    ResourceContextRecord,
    SerialRegistryRecord,
    SideResourceRecord,
    TrainingTargetsRecord,
    TurnUsageRecord,
)
from .legal_options import build_action_target, infer_action_semantics, policy_loss_mask
from .observation_parser import parse_observation
from .state_schema import ParsedObservation


REPLAY_DATE_PATTERN = re.compile(r"(20\d{2}-\d{2}-\d{2})")


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
    source_path: str | None = None
    source_date: str | None = None
    action_step_index: int | None = None
    observation_fingerprint: str | None = None
    action_semantics: ActionSemantics = ActionSemantics.SINGLE_INDEX
    action_target: LegalActionTarget | None = None

    @property
    def episode_key(self) -> str | None:
        if self.episode_id is not None:
            return f"episode:{self.episode_id}"
        if self.replay_id is not None:
            return f"replay:{self.replay_id}"
        if self.source_path is not None:
            return f"path:{self.source_path}"
        return None

    @property
    def decision_step_index(self) -> int:
        return self.step_index

    @property
    def decision_key(self) -> DecisionKey:
        if self.action_step_index is None:
            raise ValueError("a finalized decision must have an action_step_index")
        return DecisionKey(
            episode_id=self.episode_id,
            decision_step_index=self.decision_step_index,
            action_step_index=self.action_step_index,
            player_index=self.agent_index,
        )


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
    duplicate_old_observations: int = 0
    deck_configuration_actions: int = 0
    unpaired_pending: int = 0
    illegal_action_indices: int = 0
    unexpected_unpaired_actions: int = 0
    action_semantics_counts: Counter[str] = field(default_factory=Counter)


def _int(value: Any, default: int | None = -1) -> int | None:
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
        return [int(_int(item, 0) or 0) for item in value]
    return [int(_int(value, 0) or 0)]


def _option_type(option: dict[str, Any]) -> int:
    value = _int(option.get("type"), -1)
    return int(value) if value is not None else -1


def observation_fingerprint(observation: dict[str, Any]) -> str:
    """Fingerprint policy-visible state while excluding replay transport metadata."""

    payload = {
        "current": observation.get("current"),
        "logs": observation.get("logs"),
        "select": observation.get("select"),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass
class _PendingDecision:
    episode_id: int | None
    decision_step_index: int
    player_index: int
    observation: dict[str, Any]
    parsed: ParsedObservation
    memory_before: GameMemoryState
    memory_after: GameMemoryState
    fingerprint: str


def replay_source_date(path: str | Path) -> str | None:
    match = REPLAY_DATE_PATTERN.search(str(path))
    if match is None:
        return None
    value = match.group(1)
    try:
        date.fromisoformat(value)
    except ValueError:
        return None
    return value


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
        self.summary = ReplayDatasetSummary()
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
                try:
                    replay = json.loads(replay_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as exc:
                    self._record_error(replay_path, f"{type(exc).__name__}: {exc}", stage="replay_load")
                    continue
                self._append_replay(replay, replay_path, max_samples)
                if max_samples is not None and len(self.samples) >= max_samples:
                    return
        self.summary.sample_count = len(self.samples)

    def _load_jsonl(self, replay_path: Path) -> Iterable[dict[str, Any]]:
        try:
            handle = replay_path.open("r", encoding="utf-8")
        except OSError as exc:
            self._record_error(replay_path, f"{type(exc).__name__}: {exc}", stage="replay_load")
            return
        with handle:
            for line_number, line in enumerate(handle, start=1):
                if line.strip():
                    try:
                        replay = json.loads(line)
                    except json.JSONDecodeError as exc:
                        self._record_error(
                            replay_path,
                            f"{type(exc).__name__}: {exc}",
                            stage="replay_load",
                            line=line_number,
                        )
                        continue
                    yield replay

    def _record_error(
        self,
        replay_path: Path,
        error: str,
        *,
        stage: str,
        step: int | None = None,
        agent: int | None = None,
        line: int | None = None,
    ) -> None:
        row: dict[str, Any] = {"replay": str(replay_path), "stage": stage, "error": error}
        if step is not None:
            row["step"] = step
        if agent is not None:
            row["agent"] = agent
        if line is not None:
            row["line"] = line
        self.summary.parser_errors.append(row)

    def _append_replay(self, replay: dict[str, Any], replay_path: Path, max_samples: int | None) -> None:
        if not isinstance(replay, dict) or not isinstance(replay.get("steps"), list):
            self._record_error(
                replay_path,
                "ReplayFormatError: replay must be an object containing a steps list",
                stage="replay_structure",
            )
            return
        self.summary.replay_count += 1
        memories: dict[int, GameMemoryState] = {}
        pending: dict[int, _PendingDecision] = {}
        last_fingerprint: dict[int, str] = {}
        replay_id = replay.get("id")
        info = replay.get("info")
        episode_id = info.get("EpisodeId") if isinstance(info, dict) else None
        source_date = replay_source_date(replay_path)
        for step_index, step in enumerate(replay.get("steps", []) or []):
            if not isinstance(step, list):
                self._record_error(
                    replay_path,
                    "ReplayFormatError: step must be a list",
                    stage="replay_structure",
                    step=step_index,
                )
                continue
            for agent_index, agent_step in enumerate(step or []):
                if self.controlled_agents is not None and agent_index not in self.controlled_agents:
                    continue
                if not isinstance(agent_step, dict):
                    self._record_error(
                        replay_path,
                        "ReplayFormatError: agent step must be an object",
                        stage="replay_structure",
                        step=step_index,
                        agent=agent_index,
                    )
                    continue
                action = _action_list(agent_step.get("action"))
                previous = pending.pop(agent_index, None)
                if previous is not None:
                    option_types = [_option_type(option) for option in previous.parsed.select_options]
                    select = previous.observation.get("select") or {}
                    action_semantics = infer_action_semantics(select, action)
                    try:
                        action_target = build_action_target(previous.observation, action, action_semantics)
                    except ValueError as exc:
                        self.summary.illegal_action_indices += 1
                        self._record_error(
                            replay_path,
                            f"ActionAlignmentError: {exc}",
                            stage="action_alignment",
                            step=step_index,
                            agent=agent_index,
                        )
                        action_target = None
                    if action_target is not None:
                        token_estimate = (
                            1
                            + len(previous.parsed.card_instances)
                            + 1
                            + 1
                            + 2
                            + min(
                                len(previous.memory_after.recent_events),
                                previous.memory_after.max_recent_events,
                            )
                        )
                        self.summary.max_instances = max(
                            self.summary.max_instances, len(previous.parsed.card_instances)
                        )
                        self.summary.max_options = max(
                            self.summary.max_options, len(previous.parsed.select_options)
                        )
                        self.summary.max_events = max(
                            self.summary.max_events, len(previous.parsed.events)
                        )
                        self.summary.max_token_estimate = max(
                            self.summary.max_token_estimate, token_estimate
                        )
                        self.samples.append(
                            ReplayDecisionSample(
                            replay_id=str(replay_id) if replay_id is not None else None,
                            episode_id=_int(episode_id, None),
                            step_index=previous.decision_step_index,
                            action_step_index=step_index,
                            agent_index=agent_index,
                            observation=previous.observation,
                            action=action,
                            reward=float(agent_step.get("reward") or 0.0),
                            status=agent_step.get("status"),
                            parsed=previous.parsed,
                            memory_before=previous.memory_before,
                            memory_after=previous.memory_after,
                            option_count=len(previous.parsed.select_options),
                            legal_option_types=option_types,
                            select_type=previous.parsed.global_snapshot.select_type,
                            select_context=previous.parsed.global_snapshot.select_context,
                            done=str(agent_step.get("status", "")).upper() == "DONE",
                            source_path=str(replay_path),
                            source_date=source_date,
                            observation_fingerprint=previous.fingerprint,
                            action_semantics=action_semantics,
                            action_target=action_target,
                            )
                        )
                        self.summary.action_semantics_counts[action_semantics.value] += 1
                        if max_samples is not None and len(self.samples) >= max_samples:
                            self.summary.sample_count = len(self.samples)
                            return
                elif len(action) == 60:
                    # The initial deck submission is configuration, not behavior cloning.
                    self.summary.deck_configuration_actions += 1
                elif action:
                    self.summary.unexpected_unpaired_actions += 1
                observation = agent_step.get("observation")
                if observation is None:
                    continue
                if not isinstance(observation, dict):
                    self._record_error(
                        replay_path,
                        "ReplayFormatError: observation must be an object",
                        stage="replay_structure",
                        step=step_index,
                        agent=agent_index,
                    )
                    continue
                fingerprint = observation_fingerprint(observation)
                if last_fingerprint.get(agent_index) == fingerprint:
                    self.summary.duplicate_old_observations += 1
                    continue
                last_fingerprint[agent_index] = fingerprint
                try:
                    parsed = parse_observation(observation)
                except Exception as exc:
                    self._record_error(
                        replay_path,
                        f"{type(exc).__name__}: {exc}",
                        stage="observation_parse",
                        step=step_index,
                        agent=agent_index,
                    )
                    continue
                memory = memories.setdefault(agent_index, GameMemoryState())
                memory_before = copy.deepcopy(memory)
                memory.update_from_parsed(parsed)
                memory_after = copy.deepcopy(memory)
                if observation.get("select") is None:
                    self.summary.skipped_no_select += 1
                    continue
                pending[agent_index] = _PendingDecision(
                    episode_id=_int(episode_id, None),
                    decision_step_index=step_index,
                    player_index=agent_index,
                    observation=observation,
                    parsed=parsed,
                    memory_before=memory_before,
                    memory_after=memory_after,
                    fingerprint=fingerprint,
                )
        self.summary.unpaired_pending += len(pending)
        self.summary.sample_count = len(self.samples)

    def to_index_rows(self) -> list[dict[str, Any]]:
        rows = []
        for index, sample in enumerate(self.samples):
            rows.append(
                {
                    "index": index,
                    "replay_id": sample.replay_id,
                    "episode_id": sample.episode_id,
                    "episode_key": sample.episode_key,
                    "source_path": sample.source_path,
                    "source_date": sample.source_date,
                    "step_index": sample.step_index,
                    "decision_step_index": sample.decision_step_index,
                    "action_step_index": sample.action_step_index,
                    "agent_index": sample.agent_index,
                    "action": sample.action,
                    "reward": sample.reward,
                    "status": sample.status,
                    "done": sample.done,
                    "option_count": sample.option_count,
                    "select_type": sample.select_type,
                    "select_context": sample.select_context,
                    "action_semantics": sample.action_semantics.value,
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
                "episode_key": sample.episode_key,
                "source_path": sample.source_path,
                "source_date": sample.source_date,
                "step_index": sample.step_index,
                "decision_step_index": sample.decision_step_index,
                "action_step_index": sample.action_step_index,
                "agent_index": sample.agent_index,
                "option_count": sample.option_count,
                "select_type": sample.select_type,
                "select_context": sample.select_context,
                "action_semantics": sample.action_semantics.value,
                "done": sample.done,
            }
            for sample in samples
        ],
    }


def replay_decision_reference_dict(
    sample: ReplayDecisionSample,
    final_rewards: list[Any] | None = None,
) -> dict[str, Any]:
    """Return the compact canonical label row without copying replay state.

    The source replay remains the only observation/log/card-state store. Loading
    resolves this row by ``source_path`` and ``decision_key`` and rebuilds the
    eight model-facing classes with the current schema implementation.
    """

    final_reward = None
    if final_rewards is not None and sample.agent_index < len(final_rewards):
        value = final_rewards[sample.agent_index]
        final_reward = float(value) if value is not None else None
    outcome = None
    if final_reward is not None:
        outcome = 1 if final_reward > 0 else -1 if final_reward < 0 else 0
    select = sample.observation.get("select") or {}
    return {
        "schema_version": "replay_decision_reference_v1",
        "decision_key": asdict(sample.decision_key),
        "source": {
            "replay_path": sample.source_path,
            "replay_id": sample.replay_id,
        },
        "observation_fingerprint": sample.observation_fingerprint,
        "select_type": sample.select_type,
        "select_context": sample.select_context,
        "min_count": select.get("minCount"),
        "min_count_state": _snapshot_state_name(sample, "select.minCount"),
        "max_count": select.get("maxCount"),
        "max_count_state": _snapshot_state_name(sample, "select.maxCount"),
        "effect_reference": sample.parsed.effect_reference,
        "effect_presence": sample.parsed.effect_presence.name,
        "context_card_reference": sample.parsed.context_card_reference,
        "context_card_presence": sample.parsed.context_card_presence.name,
        "option_count": sample.option_count,
        "action": sample.action,
        "action_semantics": sample.action_semantics.value,
        "action_target": asdict(sample.action_target) if sample.action_target is not None else None,
        "policy_loss_mask": (
            policy_loss_mask(select, sample.action_target)
            if sample.action_target is not None
            else False
        ),
        "reward": sample.reward,
        "status": sample.status,
        "final_outcome": outcome,
    }


def _snapshot_state_name(sample: ReplayDecisionSample, name: str) -> str:
    return sample.parsed.global_snapshot.field_states.get(name, FieldState.MISSING).name


def _optional(value: Any, state: FieldState, placeholder: Any = 0) -> OptionalField:
    return OptionalField(value=value if state is FieldState.PRESENT else placeholder, state=state)


def _combined_state(*states: FieldState) -> FieldState:
    if all(state is FieldState.PRESENT for state in states):
        return FieldState.PRESENT
    for state in (
        FieldState.EXPLICIT_NULL,
        FieldState.MISSING,
        FieldState.UNKNOWN,
        FieldState.NOT_APPLICABLE,
    ):
        if state in states:
            return state
    return FieldState.UNKNOWN


def build_replay_decision_contract(
    sample: ReplayDecisionSample,
    *,
    final_outcome: int | None = None,
    hidden_belief: Any | None = None,
) -> ReplayDecisionContract:
    """Materialize the frozen eight-class view without persisting a second copy."""

    snapshot = sample.parsed.global_snapshot
    states = snapshot.field_states
    your_index = snapshot.your_index
    current = sample.observation.get("current") or {}
    players = current.get("players") or []

    def state(name: str) -> FieldState:
        return states.get(name, FieldState.MISSING)

    def side_resources(absolute_index: int) -> SideResourceRecord:
        counts = (
            snapshot.player_counts[absolute_index]
            if 0 <= absolute_index < len(snapshot.player_counts)
            else {}
        )
        player = players[absolute_index] if 0 <= absolute_index < len(players) else {}
        prefix = f"players[{absolute_index}]"
        bench_state = _combined_state(state(f"{prefix}.benchMax"), state(f"{prefix}.bench"))
        bench_free = 0
        if bench_state is FieldState.PRESENT:
            bench_free = int(player.get("benchMax")) - len(player.get("bench") or [])
        return SideResourceRecord(
            deck_count=_optional(counts.get("deck", 0), state(f"{prefix}.deckCount")),
            hand_count=_optional(counts.get("hand", 0), state(f"{prefix}.handCount")),
            prize_count=_optional(counts.get("prize", 0), state(f"{prefix}.prize")),
            discard_count=_optional(counts.get("discard", 0), state(f"{prefix}.discard")),
            bench_free_slots=_optional(bench_free, bench_state),
        )

    first_state = state("firstPlayer")
    first_player = snapshot.first_player
    starting = first_player == sample.agent_index if first_state is FieldState.PRESENT else False
    match = MatchContextRecord(
        turn=_optional(snapshot.turn, state("turn")),
        turn_action_count=_optional(snapshot.turn_action_count, state("turnActionCount")),
        is_starting_player=_optional(starting, first_state, False),
        # The observation has no explicit turn-owner field. Do not infer it from
        # parity until that engine rule is version-audited.
        is_turn_owner=_optional(False, FieldState.UNKNOWN, False),
    )
    opponent_index = 1 - your_index if your_index in (0, 1) else -1
    resources = ResourceContextRecord(
        self_resources=side_resources(your_index),
        opponent_resources=side_resources(opponent_index),
        turn_usage=TurnUsageRecord(
            energy_attached=_optional(snapshot.energy_attached, state("energyAttached"), False),
            supporter_played=_optional(snapshot.supporter_played, state("supporterPlayed"), False),
            stadium_played=_optional(snapshot.stadium_played, state("stadiumPlayed"), False),
            retreated=_optional(snapshot.retreated, state("retreated"), False),
            turn_owner_relative=_optional(0, FieldState.UNKNOWN),
        ),
    )
    serial_registry = tuple(
        SerialRegistryRecord(
            serial=memory.serial,
            card_id=memory.card_id,
            owner_relative=(
                None
                if memory.player_index is None
                else 0 if memory.player_index == your_index else 1
            ),
            exact_zone=memory.current_area,
            previous_exact_zone=memory.previous_area,
            possible_hidden_zone_mask=memory.possible_hidden_zone_mask,
            currently_visible=memory.currently_visible,
            last_seen_turn=memory.last_seen_turn,
            last_seen_observation=memory.last_seen_observation,
            last_event_type=memory.last_event_type,
            field_states={
                "card_id": (
                    FieldState.PRESENT if memory.card_id is not None else FieldState.UNKNOWN
                ),
                "owner_relative": (
                    FieldState.PRESENT
                    if memory.player_index is not None
                    else FieldState.UNKNOWN
                ),
                "exact_zone": (
                    FieldState.PRESENT
                    if memory.current_area is not None
                    else FieldState.UNKNOWN
                ),
                "previous_exact_zone": (
                    FieldState.PRESENT
                    if memory.previous_area is not None
                    else FieldState.NOT_APPLICABLE
                ),
                "last_event_type": (
                    FieldState.PRESENT
                    if memory.last_event_type is not None
                    else FieldState.NOT_APPLICABLE
                ),
            },
        )
        for _, memory in sorted(sample.memory_after.serials.items())
    )
    decision = DecisionContextRecord(
        select_type=_optional(snapshot.select_type, state("select.type")),
        select_context=_optional(snapshot.select_context, state("select.context")),
        min_count=_optional(snapshot.select_min_count, state("select.minCount")),
        max_count=_optional(snapshot.select_max_count, state("select.maxCount")),
        remain_energy_cost=_optional(
            snapshot.remain_energy_cost, state("select.remainEnergyCost")
        ),
        remain_damage_counter=_optional(
            snapshot.remain_damage_counter, state("select.remainDamageCounter")
        ),
        effect_reference=_optional(
            sample.parsed.effect_reference, sample.parsed.effect_presence, None
        ),
        context_card_reference=_optional(
            sample.parsed.context_card_reference,
            sample.parsed.context_card_presence,
            None,
        ),
    )
    if sample.action_target is None:
        raise ValueError("a replay decision contract requires a valid legal action target")
    targets = TrainingTargetsRecord(
        legal_action=sample.action_target,
        final_outcome=final_outcome,
        hidden_truth_reference=sample.source_path,
        policy_loss_mask=policy_loss_mask(sample.observation.get("select") or {}, sample.action_target),
    )
    card_id_memory = tuple(
        sample.memory_after.card_id_memory_records(
            your_index,
            expected_zone_counts=(
                hidden_belief.expected_zone_counts if hidden_belief is not None else None
            ),
            presence_predictions=(
                hidden_belief.presence_predictions if hidden_belief is not None else None
            ),
            uncertainty=(
                hidden_belief.unresolved_zone_entropy if hidden_belief is not None else None
            ),
        )
    )
    memory_index = {
        (record.owner_relative, record.card_id): index
        for index, record in enumerate(card_id_memory)
    }
    instance_memory_edges = tuple(
        (instance_index, memory_index[(instance.relative_player, int(instance.card_id))])
        for instance_index, instance in enumerate(sample.parsed.card_instances)
        if instance.card_id is not None
        and instance.relative_player is not None
        and (instance.relative_player, int(instance.card_id)) in memory_index
    )
    return ReplayDecisionContract(
        schema_version="replay_decision_contract_v1",
        key=sample.decision_key,
        match=match,
        resources=resources,
        card_instances=tuple(sample.parsed.card_instances),
        recent_events=tuple(sample.memory_after.recent_events),
        serial_registry=serial_registry,
        anonymous_hidden_pools=sample.memory_after.anonymous_hidden_pools_record(your_index),
        card_id_memory=card_id_memory,
        instance_card_id_memory_edges=instance_memory_edges,
        hidden_belief=hidden_belief,
        hidden_belief_state=(
            FieldState.PRESENT if hidden_belief is not None else FieldState.NOT_APPLICABLE
        ),
        decision=decision,
        legal_options=tuple(sample.parsed.select_options),
        targets=targets,
    )


def _iter_replay_payloads(path: Path) -> Iterator[tuple[dict[str, Any], int | None]]:
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if line.strip():
                    yield json.loads(line), line_number
    else:
        yield json.loads(path.read_text(encoding="utf-8")), None


def iter_replay_decision_samples(
    paths: Iterable[str | Path],
    include_no_select: bool = False,
    controlled_agents: set[int] | None = None,
    max_samples: int | None = None,
) -> Iterator[ReplayDecisionSample]:
    """Stream samples while retaining only one replay payload and its decisions."""
    emitted = 0
    for replay_path in iter_replay_paths([Path(path) for path in paths]):
        try:
            payloads = _iter_replay_payloads(replay_path)
            for replay, _line in payloads:
                remaining = None if max_samples is None else max_samples - emitted
                if remaining is not None and remaining <= 0:
                    return
                dataset = ReplayDecisionDataset(
                    [], include_no_select=include_no_select,
                    controlled_agents=controlled_agents,
                )
                dataset._append_replay(replay, replay_path, remaining)
                for sample in dataset.samples:
                    yield sample
                    emitted += 1
                    if max_samples is not None and emitted >= max_samples:
                        return
        except (OSError, json.JSONDecodeError):
            continue


def _distribution(values: list[int]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "min": 0, "max": 0, "mean": 0.0, "counts": {}}
    counts = Counter(values)
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": sum(values) / len(values),
        "counts": {str(key): value for key, value in sorted(counts.items())},
    }


def export_replay_decisions(
    paths: Iterable[str | Path],
    output_dir: Path,
    include_no_select: bool = False,
    controlled_agents: set[int] | None = None,
    max_samples: int | None = None,
) -> dict[str, Any]:
    """Write compact decision references; never duplicate full replay observations."""
    output_dir.mkdir(parents=True, exist_ok=True)
    decisions_dir = output_dir / "decisions"
    reports_dir = output_dir / "reports"
    decisions_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    reference_path = decisions_dir / "decision_references.jsonl.gz"
    error_path = reports_dir / "extraction_errors.jsonl"
    audit_path = reports_dir / "replay_feature_audit.json"
    replay_paths = iter_replay_paths([Path(path) for path in paths])
    summary: dict[str, Any] = {
        "replay_count": 0, "successful_replay_count": 0, "failed_replay_count": 0,
        "decision_sample_count": 0, "observation_parser_error_count": 0,
        "output_serialization_error_count": 0, "max_card_instances": 0, "max_legal_options": 0,
        "duplicate_old_observation": 0, "deck_configuration_action": 0,
        "unpaired_pending": 0, "illegal_action_index": 0,
        "unexpected_unpaired_action": 0, "action_semantics": Counter(),
    }
    step_counts: list[int] = []
    decision_counts: list[int] = []
    errors: list[dict[str, Any]] = []
    if error_path.exists():
        for line in error_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    errors.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    with gzip.open(reference_path, "wt", encoding="utf-8") as reference_handle:
        stop = False
        for replay_path in replay_paths:
            try:
                payloads = _iter_replay_payloads(replay_path)
                for replay, line_number in payloads:
                    summary["replay_count"] += 1
                    steps = replay.get("steps") if isinstance(replay, dict) else None
                    if not isinstance(steps, list):
                        raise ValueError("replay must contain a steps list")
                    step_counts.append(len(steps))
                    dataset = ReplayDecisionDataset(
                        [], include_no_select=include_no_select,
                        controlled_agents=controlled_agents,
                    )
                    remaining = None if max_samples is None else max_samples - summary["decision_sample_count"]
                    dataset._append_replay(replay, replay_path, remaining)
                    if dataset.summary.replay_count == 0:
                        message = dataset.summary.parser_errors[-1]["error"] if dataset.summary.parser_errors else "invalid replay"
                        raise ValueError(message)
                    summary["observation_parser_error_count"] += len(dataset.summary.parser_errors)
                    summary["duplicate_old_observation"] += dataset.summary.duplicate_old_observations
                    summary["deck_configuration_action"] += dataset.summary.deck_configuration_actions
                    summary["unpaired_pending"] += dataset.summary.unpaired_pending
                    summary["illegal_action_index"] += dataset.summary.illegal_action_indices
                    summary["unexpected_unpaired_action"] += dataset.summary.unexpected_unpaired_actions
                    summary["action_semantics"].update(dataset.summary.action_semantics_counts)
                    for error in dataset.summary.parser_errors:
                        errors.append({
                            "stage": "observation parse" if error.get("stage") == "observation_parse" else "replay JSON",
                            "source_file": error.get("replay", str(replay_path)),
                            "seat": error.get("agent"), "step": error.get("step"), "agent": error.get("agent"),
                            "error_type": str(error.get("error", "Error")).split(":", 1)[0],
                            "message": error.get("error", ""),
                        })
                    final_rewards = replay.get("rewards") if isinstance(replay.get("rewards"), list) else None
                    replay_decisions = 0
                    for sample in dataset.samples:
                        try:
                            payload = replay_decision_reference_dict(sample, final_rewards)
                            payload["source"]["replay_record_line"] = line_number
                            reference_handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
                        except (TypeError, ValueError, OSError) as exc:
                            summary["output_serialization_error_count"] += 1
                            errors.append({
                                "stage": "output serialization", "source_file": str(replay_path),
                                "seat": sample.agent_index, "step": sample.step_index, "agent": sample.agent_index,
                                "error_type": type(exc).__name__, "message": str(exc),
                            })
                            continue
                        replay_decisions += 1
                        summary["decision_sample_count"] += 1
                        summary["max_card_instances"] = max(summary["max_card_instances"], len(sample.parsed.card_instances))
                        summary["max_legal_options"] = max(summary["max_legal_options"], sample.option_count)
                        if max_samples is not None and summary["decision_sample_count"] >= max_samples:
                            stop = True
                            break
                    decision_counts.append(replay_decisions)
                    summary["successful_replay_count"] += 1
                    if stop:
                        break
            except (OSError, json.JSONDecodeError, ValueError, TypeError) as exc:
                summary["failed_replay_count"] += 1
                errors.append({
                    "stage": "replay JSON", "source_file": str(replay_path), "seat": None,
                    "step": None, "agent": None, "error_type": type(exc).__name__, "message": str(exc),
                })
            if stop:
                break
    with error_path.open("w", encoding="utf-8") as handle:
        for error in errors:
            handle.write(json.dumps(error, ensure_ascii=False) + "\n")
    summary["steps_per_replay"] = _distribution(step_counts)
    summary["decisions_per_replay"] = _distribution(decision_counts)
    audit = {
        "schema_version": "replay_decision_reference_export_v1",
        "storage_contract": "Replay state is referenced by decision key and is not copied.",
        "decision_reference_file": str(reference_path.relative_to(output_dir)),
        **summary,
        "errors": errors[:100],
    }
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if summary["successful_replay_count"] == 0:
        raise RuntimeError("all replay files failed to parse")
    return summary
