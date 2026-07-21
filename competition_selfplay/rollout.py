from __future__ import annotations

import copy
import hashlib
import importlib
import json
import sys
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol, Sequence

import torch
from torch import Tensor

from .config import ModelConfig, RewardConfig
from .features import GLOBAL_FEATURE_NAMES, compact_global_features
from .model import SemanticSelfPlayModel, TargetSemanticRewardModule, freeze_module
from .phase import BatchPhase, PhaseController
from .reward import mark_terminal_outcomes
from .semantic import TrajectoryLabelBuilder, semantic_applicability, semantic_state_facts
from .transactions import (
    SelectRecord,
    Transaction,
    TransactionAssembler,
    TransactionEvent,
    closes_transaction,
    is_terminal_state,
    normalize_event,
    selection_is_forced,
    state_seat,
)


class BattleEnvironment(Protocol):
    def start(self, deck0: Sequence[int], deck1: Sequence[int]) -> Mapping[str, Any]: ...

    def select(self, action: Sequence[int]) -> Mapping[str, Any]: ...

    def finish(self) -> None: ...


class CgBattleEnvironment:
    """Thin adapter around the official local ``cg.game`` runtime."""

    def __init__(self, runtime_root: str | None = None) -> None:
        if runtime_root is not None and runtime_root not in sys.path:
            sys.path.insert(0, runtime_root)
        try:
            game = importlib.import_module("cg.game")
        except ImportError as exc:
            raise RuntimeError(
                "cg runtime is unavailable; pass the directory containing the cg package"
            ) from exc
        self._battle_start = game.battle_start
        self._battle_select = game.battle_select
        self._battle_finish = game.battle_finish
        self._visualize_data = game.visualize_data
        self._started = False

    def start(self, deck0: Sequence[int], deck1: Sequence[int]) -> Mapping[str, Any]:
        observation, start_data = self._battle_start(list(deck0), list(deck1))
        if observation is None:
            raise RuntimeError(
                "battle_start failed: "
                f"player={getattr(start_data, 'errorPlayer', None)}, "
                f"type={getattr(start_data, 'errorType', None)}"
            )
        self._started = True
        return observation

    def select(self, action: Sequence[int]) -> Mapping[str, Any]:
        if not self._started:
            raise RuntimeError("battle has not started")
        return self._battle_select(list(action))

    def finish(self) -> None:
        if self._started:
            self._battle_finish()
            self._started = False

    def perspective_observation(
        self,
        seat: int,
        observation: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """Build the opposite legitimate private view for holdout calibration only."""

        frames = json.loads(self._visualize_data())
        if not frames or not isinstance(frames[-1], Mapping):
            raise RuntimeError("cg visualizer did not expose a calibration state")
        full_current = copy.deepcopy(frames[-1].get("current"))
        if not isinstance(full_current, Mapping):
            raise RuntimeError("cg visualizer frame has no current state")
        result = copy.deepcopy(dict(observation))
        result.pop("semantic_facts", None)
        result["current"] = dict(full_current)
        result["current"].pop("semantic_facts", None)
        result["current"]["yourIndex"] = int(seat)
        players = result["current"].get("players") or []
        for player_index, player in enumerate(players):
            if player_index != seat and isinstance(player, dict):
                player["hand"] = None
                player.pop("deck", None)
        return result


@dataclass(frozen=True)
class RolloutBatch:
    transactions: tuple[Transaction, ...]
    phase: BatchPhase
    completed_games: int

    @property
    def learner_transactions(self) -> tuple[Transaction, ...]:
        return tuple(transaction for transaction in self.transactions if transaction.learner_controlled)

    @property
    def transaction_count(self) -> int:
        return len(self.transactions)

    @property
    def non_forced_select_count(self) -> int:
        return sum(transaction.non_forced_select_count for transaction in self.transactions)

    @property
    def forced_select_count(self) -> int:
        return sum(transaction.forced_select_count for transaction in self.transactions)


def _stable_bucket(text: str, size: int) -> tuple[int, float]:
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    number = int.from_bytes(digest, "little")
    return number % size, -1.0 if (number >> 63) else 1.0


class ObservationVectorizer:
    """Deterministic adapter from simulator JSON to the shared encoder input.

    It keeps the existing compact global fields and adds hashed identity/copy
    facts for visible card instances. Opponent hand/deck/discard contents and
    counts are never traversed.
    """

    PUBLIC_OPPONENT_ZONES = ("active", "bench")
    OWN_VISIBLE_ZONES = ("hand", "active", "bench", "discard")

    def __init__(self, model_config: ModelConfig) -> None:
        self.observation_dimensions = model_config.observation_dimensions
        self.option_dimensions = model_config.option_dimensions
        if self.observation_dimensions < len(GLOBAL_FEATURE_NAMES) + 16:
            raise ValueError("observation_dimensions is too small for the fixed globals")

    def state(self, observation: Mapping[str, Any], seat: int) -> Tensor:
        current = observation.get("current") or {}
        select = observation.get("select") or {}
        players = current.get("players") or ({}, {})
        own = players[seat] if seat < len(players) and isinstance(players[seat], Mapping) else {}
        opponent_seat = 1 - seat
        opponent = (
            players[opponent_seat]
            if opponent_seat < len(players) and isinstance(players[opponent_seat], Mapping)
            else {}
        )
        values = {
            "turn": current.get("turn", 0),
            "turn_action_count": current.get("turnActionCount", 0),
            "is_first_player": int(current.get("firstPlayer", -1) == seat),
            "relative_current_player": int(state_seat(observation) not in (None, seat)),
            "self_prize_count": len(own.get("prize") or []),
            "opponent_prize_count": len(opponent.get("prize") or []),
            "self_active_count": len(own.get("active") or []),
            "self_bench_count": len(own.get("bench") or []),
            "opponent_active_count": len(opponent.get("active") or []),
            "opponent_bench_count": len(opponent.get("bench") or []),
            "energy_attached": current.get("energyAttached", False),
            "supporter_played": current.get("supporterPlayed", False),
            "stadium_played": current.get("stadiumPlayed", False),
            "retreated": current.get("retreated", False),
            "select_type": select.get("type", 0),
            "select_context": select.get("context", 0),
            "remain_damage_counter": select.get("remainDamageCounter", 0),
            "remain_energy_cost": select.get("remainEnergyCost", 0),
            "min_count": select.get("minCount", 0),
            "max_count": select.get("maxCount", 0),
        }
        vector = torch.zeros(self.observation_dimensions, dtype=torch.float32)
        global_values = compact_global_features(values)
        vector[: len(global_values)] = torch.tensor(global_values, dtype=torch.float32)
        # Bound raw scales while preserving exact zero/boolean distinctions.
        vector[: len(global_values)] = torch.sign(vector[: len(global_values)]) * torch.log1p(
            vector[: len(global_values)].abs()
        )
        facts = semantic_state_facts(observation, seat)
        reserved = (
            facts.self_deck_count or 0,
            own.get("handCount", len(own.get("hand") or [])),
            int(bool(own.get("poisoned", False))),
            int(bool(own.get("burned", False))),
            int(bool(own.get("asleep", False))),
            int(bool(own.get("paralyzed", False))),
            int(bool(own.get("confused", False))),
            len(facts.rule_lock_ids),
            len(facts.armed_delayed_effect_ids),
            int(facts.recovery_needed),
        )
        reserved_tensor = torch.tensor(reserved, dtype=torch.float32)
        reserved_tensor = torch.sign(reserved_tensor) * torch.log1p(reserved_tensor.abs())
        reserved_start = len(global_values)
        reserved_end = reserved_start + len(reserved)
        vector[reserved_start:reserved_end] = reserved_tensor
        offset = reserved_end
        hashed_size = self.observation_dimensions - offset
        for zone in self.OWN_VISIBLE_ZONES:
            for card in own.get(zone) or []:
                self._add_card(vector, offset, hashed_size, card, owner="self", zone=zone)
        for zone in self.PUBLIC_OPPONENT_ZONES:
            for card in opponent.get(zone) or []:
                self._add_card(vector, offset, hashed_size, card, owner="opponent", zone=zone)
        for card in current.get("stadium") or []:
            owner = "self" if int(card.get("playerIndex", -1)) == seat else "opponent"
            self._add_card(vector, offset, hashed_size, card, owner=owner, zone="stadium")
        return vector

    @staticmethod
    def _add_card(
        vector: Tensor,
        offset: int,
        size: int,
        card: Any,
        *,
        owner: str,
        zone: str,
    ) -> None:
        if not isinstance(card, Mapping):
            return
        card_id = card.get("id", "unknown")
        serial = card.get("serial", "unknown")
        token = f"{owner}:{zone}:id={card_id}:serial={serial}"
        bucket, sign = _stable_bucket(token, size)
        vector[offset + bucket] += sign
        for child_zone in ("energies", "tools", "preEvolution"):
            children = card.get(child_zone) or []
            child_token = f"{owner}:{zone}:{card_id}:{child_zone}:count={len(children)}"
            child_bucket, child_sign = _stable_bucket(child_token, size)
            vector[offset + child_bucket] += child_sign

    def options(self, observation: Mapping[str, Any], seat: int) -> Tensor:
        options = list((observation.get("select") or {}).get("option") or [])
        result = torch.zeros((len(options), self.option_dimensions), dtype=torch.float32)
        for row, option in enumerate(options):
            if not isinstance(option, Mapping):
                continue
            numeric_keys = (
                "type",
                "area",
                "index",
                "playerIndex",
                "cardId",
                "serial",
                "attackId",
                "abilityId",
                "value",
            )
            for column, key in enumerate(numeric_keys):
                if column >= self.option_dimensions:
                    break
                value = option.get(key, 0)
                try:
                    numeric = float(value)
                except (TypeError, ValueError):
                    numeric = 0.0
                if key == "playerIndex" and value in (0, 1):
                    numeric = 0.0 if int(value) == seat else 1.0
                result[row, column] = torch.sign(torch.tensor(numeric)) * torch.log1p(
                    torch.tensor(abs(numeric))
                )
            for key, value in sorted(option.items()):
                bucket, sign = _stable_bucket(f"{key}={value}", self.option_dimensions)
                result[row, bucket] += 0.25 * sign
            card = self.resolve_option_card(observation, option, seat)
            if card is not None:
                for key in ("id", "serial"):
                    bucket, sign = _stable_bucket(
                        f"resolved_{key}={card.get(key, 'unknown')}",
                        self.option_dimensions,
                    )
                    result[row, bucket] += sign
                for child_zone in ("energyCards", "tools", "preEvolution"):
                    bucket, sign = _stable_bucket(
                        f"resolved_{child_zone}={len(card.get(child_zone) or [])}",
                        self.option_dimensions,
                    )
                    result[row, bucket] += 0.5 * sign
        return result

    @staticmethod
    def resolve_option_card(
        observation: Mapping[str, Any],
        option: Mapping[str, Any],
        seat: int,
    ) -> Mapping[str, Any] | None:
        direct_id = option.get("cardId")
        direct_serial = option.get("serial")
        if direct_id is not None or direct_serial is not None:
            return {"id": direct_id, "serial": direct_serial}
        try:
            area = int(option.get("area", 2 if int(option.get("type", -1)) == 7 else -1))
            index = int(option.get("index", -1))
            owner = int(option.get("playerIndex", seat))
        except (TypeError, ValueError):
            return None
        select = observation.get("select") or {}
        if area == 1:
            cards = select.get("deck") or []
            if 0 <= index < len(cards) and isinstance(cards[index], Mapping):
                return cards[index]
            for option_index, candidate in enumerate(select.get("option") or []):
                if not isinstance(candidate, Mapping):
                    continue
                if candidate.get("area") == area and candidate.get("index") == index:
                    if 0 <= option_index < len(cards) and isinstance(cards[option_index], Mapping):
                        return cards[option_index]
            return None
        players = (observation.get("current") or {}).get("players") or ()
        if owner not in (0, 1) or owner >= len(players):
            return None
        zone = {2: "hand", 3: "discard", 4: "active", 5: "bench"}.get(area)
        if zone is None:
            return None
        cards = players[owner].get(zone) or []
        if not (0 <= index < len(cards)) or not isinstance(cards[index], Mapping):
            return None
        parent = cards[index]
        try:
            option_type = int(option.get("type", -1))
        except (TypeError, ValueError):
            option_type = -1
        child_key = "energyCards" if option_type in (5, 6) else "tools" if option_type == 4 else None
        child_index_key = "energyIndex" if option_type in (5, 6) else "toolIndex"
        if child_key is not None:
            try:
                child_index = int(option.get(child_index_key, -1))
            except (TypeError, ValueError):
                child_index = -1
            children = parent.get(child_key) or []
            if 0 <= child_index < len(children) and isinstance(children[child_index], Mapping):
                return children[child_index]
        return parent


def engine_events(observation: Mapping[str, Any]) -> list[TransactionEvent]:
    events: list[TransactionEvent] = []
    current = observation.get("current") or {}
    turn = current.get("turn")
    kind_by_type = {
        10: "card_played",
        11: "energy_attached",
        12: "evolved",
        15: "attack_executed",
        16: "damage",
        23: "result",
    }
    for raw in observation.get("logs") or []:
        if not isinstance(raw, Mapping):
            continue
        event_type = raw.get("type")
        kind = raw.get("kind", kind_by_type.get(event_type, str(event_type)))
        payload = dict(raw)
        payload["kind"] = kind
        payload.setdefault("seat", raw.get("playerIndex"))
        payload.setdefault("turn", turn)
        payload.setdefault("source_card_serial", raw.get("serial"))
        payload.setdefault("cause_transaction_id", raw.get("causeTransactionId"))
        events.append(normalize_event(payload))
    for raw in observation.get("causal_events") or []:
        if isinstance(raw, Mapping):
            events.append(normalize_event(raw))
    return events


class ModelActor:
    def __init__(
        self,
        model: SemanticSelfPlayModel,
        vectorizer: ObservationVectorizer,
        *,
        device: torch.device,
        deterministic: bool = False,
    ) -> None:
        self.model = model
        self.vectorizer = vectorizer
        self.device = device
        self.deterministic = deterministic

    def act(self, observation: Mapping[str, Any], seat: int) -> tuple[list[int], SelectRecord, Tensor]:
        state = self.vectorizer.state(observation, seat).to(self.device)
        options = self.vectorizer.options(observation, seat).to(self.device)
        if len(options) == 0:
            empty = SelectRecord(state.cpu(), options.cpu(), (), 0.0, True, 0.0, 0, 0, False)
            return [], empty, torch.tensor(0.0)
        forced = selection_is_forced(observation)
        select = observation.get("select") or {}
        minimum = int(select.get("minCount", 1) or 0)
        maximum = int(select.get("maxCount", minimum) or minimum)
        minimum = max(0, min(minimum, len(options)))
        maximum = max(minimum, min(maximum, len(options)))
        if forced:
            count = maximum
            actions = tuple(range(count))
            old_log_prob = entropy = 0.0
            stopped_early = False
        else:
            actions, old_log_prob, entropy, stopped_early = self.model.sample_selection(
                state,
                options,
                minimum_count=minimum,
                maximum_count=maximum,
                deterministic=self.deterministic,
            )
        record = SelectRecord(
            state_features=state.detach().cpu(),
            option_features=options.detach().cpu(),
            action_indices=actions,
            old_log_prob=old_log_prob,
            forced=forced,
            entropy=entropy,
            minimum_count=minimum,
            maximum_count=maximum,
            stopped_early=stopped_early,
        )
        return list(actions), record, torch.tensor(old_log_prob)


class AsymmetricRolloutCollector:
    def __init__(
        self,
        *,
        online_model: SemanticSelfPlayModel,
        frozen_opponent: SemanticSelfPlayModel,
        target_semantic: TargetSemanticRewardModule,
        model_config: ModelConfig,
        reward_config: RewardConfig,
        phase_controller: PhaseController,
        device: torch.device | str = "cpu",
    ) -> None:
        self.online_model = online_model
        self.frozen_opponent = freeze_module(frozen_opponent)
        self.target_semantic = freeze_module(target_semantic)
        self.vectorizer = ObservationVectorizer(model_config)
        self.reward_config = reward_config
        self.phase_controller = phase_controller
        self.device = torch.device(device)
        self._next_transaction_id = 0

    @classmethod
    def from_online(
        cls,
        *,
        online_model: SemanticSelfPlayModel,
        model_config: ModelConfig,
        reward_config: RewardConfig,
        phase_controller: PhaseController,
        device: torch.device | str = "cpu",
    ) -> "AsymmetricRolloutCollector":
        opponent = copy.deepcopy(online_model)
        target = online_model.build_target_semantic_module()
        return cls(
            online_model=online_model,
            frozen_opponent=opponent,
            target_semantic=target,
            model_config=model_config,
            reward_config=reward_config,
            phase_controller=phase_controller,
            device=device,
        )

    def collect_batch(
        self,
        *,
        environment_factory: Callable[[], BattleEnvironment],
        deck: Sequence[int],
        games: int,
        max_selects_per_game: int = 2_000,
        count_completed_games: bool = True,
        capture_swapped_states: bool = False,
    ) -> RolloutBatch:
        if games <= 0:
            raise ValueError("games must be positive")
        batch_phase = self.phase_controller.begin_batch()
        transactions: list[Transaction] = []
        for batch_game in range(games):
            absolute_game = batch_phase.completed_games_at_start + batch_game
            learner_seat = self.phase_controller.learner_seat(absolute_game)
            game_transactions = self._collect_game(
                environment_factory(),
                deck=deck,
                learner_seat=learner_seat,
                max_selects=max_selects_per_game,
                capture_swapped_states=capture_swapped_states,
            )
            transactions.extend(game_transactions)
        if count_completed_games:
            self.phase_controller.record_completed_games(games)
        return RolloutBatch(tuple(transactions), batch_phase, games)

    def _state_values(
        self,
        observation: Mapping[str, Any],
        seat: int,
    ) -> tuple[Tensor, Tensor, float, float]:
        features = self.vectorizer.state(observation, seat).to(self.device)
        applicable = semantic_applicability(observation, seat).to(self.device)
        with torch.no_grad():
            online = self.online_model(features.unsqueeze(0), applicable.unsqueeze(0))
            target_phi, _ = self.target_semantic(features.unsqueeze(0), applicable.unsqueeze(0))
        return (
            features.detach().cpu(),
            applicable.detach().cpu(),
            float(online.full_value[0].cpu()),
            float(target_phi[0].cpu()),
        )

    def _collect_game(
        self,
        environment: BattleEnvironment,
        *,
        deck: Sequence[int],
        learner_seat: int,
        max_selects: int,
        capture_swapped_states: bool,
    ) -> list[Transaction]:
        assembler = TransactionAssembler(self._next_transaction_id)
        learner_actor = ModelActor(self.online_model, self.vectorizer, device=self.device)
        opponent_actor = ModelActor(self.frozen_opponent, self.vectorizer, device=self.device)
        winner: int | None = None
        observation: Mapping[str, Any] | None = None
        source_transactions_by_serial: dict[int, int] = {}
        try:
            observation = environment.start(deck, deck)
            for _ in range(max_selects):
                if is_terminal_state(observation):
                    current = observation.get("current") or {}
                    winner = int(current.get("result", 2))
                    if assembler.current is not None:
                        assembler.close(
                            end_state=observation,
                            terminal=True,
                            final_events=engine_events(observation),
                        )
                    break
                seat = state_seat(observation)
                if seat not in (0, 1):
                    raise RuntimeError("simulator observation has no acting seat")
                if assembler.current is None:
                    _, applicable, old_value, target_phi = self._state_values(observation, seat)
                    swapped_state = None
                    if capture_swapped_states:
                        perspective = getattr(environment, "perspective_observation", None)
                        if perspective is not None:
                            swapped_state = perspective(1 - seat, observation)
                    assembler.begin(
                        seat=seat,
                        start_state=observation,
                        learner_controlled=seat == learner_seat,
                        old_full_value=old_value,
                        target_phi_before=target_phi,
                        semantic_applicable=applicable,
                        seat_swapped_state=swapped_state,
                    )
                actor = learner_actor if seat == learner_seat else opponent_actor
                action, record, log_prob = actor.act(observation, seat)
                self._record_selected_sources(
                    observation,
                    action,
                    seat=seat,
                    transaction_id=assembler.current.transaction_id,
                    registry=source_transactions_by_serial,
                )
                next_observation = environment.select(action)
                events = engine_events(next_observation)
                delayed_event = self._delayed_trigger_event(
                    next_observation,
                    trigger_transaction_id=assembler.current.transaction_id,
                    registry=source_transactions_by_serial,
                )
                if delayed_event is not None:
                    events.append(delayed_event)
                assembler.record_select(
                    log_prob=log_prob,
                    forced=record.forced,
                    record=record,
                    events=events,
                )
                if closes_transaction(assembler.current, next_observation):
                    terminal = is_terminal_state(next_observation)
                    assembler.close(end_state=next_observation, terminal=terminal)
                    if terminal:
                        current = next_observation.get("current") or {}
                        winner = int(current.get("result", 2))
                        observation = next_observation
                        break
                observation = next_observation
            else:
                raise RuntimeError(f"game exceeded {max_selects} simulator selects")
        finally:
            environment.finish()

        game_transactions = assembler.completed
        if not game_transactions or winner is None:
            raise RuntimeError("game did not produce a completed transactional trajectory")
        terminal_reason = next(
            (
                int(raw["reason"])
                for raw in (observation or {}).get("logs") or []
                if isinstance(raw, Mapping)
                and raw.get("type") == 23
                and raw.get("reason") is not None
            ),
            None,
        )
        for seat in (0, 1):
            seat_transactions = [transaction for transaction in game_transactions if transaction.seat == seat]
            if seat_transactions:
                seat_transactions[-1].trajectory_terminal_state = observation
                seat_transactions[-1].terminal_reason = terminal_reason
        mark_terminal_outcomes(
            game_transactions,
            winner=winner,
            config=self.reward_config,
        )
        self._set_next_potentials(game_transactions)
        TrajectoryLabelBuilder().build(game_transactions)
        self._next_transaction_id = max(tx.transaction_id for tx in game_transactions) + 1
        return game_transactions

    def _record_selected_sources(
        self,
        observation: Mapping[str, Any],
        action: Sequence[int],
        *,
        seat: int,
        transaction_id: int,
        registry: dict[int, int],
    ) -> None:
        options = list((observation.get("select") or {}).get("option") or [])
        for raw_index in action:
            if not 0 <= int(raw_index) < len(options):
                continue
            option = options[int(raw_index)]
            if not isinstance(option, Mapping):
                continue
            try:
                option_type = int(option.get("type", -1))
            except (TypeError, ValueError):
                continue
            # Playing/attaching a card or explicitly activating its ability
            # establishes the current transaction as the immediate cause. A
            # later automatic resolution from the same serial can then link
            # back without any Card-ID rule table.
            if option_type not in {7, 8, 10, 15}:
                continue
            card = self.vectorizer.resolve_option_card(observation, option, seat)
            if not isinstance(card, Mapping):
                continue
            try:
                serial = int(card.get("serial"))
            except (TypeError, ValueError):
                continue
            registry[serial] = transaction_id

    @staticmethod
    def _delayed_trigger_event(
        observation: Mapping[str, Any],
        *,
        trigger_transaction_id: int,
        registry: Mapping[int, int],
    ) -> TransactionEvent | None:
        select = observation.get("select") or {}
        effect = select.get("effect") or select.get("contextCard")
        if not isinstance(effect, Mapping):
            return None
        try:
            serial = int(effect.get("serial"))
        except (TypeError, ValueError):
            return None
        cause_transaction_id = registry.get(serial)
        if cause_transaction_id is None or cause_transaction_id == trigger_transaction_id:
            return None
        try:
            owner = int(effect.get("playerIndex"))
        except (TypeError, ValueError):
            owner = None
        return TransactionEvent(
            kind="delayed_trigger",
            seat=owner,
            source_card_serial=serial,
            cause_transaction_id=cause_transaction_id,
            details={"cause_kind": "deferred_effect"},
        )

    @staticmethod
    def _set_next_potentials(transactions: Sequence[Transaction]) -> None:
        for seat in (0, 1):
            seat_transactions = [transaction for transaction in transactions if transaction.seat == seat]
            for current, following in zip(seat_transactions, seat_transactions[1:]):
                current.target_phi_after = following.target_phi_before
            if seat_transactions:
                seat_transactions[-1].target_phi_after = 0.0
