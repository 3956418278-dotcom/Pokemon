from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

from .config import SelfPlayConfig
from .league import (
    EvaluationResult,
    LeagueAction,
    LeagueController,
    LeagueDecision,
    LeagueState,
)
from .model import SemanticSelfPlayModel, TargetSemanticRewardModule, freeze_module
from .phase import CalibrationMetrics, PhaseController, calibration_metrics
from .reward import (
    discounted_outcome_returns,
    rewards_from_transactions,
    transaction_gae,
)
from .rollout import ObservationVectorizer, RolloutBatch
from .semantic import SemanticConceptTargets, semantic_applicability, semantic_concept_loss
from .transactions import Transaction


@dataclass(frozen=True)
class PreparedTargets:
    advantages: Tensor
    outcome_returns: Tensor


@dataclass(frozen=True)
class PPOUpdateMetrics:
    metrics: dict[str, float | int | str | None]


@dataclass(frozen=True)
class FixedCalibrationHoldout:
    """Immutable compact trajectory-derived holdout excluded from optimizer batches."""

    states: Tensor
    applicable: Tensor
    concept_targets: Tensor
    target_applicable: Tensor
    swapped_states: Tensor
    swapped_applicable: Tensor
    outcome_returns: Tensor

    @classmethod
    def from_transactions(
        cls,
        transactions: Sequence[Transaction],
        vectorizer: ObservationVectorizer,
        *,
        gamma: float,
    ) -> "FixedCalibrationHoldout":
        if not transactions:
            raise ValueError("calibration holdout is empty")
        if any(
            transaction.semantic_target_values is None
            or transaction.semantic_target_applicable is None
            for transaction in transactions
        ):
            raise ValueError("calibration holdout contains an unlabeled transaction")
        return cls(
            states=torch.stack(
                [vectorizer.state(transaction.start_state, transaction.seat) for transaction in transactions]
            ).cpu(),
            applicable=torch.stack(
                [semantic_applicability(transaction.start_state, transaction.seat) for transaction in transactions]
            ).cpu(),
            concept_targets=torch.stack(
                [transaction.semantic_target_values for transaction in transactions]
            ).cpu(),
            target_applicable=torch.stack(
                [transaction.semantic_target_applicable for transaction in transactions]
            ).cpu(),
            swapped_states=torch.stack(
                [
                    vectorizer.state(
                        transaction.seat_swapped_state or transaction.start_state,
                        1 - transaction.seat,
                    )
                    for transaction in transactions
                ]
            ).cpu(),
            swapped_applicable=torch.stack(
                [
                    semantic_applicability(
                        transaction.seat_swapped_state or transaction.start_state,
                        1 - transaction.seat,
                    )
                    for transaction in transactions
                ]
            ).cpu(),
            outcome_returns=discounted_outcome_returns(transactions, gamma=gamma).cpu(),
        )

    @torch.no_grad()
    def evaluate(
        self,
        model: SemanticSelfPlayModel,
        *,
        device: torch.device | str = "cpu",
    ) -> CalibrationMetrics:
        target_device = torch.device(device)
        output = model(self.states.to(target_device), self.applicable.to(target_device))
        swapped_output = model(
            self.swapped_states.to(target_device),
            self.swapped_applicable.to(target_device),
        )
        return calibration_metrics(
            concept_predictions=output.concepts.values,
            concept_targets=self.concept_targets.to(target_device),
            applicable=self.target_applicable.to(target_device),
            full_values=output.full_value,
            outcome_returns=self.outcome_returns.to(target_device),
            swapped_full_values=swapped_output.full_value,
            swapped_concept_predictions=swapped_output.concepts.values,
        )


def _transaction_device_state(
    transaction: Transaction,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    if not transaction.select_records:
        raise ValueError(f"transaction {transaction.transaction_id} has no select records")
    if transaction.semantic_applicable is None:
        raise ValueError(f"transaction {transaction.transaction_id} has no deterministic mask")
    return (
        transaction.select_records[0].state_features.to(device),
        transaction.semantic_applicable.to(device=device, dtype=torch.bool),
    )


def prepare_targets(
    batch: RolloutBatch,
    *,
    gamma: float,
    gae_lambda: float,
    normalize_advantage: bool,
) -> PreparedTargets:
    transactions = batch.transactions
    advantages = torch.zeros(len(transactions), dtype=torch.float32)
    for seat in (0, 1):
        indices = [index for index, tx in enumerate(transactions) if tx.seat == seat]
        if not indices:
            continue
        seat_transactions = [transactions[index] for index in indices]
        rewards = rewards_from_transactions(
            seat_transactions,
            gamma=gamma,
            shaping_alpha=batch.phase.shaping_alpha,
        )
        values = torch.tensor(
            [transaction.old_full_value for transaction in seat_transactions],
            dtype=torch.float32,
        )
        terminals = torch.tensor(
            [transaction.terminal for transaction in seat_transactions],
            dtype=torch.bool,
        )
        next_values = torch.cat((values[1:], values.new_zeros(1)))
        seat_advantages, _ = transaction_gae(
            rewards,
            values,
            terminals,
            gamma=gamma,
            gae_lambda=gae_lambda,
            next_values=next_values,
        )
        advantages[indices] = seat_advantages
    learner_mask = torch.tensor(
        [transaction.learner_controlled for transaction in transactions],
        dtype=torch.bool,
    )
    if normalize_advantage and learner_mask.any():
        learner_advantages = advantages[learner_mask]
        advantages[learner_mask] = (
            learner_advantages - learner_advantages.mean()
        ) / learner_advantages.std(unbiased=False).clamp_min(1e-8)
    return PreparedTargets(
        advantages=advantages,
        outcome_returns=discounted_outcome_returns(transactions, gamma=gamma),
    )


class TransactionalPPO:
    """Joint-probability PPO; only learner-controlled transactions reach actor loss."""

    def __init__(
        self,
        model: SemanticSelfPlayModel,
        optimizer: torch.optim.Optimizer,
        config: SelfPlayConfig,
        *,
        device: torch.device | str = "cpu",
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.config = config
        self.device = torch.device(device)
        self.model.to(self.device)

    @staticmethod
    def actor_transaction_ids(batch: RolloutBatch) -> tuple[int, ...]:
        return tuple(
            transaction.transaction_id
            for transaction in batch.transactions
            if transaction.learner_controlled and transaction.non_forced_select_count > 0
        )

    def _joint_policy_terms(self, transaction: Transaction) -> tuple[Tensor, Tensor]:
        log_probabilities: list[Tensor] = []
        entropies: list[Tensor] = []
        for record in transaction.select_records:
            if record.forced:
                continue
            log_probability, entropy = self.model.selection_log_probability(
                record.state_features.to(self.device),
                record.option_features.to(self.device),
                record.action_indices,
                minimum_count=record.minimum_count,
                maximum_count=record.maximum_count,
                stopped_early=record.stopped_early,
            )
            log_probabilities.append(log_probability)
            entropies.append(entropy)
        if not log_probabilities:
            anchor = next(self.model.parameters()).sum() * 0.0
            return anchor, anchor
        return torch.stack(log_probabilities).sum(), torch.stack(entropies).mean()

    def _minibatch_loss(
        self,
        transactions: Sequence[Transaction],
        global_indices: Tensor,
        prepared: PreparedTargets,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        states_and_masks = [
            _transaction_device_state(transaction, self.device)
            for transaction in transactions
        ]
        states = torch.stack([item[0] for item in states_and_masks])
        applicable = torch.stack([item[1] for item in states_and_masks])
        output = self.model(states, applicable)

        target_values = []
        target_masks = []
        for transaction in transactions:
            if transaction.semantic_target_values is None or transaction.semantic_target_applicable is None:
                raise ValueError(
                    f"transaction {transaction.transaction_id} is missing trajectory concept labels"
                )
            target_values.append(transaction.semantic_target_values)
            target_masks.append(transaction.semantic_target_applicable)
        concept_targets = SemanticConceptTargets(
            values=torch.stack(target_values).to(self.device),
            applicable=torch.stack(target_masks).to(self.device),
        )
        concept_loss = semantic_concept_loss(output.concepts, concept_targets)

        learner_positions = [
            position
            for position, transaction in enumerate(transactions)
            if transaction.learner_controlled
        ]
        if learner_positions:
            learner_index = torch.tensor(learner_positions, device=self.device)
            global_learner_index = global_indices[learner_positions]
            returns = prepared.outcome_returns[global_learner_index].to(self.device)
            semantic_values = output.semantic_value[learner_index]
            residual_values = output.residual_value[learner_index]
            full_values = output.full_value[learner_index]
            semantic_loss = F.huber_loss(semantic_values, returns)
            # Both operands are detached together: residual cannot move semantic
            # value to manufacture an easier target.
            residual_target = (returns - semantic_values).detach()
            residual_loss = F.huber_loss(residual_values, residual_target)
            full_loss = F.huber_loss(full_values, returns)
        else:
            zero = output.full_value.sum() * 0.0
            semantic_loss = residual_loss = full_loss = zero

        policy_losses: list[Tensor] = []
        entropy_values: list[Tensor] = []
        approximate_kls: list[Tensor] = []
        clipped: list[Tensor] = []
        clip_epsilon = self.config.training.clip_epsilon
        for position, transaction in enumerate(transactions):
            if not transaction.learner_controlled or transaction.non_forced_select_count == 0:
                continue
            new_log_probability, entropy = self._joint_policy_terms(transaction)
            old_log_probability = torch.tensor(
                transaction.old_log_prob_sum,
                dtype=new_log_probability.dtype,
                device=self.device,
            )
            ratio = torch.exp(new_log_probability - old_log_probability)
            advantage = prepared.advantages[global_indices[position]].to(self.device)
            unclipped = ratio * advantage
            clipped_ratio = ratio.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon)
            policy_losses.append(-torch.minimum(unclipped, clipped_ratio * advantage))
            entropy_values.append(entropy)
            approximate_kls.append(old_log_probability - new_log_probability)
            clipped.append((ratio != clipped_ratio).to(dtype=ratio.dtype))
        if policy_losses:
            actor_loss = torch.stack(policy_losses).mean()
            entropy = torch.stack(entropy_values).mean()
            approximate_kl = torch.stack(approximate_kls).mean()
            clip_fraction = torch.stack(clipped).mean()
        else:
            zero = output.full_value.sum() * 0.0
            actor_loss = entropy = approximate_kl = clip_fraction = zero

        total = (
            actor_loss
            - self.config.training.entropy_coefficient * entropy
            + self.config.training.concept_coefficient * concept_loss
            + self.config.training.semantic_value_coefficient * semantic_loss
            + self.config.training.residual_value_coefficient * residual_loss
            + self.config.training.value_coefficient * full_loss
        )
        terms = {
            "actor_loss": actor_loss,
            "concept_loss": concept_loss,
            "semantic_loss": semantic_loss,
            "residual_loss": residual_loss,
            "full_loss": full_loss,
            "entropy": entropy,
            "approximate_kl": approximate_kl,
            "clip_fraction": clip_fraction,
            "semantic_potential_mean": output.semantic_value.mean(),
            "semantic_potential_std": output.semantic_value.std(unbiased=False),
            "semantic_confidence_mean": output.concepts.confidence.mean(),
        }
        return total, terms

    def update_online(self, batch: RolloutBatch) -> PPOUpdateMetrics:
        if not batch.transactions:
            raise ValueError("cannot update from an empty rollout batch")
        prepared = prepare_targets(
            batch,
            gamma=self.config.training.gamma,
            gae_lambda=self.config.training.gae_lambda,
            normalize_advantage=self.config.training.normalize_advantage,
        )
        accumulator: dict[str, float] = {}
        update_count = 0
        stopped_for_kl = False
        transaction_count = len(batch.transactions)
        for _ in range(self.config.training.update_epochs):
            permutation = torch.randperm(transaction_count)
            for start in range(0, transaction_count, self.config.training.minibatch_size):
                indices = permutation[start : start + self.config.training.minibatch_size]
                transactions = [batch.transactions[int(index)] for index in indices]
                self.optimizer.zero_grad(set_to_none=True)
                total, terms = self._minibatch_loss(transactions, indices, prepared)
                total.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.training.max_grad_norm,
                )
                self.optimizer.step()
                update_count += 1
                for name, value in terms.items():
                    accumulator[name] = accumulator.get(name, 0.0) + float(value.detach().cpu())
                if float(terms["approximate_kl"].detach().cpu()) > self.config.training.target_kl:
                    stopped_for_kl = True
                    break
            if stopped_for_kl:
                break
        averaged = {
            name: value / max(1, update_count)
            for name, value in accumulator.items()
        }
        metrics: dict[str, float | int | str | None] = {
            **averaged,
            "phase": batch.phase.phase.value,
            "shaping_alpha": batch.phase.shaping_alpha,
            "transaction_count": batch.transaction_count,
            "non_forced_select_count": batch.non_forced_select_count,
            "forced_select_count": batch.forced_select_count,
            "actor_transaction_count": len(self.actor_transaction_ids(batch)),
            "optimizer_update_count": update_count,
            "early_stop_kl": int(stopped_for_kl),
        }
        return PPOUpdateMetrics(metrics)


class AsymmetricSelfPlayTrainer:
    """Coordinates online updates without moving rollout-frozen modules."""

    def __init__(
        self,
        *,
        config: SelfPlayConfig,
        learner: SemanticSelfPlayModel,
        frozen_opponent: SemanticSelfPlayModel,
        target_semantic: TargetSemanticRewardModule,
        optimizer: torch.optim.Optimizer,
        phase_controller: PhaseController,
        device: torch.device | str = "cpu",
        frozen_opponent_revision: int = 0,
        league_state: LeagueState | None = None,
        calibration_holdout: FixedCalibrationHoldout | None = None,
    ) -> None:
        self.config = config
        self.learner = learner
        self.frozen_opponent = freeze_module(frozen_opponent)
        self.target_semantic = freeze_module(target_semantic)
        self.optimizer = optimizer
        self.phase_controller = phase_controller
        self.frozen_opponent_revision = int(frozen_opponent_revision)
        self.calibration_holdout = calibration_holdout
        self.league_controller = LeagueController(config.promotion)
        self.league_state = league_state or LeagueState(
            frozen_opponent_revision=frozen_opponent_revision
        )
        self.ppo = TransactionalPPO(learner, optimizer, config, device=device)

    def update_online(self, batch: RolloutBatch) -> PPOUpdateMetrics:
        """Update learner only; target/opponent remain bitwise fixed here."""
        update = self.ppo.update_online(batch)
        metrics = dict(update.metrics)
        calibration = self.phase_controller.last_calibration
        if calibration is None:
            metrics.update(
                {
                    "semantic_brier": None,
                    "semantic_ece": None,
                    "semantic_antisymmetry_error": None,
                    "semantic_ranking_accuracy": None,
                }
            )
        else:
            metrics.update(calibration.as_dict())
        return PPOUpdateMetrics(metrics)

    @torch.no_grad()
    def finish_rollout_batch(self) -> None:
        """EMA is deliberately a separate post-update boundary."""

        self.target_semantic.ema_update_from(
            self.learner,
            self.config.reward.target_ema_tau,
        )

    @torch.no_grad()
    def promote_frozen_opponent(self, learner_revision: int) -> None:
        self.frozen_opponent.load_state_dict(self.learner.state_dict())
        freeze_module(self.frozen_opponent)
        self.frozen_opponent_revision = int(learner_revision)

    def record_training_iteration(self) -> LeagueState:
        self.league_state = self.league_controller.record_training_iteration(self.league_state)
        return self.league_state

    def apply_league_evaluation(self, result: EvaluationResult) -> LeagueDecision:
        decision = self.league_controller.evaluate(self.league_state, result)
        self.league_state = decision.state
        if decision.action in {LeagueAction.PROMOTE_AND_COPY, LeagueAction.FREEZE_CONVERGED}:
            if decision.state.frozen_opponent_revision == decision.state.learner_revision:
                self.promote_frozen_opponent(decision.state.learner_revision)
        return decision

    def checkpoint_payload(self) -> dict[str, Any]:
        calibration = self.phase_controller.last_calibration
        return {
            "schema_version": self.config.schema_version,
            "learner": self.learner.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "full_critic": {
                "shared_state_encoder": self.learner.shared_state_encoder.state_dict(),
                "semantic_potential_head": self.learner.semantic_potential_head.state_dict(),
                "residual_value_head": self.learner.residual_value_head.state_dict(),
            },
            "online_semantic_heads": self.learner.semantic_concept_heads.state_dict(),
            "target_semantic": self.target_semantic.state_dict(),
            "residual_value": self.learner.residual_value_head.state_dict(),
            "frozen_opponent": self.frozen_opponent.state_dict(),
            "frozen_opponent_reference": self.frozen_opponent_revision,
            "league_state": asdict(self.league_state),
            "current_phase": self.phase_controller.phase.value,
            "completed_games": self.phase_controller.completed_games,
            "shaping_alpha": self.phase_controller.shaping_alpha(),
            "calibration_metrics": None if calibration is None else asdict(calibration),
            "phase_controller": self.phase_controller.metadata(),
            "config_snapshot": asdict(self.config),
            "calibration_holdout": self.calibration_holdout,
        }

    def save_checkpoint(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.checkpoint_payload(), path)
        return path

    def load_checkpoint(self, path: str | Path) -> dict[str, Any]:
        payload = torch.load(Path(path), map_location=self.ppo.device)
        if payload.get("schema_version") != self.config.schema_version:
            raise ValueError("checkpoint schema does not match transactional semantic v2")
        self.learner.load_state_dict(payload["learner"])
        self.optimizer.load_state_dict(payload["optimizer"])
        self.target_semantic.load_state_dict(payload["target_semantic"])
        self.frozen_opponent.load_state_dict(payload["frozen_opponent"])
        self.frozen_opponent_revision = int(payload["frozen_opponent_reference"])
        self.calibration_holdout = payload.get("calibration_holdout")
        if "league_state" in payload:
            self.league_state = LeagueState(**payload["league_state"])
        if "phase_controller" in payload:
            self.phase_controller.load_metadata(payload["phase_controller"])
        freeze_module(self.target_semantic)
        freeze_module(self.frozen_opponent)
        return payload
