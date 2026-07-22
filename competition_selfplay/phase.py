from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum

import torch
from torch import Tensor

from .config import RewardConfig
from .semantic import BINARY_CONCEPT_INDICES


class TrainingPhase(str, Enum):
    PHASE_A = "A"
    PHASE_B = "B"


@dataclass(frozen=True)
class CalibrationMetrics:
    concept_brier_score: float
    concept_constant_prior_brier: float
    concept_brier_improvement: float
    concept_ece: float
    semantic_antisymmetry_error: float
    semantic_ranking_accuracy: float
    full_antisymmetry_error: float
    full_ranking_accuracy: float

    def as_dict(self) -> dict[str, float]:
        return {
            "concept_brier_score": self.concept_brier_score,
            "concept_constant_prior_brier": self.concept_constant_prior_brier,
            "concept_brier_improvement": self.concept_brier_improvement,
            "concept_ece": self.concept_ece,
            "semantic_antisymmetry_error": self.semantic_antisymmetry_error,
            "semantic_ranking_accuracy": self.semantic_ranking_accuracy,
            "full_antisymmetry_error": self.full_antisymmetry_error,
            "full_ranking_accuracy": self.full_ranking_accuracy,
        }


@dataclass(frozen=True)
class CalibrationDecision:
    passed: bool
    failures: tuple[str, ...]


@dataclass(frozen=True)
class BatchPhase:
    phase: TrainingPhase
    shaping_alpha: float
    completed_games_at_start: int


def _masked_binary_values(
    predictions: Tensor,
    targets: Tensor,
    applicable: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    indices = list(BINARY_CONCEPT_INDICES)
    prediction = predictions[..., indices].reshape(-1)
    target = targets[..., indices].reshape(-1)
    mask = applicable[..., indices].reshape(-1).to(dtype=torch.bool)
    return prediction[mask], target[mask], mask


def expected_calibration_error(
    probabilities: Tensor,
    targets: Tensor,
    *,
    bins: int = 10,
) -> float:
    if probabilities.numel() == 0:
        return float("inf")
    probabilities = probabilities.detach().float().clamp(0.0, 1.0)
    targets = targets.detach().float()
    boundaries = torch.linspace(0.0, 1.0, bins + 1, device=probabilities.device)
    result = probabilities.new_zeros(())
    for index in range(bins):
        if index == bins - 1:
            selected = (probabilities >= boundaries[index]) & (probabilities <= boundaries[index + 1])
        else:
            selected = (probabilities >= boundaries[index]) & (probabilities < boundaries[index + 1])
        if selected.any():
            confidence = probabilities[selected].mean()
            accuracy = targets[selected].mean()
            result = result + selected.float().mean() * (confidence - accuracy).abs()
    return float(result.cpu())


def value_ranking_accuracy(values: Tensor, returns: Tensor) -> float:
    values = values.detach().flatten().float()
    returns = returns.detach().flatten().float()
    if len(values) != len(returns):
        raise ValueError("values and returns must have equal length")
    correct = total = 0
    for left in range(len(values)):
        for right in range(left + 1, len(values)):
            target_delta = float(returns[left] - returns[right])
            if target_delta == 0.0:
                continue
            value_delta = float(values[left] - values[right])
            correct += int(value_delta * target_delta > 0.0)
            total += 1
    return correct / total if total else 0.0


def calibration_metrics(
    *,
    concept_predictions: Tensor,
    concept_targets: Tensor,
    applicable: Tensor,
    semantic_values: Tensor,
    full_values: Tensor,
    outcome_returns: Tensor,
    swapped_semantic_values: Tensor,
    swapped_full_values: Tensor,
    prior_probabilities: Tensor | None = None,
) -> CalibrationMetrics:
    probabilities, targets, _ = _masked_binary_values(
        concept_predictions,
        concept_targets,
        applicable,
    )
    if probabilities.numel() == 0:
        brier = prior_brier = float("inf")
        improvement = float("-inf")
        ece = float("inf")
    else:
        brier = float(((probabilities - targets) ** 2).mean().cpu())
        if prior_probabilities is None:
            binary_indices = list(BINARY_CONCEPT_INDICES)
            raw_targets = concept_targets[..., binary_indices].reshape(
                -1, len(binary_indices)
            )
            raw_mask = applicable[..., binary_indices].reshape(
                -1, len(binary_indices)
            ).to(dtype=torch.bool)
            prior_matrix = torch.zeros_like(raw_targets)
            for column in range(len(binary_indices)):
                selected_targets = raw_targets[:, column][raw_mask[:, column]]
                prior_value = (
                    selected_targets.mean()
                    if selected_targets.numel()
                    else raw_targets.new_tensor(0.5)
                )
                prior_matrix[:, column] = prior_value
            prior = prior_matrix.reshape(-1)[raw_mask.reshape(-1)]
        else:
            supplied = prior_probabilities.detach().to(targets.device).float()
            if supplied.numel() == 1:
                prior = supplied.expand_as(targets)
            elif supplied.shape == targets.shape:
                prior = supplied
            else:
                raise ValueError("prior_probabilities must be scalar or match masked targets")
        prior_brier = float(((prior - targets) ** 2).mean().cpu())
        improvement = (prior_brier - brier) / max(prior_brier, 1e-12)
        ece = expected_calibration_error(probabilities, targets)
    semantic = semantic_values.detach().flatten().float()
    swapped_semantic = swapped_semantic_values.detach().flatten().float()
    if semantic.shape != swapped_semantic.shape:
        raise ValueError("seat-swapped semantic values must align one-to-one")
    semantic_antisymmetry = (
        float((semantic + swapped_semantic).abs().mean().cpu())
        if semantic.numel()
        else float("inf")
    )
    full = full_values.detach().flatten().float()
    swapped_full = swapped_full_values.detach().flatten().float()
    if full.shape != swapped_full.shape:
        raise ValueError("seat-swapped full values must align one-to-one")
    full_antisymmetry = (
        float((full + swapped_full).abs().mean().cpu())
        if full.numel()
        else float("inf")
    )
    return CalibrationMetrics(
        concept_brier_score=brier,
        concept_constant_prior_brier=prior_brier,
        concept_brier_improvement=improvement,
        concept_ece=ece,
        semantic_antisymmetry_error=semantic_antisymmetry,
        semantic_ranking_accuracy=value_ranking_accuracy(
            semantic_values,
            outcome_returns,
        ),
        full_antisymmetry_error=full_antisymmetry,
        full_ranking_accuracy=value_ranking_accuracy(full_values, outcome_returns),
    )


class CalibrationGate:
    def __init__(self, config: RewardConfig) -> None:
        self.config = config

    def evaluate(self, metrics: CalibrationMetrics) -> CalibrationDecision:
        failures: list[str] = []
        if (
            metrics.concept_brier_improvement
            < self.config.calibration_brier_improvement
        ):
            failures.append("concept_brier_improvement")
        if metrics.concept_ece > self.config.calibration_ece_max:
            failures.append("concept_ece")
        if (
            metrics.semantic_antisymmetry_error
            > self.config.calibration_antisymmetry_max
        ):
            failures.append("semantic_antisymmetry")
        if metrics.semantic_ranking_accuracy < self.config.calibration_ranking_min:
            failures.append("semantic_ranking")
        return CalibrationDecision(passed=not failures, failures=tuple(failures))


class PhaseController:
    """Owns the one-way calibrated transition and freezes alpha per batch."""

    def __init__(
        self,
        config: RewardConfig,
        *,
        completed_games: int = 0,
        gate_passed_at_game: int | None = None,
        calibration: CalibrationMetrics | None = None,
    ) -> None:
        self.config = config
        self.completed_games = int(completed_games)
        if gate_passed_at_game is not None and gate_passed_at_game < config.phase_a_games:
            raise ValueError("Phase B gate cannot precede the locked Phase A game count")
        if gate_passed_at_game is not None and self.completed_games < gate_passed_at_game:
            raise ValueError("completed_games cannot precede the recorded gate game")
        self.gate_passed_at_game = gate_passed_at_game
        self.last_calibration = calibration
        self.gate = CalibrationGate(config)

    @property
    def phase(self) -> TrainingPhase:
        return TrainingPhase.PHASE_B if self.gate_passed_at_game is not None else TrainingPhase.PHASE_A

    def consider_calibration(self, metrics: CalibrationMetrics) -> CalibrationDecision:
        self.last_calibration = metrics
        decision = self.gate.evaluate(metrics)
        if (
            decision.passed
            and self.completed_games >= self.config.phase_a_games
            and self.gate_passed_at_game is None
        ):
            self.gate_passed_at_game = self.completed_games
        return decision

    def shaping_alpha(self, completed_games: int | None = None) -> float:
        games = self.completed_games if completed_games is None else int(completed_games)
        if self.gate_passed_at_game is None:
            return 0.0
        progress = max(0, games - self.gate_passed_at_game)
        fraction = min(1.0, progress / self.config.phase_b_ramp_games)
        return self.config.max_shaping_alpha * fraction

    def begin_batch(self) -> BatchPhase:
        return BatchPhase(
            phase=self.phase,
            shaping_alpha=self.shaping_alpha(),
            completed_games_at_start=self.completed_games,
        )

    def record_completed_games(self, games: int) -> None:
        if games < 0:
            raise ValueError("games cannot be negative")
        self.completed_games += int(games)

    @staticmethod
    def learner_seat(game_index: int) -> int:
        return int(game_index) % 2

    def metadata(self) -> dict[str, object]:
        return {
            "phase": self.phase.value,
            "completed_games": self.completed_games,
            "gate_passed_at_game": self.gate_passed_at_game,
            "shaping_alpha": self.shaping_alpha(),
            "calibration_metrics": (
                None if self.last_calibration is None else asdict(self.last_calibration)
            ),
        }

    def load_metadata(self, metadata: dict[str, object]) -> None:
        self.completed_games = int(metadata.get("completed_games", 0))
        gate_game = metadata.get("gate_passed_at_game")
        self.gate_passed_at_game = None if gate_game is None else int(gate_game)
        if (
            self.gate_passed_at_game is not None
            and self.gate_passed_at_game < self.config.phase_a_games
        ):
            raise ValueError("checkpoint Phase B gate predates the locked Phase A game count")
        if (
            self.gate_passed_at_game is not None
            and self.completed_games < self.gate_passed_at_game
        ):
            raise ValueError("checkpoint completed_games predates its Phase B gate")
        raw_calibration = metadata.get("calibration_metrics")
        self.last_calibration = (
            None
            if not isinstance(raw_calibration, dict)
            else CalibrationMetrics(**raw_calibration)
        )
