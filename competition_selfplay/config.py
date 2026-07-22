from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


SCHEMA_VERSION = "transactional_semantic_selfplay_v3"
LEGACY_SCHEMA_VERSIONS = frozenset(
    {
        "fixed_deck_selfplay_v1",
        "transactional_semantic_selfplay_v2",
    }
)


def _legacy_schema_error(schema_version: str) -> ValueError:
    if schema_version == "transactional_semantic_selfplay_v2":
        return ValueError(
            "transactional semantic v2 is incompatible with the v3 ten-dimensional "
            f"semantic heads; start a new run with schema_version={SCHEMA_VERSION!r}"
        )
    return ValueError(
        "legacy three-dimensional reward config is obsolete; migrate to "
        f"schema_version={SCHEMA_VERSION!r} and remove actor/setup weights"
    )


@dataclass(frozen=True)
class TargetDeckConfig:
    source: str
    index: int
    expected_name: str
    expected_deck_sha256: str
    source_submission_ref: int


@dataclass(frozen=True)
class RewardConfig:
    terminal_win: float = 1.0
    terminal_loss: float = -1.0
    terminal_draw: float = 0.0
    max_shaping_alpha: float = 0.15
    potential_clip: float = 0.8
    target_ema_tau: float = 0.01
    phase_a_games: int = 20_000
    phase_b_ramp_games: int = 50_000
    calibration_brier_improvement: float = 0.15
    calibration_ece_max: float = 0.10
    calibration_antisymmetry_max: float = 0.08
    calibration_ranking_min: float = 0.60


@dataclass(frozen=True)
class PromotionConfig:
    win_rate_threshold: float
    evaluation_games: int
    minimum_games: int
    draws_as_half_win: bool
    alternate_seats: bool
    max_promotions: int
    max_evaluations_without_promotion: int


@dataclass(frozen=True)
class TrainingConfig:
    seed: int
    device: str
    train_steps_per_iteration: int
    rollout_episodes: int
    learning_rate: float
    gamma: float = 0.997
    gae_lambda: float = 0.95
    update_epochs: int = 4
    minibatch_size: int = 256
    clip_epsilon: float = 0.2
    entropy_coefficient: float = 0.01
    value_coefficient: float = 0.5
    concept_coefficient: float = 1.0
    semantic_value_coefficient: float = 0.5
    residual_value_coefficient: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = 0.02
    normalize_advantage: bool = True
    checkpoint_root: str = "outputs/competition_selfplay/checkpoints"
    rollout_root: str = "outputs/competition_selfplay/rollouts"
    metrics_root: str = "outputs/competition_selfplay/metrics"


@dataclass(frozen=True)
class ModelConfig:
    card_embedding_dim: int
    copy_embedding_dim: int
    instance_embedding_dim: int
    hidden_dim: int
    value_dimensions: int = 1
    observation_dimensions: int = 128
    option_dimensions: int = 32
    semantic_spline_knots: int = 5


@dataclass(frozen=True)
class SelfPlayConfig:
    schema_version: str
    target_deck: TargetDeckConfig
    reward: RewardConfig
    promotion: PromotionConfig
    training: TrainingConfig
    model: ModelConfig

    def validate(self) -> None:
        if self.schema_version in LEGACY_SCHEMA_VERSIONS:
            raise _legacy_schema_error(self.schema_version)
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version: {self.schema_version}")
        if self.model.value_dimensions != 1:
            raise ValueError("model.value_dimensions must be 1 (the full critic is scalar)")
        if not 0.5 < self.promotion.win_rate_threshold <= 1.0:
            raise ValueError("promotion threshold must be in (0.5, 1.0]")
        if self.promotion.minimum_games > self.promotion.evaluation_games:
            raise ValueError("minimum_games cannot exceed evaluation_games")
        if min(self.training.rollout_episodes, self.training.minibatch_size) <= 0:
            raise ValueError("rollout_episodes and minibatch_size must be positive")
        if not 0.0 < self.training.gamma <= 1.0:
            raise ValueError("training.gamma must be in (0, 1]")
        if not 0.0 <= self.training.gae_lambda <= 1.0:
            raise ValueError("training.gae_lambda must be in [0, 1]")
        if self.reward.terminal_win != 1.0 or self.reward.terminal_loss != -1.0:
            raise ValueError("terminal_win/loss are locked to +1/-1")
        if self.reward.terminal_draw != 0.0:
            raise ValueError("terminal_draw is locked to 0")
        if not 0.0 <= self.reward.max_shaping_alpha <= 0.15:
            raise ValueError("max_shaping_alpha must be in [0, 0.15]")
        if self.reward.potential_clip != 0.8:
            raise ValueError("potential_clip is locked to 0.8")
        if not 0.0 < self.reward.target_ema_tau <= 1.0:
            raise ValueError("target_ema_tau must be in (0, 1]")
        if min(self.reward.phase_a_games, self.reward.phase_b_ramp_games) <= 0:
            raise ValueError("phase game counts must be positive")
        if not 0.0 <= self.reward.calibration_brier_improvement <= 1.0:
            raise ValueError("calibration_brier_improvement must be in [0, 1]")
        if self.reward.calibration_ece_max < 0.0 or self.reward.calibration_antisymmetry_max < 0.0:
            raise ValueError("calibration error thresholds cannot be negative")
        if not 0.0 <= self.reward.calibration_ranking_min <= 1.0:
            raise ValueError("calibration_ranking_min must be in [0, 1]")
        if min(
            self.model.hidden_dim,
            self.model.observation_dimensions,
            self.model.option_dimensions,
        ) <= 0:
            raise ValueError("model dimensions must be positive")
        if self.model.semantic_spline_knots < 2:
            raise ValueError("semantic_spline_knots must be at least two")


def _section(payload: dict[str, Any], name: str) -> dict[str, Any]:
    value = payload.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"missing mapping section: {name}")
    return value


def load_config(path: str | Path) -> SelfPlayConfig:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("self-play config must be a mapping")
    schema_version = str(payload.get("schema_version", ""))
    if schema_version in LEGACY_SCHEMA_VERSIONS:
        raise _legacy_schema_error(schema_version)
    reward_payload = _section(payload, "reward")
    config = SelfPlayConfig(
        schema_version=schema_version,
        target_deck=TargetDeckConfig(**_section(payload, "target_deck")),
        reward=RewardConfig(**reward_payload),
        promotion=PromotionConfig(**_section(payload, "promotion")),
        training=TrainingConfig(**_section(payload, "training")),
        model=ModelConfig(**_section(payload, "model")),
    )
    config.validate()
    return config
