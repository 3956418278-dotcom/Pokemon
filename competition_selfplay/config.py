from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class TargetDeckConfig:
    source: str
    index: int
    expected_name: str
    expected_deck_sha256: str
    card_vocab_path: str
    source_submission_ref: int


@dataclass(frozen=True)
class RewardConfig:
    actor_weights: tuple[float, float, float]
    win: float
    loss: float
    draw: float
    opponent_deck_out_win: float
    self_deck_out_loss: float
    prize_scale: float
    own_active_weight: float
    own_bench_weight: float
    own_energy_weight: float
    opponent_active_weight: float
    opponent_bench_weight: float
    setup_delta_clip: float


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
    gamma: float
    gae_lambda: float
    update_epochs: int
    minibatch_size: int
    clip_epsilon: float
    value_clip_epsilon: float
    entropy_coefficient: float
    value_coefficient: float
    max_grad_norm: float
    target_kl: float
    checkpoint_root: str
    rollout_root: str
    metrics_root: str


@dataclass(frozen=True)
class ModelConfig:
    card_embedding_dim: int
    copy_embedding_dim: int
    instance_embedding_dim: int
    hidden_dim: int
    value_dimensions: int


@dataclass(frozen=True)
class SelfPlayConfig:
    schema_version: str
    target_deck: TargetDeckConfig
    reward: RewardConfig
    promotion: PromotionConfig
    training: TrainingConfig
    model: ModelConfig

    def validate(self) -> None:
        if self.schema_version != "fixed_deck_selfplay_v1":
            raise ValueError(f"unsupported schema_version: {self.schema_version}")
        if len(self.reward.actor_weights) != 3:
            raise ValueError("reward.actor_weights must contain exactly three values")
        if self.model.value_dimensions != 3:
            raise ValueError("model.value_dimensions must be 3")
        if not 0.5 < self.promotion.win_rate_threshold <= 1.0:
            raise ValueError("promotion threshold must be in (0.5, 1.0]")
        if self.promotion.minimum_games > self.promotion.evaluation_games:
            raise ValueError("minimum_games cannot exceed evaluation_games")
        if min(self.training.rollout_episodes, self.training.minibatch_size) <= 0:
            raise ValueError("rollout_episodes and minibatch_size must be positive")


def _section(payload: dict[str, Any], name: str) -> dict[str, Any]:
    value = payload.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"missing mapping section: {name}")
    return value


def load_config(path: str | Path) -> SelfPlayConfig:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("self-play config must be a mapping")
    reward = dict(_section(payload, "reward"))
    reward["actor_weights"] = tuple(float(value) for value in reward["actor_weights"])
    config = SelfPlayConfig(
        schema_version=str(payload.get("schema_version", "")),
        target_deck=TargetDeckConfig(**_section(payload, "target_deck")),
        reward=RewardConfig(**reward),
        promotion=PromotionConfig(**_section(payload, "promotion")),
        training=TrainingConfig(**_section(payload, "training")),
        model=ModelConfig(**_section(payload, "model")),
    )
    config.validate()
    return config
