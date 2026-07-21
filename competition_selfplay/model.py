from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.distributions import Categorical

from .config import ModelConfig
from .semantic import (
    SemanticConceptHeads,
    SemanticConceptOutput,
    SemanticConceptTargets,
    SemanticPotentialHead,
    semantic_concept_loss,
)


@dataclass(frozen=True)
class ModelOutput:
    concepts: SemanticConceptOutput
    semantic_value: Tensor
    residual_value: Tensor
    full_value: Tensor


@dataclass(frozen=True)
class CriticLosses:
    concept: Tensor
    semantic: Tensor
    residual: Tensor
    full: Tensor

    def detached_metrics(self) -> dict[str, float]:
        return {
            "concept_loss": float(self.concept.detach().cpu()),
            "semantic_loss": float(self.semantic.detach().cpu()),
            "residual_loss": float(self.residual.detach().cpu()),
            "full_loss": float(self.full.detach().cpu()),
        }


class SharedStateEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

    def forward(self, state_features: Tensor) -> Tensor:
        return self.network(state_features)


class OptionPolicyHead(nn.Module):
    def __init__(self, hidden_dim: int, option_dim: int) -> None:
        super().__init__()
        self.option_encoder = nn.Sequential(
            nn.LayerNorm(option_dim),
            nn.Linear(option_dim, hidden_dim),
            nn.GELU(),
        )
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.option_bias = nn.Linear(hidden_dim, 1)
        self.stop_logit = nn.Linear(hidden_dim, 1)
        self.scale = hidden_dim**-0.5

    def forward(self, encoded_state: Tensor, option_features: Tensor) -> Tensor:
        single = encoded_state.ndim == 1
        if single:
            encoded_state = encoded_state.unsqueeze(0)
            option_features = option_features.unsqueeze(0)
        if option_features.ndim != 3 or option_features.shape[0] != encoded_state.shape[0]:
            raise ValueError("options must have shape [batch, options, option_dim]")
        options = self.option_encoder(option_features)
        query = self.query(encoded_state).unsqueeze(1)
        logits = (query * options).sum(dim=-1) * self.scale + self.option_bias(options).squeeze(-1)
        return logits[0] if single else logits

    def stop(self, encoded_state: Tensor) -> Tensor:
        return self.stop_logit(encoded_state).squeeze(-1)


class FullValueHead(nn.Module):
    """The scalar full critic is the identifiable semantic/residual sum."""

    def forward(self, semantic_value: Tensor, residual_value: Tensor) -> Tensor:
        return semantic_value + residual_value


class SemanticSelfPlayModel(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if config.value_dimensions != 1:
            raise ValueError("transactional semantic PPO requires a scalar value")
        self.config = config
        self.shared_state_encoder = SharedStateEncoder(
            config.observation_dimensions,
            config.hidden_dim,
        )
        self.policy_head = OptionPolicyHead(config.hidden_dim, config.option_dimensions)
        self.semantic_concept_heads = SemanticConceptHeads(config.hidden_dim)
        self.semantic_potential_head = SemanticPotentialHead(
            knot_count=config.semantic_spline_knots,
            potential_clip=0.8,
        )
        self.residual_value_head = nn.Linear(config.hidden_dim, 1)
        self.full_value_head = FullValueHead()

    def encode(self, state_features: Tensor) -> Tensor:
        return self.shared_state_encoder(state_features)

    def policy_logits(self, state_features: Tensor, option_features: Tensor) -> Tensor:
        return self.policy_head(self.encode(state_features), option_features)

    def forward(
        self,
        state_features: Tensor,
        applicable: Tensor,
        *,
        terminal: Tensor | None = None,
    ) -> ModelOutput:
        encoded = self.encode(state_features)
        concepts = self.semantic_concept_heads(encoded, applicable)
        # The value path cannot repurpose concept coordinates or game their
        # confidence. Applicability was computed before entering the model.
        semantic_value = self.semantic_potential_head(
            concepts.values.detach(),
            concepts.applicable,
            concepts.confidence.detach(),
            terminal=terminal,
        )
        residual_value = self.residual_value_head(encoded).squeeze(-1)
        full_value = self.full_value_head(semantic_value, residual_value)
        return ModelOutput(
            concepts=concepts,
            semantic_value=semantic_value,
            residual_value=residual_value,
            full_value=full_value,
        )

    def selection_log_probability(
        self,
        state_features: Tensor,
        option_features: Tensor,
        action_indices: Iterable[int],
        *,
        minimum_count: int | None = None,
        maximum_count: int | None = None,
        stopped_early: bool = False,
    ) -> tuple[Tensor, Tensor]:
        """Joint probability for sampling distinct options without replacement."""

        encoded = self.encode(state_features)
        logits = self.policy_head(encoded, option_features)
        stop_logit = self.policy_head.stop(encoded).reshape(())
        if logits.ndim != 1:
            raise ValueError("selection_log_probability accepts one select record")
        available = torch.ones_like(logits, dtype=torch.bool)
        actions = tuple(int(index) for index in action_indices)
        minimum = len(actions) if minimum_count is None else int(minimum_count)
        maximum = len(actions) if maximum_count is None else int(maximum_count)
        if not 0 <= minimum <= maximum <= len(logits):
            raise ValueError("invalid selection count bounds")
        log_probs: list[Tensor] = []
        entropies: list[Tensor] = []
        for step, raw_index in enumerate(actions):
            index = int(raw_index)
            if index < 0 or index >= len(logits) or not bool(available[index]):
                raise ValueError(f"invalid or repeated action index: {index}")
            allow_stop = step >= minimum
            candidate_logits = torch.cat(
                (
                    logits.masked_fill(~available.clone(), float("-inf")),
                    stop_logit.unsqueeze(0) if allow_stop else stop_logit.new_full((1,), float("-inf")),
                )
            )
            distribution = Categorical(logits=candidate_logits)
            chosen = torch.tensor(index, device=logits.device)
            log_probs.append(distribution.log_prob(chosen))
            entropies.append(distribution.entropy())
            available[index] = False
        if stopped_early:
            if len(actions) < minimum or len(actions) >= maximum:
                raise ValueError("stopped_early is inconsistent with selection bounds")
            candidate_logits = torch.cat(
                (
                    logits.masked_fill(~available.clone(), float("-inf")),
                    stop_logit.unsqueeze(0),
                )
            )
            distribution = Categorical(logits=candidate_logits)
            stop_index = torch.tensor(len(logits), device=logits.device)
            log_probs.append(distribution.log_prob(stop_index))
            entropies.append(distribution.entropy())
        elif len(actions) < maximum:
            raise ValueError("a selection below maximum_count must record its stop action")
        if not log_probs:
            zero = logits.sum() * 0.0
            return zero, zero
        return torch.stack(log_probs).sum(), torch.stack(entropies).mean()

    @torch.no_grad()
    def sample_selection(
        self,
        state_features: Tensor,
        option_features: Tensor,
        *,
        count: int | None = None,
        minimum_count: int | None = None,
        maximum_count: int | None = None,
        deterministic: bool = False,
    ) -> tuple[tuple[int, ...], float, float, bool]:
        encoded = self.encode(state_features)
        logits = self.policy_head(encoded, option_features)
        stop_logit = self.policy_head.stop(encoded).reshape(())
        if count is not None:
            minimum = maximum = int(count)
        else:
            minimum = 0 if minimum_count is None else int(minimum_count)
            maximum = len(logits) if maximum_count is None else int(maximum_count)
        if not 0 <= minimum <= maximum <= len(logits):
            raise ValueError("invalid selection count bounds")
        available = torch.ones_like(logits, dtype=torch.bool)
        actions: list[int] = []
        log_probs: list[Tensor] = []
        entropies: list[Tensor] = []
        stopped_early = False
        for step in range(maximum):
            allow_stop = step >= minimum
            candidate_logits = torch.cat(
                (
                    logits.masked_fill(~available, float("-inf")),
                    stop_logit.unsqueeze(0) if allow_stop else stop_logit.new_full((1,), float("-inf")),
                )
            )
            distribution = Categorical(logits=candidate_logits)
            action = candidate_logits.argmax() if deterministic else distribution.sample()
            if int(action) == len(logits):
                log_probs.append(distribution.log_prob(action))
                entropies.append(distribution.entropy())
                stopped_early = True
                break
            actions.append(int(action.cpu()))
            log_probs.append(distribution.log_prob(action))
            entropies.append(distribution.entropy())
            available[action] = False
        joint = torch.stack(log_probs).sum() if log_probs else logits.new_zeros(())
        entropy = torch.stack(entropies).mean() if entropies else logits.new_zeros(())
        return tuple(actions), float(joint.cpu()), float(entropy.cpu()), stopped_early

    def build_target_semantic_module(self) -> "TargetSemanticRewardModule":
        return TargetSemanticRewardModule.from_online(self)


class TargetSemanticRewardModule(nn.Module):
    """Independent frozen copy of the complete shaping path."""

    def __init__(
        self,
        encoder: SharedStateEncoder,
        concept_heads: SemanticConceptHeads,
        potential_head: SemanticPotentialHead,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.concept_heads = concept_heads
        self.potential_head = potential_head
        freeze_module(self)

    @classmethod
    def from_online(cls, model: SemanticSelfPlayModel) -> "TargetSemanticRewardModule":
        return cls(
            copy.deepcopy(model.shared_state_encoder),
            copy.deepcopy(model.semantic_concept_heads),
            copy.deepcopy(model.semantic_potential_head),
        )

    @torch.no_grad()
    def forward(
        self,
        state_features: Tensor,
        applicable: Tensor,
        *,
        terminal: Tensor | None = None,
    ) -> tuple[Tensor, SemanticConceptOutput]:
        encoded = self.encoder(state_features)
        concepts = self.concept_heads(encoded, applicable)
        potential = self.potential_head(
            concepts.values,
            concepts.applicable,
            concepts.confidence,
            terminal=terminal,
        )
        return potential, concepts

    @torch.no_grad()
    def ema_update_from(self, model: SemanticSelfPlayModel, tau: float) -> None:
        if not 0.0 < tau <= 1.0:
            raise ValueError("tau must be in (0, 1]")
        sources = (
            model.shared_state_encoder,
            model.semantic_concept_heads,
            model.semantic_potential_head,
        )
        targets = (self.encoder, self.concept_heads, self.potential_head)
        for target_module, source_module in zip(targets, sources):
            target_state = target_module.state_dict()
            source_state = source_module.state_dict()
            if target_state.keys() != source_state.keys():
                raise RuntimeError("online and target semantic module structures diverged")
            for name, target_value in target_state.items():
                source_value = source_state[name].to(device=target_value.device)
                if torch.is_floating_point(target_value):
                    target_value.lerp_(source_value, tau)
                else:
                    target_value.copy_(source_value)
        freeze_module(self)


def freeze_module(module: nn.Module) -> nn.Module:
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    return module


def compute_critic_losses(
    output: ModelOutput,
    concept_targets: SemanticConceptTargets,
    outcome_returns: Tensor,
) -> CriticLosses:
    returns = outcome_returns.to(device=output.full_value.device, dtype=output.full_value.dtype)
    concept = semantic_concept_loss(output.concepts, SemanticConceptTargets(
        values=concept_targets.values.to(output.concepts.values.device),
        applicable=concept_targets.applicable.to(output.concepts.values.device),
    ))
    semantic = F.huber_loss(output.semantic_value, returns)
    residual_target = (returns - output.semantic_value).detach()
    residual = F.huber_loss(output.residual_value, residual_target)
    full = F.huber_loss(output.full_value, returns)
    return CriticLosses(concept=concept, semantic=semantic, residual=residual, full=full)
