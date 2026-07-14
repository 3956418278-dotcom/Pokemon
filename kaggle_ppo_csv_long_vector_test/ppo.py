"""Actor-critic PPO utilities for variable legal-action sets.

Each decision has one fixed-width state vector and a variable number of legal
candidate action vectors.  The actor scores every candidate independently from
``concat(state, candidate)``; a categorical distribution over those scores is
then used to select a legal candidate.  The critic deliberately receives only
the state vector.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NamedTuple, Sequence

import torch
from torch import Tensor, nn
from torch.distributions import Categorical


class ActionSample(NamedTuple):
    """A rollout action and the old-policy statistics PPO must retain."""

    action_index: int
    log_prob: float
    value: float


class PolicyEvaluation(NamedTuple):
    """Differentiable evaluation of one action in one variable candidate set."""

    log_prob: Tensor
    entropy: Tensor
    value: Tensor
    logits: Tensor


@dataclass
class Transition:
    """One on-policy transition.

    ``candidates`` has shape ``[num_candidates, action_dim]``.  Consequently,
    consecutive transitions may have different candidate counts without
    padding or an action mask.
    """

    state: Tensor
    candidates: Tensor
    action_index: int
    old_log_prob: float
    reward: float
    done: bool
    value: float

    def __post_init__(self) -> None:
        self.state = torch.as_tensor(self.state, dtype=torch.float32).detach().clone()
        self.candidates = torch.as_tensor(self.candidates, dtype=torch.float32).detach().clone()
        self.action_index = int(self.action_index)
        self.old_log_prob = _as_finite_float(self.old_log_prob, "old_log_prob")
        self.reward = _as_finite_float(self.reward, "reward")
        self.done = bool(self.done)
        self.value = _as_finite_float(self.value, "value")

        if self.state.ndim != 1:
            raise ValueError(f"state must have shape [state_dim], got {tuple(self.state.shape)}")
        if self.candidates.ndim != 2:
            raise ValueError(
                "candidates must have shape [num_candidates, action_dim], "
                f"got {tuple(self.candidates.shape)}"
            )
        if self.candidates.shape[0] == 0:
            raise ValueError("a transition must contain at least one legal candidate")
        if not 0 <= self.action_index < self.candidates.shape[0]:
            raise ValueError(
                f"action_index {self.action_index} is outside "
                f"[0, {self.candidates.shape[0]})"
            )
        if not torch.isfinite(self.state).all():
            raise ValueError("state contains a non-finite value")
        if not torch.isfinite(self.candidates).all():
            raise ValueError("candidates contain a non-finite value")

    @property
    def num_candidates(self) -> int:
        return int(self.candidates.shape[0])


def _as_finite_float(value: float | Tensor, name: str) -> float:
    if isinstance(value, Tensor):
        if value.numel() != 1:
            raise ValueError(f"{name} must be scalar")
        result = float(value.detach().cpu().item())
    else:
        result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite, got {result!r}")
    return result


class ActorCritic(nn.Module):
    """Candidate-scoring actor and state-only critic."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        if self.state_dim <= 0 or self.action_dim <= 0 or self.hidden_dim <= 0:
            raise ValueError("state_dim, action_dim, and hidden_dim must all be positive")

        self.actor = nn.Sequential(
            nn.Linear(self.state_dim + self.action_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, 1),
        )
        self.critic = nn.Sequential(
            nn.Linear(self.state_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, 1),
        )
        self._initialize_parameters()

    def _initialize_parameters(self) -> None:
        actor_linears = [module for module in self.actor if isinstance(module, nn.Linear)]
        critic_linears = [module for module in self.critic if isinstance(module, nn.Linear)]
        for module in [*actor_linears, *critic_linears]:
            nn.init.orthogonal_(module.weight, gain=math.sqrt(2.0))
            nn.init.zeros_(module.bias)
        # Small initial actor logits encourage exploration without making all
        # candidates exactly equal.  The critic uses the conventional unit gain.
        nn.init.orthogonal_(actor_linears[-1].weight, gain=0.01)
        nn.init.orthogonal_(critic_linears[-1].weight, gain=1.0)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def _state_tensor(self, state: Tensor | Sequence[float]) -> Tensor:
        result = torch.as_tensor(state, device=self.device, dtype=self.dtype)
        if result.ndim not in (1, 2):
            raise ValueError(
                "state must have shape [state_dim] or [batch, state_dim], "
                f"got {tuple(result.shape)}"
            )
        if result.shape[-1] != self.state_dim:
            raise ValueError(
                f"state has width {result.shape[-1]}, expected {self.state_dim}"
            )
        return result

    def _candidate_tensor(self, candidates: Tensor | Sequence[Sequence[float]]) -> Tensor:
        result = torch.as_tensor(candidates, device=self.device, dtype=self.dtype)
        if result.ndim not in (2, 3):
            raise ValueError(
                "candidates must have shape [candidates, action_dim] or "
                f"[batch, candidates, action_dim], got {tuple(result.shape)}"
            )
        if result.shape[-1] != self.action_dim:
            raise ValueError(
                f"candidate vectors have width {result.shape[-1]}, expected {self.action_dim}"
            )
        if result.shape[-2] == 0:
            raise ValueError("at least one legal candidate is required")
        return result

    def score_candidates(
        self,
        state: Tensor | Sequence[float],
        candidates: Tensor | Sequence[Sequence[float]],
    ) -> Tensor:
        """Return candidate logits.

        For one state, shapes are ``[state_dim]`` and
        ``[num_candidates, action_dim]`` and the output is ``[num_candidates]``.
        Equal-width batches are also supported with shapes ``[B, state_dim]``
        and ``[B, N, action_dim]``, producing ``[B, N]``.  Rollouts do not need
        equal candidate counts because :meth:`act` and :meth:`evaluate` operate
        on one decision at a time.
        """

        state_tensor = self._state_tensor(state)
        candidate_tensor = self._candidate_tensor(candidates)
        if state_tensor.ndim == 1 and candidate_tensor.ndim == 2:
            expanded_state = state_tensor.unsqueeze(0).expand(candidate_tensor.shape[0], -1)
        elif state_tensor.ndim == 2 and candidate_tensor.ndim == 3:
            if state_tensor.shape[0] != candidate_tensor.shape[0]:
                raise ValueError("state and candidate batch sizes differ")
            expanded_state = state_tensor.unsqueeze(1).expand(
                -1, candidate_tensor.shape[1], -1
            )
        else:
            raise ValueError("state and candidates must both be unbatched or both be batched")
        actor_input = torch.cat((expanded_state, candidate_tensor), dim=-1)
        return self.actor(actor_input).squeeze(-1)

    def state_value(self, state: Tensor | Sequence[float]) -> Tensor:
        """Return a scalar value for one state, or ``[B]`` for a state batch."""

        state_tensor = self._state_tensor(state)
        return self.critic(state_tensor).squeeze(-1)

    def forward(
        self,
        state: Tensor | Sequence[float],
        candidates: Tensor | Sequence[Sequence[float]],
    ) -> tuple[Tensor, Tensor]:
        return self.score_candidates(state, candidates), self.state_value(state)

    @torch.no_grad()
    def act(
        self,
        state: Tensor | Sequence[float],
        candidates: Tensor | Sequence[Sequence[float]],
        *,
        deterministic: bool = False,
    ) -> ActionSample:
        """Choose a legal candidate and return rollout statistics."""

        logits = self.score_candidates(state, candidates)
        if logits.ndim != 1:
            raise ValueError("act expects one state and one candidate set")
        distribution = Categorical(logits=logits)
        action = torch.argmax(logits) if deterministic else distribution.sample()
        return ActionSample(
            action_index=int(action.item()),
            log_prob=float(distribution.log_prob(action).item()),
            value=float(self.state_value(state).item()),
        )

    def evaluate(
        self,
        state: Tensor | Sequence[float],
        candidates: Tensor | Sequence[Sequence[float]],
        action_index: int,
    ) -> PolicyEvaluation:
        """Evaluate a recorded action under the current policy."""

        logits = self.score_candidates(state, candidates)
        if logits.ndim != 1:
            raise ValueError("evaluate expects one state and one candidate set")
        action_index = int(action_index)
        if not 0 <= action_index < logits.shape[0]:
            raise ValueError(f"action_index {action_index} is outside [0, {logits.shape[0]})")
        action = torch.tensor(action_index, device=logits.device, dtype=torch.long)
        distribution = Categorical(logits=logits)
        return PolicyEvaluation(
            log_prob=distribution.log_prob(action),
            entropy=distribution.entropy(),
            value=self.state_value(state),
            logits=logits,
        )


def compute_gae(
    rewards: Sequence[float] | Tensor,
    values: Sequence[float] | Tensor,
    dones: Sequence[bool] | Tensor,
    *,
    next_value: float | Tensor = 0.0,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> tuple[Tensor, Tensor]:
    """Compute generalized advantages and value targets.

    ``done[t]`` cuts both value bootstrapping and recursive advantage flow at
    that transition, so a flat rollout may safely contain multiple episodes.
    ``next_value`` is used only when the final transition is non-terminal.
    """

    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be in [0, 1]")
    if not 0.0 <= gae_lambda <= 1.0:
        raise ValueError("gae_lambda must be in [0, 1]")

    device = values.device if isinstance(values, Tensor) else None
    value_tensor = torch.as_tensor(values, dtype=torch.float32, device=device).detach()
    reward_tensor = torch.as_tensor(rewards, dtype=torch.float32, device=value_tensor.device).detach()
    done_tensor = torch.as_tensor(dones, dtype=torch.bool, device=value_tensor.device).detach()
    if value_tensor.ndim != 1 or reward_tensor.ndim != 1 or done_tensor.ndim != 1:
        raise ValueError("rewards, values, and dones must be one-dimensional")
    if not (len(reward_tensor) == len(value_tensor) == len(done_tensor)):
        raise ValueError("rewards, values, and dones must have equal lengths")
    if not torch.isfinite(reward_tensor).all() or not torch.isfinite(value_tensor).all():
        raise ValueError("rewards and values must be finite")

    advantages = torch.zeros_like(value_tensor)
    if value_tensor.numel() == 0:
        return advantages, advantages.clone()

    bootstrap = torch.as_tensor(
        next_value, dtype=value_tensor.dtype, device=value_tensor.device
    ).detach()
    if bootstrap.numel() != 1 or not torch.isfinite(bootstrap):
        raise ValueError("next_value must be one finite scalar")

    next_advantage = torch.zeros((), dtype=value_tensor.dtype, device=value_tensor.device)
    for index in range(value_tensor.numel() - 1, -1, -1):
        nonterminal = (~done_tensor[index]).to(value_tensor.dtype)
        following_value = bootstrap if index == value_tensor.numel() - 1 else value_tensor[index + 1]
        delta = reward_tensor[index] + gamma * following_value * nonterminal - value_tensor[index]
        next_advantage = (
            delta + gamma * gae_lambda * nonterminal * next_advantage
        )
        advantages[index] = next_advantage
    returns = advantages + value_tensor
    return advantages, returns


def ppo_update(
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    transitions: Sequence[Transition],
    *,
    epochs: int = 4,
    minibatch_size: int | None = None,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    bootstrap_value: float = 0.0,
    clip_epsilon: float = 0.2,
    value_clip_epsilon: float | None = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
    max_grad_norm: float = 0.5,
    normalize_advantages: bool = True,
) -> dict[str, float]:
    """Run PPO optimization over a variable-candidate rollout.

    Candidate sets are evaluated transition-by-transition, while the resulting
    scalar log probabilities, values, and entropies are batched for the PPO
    losses.  This avoids padding candidate sets and keeps the policy exactly
    permutation-equivariant over legal candidates.
    """

    if not transitions:
        raise ValueError("ppo_update requires at least one transition")
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if clip_epsilon < 0.0:
        raise ValueError("clip_epsilon must be non-negative")
    if value_clip_epsilon is not None and value_clip_epsilon < 0.0:
        raise ValueError("value_clip_epsilon must be non-negative or None")
    if value_coef < 0.0 or entropy_coef < 0.0:
        raise ValueError("value_coef and entropy_coef must be non-negative")
    if max_grad_norm <= 0.0:
        raise ValueError("max_grad_norm must be positive")

    count = len(transitions)
    minibatch_size = count if minibatch_size is None else int(minibatch_size)
    if minibatch_size <= 0:
        raise ValueError("minibatch_size must be positive")

    for transition in transitions:
        if transition.state.numel() != model.state_dim:
            raise ValueError("a transition state width does not match the model")
        if transition.candidates.shape[1] != model.action_dim:
            raise ValueError("a transition candidate width does not match the model")

    device = model.device
    dtype = model.dtype
    rewards = torch.tensor([item.reward for item in transitions], device=device, dtype=dtype)
    old_values = torch.tensor([item.value for item in transitions], device=device, dtype=dtype)
    dones = torch.tensor([item.done for item in transitions], device=device, dtype=torch.bool)
    old_log_probs = torch.tensor(
        [item.old_log_prob for item in transitions], device=device, dtype=dtype
    )
    advantages, returns = compute_gae(
        rewards,
        old_values,
        dones,
        next_value=bootstrap_value,
        gamma=gamma,
        gae_lambda=gae_lambda,
    )
    advantages = advantages.to(dtype=dtype)
    returns = returns.to(dtype=dtype)
    raw_advantage_mean = advantages.mean()
    raw_advantage_std = advantages.std(unbiased=False)
    if normalize_advantages and count > 1:
        advantages = (advantages - raw_advantage_mean) / raw_advantage_std.clamp_min(1e-8)

    totals = {
        "loss": 0.0,
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "entropy": 0.0,
        "approx_kl": 0.0,
        "clip_fraction": 0.0,
        "grad_norm": 0.0,
    }
    metric_weight = 0
    update_count = 0

    for _epoch in range(epochs):
        permutation = torch.randperm(count, device=device)
        for start in range(0, count, minibatch_size):
            indices = permutation[start : start + minibatch_size]
            evaluations = [
                model.evaluate(
                    transitions[index].state,
                    transitions[index].candidates,
                    transitions[index].action_index,
                )
                for index in indices.detach().cpu().tolist()
            ]
            new_log_probs = torch.stack([item.log_prob for item in evaluations])
            entropies = torch.stack([item.entropy for item in evaluations])
            new_values = torch.stack([item.value for item in evaluations])

            batch_old_log_probs = old_log_probs[indices]
            batch_old_values = old_values[indices]
            batch_advantages = advantages[indices]
            batch_returns = returns[indices]

            log_ratio = new_log_probs - batch_old_log_probs
            ratio = torch.exp(log_ratio)
            unclipped_objective = ratio * batch_advantages
            clipped_objective = torch.clamp(
                ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon
            ) * batch_advantages
            policy_loss = -torch.minimum(unclipped_objective, clipped_objective).mean()

            value_error = (new_values - batch_returns).pow(2)
            if value_clip_epsilon is None:
                value_loss = 0.5 * value_error.mean()
            else:
                clipped_values = batch_old_values + torch.clamp(
                    new_values - batch_old_values,
                    -value_clip_epsilon,
                    value_clip_epsilon,
                )
                clipped_error = (clipped_values - batch_returns).pow(2)
                value_loss = 0.5 * torch.maximum(value_error, clipped_error).mean()

            entropy = entropies.mean()
            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite PPO loss")

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            if not torch.isfinite(grad_norm):
                raise FloatingPointError("non-finite PPO gradient norm")
            optimizer.step()

            with torch.no_grad():
                # Schulman's commonly used low-variance approximate KL.
                approx_kl = ((torch.exp(log_ratio) - 1.0) - log_ratio).mean()
                clip_fraction = ((ratio - 1.0).abs() > clip_epsilon).to(dtype).mean()
            weight = int(indices.numel())
            values_to_record = {
                "loss": loss,
                "policy_loss": policy_loss,
                "value_loss": value_loss,
                "entropy": entropy,
                "approx_kl": approx_kl,
                "clip_fraction": clip_fraction,
                "grad_norm": grad_norm,
            }
            for name, value in values_to_record.items():
                totals[name] += float(value.detach().cpu().item()) * weight
            metric_weight += weight
            update_count += 1

    metrics = {name: total / metric_weight for name, total in totals.items()}
    metrics.update(
        {
            "updates": float(update_count),
            "transitions": float(count),
            "advantage_mean": float(raw_advantage_mean.detach().cpu().item()),
            "advantage_std": float(raw_advantage_std.detach().cpu().item()),
            "return_mean": float(returns.mean().detach().cpu().item()),
        }
    )
    non_finite = {name: value for name, value in metrics.items() if not math.isfinite(value)}
    if non_finite:
        raise FloatingPointError(f"non-finite PPO metrics: {non_finite}")
    return metrics


__all__ = [
    "ActionSample",
    "ActorCritic",
    "PolicyEvaluation",
    "Transition",
    "compute_gae",
    "ppo_update",
]
