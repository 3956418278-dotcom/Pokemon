from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


TEST_ROOT = Path(__file__).resolve().parents[1]
if str(TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_ROOT))

from ppo import ActorCritic, Transition, compute_gae, ppo_update


def test_actor_critic_output_shapes_and_variable_candidates() -> None:
    torch.manual_seed(1)
    model = ActorCritic(state_dim=5, action_dim=3, hidden_dim=16)
    state = torch.randn(5)

    logits_two, value = model(state, torch.randn(2, 3))
    logits_seven = model.score_candidates(state, torch.randn(7, 3))
    evaluation = model.evaluate(state, torch.randn(4, 3), action_index=2)
    sample = model.act(state, torch.randn(3, 3), deterministic=True)

    assert logits_two.shape == (2,)
    assert logits_seven.shape == (7,)
    assert value.shape == ()
    assert evaluation.log_prob.shape == ()
    assert evaluation.entropy.shape == ()
    assert evaluation.value.shape == ()
    assert evaluation.logits.shape == (4,)
    assert 0 <= sample.action_index < 3
    assert math.isfinite(sample.log_prob)
    assert math.isfinite(sample.value)

    batched_logits, batched_values = model(
        torch.randn(3, 5), torch.randn(3, 6, 3)
    )
    assert batched_logits.shape == (3, 6)
    assert batched_values.shape == (3,)


def test_candidate_reordering_only_reorders_logits_and_probabilities() -> None:
    torch.manual_seed(2)
    model = ActorCritic(state_dim=4, action_dim=2, hidden_dim=12)
    state = torch.randn(4)
    candidates = torch.tensor(
        [[-1.0, 0.2], [0.5, 1.5], [2.0, -0.3], [0.1, -2.0]],
        dtype=torch.float32,
    )
    permutation = torch.tensor([2, 0, 3, 1])

    original_logits = model.score_candidates(state, candidates)
    reordered_logits = model.score_candidates(state, candidates[permutation])
    assert torch.allclose(reordered_logits, original_logits[permutation], atol=1e-7)
    assert torch.allclose(
        reordered_logits.softmax(dim=0),
        original_logits.softmax(dim=0)[permutation],
        atol=1e-7,
    )

    original_action = 1
    reordered_action = int((permutation == original_action).nonzero().item())
    original = model.evaluate(state, candidates, original_action)
    reordered = model.evaluate(state, candidates[permutation], reordered_action)
    assert torch.allclose(original.log_prob, reordered.log_prob, atol=1e-7)
    assert torch.allclose(original.entropy, reordered.entropy, atol=1e-7)
    assert torch.allclose(original.value, reordered.value, atol=1e-7)


def test_gae_stops_at_terminal_episode_boundaries() -> None:
    # The large reward in the second episode must not leak through the terminal
    # transition at index 1 into the first episode.
    advantages, returns = compute_gae(
        rewards=[1.0, 2.0, 10.0],
        values=[0.0, 0.0, 0.0],
        dones=[False, True, True],
        next_value=999.0,
        gamma=0.9,
        gae_lambda=1.0,
    )

    expected = torch.tensor([2.8, 2.0, 10.0])
    torch.testing.assert_close(advantages, expected)
    torch.testing.assert_close(returns, expected)


def test_ppo_update_changes_parameters_and_returns_finite_metrics() -> None:
    torch.manual_seed(3)
    model = ActorCritic(state_dim=6, action_dim=4, hidden_dim=24)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    transitions: list[Transition] = []

    for index in range(12):
        state = torch.randn(6)
        # Vary candidate count on every decision to exercise the unpadded path.
        candidates = torch.randn(2 + index % 5, 4)
        sample = model.act(state, candidates, deterministic=False)
        transitions.append(
            Transition(
                state=state,
                candidates=candidates,
                action_index=sample.action_index,
                old_log_prob=sample.log_prob,
                reward=1.0 if index % 4 == 0 else -0.2,
                done=index in {5, 11},
                value=sample.value,
            )
        )

    before = [parameter.detach().clone() for parameter in model.parameters()]
    metrics = ppo_update(
        model,
        optimizer,
        transitions,
        epochs=3,
        minibatch_size=4,
        value_clip_epsilon=0.2,
        max_grad_norm=0.5,
    )

    assert any(
        not torch.allclose(previous, current.detach())
        for previous, current in zip(before, model.parameters())
    )
    expected_metrics = {
        "loss",
        "policy_loss",
        "value_loss",
        "entropy",
        "approx_kl",
        "clip_fraction",
        "grad_norm",
        "updates",
        "transitions",
        "advantage_mean",
        "advantage_std",
        "return_mean",
    }
    assert expected_metrics <= metrics.keys()
    assert all(math.isfinite(value) for value in metrics.values())
    assert metrics["updates"] == 9.0
    assert metrics["transitions"] == 12.0
