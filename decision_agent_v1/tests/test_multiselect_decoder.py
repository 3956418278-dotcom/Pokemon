from __future__ import annotations

import torch

from decision_agent_v1.models.multiselect_decoder import MultiSelectDecoder
from decision_agent_v1.training.losses import autoregressive_policy_loss


def test_single_and_without_replacement_decode() -> None:
    decoder = MultiSelectDecoder(model_dim=4)
    logits = torch.tensor([[4.0, 3.0, 2.0]])
    options = torch.zeros(1, 3, 4)
    mask = torch.tensor([[True, True, True]])
    assert decoder.decode(logits, options, mask, 1) == [[0]]
    sequence = decoder.decode(logits, options, mask, 3)[0]
    assert sequence == [0, 1, 2]
    assert len(set(sequence)) == 3


def test_equivalence_group_loss_marginalizes_members() -> None:
    decoder = MultiSelectDecoder(model_dim=4)
    outputs = {
        "policy_logits": torch.tensor([[0.0, 0.0, -4.0]], requires_grad=True),
        "option_embeddings": torch.zeros(1, 3, 4, requires_grad=True),
    }
    batch = {
        "option_mask": torch.tensor([[True, True, True]]),
        "policy_sample_mask": torch.tensor([True]),
        "target_sequences": [(0,)],
        "target_equivalence_groups": [(7,)],
        "option_equivalence_group": torch.tensor([[7, 7, 8]]),
    }
    loss, _ = autoregressive_policy_loss(decoder, outputs, batch)
    assert loss < 0.02
    loss.backward()
    assert outputs["policy_logits"].grad is not None


def test_masked_inactive_rows_do_not_create_nan_gradients() -> None:
    decoder = MultiSelectDecoder(model_dim=4)
    outputs = {
        "policy_logits": torch.tensor(
            [[0.0, 0.0, -4.0], [1.0, -1.0, float("-inf")]], requires_grad=True
        ),
        "option_embeddings": torch.zeros(2, 3, 4, requires_grad=True),
    }
    batch = {
        "option_mask": torch.tensor([[True, True, True], [True, True, False]]),
        "policy_sample_mask": torch.tensor([True, False]),
        "target_sequences": [(0,), (0,)],
        "target_equivalence_groups": [(7,), (-1,)],
        "option_equivalence_group": torch.tensor([[7, 7, 8], [1, 2, -1]]),
    }
    loss, _ = autoregressive_policy_loss(decoder, outputs, batch)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(outputs["policy_logits"].grad).all()
    assert torch.isfinite(outputs["option_embeddings"].grad).all()


def test_ordered_and_unordered_targets_preserve_original_index_space() -> None:
    original_option_indices = torch.tensor([[8, 3, 11]])
    ordered_positions = [2, 0]
    unordered_positions = sorted([2, 0], key=lambda index: int(original_option_indices[0, index]))
    assert [int(original_option_indices[0, index]) for index in ordered_positions] == [11, 8]
    assert [int(original_option_indices[0, index]) for index in unordered_positions] == [8, 11]
