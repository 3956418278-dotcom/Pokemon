from __future__ import annotations

from pathlib import Path

import torch

from data.replay_dataset import ReplayDecisionDataset

from decision_agent_v1.adapters.card_vocab_adapter import CardVocabulary
from decision_agent_v1.adapters.observation_adapter import ObservationAdapter
from decision_agent_v1.adapters.replay_adapter import adapt_replay_dataset
from decision_agent_v1.contracts.action_contract import ActionSemanticsContract
from decision_agent_v1.data.collate import collate_decision_samples
from decision_agent_v1.models.policy_value_model import PolicyValueModel
from decision_agent_v1.training.losses import joint_policy_value_loss
from decision_agent_v1.training.train_policy_value import gradient_norm


ROOT = Path(__file__).resolve().parents[2]


def _batch():
    vocabulary = CardVocabulary.from_json(ROOT / "static_card/artifacts/card_data/card_id_to_index.json")
    contract = ActionSemanticsContract.load(ROOT / "decision_agent_v1/contracts/action_semantics.json")
    dataset = ReplayDecisionDataset.from_paths(
        [ROOT / "tests/fixtures/replay/episode-84817357-replay.json"], max_replays=1
    )
    samples, _ = adapt_replay_dataset(dataset, ObservationAdapter(vocabulary), contract)
    selected = [sample for sample in samples if sample.policy_supervision][:4]
    return collate_decision_samples(selected), vocabulary


def test_variable_batch_forward_and_masks() -> None:
    batch, vocabulary = _batch()
    model = PolicyValueModel(len(vocabulary), dropout=0.0)
    outputs = model(batch)
    assert outputs["policy_logits"].shape == batch["option_mask"].shape
    assert outputs["value_logits"].shape == (len(batch["target_sequences"]), 3)
    assert torch.isfinite(outputs["policy_logits"][batch["option_mask"]]).all()
    assert (outputs["policy_logits"][~batch["option_mask"]] == float("-inf")).all()
    assert outputs["contextual_card_tokens"].shape[:2] == batch["card_index"].shape


def test_policy_value_and_joint_gradients_reach_shared_modules() -> None:
    batch, vocabulary = _batch()
    model = PolicyValueModel(len(vocabulary), dropout=0.0)
    model.zero_grad(set_to_none=True)
    outputs = model(batch)
    losses = joint_policy_value_loss(model.multiselect_decoder, outputs, batch)
    losses["policy_loss"].backward()
    assert gradient_norm(model.board_encoder) > 0
    assert gradient_norm(model.policy_head) > 0

    model.zero_grad(set_to_none=True)
    outputs = model(batch)
    losses = joint_policy_value_loss(model.multiselect_decoder, outputs, batch)
    losses["value_loss"].backward()
    assert gradient_norm(model.board_encoder) > 0
    assert gradient_norm(model.value_head) > 0

    model.zero_grad(set_to_none=True)
    outputs = model(batch)
    losses = joint_policy_value_loss(model.multiselect_decoder, outputs, batch)
    losses["total_loss"].backward()
    assert model.card_instance_encoder.card_id_embedding.weight.grad is not None
    assert model.card_instance_encoder.card_id_embedding.weight.grad.norm() > 0
    referenced = batch["option_card_token_index"] >= 0
    assert referenced.any()
