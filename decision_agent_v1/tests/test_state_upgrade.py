from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from data.replay_dataset import ReplayDecisionDataset

from decision_agent_v1.adapters.card_vocab_adapter import CardVocabulary
from decision_agent_v1.adapters.observation_adapter import ObservationAdapter
from decision_agent_v1.adapters.replay_adapter import adapt_replay_dataset
from decision_agent_v1.contracts.action_contract import ActionSemanticsContract
from decision_agent_v1.data.collate import collate_decision_samples
from decision_agent_v1.models.policy_value_v2_model import PolicyValueV2Model
from decision_agent_v1.state_upgrade.collate import collate_state_upgrade
from decision_agent_v1.state_upgrade.deck_prior import DeckPrior
from decision_agent_v1.state_upgrade.features import build_state_upgrade_features, event_decision_ages
from decision_agent_v1.training.state_upgrade_losses import state_upgrade_loss


ROOT = Path(__file__).resolve().parents[2]


def _fixture():
    path = ROOT / "tests/fixtures/replay/episode-84817357-replay.json"
    replay = json.loads(path.read_text(encoding="utf-8"))
    decks = [list(replay["steps"][1][seat]["action"]) for seat in (0, 1)]
    vocabulary = CardVocabulary.from_json(ROOT / "static_card/artifacts/card_data/card_id_to_index.json")
    contract = ActionSemanticsContract.load(ROOT / "decision_agent_v1/contracts/action_semantics.json")
    dataset = ReplayDecisionDataset.from_paths([path], max_replays=1)
    samples, _ = adapt_replay_dataset(dataset, ObservationAdapter(vocabulary), contract)
    raw = {(row.agent_index, row.step_index): row for row in dataset.samples if row.action_target is not None}
    card_ids = tuple(sorted(set(decks[0] + decks[1])))
    columns = {card_id: index for index, card_id in enumerate(card_ids)}
    matrix = np.zeros((2, len(card_ids)), dtype=np.float32)
    for template, deck in enumerate(decks):
        for card_id in deck:
            matrix[template, columns[card_id]] += 1
    prior = DeckPrior(
        card_ids=card_ids,
        template_fingerprints=("self", "opponent"),
        template_counts=matrix,
        prior=np.asarray([0.5, 0.5]),
        card_kinds=tuple("TRAINER" for _ in card_ids),
        source_dates=("train",),
        template_hash="fixture",
    )
    return samples, raw, decks, vocabulary, prior


def test_opponent_complete_deck_changes_only_auxiliary_label() -> None:
    samples, raw, decks, vocabulary, prior = _fixture()
    sample = next(row for row in samples if row.policy_supervision)
    memory = raw[(sample.agent_index, sample.step)].memory_after
    common = dict(
        sample=sample,
        memory=memory,
        self_deck=decks[sample.agent_index],
        prior=prior,
        vocabulary=vocabulary,
        next_public_card_id=None,
    )
    first = build_state_upgrade_features(opponent_deck=decks[1 - sample.agent_index], **common)
    second = build_state_upgrade_features(opponent_deck=decks[sample.agent_index], **common)
    assert {**first.__dict__, "archetype_target": None} == {**second.__dict__, "archetype_target": None}


def test_v2_forward_auxiliary_losses_and_token_order() -> None:
    samples, raw, decks, vocabulary, prior = _fixture()
    selected = [row for row in samples if row.policy_supervision][:4]
    base = collate_decision_samples(selected)
    features = [
        build_state_upgrade_features(
            sample,
            raw[(sample.agent_index, sample.step)].memory_after,
            decks[sample.agent_index],
            decks[1 - sample.agent_index],
            prior,
            vocabulary,
            next_public_card_id=decks[1 - sample.agent_index][0],
        )
        for sample in selected
    ]
    base.update(collate_state_upgrade(features))
    model = PolicyValueV2Model(len(vocabulary), prior.template_count, dropout=0.0)
    outputs = model(base)
    losses = state_upgrade_loss(model, outputs, base)
    assert outputs["policy_logits"].shape == base["option_mask"].shape
    assert outputs["archetype_logits"].shape == (4, prior.template_count)
    assert outputs["next_public_logits"].shape == (4, len(vocabulary))
    assert outputs["board_token_mask"].shape[1] == 4 + 16 + base["card_mask"].shape[1]
    assert all(torch.isfinite(losses[key]) for key in ("total_loss", "archetype_loss", "next_public_loss"))
    losses["total_loss"].backward()
    assert model.archetype_head.weight.grad is not None
    assert model.next_public_head.weight.grad is not None


def test_self_prize_identity_is_not_subtracted_from_remaining_pool() -> None:
    samples, raw, decks, vocabulary, prior = _fixture()
    sample = samples[0]
    memory = raw[(sample.agent_index, sample.step)].memory_after
    feature = build_state_upgrade_features(
        sample,
        memory,
        decks[sample.agent_index],
        decks[1 - sample.agent_index],
        prior,
        vocabulary,
        next_public_card_id=None,
    )
    # Deck and face-down prizes are represented as one legitimate unknown pool.
    assert feature.self_deck_summary[10] == 1.0


def test_recent_events_use_exact_agent_decision_distance() -> None:
    samples, raw, _, _, _ = _fixture()
    paired = [
        (sample, raw[(sample.agent_index, sample.step)].memory_after)
        for sample in samples
    ]
    ages = event_decision_ages(paired)
    assert ages
    for (agent, decision), values in ages.items():
        assert all(0 <= value <= decision for value in values)
