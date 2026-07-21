from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path

from data.replay_dataset import ReplayDecisionDataset

from decision_agent_v1.adapters.card_vocab_adapter import CardVocabulary
from decision_agent_v1.adapters.observation_adapter import ObservationAdapter
from decision_agent_v1.adapters.replay_adapter import adapt_replay_dataset, split_complete_episodes
from decision_agent_v1.contracts.action_contract import ActionSemanticsContract


ROOT = Path(__file__).resolve().parents[2]


def _real_samples():
    vocabulary = CardVocabulary.from_json(ROOT / "static_card/artifacts/card_data/card_id_to_index.json")
    contract = ActionSemanticsContract.load(ROOT / "decision_agent_v1/contracts/action_semantics.json")
    dataset = ReplayDecisionDataset.from_paths(
        [ROOT / "tests/fixtures/replay/episode-84817357-replay.json"], max_replays=1
    )
    return adapt_replay_dataset(dataset, ObservationAdapter(vocabulary), contract)[0]


def test_real_replay_visibility_identity_and_original_indices() -> None:
    samples = _real_samples()
    assert samples
    serialized = str([asdict(sample) for sample in samples[:3]])
    assert "visualize" not in serialized
    assert all(set(sample.visibility_sources) == {"observation.current", "observation.logs", "observation.select"} for sample in samples)
    assert all(
        option.original_option_index == index
        for sample in samples
        for index, option in enumerate(sample.options)
    )
    by_card = {}
    for sample in samples:
        for card in sample.cards:
            if card.card_id is not None and card.serial is not None:
                by_card.setdefault(card.card_id, set()).add(card.serial)
    assert any(len(serials) > 1 for serials in by_card.values())
    assert all(
        card.relative_owner in {0, 1, 2}
        for sample in samples
        for card in sample.cards
    )


def test_terminal_outcome_is_agent_relative() -> None:
    samples = _real_samples()
    outcomes = {sample.agent_index: sample.terminal_outcome.name for sample in samples}
    assert outcomes == {0: "WIN", 1: "LOSS"}


def test_episode_split_never_splits_decisions() -> None:
    sample = _real_samples()[0]
    rows = [replace(sample, source_date="2026-07-10", decision_index=index) for index in range(3)]
    splits = split_complete_episodes(rows, {"2026-07-10"}, set(), set())
    assert len(splits["train"]) == 3
    assert not splits["validation"] and not splits["test"]
