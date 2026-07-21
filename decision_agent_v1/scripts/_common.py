from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import torch

from data.replay_dataset import ReplayDecisionDataset

from decision_agent_v1.adapters.card_vocab_adapter import CardVocabulary
from decision_agent_v1.adapters.observation_adapter import ObservationAdapter
from decision_agent_v1.adapters.replay_adapter import ReplayAdapterReport, adapt_replay_dataset
from decision_agent_v1.contracts.action_contract import ActionSemanticsContract
from decision_agent_v1.contracts.schemas import DecisionSampleV1, SelectionMode


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPLAY = ROOT / "outputs/replay_extract/replays/replays.zip"
DEFAULT_FIXTURE = ROOT / "tests/fixtures/replay/episode-84817357-replay.json"
DEFAULT_VOCAB = ROOT / "static_card/artifacts/card_data/card_id_to_index.json"
DEFAULT_CONTRACT = ROOT / "decision_agent_v1/contracts/action_semantics.json"
OUTPUT_ROOT = ROOT / "outputs/decision_agent_v1"


def add_data_arguments(parser: argparse.ArgumentParser, default_replays: int = 4) -> None:
    parser.add_argument("--replay-path", type=Path, default=DEFAULT_REPLAY)
    parser.add_argument("--card-vocab", type=Path, default=DEFAULT_VOCAB)
    parser.add_argument("--action-contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--max-replays", type=int, default=default_replays)
    parser.add_argument("--max-decisions", type=int, default=None)


def load_samples(args: argparse.Namespace) -> tuple[list[DecisionSampleV1], ReplayAdapterReport, CardVocabulary, ReplayDecisionDataset]:
    vocabulary = CardVocabulary.from_json(args.card_vocab)
    contract = ActionSemanticsContract.load(args.action_contract)
    dataset = ReplayDecisionDataset.from_paths(
        [args.replay_path],
        max_replays=args.max_replays,
        archive_member_selection="EVENLY_SPACED",
    )
    samples, report = adapt_replay_dataset(
        dataset,
        ObservationAdapter(vocabulary),
        contract,
        max_decisions=args.max_decisions,
    )
    if not samples:
        raise RuntimeError("no terminal-labeled DecisionSampleV1 rows were produced")
    return samples, report, vocabulary, dataset


def stratified_tiny_samples(samples: list[DecisionSampleV1], count: int = 32) -> list[DecisionSampleV1]:
    groups: dict[tuple[str, int], list[DecisionSampleV1]] = defaultdict(list)
    for sample in samples:
        groups[(sample.episode_id, sample.agent_index)].append(sample)
    chosen: list[DecisionSampleV1] = []
    depth = 0
    ordered_groups = sorted(groups)
    while len(chosen) < count:
        added = False
        for key in ordered_groups:
            rows = groups[key]
            if depth < len(rows):
                chosen.append(rows[depth])
                added = True
                if len(chosen) >= count:
                    break
        if not added:
            break
        depth += 1
    multi = next(
        (
            row
            for row in samples
            if row.policy_supervision
            and row.selection_mode in {SelectionMode.ORDERED_SEQUENCE, SelectionMode.UNORDERED_UNIQUE_SUBSET}
        ),
        None,
    )
    if multi is not None and all(row is not multi for row in chosen):
        chosen[-1] = multi
    return chosen


def seed_everything(seed: int = 20260718) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
