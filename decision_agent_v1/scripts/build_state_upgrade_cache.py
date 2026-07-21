from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import torch
import yaml

from data.replay_dataset import ReplayDecisionDataset

from decision_agent_v1.adapters.card_vocab_adapter import CardVocabulary
from decision_agent_v1.adapters.observation_adapter import ObservationAdapter
from decision_agent_v1.adapters.replay_adapter import adapt_replay_dataset
from decision_agent_v1.contracts.action_contract import ActionSemanticsContract, sha256_file
from decision_agent_v1.state_upgrade.collate import V2_TENSOR_KEYS, collate_state_upgrade
from decision_agent_v1.state_upgrade.deck_prior import DeckPrior
from decision_agent_v1.state_upgrade.features import (
    build_state_upgrade_features,
    event_decision_ages,
    next_public_cards,
)


ROOT = Path(__file__).resolve().parents[2]
_ZIP: zipfile.ZipFile | None = None
_PRIOR: DeckPrior | None = None
_VOCAB: CardVocabulary | None = None
_ADAPTER: ObservationAdapter | None = None
_CONTRACT: ActionSemanticsContract | None = None
_MEMBER_BY_EPISODE: dict[str, str] = {}


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _init_worker(zip_path: str, prior_path: str, vocab_path: str, contract_path: str, member_map: dict[str, str]) -> None:
    global _ZIP, _PRIOR, _VOCAB, _ADAPTER, _CONTRACT, _MEMBER_BY_EPISODE
    torch.set_num_threads(1)
    _ZIP = zipfile.ZipFile(zip_path)
    _PRIOR = DeckPrior.load(prior_path)
    _VOCAB = CardVocabulary.from_json(vocab_path)
    _ADAPTER = ObservationAdapter(_VOCAB)
    _CONTRACT = ActionSemanticsContract.load(contract_path)
    _MEMBER_BY_EPISODE = member_map


def _decks_from_actions(replay: dict[str, Any]) -> list[list[int]]:
    decks: list[list[int]] = [[], []]
    for step in replay.get("steps") or []:
        if not isinstance(step, list):
            continue
        for agent, row in enumerate(step[:2]):
            action = row.get("action") if isinstance(row, dict) else None
            if not decks[agent] and isinstance(action, list) and len(action) == 60 and all(isinstance(x, int) for x in action):
                decks[agent] = [int(x) for x in action]
        if all(decks):
            break
    if not all(len(deck) == 60 for deck in decks):
        raise RuntimeError("complete initial deck actions are required for both agents")
    return decks


def _episode_rows(episode_id: str) -> list[tuple[Any, Any, list[int], list[int]]]:
    assert _ZIP is not None and _PRIOR is not None and _VOCAB is not None and _ADAPTER is not None and _CONTRACT is not None
    member = _MEMBER_BY_EPISODE[episode_id]
    raw = _ZIP.read(member)
    replay = json.loads(raw)
    decks = _decks_from_actions(replay)
    dataset = ReplayDecisionDataset([])
    dataset._append_replay(
        replay, Path("replays.zip"), None,
        source_archive_member=member,
        source_content_hash=hashlib.sha256(raw).hexdigest(),
    )
    samples, report = adapt_replay_dataset(dataset, _ADAPTER, _CONTRACT)
    if report.missing_terminal_groups or dataset.summary.parser_errors or dataset.summary.illegal_action_indices:
        raise RuntimeError(f"V2 adaptation principle failure for {member}")
    raw_by_key = {(row.agent_index, row.step_index): row for row in dataset.samples if row.action_target is not None}
    paired = [(sample, raw_by_key[(sample.agent_index, sample.step)].memory_after) for sample in samples]
    targets = next_public_cards(paired)
    decision_ages = event_decision_ages(paired)
    return [
        (
            sample,
            build_state_upgrade_features(
                sample,
                memory,
                decks[sample.agent_index],
                decks[1 - sample.agent_index],
                _PRIOR,
                _VOCAB,
                next_public_card_id=targets[(sample.agent_index, sample.decision_index)],
                recent_event_decision_ages=decision_ages[(sample.agent_index, sample.decision_index)],
            ),
        )
        for sample, memory in paired
    ]


def _build_shard(task: tuple[str, str, str, list[dict[str, Any]]]) -> dict[str, Any]:
    cache_dir_value, split, base_tensor_file, metadata = task
    episode_ids = sorted({str(row["episode_id"]).split(":")[-1] for row in metadata})
    lookup = {}
    for episode_id in episode_ids:
        for sample, features in _episode_rows(episode_id):
            lookup[(sample.episode_id, sample.agent_index, sample.decision_index)] = features
    ordered = []
    for row in metadata:
        key = (row["episode_id"], int(row["agent_index"]), int(row["decision_index"]))
        if key not in lookup:
            raise RuntimeError(f"missing V2 overlay row {key}")
        ordered.append(lookup[key])
    tensors = collate_state_upgrade(ordered)
    overlay_rel = str(Path(split) / (Path(base_tensor_file).stem + "_v2.pt"))
    output = Path(cache_dir_value) / overlay_rel
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".pt.incomplete")
    torch.save(tensors, temporary)
    temporary.replace(output)
    return {
        "base_tensor_file": base_tensor_file,
        "overlay_file": overlay_rel,
        "overlay_sha256": sha256_file(output),
        "decision_count": len(ordered),
        "episode_count": len(episode_ids),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "decision_agent_v1/configs/policy_value_v2.yaml")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--max-shards-per-split", type=int, default=None)
    parser.add_argument("--tag", default="full")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    base_dir = ROOT / config["data"]["base_cache_dir"]
    base_manifest = json.loads((base_dir / "manifest.json").read_text(encoding="utf-8"))
    prior_path = ROOT / config["data"]["belief_output_dir"] / "deck_templates.json"
    prior = DeckPrior.load(prior_path)
    source_hashes = {
        name: sha256_file(ROOT / name)
        for name in (
            "decision_agent_v1/state_upgrade/features.py",
            "decision_agent_v1/state_upgrade/collate.py",
            "decision_agent_v1/state_upgrade/deck_prior.py",
            "data/game_memory.py",
        )
    }
    descriptor = {
        "schema_version": config["cache"]["schema_version"],
        "base_schema_hash": base_manifest["schema_hash"],
        "tensor_keys": list(V2_TENSOR_KEYS),
        "recent_event_count": int(config["state_upgrade"]["recent_event_count"]),
        "deck_prior_source": "train_dates_only",
        "archetype_clustering_version": "top_exact_fingerprint_v1",
        "belief_template_hash": prior.template_hash,
        "ledger_schema": {
            "summary": "two sanitized relative-owner rows + cumulative event_type[24] + current-turn event_type[24] + turn usage[4]",
            "per_card": "owner, known/ambiguous counts, observation/movement counts, first/last turn, currently-visible exact zone counts[12], exact-zone mask",
            "stale_zone_policy": "last public zone is historical only; exact current zone requires currently_visible",
        },
        "recent_event_schema": "type, relative player, public Card ID, source/target zone, turn distance, exact decision distance, within-turn positions, visibility/reverse masks",
        "self_deck_schema": "initial/known-out/remaining per Card ID; deck+face-down-prize remain one unknown multiset; remaining category summary",
        "source_hashes": source_hashes,
        "actor_visibility_sources": ["GameMemory public observations", "own initial deck action", "train-only deck prior"],
        "forbidden_actor_sources": ["complete opponent deck", "hidden prize identity", "validation/test deck prior"],
    }
    schema_hash = hashlib.sha256(json.dumps(descriptor, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    cache_dir = ROOT / config["data"]["output_root"] / "cache" / f"policy_value_v2_{schema_hash[:12]}" / args.tag
    cache_dir.mkdir(parents=True, exist_ok=True)
    index_rows = []
    import csv
    with (ROOT / config["data"]["replay_index_path"]).open(encoding="utf-8-sig", newline="") as handle:
        index_rows = list(csv.DictReader(handle))
    member_map = {str(row["episode_id"]): row["archive_name"] for row in index_rows}
    manifest_path = cache_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema_hash") != schema_hash:
            raise RuntimeError("existing V2 cache schema hash mismatch")
        manifest["status"] = "building"
    else:
        manifest = {
            "schema_version": config["cache"]["schema_version"],
            "status": "building",
            "schema_hash": schema_hash,
            "schema": descriptor,
            "base_cache_dir": str(base_dir),
            "base_schema_hash": base_manifest["schema_hash"],
            "action_contract_hash": base_manifest["action_contract_hash"],
            "card_vocabulary_hash": base_manifest["card_vocabulary_hash"],
            "belief_template_hash": prior.template_hash,
            "source_dates": base_manifest["source_dates"],
            "splits": {split: {"shards": [], "decision_count": 0, "episode_count": 0} for split in ("train", "validation", "test")},
        }
    tasks = []
    for split in ("train", "validation", "test"):
        completed = {
            row["base_tensor_file"] for row in manifest["splits"][split]["shards"]
        }
        entries = base_manifest["splits"][split]["shards"]
        if args.max_shards_per_split is not None:
            entries = entries[: args.max_shards_per_split]
        for entry in entries:
            if entry["tensor_file"] in completed:
                continue
            metadata = json.loads((base_dir / entry["metadata_file"]).read_text(encoding="utf-8"))["records"]
            tasks.append((str(cache_dir), split, entry["tensor_file"], metadata))
    workers = args.workers or int(config["cache"]["workers"])
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker,
        initargs=(
            str(ROOT / config["data"]["replay_paths"][0]), str(prior_path),
            str(ROOT / config["data"]["card_vocab_path"]),
            str(ROOT / config["data"]["action_contract_path"]), member_map,
        ),
    ) as executor:
        futures = [executor.submit(_build_shard, task) for task in tasks]
        for future in as_completed(futures):
            entry = future.result()
            split = Path(entry["overlay_file"]).parts[0]
            manifest["splits"][split]["shards"].append(entry)
            manifest["splits"][split]["decision_count"] += entry["decision_count"]
            manifest["splits"][split]["episode_count"] += entry["episode_count"]
            _write_json_atomic(manifest_path, manifest)
    for split in manifest["splits"]:
        manifest["splits"][split]["shards"].sort(key=lambda row: row["base_tensor_file"])
    manifest["status"] = "complete"
    (cache_dir / "schema.json").write_text(json.dumps(descriptor, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_json_atomic(manifest_path, manifest)
    print(cache_dir)


if __name__ == "__main__":
    main()
