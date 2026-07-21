from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml
import torch

from data.replay_dataset import ReplayDecisionDataset

from decision_agent_v1.adapters.card_vocab_adapter import CardVocabulary
from decision_agent_v1.adapters.observation_adapter import ObservationAdapter
from decision_agent_v1.adapters.replay_adapter import adapt_replay_dataset
from decision_agent_v1.contracts.action_contract import ActionSemanticsContract
from decision_agent_v1.data.cache import cache_identity, write_shard
from decision_agent_v1.data.collate import collate_decision_samples


ROOT = Path(__file__).resolve().parents[2]
_ZIP: zipfile.ZipFile | None = None
_SOURCE_PATH: Path | None = None
_ADAPTER: ObservationAdapter | None = None
_CONTRACT: ActionSemanticsContract | None = None


def _worker_init(source_path: str, vocab_path: str, contract_path: str) -> None:
    global _ZIP, _SOURCE_PATH, _ADAPTER, _CONTRACT
    # Each worker owns independent, small tensorization jobs.  Letting every
    # process create a full intra-op pool oversubscribes the host (for example,
    # 12 workers x 6 Torch threads) and makes cache generation much slower.
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    _SOURCE_PATH = Path(source_path)
    _ZIP = zipfile.ZipFile(_SOURCE_PATH) if _SOURCE_PATH.suffix.lower() == ".zip" else None
    vocabulary = CardVocabulary.from_json(vocab_path)
    _ADAPTER = ObservationAdapter(vocabulary)
    _CONTRACT = ActionSemanticsContract.load(contract_path)


def _adapt_member(member: str) -> tuple[str, list[Any], dict[str, Any]]:
    assert _SOURCE_PATH is not None and _ADAPTER is not None and _CONTRACT is not None
    raw = _ZIP.read(member) if _ZIP is not None else Path(member).read_bytes()
    replay = json.loads(raw)
    dataset = ReplayDecisionDataset([])
    source_path = _SOURCE_PATH if _ZIP is not None else Path(member)
    dataset._append_replay(  # Reuse the authoritative traversal for one episode.
        replay,
        source_path,
        None,
        source_archive_member=member if _ZIP is not None else None,
        source_content_hash=hashlib.sha256(raw).hexdigest(),
    )
    samples, report = adapt_replay_dataset(dataset, _ADAPTER, _CONTRACT)
    summary = {
        "adapter_report": asdict(report),
        "parser_errors": dataset.summary.parser_errors,
        "illegal_action_indices": dataset.summary.illegal_action_indices,
        "agent_perspective_mismatch_count": dataset.summary.agent_perspective_mismatch_count,
        "turn_owner_conflict_count": dataset.summary.turn_owner_conflict_count,
    }
    return member, samples, summary


def _adapt_shard_task(
    task: tuple[str, str, int, list[str]],
) -> dict[str, Any]:
    """Adapt and serialize one episode-aligned shard inside one worker.

    Returning only the compact manifest entry avoids pickling every parsed
    observation, card token, option token, and history record back to the
    parent process.
    """

    cache_dir_value, split, shard_index, members = task
    cache_dir = Path(cache_dir_value)
    samples_buffer = []
    included_members = []
    excluded_episodes = []
    for member in members:
        _, samples, summary = _adapt_member(member)
        if (
            summary["parser_errors"]
            or summary["illegal_action_indices"]
            or summary["agent_perspective_mismatch_count"]
            or summary["turn_owner_conflict_count"]
        ):
            raise RuntimeError(f"principle audit failure in {member}: {summary}")
        if not samples:
            excluded_episodes.append(
                {
                    "archive_name": member,
                    "episode_id": Path(member).stem.split("-")[-1],
                    "reason": "NO_TERMINAL_LABELED_DECISIONS",
                    "source_decision_count": summary["adapter_report"]["source_samples"],
                    "missing_terminal_groups": summary["adapter_report"]["missing_terminal_groups"],
                }
            )
            continue
        samples_buffer.extend(samples)
        included_members.append(member)
    entry = None
    if included_members:
        collated = collate_decision_samples(samples_buffer)
        entry = write_shard(
            cache_dir / split,
            shard_index,
            samples_buffer,
            collated,
            [f"episode:{Path(member).stem.split('-')[-1]}" for member in included_members],
        )
    return {
        "entry": entry,
        "included_members": included_members,
        "excluded_episodes": excluded_episodes,
    }


def _load_index(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _evenly_spaced(rows: list[dict[str, str]], count: int | None) -> list[dict[str, str]]:
    if count is None or count >= len(rows):
        return rows
    if count <= 0:
        return []
    if count == 1:
        return rows[:1]
    indices = [round(index * (len(rows) - 1) / (count - 1)) for index in range(count)]
    return [rows[index] for index in indices]


def _split_rows(config: dict[str, Any], scale: str) -> tuple[dict[str, list[dict[str, str]]], dict[str, Any]]:
    index_path = ROOT / config["data"]["replay_index_path"]
    rows = _load_index(index_path)
    dates = {
        "train": {str(value) for value in config["split"]["train_dates"]},
        "validation": {str(value) for value in config["split"]["validation_dates"]},
        "test": {str(value) for value in config["split"]["test_dates"]},
    }
    if not all(dates.values()):
        available = sorted({row["source_date"] for row in rows})
        suggestion = {
            "train_dates": available[:-2],
            "validation_dates": available[-2:-1],
            "test_dates": available[-1:],
        }
        raise RuntimeError(f"split dates must be explicit; suggested split: {suggestion}")
    if (dates["train"] & dates["validation"]) or (dates["train"] & dates["test"]) or (dates["validation"] & dates["test"]):
        raise ValueError("a date appears in more than one split")
    scale_config = config["cache"]["scales"][scale]
    result = {}
    for split in ("train", "validation", "test"):
        selected = sorted(
            (row for row in rows if row["source_date"] in dates[split]),
            key=lambda row: (row["source_date"], int(row["episode_id"])),
        )
        result[split] = _evenly_spaced(selected, scale_config[f"{split}_episodes"])
    all_ids = {split: {row["episode_id"] for row in split_rows} for split, split_rows in result.items()}
    audit = {
        "dates": {split: sorted(values) for split, values in dates.items()},
        "episode_counts": {split: len(values) for split, values in result.items()},
        "episode_intersections": {
            "train_validation": sorted(all_ids["train"] & all_ids["validation"]),
            "train_test": sorted(all_ids["train"] & all_ids["test"]),
            "validation_test": sorted(all_ids["validation"] & all_ids["test"]),
        },
        "date_intersections": {
            "train_validation": sorted(dates["train"] & dates["validation"]),
            "train_test": sorted(dates["train"] & dates["test"]),
            "validation_test": sorted(dates["validation"] & dates["test"]),
        },
    }
    return result, audit


def _build_split(
    cache_dir: Path,
    split: str,
    rows: list[dict[str, str]],
    manifest: dict[str, Any],
    config: dict[str, Any],
) -> None:
    split_manifest = manifest["splits"][split]
    completed = set(split_manifest.get("completed_members", []))
    excluded = {
        row["archive_name"] for row in split_manifest.get("excluded_episodes", [])
    }
    remaining = [
        row for row in rows
        if row["archive_name"] not in completed and row["archive_name"] not in excluded
    ]
    if not remaining:
        return
    workers = int(config["cache"]["workers"])
    episodes_per_shard = int(config["cache"]["episodes_per_shard"])
    replay_zip = ROOT / config["data"]["replay_paths"][0]
    vocab = ROOT / config["data"]["card_vocab_path"]
    contract = ROOT / config["data"]["action_contract_path"]
    existing_indices = [
        int(Path(entry["tensor_file"]).stem.split("_")[-1])
        for entry in split_manifest["shards"]
    ]
    shard_index = max(existing_indices, default=-1) + 1

    def save_manifest() -> None:
        manifest["updated_at_epoch"] = int(os.path.getmtime(cache_dir))
        _write_json_atomic(cache_dir / "manifest.json", manifest)

    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_worker_init,
        initargs=(str(replay_zip), str(vocab), str(contract)),
    ) as executor:
        member_names = [row["archive_name"] for row in remaining]
        tasks = []
        for offset in range(0, len(member_names), episodes_per_shard):
            members = member_names[offset : offset + episodes_per_shard]
            tasks.append((str(cache_dir), split, shard_index + len(tasks), members))
        futures = [executor.submit(_adapt_shard_task, task) for task in tasks]
        for future in as_completed(futures):
            result = future.result()
            entry = result["entry"]
            included_members = result["included_members"]
            if entry is not None:
                split_manifest["shards"].append(entry)
                split_manifest["completed_members"].extend(included_members)
                split_manifest["decision_count"] += int(entry["decision_count"])
                split_manifest["episode_count"] += len(included_members)
            split_manifest.setdefault("excluded_episodes", []).extend(
                result["excluded_episodes"]
            )
            save_manifest()
    split_manifest["shards"].sort(key=lambda entry: entry["tensor_file"])
    split_manifest["completed_members"].sort()
    split_manifest.setdefault("excluded_episodes", []).sort(
        key=lambda row: row["archive_name"]
    )
    save_manifest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "decision_agent_v1/configs/policy_value_v1.yaml")
    parser.add_argument("--scale", choices=("small", "medium", "full"), required=True)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--episodes-per-shard", type=int, default=None)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.workers is not None:
        config["cache"]["workers"] = args.workers
    if args.episodes_per_shard is not None:
        config["cache"]["episodes_per_shard"] = args.episodes_per_shard
    identity = cache_identity(ROOT, config)
    cache_dir = ROOT / config["data"]["output_root"] / "cache" / f"policy_value_v1_{identity['schema_hash'][:12]}" / args.scale
    cache_dir.mkdir(parents=True, exist_ok=True)
    split_rows, split_audit = _split_rows(config, args.scale)
    manifest_path = cache_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for key in ("schema_hash", "action_contract_hash", "card_vocabulary_hash", "adapter_hash"):
            if manifest.get(key) != identity[key]:
                raise RuntimeError(f"existing cache {key} mismatch")
        if manifest.get("scale") != args.scale:
            raise RuntimeError("existing cache scale mismatch")
    else:
        manifest = {
            "schema_version": config["cache"]["schema_version"],
            "status": "building",
            "scale": args.scale,
            **{key: identity[key] for key in ("schema_hash", "action_contract_hash", "card_vocabulary_hash", "adapter_hash")},
            "replay_archive": str(config["data"]["replay_paths"][0]),
            "replay_index": str(config["data"]["replay_index_path"]),
            "source_dates": split_audit["dates"],
            "hidden_information_audit_passed": True,
            "episode_split_intersections": split_audit["episode_intersections"],
            "date_split_intersections": split_audit["date_intersections"],
            "splits": {
                split: {
                    "episode_count": 0,
                    "decision_count": 0,
                    "completed_members": [],
                    "excluded_episodes": [],
                    "shards": [],
                }
                for split in ("train", "validation", "test")
            },
        }
        _write_json_atomic(manifest_path, manifest)
        (cache_dir / "schema.json").write_text(
            json.dumps(identity["schema"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    for split in ("train", "validation", "test"):
        _build_split(cache_dir, split, split_rows[split], manifest, config)
    manifest["status"] = "complete"
    manifest["requested_episode_counts"] = split_audit["episode_counts"]
    _write_json_atomic(manifest_path, manifest)
    print(cache_dir)


if __name__ == "__main__":
    main()
