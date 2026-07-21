from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any, Iterator

import torch

from decision_agent_v1.contracts.action_contract import sha256_file
from decision_agent_v1.contracts.schemas import DecisionSampleV1, SelectionMode


TENSOR_KEYS = (
    "card_index",
    "card_owner",
    "card_zone",
    "card_position",
    "card_dynamic",
    "card_mask",
    "global_features",
    "history_features",
    "option_type",
    "option_select_type",
    "option_context",
    "option_owner",
    "option_area",
    "option_position",
    "option_card_index",
    "option_numeric",
    "option_card_token_index",
    "option_equivalence_group",
    "original_option_index",
    "option_mask",
    "value_target",
    "policy_sample_mask",
)


def canonical_json_hash(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def schema_descriptor(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    source_paths = [
        root / "decision_agent_v1/contracts/schemas.py",
        root / "decision_agent_v1/adapters/observation_adapter.py",
        root / "decision_agent_v1/adapters/replay_adapter.py",
        root / "decision_agent_v1/data/collate.py",
    ]
    return {
        "schema_version": config["cache"]["schema_version"],
        "decision_sample_fields": [field.name for field in fields(DecisionSampleV1)],
        "tensor_keys": list(TENSOR_KEYS),
        "source_hashes": {str(path.relative_to(root)): sha256_file(path) for path in source_paths},
        "visibility_sources": ["observation.current", "observation.logs", "observation.select"],
        "forbidden_sources": ["visualize", "visualize.current", "complete opponent hand/deck", "hidden prize identity"],
    }


def cache_identity(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    contract_path = root / config["data"]["action_contract_path"]
    vocab_path = root / config["data"]["card_vocab_path"]
    schema = schema_descriptor(root, config)
    return {
        "schema_hash": canonical_json_hash(schema),
        "action_contract_hash": sha256_file(contract_path),
        "card_vocabulary_hash": sha256_file(vocab_path),
        "adapter_hash": canonical_json_hash(schema["source_hashes"]),
        "schema": schema,
    }


def record_metadata(sample: DecisionSampleV1) -> dict[str, Any]:
    return {
        "episode_id": sample.episode_id,
        "source_date": sample.source_date,
        "agent_index": sample.agent_index,
        "decision_index": sample.decision_index,
        "step": sample.step,
        "turn": sample.turn,
        "turn_action_count": sample.turn_action_count,
        "select_type": sample.global_state.select_type,
        "select_context": sample.global_state.select_context,
        "min_count": sample.min_count,
        "max_count": sample.max_count,
        "selection_mode": sample.selection_mode.value,
        "terminal_outcome": sample.terminal_outcome.name,
        "episode_decision_count": sample.episode_decision_count,
        "policy_supervision": sample.policy_supervision,
        "policy_mask_reason": sample.policy_mask_reason,
        "target_sequence": list(sample.selected_option_indices),
        "target_equivalence_groups": list(sample.selected_equivalence_groups),
        "history_token_count": len(sample.history.recent_events),
        "visibility_sources": list(sample.visibility_sources),
    }


def write_shard(
    split_dir: Path,
    shard_index: int,
    samples: list[DecisionSampleV1],
    collated: dict[str, Any],
    episode_ids: list[str],
) -> dict[str, Any]:
    split_dir.mkdir(parents=True, exist_ok=True)
    stem = f"shard_{shard_index:05d}"
    tensor_path = split_dir / f"{stem}.pt"
    metadata_path = split_dir / f"{stem}.json"
    if tensor_path.exists() or metadata_path.exists():
        raise FileExistsError(f"refusing to overwrite existing shard {stem}")
    tensors = {key: collated[key].cpu() for key in TENSOR_KEYS}
    metadata = {
        "schema_version": "policy_value_cache_shard_v1",
        "tensor_file": tensor_path.name,
        "decision_count": len(samples),
        "episode_ids": episode_ids,
        "records": [record_metadata(sample) for sample in samples],
    }
    tensor_tmp = tensor_path.with_suffix(".pt.incomplete")
    metadata_tmp = metadata_path.with_suffix(".json.incomplete")
    torch.save(tensors, tensor_tmp)
    metadata_tmp.write_text(json.dumps(metadata, ensure_ascii=False) + "\n", encoding="utf-8")
    tensor_tmp.replace(tensor_path)
    metadata_tmp.replace(metadata_path)
    return {
        "tensor_file": str(tensor_path.relative_to(split_dir.parent)),
        "metadata_file": str(metadata_path.relative_to(split_dir.parent)),
        "tensor_sha256": sha256_file(tensor_path),
        "metadata_sha256": sha256_file(metadata_path),
        "decision_count": len(samples),
        "episode_ids": episode_ids,
    }


class CachedDecisionCorpus:
    def __init__(self, cache_dir: str | Path, expected_identity: dict[str, Any] | None = None) -> None:
        self.cache_dir = Path(cache_dir)
        self.manifest = json.loads((self.cache_dir / "manifest.json").read_text(encoding="utf-8"))
        if self.manifest.get("status") != "complete":
            raise RuntimeError("cache manifest is not complete")
        if expected_identity is not None:
            for key in ("schema_hash", "action_contract_hash", "card_vocabulary_hash", "adapter_hash"):
                if self.manifest.get(key) != expected_identity.get(key):
                    raise RuntimeError(f"cache identity mismatch for {key}")
        self._validated_shards: set[tuple[str, str]] = set()

    def shard_entries(self, split: str) -> list[dict[str, Any]]:
        return list(self.manifest["splits"][split]["shards"])

    def iter_batches(
        self,
        split: str,
        batch_size: int,
        *,
        shuffle: bool = False,
        seed: int = 0,
    ) -> Iterator[dict[str, Any]]:
        entries = self.shard_entries(split)
        rng = random.Random(seed)
        if shuffle:
            rng.shuffle(entries)
        for entry in entries:
            tensor_path = self.cache_dir / entry["tensor_file"]
            metadata_path = self.cache_dir / entry["metadata_file"]
            validation_key = (str(tensor_path), str(metadata_path))
            if validation_key not in self._validated_shards:
                if sha256_file(tensor_path) != entry["tensor_sha256"]:
                    raise RuntimeError(f"tensor shard hash mismatch: {tensor_path}")
                if sha256_file(metadata_path) != entry["metadata_sha256"]:
                    raise RuntimeError(f"metadata shard hash mismatch: {metadata_path}")
                self._validated_shards.add(validation_key)
            tensors = torch.load(tensor_path, map_location="cpu")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            records = metadata["records"]
            indices = list(range(len(records)))
            if shuffle:
                rng.shuffle(indices)
            for offset in range(0, len(indices), batch_size):
                selected = indices[offset : offset + batch_size]
                index_tensor = torch.tensor(selected, dtype=torch.long)
                batch = {key: value.index_select(0, index_tensor) for key, value in tensors.items()}
                rows = [records[index] for index in selected]
                inverse_length = torch.tensor(
                    [1.0 / max(int(row["episode_decision_count"]), 1) for row in rows]
                )
                batch["episode_weight"] = inverse_length / inverse_length.sum() * len(rows)
                batch["target_sequences"] = [tuple(row["target_sequence"]) for row in rows]
                batch["target_equivalence_groups"] = [
                    tuple(row["target_equivalence_groups"]) for row in rows
                ]
                batch["selection_modes"] = [SelectionMode(row["selection_mode"]) for row in rows]
                batch["metadata"] = rows
                max_target = max((len(row["target_sequence"]) for row in rows), default=0)
                batch["target_sequence_mask"] = torch.tensor(
                    [
                        [position < len(row["target_sequence"]) for position in range(max_target)]
                        for row in rows
                    ],
                    dtype=torch.bool,
                )
                yield batch

    def class_counts(self, split: str) -> list[int]:
        counts = [0, 0, 0]
        for batch in self.iter_batches(split, 4096):
            for label in batch["value_target"].tolist():
                counts[int(label)] += 1
        return counts
