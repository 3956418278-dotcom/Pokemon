from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Iterator

import torch

from decision_agent_v1.contracts.action_contract import sha256_file

from .cache import CachedDecisionCorpus


class StateUpgradeCorpus:
    """V1 tensors plus a hash-checked, row-aligned V2 feature overlay."""

    def __init__(self, cache_dir: str | Path) -> None:
        self.cache_dir = Path(cache_dir)
        self.manifest = json.loads((self.cache_dir / "manifest.json").read_text(encoding="utf-8"))
        if self.manifest.get("status") != "complete":
            raise RuntimeError("V2 cache manifest is not complete")
        self.base = CachedDecisionCorpus(self.manifest["base_cache_dir"])

    def class_counts(self, split: str) -> list[int]:
        return self.base.class_counts(split)

    def iter_batches(
        self, split: str, batch_size: int, *, shuffle: bool = False, seed: int = 0
    ) -> Iterator[dict[str, Any]]:
        entries = list(self.manifest["splits"][split]["shards"])
        rng = random.Random(seed)
        if shuffle:
            rng.shuffle(entries)
        base_entries = {Path(row["tensor_file"]).name: row for row in self.base.shard_entries(split)}
        for entry in entries:
            overlay_path = self.cache_dir / entry["overlay_file"]
            if sha256_file(overlay_path) != entry["overlay_sha256"]:
                raise RuntimeError(f"V2 overlay hash mismatch: {overlay_path}")
            overlay = torch.load(overlay_path, map_location="cpu")
            base_name = Path(entry["base_tensor_file"]).name
            base_entry = base_entries[base_name]
            base_tensor = torch.load(self.base.cache_dir / base_entry["tensor_file"], map_location="cpu")
            metadata = json.loads((self.base.cache_dir / base_entry["metadata_file"]).read_text(encoding="utf-8"))["records"]
            if len(metadata) != int(entry["decision_count"]):
                raise RuntimeError("V2 overlay row count mismatch")
            indices = list(range(len(metadata)))
            if shuffle:
                rng.shuffle(indices)
            for offset in range(0, len(indices), batch_size):
                selected = indices[offset: offset + batch_size]
                index = torch.tensor(selected, dtype=torch.long)
                batch = {key: value.index_select(0, index) for key, value in base_tensor.items()}
                batch.update({key: value.index_select(0, index) for key, value in overlay.items()})
                rows = [metadata[i] for i in selected]
                inverse = torch.tensor([1.0 / max(int(row["episode_decision_count"]), 1) for row in rows])
                batch["episode_weight"] = inverse / inverse.sum() * len(rows)
                batch["target_sequences"] = [tuple(row["target_sequence"]) for row in rows]
                batch["target_equivalence_groups"] = [tuple(row["target_equivalence_groups"]) for row in rows]
                from decision_agent_v1.contracts.schemas import SelectionMode
                batch["selection_modes"] = [SelectionMode(row["selection_mode"]) for row in rows]
                batch["metadata"] = rows
                yield batch
