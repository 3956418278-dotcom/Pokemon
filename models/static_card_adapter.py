from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


V2_SCHEMA_VERSION = "static_card_artifacts_v2"
V2_REQUIRED_FILES = {
    "card_embeddings.pt",
    "detail_embeddings.pt",
    "detail_offsets.pt",
    "detail_type_ids.pt",
    "detail_metadata.json",
    "card_id_to_index.json",
}
LEGACY_DETAIL_FILES = {
    "card_detail_tokens.pt",
    "card_detail_masks.pt",
    "card_detail_type_ids.pt",
    "card_detail_metadata.json",
    "card_embedding_metadata.json",
}
DETAIL_TYPE_NAMES = {1: "attack", 2: "ability", 3: "card_effect"}
DETAIL_TYPE_ALIASES = {
    "attack": "attack",
    "ability": "ability",
    "card_effect": "card_effect",
    "effect": "card_effect",
    "special_effect": "card_effect",
}


@dataclass
class StaticCardFeatureOutput:
    summary: torch.Tensor
    known_mask: torch.Tensor
    detail_tokens: torch.Tensor | None = None
    detail_mask: torch.Tensor | None = None
    detail_type_ids: torch.Tensor | None = None


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _extract_embedding_tensor(value: Any, path: Path) -> torch.Tensor:
    if isinstance(value, dict):
        tensor = value.get("embeddings")
        if tensor is None:
            tensor = value.get("card_embeddings")
        if tensor is None:
            raise KeyError(f"{path} does not contain embeddings")
        value = tensor
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{path} does not contain a tensor")
    return value


def _load_mapping(path: Path) -> dict[str, int]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "card_id_to_index" in raw:
        raw = raw["card_id_to_index"]
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain a card-id mapping")
    return {str(card_id): int(index) for card_id, index in raw.items()}


class StaticCardEmbeddingAdapter(nn.Module):
    """Map public card ids to frozen static summary/detail artifacts.

    V2 artifacts remain flat on disk and as registered module buffers. A dense
    detail table is created only for the requested card ids in
    :meth:`forward_features`. The legacy dense constructor remains supported for
    unit tests and in-memory callers; loading legacy files requires explicit
    opt-in.
    """

    def __init__(
        self,
        embedding_weight: torch.Tensor,
        card_id_to_index: dict[str, int],
        freeze: bool = True,
        detail_tokens: torch.Tensor | None = None,
        detail_mask: torch.Tensor | None = None,
        detail_type_ids: torch.Tensor | None = None,
        *,
        flat_detail_tokens: torch.Tensor | None = None,
        detail_offsets: torch.Tensor | None = None,
        flat_detail_type_ids: torch.Tensor | None = None,
        artifact_manifest: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        if embedding_weight.dim() != 2:
            raise ValueError("embedding_weight must have shape [num_cards, embedding_dim]")
        if detail_tokens is not None and flat_detail_tokens is not None:
            raise ValueError("dense and flat detail storage cannot both be provided")

        self.card_id_to_index = {str(key): int(value) for key, value in card_id_to_index.items()}
        expected_indices = set(range(int(embedding_weight.shape[0])))
        actual_indices = list(self.card_id_to_index.values())
        if (
            len(self.card_id_to_index) != int(embedding_weight.shape[0])
            or len(actual_indices) != len(set(actual_indices))
            or set(actual_indices) != expected_indices
        ):
            raise ValueError(
                "card_id_to_index must map every embedding row exactly once with contiguous indices"
            )
        self.artifact_manifest = artifact_manifest
        max_card_id = max(
            [int(key) for key in self.card_id_to_index if str(key).isdigit()] + [0]
        )
        index_lookup = torch.zeros(max_card_id + 1, dtype=torch.long)
        known_lookup = torch.zeros(max_card_id + 1, dtype=torch.bool)
        for card_id, index in self.card_id_to_index.items():
            if card_id.isdigit():
                index_lookup[int(card_id)] = int(index) + 1
                known_lookup[int(card_id)] = True
        padded_weight = torch.cat(
            [
                embedding_weight.new_zeros(1, embedding_weight.size(1)),
                embedding_weight.float(),
            ],
            dim=0,
        )
        self.embedding = nn.Embedding.from_pretrained(
            padded_weight,
            freeze=freeze,
            padding_idx=0,
        )

        dense_tokens: torch.Tensor | None = None
        dense_mask: torch.Tensor | None = None
        dense_types: torch.Tensor | None = None
        if detail_tokens is not None:
            if detail_tokens.dim() != 3:
                raise ValueError("detail_tokens must have shape [num_cards, max_details, detail_dim]")
            if int(detail_tokens.shape[0]) != int(embedding_weight.shape[0]):
                raise ValueError("dense detail card count does not match embedding card count")
            if detail_mask is not None and detail_mask.shape != detail_tokens.shape[:2]:
                raise ValueError("detail_mask shape does not match dense detail tokens")
            if detail_type_ids is not None and detail_type_ids.shape != detail_tokens.shape[:2]:
                raise ValueError("detail_type_ids shape does not match dense detail tokens")
            dense_tokens = torch.cat(
                [
                    detail_tokens.new_zeros(1, detail_tokens.size(1), detail_tokens.size(2)),
                    detail_tokens.float(),
                ],
                dim=0,
            )
            source_mask = (
                detail_mask.float()
                if detail_mask is not None
                else torch.ones(detail_tokens.shape[:2], dtype=torch.float32)
            )
            source_types = (
                detail_type_ids.long()
                if detail_type_ids is not None
                else torch.zeros(detail_tokens.shape[:2], dtype=torch.long)
            )
            dense_mask = torch.cat(
                [source_mask.new_zeros(1, source_mask.size(1)), source_mask],
                dim=0,
            )
            dense_types = torch.cat(
                [source_types.new_zeros(1, source_types.size(1)), source_types],
                dim=0,
            )

        flat_tokens: torch.Tensor | None = None
        flat_types: torch.Tensor | None = None
        offsets: torch.Tensor | None = None
        if flat_detail_tokens is not None:
            if detail_offsets is None or flat_detail_type_ids is None:
                raise ValueError("flat detail storage requires offsets and type ids")
            flat_tokens, offsets, flat_types = self._validate_flat_details(
                flat_detail_tokens,
                detail_offsets,
                flat_detail_type_ids,
                card_count=int(embedding_weight.shape[0]),
            )

        self.register_buffer("_dense_detail_tokens", dense_tokens, persistent=False)
        self.register_buffer("_dense_detail_mask", dense_mask, persistent=False)
        self.register_buffer("_dense_detail_type_ids", dense_types, persistent=False)
        self.register_buffer("_flat_detail_tokens", flat_tokens, persistent=False)
        self.register_buffer("_detail_offsets", offsets, persistent=False)
        self.register_buffer("_flat_detail_type_ids", flat_types, persistent=False)
        self.register_buffer("index_lookup", index_lookup, persistent=False)
        self.register_buffer("known_lookup", known_lookup, persistent=False)

        if dense_tokens is not None:
            self._max_detail_count = int(dense_tokens.shape[1])
        elif offsets is not None and offsets.numel() > 1:
            self._max_detail_count = int((offsets[1:] - offsets[:-1]).max().item())
        else:
            self._max_detail_count = 0

    @staticmethod
    def _validate_flat_details(
        detail_tokens: torch.Tensor,
        detail_offsets: torch.Tensor,
        detail_type_ids: torch.Tensor,
        *,
        card_count: int,
        allowed_type_ids: set[int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if detail_tokens.dim() != 2:
            raise ValueError("detail_embeddings must have shape [detail_count, detail_dim]")
        offsets = detail_offsets.long().flatten()
        types = detail_type_ids.long().flatten()
        if detail_offsets.dim() != 1:
            raise ValueError("detail_offsets must be one-dimensional")
        if detail_type_ids.dim() != 1:
            raise ValueError("detail_type_ids must be one-dimensional")
        if offsets.numel() != card_count + 1:
            raise ValueError(
                f"detail_offsets length {offsets.numel()} does not equal card_count + 1"
            )
        if int(offsets[0].item()) != 0:
            raise ValueError("detail_offsets must start at zero")
        if bool((offsets[1:] < offsets[:-1]).any().item()):
            raise ValueError("detail_offsets must be nondecreasing")
        detail_count = int(detail_tokens.shape[0])
        if int(offsets[-1].item()) != detail_count:
            raise ValueError("detail_offsets final value does not match detail count")
        if types.shape != (detail_count,):
            raise ValueError("detail_type_ids length does not match detail count")
        allowed_type_ids = allowed_type_ids or {1, 2, 3}
        if detail_count:
            actual_type_ids = {int(value) for value in types.detach().cpu().tolist()}
            if not actual_type_ids <= allowed_type_ids:
                raise ValueError(
                    f"detail_type_ids contain unsupported values: "
                    f"{sorted(actual_type_ids - allowed_type_ids)}"
                )
        if not torch.isfinite(detail_tokens).all():
            raise ValueError("detail_embeddings contain non-finite values")
        return detail_tokens.float(), offsets, types

    @classmethod
    def from_artifacts(
        cls,
        artifact_dir: str | Path,
        freeze: bool = True,
        *,
        allow_legacy_v1: bool = False,
    ) -> "StaticCardEmbeddingAdapter":
        artifact_dir = Path(artifact_dir)
        manifest_path = artifact_dir / "artifact_manifest.json"
        if manifest_path.exists():
            return cls._from_v2_artifacts(artifact_dir, manifest_path, freeze=freeze)
        v2_without_manifest = sorted(
            name for name in V2_REQUIRED_FILES - {"card_embeddings.pt", "card_id_to_index.json"}
            if (artifact_dir / name).exists()
        )
        if v2_without_manifest:
            raise ValueError(
                f"v2 artifact files exist without artifact_manifest.json in {artifact_dir}: "
                f"{v2_without_manifest}"
            )
        if not allow_legacy_v1:
            raise ValueError(
                "legacy static artifacts are disabled by default; pass "
                "allow_legacy_v1=True only for an explicit v1 rollback"
            )
        return cls._from_legacy_artifacts(artifact_dir, freeze=freeze)

    @classmethod
    def _from_v2_artifacts(
        cls,
        artifact_dir: Path,
        manifest_path: Path,
        *,
        freeze: bool,
    ) -> "StaticCardEmbeddingAdapter":
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != V2_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported static artifact schema: {manifest.get('schema_version')!r}"
            )
        mixed = sorted(name for name in LEGACY_DETAIL_FILES if (artifact_dir / name).exists())
        if mixed:
            raise ValueError(f"v2 artifact directory contains legacy files: {mixed}")
        missing = sorted(name for name in V2_REQUIRED_FILES if not (artifact_dir / name).is_file())
        if missing:
            raise FileNotFoundError(f"v2 artifact directory is incomplete; missing {missing}")
        file_records = manifest.get("files")
        if not isinstance(file_records, dict):
            raise ValueError("v2 manifest is missing file hashes")
        for name in sorted(V2_REQUIRED_FILES):
            record = file_records.get(name)
            if not isinstance(record, dict) or not record.get("sha256"):
                raise ValueError(f"v2 manifest has no hash for {name}")
            actual = _sha256(artifact_dir / name)
            if actual != str(record["sha256"]):
                raise ValueError(f"v2 artifact hash mismatch for {name}")

        weight_path = artifact_dir / "card_embeddings.pt"
        embedding_weight = _extract_embedding_tensor(_torch_load(weight_path), weight_path).float()
        mapping = _load_mapping(artifact_dir / "card_id_to_index.json")
        detail_tokens = _torch_load(artifact_dir / "detail_embeddings.pt")
        detail_offsets = _torch_load(artifact_dir / "detail_offsets.pt")
        detail_type_ids = _torch_load(artifact_dir / "detail_type_ids.pt")
        if not all(isinstance(value, torch.Tensor) for value in (detail_tokens, detail_offsets, detail_type_ids)):
            raise TypeError("v2 detail tensor files must each contain a tensor")
        card_count = int(embedding_weight.shape[0])
        raw_type_vocab = manifest.get("detail_type_vocab")
        if not isinstance(raw_type_vocab, dict):
            raise ValueError("v2 manifest is missing detail_type_vocab")
        source_type_vocab = {
            canonical: int(raw_type_vocab[canonical])
            for canonical in ("attack", "ability", "card_effect")
            if canonical in raw_type_vocab
        }
        if set(source_type_vocab) != {"attack", "ability", "card_effect"}:
            raise ValueError("v2 manifest detail_type_vocab is incomplete")
        if len(set(source_type_vocab.values())) != len(source_type_vocab):
            raise ValueError("v2 manifest detail_type_vocab ids must be unique")
        cls._validate_flat_details(
            detail_tokens,
            detail_offsets,
            detail_type_ids,
            card_count=card_count,
            allowed_type_ids=set(source_type_vocab.values()),
        )
        mapping_indices = list(mapping.values())
        if (
            len(mapping) != card_count
            or len(mapping_indices) != len(set(mapping_indices))
            or set(mapping_indices) != set(range(card_count))
        ):
            raise ValueError("v2 card_id_to_index does not match card embedding rows")
        detail_count = int(detail_tokens.shape[0])
        expected_manifest_values = {
            "card_count": card_count,
            "detail_count": detail_count,
            "card_embedding_dim": int(embedding_weight.shape[1]),
            "detail_embedding_dim": int(detail_tokens.shape[1]),
        }
        for key, actual in expected_manifest_values.items():
            if int(manifest.get(key, -1)) != actual:
                raise ValueError(
                    f"v2 manifest {key}={manifest.get(key)!r} does not match tensor value {actual}"
                )
        actual_max_details = int(
            (detail_offsets[1:] - detail_offsets[:-1]).max().item()
        ) if card_count else 0
        if int(manifest.get("max_details_per_card", -1)) != actual_max_details:
            raise ValueError(
                "v2 manifest max_details_per_card does not match detail_offsets"
            )
        source_to_runtime = {
            source_type_vocab["attack"]: 1,
            source_type_vocab["ability"]: 2,
            source_type_vocab["card_effect"]: 3,
        }
        runtime_detail_type_ids = torch.tensor(
            [source_to_runtime[int(value)] for value in detail_type_ids.tolist()],
            dtype=torch.long,
        )
        cls._validate_v2_metadata(
            artifact_dir / "detail_metadata.json",
            mapping,
            detail_offsets,
            runtime_detail_type_ids,
        )
        return cls(
            embedding_weight,
            mapping,
            freeze=freeze,
            flat_detail_tokens=detail_tokens,
            detail_offsets=detail_offsets,
            flat_detail_type_ids=runtime_detail_type_ids,
            artifact_manifest=manifest,
        )

    @staticmethod
    def _validate_v2_metadata(
        path: Path,
        mapping: dict[str, int],
        offsets: torch.Tensor,
        type_ids: torch.Tensor,
    ) -> None:
        rows = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(rows, list) or len(rows) != int(type_ids.numel()):
            raise ValueError("detail_metadata.json must be a flat list matching detail_count")
        reverse_mapping = {index: card_id for card_id, index in mapping.items()}
        for card_index in range(len(mapping)):
            start = int(offsets[card_index].item())
            end = int(offsets[card_index + 1].item())
            for local_index, global_index in enumerate(range(start, end)):
                row = rows[global_index]
                if not isinstance(row, dict):
                    raise ValueError(f"detail metadata row {global_index} is not a mapping")
                required = {
                    "global_detail_index",
                    "local_detail_index",
                    "card_index",
                    "card_id",
                    "source_row",
                    "detail_type",
                    "attack_id",
                    "move_name",
                    "subtype",
                }
                missing = sorted(required - set(row))
                if missing:
                    raise ValueError(f"detail metadata row {global_index} is missing {missing}")
                if int(row["global_detail_index"]) != global_index:
                    raise ValueError("detail metadata global indices do not match file row numbers")
                if int(row["local_detail_index"]) != local_index:
                    raise ValueError("detail metadata local indices do not match detail_offsets")
                if int(row["card_index"]) != card_index:
                    raise ValueError("detail metadata card indices do not match detail_offsets")
                if str(row["card_id"]) != reverse_mapping.get(card_index):
                    raise ValueError("detail metadata card ids do not match card_id_to_index")
                type_id = int(type_ids[global_index].item())
                expected_type = DETAIL_TYPE_NAMES.get(type_id)
                declared_type = DETAIL_TYPE_ALIASES.get(str(row["detail_type"]).lower())
                if expected_type is None or declared_type != expected_type:
                    raise ValueError("detail metadata types do not match detail_type_ids")
                if expected_type == "attack" and row.get("attack_id") is None:
                    raise ValueError("attack detail metadata must contain attack_id")

    @classmethod
    def _from_legacy_artifacts(
        cls,
        artifact_dir: Path,
        *,
        freeze: bool,
    ) -> "StaticCardEmbeddingAdapter":
        weight_path = artifact_dir / "card_embeddings.pt"
        mapping_path = artifact_dir / "card_id_to_index.json"
        if not weight_path.is_file() or not mapping_path.is_file():
            raise FileNotFoundError("legacy artifact directory is missing summary embeddings or id mapping")
        embedding_weight = _extract_embedding_tensor(_torch_load(weight_path), weight_path).float()
        mapping = _load_mapping(mapping_path)
        detail_tokens = (
            _torch_load(artifact_dir / "card_detail_tokens.pt")
            if (artifact_dir / "card_detail_tokens.pt").exists()
            else None
        )
        detail_mask = (
            _torch_load(artifact_dir / "card_detail_masks.pt")
            if (artifact_dir / "card_detail_masks.pt").exists()
            else None
        )
        detail_type_ids = (
            _torch_load(artifact_dir / "card_detail_type_ids.pt")
            if (artifact_dir / "card_detail_type_ids.pt").exists()
            else None
        )
        return cls(
            embedding_weight,
            mapping,
            freeze=freeze,
            detail_tokens=detail_tokens,
            detail_mask=detail_mask,
            detail_type_ids=detail_type_ids,
            artifact_manifest={"schema_version": "static_card_artifacts_v1_legacy"},
        )

    @property
    def embedding_dim(self) -> int:
        return int(self.embedding.embedding_dim)

    @property
    def max_detail_count(self) -> int:
        return self._max_detail_count

    @property
    def detail_offsets(self) -> torch.Tensor | None:
        return self._detail_offsets

    @property
    def flat_detail_tokens(self) -> torch.Tensor | None:
        return self._flat_detail_tokens

    @property
    def flat_detail_type_ids(self) -> torch.Tensor | None:
        return self._flat_detail_type_ids

    def _materialize_details(
        self,
        embedding_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        if self._dense_detail_tokens is not None:
            return (
                self._dense_detail_tokens[embedding_indices],
                self._dense_detail_mask[embedding_indices],
                self._dense_detail_type_ids[embedding_indices],
            )
        if self._flat_detail_tokens is None or self._detail_offsets is None:
            return None
        original_shape = tuple(embedding_indices.shape)
        flat_indices = embedding_indices.reshape(-1)
        count = int(flat_indices.numel())
        detail_dim = int(self._flat_detail_tokens.shape[1])
        tokens = self._flat_detail_tokens.new_zeros(
            (count, self._max_detail_count, detail_dim)
        )
        mask = self._flat_detail_tokens.new_zeros((count, self._max_detail_count))
        types = self._flat_detail_type_ids.new_zeros((count, self._max_detail_count))
        for output_row, padded_index in enumerate(flat_indices.tolist()):
            if padded_index <= 0:
                continue
            card_index = int(padded_index) - 1
            start = int(self._detail_offsets[card_index].item())
            end = int(self._detail_offsets[card_index + 1].item())
            length = end - start
            if length:
                tokens[output_row, :length] = self._flat_detail_tokens[start:end]
                mask[output_row, :length] = 1.0
                types[output_row, :length] = self._flat_detail_type_ids[start:end]
        return (
            tokens.reshape(*original_shape, self._max_detail_count, detail_dim),
            mask.reshape(*original_shape, self._max_detail_count),
            types.reshape(*original_shape, self._max_detail_count),
        )

    @property
    def detail_tokens(self) -> torch.Tensor | None:
        if self._dense_detail_tokens is not None:
            return self._dense_detail_tokens
        if self._flat_detail_tokens is None:
            return None
        indices = torch.arange(
            self.embedding.num_embeddings,
            device=self._flat_detail_tokens.device,
            dtype=torch.long,
        )
        materialized = self._materialize_details(indices)
        return materialized[0] if materialized is not None else None

    @property
    def detail_mask(self) -> torch.Tensor | None:
        if self._dense_detail_mask is not None:
            return self._dense_detail_mask
        if self._flat_detail_tokens is None:
            return None
        indices = torch.arange(
            self.embedding.num_embeddings,
            device=self._flat_detail_tokens.device,
            dtype=torch.long,
        )
        materialized = self._materialize_details(indices)
        return materialized[1] if materialized is not None else None

    @property
    def detail_type_ids(self) -> torch.Tensor | None:
        if self._dense_detail_type_ids is not None:
            return self._dense_detail_type_ids
        if self._flat_detail_tokens is None:
            return None
        indices = torch.arange(
            self.embedding.num_embeddings,
            device=self._flat_detail_tokens.device,
            dtype=torch.long,
        )
        materialized = self._materialize_details(indices)
        return materialized[2] if materialized is not None else None

    def forward(self, card_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedding_indices, known = self.lookup_indices(card_ids)
        return self.embedding(embedding_indices), known.float()

    def lookup_indices(self, card_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        safe_ids = card_ids.long().clamp_min(0)
        in_range = safe_ids < self.index_lookup.numel()
        lookup_ids = torch.where(in_range, safe_ids, torch.zeros_like(safe_ids))
        embedding_indices = self.index_lookup[lookup_ids]
        known = in_range & self.known_lookup[lookup_ids] & (card_ids.long() > 0)
        return embedding_indices, known

    def forward_features(self, card_ids: torch.Tensor) -> StaticCardFeatureOutput:
        embedding_indices, known = self.lookup_indices(card_ids)
        summary = self.embedding(embedding_indices)
        details = self._materialize_details(embedding_indices)
        if details is None:
            return StaticCardFeatureOutput(summary=summary, known_mask=known.float())
        detail_tokens, detail_mask, detail_type_ids = details
        return StaticCardFeatureOutput(
            summary=summary,
            known_mask=known.float(),
            detail_tokens=detail_tokens,
            detail_mask=detail_mask,
            detail_type_ids=detail_type_ids,
        )
