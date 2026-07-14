from __future__ import annotations

import argparse
import base64
import hashlib
import inspect
import json
import shutil
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.card_dataset import CardDataset, collate_cards
from data.card_preprocessing import SCHEMA_VERSION
from models.card_encoder import CardEncoder


ARTIFACT_SCHEMA_VERSION = "static_card_artifacts_v3"
MODEL_VERSION = "card_encoder_v3"
TRAINING_SCHEMA_VERSION = "static_card_training_v3"
DATA_SCHEMA_VERSION = SCHEMA_VERSION
SOURCE_MAPPING_ROLE = "PREPROCESS_CARD_RECORD_ROW"
EXPORT_MAPPING_ROLE = "EMBEDDING_TENSOR_ROW"
DERIVED_FILENAMES = (
    "base_card_summary.npy",
    "independent_detail_tokens.npy",
    "detail_offsets.npy",
    "card_metadata.jsonl",
    "detail_metadata.jsonl",
    "text_references.jsonl",
    "cross_card_references.jsonl",
    "card_id_to_index.json",
    "field_vocabs.json",
    "sentencepiece.model",
    "encoder_config.json",
    "card_feature_schema.json",
)


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path) -> dict[str, Any]:
    return {"sha256": _sha256(path), "size_bytes": path.stat().st_size}


def _read_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            count += 1
    return count


def _data_version(cache_dir: Path) -> dict[str, Any]:
    names = (
        "cards.json",
        "details.json",
        "detail_offsets.json",
        "card_id_to_index.json",
        "preprocess_manifest.json",
    )
    files = {}
    digest = hashlib.sha256()
    for name in names:
        path = cache_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"canonical v3 cache is missing {name}")
        files[name] = _file_record(path)
        digest.update(name.encode("utf-8"))
        digest.update(files[name]["sha256"].encode("ascii"))
    return {"path": str(cache_dir), "sha256": digest.hexdigest(), "files": files}


def _source_mapping_record(cache_dir: Path) -> dict[str, Any]:
    manifest = _read_json(cache_dir / "preprocess_manifest.json")
    mapping_path = cache_dir / "card_id_to_index.json"
    declared = (manifest.get("mappings") or {}).get("card_id_to_index") or {}
    if declared.get("role") != SOURCE_MAPPING_ROLE:
        raise ValueError(
            "v3 preprocess manifest must declare card_id_to_index role "
            f"{SOURCE_MAPPING_ROLE}"
        )
    actual_sha = _sha256(mapping_path)
    if declared.get("sha256") != actual_sha:
        raise ValueError("v3 preprocess mapping SHA256 does not match the canonical file")
    return {
        "role": SOURCE_MAPPING_ROLE,
        "path": str(mapping_path),
        "sha256": actual_sha,
    }


def _sentencepiece_source(
    checkpoint: dict[str, Any],
    schema: dict[str, Any],
) -> tuple[bytes, dict[str, Any]]:
    contract = schema.get("tokenizer_contract")
    if not isinstance(contract, dict) or contract.get("tokenizer_type") != "sentencepiece":
        raise ValueError("formal v3 export requires a real sentencepiece tokenizer contract")
    expected_sha = str(contract.get("model_sha256") or "")
    if not expected_sha:
        raise ValueError("sentencepiece tokenizer contract has no model_sha256")

    payload: bytes | None = None
    artifacts = checkpoint.get("tokenizer_artifacts")
    if isinstance(artifacts, dict):
        raw = artifacts.get("sentencepiece_model_bytes")
        if isinstance(raw, bytes):
            payload = raw
        elif isinstance(raw, str):
            payload = base64.b64decode(raw)
    if payload is None:
        source_path = Path(str(contract.get("model_path") or ""))
        if not source_path.is_file():
            raise FileNotFoundError(
                "checkpoint did not embed sentencepiece bytes and tokenizer model_path is unavailable"
            )
        payload = source_path.read_bytes()
    actual_sha = hashlib.sha256(payload).hexdigest()
    if actual_sha != expected_sha:
        raise ValueError(
            f"sentencepiece model SHA mismatch: contract={expected_sha} actual={actual_sha}"
        )
    return payload, contract


def _assert_safe_output_directory(output_dir: Path) -> None:
    manifest_path = output_dir / "artifact_manifest.json"
    existing = [name for name in DERIVED_FILENAMES if (output_dir / name).exists()]
    if manifest_path.exists():
        manifest = _read_json(manifest_path)
        if manifest.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
            raise FileExistsError(f"refusing to overwrite a non-v3 artifact directory: {output_dir}")
    elif existing:
        raise FileExistsError(
            f"refusing to mix unmanifested files into v3 artifact directory {output_dir}: {existing}"
        )


def _build_encoder(checkpoint: dict[str, Any], schema: dict[str, Any]) -> torch.nn.Module:
    model_config = checkpoint.get("config", {}).get("model", {})
    candidates = {
        "embedding_dim": int(model_config.get("embedding_dim", 128)),
        "detail_token_dim": int(model_config.get("detail_token_dim", 128)),
        "num_heads": int(model_config.get("attention_heads", 4)),
        "transformer_layers": int(model_config.get("transformer_layers", 2)),
        "ffn_dim": int(model_config.get("ffn_dim", 256)),
        "dropout": float(model_config.get("dropout", 0.0)),
    }
    parameters = inspect.signature(CardEncoder).parameters
    encoder = CardEncoder(
        schema,
        **{name: value for name, value in candidates.items() if name in parameters},
    )
    state = checkpoint.get("encoder")
    if not isinstance(state, dict):
        raise ValueError("v3 checkpoint has no encoder state")
    encoder.load_state_dict(state)
    encoder.eval()
    return encoder


def _metadata_rows(
    cards: list[dict[str, Any]],
    details: list[dict[str, Any]],
    mapping: dict[str, int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    card_rows = [
        {"embedding_index": index, **dict(card)}
        for index, card in enumerate(cards)
    ]
    detail_rows = [
        {"embedding_index": index, **dict(detail)}
        for index, detail in enumerate(details)
    ]
    text_references: list[dict[str, Any]] = []
    cross_references: list[dict[str, Any]] = []
    for global_detail_index, detail in enumerate(details):
        card_id = str(detail["card_id"])
        card_index = mapping[card_id]
        for reference in detail.get("text_references") or []:
            row = {
                "card_id": card_id,
                "card_index": card_index,
                "global_detail_index": global_detail_index,
                **dict(reference),
            }
            text_references.append(row)
            payload = reference.get("payload") if isinstance(reference.get("payload"), dict) else {}
            target_card_ids = [str(value) for value in payload.get("matching_target_card_ids") or []]
            if any(value != card_id for value in target_card_ids):
                cross_references.append(row)
    return card_rows, detail_rows, text_references, cross_references


def export_card_artifacts(
    *,
    dataset: Any,
    encoder: Any,
    schema: dict[str, Any],
    checkpoint: dict[str, Any],
    checkpoint_path: Path,
    cache_dir: Path,
    output_dir: Path,
    batch_size: int = 256,
    collate_fn: Callable[[list[Any]], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Export the frozen v3 derived views; the canonical cache is never rewritten."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if schema.get("schema_version") != DATA_SCHEMA_VERSION:
        raise ValueError("v3 export requires a static_card_v3 feature schema")
    if checkpoint.get("training_schema_version") != TRAINING_SCHEMA_VERSION:
        raise ValueError("v3 export requires a static_card_training_v3 checkpoint")
    if checkpoint.get("data_schema_version") != DATA_SCHEMA_VERSION:
        raise ValueError("checkpoint data schema is not static_card_v3")
    if not checkpoint_path.is_file() or _sha256(checkpoint_path) == "":
        raise FileNotFoundError(checkpoint_path)
    lineage = checkpoint.get("lineage")
    if not isinstance(lineage, dict):
        raise ValueError("v3 checkpoint is missing lineage")
    cache_dir = Path(cache_dir)
    preprocess_path = cache_dir / "preprocess_manifest.json"
    if lineage.get("preprocess_manifest_sha256") != _sha256(preprocess_path):
        raise ValueError("checkpoint preprocessing lineage does not match the export cache")

    output_dir = Path(output_dir)
    _assert_safe_output_directory(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cards = _read_json(cache_dir / "cards.json")
    details = _read_json(cache_dir / "details.json")
    source_offsets = [int(value) for value in _read_json(cache_dir / "detail_offsets.json")]
    mapping = {
        str(card_id): int(index)
        for card_id, index in _read_json(cache_dir / "card_id_to_index.json").items()
    }
    if len(dataset) != len(cards) or dataset.card_id_to_index != mapping:
        raise ValueError("export dataset does not match the canonical v3 card mapping")
    if source_offsets[-1] != len(details):
        raise ValueError("canonical v3 detail offsets do not terminate at detail_count")

    collator = collate_fn or (lambda rows: collate_cards(rows, schema))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)
    card_rows: dict[int, torch.Tensor] = {}
    detail_rows: dict[int, torch.Tensor] = {}
    with torch.no_grad():
        for batch in loader:
            output = encoder(batch, return_details=True)
            summaries = output.card_summary.detach().cpu()
            independent = output.independent_detail_tokens.detach().cpu()
            mask = output.detail_mask.detach().cpu().bool()
            card_indices = batch["card_index"].detach().cpu().long()
            global_indices = batch["detail_global_indices"].detach().cpu().long()
            if summaries.dim() != 2 or summaries.shape[1] != 128:
                raise ValueError("CardEncoder returned invalid v3 card_summary shape")
            if independent.dim() != 3 or independent.shape[-1] != 128:
                raise ValueError("CardEncoder returned invalid v3 independent_detail_tokens shape")
            if mask.shape != independent.shape[:2] or global_indices.shape != mask.shape:
                raise ValueError("v3 detail tensors, masks and global indices are not aligned")
            for batch_row, card_index_value in enumerate(card_indices.tolist()):
                card_index = int(card_index_value)
                if card_index in card_rows:
                    raise ValueError(f"duplicate exported card index {card_index}")
                card_rows[card_index] = summaries[batch_row]
                for slot in torch.nonzero(mask[batch_row], as_tuple=False).flatten().tolist():
                    global_index = int(global_indices[batch_row, slot])
                    if global_index in detail_rows:
                        raise ValueError(f"duplicate exported detail index {global_index}")
                    detail_rows[global_index] = independent[batch_row, slot]

    if set(card_rows) != set(range(len(cards))):
        raise ValueError("v3 export did not reconstruct every canonical card row")
    if set(detail_rows) != set(range(len(details))):
        raise ValueError("v3 export did not reconstruct every canonical detail row")
    card_table = torch.stack([card_rows[index] for index in range(len(cards))]).numpy().astype(np.float32)
    detail_table = torch.stack([detail_rows[index] for index in range(len(details))]).numpy().astype(np.float32)
    offsets = np.asarray(source_offsets, dtype=np.int64)
    if not np.isfinite(card_table).all() or not np.isfinite(detail_table).all():
        raise ValueError("v3 export contains a non-finite embedding")

    paths = {name: output_dir / name for name in DERIVED_FILENAMES}
    np.save(paths["base_card_summary.npy"], card_table, allow_pickle=False)
    np.save(paths["independent_detail_tokens.npy"], detail_table, allow_pickle=False)
    np.save(paths["detail_offsets.npy"], offsets, allow_pickle=False)
    metadata = _metadata_rows(cards, details, mapping)
    row_counts = {
        "card_metadata.jsonl": _write_jsonl(paths["card_metadata.jsonl"], metadata[0]),
        "detail_metadata.jsonl": _write_jsonl(paths["detail_metadata.jsonl"], metadata[1]),
        "text_references.jsonl": _write_jsonl(paths["text_references.jsonl"], metadata[2]),
        "cross_card_references.jsonl": _write_jsonl(paths["cross_card_references.jsonl"], metadata[3]),
    }
    shutil.copyfile(cache_dir / "card_id_to_index.json", paths["card_id_to_index.json"])
    _write_json(
        paths["field_vocabs.json"],
        {
            "card_field_slots": schema["card_field_slots"],
            "field_to_value_group": schema["field_to_value_group"],
            "value_vocabs": schema["value_vocabs"],
            "reference_card_field_vocab": schema["reference_card_field_vocab"],
            "reference_type_vocab": schema["reference_type_vocab"],
            "detail_type_vocab": schema["vocab"]["detail_type"],
            "detail_subtype_vocab": schema["vocab"]["detail_subtype"],
            "damage_value_vocab": schema["vocab"]["damage_value"],
            "damage_mode_vocab": schema["vocab"]["damage_mode"],
        },
    )
    sentencepiece_bytes, tokenizer_contract = _sentencepiece_source(checkpoint, schema)
    paths["sentencepiece.model"].write_bytes(sentencepiece_bytes)
    _write_json(
        paths["encoder_config.json"],
        {
            "model_version": MODEL_VERSION,
            "data_schema_version": DATA_SCHEMA_VERSION,
            "training_schema_version": TRAINING_SCHEMA_VERSION,
            "model": checkpoint["config"]["model"],
            "tokenizer_contract": tokenizer_contract,
        },
    )
    _write_json(paths["card_feature_schema.json"], schema)

    source_mapping = _source_mapping_record(cache_dir)
    export_mapping = {
        "role": EXPORT_MAPPING_ROLE,
        "path": "card_id_to_index.json",
        "sha256": _sha256(paths["card_id_to_index.json"]),
        "source": source_mapping,
    }
    if export_mapping["sha256"] != source_mapping["sha256"]:
        raise ValueError("derived embedding-row mapping differs from the canonical preprocess mapping")
    file_records = {name: _file_record(path) for name, path in paths.items()}
    manifest = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "artifact_role": "DERIVED_FORMAL_RUN_EXPORT",
        "canonical_source_mutated": False,
        "representations": {
            "base_card_summary": "CardEncoder.card_summary under static_card_v3",
            "independent_detail_tokens": (
                "DetailTransformer outputs before CardTransformer contextualization"
            ),
            "storage": "flat_details_with_offsets",
        },
        "card_count": int(card_table.shape[0]),
        "detail_count": int(detail_table.shape[0]),
        "card_embedding_dim": int(card_table.shape[1]),
        "detail_embedding_dim": int(detail_table.shape[1]),
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": _sha256(checkpoint_path),
            "stage": checkpoint.get("stage"),
            "lineage": lineage,
        },
        "data": _data_version(cache_dir),
        "mappings": {"card_id_to_index": export_mapping},
        "tokenizer": {
            **tokenizer_contract,
            "export_path": "sentencepiece.model",
            "export_sha256": file_records["sentencepiece.model"]["sha256"],
        },
        "row_counts": row_counts,
        "files": file_records,
    }
    _write_json(output_dir / "artifact_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Export frozen static CardEncoder v3 artifacts")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("artifacts/card_data_v3"))
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    checkpoint = _torch_load(args.checkpoint)
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint must be a mapping")
    schema = checkpoint["schema"]
    dataset = CardDataset.from_cache(args.cache_dir)
    dataset.schema = schema
    encoder = _build_encoder(checkpoint, schema)
    manifest = export_card_artifacts(
        dataset=dataset,
        encoder=encoder,
        schema=schema,
        checkpoint=checkpoint,
        checkpoint_path=args.checkpoint,
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
    )
    print(
        f"exported {manifest['card_count']} cards and {manifest['detail_count']} independent details "
        f"to {args.output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
