from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np


VALIDATION_SCHEMA_VERSION = "static_card_artifact_validation_v3"
ARTIFACT_SCHEMA_VERSION = "static_card_artifacts_v3"
DATA_SCHEMA_VERSION = "static_card_v3"
SOURCE_MAPPING_ROLE = "PREPROCESS_CARD_RECORD_ROW"
EXPORT_MAPPING_ROLE = "EMBEDDING_TENSOR_ROW"
REQUIRED_FILES = {
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
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"{path} contains a blank JSONL row at line {line_number}")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path} line {line_number} is not an object")
            rows.append(value)
    return rows


def _mapping(value: Any, path: Path) -> dict[str, int]:
    if not isinstance(value, dict):
        raise ValueError(f"mapping in {path} is not an object")
    result = {str(card_id): int(index) for card_id, index in value.items()}
    if set(result.values()) != set(range(len(result))):
        raise ValueError(f"mapping in {path} is not contiguous and one-to-one")
    return result


def _reference_rows(details: list[dict[str, Any]], mapping: dict[str, int]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    all_rows: list[dict[str, Any]] = []
    cross_rows: list[dict[str, Any]] = []
    for global_detail_index, detail in enumerate(details):
        card_id = str(detail["card_id"])
        for reference in detail.get("text_references") or []:
            row = {
                "card_id": card_id,
                "card_index": mapping[card_id],
                "global_detail_index": global_detail_index,
                **dict(reference),
            }
            all_rows.append(row)
            payload = reference.get("payload") if isinstance(reference.get("payload"), dict) else {}
            if any(str(value) != card_id for value in payload.get("matching_target_card_ids") or []):
                cross_rows.append(row)
    return all_rows, cross_rows


def _verify_sentencepiece(path: Path, expected_sha256: str) -> dict[str, Any]:
    if _sha256(path) != expected_sha256:
        raise ValueError("exported sentencepiece.model SHA256 does not match its manifest")
    try:
        import sentencepiece as sentencepiece
    except ImportError as exc:  # pragma: no cover - formal Kaggle image supplies it.
        raise RuntimeError("sentencepiece is required by the formal v3 integrity gate") from exc
    processor = sentencepiece.SentencePieceProcessor()
    if not processor.load(str(path)):
        raise ValueError("exported sentencepiece.model is not loadable")
    if processor.pad_id() != 0 or processor.unk_id() != 1:
        raise ValueError("exported SentencePiece special IDs do not match the frozen contract")
    if processor.bos_id() != -1 or processor.eos_id() != -1:
        raise ValueError("exported SentencePiece unexpectedly has BOS/EOS IDs")
    mask_id = int(processor.piece_to_id("[MASK_TEXT]"))
    if mask_id < 0 or processor.id_to_piece(mask_id) != "[MASK_TEXT]":
        raise ValueError("exported SentencePiece has no [MASK_TEXT] user-defined symbol")
    return {"loadable": True, "piece_count": int(processor.get_piece_size()), "mask_text_id": mask_id}


def validate_artifact_directory(
    cache_dir: Path,
    artifact_dir: Path,
    *,
    expected_embedding_dim: int = 128,
) -> dict[str, Any]:
    """Reconstruct every v3 exported row against the five canonical cache files."""

    cache_dir, artifact_dir = Path(cache_dir), Path(artifact_dir)
    cards = _load_json(cache_dir / "cards.json")
    details = _load_json(cache_dir / "details.json")
    source_offsets = np.asarray(_load_json(cache_dir / "detail_offsets.json"), dtype=np.int64)
    source_mapping_path = cache_dir / "card_id_to_index.json"
    source_mapping = _mapping(_load_json(source_mapping_path), source_mapping_path)
    preprocess = _load_json(cache_dir / "preprocess_manifest.json")
    if preprocess.get("schema_version") != DATA_SCHEMA_VERSION:
        raise ValueError("canonical cache is not static_card_v3")
    source_mapping_record = (preprocess.get("mappings") or {}).get("card_id_to_index") or {}
    if source_mapping_record.get("role") != SOURCE_MAPPING_ROLE:
        raise ValueError("canonical mapping role is not PREPROCESS_CARD_RECORD_ROW")
    if source_mapping_record.get("sha256") != _sha256(source_mapping_path):
        raise ValueError("canonical mapping lineage SHA256 is invalid")

    manifest = _load_json(artifact_dir / "artifact_manifest.json")
    if manifest.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise ValueError("artifact manifest is not static_card_artifacts_v3")
    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != REQUIRED_FILES:
        raise ValueError(f"artifact file set differs from the frozen v3 contract: {set(files or {})}")
    for filename, record in files.items():
        path = artifact_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"artifact manifest file is missing: {filename}")
        if path.stat().st_size != int(record["size_bytes"]):
            raise ValueError(f"artifact size mismatch for {filename}")
        if _sha256(path) != str(record["sha256"]):
            raise ValueError(f"artifact SHA256 mismatch for {filename}")

    mapping_path = artifact_dir / "card_id_to_index.json"
    mapping = _mapping(_load_json(mapping_path), mapping_path)
    mapping_lineage = (manifest.get("mappings") or {}).get("card_id_to_index") or {}
    if mapping_lineage.get("role") != EXPORT_MAPPING_ROLE:
        raise ValueError("derived mapping role is not EMBEDDING_TENSOR_ROW")
    if mapping_lineage.get("sha256") != _sha256(mapping_path):
        raise ValueError("derived mapping SHA256 is invalid")
    source_lineage = mapping_lineage.get("source") or {}
    if source_lineage.get("role") != SOURCE_MAPPING_ROLE:
        raise ValueError("derived mapping does not identify its canonical source role")
    if source_lineage.get("sha256") != _sha256(source_mapping_path):
        raise ValueError("derived mapping source SHA256 does not match canonical mapping")
    if mapping != source_mapping:
        raise ValueError("derived embedding-row mapping differs from canonical card rows")

    cards_array = np.load(artifact_dir / "base_card_summary.npy", allow_pickle=False)
    details_array = np.load(artifact_dir / "independent_detail_tokens.npy", allow_pickle=False)
    offsets = np.load(artifact_dir / "detail_offsets.npy", allow_pickle=False)
    if cards_array.shape != (len(cards), expected_embedding_dim):
        raise ValueError("base_card_summary.npy has the wrong shape")
    if details_array.shape != (len(details), expected_embedding_dim):
        raise ValueError("independent_detail_tokens.npy has the wrong shape")
    if offsets.dtype.kind not in "iu" or not np.array_equal(offsets.astype(np.int64), source_offsets):
        raise ValueError("detail_offsets.npy does not exactly reproduce canonical offsets")
    if not np.isfinite(cards_array).all() or not np.isfinite(details_array).all():
        raise ValueError("exported embeddings contain NaN or infinity")

    card_metadata = _load_jsonl(artifact_dir / "card_metadata.jsonl")
    detail_metadata = _load_jsonl(artifact_dir / "detail_metadata.jsonl")
    if len(card_metadata) != len(cards) or len(detail_metadata) != len(details):
        raise ValueError("metadata JSONL row counts do not match canonical tables")
    for index, (source, exported) in enumerate(zip(cards, card_metadata)):
        if int(exported.get("embedding_index", -1)) != index:
            raise ValueError(f"card metadata row {index} has the wrong embedding_index")
        if str(exported.get("card_id")) != str(source.get("card_id")):
            raise ValueError(f"card metadata row {index} has the wrong card_id")
    for index, (source, exported) in enumerate(zip(details, detail_metadata)):
        if int(exported.get("embedding_index", -1)) != index:
            raise ValueError(f"detail metadata row {index} has the wrong embedding_index")
        for field in ("card_id", "global_detail_index", "local_detail_index", "detail_type"):
            if exported.get(field) != source.get(field):
                raise ValueError(f"detail metadata row {index} field {field} differs from canonical")

    expected_references, expected_cross = _reference_rows(details, mapping)
    if _load_jsonl(artifact_dir / "text_references.jsonl") != expected_references:
        raise ValueError("text_references.jsonl cannot be reconstructed from canonical details")
    if _load_jsonl(artifact_dir / "cross_card_references.jsonl") != expected_cross:
        raise ValueError("cross_card_references.jsonl cannot be reconstructed from canonical details")

    schema = _load_json(artifact_dir / "card_feature_schema.json")
    encoder_config = _load_json(artifact_dir / "encoder_config.json")
    field_vocabs = _load_json(artifact_dir / "field_vocabs.json")
    if schema.get("schema_version") != DATA_SCHEMA_VERSION:
        raise ValueError("card_feature_schema.json is not static_card_v3")
    if encoder_config.get("data_schema_version") != DATA_SCHEMA_VERSION:
        raise ValueError("encoder_config.json has the wrong data schema")
    if field_vocabs.get("card_field_slots") != schema.get("card_field_slots"):
        raise ValueError("field_vocabs card field ordering differs from the feature schema")
    tokenizer = manifest.get("tokenizer") or {}
    tokenizer_report = _verify_sentencepiece(
        artifact_dir / "sentencepiece.model",
        str(tokenizer.get("export_sha256") or ""),
    )
    if tokenizer.get("model_sha256") != tokenizer.get("export_sha256"):
        raise ValueError("exported tokenizer bytes differ from the training tokenizer lineage")

    checkpoint = manifest.get("checkpoint") or {}
    checkpoint_path = Path(str(checkpoint.get("path") or ""))
    if not checkpoint_path.is_file() or _sha256(checkpoint_path) != checkpoint.get("sha256"):
        raise ValueError("artifact source checkpoint is unavailable or has the wrong SHA256")
    return {
        "artifact_dir": str(artifact_dir),
        "card_count": len(cards),
        "detail_count": len(details),
        "mapping_roles_and_lineage_valid": True,
        "offsets_match_source": True,
        "metadata_reconstructable": True,
        "references_reconstructable": True,
        "manifest_hashes_valid": True,
        "embeddings_finite": True,
        "checkpoint_load_source_valid": True,
        "tokenizer": tokenizer_report,
        "integrity_passed": True,
    }


def validate_artifact_directories(
    cache_dir: Path,
    artifact_dirs: list[Path],
    *,
    expected_embedding_dim: int = 128,
    absolute_tolerance: float = 5e-6,
    relative_tolerance: float = 1e-6,
) -> dict[str, Any]:
    if not artifact_dirs:
        raise ValueError("at least one artifact directory is required")
    reports = [
        validate_artifact_directory(cache_dir, path, expected_embedding_dim=expected_embedding_dim)
        for path in artifact_dirs
    ]
    reference = Path(artifact_dirs[0])
    reference_cards = np.load(reference / "base_card_summary.npy", allow_pickle=False)
    reference_details = np.load(reference / "independent_detail_tokens.npy", allow_pickle=False)
    reference_offsets = np.load(reference / "detail_offsets.npy", allow_pickle=False)
    reference_metadata = {
        name: (reference / name).read_bytes()
        for name in REQUIRED_FILES
        if name.endswith(".jsonl") or name.endswith(".json") or name == "sentencepiece.model"
    }
    comparisons: list[dict[str, Any]] = []
    for value in artifact_dirs[1:]:
        candidate = Path(value)
        cards = np.load(candidate / "base_card_summary.npy", allow_pickle=False)
        details = np.load(candidate / "independent_detail_tokens.npy", allow_pickle=False)
        offsets = np.load(candidate / "detail_offsets.npy", allow_pickle=False)
        if not np.array_equal(reference_offsets, offsets):
            raise ValueError(f"detail offsets differ between {reference} and {candidate}")
        for name, payload in reference_metadata.items():
            if (candidate / name).read_bytes() != payload:
                raise ValueError(f"derived metadata/tokenizer {name} differs for {candidate}")
        if not np.allclose(reference_cards, cards, atol=absolute_tolerance, rtol=relative_tolerance):
            raise ValueError(f"card embeddings differ beyond tolerance for {candidate}")
        if not np.allclose(reference_details, details, atol=absolute_tolerance, rtol=relative_tolerance):
            raise ValueError(f"detail embeddings differ beyond tolerance for {candidate}")
        comparisons.append(
            {
                "artifact_dir": str(candidate),
                "offsets_exact": True,
                "metadata_and_tokenizer_exact": True,
                "card_embedding_max_abs_diff": float(np.max(np.abs(reference_cards - cards))),
                "detail_embedding_max_abs_diff": float(np.max(np.abs(reference_details - details))),
            }
        )
    return {
        "schema_version": VALIDATION_SCHEMA_VERSION,
        "success": True,
        "cache_dir": str(cache_dir),
        "artifacts": reports,
        "comparisons": comparisons,
        "absolute_tolerance": absolute_tolerance,
        "relative_tolerance": relative_tolerance,
        "integrity_gate": "pass",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate static CardEncoder v3 derived artifacts")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = validate_artifact_directories(args.cache_dir, args.artifact_dir)
    payload = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="", flush=True)


if __name__ == "__main__":
    main()
