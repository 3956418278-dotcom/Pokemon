from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from models.static_card_adapter import StaticCardEmbeddingAdapter
from training.export_card_embeddings import export_card_artifacts
from training.validate_card_artifacts import validate_artifact_directories


TYPE_IDS = {"attack": 3, "ability": 4, "card_effect": 5}
SCHEMA = {
    "schema_version": "static_card_corpus_v2",
    "vocab": {
        "detail_type": {
            "<PAD>": 0,
            "<MASK>": 1,
            "<UNK>": 2,
            "ATTACK": 3,
            "ABILITY": 4,
            "CARD_EFFECT": 5,
        }
    },
}


class _FakeDataset:
    def __init__(self) -> None:
        self.rows = [
            {
                "index": 0,
                "card_id": "10",
                "details": [
                    _detail(0, 0, "attack", 101, "First Hit"),
                    _detail(0, 1, "ability", None, "First Ability"),
                ],
            },
            {
                "index": 1,
                "card_id": "11",
                "details": [
                    _detail(1, 0, "card_effect", None, "Trainer Effect"),
                ],
            },
            {
                "index": 2,
                "card_id": "12",
                "details": [
                    _detail(2, 0, "ability", None, "Second Ability"),
                    _detail(2, 1, "attack", 202, "Second Hit"),
                    _detail(2, 2, "card_effect", None, "Rule Effect"),
                ],
            },
        ]
        self.card_id_to_index = {row["card_id"]: row["index"] for row in self.rows}

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


def _detail(
    card_index: int,
    local_index: int,
    detail_type: str,
    attack_id: int | None,
    move_name: str,
) -> dict[str, Any]:
    return {
        "card_index": card_index,
        "local_detail_index": local_index,
        "source_row_index": 100 + card_index * 10 + local_index,
        "source_line_number": 102 + card_index * 10 + local_index,
        "detail_type": detail_type,
        "detail_subtype": "rule" if detail_type == "card_effect" else None,
        "move_name": move_name,
        "attack_id": attack_id,
    }


def _collate(items: list[dict[str, Any]]) -> dict[str, Any]:
    width = max(len(item["details"]) for item in items)
    detail_dim = 4
    tokens = torch.zeros(len(items), width, detail_dim)
    masks = torch.zeros(len(items), width)
    types = torch.zeros(len(items), width, dtype=torch.long)
    summaries = torch.zeros(len(items), detail_dim)
    metadata = []
    for row_index, item in enumerate(items):
        summaries[row_index] = float(item["index"] + 1)
        for local_index, detail in enumerate(item["details"]):
            value = float(item["index"] * 10 + local_index + 1)
            tokens[row_index, local_index] = value
            masks[row_index, local_index] = 1.0
            types[row_index, local_index] = TYPE_IDS[detail["detail_type"]]
        metadata.append({"card_id": item["card_id"], "details": item["details"]})
    return {
        "card_index": torch.tensor([item["index"] for item in items]),
        "card_ids": [item["card_id"] for item in items],
        "detail_metadata": metadata,
        "fake_summaries": summaries,
        "fake_tokens": tokens,
        "fake_masks": masks,
        "fake_types": types,
    }


class _FakeEncoder:
    def __call__(self, batch: dict[str, Any], return_details: bool = False) -> SimpleNamespace:
        assert return_details
        return SimpleNamespace(
            card_summary=batch["fake_summaries"],
            detail_tokens=batch["fake_tokens"],
            detail_mask=batch["fake_masks"],
            detail_type_ids=batch["fake_types"],
        )


def _export(tmp_path: Path, name: str, batch_size: int) -> Path:
    dataset = _FakeDataset()
    checkpoint_path = tmp_path / "checkpoint.pt"
    if not checkpoint_path.exists():
        torch.save({"encoder": {}}, checkpoint_path)
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(exist_ok=True)
    (cache_dir / "card_id_to_index.json").write_text(
        json.dumps(dataset.card_id_to_index), encoding="utf-8"
    )
    (cache_dir / "cards.json").write_text(
        json.dumps([{"card_id": row["card_id"]} for row in dataset.rows]), encoding="utf-8"
    )
    flat_details = [
        {"card_id": row["card_id"], **detail}
        for row in dataset.rows
        for detail in row["details"]
    ]
    (cache_dir / "details.json").write_text(json.dumps(flat_details), encoding="utf-8")
    offsets = [0]
    for row in dataset.rows:
        offsets.append(offsets[-1] + len(row["details"]))
    (cache_dir / "detail_offsets.json").write_text(json.dumps(offsets), encoding="utf-8")
    output_dir = tmp_path / name
    export_card_artifacts(
        dataset=dataset,
        encoder=_FakeEncoder(),
        schema=SCHEMA,
        checkpoint={"config": {"model": {"detail_token_dim": 4}}},
        checkpoint_path=checkpoint_path,
        cache_dir=cache_dir,
        output_dir=output_dir,
        batch_size=batch_size,
        collate_fn=_collate,
    )
    return output_dir


def test_v2_export_is_flat_aligned_and_batch_size_independent(tmp_path: Path) -> None:
    one = _export(tmp_path, "batch_one", 1)
    many = _export(tmp_path, "batch_many", 3)

    for filename in (
        "card_embeddings.pt",
        "detail_embeddings.pt",
        "detail_offsets.pt",
        "detail_type_ids.pt",
    ):
        assert torch.equal(torch.load(one / filename), torch.load(many / filename))
    metadata_one = json.loads((one / "detail_metadata.json").read_text(encoding="utf-8"))
    metadata_many = json.loads((many / "detail_metadata.json").read_text(encoding="utf-8"))
    assert metadata_one == metadata_many
    assert [row["global_detail_index"] for row in metadata_one] == list(range(6))
    assert [row["local_detail_index"] for row in metadata_one] == [0, 1, 0, 0, 1, 2]
    assert [row["source_row"] for row in metadata_one] == [100, 101, 110, 120, 121, 122]
    assert [row["detail_type"] for row in metadata_one] == [
        "attack",
        "ability",
        "card_effect",
        "ability",
        "attack",
        "card_effect",
    ]
    assert torch.load(one / "detail_offsets.pt").tolist() == [0, 2, 3, 6]
    assert torch.load(one / "detail_type_ids.pt").tolist() == [3, 4, 5, 4, 3, 5]
    assert tuple(torch.load(one / "detail_embeddings.pt").shape) == (6, 4)
    manifest = json.loads((one / "artifact_manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "static_card_artifacts_v2"
    assert manifest["card_count"] == 3
    assert manifest["detail_count"] == 6
    assert manifest["max_details_per_card"] == 3
    assert manifest["detail_type_vocab"] == {
        "padding": 0,
        "attack": 3,
        "ability": 4,
        "card_effect": 5,
    }
    assert set(manifest["files"]) == {
        "card_embeddings.pt",
        "detail_embeddings.pt",
        "detail_offsets.pt",
        "detail_type_ids.pt",
        "detail_metadata.json",
        "card_id_to_index.json",
    }


def test_v2_adapter_validates_manifest_and_densifies_only_in_memory(tmp_path: Path) -> None:
    output_dir = _export(tmp_path, "v2", 2)
    adapter = StaticCardEmbeddingAdapter.from_artifacts(output_dir)
    assert tuple(adapter.flat_detail_tokens.shape) == (6, 4)
    assert adapter.detail_offsets.tolist() == [0, 2, 3, 6]
    assert adapter.max_detail_count == 3

    features = adapter.forward_features(torch.tensor([12, 999, 10]))
    assert features.known_mask.tolist() == [1.0, 0.0, 1.0]
    assert tuple(features.detail_tokens.shape) == (3, 3, 4)
    assert features.detail_mask.tolist() == [
        [1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
    ]
    assert features.detail_type_ids.tolist() == [
        [2, 1, 3],
        [0, 0, 0],
        [1, 2, 0],
    ]
    # Compatibility view includes the unknown/padding embedding row, as v1 did.
    assert tuple(adapter.detail_tokens.shape) == (4, 3, 4)

    torch.save(torch.zeros(3, 3), output_dir / "card_detail_masks.pt")
    with pytest.raises(ValueError, match="legacy files"):
        StaticCardEmbeddingAdapter.from_artifacts(output_dir)


def test_legacy_artifact_loading_requires_explicit_opt_in(tmp_path: Path) -> None:
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    torch.save(torch.zeros(2, 4), legacy / "card_embeddings.pt")
    torch.save(torch.ones(2, 2, 4), legacy / "card_detail_tokens.pt")
    torch.save(torch.ones(2, 2), legacy / "card_detail_masks.pt")
    torch.save(torch.ones(2, 2, dtype=torch.long), legacy / "card_detail_type_ids.pt")
    (legacy / "card_id_to_index.json").write_text(
        json.dumps({"10": 0, "11": 1}), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="disabled by default"):
        StaticCardEmbeddingAdapter.from_artifacts(legacy)
    adapter = StaticCardEmbeddingAdapter.from_artifacts(legacy, allow_legacy_v1=True)
    features = adapter.forward_features(torch.tensor([10, 999]))
    assert features.known_mask.tolist() == [1.0, 0.0]
    assert tuple(features.detail_tokens.shape) == (2, 2, 4)


def test_export_refuses_to_mix_v2_with_legacy_artifacts(tmp_path: Path) -> None:
    output_dir = tmp_path / "mixed"
    output_dir.mkdir()
    torch.save(torch.zeros(1), output_dir / "card_detail_tokens.pt")
    with pytest.raises(FileExistsError, match="legacy files"):
        export_card_artifacts(
            dataset=_FakeDataset(),
            encoder=_FakeEncoder(),
            schema=SCHEMA,
            checkpoint={"config": {"model": {"detail_token_dim": 4}}},
            checkpoint_path=tmp_path / "missing-checkpoint.pt",
            cache_dir=tmp_path / "cache",
            output_dir=output_dir,
            batch_size=2,
            collate_fn=_collate,
        )


def test_full_artifact_validator_checks_source_rows_and_batch_independence(tmp_path: Path) -> None:
    one = _export(tmp_path, "batch_one", 1)
    many = _export(tmp_path, "batch_many", 3)
    report = validate_artifact_directories(
        tmp_path / "cache",
        [one, many],
        expected_embedding_dim=4,
    )
    assert report["success"] is True
    assert report["artifacts"][0]["detail_count"] == 6
    assert report["comparisons"][0]["offsets_exact"] is True
    assert report["comparisons"][0]["metadata_exact"] is True
