from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.card_dataset import CardDataset, collate_cards
from data.card_preprocessing import DEFAULT_CACHE_DIR
from models.card_encoder import CardEncoder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/card_embeddings.pt"))
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    schema = checkpoint["schema"]
    dataset = CardDataset.from_cache(args.cache_dir)
    encoder = CardEncoder(
        schema,
        embedding_dim=int(checkpoint["config"]["model"].get("embedding_dim", 128)),
        detail_token_dim=int(checkpoint["config"]["model"].get("detail_token_dim", checkpoint["config"]["model"].get("embedding_dim", 128))),
    )
    encoder.load_state_dict(checkpoint["encoder"])
    encoder.eval()
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=lambda items: collate_cards(items, schema))
    embeddings = []
    detail_tokens = []
    detail_masks = []
    detail_type_ids = []
    detail_metadata = []
    with torch.no_grad():
        for batch in loader:
            output = encoder(batch, return_details=True)
            embeddings.append(output.card_summary)
            detail_tokens.append(output.detail_tokens)
            detail_masks.append(output.detail_mask)
            detail_type_ids.append(output.detail_type_ids)
            for card_meta in batch["detail_metadata"]:
                rows = []
                detail_index = 0
                for attack in card_meta["attacks"]:
                    rows.append({"detail_index": detail_index, "detail_type": "attack", **attack})
                    detail_index += 1
                for ability in card_meta["abilities"]:
                    rows.append({"detail_index": detail_index, "detail_type": "ability", **ability})
                    detail_index += 1
                for effect in card_meta["effects"]:
                    rows.append({"detail_index": detail_index, "detail_type": "special_effect", **effect})
                    detail_index += 1
                detail_metadata.append({"card_id": card_meta["card_id"], "details": rows})
    table = torch.cat(embeddings, dim=0)
    max_detail_count = max(tensor.size(1) for tensor in detail_tokens)

    def pad_details(tensor: torch.Tensor, value: float = 0.0) -> torch.Tensor:
        if tensor.size(1) == max_detail_count:
            return tensor
        pad_shape = list(tensor.shape)
        pad_shape[1] = max_detail_count - tensor.size(1)
        pad = tensor.new_full(pad_shape, value)
        return torch.cat([tensor, pad], dim=1)

    detail_table = torch.cat([pad_details(tensor) for tensor in detail_tokens], dim=0)
    detail_mask_table = torch.cat([pad_details(tensor) for tensor in detail_masks], dim=0)
    detail_type_table = torch.cat([pad_details(tensor) for tensor in detail_type_ids], dim=0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(table, args.output)
    np.save(args.output.with_suffix(".npy"), table.numpy())
    torch.save(detail_table, args.output.parent / "card_detail_tokens.pt")
    torch.save(detail_mask_table, args.output.parent / "card_detail_masks.pt")
    torch.save(detail_type_table, args.output.parent / "card_detail_type_ids.pt")
    id_path = args.output.parent / "card_id_to_index.json"
    id_path.write_text(json.dumps(dataset.card_id_to_index, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    metadata = {
        "embedding_dim": int(table.shape[1]),
        "detail_token_dim": int(detail_table.shape[-1]),
        "max_detail_count": int(detail_table.shape[1]),
        "card_count": int(table.shape[0]),
        "checkpoint": str(args.checkpoint),
        "data_version": str(args.cache_dir),
        "field_vocab": schema["vocab"],
        "normalization": schema["normalization"],
        "text_encoder": {
            "name": checkpoint["config"]["model"].get("text_encoder_name", "hashing"),
            "hash_dim": schema.get("text_hash_dim", 2048),
            "frozen": bool(checkpoint["config"]["model"].get("freeze_text_encoder", False)),
        },
    }
    (args.output.parent / "card_embedding_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (args.output.parent / "card_detail_metadata.json").write_text(
        json.dumps(
            {
                "schema_version": schema.get("schema_version"),
                "model_version": "card_encoder_static_detail_v1",
                "cards": detail_metadata,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"exported {tuple(table.shape)} to {args.output}")


if __name__ == "__main__":
    main()
