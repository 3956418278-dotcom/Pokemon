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
    encoder = CardEncoder(schema, embedding_dim=int(checkpoint["config"]["model"].get("embedding_dim", 128)))
    encoder.load_state_dict(checkpoint["encoder"])
    encoder.eval()
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=lambda items: collate_cards(items, schema))
    embeddings = []
    with torch.no_grad():
        for batch in loader:
            embeddings.append(encoder(batch))
    table = torch.cat(embeddings, dim=0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(table, args.output)
    np.save(args.output.with_suffix(".npy"), table.numpy())
    id_path = args.output.parent / "card_id_to_index.json"
    id_path.write_text(json.dumps(dataset.card_id_to_index, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    metadata = {
        "embedding_dim": int(table.shape[1]),
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
    print(f"exported {tuple(table.shape)} to {args.output}")


if __name__ == "__main__":
    main()

