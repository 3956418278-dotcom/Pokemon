from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from data.card_preprocessing import DEFAULT_CACHE_DIR, load_or_create_records


def cosine_neighbors(table: torch.Tensor, k: int) -> torch.Tensor:
    norm = torch.nn.functional.normalize(table, dim=-1)
    sims = norm @ norm.t()
    sims.fill_diagonal_(-1.0)
    return sims.topk(k, dim=1).indices


def purity(records: list[dict], neighbors: torch.Tensor, field: str) -> float:
    scores = []
    for index, row in enumerate(records):
        value = row.get(field)
        if value is None:
            continue
        neigh_values = [records[int(j)].get(field) for j in neighbors[index]]
        scores.append(sum(v == value for v in neigh_values) / max(1, len(neigh_values)))
    return sum(scores) / len(scores) if scores else 0.0


def write_pca(table: torch.Tensor, output_dir: Path) -> None:
    arr = table.numpy()
    arr = arr - arr.mean(axis=0, keepdims=True)
    u, s, _v = np.linalg.svd(arr, full_matrices=False)
    coords = u[:, :2] * s[:2]
    np.save(output_dir / "pca_2d.npy", coords)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", type=Path, default=Path("artifacts/card_embeddings.pt"))
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/card_embedding_analysis"))
    parser.add_argument("--neighbors", type=int, default=5)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table = torch.load(args.embeddings, map_location="cpu")
    records, _mapping, summary = load_or_create_records(args.cache_dir)
    neighbors = cosine_neighbors(table, args.neighbors)
    write_pca(table, args.output_dir)

    neighbor_rows = []
    for index, neigh in enumerate(neighbors):
        row = records[index]
        neighbor_rows.append(
            {
                "card_id": row["card_id"],
                "name": row["name"],
                "card_type": row["card_type"],
                "neighbors": [
                    {
                        "card_id": records[int(j)]["card_id"],
                        "name": records[int(j)]["name"],
                        "card_type": records[int(j)]["card_type"],
                    }
                    for j in neigh
                ],
            }
        )
    (args.output_dir / "nearest_neighbors.json").write_text(json.dumps(neighbor_rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    metrics = {
        "card_type_neighbor_purity": purity(records, neighbors, "card_type"),
        "pokemon_type_neighbor_purity": purity(records, neighbors, "pokemon_type"),
        "stage_neighbor_purity": purity(records, neighbors, "stage"),
        "preprocess_summary": summary,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# Card Embedding Analysis",
        "",
        f"Cards: {len(records)}",
        f"Embedding shape: {tuple(table.shape)}",
        "",
        "## Metrics",
        "",
        f"- Card type neighbor purity: {metrics['card_type_neighbor_purity']:.4f}",
        f"- Pokemon type neighbor purity: {metrics['pokemon_type_neighbor_purity']:.4f}",
        f"- Stage neighbor purity: {metrics['stage_neighbor_purity']:.4f}",
        "",
        "## Representative nearest neighbors",
        "",
    ]
    for row in neighbor_rows[:20]:
        lines.append(f"### {row['name']} ({row['card_type']})")
        for neigh in row["neighbors"]:
            lines.append(f"- {neigh['name']} ({neigh['card_type']})")
        lines.append("")
    lines.extend(
        [
            "## Interpretation",
            "",
            "Nearest-neighbor similarity can come from shared card type, numeric structure, text hash semantics, and evolution/name relations learned during pretraining.",
            "The PCA coordinates are saved as `pca_2d.npy`; plot them in Kaggle if an image artifact is needed.",
        ]
    )
    (args.output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

