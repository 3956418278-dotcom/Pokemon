from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.online_replay_importer import import_mounted_daily_replay_dataset, select_daily_dataset_refs
from data.replay_dataset import ReplayDecisionDataset


def require_torch():
    try:
        import torch
        import torch.nn as nn
    except ModuleNotFoundError as exc:
        raise SystemExit("This training script requires torch. Run it in Kaggle or an environment with PyTorch installed.") from exc
    return torch, nn


def make_heads(nn):
    class DynamicReplayFeatureHeads(nn.Module):
        def __init__(self, input_dim: int = 128, hidden_dim: int = 128) -> None:
            super().__init__()
            self.shared = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.LayerNorm(hidden_dim))
            self.select_type = nn.Linear(hidden_dim, 16)
            self.select_context = nn.Linear(hidden_dim, 64)
            self.option_count = nn.Linear(hidden_dim, 32)
            self.reward = nn.Linear(hidden_dim, 1)

        def forward(self, board_embeddings):
            hidden = self.shared(board_embeddings)
            return {
                "select_type": self.select_type(hidden),
                "select_context": self.select_context(hidden),
                "option_count": self.option_count(hidden),
                "reward": self.reward(hidden).squeeze(-1),
            }

    return DynamicReplayFeatureHeads()


def build_observed_static_adapter(samples, torch, StaticCardEmbeddingAdapter, freeze: bool = False):
    card_ids = sorted(
        {
            int(instance.static_card_id)
            for sample in samples
            for instance in sample.parsed.card_instances
            if int(instance.static_card_id) > 0
        }
    )
    if not card_ids:
        card_ids = [1]
    generator = torch.Generator().manual_seed(17)
    weights = torch.randn(len(card_ids), 128, generator=generator) * 0.02
    mapping = {str(card_id): index for index, card_id in enumerate(card_ids)}
    details = torch.zeros(len(card_ids), 1, 128)
    detail_mask = torch.zeros(len(card_ids), 1)
    detail_type_ids = torch.zeros(len(card_ids), 1, dtype=torch.long)
    return StaticCardEmbeddingAdapter(
        weights,
        mapping,
        freeze=freeze,
        detail_tokens=details,
        detail_mask=detail_mask,
        detail_type_ids=detail_type_ids,
    )


def load_static_adapter(args: argparse.Namespace, samples, torch, StaticCardEmbeddingAdapter):
    if args.static_artifact_dir is not None and (args.static_artifact_dir / "card_embeddings.pt").exists():
        return StaticCardEmbeddingAdapter.from_artifacts(args.static_artifact_dir, freeze=args.freeze_static)
    return build_observed_static_adapter(samples, torch, StaticCardEmbeddingAdapter, freeze=False)


def choose_daily_dirs(args: argparse.Namespace) -> list[Path]:
    daily_dirs = list(args.daily_replay_dir)
    if args.use_daily_manifest:
        if args.episodes_index_dir is None:
            raise SystemExit("--use-daily-manifest requires --episodes-index-dir")
        refs = select_daily_dataset_refs(
            args.episodes_index_dir,
            mount_root=args.daily_dataset_mount_root,
            reserve_recent_days=args.reserve_recent_days,
            import_split=args.import_split,
            max_days=args.max_days,
        )
        daily_dirs.extend([ref.mount_path for ref in refs if ref.mount_path is not None])
    if not daily_dirs:
        raise SystemExit("Provide --daily-replay-dir or --use-daily-manifest with mounted daily datasets.")
    return daily_dirs


def load_replay_dataset(args: argparse.Namespace) -> tuple[ReplayDecisionDataset, dict]:
    daily_dirs = choose_daily_dirs(args)
    return import_mounted_daily_replay_dataset(
        daily_dirs,
        output_dir=args.output_dir / "dataset",
        include_no_select=args.include_no_select,
        controlled_agents=set(args.controlled_agent) if args.controlled_agent is not None else None,
        max_samples=args.max_samples,
    )


def compute_loss(outputs: dict[str, Any], batch, nn) -> tuple[Any, dict[str, float]]:
    select_type = batch.select_type.clamp_min(0).clamp_max(15)
    select_context = batch.select_context.clamp_min(0).clamp_max(63)
    option_count = batch.option_count.clamp_min(0).clamp_max(31)
    loss_select_type = nn.functional.cross_entropy(outputs["select_type"], select_type)
    loss_select_context = nn.functional.cross_entropy(outputs["select_context"], select_context)
    loss_option_count = nn.functional.cross_entropy(outputs["option_count"], option_count)
    loss_reward = nn.functional.mse_loss(outputs["reward"], batch.rewards)
    loss = loss_select_type + 0.5 * loss_select_context + 0.25 * loss_option_count + 0.1 * loss_reward
    metrics = {
        "loss": float(loss.detach().cpu().item()),
        "select_type_loss": float(loss_select_type.detach().cpu().item()),
        "select_context_loss": float(loss_select_context.detach().cpu().item()),
        "option_count_loss": float(loss_option_count.detach().cpu().item()),
        "reward_loss": float(loss_reward.detach().cpu().item()),
        "select_type_acc": float((outputs["select_type"].argmax(dim=-1) == select_type).float().mean().detach().cpu().item()),
    }
    return loss, metrics


def train(args: argparse.Namespace) -> dict:
    torch, nn = require_torch()
    from data.replay_training_features import encode_replay_samples
    from models.dynamic_state_encoder import DynamicStateEncoder
    from models.static_card_adapter import StaticCardEmbeddingAdapter

    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset, import_metadata = load_replay_dataset(args)
    if len(dataset) == 0:
        raise RuntimeError("No replay decision samples were loaded.")
    device = torch.device(args.device)
    static_adapter = load_static_adapter(args, dataset.samples, torch, StaticCardEmbeddingAdapter)
    encoder = DynamicStateEncoder(static_adapter).to(device)
    heads = make_heads(nn).to(device)
    parameters = list(encoder.parameters()) + list(heads.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=args.weight_decay)

    metrics_rows: list[dict] = []
    indices = list(range(len(dataset)))
    for epoch in range(1, args.epochs + 1):
        random.Random(args.seed + epoch).shuffle(indices)
        epoch_rows = []
        encoder.train()
        heads.train()
        for start in range(0, len(indices), args.batch_size):
            batch_indices = indices[start : start + args.batch_size]
            samples = [dataset.samples[index] for index in batch_indices]
            optimizer.zero_grad(set_to_none=True)
            batch = encode_replay_samples(samples, encoder, device=device)
            outputs = heads(batch.board_embeddings)
            loss, metrics = compute_loss(outputs, batch, nn)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, args.grad_clip)
            optimizer.step()
            metrics["epoch"] = epoch
            metrics["batch_start"] = start
            metrics["batch_size"] = len(samples)
            epoch_rows.append(metrics)
            metrics_rows.append(metrics)
        epoch_summary = {
            key: sum(row[key] for row in epoch_rows) / max(len(epoch_rows), 1)
            for key in ["loss", "select_type_loss", "select_context_loss", "option_count_loss", "reward_loss", "select_type_acc"]
        }
        epoch_summary["epoch"] = epoch
        print(json.dumps(epoch_summary, ensure_ascii=False))

    checkpoint = {
        "encoder": encoder.state_dict(),
        "heads": heads.state_dict(),
        "args": vars(args),
        "dataset_summary": {
            "sample_count": len(dataset),
            "replay_count": dataset.summary.replay_count,
            "skipped_no_select": dataset.summary.skipped_no_select,
            "parser_errors": dataset.summary.parser_errors[:20],
            "max_instances": dataset.summary.max_instances,
            "max_options": dataset.summary.max_options,
            "max_events": dataset.summary.max_events,
            "max_token_estimate": dataset.summary.max_token_estimate,
        },
        "import_metadata": import_metadata,
    }
    checkpoint_path = args.output_dir / "dynamic_replay_feature_encoder.pt"
    torch.save(checkpoint, checkpoint_path)
    metrics_path = args.output_dir / "dynamic_replay_feature_metrics.jsonl"
    metrics_path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in metrics_rows) + "\n", encoding="utf-8")
    summary = {
        "checkpoint": str(checkpoint_path),
        "metrics": str(metrics_path),
        "sample_count": len(dataset),
        "replay_count": dataset.summary.replay_count,
        "max_instances": dataset.summary.max_instances,
        "max_options": dataset.summary.max_options,
        "max_events": dataset.summary.max_events,
        "max_token_estimate": dataset.summary.max_token_estimate,
        "last_metrics": metrics_rows[-1] if metrics_rows else None,
    }
    (args.output_dir / "dynamic_replay_feature_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train dynamic replay state features from mounted Kaggle replay JSON datasets.")
    parser.add_argument("--daily-replay-dir", type=Path, action="append", default=[])
    parser.add_argument("--episodes-index-dir", type=Path, default=None)
    parser.add_argument("--use-daily-manifest", action="store_true")
    parser.add_argument("--daily-dataset-mount-root", type=Path, default=Path("/kaggle/input"))
    parser.add_argument("--reserve-recent-days", type=int, default=3)
    parser.add_argument("--import-split", choices=["train", "reserved"], default="train")
    parser.add_argument("--max-days", type=int, default=None)
    parser.add_argument("--include-no-select", action="store_true")
    parser.add_argument("--controlled-agent", type=int, action="append")
    parser.add_argument("--max-samples", type=int, default=4096)
    parser.add_argument("--static-artifact-dir", type=Path, default=Path("outputs/card_pretrain/artifacts"))
    parser.add_argument("--freeze-static", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/dynamic_replay_features"))
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.device == "auto":
        try:
            import torch

            args.device = "cuda" if torch.cuda.is_available() else "cpu"
        except ModuleNotFoundError:
            args.device = "cpu"
    random.seed(args.seed)
    summary = train(args)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
