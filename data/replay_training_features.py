from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch

from .replay_dataset import ReplayDecisionSample
from models.dynamic_state_encoder import DynamicStateEncoder, DynamicStateEncoderOutput


@dataclass
class ReplayTrainingBatch:
    """Model-ready features from online replay decision samples.

    The replay JSON can have variable game lengths and variable board sizes.
    This batch keeps per-step policy targets as metadata while exposing the
    encoded board state for downstream heads.
    """

    board_embeddings: torch.Tensor
    rewards: torch.Tensor
    done: torch.Tensor
    select_type: torch.Tensor
    select_context: torch.Tensor
    option_count: torch.Tensor
    actions: list[list[int]]
    encoder_outputs: list[DynamicStateEncoderOutput]
    metadata: list[dict[str, Any]]


def encode_replay_samples(
    samples: Sequence[ReplayDecisionSample],
    encoder: DynamicStateEncoder,
    device: torch.device | str | None = None,
) -> ReplayTrainingBatch:
    """Encode decision samples imported from Kaggle online replays.

    Each sample is encoded independently because observations contain variable
    numbers of card, ledger, and event tokens. Batching/padding is handled inside
    the board tokenizer for a single state; policy heads can batch the returned
    pooled board embeddings directly.
    """

    if device is not None:
        encoder = encoder.to(device)
    outputs: list[DynamicStateEncoderOutput] = []
    pooled: list[torch.Tensor] = []
    for sample in samples:
        output = encoder.forward_parsed(sample.parsed, sample.memory_after)
        outputs.append(output)
        pooled.append(output.board.pooled.squeeze(0))
    if pooled:
        board_embeddings = torch.stack(pooled, dim=0)
    else:
        output_dim = int(encoder.board_transformer.norm.normalized_shape[0])
        board_embeddings = torch.zeros(0, output_dim, dtype=torch.float32, device=next(encoder.parameters()).device)
    tensor_device = board_embeddings.device
    return ReplayTrainingBatch(
        board_embeddings=board_embeddings,
        rewards=torch.tensor([sample.reward for sample in samples], dtype=torch.float32, device=tensor_device),
        done=torch.tensor([float(sample.done) for sample in samples], dtype=torch.float32, device=tensor_device),
        select_type=torch.tensor([sample.select_type for sample in samples], dtype=torch.long, device=tensor_device),
        select_context=torch.tensor([sample.select_context for sample in samples], dtype=torch.long, device=tensor_device),
        option_count=torch.tensor([sample.option_count for sample in samples], dtype=torch.long, device=tensor_device),
        actions=[sample.action for sample in samples],
        encoder_outputs=outputs,
        metadata=[
            {
                "replay_id": sample.replay_id,
                "episode_id": sample.episode_id,
                "step_index": sample.step_index,
                "agent_index": sample.agent_index,
                "option_count": sample.option_count,
            }
            for sample in samples
        ],
    )
