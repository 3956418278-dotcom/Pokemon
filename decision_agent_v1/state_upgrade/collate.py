from __future__ import annotations

from dataclasses import fields
from typing import Any

import torch

from .features import (
    BELIEF_TOP_K,
    MAX_BELIEF_CARDS,
    MAX_LEDGER_CARDS,
    MAX_SELF_DECK_CARDS,
    RECENT_EVENT_COUNT,
    LEDGER_SUMMARY_DIM,
    LEDGER_CARD_NUMERIC_DIM,
    SELF_DECK_SUMMARY_DIM,
    StateUpgradeFeatures,
)


V2_TENSOR_KEYS = tuple(field.name for field in fields(StateUpgradeFeatures))


def collate_state_upgrade(rows: list[StateUpgradeFeatures]) -> dict[str, torch.Tensor]:
    if not rows:
        raise ValueError("cannot collate empty state-upgrade rows")
    size = len(rows)

    def matrix(name: str, width: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        result = torch.zeros(size, width, dtype=dtype)
        for index, row in enumerate(rows):
            values = getattr(row, name)
            if values:
                result[index, : min(width, len(values))] = torch.tensor(values[:width], dtype=dtype)
        return result

    def cube(name: str, count: int, width: int) -> torch.Tensor:
        result = torch.zeros(size, count, width)
        for index, row in enumerate(rows):
            values = getattr(row, name)
            if values:
                tensor = torch.tensor(values[:count], dtype=torch.float32)
                result[index, : tensor.shape[0], : tensor.shape[1]] = tensor
        return result

    result = {
        "ledger_summary": matrix("ledger_summary", LEDGER_SUMMARY_DIM),
        "ledger_card_index": matrix("ledger_card_index", MAX_LEDGER_CARDS, torch.long),
        "ledger_card_numeric": cube("ledger_card_numeric", MAX_LEDGER_CARDS, LEDGER_CARD_NUMERIC_DIM),
        "recent_event_type": matrix("recent_event_type", RECENT_EVENT_COUNT, torch.long),
        "recent_event_player": matrix("recent_event_player", RECENT_EVENT_COUNT, torch.long),
        "recent_event_card_index": matrix("recent_event_card_index", RECENT_EVENT_COUNT, torch.long),
        "recent_event_source": matrix("recent_event_source", RECENT_EVENT_COUNT, torch.long),
        "recent_event_target": matrix("recent_event_target", RECENT_EVENT_COUNT, torch.long),
        "recent_event_numeric": cube("recent_event_numeric", RECENT_EVENT_COUNT, 6),
        "self_deck_summary": matrix("self_deck_summary", SELF_DECK_SUMMARY_DIM),
        "self_deck_card_index": matrix("self_deck_card_index", MAX_SELF_DECK_CARDS, torch.long),
        "self_deck_card_numeric": cube("self_deck_card_numeric", MAX_SELF_DECK_CARDS, 3),
        "belief_summary": matrix("belief_summary", 8),
        "belief_template_index": matrix("belief_template_index", BELIEF_TOP_K, torch.long),
        "belief_template_probability": matrix("belief_template_probability", BELIEF_TOP_K),
        "belief_card_index": matrix("belief_card_index", MAX_BELIEF_CARDS, torch.long),
        "belief_card_expected": matrix("belief_card_expected", MAX_BELIEF_CARDS),
        "archetype_target": torch.tensor([row.archetype_target for row in rows], dtype=torch.long),
        "next_public_target": torch.tensor([row.next_public_target for row in rows], dtype=torch.long),
        "next_public_mask": torch.tensor([row.next_public_mask for row in rows], dtype=torch.bool),
    }
    result["ledger_card_mask"] = result["ledger_card_index"] != 0
    result["recent_event_mask"] = result["recent_event_type"] != 0
    result["self_deck_card_mask"] = result["self_deck_card_index"] != 0
    result["belief_template_mask"] = result["belief_template_index"] != 0
    result["belief_card_mask"] = result["belief_card_index"] != 0
    return result
