from __future__ import annotations

from typing import Any

import torch

from decision_agent_v1.contracts.schemas import DecisionSampleV1


def collate_decision_samples(samples: list[DecisionSampleV1]) -> dict[str, Any]:
    if not samples:
        raise ValueError("cannot collate an empty batch")
    batch_size = len(samples)
    max_cards = max(1, max(len(sample.cards) for sample in samples))
    max_options = max(1, max(len(sample.options) for sample in samples))
    card_dynamic_dim = len(samples[0].cards[0].dynamic_features()) if samples[0].cards else 33
    option_numeric_dim = 12

    card_index = torch.zeros(batch_size, max_cards, dtype=torch.long)
    card_owner = torch.zeros(batch_size, max_cards, dtype=torch.long)
    card_zone = torch.zeros(batch_size, max_cards, dtype=torch.long)
    card_position = torch.zeros(batch_size, max_cards, dtype=torch.long)
    card_dynamic = torch.zeros(batch_size, max_cards, card_dynamic_dim)
    card_mask = torch.zeros(batch_size, max_cards, dtype=torch.bool)

    option_type = torch.full((batch_size, max_options), -1, dtype=torch.long)
    option_select_type = torch.full((batch_size, max_options), -1, dtype=torch.long)
    option_context = torch.full((batch_size, max_options), -1, dtype=torch.long)
    option_owner = torch.full((batch_size, max_options), -1, dtype=torch.long)
    option_area = torch.full((batch_size, max_options), -1, dtype=torch.long)
    option_position = torch.full((batch_size, max_options), -1, dtype=torch.long)
    option_card_index = torch.zeros(batch_size, max_options, dtype=torch.long)
    option_numeric = torch.zeros(batch_size, max_options, option_numeric_dim)
    option_card_token_index = torch.full((batch_size, max_options), -1, dtype=torch.long)
    option_equivalence_group = torch.full((batch_size, max_options), -1, dtype=torch.long)
    original_option_index = torch.full((batch_size, max_options), -1, dtype=torch.long)
    option_mask = torch.zeros(batch_size, max_options, dtype=torch.bool)

    for batch_index, sample in enumerate(samples):
        serial_to_position: dict[int, int] = {}
        for position, card in enumerate(sample.cards):
            card_index[batch_index, position] = card.card_index
            card_owner[batch_index, position] = card.relative_owner
            card_zone[batch_index, position] = card.zone
            card_position[batch_index, position] = max(card.zone_position, 0)
            card_dynamic[batch_index, position] = torch.tensor(card.dynamic_features())
            card_mask[batch_index, position] = True
            if card.serial is not None:
                serial_to_position.setdefault(card.serial, position)
        for position, option in enumerate(sample.options):
            option_type[batch_index, position] = option.option_type
            option_select_type[batch_index, position] = option.select_type
            option_context[batch_index, position] = option.select_context
            option_owner[batch_index, position] = option.relative_player
            option_area[batch_index, position] = option.area
            option_position[batch_index, position] = option.position_index
            option_card_index[batch_index, position] = option.card_index
            option_numeric[batch_index, position] = torch.tensor(
                (
                    option.position_index / 60.0,
                    option.damage_value / 300.0,
                    float(option.has_card_reference),
                    float(option.has_serial_reference),
                    *option.field_visibility_mask,
                )
            )
            if option.serial is not None and option.serial in serial_to_position:
                option_card_token_index[batch_index, position] = serial_to_position[option.serial]
            option_equivalence_group[batch_index, position] = option.equivalence_group
            original_option_index[batch_index, position] = option.original_option_index
            option_mask[batch_index, position] = True

    inverse_episode_length = torch.tensor(
        [1.0 / max(sample.episode_decision_count, 1) for sample in samples], dtype=torch.float32
    )
    episode_weight = inverse_episode_length / inverse_episode_length.sum() * batch_size
    return {
        "card_index": card_index,
        "card_owner": card_owner,
        "card_zone": card_zone,
        "card_position": card_position,
        "card_dynamic": card_dynamic,
        "card_mask": card_mask,
        "global_features": torch.tensor([sample.global_state.features() for sample in samples]),
        "history_features": torch.tensor([sample.history.features() for sample in samples]),
        "option_type": option_type,
        "option_select_type": option_select_type,
        "option_context": option_context,
        "option_owner": option_owner,
        "option_area": option_area,
        "option_position": option_position,
        "option_card_index": option_card_index,
        "option_numeric": option_numeric,
        "option_card_token_index": option_card_token_index,
        "option_equivalence_group": option_equivalence_group,
        "original_option_index": original_option_index,
        "option_mask": option_mask,
        "target_sequences": [sample.selected_option_indices for sample in samples],
        "target_equivalence_groups": [sample.selected_equivalence_groups for sample in samples],
        "target_sequence_mask": torch.nn.utils.rnn.pad_sequence(
            [torch.ones(len(sample.selected_option_indices), dtype=torch.bool) for sample in samples],
            batch_first=True,
        ),
        "value_target": torch.tensor([int(sample.terminal_outcome) for sample in samples]),
        "policy_sample_mask": torch.tensor([sample.policy_supervision for sample in samples]),
        "episode_weight": episode_weight,
        "selection_modes": [sample.selection_mode for sample in samples],
        "metadata": [
            {
                "episode_id": sample.episode_id,
                "agent_index": sample.agent_index,
                "decision_index": sample.decision_index,
                "turn": sample.turn,
                "select_context": sample.global_state.select_context,
                "policy_mask_reason": sample.policy_mask_reason,
            }
            for sample in samples
        ],
    }
