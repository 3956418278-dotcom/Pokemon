from __future__ import annotations

import argparse
import copy
from collections import Counter
from dataclasses import asdict

from data.observation_parser import parse_observation

from decision_agent_v1.adapters.observation_adapter import ObservationAdapter
from ._common import OUTPUT_ROOT, add_data_arguments, load_samples, write_json


def _distribution(values: list[int]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "min": 0, "max": 0, "mean": 0.0}
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": sum(values) / len(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    add_data_arguments(parser, default_replays=8)
    args = parser.parse_args()
    samples, report, vocabulary, dataset = load_samples(args)
    serial_references = 0
    serial_matches = 0
    missing_card_ids = 0
    equivalence_references = 0
    for sample in samples:
        visible_serials = {card.serial for card in sample.cards if card.serial is not None}
        for card in sample.cards:
            missing_card_ids += int(card.card_id is not None and card.card_index == vocabulary.unk_index)
        for option in sample.options:
            if option.serial is not None:
                serial_references += 1
                serial_matches += int(option.serial in visible_serials)
            equivalence_references += int(option.equivalence_group >= 0)
    serialized = "\n".join(str(asdict(sample)) for sample in samples[:5])
    adapter = ObservationAdapter(vocabulary)
    raw = dataset.samples[0]
    adversarial = copy.deepcopy(raw.observation)
    adversarial["visualize"] = {
        "current": {
            "opponent_hand": [{"id": 999999, "serial": 999999}],
            "opponent_deck": [{"id": 999998, "serial": 999998}],
            "hidden_prizes": [{"id": 999997, "serial": 999997}],
        }
    }
    parsed_original = parse_observation(raw.observation)
    parsed_adversarial = parse_observation(adversarial)
    visualize_invariant = (
        adapter.cards(parsed_original) == adapter.cards(parsed_adversarial)
        and adapter.global_state(parsed_original) == adapter.global_state(parsed_adversarial)
    )
    no_opponent_hidden_hand_identity = all(
        not (card.relative_owner == 2 and card.zone == 2 and card.card_id is not None)
        for sample in samples
        for card in sample.cards
    )
    no_hidden_prize_identity = all(
        not (card.zone == 6 and card.card_id is not None)
        for sample in samples
        for card in sample.cards
    )
    visibility_passed = (
        "visualize" not in serialized
        and visualize_invariant
        and no_opponent_hidden_hand_identity
        and no_hidden_prize_identity
        and all(sample.visibility_sources == ("observation.current", "observation.logs", "observation.select") for sample in samples)
    )
    payload = {
        "source_replay_path": str(args.replay_path),
        "successful_episode_count": report.episodes,
        "valid_decision_count": len(samples),
        "parser_error_count": len(dataset.summary.parser_errors),
        "terminal_outcome_distribution": report.outcome_counts,
        "selection_mode_distribution": report.selection_mode_counts,
        "unknown_combinations": report.unknown_combinations,
        "unknown_semantics_count": sum(report.unknown_combinations.values()),
        "card_instance_distribution": _distribution([len(sample.cards) for sample in samples]),
        "option_distribution": _distribution([len(sample.options) for sample in samples]),
        "multi_select_distribution": dict(Counter(len(sample.selected_option_indices) for sample in samples)),
        "card_id_mapping_missing": missing_card_ids,
        "serial_reference_count": serial_references,
        "serial_reference_match_rate": serial_matches / max(serial_references, 1),
        "equivalence_group_coverage": equivalence_references / max(sum(len(sample.options) for sample in samples), 1),
        "episode_decision_distribution": _distribution([sample.episode_decision_count for sample in samples]),
        "visibility_check": {
            "passed": visibility_passed,
            "allowed_sources": ["observation.current", "observation.logs", "observation.select"],
            "forbidden_sources": ["visualize", "visualize.current", "opponent hidden hand/deck", "hidden prize identity"],
            "decision_sample_retains_raw_observation": False,
            "adversarial_visualize_invariance": visualize_invariant,
            "opponent_hidden_hand_identity_absent": no_opponent_hidden_hand_identity,
            "hidden_prize_identity_absent": no_hidden_prize_identity,
        },
        "terminal_outcome_agent_perspective": {
            "passed": "WIN" in report.outcome_counts and "LOSS" in report.outcome_counts,
            "rule": "Use terminal reward from the same agent_index stream; when the existing BC dataset omits the other side's empty terminal action, derive its label by the audited two-player zero-sum complement.",
        },
        "existing_dataset_summary": {
            "replay_count": dataset.summary.replay_count,
            "agent_perspective_mismatch_count": dataset.summary.agent_perspective_mismatch_count,
            "turn_owner_conflict_count": dataset.summary.turn_owner_conflict_count,
            "illegal_action_indices": dataset.summary.illegal_action_indices,
        },
    }
    if not visibility_passed:
        raise RuntimeError("visibility audit failed")
    output_json = OUTPUT_ROOT / "audits/interface_audit.json"
    write_json(output_json, payload)
    lines = [
        "# Decision Agent V1 interface audit",
        "",
        f"- Episodes: {report.episodes}",
        f"- Decisions: {len(samples)}",
        f"- Visibility check: {'PASS' if visibility_passed else 'FAIL'}",
        f"- Agent-perspective terminal outcome check: {'PASS' if payload['terminal_outcome_agent_perspective']['passed'] else 'FAIL'}",
        f"- UNKNOWN semantics: {payload['unknown_semantics_count']}",
        f"- Serial reference match rate: {payload['serial_reference_match_rate']:.6f}",
        f"- Card vocabulary misses: {missing_card_ids}",
    ]
    (OUTPUT_ROOT / "audits/interface_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output_json)


if __name__ == "__main__":
    main()
