from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch

from decision_agent_v1.data.cache import CachedDecisionCorpus


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = ROOT / "outputs/decision_agent_v1"


def _distribution(values: list[int]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "min": 0, "max": 0, "mean": 0.0}
    return {"count": len(values), "min": min(values), "max": max(values), "mean": sum(values) / len(values)}


def audit_cache(cache_dir: Path) -> dict[str, Any]:
    corpus = CachedDecisionCorpus(cache_dir)
    manifest = corpus.manifest
    split_episode_ids: dict[str, set[str]] = {}
    split_dates: dict[str, set[str]] = {}
    split_payloads = {}
    principle_errors = []
    exclusion_audit: dict[str, Any] = {}
    episode_outcomes: dict[tuple[str, int], str] = {}
    for split in ("train", "validation", "test"):
        episodes = set()
        dates = set()
        decisions_per_stream = Counter()
        agent_counts = Counter()
        outcomes = Counter()
        outcomes_by_agent: dict[str, Counter[str]] = defaultdict(Counter)
        modes = Counter()
        select_types = Counter()
        contexts = Counter()
        option_counts = []
        card_counts = []
        history_counts = []
        unknown = 0
        visible_card_count = 0
        unk_card_count = 0
        serial_refs = 0
        serial_matches = 0
        equivalence_count = 0
        total_options = 0
        for batch in corpus.iter_batches(split, 2048):
            batch_size = len(batch["metadata"])
            for row_index, row in enumerate(batch["metadata"]):
                episode = str(row["episode_id"])
                episodes.add(episode)
                if row["source_date"]:
                    dates.add(str(row["source_date"]))
                stream = (episode, int(row["agent_index"]))
                decisions_per_stream[stream] += 1
                agent_counts[str(row["agent_index"])] += 1
                outcomes[row["terminal_outcome"]] += 1
                outcomes_by_agent[str(row["agent_index"])][row["terminal_outcome"]] += 1
                modes[row["selection_mode"]] += 1
                select_types[str(row["select_type"])] += 1
                contexts[str(row["select_context"])] += 1
                history_counts.append(int(row["history_token_count"]))
                episode_outcomes.setdefault(stream, row["terminal_outcome"])
                if episode_outcomes[stream] != row["terminal_outcome"]:
                    principle_errors.append(f"inconsistent terminal outcome in {stream}")
                target = list(row["target_sequence"])
                valid_option_count = int(batch["option_mask"][row_index].sum())
                if any(index < 0 or index >= valid_option_count for index in target):
                    principle_errors.append(f"out-of-range target in {stream}:{row['decision_index']}")
                if not (int(row["min_count"]) <= len(target) <= int(row["max_count"])):
                    principle_errors.append(f"target count contract violation in {stream}:{row['decision_index']}")
                if row["selection_mode"] == "UNKNOWN":
                    unknown += 1
                    if row["policy_supervision"]:
                        principle_errors.append(f"UNKNOWN Policy supervision in {stream}:{row['decision_index']}")
                if row["visibility_sources"] != ["observation.current", "observation.logs", "observation.select"]:
                    principle_errors.append(f"visibility source violation in {stream}:{row['decision_index']}")
            cards = batch["card_mask"]
            options = batch["option_mask"]
            card_counts.extend(cards.sum(dim=1).tolist())
            option_counts.extend(options.sum(dim=1).tolist())
            identity_visible = cards & (batch["card_dynamic"][..., 25] > 0.5)
            visible_card_count += int(identity_visible.sum())
            unk_card_count += int(((batch["card_index"] == 1) & identity_visible).sum())
            serial_reference = options & (batch["option_numeric"][..., 3] > 0.5)
            serial_refs += int(serial_reference.sum())
            serial_matches += int((serial_reference & (batch["option_card_token_index"] >= 0)).sum())
            equivalence_count += int((options & (batch["option_equivalence_group"] >= 0)).sum())
            total_options += int(options.sum())
        split_episode_ids[split] = episodes
        split_dates[split] = dates
        split_manifest = manifest["splits"][split]
        excluded = list(split_manifest.get("excluded_episodes", []))
        requested = int(
            manifest.get("requested_episode_counts", {}).get(
                split, int(split_manifest.get("episode_count", 0)) + len(excluded)
            )
        )
        included = int(split_manifest.get("episode_count", 0))
        if included != len(episodes):
            principle_errors.append(
                f"{split} manifest/cache episode mismatch: {included} != {len(episodes)}"
            )
        if included + len(excluded) != requested:
            principle_errors.append(
                f"{split} requested episode accounting mismatch: "
                f"{included} + {len(excluded)} != {requested}"
            )
        excluded_names = [str(row.get("archive_name", "")) for row in excluded]
        if len(excluded_names) != len(set(excluded_names)):
            principle_errors.append(f"{split} contains duplicate excluded episodes")
        exclusion_audit[split] = {
            "requested_episode_count": requested,
            "included_episode_count": included,
            "excluded_episode_count": len(excluded),
            "excluded_episodes": excluded,
            "accounting_complete": included + len(excluded) == requested,
        }
        side_coverage = Counter()
        outcome_pairs = Counter()
        for episode in episodes:
            sides = {agent: outcome for (key, agent), outcome in episode_outcomes.items() if key == episode}
            if set(sides) == {0, 1}:
                side_coverage["two_sided"] += 1
                outcome_pairs[f"agent0_{sides[0]}__agent1_{sides[1]}"] += 1
                valid = sides[0] == sides[1] == "DRAW" or {sides[0], sides[1]} == {"WIN", "LOSS"}
                if not valid:
                    principle_errors.append(f"non-opposite two-player outcomes for {episode}: {sides}")
            else:
                side_coverage["one_sided"] += 1
        split_payloads[split] = {
            "dates": sorted(dates),
            "episode_count": len(episodes),
            "decision_count": sum(agent_counts.values()),
            "decision_count_by_agent": dict(agent_counts),
            "terminal_outcome_distribution": dict(outcomes),
            "terminal_outcome_distribution_by_agent": {
                key: dict(values) for key, values in sorted(outcomes_by_agent.items())
            },
            "episode_side_coverage": dict(side_coverage),
            "terminal_outcome_pair_distribution": dict(outcome_pairs),
            "selection_mode_distribution": dict(modes),
            "select_type_distribution": dict(select_types),
            "select_context_distribution": dict(contexts),
            "unknown_sample_count": unknown,
            "decision_per_agent_episode_distribution": _distribution(list(decisions_per_stream.values())),
            "option_count_distribution": _distribution([int(value) for value in option_counts]),
            "card_token_count_distribution": _distribution([int(value) for value in card_counts]),
            "history_token_count_distribution": _distribution(history_counts),
            "unk_card_id_ratio": unk_card_count / max(visible_card_count, 1),
            "serial_reference_match_rate": serial_matches / max(serial_refs, 1),
            "equivalence_group_coverage": equivalence_count / max(total_options, 1),
        }
    episode_intersections = {
        "train_validation": sorted(split_episode_ids["train"] & split_episode_ids["validation"]),
        "train_test": sorted(split_episode_ids["train"] & split_episode_ids["test"]),
        "validation_test": sorted(split_episode_ids["validation"] & split_episode_ids["test"]),
    }
    date_intersections = {
        "train_validation": sorted(split_dates["train"] & split_dates["validation"]),
        "train_test": sorted(split_dates["train"] & split_dates["test"]),
        "validation_test": sorted(split_dates["validation"] & split_dates["test"]),
    }
    payload = {
        "schema_version": "replay_corpus_audit_v1",
        "cache_dir": str(cache_dir),
        "cache_identity": {
            key: manifest[key]
            for key in ("schema_hash", "action_contract_hash", "card_vocabulary_hash", "adapter_hash")
        },
        "splits": split_payloads,
        "episode_inclusion_audit": exclusion_audit,
        "episode_split_intersections": episode_intersections,
        "date_split_intersections": date_intersections,
        "hidden_information_audit_passed": manifest["hidden_information_audit_passed"],
        "principle_errors": principle_errors,
        "passed": (
            manifest["hidden_information_audit_passed"]
            and not any(episode_intersections.values())
            and not any(date_intersections.values())
            and not principle_errors
        ),
    }
    (cache_dir / "statistics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("cache_dir", type=Path)
    args = parser.parse_args()
    payload = audit_cache(args.cache_dir)
    json_path = OUTPUT_ROOT / "audits/replay_corpus_audit.json"
    md_path = OUTPUT_ROOT / "audits/replay_corpus_audit.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# Replay corpus audit", "", f"- Cache: `{args.cache_dir}`", f"- Status: {'PASS' if payload['passed'] else 'FAIL'}"]
    for split, values in payload["splits"].items():
        inclusion = payload["episode_inclusion_audit"][split]
        lines.append(
            f"- {split}: dates={values['dates']}, requested={inclusion['requested_episode_count']}, "
            f"included={inclusion['included_episode_count']}, excluded={inclusion['excluded_episode_count']}, "
            f"decisions={values['decision_count']}, UNKNOWN={values['unknown_sample_count']}"
        )
    lines.extend(
        [
            f"- Episode intersections: `{payload['episode_split_intersections']}`",
            f"- Date intersections: `{payload['date_split_intersections']}`",
            f"- Principle errors: {len(payload['principle_errors'])}",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if not payload["passed"]:
        raise RuntimeError(f"corpus audit failed; see {json_path}")
    print(json_path)


if __name__ == "__main__":
    main()
