from __future__ import annotations

import argparse
import csv
import hashlib
import json
import multiprocessing as mp
import sys
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.decision_schema import ActionSemantics
from data.legal_options import build_action_target, infer_action_semantics, policy_mask_decision
from data.observation_parser import parse_observation
from data.replay_dataset import (
    ActionAlignmentError,
    agent_perspective_state,
    observation_fingerprint,
    parse_action,
    replay_content_hash,
    resolve_turn_owner,
    stable_replay_key,
)


OUTPUT_FILES = {
    "audit.json",
    "action_semantics.csv",
    "equivalence_resolution.csv",
    "policy_mask_reasons.csv",
    "turn_owner_audit.csv",
    "errors.jsonl",
}
SEMANTICS = tuple(item.value for item in ActionSemantics)
RESOLUTION_STATES = ("FULLY_RESOLVED", "PARTIALLY_RESOLVED", "UNRESOLVED")
EQUIVALENCE_MASK_REASONS = {"ONE_FEASIBLE_CLASS_TARGET", "FORCED_FULL_SUBSET"}
HARD_ZERO_FIELDS = (
    "invalid_action_index_count",
    "duplicate_decision_key_count",
    "agent_perspective_mismatch_count",
    "conflicting_turn_owner_count",
    "invariant_violation_count",
)

_ZIP: zipfile.ZipFile | None = None


def _init_worker(archive_path: str) -> None:
    global _ZIP
    _ZIP = zipfile.ZipFile(archive_path)


def _int(value: Any, default: int = -1) -> int:
    if isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _option_types(select: dict[str, Any]) -> tuple[int, ...]:
    values = {
        _int(option.get("type"))
        for option in (select.get("option") or [])
        if isinstance(option, dict)
    }
    return tuple(sorted(values)) or (-1,)


def _error(
    *,
    member: str,
    replay_key: str | None,
    stage: str,
    message: str,
    step: int | None = None,
    agent_index: int | None = None,
    **details: Any,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "archive_member": member,
        "replay_key": replay_key,
        "stage": stage,
        "message": message,
    }
    if step is not None:
        row["step"] = step
    if agent_index is not None:
        row["agent_index"] = agent_index
    row.update(details)
    return row


def _audit_member(member: str) -> dict[str, Any]:
    assert _ZIP is not None
    try:
        raw = _ZIP.read(member)
        replay = json.loads(raw)
        if not isinstance(replay, dict) or not isinstance(replay.get("steps"), list):
            raise ValueError("Replay must be a JSON object containing a steps list")
    except Exception as exc:
        return {
            "member": member,
            "successful": False,
            "errors": [
                _error(
                    member=member,
                    replay_key=None,
                    stage="replay_load",
                    message=f"{type(exc).__name__}: {exc}",
                )
            ],
        }

    replay_key = stable_replay_key(replay)
    canonical_hash = replay_content_hash(replay)
    raw_content_hash = hashlib.sha256(raw).hexdigest()
    info = replay.get("info")
    has_external_id = (
        isinstance(info, dict) and info.get("EpisodeId") is not None
    ) or replay.get("id") is not None
    summary: Counter[str] = Counter()
    action_rows: Counter[tuple[int, int, int, str]] = Counter()
    equivalence_rows: Counter[tuple[int, int, int, str, str]] = Counter()
    policy_rows: Counter[tuple[str, str, str, int, int, int]] = Counter()
    turn_rows: Counter[tuple[int, int, int, str, str]] = Counter()
    errors: list[dict[str, Any]] = []
    decision_keys: list[tuple[str, int | None, int, int, int]] = []
    pending: dict[int, dict[str, Any]] = {}
    last_fingerprint: dict[int, str] = {}

    episode_id = info.get("EpisodeId") if isinstance(info, dict) else None
    normalized_episode_id = _int(episode_id) if episode_id is not None else None

    for step_index, step in enumerate(replay["steps"]):
        if not isinstance(step, list):
            summary["parser_error_count"] += 1
            errors.append(
                _error(
                    member=member,
                    replay_key=replay_key,
                    stage="replay_structure",
                    message="Replay step is not a list",
                    step=step_index,
                )
            )
            continue
        for agent_index, agent_step in enumerate(step):
            if not isinstance(agent_step, dict):
                continue

            raw_action = agent_step.get("action")
            try:
                action = parse_action(raw_action)
            except ActionAlignmentError as exc:
                action = None
                summary["action_alignment_error_count"] += 1
                errors.append(
                    _error(
                        member=member,
                        replay_key=replay_key,
                        stage="action_alignment",
                        message=str(exc),
                        step=step_index,
                        agent_index=agent_index,
                        action=raw_action,
                    )
                )

            previous = pending.pop(agent_index, None)
            if previous is not None:
                summary["paired_decision_count"] += 1
                if action is not None:
                    select = previous["select"]
                    options = select.get("option") or []
                    invalid = [index for index in action if index < 0 or index >= len(options)]
                    if invalid:
                        summary["invalid_action_index_count"] += 1
                        errors.append(
                            _error(
                                member=member,
                                replay_key=replay_key,
                                stage="invalid_action_index",
                                message="Action contains an index outside the legal option range",
                                step=step_index,
                                agent_index=agent_index,
                                decision_step=previous["step"],
                                action=action,
                                option_count=len(options),
                                invalid_indices=invalid,
                            )
                        )
                    else:
                        semantics = infer_action_semantics(select, action)
                        try:
                            target = build_action_target(
                                previous["observation"], action, semantics
                            )
                        except ValueError as exc:
                            summary["action_alignment_error_count"] += 1
                            errors.append(
                                _error(
                                    member=member,
                                    replay_key=replay_key,
                                    stage="action_alignment",
                                    message=str(exc),
                                    step=step_index,
                                    agent_index=agent_index,
                                    decision_step=previous["step"],
                                    action=action,
                                )
                            )
                        else:
                            policy_loss_applies, mask_reason = policy_mask_decision(select, target)
                            resolution = target.equivalence_resolution_status
                            option_types = _option_types(select)
                            summary[f"action_semantics:{semantics.value}"] += 1
                            summary[f"resolution:{resolution}"] += 1
                            summary[f"policy_mask_reason:{mask_reason}"] += 1
                            for option_type in option_types:
                                action_rows[
                                    (
                                        _int(select.get("type")),
                                        _int(select.get("context")),
                                        option_type,
                                        semantics.value,
                                    )
                                ] += 1
                                equivalence_rows[
                                    (
                                        _int(select.get("type")),
                                        _int(select.get("context")),
                                        option_type,
                                        semantics.value,
                                        resolution,
                                    )
                                ] += 1
                                turn_rows[
                                    (
                                        _int(select.get("type")),
                                        _int(select.get("context")),
                                        option_type,
                                        previous["turn_owner_state"],
                                        previous["turn_owner_source"],
                                    )
                                ] += 1
                            policy_rows[
                                (
                                    mask_reason,
                                    semantics.value,
                                    resolution,
                                    len(options),
                                    len(target.equivalence_class_capacities),
                                    target.selected_count,
                                )
                            ] += 1
                            decision_keys.append(
                                (
                                    replay_key,
                                    normalized_episode_id,
                                    previous["step"],
                                    step_index,
                                    agent_index,
                                )
                            )

                            masked_by_equivalence = (
                                not policy_loss_applies
                                and mask_reason in EQUIVALENCE_MASK_REASONS
                            )
                            invariant_message = None
                            if resolution != "FULLY_RESOLVED" and masked_by_equivalence:
                                invariant_message = (
                                    f"{resolution} decision was masked because of equivalence"
                                )
                            elif (
                                semantics is ActionSemantics.ORDERED_INDEX_SEQUENCE
                                and masked_by_equivalence
                            ):
                                invariant_message = (
                                    "ORDERED_INDEX_SEQUENCE decision was masked because of equivalence"
                                )
                            if invariant_message is not None:
                                summary["invariant_violation_count"] += 1
                                errors.append(
                                    _error(
                                        member=member,
                                        replay_key=replay_key,
                                        stage="policy_mask_invariant",
                                        message=invariant_message,
                                        step=previous["step"],
                                        agent_index=agent_index,
                                        action_semantics=semantics.value,
                                        resolution_status=resolution,
                                        policy_mask_reason=mask_reason,
                                    )
                                )

            observation = agent_step.get("observation")
            if not isinstance(observation, dict):
                continue

            perspective, your_index = agent_perspective_state(observation, agent_index)
            summary[f"agent_perspective_{perspective.lower()}_count"] += 1
            if perspective == "MISMATCH":
                errors.append(
                    _error(
                        member=member,
                        replay_key=replay_key,
                        stage="agent_perspective_mismatch",
                        message="agent_index does not equal current.yourIndex",
                        step=step_index,
                        agent_index=agent_index,
                        yourIndex=your_index,
                    )
                )
                continue

            owner = resolve_turn_owner(observation.get("current"))
            if owner.conflict:
                summary["conflicting_turn_owner_count"] += 1
                errors.append(
                    _error(
                        member=member,
                        replay_key=replay_key,
                        stage="turn_owner_conflict",
                        message="Explicit owner conflicts with turn + firstPlayer formula",
                        step=step_index,
                        agent_index=agent_index,
                        formula_owner=owner.owner,
                        explicit_fields=dict(owner.explicit_fields),
                    )
                )
                continue

            fingerprint = observation_fingerprint(observation)
            if last_fingerprint.get(agent_index) == fingerprint:
                continue
            last_fingerprint[agent_index] = fingerprint
            try:
                parse_observation(observation)
            except Exception as exc:
                summary["parser_error_count"] += 1
                errors.append(
                    _error(
                        member=member,
                        replay_key=replay_key,
                        stage="observation_parse",
                        message=f"{type(exc).__name__}: {exc}",
                        step=step_index,
                        agent_index=agent_index,
                    )
                )
                continue

            select = observation.get("select")
            if not isinstance(select, dict):
                continue
            summary["decision_count"] += 1
            if owner.state.name == "UNKNOWN":
                summary["unknown_turn_owner_count"] += 1
            elif owner.source == "EXPLICIT_ENGINE_FIELD":
                summary["explicit_turn_owner_count"] += 1
            else:
                summary["inferred_turn_owner_count"] += 1
            pending[agent_index] = {
                "step": step_index,
                "select": select,
                "observation": observation,
                "turn_owner_state": owner.state.name,
                "turn_owner_source": owner.source or "UNKNOWN",
            }

    summary["unpaired_decision_count"] += len(pending)
    return {
        "member": member,
        "successful": True,
        "replay_key": replay_key,
        "canonical_hash": canonical_hash,
        "raw_content_hash": raw_content_hash,
        "has_external_id": has_external_id,
        "summary": summary,
        "action_rows": action_rows,
        "equivalence_rows": equivalence_rows,
        "policy_rows": policy_rows,
        "turn_rows": turn_rows,
        "decision_keys": decision_keys,
        "errors": errors,
    }


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _counter_values(summary: Counter[str], prefix: str, names: tuple[str, ...]) -> dict[str, int]:
    return {name: summary[f"{prefix}:{name}"] for name in names}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit Replay decision identity, perspective, turn owner, and labels."
    )
    parser.add_argument("archive", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/replay_decision_contract_audit_v2"),
    )
    parser.add_argument("--workers", type=int, default=max(1, min(4, mp.cpu_count())))
    parser.add_argument("--max-replays", type=int)
    args = parser.parse_args()

    with zipfile.ZipFile(args.archive) as archive:
        members = [name for name in archive.namelist() if name.endswith(".json")]
    if args.max_replays is not None:
        members = members[: args.max_replays]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for child in args.output_dir.iterdir():
        if child.is_file() and child.name not in OUTPUT_FILES:
            child.unlink()

    summary: Counter[str] = Counter()
    action_rows: Counter[tuple[int, int, int, str]] = Counter()
    equivalence_rows: Counter[tuple[int, int, int, str, str]] = Counter()
    policy_rows: Counter[tuple[str, str, str, int, int, int]] = Counter()
    turn_rows: Counter[tuple[int, int, int, str, str]] = Counter()
    errors: list[dict[str, Any]] = []
    seen_no_id_hashes: set[str] = set()
    seen_decision_keys: set[tuple[str, int | None, int, int, int]] = set()
    successful_replay_count = 0
    duplicate_replay_content_count = 0

    with mp.Pool(
        args.workers,
        initializer=_init_worker,
        initargs=(str(args.archive),),
    ) as pool:
        for index, result in enumerate(pool.imap(_audit_member, members, chunksize=1), start=1):
            errors.extend(result["errors"])
            if not result["successful"]:
                if index % 100 == 0:
                    print(json.dumps({"processed": index, "total": len(members)}), flush=True)
                continue
            successful_replay_count += 1
            if not result["has_external_id"]:
                canonical_hash = result["canonical_hash"]
                if canonical_hash in seen_no_id_hashes:
                    duplicate_replay_content_count += 1
                    if index % 100 == 0:
                        print(json.dumps({"processed": index, "total": len(members)}), flush=True)
                    continue
                seen_no_id_hashes.add(canonical_hash)

            summary.update(result["summary"])
            action_rows.update(result["action_rows"])
            equivalence_rows.update(result["equivalence_rows"])
            policy_rows.update(result["policy_rows"])
            turn_rows.update(result["turn_rows"])
            for key in result["decision_keys"]:
                if key in seen_decision_keys:
                    summary["duplicate_decision_key_count"] += 1
                    errors.append(
                        _error(
                            member=result["member"],
                            replay_key=result["replay_key"],
                            stage="duplicate_decision_key",
                            message="DecisionKey already appeared in the audit",
                            decision_key={
                                "replay_key": key[0],
                                "episode_id": key[1],
                                "decision_step_index": key[2],
                                "action_step_index": key[3],
                                "player_index": key[4],
                            },
                        )
                    )
                else:
                    seen_decision_keys.add(key)
            if index % 100 == 0:
                print(
                    json.dumps(
                        {
                            "processed": index,
                            "total": len(members),
                            "decisions": summary["decision_count"],
                        }
                    ),
                    flush=True,
                )

    _write_csv(
        args.output_dir / "action_semantics.csv",
        ["select_type", "select_context", "option_type", "action_semantics", "decision_count"],
        [
            {
                "select_type": key[0],
                "select_context": key[1],
                "option_type": key[2],
                "action_semantics": key[3],
                "decision_count": count,
            }
            for key, count in sorted(action_rows.items())
        ],
    )
    _write_csv(
        args.output_dir / "equivalence_resolution.csv",
        [
            "select_type",
            "select_context",
            "option_type",
            "action_semantics",
            "resolution_status",
            "decision_count",
        ],
        [
            {
                "select_type": key[0],
                "select_context": key[1],
                "option_type": key[2],
                "action_semantics": key[3],
                "resolution_status": key[4],
                "decision_count": count,
            }
            for key, count in sorted(equivalence_rows.items())
        ],
    )
    _write_csv(
        args.output_dir / "policy_mask_reasons.csv",
        [
            "policy_mask_reason",
            "action_semantics",
            "resolution_status",
            "option_count",
            "equivalence_class_count",
            "selected_count",
            "decision_count",
        ],
        [
            {
                "policy_mask_reason": key[0],
                "action_semantics": key[1],
                "resolution_status": key[2],
                "option_count": key[3],
                "equivalence_class_count": key[4],
                "selected_count": key[5],
                "decision_count": count,
            }
            for key, count in sorted(policy_rows.items())
        ],
    )
    _write_csv(
        args.output_dir / "turn_owner_audit.csv",
        [
            "select_type",
            "select_context",
            "option_type",
            "turn_owner_state",
            "turn_owner_source",
            "decision_count",
        ],
        [
            {
                "select_type": key[0],
                "select_context": key[1],
                "option_type": key[2],
                "turn_owner_state": key[3],
                "turn_owner_source": key[4],
                "decision_count": count,
            }
            for key, count in sorted(turn_rows.items())
        ],
    )
    with (args.output_dir / "errors.jsonl").open("w", encoding="utf-8") as handle:
        for row in errors:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    action_semantics_counts = _counter_values(summary, "action_semantics", SEMANTICS)
    resolution_counts = _counter_values(summary, "resolution", RESOLUTION_STATES)
    policy_mask_reason_counts = {
        key.split(":", 1)[1]: count
        for key, count in sorted(summary.items())
        if key.startswith("policy_mask_reason:")
    }
    report = {
        "schema_version": "replay_decision_contract_audit_v2",
        "archive": str(args.archive),
        "requested_replay_count": len(members),
        "successful_replay_count": successful_replay_count,
        "duplicate_replay_content_count": duplicate_replay_content_count,
        "decision_count": summary["decision_count"],
        "paired_decision_count": summary["paired_decision_count"],
        "unpaired_decision_count": summary["unpaired_decision_count"],
        "parser_error_count": summary["parser_error_count"],
        "action_alignment_error_count": summary["action_alignment_error_count"],
        "invalid_action_index_count": summary["invalid_action_index_count"],
        "duplicate_decision_key_count": summary["duplicate_decision_key_count"],
        "agent_perspective_match_count": summary["agent_perspective_match_count"],
        "agent_perspective_mismatch_count": summary["agent_perspective_mismatch_count"],
        "agent_perspective_unknown_count": summary["agent_perspective_unknown_count"],
        "explicit_turn_owner_count": summary["explicit_turn_owner_count"],
        "inferred_turn_owner_count": summary["inferred_turn_owner_count"],
        "unknown_turn_owner_count": summary["unknown_turn_owner_count"],
        "conflicting_turn_owner_count": summary["conflicting_turn_owner_count"],
        "FULLY_RESOLVED_count": resolution_counts["FULLY_RESOLVED"],
        "PARTIALLY_RESOLVED_count": resolution_counts["PARTIALLY_RESOLVED"],
        "UNRESOLVED_count": resolution_counts["UNRESOLVED"],
        "policy_mask_reason_counts": policy_mask_reason_counts,
        "action_semantics_counts": action_semantics_counts,
        "invariant_violation_count": summary["invariant_violation_count"],
        "error_count": len(errors),
        "hard_zero_checks": {
            field: summary[field] == 0 for field in HARD_ZERO_FIELDS
        },
        "audit_passed": all(summary[field] == 0 for field in HARD_ZERO_FIELDS),
    }
    (args.output_dir / "audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return 0 if report["audit_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
