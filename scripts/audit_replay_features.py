from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.dynamic_card_dataset import AttackCostCatalog, SPECIAL_ENERGY_RESOLUTION_REASONS
from data.replay_dataset import ReplayDecisionDataset
from data.state_schema import ENERGY_TYPE_NAMES, SPECIAL_CONDITION_NAMES


def percentile(values: list[float | int], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * q)))
    return ordered[index]


def distribution(values: list[float | int]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "mean": float(mean(values)) if values else 0.0,
        "p50": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "p99": percentile(values, 0.99),
        "min": min(values) if values else 0,
        "max": max(values) if values else 0,
    }


def ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def optional_ratio(numerator: int, denominator: int, *, checked: bool = True) -> float | None:
    if not checked or denominator == 0:
        return None
    return float(numerator / denominator)


def load_known_static_ids(path: Path) -> set[int]:
    if not path.exists():
        return set()
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "cards" in raw:
        return {int(card["card_id"]) for card in raw["cards"] if card.get("card_id") is not None}
    if isinstance(raw, dict) and "card_id_to_index" in raw:
        raw = raw["card_id_to_index"]
    if isinstance(raw, dict):
        return {int(card_id) for card_id in raw}
    return set()


def _role(instance: Any) -> str:
    if instance.attachment_kind not in (None, 0):
        return "attachment"
    if instance.zone == "active":
        return "active"
    if instance.zone == "bench":
        return "bench"
    return "none"


def build_report(
    dataset: ReplayDecisionDataset | Any,
    known_static_ids: set[int],
    catalog: AttackCostCatalog | None = None,
) -> dict[str, Any]:
    values: dict[str, list[float | int]] = {
        "instances_per_decision": [],
        "visible_instances_per_decision": [],
        "hidden_instances_per_decision": [],
        "events_per_decision": [],
        "options_per_decision": [],
        "current_hp": [],
        "max_hp": [],
        "damage": [],
        "hp_ratio": [],
        "damage_ratio": [],
        "energy_count": [],
        "tool_count": [],
        "pre_evolution_count": [],
        "copy_count": [],
    }
    counters: dict[str, Counter[Any]] = {
        "source_date": Counter(),
        "zone": Counter(),
        "field_role": Counter(),
        "owner": Counter(),
        "visibility": Counter(),
        "knowledge": Counter(),
        "energy_type": Counter(),
        "special_condition": Counter(),
        "event_type": Counter(),
        "option_type": Counter(),
        "select_type": Counter(),
        "select_context": Counter(),
        "parser_error_type": Counter(),
        "energy_unresolved_reason": Counter(),
        "training_payment_unsupervised_reason": Counter(),
        "catalog_alignment_anomaly": Counter(),
    }
    unique_episodes: set[str] = set()
    unique_source_paths: set[str] = set()
    missing_episode_key_samples = 0
    missing_source_date_samples = 0
    unique_visible_card_ids: set[int] = set()
    static_lookup_total = 0
    static_lookup_known = 0
    detail_alignment_total = 0
    detail_alignment_known = 0
    attack_detail_total = 0
    attack_cost_aligned = 0
    pokemon_count = 0
    hp_known = 0
    max_hp_known = 0
    ratio_known = 0
    energy_known = 0
    tool_known = 0
    evolution_known = 0
    condition_known = 0
    appear_known = 0
    appear_true = 0
    serial_known = 0
    serial_missing = 0
    duplicate_serial_rows = 0
    anonymous_instances = 0
    unknown_static_ids = 0
    empty_detail_instances = 0
    special_energy_attachments = 0
    unresolved_special_energy_attachments = 0
    payment_candidates = 0
    payment_resolved = 0
    payment_candidate_details = 0
    payment_cost_known_details = 0
    payment_resolved_details = 0
    payment_unresolved_details = 0
    special_energy_candidate_details = 0
    special_energy_unresolved_details = 0
    special_energy_dynamic_units = 0
    training_attack_detail_count = 0
    training_payment_supervision_count = 0
    training_unsupervised_attack_detail_count = 0

    if catalog is not None:
        counters["catalog_alignment_anomaly"].update(
            anomaly.get("kind", "unknown")
            for anomaly in catalog.static_catalog.alignment_anomalies
        )

    for error in dataset.summary.parser_errors:
        error_text = str(error.get("error", "unknown"))
        counters["parser_error_type"].update([error_text.split(":", 1)[0]])

    for sample in dataset.samples:
        parsed = sample.parsed
        instances = parsed.card_instances
        source_date = getattr(sample, "source_date", None)
        source_path = getattr(sample, "source_path", None)
        counters["source_date"].update([source_date or "unknown"])
        missing_source_date_samples += int(source_date is None)
        if source_path is not None:
            unique_source_paths.add(str(source_path))
        episode_key = getattr(sample, "episode_key", None)
        if episode_key is None:
            episode_id = getattr(sample, "episode_id", None)
            replay_id = getattr(sample, "replay_id", None)
            if episode_id is not None:
                episode_key = f"episode:{episode_id}"
            elif replay_id is not None:
                episode_key = f"replay:{replay_id}"
            elif source_path is not None:
                episode_key = f"path:{source_path}"
        if episode_key is not None:
            unique_episodes.add(str(episode_key))
        else:
            missing_episode_key_samples += 1
        values["instances_per_decision"].append(len(instances))
        values["visible_instances_per_decision"].append(sum(item.is_visible for item in instances))
        values["hidden_instances_per_decision"].append(sum(not item.is_visible for item in instances))
        values["events_per_decision"].append(len(parsed.events))
        values["options_per_decision"].append(len(parsed.select_options))
        counters["select_type"].update([sample.select_type])
        counters["select_context"].update([sample.select_context])
        serial_counts = Counter(
            int(instance.serial)
            for instance in instances
            if instance.serial is not None
        )
        duplicate_serial_rows += sum(count - 1 for count in serial_counts.values() if count > 1)

        for instance in instances:
            counters["zone"].update([instance.zone])
            counters["field_role"].update([_role(instance)])
            counters["owner"].update([
                "self" if instance.relative_player == 0 else "opponent" if instance.relative_player == 1 else "unknown"
            ])
            counters["visibility"].update(["visible" if instance.is_visible else "hidden"])
            counters["knowledge"].update([
                "known_id" if instance.card_id is not None else "anonymous"
            ])
            if instance.card_id is None:
                anonymous_instances += 1
            else:
                card_id = int(instance.card_id)
                if instance.is_visible:
                    unique_visible_card_ids.add(card_id)
                    static_lookup_total += 1
                    if card_id in known_static_ids:
                        static_lookup_known += 1
                    elif known_static_ids:
                        unknown_static_ids += 1
                if catalog is not None and catalog.card_known(card_id):
                    detail_alignment_total += 1
                    if card_id in catalog.static_catalog.details_by_card_id:
                        detail_alignment_known += 1
                    if not catalog.detail_exists(card_id):
                        empty_detail_instances += 1
                    for attack in catalog.attack_details(card_id):
                        attack_detail_total += 1
                        attack_cost_aligned += int(attack.cost_known)

            if instance.serial is None:
                serial_missing += 1
            else:
                serial_known += 1
            if instance.copy_count is not None:
                values["copy_count"].append(instance.copy_count)

            if catalog is not None:
                training_energy_reason = catalog.energy_resolution_reason(instance)
                for attack in catalog.attack_details(instance.card_id):
                    training_attack_detail_count += 1
                    if attack.cost_known and training_energy_reason is None:
                        training_payment_supervision_count += 1
                    else:
                        training_unsupervised_attack_detail_count += 1
                        reason = "attack_cost_unknown" if not attack.cost_known else training_energy_reason
                        counters["training_payment_unsupervised_reason"].update([reason or "unknown"])

            if instance.is_pokemon:
                pokemon_count += 1
                hp_known += int(instance.hp is not None)
                max_hp_known += int(instance.max_hp is not None)
                if instance.hp is not None:
                    values["current_hp"].append(instance.hp)
                if instance.max_hp is not None:
                    values["max_hp"].append(instance.max_hp)
                if instance.hp is not None and instance.max_hp is not None:
                    damage = max(instance.max_hp - instance.hp, 0)
                    values["damage"].append(damage)
                    if instance.max_hp > 0:
                        ratio_known += 1
                        values["hp_ratio"].append(instance.hp / instance.max_hp)
                        values["damage_ratio"].append(damage / instance.max_hp)
                energy_known += int(instance.energy_counts_valid)
                tool_known += int(instance.tools_valid)
                evolution_known += int(instance.pre_evolution_valid)
                condition_known += int(instance.special_conditions_valid)
                appear_known += int(instance.appear_this_turn_valid)
                appear_true += int(instance.appear_this_turn)
                if instance.energy_counts_valid:
                    values["energy_count"].append(sum(instance.energy_counts))
                    for index, count in enumerate(instance.energy_counts[: len(ENERGY_TYPE_NAMES)]):
                        if count:
                            counters["energy_type"].update({ENERGY_TYPE_NAMES[index]: int(count)})
                if instance.tools_valid:
                    values["tool_count"].append(instance.tool_count)
                if instance.pre_evolution_valid:
                    values["pre_evolution_count"].append(instance.pre_evolution_count)
                if instance.special_conditions_valid:
                    for name, active in zip(SPECIAL_CONDITION_NAMES, instance.special_conditions):
                        if active:
                            counters["special_condition"].update([name])
                if catalog is not None:
                    attacks = catalog.attack_details(instance.card_id)
                    has_special_energy = any(
                        int(value) > 0
                        for value in instance.energy_counts[10 : len(ENERGY_TYPE_NAMES)]
                    ) or any(
                        str(
                            catalog.static_catalog.card_records.get(int(energy_card_id), {}).get(
                                "card_type", ""
                            )
                        ).upper()
                        == "SPECIAL_ENERGY"
                        for energy_card_id in instance.energy_card_ids
                    )
                    if attacks:
                        payment_candidates += 1
                        energy_reason = catalog.energy_resolution_reason(instance)
                        payment_resolved += int(energy_reason is None)
                        for attack in attacks:
                            payment_candidate_details += 1
                            special_energy_candidate_details += int(has_special_energy)
                            reason = energy_reason
                            if not attack.cost_known:
                                reason = "attack_cost_unknown"
                            else:
                                payment_cost_known_details += 1
                            if reason is None:
                                payment_resolved_details += 1
                            else:
                                payment_unresolved_details += 1
                                counters["energy_unresolved_reason"].update([reason])
                                special_energy_unresolved_details += int(
                                    reason in SPECIAL_ENERGY_RESOLUTION_REASONS
                                )
                    special_energy_dynamic_units += sum(
                        int(value)
                        for value in instance.energy_counts[10 : len(ENERGY_TYPE_NAMES)]
                        if int(value) > 0
                    )
                if catalog is not None:
                    for energy_card_id in instance.energy_card_ids:
                        record = catalog.static_catalog.card_records.get(int(energy_card_id), {})
                        if str(record.get("card_type", "")).upper() == "SPECIAL_ENERGY":
                            special_energy_attachments += 1
                            unresolved = (
                                int(energy_card_id) in catalog.static_catalog.ambiguous_special_energy_ids
                                or int(energy_card_id) in catalog.static_catalog.unknown_special_energy_ids
                            )
                            unresolved_special_energy_attachments += int(unresolved)

        for event in parsed.events:
            counters["event_type"].update([event.event_type])
        for option in parsed.select_options:
            counters["option_type"].update([option.get("type", -1)])

    parser_errors = list(dataset.summary.parser_errors)
    static_checked = bool(known_static_ids)
    catalog_checked = catalog is not None
    layout = catalog.static_catalog.detail_layout if catalog is not None else None
    alignment_anomalies = catalog.static_catalog.alignment_anomalies if catalog is not None else []
    return {
        "schema_version": "dynamic_replay_feature_audit_v1",
        "replay_count": dataset.summary.replay_count,
        "episode_count": len(unique_episodes),
        "decision_sample_count": len(dataset.samples),
        "skipped_no_select": dataset.summary.skipped_no_select,
        "parser_error_count": len(parser_errors),
        "parser_errors": parser_errors[:100],
        "parser_error_types": dict(counters["parser_error_type"]),
        "source_dates": dict(counters["source_date"]),
        "source_provenance": {
            "unique_source_paths": len(unique_source_paths),
            "missing_source_date_samples": missing_source_date_samples,
            "missing_episode_key_samples": missing_episode_key_samples,
        },
        "unique_visible_card_ids": len(unique_visible_card_ids),
        "anonymous_instance_count": anonymous_instances,
        "serial_coverage": {
            "known": serial_known,
            "missing": serial_missing,
            "coverage": optional_ratio(serial_known, serial_known + serial_missing),
            "duplicate_serial_rows_within_decision": duplicate_serial_rows,
        },
        "unknown_static_instance_count": unknown_static_ids,
        "empty_detail_instance_count": empty_detail_instances,
        "static_lookup": {
            "checked": static_checked,
            "known": static_lookup_known if static_checked else None,
            "total_visible": static_lookup_total,
            "coverage": optional_ratio(static_lookup_known, static_lookup_total, checked=static_checked),
        },
        "detail_alignment": {
            "checked": catalog_checked,
            "aligned": detail_alignment_known if catalog_checked else None,
            "known_card_instances": detail_alignment_total if catalog_checked else None,
            "metadata_card_coverage": optional_ratio(
                detail_alignment_known,
                detail_alignment_total,
                checked=catalog_checked,
            ),
            "coverage": (
                optional_ratio(
                    detail_alignment_known,
                    detail_alignment_total,
                    checked=catalog_checked,
                )
                if layout is not None and layout.all_type_slots_verified
                else None
            ),
            "attack_cost_aligned": attack_cost_aligned if catalog_checked else None,
            "attack_detail_total": attack_detail_total if catalog_checked else None,
            "attack_cost_coverage": optional_ratio(
                attack_cost_aligned,
                attack_detail_total,
                checked=catalog_checked,
            ),
            "physical_layout": layout.as_dict() if layout is not None else None,
            "physical_width_validated": catalog_checked,
            "all_detail_physical_slots_verified": (
                layout.all_type_slots_verified if layout is not None else None
            ),
            "physical_slot_coverage": (
                1.0 if layout is not None and layout.all_type_slots_verified else None
            ),
            "alignment_anomaly_count": len(alignment_anomalies) if catalog_checked else None,
            "alignment_anomaly_types": (
                dict(counters["catalog_alignment_anomaly"]) if catalog_checked else None
            ),
            "alignment_anomaly_examples": alignment_anomalies[:100] if catalog_checked else None,
        },
        "field_coverage": {
            "pokemon_instances": pokemon_count,
            "hp": ratio(hp_known, pokemon_count),
            "max_hp": ratio(max_hp_known, pokemon_count),
            "hp_and_damage_ratio": ratio(ratio_known, pokemon_count),
            "energy_counts": ratio(energy_known, pokemon_count),
            "tool": ratio(tool_known, pokemon_count),
            "pre_evolution": ratio(evolution_known, pokemon_count),
            "special_conditions": ratio(condition_known, pokemon_count),
            "appear_this_turn": ratio(appear_known, pokemon_count),
            "appear_this_turn_true": appear_true,
        },
        "energy_resolution": {
            "checked": catalog_checked,
            "payment_candidate_instances": payment_candidates if catalog_checked else None,
            "resolved_instances": payment_resolved if catalog_checked else None,
            "resolved_ratio": optional_ratio(
                payment_resolved,
                payment_candidates,
                checked=catalog_checked,
            ),
            "unresolved_ratio": optional_ratio(
                payment_candidates - payment_resolved,
                payment_candidates,
                checked=catalog_checked,
            ),
            "payment_candidate_details": payment_candidate_details if catalog_checked else None,
            "cost_known_details": payment_cost_known_details if catalog_checked else None,
            "supervised_details": payment_resolved_details if catalog_checked else None,
            "unresolved_details": payment_unresolved_details if catalog_checked else None,
            "detail_unresolved_ratio": optional_ratio(
                payment_unresolved_details,
                payment_candidate_details,
                checked=catalog_checked,
            ),
            "unresolved_reasons": (
                dict(counters["energy_unresolved_reason"]) if catalog_checked else None
            ),
            "special_energy_dynamic_units": special_energy_dynamic_units if catalog_checked else None,
            "special_energy_candidate_details": (
                special_energy_candidate_details if catalog_checked else None
            ),
            "special_energy_unresolved_details": (
                special_energy_unresolved_details if catalog_checked else None
            ),
            "special_energy_unresolved_detail_ratio": optional_ratio(
                special_energy_unresolved_details,
                special_energy_candidate_details,
                checked=catalog_checked,
            ),
            "special_energy_attachment_count": special_energy_attachments if catalog_checked else None,
            "unresolved_special_energy_attachment_count": (
                unresolved_special_energy_attachments if catalog_checked else None
            ),
            "unresolved_special_energy_ratio": optional_ratio(
                unresolved_special_energy_attachments,
                special_energy_attachments,
                checked=catalog_checked,
            ),
        },
        "training_payment_supervision": {
            "checked": catalog_checked,
            "attack_detail_count": training_attack_detail_count if catalog_checked else None,
            "payment_supervision_count": (
                training_payment_supervision_count if catalog_checked else None
            ),
            "unsupervised_attack_detail_count": (
                training_unsupervised_attack_detail_count if catalog_checked else None
            ),
            "supervision_ratio": optional_ratio(
                training_payment_supervision_count,
                training_attack_detail_count,
                checked=catalog_checked,
            ),
            "unsupervised_ratio": optional_ratio(
                training_unsupervised_attack_detail_count,
                training_attack_detail_count,
                checked=catalog_checked,
            ),
            "unsupervised_reasons": (
                dict(counters["training_payment_unsupervised_reason"])
                if catalog_checked
                else None
            ),
        },
        "distributions": {name: distribution(rows) for name, rows in values.items()},
        "counters": {
            name: [[key, value] for key, value in rows.most_common(100)]
            for name, rows in counters.items()
            if name not in {
                "source_date",
                "parser_error_type",
                "energy_unresolved_reason",
                "training_payment_unsupervised_reason",
                "catalog_alignment_anomaly",
            }
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit model-facing fields across replay decision points.")
    parser.add_argument("paths", nargs="+", type=Path, help="Replay JSON/JSONL files or directories.")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--static-id-map",
        type=Path,
        default=Path("outputs/card_pretrain/artifacts/card_id_to_index.json"),
    )
    parser.add_argument("--card-records", type=Path, default=Path("artifacts/card_data/card_records.json"))
    parser.add_argument(
        "--detail-metadata",
        type=Path,
        default=Path("outputs/card_pretrain/artifacts/card_detail_metadata.json"),
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/audit/replay_feature_audit.json"))
    args = parser.parse_args()

    known_ids = load_known_static_ids(args.static_id_map)
    catalog = None
    if args.card_records.exists() and args.detail_metadata.exists() and args.static_id_map.exists():
        catalog = AttackCostCatalog.from_files(args.card_records, args.detail_metadata, args.static_id_map)
    dataset = ReplayDecisionDataset.from_paths(args.paths, max_samples=args.max_samples)
    report = build_report(dataset, known_ids, catalog)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print("wrote", args.output)


if __name__ == "__main__":
    main()
