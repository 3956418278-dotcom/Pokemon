from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any, Sequence, TYPE_CHECKING

from .state_schema import ENERGY_TYPE_NAMES, CardDynamicBatch, CardInstanceState, collate_card_dynamic

if TYPE_CHECKING:
    import torch

    from .replay_dataset import ReplayDecisionSample


DETAIL_TYPE_ORDER = ("attack", "ability", "special_effect")
DETAIL_SLOT_RULE = "export_batch_group_padded_attack_ability_special_effect"
SPECIAL_ENERGY_RESOLUTION_REASONS = {
    "special_energy_count_vector",
    "ambiguous_special_energy_card",
    "unknown_special_energy_provider",
}
ENERGY_TYPE_INDEX = {
    **{name: index for index, name in enumerate(ENERGY_TYPE_NAMES)},
    **{
        symbol: index
        for index, symbol in enumerate(("C", "G", "R", "W", "L", "P", "F", "D", "M", "N", "Y", "A"))
    },
}


@dataclass(frozen=True)
class AttackDetail:
    detail_index: int
    attack_id: int
    energy_cost: tuple[int, ...]
    cost_known: bool
    metadata_detail_index: int | None = None


@dataclass(frozen=True)
class DetailLayout:
    """Describe the physical static-detail table without pretending packed JSON is physical.

    The legacy exporter pads attack, ability and special-effect groups to the
    maxima of each export batch, then concatenates the groups. Attack slots are
    therefore stable because attacks are first. Ability/effect offsets require
    the export-batch maxima and cannot be reconstructed from the legacy packed
    JSON metadata alone.
    """

    physical_width: int
    physical_width_source: str
    metadata_packed_width: int
    type_capacities: dict[str, int]
    slot_rule: str = DETAIL_SLOT_RULE
    attack_slots_verified: bool = True
    all_type_slots_verified: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "physical_width": self.physical_width,
            "physical_width_source": self.physical_width_source,
            "metadata_packed_width": self.metadata_packed_width,
            "type_capacities": dict(self.type_capacities),
            "slot_rule": self.slot_rule,
            "attack_slots_verified": self.attack_slots_verified,
            "all_type_slots_verified": self.all_type_slots_verified,
        }


@dataclass
class StaticCardCatalog:
    card_id_to_index: dict[int, int]
    card_records: dict[int, dict[str, Any]]
    details_by_card_id: dict[int, list[dict[str, Any]]]
    attack_details_by_card_id: dict[int, list[AttackDetail]]
    ambiguous_special_energy_ids: set[int]
    unknown_special_energy_ids: set[int]
    max_details: int
    detail_layout: DetailLayout
    alignment_anomalies: list[dict[str, Any]]
    invalid_detail_slots_by_card_id: dict[int, set[int]]

    @classmethod
    def from_artifact_dir(cls, artifact_dir: str | Path) -> "StaticCardCatalog":
        artifact_dir = Path(artifact_dir)
        card_records = artifact_dir / "card_records.json"
        if not card_records.exists():
            matches = sorted(artifact_dir.parent.rglob("card_records.json"))
            if not matches:
                raise FileNotFoundError(f"card_records.json was not found below {artifact_dir.parent}")
            card_records = matches[0]
        return cls.from_files(
            card_records,
            artifact_dir / "card_detail_metadata.json",
            artifact_dir / "card_id_to_index.json",
        )

    @classmethod
    def from_files(
        cls,
        card_records_path: str | Path,
        detail_metadata_path: str | Path,
        card_id_to_index_path: str | Path,
    ) -> "StaticCardCatalog":
        card_records_path = Path(card_records_path)
        detail_metadata_path = Path(detail_metadata_path)
        card_id_to_index_path = Path(card_id_to_index_path)
        records_raw = json.loads(card_records_path.read_text(encoding="utf-8"))
        if isinstance(records_raw, dict):
            records_raw = records_raw.get("cards", records_raw.get("records", []))
        detail_raw = json.loads(detail_metadata_path.read_text(encoding="utf-8"))
        detail_rows = detail_raw.get("cards", detail_raw) if isinstance(detail_raw, dict) else detail_raw
        mapping_raw = json.loads(card_id_to_index_path.read_text(encoding="utf-8"))
        if isinstance(mapping_raw, dict) and "card_id_to_index" in mapping_raw:
            mapping_raw = mapping_raw["card_id_to_index"]
        if not isinstance(records_raw, list) or not isinstance(detail_rows, list) or not isinstance(mapping_raw, dict):
            raise ValueError("static catalog artifacts have invalid top-level shapes")

        card_id_to_index = {int(card_id): int(index) for card_id, index in mapping_raw.items()}
        indices = list(card_id_to_index.values())
        if any(index < 0 for index in indices) or len(indices) != len(set(indices)):
            raise ValueError("static card mapping indices must be unique non-negative integers")
        if set(indices) != set(range(len(indices))):
            raise ValueError("static card mapping indices must be contiguous from zero")

        def rows_by_card_id(rows: list[dict[str, Any]], artifact_name: str) -> dict[int, dict[str, Any]]:
            result: dict[int, dict[str, Any]] = {}
            for row in rows:
                if not isinstance(row, dict) or row.get("card_id") is None:
                    raise ValueError(f"{artifact_name} contains a row without card_id")
                card_id = int(row["card_id"])
                if card_id in result:
                    raise ValueError(f"{artifact_name} contains duplicate card_id {card_id}")
                result[card_id] = dict(row)
            return result

        records = rows_by_card_id(records_raw, "card_records")
        detail_card_rows = rows_by_card_id(detail_rows, "card_detail_metadata")
        details_by_card_id: dict[int, list[dict[str, Any]]] = {}
        metadata_packed_width = 0
        type_capacities = {detail_type: 0 for detail_type in DETAIL_TYPE_ORDER}
        for card_id, card_row in detail_card_rows.items():
            raw_details = card_row.get("details", [])
            if not isinstance(raw_details, list):
                raise ValueError(f"card {card_id} details must be a list")
            details = [dict(detail) for detail in raw_details]
            metadata_indices: list[int] = []
            type_counts = Counter()
            for detail in details:
                try:
                    metadata_index = int(detail.get("detail_index"))
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"card {card_id} has invalid detail_index") from exc
                if metadata_index < 0:
                    raise ValueError(f"card {card_id} has negative detail_index {metadata_index}")
                detail_type = str(detail.get("detail_type", "")).strip().lower()
                if detail_type not in DETAIL_TYPE_ORDER:
                    raise ValueError(f"card {card_id} has unknown detail_type {detail_type!r}")
                metadata_indices.append(metadata_index)
                type_counts[detail_type] += 1
            if len(metadata_indices) != len(set(metadata_indices)):
                raise ValueError(f"card {card_id} contains duplicate detail_index values")
            if sorted(metadata_indices) != list(range(len(metadata_indices))):
                raise ValueError(f"card {card_id} packed detail indices must be contiguous from zero")
            metadata_packed_width = max(metadata_packed_width, len(details))
            for detail_type in DETAIL_TYPE_ORDER:
                type_capacities[detail_type] = max(type_capacities[detail_type], int(type_counts[detail_type]))
            details_by_card_id[card_id] = details

        missing_records = set(card_id_to_index) - set(records)
        if missing_records:
            raise ValueError(f"static mapping contains {len(missing_records)} cards missing from card_records.json")
        missing_detail_rows = set(card_id_to_index) - set(details_by_card_id)
        if missing_detail_rows:
            raise ValueError(
                f"static mapping contains {len(missing_detail_rows)} cards missing from card_detail_metadata.json"
            )

        inferred_capacity_width = sum(type_capacities.values())
        physical_width = inferred_capacity_width
        physical_width_source = "inferred_type_capacities"
        embedding_metadata_path = detail_metadata_path.with_name("card_embedding_metadata.json")
        if embedding_metadata_path.exists():
            embedding_metadata = json.loads(embedding_metadata_path.read_text(encoding="utf-8"))
            try:
                physical_width = int(embedding_metadata["max_detail_count"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("card_embedding_metadata.json has no valid max_detail_count") from exc
            physical_width_source = str(embedding_metadata_path)
            artifact_card_count = embedding_metadata.get("card_count")
            if artifact_card_count is not None and int(artifact_card_count) != len(card_id_to_index):
                raise ValueError(
                    "card_embedding_metadata card_count does not match card_id_to_index: "
                    f"{artifact_card_count} != {len(card_id_to_index)}"
                )
        if physical_width < metadata_packed_width:
            raise ValueError(
                "physical detail width is smaller than packed metadata width: "
                f"{physical_width} < {metadata_packed_width}"
            )
        if inferred_capacity_width and physical_width > inferred_capacity_width:
            raise ValueError(
                "physical detail width exceeds the maximum grouped detail capacity: "
                f"{physical_width} > {inferred_capacity_width}"
            )
        detail_layout = DetailLayout(
            physical_width=physical_width,
            physical_width_source=physical_width_source,
            metadata_packed_width=metadata_packed_width,
            type_capacities=type_capacities,
            all_type_slots_verified=bool(
                isinstance(detail_raw, dict)
                and detail_raw.get("physical_slot_rule") == "fixed_global_type_capacities"
                and all(
                    "physical_detail_index" in detail
                    for details in details_by_card_id.values()
                    for detail in details
                )
            ),
        )

        attack_details: dict[int, list[AttackDetail]] = {}
        alignment_anomalies: list[dict[str, Any]] = []
        invalid_detail_slots_by_card_id: dict[int, set[int]] = {}
        for card_id, details in details_by_card_id.items():
            try:
                attack_ids = [
                    int(detail["attack_id"])
                    for detail in details
                    if str(detail.get("detail_type", "")).lower() == "attack"
                    and detail.get("attack_id") is not None
                ]
            except (TypeError, ValueError) as exc:
                raise ValueError(f"card {card_id} has a non-integer attack_id in detail metadata") from exc
            if len(attack_ids) != len(set(attack_ids)):
                raise ValueError(f"card {card_id} has duplicate attack_ids in detail metadata")
            attack_costs: list[Any] = []
            for detail in details:
                if str(detail.get("detail_type", "")).lower() != "attack" or detail.get("attack_id") is None:
                    continue
                source = detail.get("energy_counts")
                if isinstance(source, (list, tuple)):
                    source = {
                        energy_type: int(count)
                        for energy_type, count in zip(ENERGY_TYPE_NAMES, source)
                        if int(count)
                    }
                attack_costs.append(source)
            rows: list[AttackDetail] = []
            seen_metadata_attack_ids: set[int] = set()
            attack_ordinal = 0
            for detail in details:
                if str(detail.get("detail_type", "")).lower() != "attack":
                    continue
                metadata_detail_index = int(detail["detail_index"])
                physical_detail_index = attack_ordinal
                attack_ordinal += 1
                if metadata_detail_index != physical_detail_index:
                    alignment_anomalies.append(
                        {
                            "kind": "attack_metadata_slot_mismatch",
                            "card_id": card_id,
                            "metadata_detail_index": metadata_detail_index,
                            "physical_detail_index": physical_detail_index,
                        }
                    )
                if not 0 <= physical_detail_index < physical_width:
                    raise ValueError(f"card {card_id} attack slot {physical_detail_index} exceeds physical width")
                try:
                    attack_id = int(detail.get("attack_id"))
                except (TypeError, ValueError):
                    invalid_detail_slots_by_card_id.setdefault(card_id, set()).add(physical_detail_index)
                    alignment_anomalies.append(
                        {
                            "kind": "invalid_attack_detail",
                            "card_id": card_id,
                            "metadata_detail_index": metadata_detail_index,
                            "attack_id": detail.get("attack_id"),
                            "attack_name": detail.get("attack_name"),
                        }
                    )
                    continue
                if attack_id in seen_metadata_attack_ids:
                    raise ValueError(f"card {card_id} repeats attack_id {attack_id} in detail metadata")
                seen_metadata_attack_ids.add(attack_id)
                cost = [0] * len(ENERGY_TYPE_NAMES)
                cost_known = attack_id in attack_ids
                if not cost_known:
                    alignment_anomalies.append(
                        {
                            "kind": "attack_id_missing_from_record",
                            "card_id": card_id,
                            "attack_id": attack_id,
                            "metadata_detail_index": metadata_detail_index,
                        }
                    )
                if cost_known:
                    cost_index = attack_ids.index(attack_id)
                    source = attack_costs[cost_index] if cost_index < len(attack_costs) else None
                    if source is None:
                        cost_known = False
                        alignment_anomalies.append(
                            {
                                "kind": "attack_cost_missing",
                                "card_id": card_id,
                                "attack_id": attack_id,
                            }
                        )
                    elif not isinstance(source, dict):
                        cost_known = False
                        alignment_anomalies.append(
                            {
                                "kind": "attack_cost_not_mapping",
                                "card_id": card_id,
                                "attack_id": attack_id,
                            }
                        )
                    else:
                        for raw_type, raw_count in source.items():
                            try:
                                normalized_type = str(raw_type).strip().upper()
                                energy_type = (
                                    ENERGY_TYPE_INDEX[normalized_type]
                                    if normalized_type in ENERGY_TYPE_INDEX
                                    else int(raw_type)
                                )
                                count = int(raw_count)
                            except (TypeError, ValueError):
                                cost_known = False
                                continue
                            if not 0 <= energy_type < len(cost) or count < 0:
                                cost_known = False
                                continue
                            cost[energy_type] = count
                        if not cost_known:
                            alignment_anomalies.append(
                                {
                                    "kind": "attack_cost_invalid_value",
                                    "card_id": card_id,
                                    "attack_id": attack_id,
                                }
                            )
                rows.append(
                    AttackDetail(
                        physical_detail_index,
                        attack_id,
                        tuple(cost),
                        cost_known,
                        metadata_detail_index,
                    )
                )
            attack_details[card_id] = rows

        ambiguous_special_energy_ids: set[int] = set()
        unknown_special_energy_ids: set[int] = set()
        for card_id, record in records.items():
            if str(record.get("card_type", "")).upper() != "SPECIAL_ENERGY":
                continue
            # Special Energy has no static provider type; its printed Type
            # column and effect remain in source metadata / DetailRecord.
            ambiguous_special_energy_ids.add(card_id)

        return cls(
            card_id_to_index=card_id_to_index,
            card_records=records,
            details_by_card_id=details_by_card_id,
            attack_details_by_card_id=attack_details,
            ambiguous_special_energy_ids=ambiguous_special_energy_ids,
            unknown_special_energy_ids=unknown_special_energy_ids,
            max_details=physical_width,
            detail_layout=detail_layout,
            alignment_anomalies=alignment_anomalies,
            invalid_detail_slots_by_card_id=invalid_detail_slots_by_card_id,
        )

    def card_known(self, card_id: int | None) -> bool:
        return card_id is not None and int(card_id) in self.card_id_to_index

    def detail_exists(self, card_id: int | None) -> bool:
        return card_id is not None and bool(self.details_by_card_id.get(int(card_id), []))


@dataclass
class AttackCostCatalog:
    static_catalog: StaticCardCatalog

    @classmethod
    def from_artifact_dir(cls, artifact_dir: str | Path) -> "AttackCostCatalog":
        return cls(StaticCardCatalog.from_artifact_dir(artifact_dir))

    @classmethod
    def from_files(
        cls,
        card_records_path: str | Path,
        detail_metadata_path: str | Path,
        card_id_to_index_path: str | Path,
    ) -> "AttackCostCatalog":
        return cls(StaticCardCatalog.from_files(card_records_path, detail_metadata_path, card_id_to_index_path))

    @property
    def max_details(self) -> int:
        return self.static_catalog.max_details

    def card_known(self, card_id: int | None) -> bool:
        return self.static_catalog.card_known(card_id)

    def detail_exists(self, card_id: int | None) -> bool:
        return self.static_catalog.detail_exists(card_id)

    def attack_details(self, card_id: int | None) -> list[AttackDetail]:
        if card_id is None:
            return []
        return self.static_catalog.attack_details_by_card_id.get(int(card_id), [])

    def invalid_detail_slots(self, card_id: int | None) -> set[int]:
        if card_id is None:
            return set()
        return set(self.static_catalog.invalid_detail_slots_by_card_id.get(int(card_id), set()))

    def energy_resolution_reason(self, instance: CardInstanceState) -> str | None:
        """Return why attack-energy supervision is unsafe, or ``None`` when safe."""
        if not instance.is_pokemon:
            return "not_pokemon"
        if not instance.energy_counts_valid:
            return "energy_counts_missing_or_invalid"
        counts = (list(instance.energy_counts) + [0] * len(ENERGY_TYPE_NAMES))[: len(ENERGY_TYPE_NAMES)]
        if any(value < 0 for value in counts):
            return "negative_energy_count"
        if counts[10] > 0 or counts[11] > 0:
            return "special_energy_count_vector"
        if not instance.energy_cards_valid:
            return "energy_cards_missing"
        if len(instance.energy_card_ids) != int(instance.energy_card_count):
            return "energy_card_id_count_mismatch"
        for raw_card_id in instance.energy_card_ids:
            card_id = int(raw_card_id)
            if card_id in self.static_catalog.ambiguous_special_energy_ids:
                return "ambiguous_special_energy_card"
            if card_id in self.static_catalog.unknown_special_energy_ids:
                return "unknown_special_energy_provider"
            record = self.static_catalog.card_records.get(card_id)
            if record is None:
                return "unknown_energy_card_id"
            if str(record.get("card_type", "")).upper() not in {"BASIC_ENERGY", "SPECIAL_ENERGY"}:
                return "non_energy_card_in_energy_attachments"
        return None

    def energy_payment_is_resolved(self, instance: CardInstanceState) -> bool:
        return self.energy_resolution_reason(instance) is None

    @staticmethod
    def resolve_payment(energy_counts: Sequence[int], energy_cost: Sequence[int]) -> tuple[bool, list[int]]:
        available = [max(int(value), 0) for value in energy_counts]
        costs = [max(int(value), 0) for value in energy_cost]
        size = max(len(available), len(costs), len(ENERGY_TYPE_NAMES))
        available.extend([0] * (size - len(available)))
        costs.extend([0] * (size - len(costs)))
        remaining = [0] * size
        for energy_type in range(1, min(10, size)):
            used = min(available[energy_type], costs[energy_type])
            available[energy_type] -= used
            remaining[energy_type] = costs[energy_type] - used
        spare_for_colorless = sum(available[: min(10, size)])
        remaining[0] = max(costs[0] - spare_for_colorless, 0)
        for energy_type in range(10, size):
            remaining[energy_type] = max(costs[energy_type] - available[energy_type], 0)
        return not any(remaining), remaining[: len(ENERGY_TYPE_NAMES)]


@dataclass
class DynamicCardTrainingBatch:
    card_dynamic_batch: CardDynamicBatch
    sample_indices: "torch.Tensor"
    static_known_mask: "torch.Tensor"
    detail_exists_mask: "torch.Tensor"
    detail_valid_mask: "torch.Tensor"
    energy_resolved_mask: "torch.Tensor"
    attack_ids: "torch.Tensor"
    attack_costs: "torch.Tensor"
    attack_detail_mask: "torch.Tensor"
    payable_targets: "torch.Tensor"
    energy_remaining_targets: "torch.Tensor"
    payment_supervision_mask: "torch.Tensor"
    hp_targets: "torch.Tensor"
    hp_mask: "torch.Tensor"
    zone_targets: "torch.Tensor"
    role_targets: "torch.Tensor"

    @property
    def dynamic_batch(self) -> CardDynamicBatch:
        return self.card_dynamic_batch

    @property
    def instance_count(self) -> int:
        return self.card_dynamic_batch.batch_size

    def to(self, device: "torch.device | str") -> "DynamicCardTrainingBatch":
        values: dict[str, Any] = {}
        for item in fields(self):
            value = getattr(self, item.name)
            values[item.name] = value.to(device)
        return DynamicCardTrainingBatch(**values)


def collate_dynamic_card_samples(
    samples: Sequence["ReplayDecisionSample"],
    catalog: AttackCostCatalog,
    max_details: int | None = None,
) -> DynamicCardTrainingBatch:
    import torch

    detail_count = int(max_details if max_details is not None else catalog.max_details)
    instances: list[CardInstanceState] = []
    appearance_features: list[list[float]] = []
    sample_indices: list[int] = []
    for sample_index, sample in enumerate(samples):
        sample_appearance = sample.memory_after.appearance_features(sample.parsed.card_instances)
        for instance, appearance in zip(sample.parsed.card_instances, sample_appearance):
            static_known = catalog.card_known(instance.card_id)
            detail_exists = catalog.detail_exists(instance.card_id)
            energy_resolved = catalog.energy_payment_is_resolved(instance)
            instances.append(
                replace(
                    instance,
                    static_artifact_known=static_known,
                    detail_exists=detail_exists,
                    energy_payment_resolved=energy_resolved,
                )
            )
            appearance_features.append(appearance)
            sample_indices.append(sample_index)

    dynamic_batch = collate_card_dynamic(instances, appearance_features)
    count = len(instances)
    attack_ids = torch.full((count, detail_count), -1, dtype=torch.long)
    attack_costs = torch.zeros(count, detail_count, len(ENERGY_TYPE_NAMES), dtype=torch.float32)
    attack_detail_mask = torch.zeros(count, detail_count, dtype=torch.float32)
    detail_valid_mask = torch.ones(count, detail_count, dtype=torch.float32)
    payable_targets = torch.zeros(count, detail_count, dtype=torch.float32)
    remaining_targets = torch.zeros(count, detail_count, len(ENERGY_TYPE_NAMES), dtype=torch.float32)
    supervision_mask = torch.zeros(count, detail_count, dtype=torch.float32)
    hp_targets = torch.zeros(count, 2, dtype=torch.float32)
    hp_mask = torch.zeros(count, dtype=torch.float32)

    for row_index, instance in enumerate(instances):
        for detail_index in catalog.invalid_detail_slots(instance.card_id):
            if 0 <= detail_index < detail_count:
                detail_valid_mask[row_index, detail_index] = 0.0
        if instance.hp is not None and instance.max_hp is not None and instance.max_hp > 0:
            hp_ratio = float(instance.hp) / float(instance.max_hp)
            hp_targets[row_index, 0] = hp_ratio
            hp_targets[row_index, 1] = max(1.0 - hp_ratio, 0.0)
            hp_mask[row_index] = 1.0
        for attack in catalog.attack_details(instance.card_id):
            if not 0 <= attack.detail_index < detail_count:
                continue
            detail_index = attack.detail_index
            attack_ids[row_index, detail_index] = attack.attack_id
            attack_costs[row_index, detail_index] = torch.tensor(attack.energy_cost, dtype=torch.float32)
            attack_detail_mask[row_index, detail_index] = 1.0
            if attack.cost_known and bool(dynamic_batch.energy_resolved_mask[row_index].item()):
                payable, remaining = catalog.resolve_payment(instance.energy_counts, attack.energy_cost)
                payable_targets[row_index, detail_index] = float(payable)
                remaining_targets[row_index, detail_index] = torch.tensor(remaining, dtype=torch.float32)
                supervision_mask[row_index, detail_index] = 1.0

    return DynamicCardTrainingBatch(
        card_dynamic_batch=dynamic_batch,
        sample_indices=torch.tensor(sample_indices, dtype=torch.long),
        static_known_mask=dynamic_batch.static_known_mask.clone(),
        detail_exists_mask=dynamic_batch.detail_exists_mask.clone(),
        detail_valid_mask=detail_valid_mask,
        energy_resolved_mask=dynamic_batch.energy_resolved_mask.clone(),
        attack_ids=attack_ids,
        attack_costs=attack_costs,
        attack_detail_mask=attack_detail_mask,
        payable_targets=payable_targets,
        energy_remaining_targets=remaining_targets,
        payment_supervision_mask=supervision_mask,
        hp_targets=hp_targets,
        hp_mask=hp_mask,
        zone_targets=dynamic_batch.zone_ids.clone(),
        role_targets=dynamic_batch.field_role_ids.clone(),
    )
