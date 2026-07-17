from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Hashable, Iterable


EngineReference = tuple[Hashable, ...]
MAPPING_STATUSES = frozenset({"EXACT", "NOT_APPLICABLE", "UNRESOLVED", "CONFLICT", "UNKNOWN_CARD"})


@dataclass(frozen=True)
class DetailCatalogEntry:
    detail_id: int
    detail_index: int
    parent_card_id: int
    detail_local_index: int
    detail_type: str
    engine_references: tuple[EngineReference, ...]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class DetailResolution:
    status: str
    detail_id: int | None = None
    detail_index: int = -1
    candidate_detail_ids: tuple[int, ...] = ()
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.status not in MAPPING_STATUSES:
            raise ValueError(f"invalid detail mapping status {self.status!r}")


def reference_key(reference: EngineReference) -> str:
    return json.dumps(list(reference), ensure_ascii=False, separators=(",", ":"))


class StaticDetailCatalog:
    """Immutable lookup layer over canonical CardRecord and DetailRecord artifacts."""

    def __init__(
        self,
        *,
        card_id_to_index: dict[int, int],
        entries: list[DetailCatalogEntry],
        card_records: dict[int, dict[str, Any]],
        reference_candidates: dict[EngineReference, tuple[int, ...]],
    ) -> None:
        self.card_id_to_index = dict(card_id_to_index)
        self.entries = list(entries)
        self.card_records = {int(card_id): dict(record) for card_id, record in card_records.items()}
        self.detail_id_to_index: dict[int | str, int] = {}
        self.card_id_to_detail_indices: dict[int, list[int]] = {
            card_id: [] for card_id in self.card_id_to_index
        }
        self.detail_index_to_parent_card_id: list[int] = []
        self.detail_index_to_type: list[str] = []
        for expected_index, entry in enumerate(self.entries):
            if entry.detail_index != expected_index:
                raise ValueError("detail catalog indices must be contiguous and ordered")
            if entry.detail_id in self.detail_id_to_index:
                raise ValueError(f"duplicate detail_id {entry.detail_id}")
            self.detail_id_to_index[entry.detail_id] = entry.detail_index
            self.detail_id_to_index[str(entry.detail_id)] = entry.detail_index
            self.card_id_to_detail_indices.setdefault(entry.parent_card_id, []).append(entry.detail_index)
            self.detail_index_to_parent_card_id.append(entry.parent_card_id)
            self.detail_index_to_type.append(entry.detail_type)

        self.reference_candidates = {
            reference: tuple(sorted(set(indices)))
            for reference, indices in reference_candidates.items()
        }
        self.engine_reference_to_detail_index = {
            reference: indices[0]
            for reference, indices in self.reference_candidates.items()
            if len(indices) == 1
        }
        self.mapping_conflicts = {
            reference: indices
            for reference, indices in self.reference_candidates.items()
            if len(indices) > 1
        }

    @classmethod
    def from_artifact_dir(cls, artifact_dir: str | Path) -> "StaticDetailCatalog":
        artifact_dir = Path(artifact_dir)
        cards_raw = json.loads((artifact_dir / "cards.json").read_text(encoding="utf-8"))
        details_raw = json.loads((artifact_dir / "details.json").read_text(encoding="utf-8"))
        mapping_raw = json.loads((artifact_dir / "card_id_to_index.json").read_text(encoding="utf-8"))
        if not isinstance(cards_raw, list) or not isinstance(details_raw, list) or not isinstance(mapping_raw, dict):
            raise ValueError("static card artifacts have invalid top-level shapes")

        card_id_to_index = {int(card_id): int(index) for card_id, index in mapping_raw.items()}
        card_records = {int(record["card_id"]): dict(record) for record in cards_raw}
        if set(card_records) != set(card_id_to_index):
            raise ValueError("card records and card_id_to_index contain different Card IDs")

        local_by_detail: dict[int, tuple[int, int]] = {}
        for card_id, record in card_records.items():
            for local_index, detail_id in enumerate(record.get("detail_ids") or []):
                detail_id = int(detail_id)
                if detail_id in local_by_detail:
                    raise ValueError(f"detail_id {detail_id} is linked from multiple cards")
                local_by_detail[detail_id] = (card_id, local_index)

        provisional: list[tuple[dict[str, Any], int, int]] = []
        references: dict[EngineReference, list[int]] = {}

        def register(reference: EngineReference, detail_index: int) -> None:
            references.setdefault(reference, []).append(detail_index)

        for expected_index, detail in enumerate(details_raw):
            detail_id = int(detail.get("detail_id", detail.get("detail_index", -1)))
            detail_index = int(detail.get("detail_index", -1))
            if detail_id != expected_index or detail_index != expected_index:
                raise ValueError("detail_id/detail_index must match stable global artifact order")
            if detail_id not in local_by_detail:
                raise ValueError(f"detail_id {detail_id} is not linked by a CardRecord")
            parent_card_id, local_index = local_by_detail[detail_id]
            if int(detail.get("card_id", -1)) != parent_card_id:
                raise ValueError(f"detail_id {detail_id} parent Card ID mismatch")
            provisional.append((dict(detail), parent_card_id, local_index))
            register(("card_detail_local_index", parent_card_id, local_index), detail_index)
            if detail.get("detail_type") == "CARD_EFFECT":
                # Playing a Trainer card identifies its printed card effect. The
                # reference remains conflict-aware for any card with more than
                # one CARD_EFFECT; callers must never pick a candidate silently.
                register(("card_effect", parent_card_id), detail_index)
            attack_id = detail.get("attack_id")
            if attack_id is not None:
                attack_id = int(attack_id)
                register(("card_attack_id", parent_card_id, attack_id), detail_index)
                register(("attack_id", attack_id), detail_index)

        normalized_candidates = {
            reference: tuple(sorted(set(indices))) for reference, indices in references.items()
        }
        entries: list[DetailCatalogEntry] = []
        for detail_index, (detail, parent_card_id, local_index) in enumerate(provisional):
            stable_references = tuple(
                sorted(
                    (
                        reference
                        for reference, indices in normalized_candidates.items()
                        if detail_index in indices
                    ),
                    key=reference_key,
                )
            )
            entries.append(
                DetailCatalogEntry(
                    detail_id=int(detail["detail_id"]),
                    detail_index=detail_index,
                    parent_card_id=parent_card_id,
                    detail_local_index=local_index,
                    detail_type=str(detail.get("detail_type") or ""),
                    engine_references=stable_references,
                    metadata={
                        "detail_subtype": detail.get("detail_subtype"),
                        "detail_name": detail.get("detail_name"),
                        "attack_id": detail.get("attack_id"),
                        "source_row": detail.get("source_row"),
                        "source_line": detail.get("source_line"),
                        "source_fields": detail.get("source_fields") or {},
                    },
                )
            )
        return cls(
            card_id_to_index=card_id_to_index,
            entries=entries,
            card_records=card_records,
            reference_candidates=normalized_candidates,
        )

    def resolve(self, reference: EngineReference, *, card_id: int | None = None) -> DetailResolution:
        if card_id is not None and int(card_id) not in self.card_id_to_index:
            return DetailResolution(
                status="UNKNOWN_CARD",
                reason=f"card_id {card_id} is absent from the static catalog",
            )
        candidates = self.reference_candidates.get(reference, ())
        if not candidates:
            return DetailResolution(status="UNRESOLVED", reason="stable reference was not found")
        if len(candidates) > 1:
            return DetailResolution(
                status="CONFLICT",
                candidate_detail_ids=tuple(self.entries[index].detail_id for index in candidates),
                reason="stable reference maps to multiple details",
            )
        detail_index = candidates[0]
        entry = self.entries[detail_index]
        if card_id is not None and entry.parent_card_id != int(card_id):
            return DetailResolution(
                status="UNRESOLVED",
                candidate_detail_ids=(entry.detail_id,),
                reason="reference resolves to a different parent Card ID",
            )
        return DetailResolution(
            status="EXACT",
            detail_id=entry.detail_id,
            detail_index=entry.detail_index,
            candidate_detail_ids=(entry.detail_id,),
        )

    def details_for_card(self, card_id: int) -> list[DetailCatalogEntry]:
        return [self.entries[index] for index in self.card_id_to_detail_indices.get(int(card_id), [])]

    def card_record(self, card_id: int | None) -> dict[str, Any] | None:
        return self.card_records.get(int(card_id)) if card_id is not None else None

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "card_count": len(self.card_id_to_index),
            "detail_count": len(self.entries),
            "entries": [asdict(entry) for entry in self.entries],
            "mapping_conflicts": [
                {
                    "engine_reference": list(reference),
                    "candidate_detail_indices": list(indices),
                    "candidate_detail_ids": [self.entries[index].detail_id for index in indices],
                }
                for reference, indices in sorted(self.mapping_conflicts.items(), key=lambda item: reference_key(item[0]))
            ],
        }

    def write(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_payload(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return path


def candidate_detail_ids(catalog: StaticDetailCatalog, indices: Iterable[int]) -> tuple[int, ...]:
    return tuple(catalog.entries[index].detail_id for index in sorted(set(indices)))
