from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .schemas import SelectionMode


SEMANTIC_TO_MODE = {
    "SINGLE_INDEX": SelectionMode.SINGLE,
    "COUNT_VALUE": SelectionMode.SINGLE,
    "ORDERED_INDEX_SEQUENCE": SelectionMode.ORDERED_SEQUENCE,
    "UNORDERED_UNIQUE_SUBSET": SelectionMode.UNORDERED_UNIQUE_SUBSET,
    "INDEX_MULTISET": SelectionMode.UNKNOWN,
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class ActionSemanticsContract:
    rules: dict[tuple[int, int, int], frozenset[str]]
    metadata: dict[str, Any]

    @classmethod
    def load(cls, path: str | Path) -> "ActionSemanticsContract":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        rules: dict[tuple[int, int, int], set[str]] = defaultdict(set)
        for row in payload.get("rules", []):
            key = (int(row["select_type"]), int(row["select_context"]), int(row["option_type"]))
            rules[key].add(str(row["source_action_semantics"]))
        return cls(
            {key: frozenset(values) for key, values in rules.items()},
            {key: value for key, value in payload.items() if key != "rules"},
        )

    @classmethod
    def build_from_audit(
        cls,
        action_semantics_csv: str | Path,
        audit_json: str | Path,
        legal_options_source: str | Path,
    ) -> tuple["ActionSemanticsContract", dict[str, Any]]:
        action_path = Path(action_semantics_csv)
        audit_path = Path(audit_json)
        legal_path = Path(legal_options_source)
        rows = list(csv.DictReader(action_path.open(encoding="utf-8", newline="")))
        rules: dict[tuple[int, int, int], set[str]] = defaultdict(set)
        serialized_rules = []
        for row in rows:
            key = (int(row["select_type"]), int(row["select_context"]), int(row["option_type"]))
            semantic = row["action_semantics"]
            rules[key].add(semantic)
            serialized_rules.append(
                {
                    "select_type": key[0],
                    "select_context": key[1],
                    "option_type": key[2],
                    "source_action_semantics": semantic,
                    "selection_mode": SEMANTIC_TO_MODE.get(semantic, SelectionMode.UNKNOWN).value,
                    "decision_count": int(row.get("decision_count") or 0),
                    "allows_duplicate": semantic == "INDEX_MULTISET",
                    "order_matters": semantic == "ORDERED_INDEX_SEQUENCE",
                    "canonical_sort_key": "option_semantic_fields_then_original_index",
                    "equivalence_rule": "data.legal_options.equivalence_class_ids",
                }
            )
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        payload = {
            "schema_version": "decision_agent_v1_action_semantics_v1",
            "sources": [
                {"path": str(action_path), "sha256": sha256_file(action_path)},
                {"path": str(audit_path), "sha256": sha256_file(audit_path)},
                {"path": str(legal_path), "sha256": sha256_file(legal_path)},
            ],
            "source_audit_passed": bool(audit.get("audit_passed")),
            "unknown_policy": "Keep sample for Value loss and exclude it from Policy loss.",
            "rules": serialized_rules,
        }
        contract = cls(
            {key: frozenset(values) for key, values in rules.items()},
            {key: value for key, value in payload.items() if key != "rules"},
        )
        return contract, payload

    def mode_for(
        self,
        select_type: int,
        select_context: int,
        option_types: set[int],
        max_count: int,
        observed_semantics: str | None = None,
    ) -> SelectionMode:
        semantics: set[str] = set()
        for option_type in option_types:
            values = self.rules.get((select_type, select_context, option_type))
            if not values:
                return SelectionMode.UNKNOWN
            semantics.update(values)
        if observed_semantics is not None:
            if observed_semantics not in semantics:
                return SelectionMode.UNKNOWN
            selected = observed_semantics
        elif max_count <= 1 and "SINGLE_INDEX" in semantics:
            selected = "SINGLE_INDEX"
        else:
            non_single = semantics - {"SINGLE_INDEX"}
            if len(non_single) != 1:
                return SelectionMode.UNKNOWN
            selected = next(iter(non_single))
        return SEMANTIC_TO_MODE.get(selected, SelectionMode.UNKNOWN)


def write_contract(
    output_path: str | Path,
    action_semantics_csv: str | Path,
    audit_json: str | Path,
    legal_options_source: str | Path,
) -> None:
    _, payload = ActionSemanticsContract.build_from_audit(
        action_semantics_csv, audit_json, legal_options_source
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
