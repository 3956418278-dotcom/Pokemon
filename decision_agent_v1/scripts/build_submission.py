from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = ROOT / "outputs/decision_agent_v1/checkpoints/best_joint.pt"
DEFAULT_VOCAB = ROOT / "static_card/artifacts/card_data/card_id_to_index.json"
DEFAULT_CONTRACT = ROOT / "decision_agent_v1/contracts/action_semantics.json"
DEFAULT_DECKS = ROOT / "decks/baseline_decks.json"
DEFAULT_RUNTIME = ROOT / "kaggle/datasets/cg_runtime/cg"
DEFAULT_OUTPUT = ROOT / "outputs/decision_agent_v1/submission_v1"


SOURCE_FILES = (
    "decision_agent_v1/__init__.py",
    "decision_agent_v1/adapters/card_vocab_adapter.py",
    "decision_agent_v1/adapters/observation_adapter.py",
    "decision_agent_v1/adapters/simulator_adapter.py",
    "decision_agent_v1/baseline/__init__.py",
    "decision_agent_v1/baseline/deterministic_legal_agent.py",
    "decision_agent_v1/contracts/__init__.py",
    "decision_agent_v1/contracts/action_contract.py",
    "decision_agent_v1/contracts/schemas.py",
    "decision_agent_v1/data/collate.py",
    "decision_agent_v1/inference/__init__.py",
    "decision_agent_v1/inference/policy_value_agent.py",
    "decision_agent_v1/models/__init__.py",
    "decision_agent_v1/models/board_encoder.py",
    "decision_agent_v1/models/card_instance_encoder.py",
    "decision_agent_v1/models/multiselect_decoder.py",
    "decision_agent_v1/models/option_encoder.py",
    "decision_agent_v1/models/policy_head.py",
    "decision_agent_v1/models/policy_value_model.py",
    "decision_agent_v1/models/value_head.py",
    "data/__init__.py",
    "data/decision_schema.py",
    "data/game_memory.py",
    "data/observation_parser.py",
    "data/state_schema.py",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_deck(
    decks_path: Path, deck_index: int, *, allow_patched_deck: bool
) -> tuple[list[int], dict[str, Any]]:
    payload = json.loads(decks_path.read_text(encoding="utf-8"))
    selected = payload["decks"][deck_index]
    deck = [int(card_id) for card_id in selected["patched_deck_ids"]]
    if len(deck) != 60:
        raise RuntimeError(f"deck {deck_index} contains {len(deck)} cards, not 60")
    if int(selected.get("replaced_total", 0)) != 0 and not allow_patched_deck:
        raise RuntimeError(
            f"deck {selected['name']} is patched; choose a source deck with no replacements"
        )
    return deck, selected


def _copy_sources(package: Path) -> None:
    for relative in SOURCE_FILES:
        source = ROOT / relative
        target = package / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    # Avoid package initializers importing training-only modules.
    (package / "decision_agent_v1/adapters/__init__.py").write_text("", encoding="utf-8")
    (package / "decision_agent_v1/data/__init__.py").write_text("", encoding="utf-8")


def _copy_runtime(package: Path, runtime: Path) -> None:
    target = package / "cg"
    target.mkdir(parents=True, exist_ok=True)
    for name in ("__init__.py", "api.py", "game.py", "sim.py", "utils.py", "libcg.so"):
        shutil.copy2(runtime / name, target / name)


def _write_checkpoint(source: Path, target: Path) -> dict[str, Any]:
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    keys = (
        "model_state_dict",
        "model_config",
        "data_schema_hash",
        "action_contract_hash",
        "card_vocabulary_hash",
        "adapter_hash",
        "epoch",
        "global_step",
        "metrics",
    )
    compact = {key: checkpoint[key] for key in keys if key in checkpoint}
    torch.save(compact, target)
    return checkpoint


def build(args: argparse.Namespace) -> Path:
    output = args.output.resolve()
    package = output / "package"
    if package.exists():
        shutil.rmtree(package)
    package.mkdir(parents=True)

    checkpoint = _write_checkpoint(args.checkpoint.resolve(), package / "model.pt")
    for path, expected_key in (
        (args.vocab.resolve(), "card_vocabulary_hash"),
        (args.contract.resolve(), "action_contract_hash"),
    ):
        actual = _sha256(path)
        expected = checkpoint.get(expected_key)
        if actual != expected:
            raise RuntimeError(f"{path.name} hash mismatch: expected {expected}, got {actual}")

    deck, selected_deck = _load_deck(
        args.decks.resolve(), args.deck_index, allow_patched_deck=args.allow_patched_deck
    )
    (package / "deck.csv").write_text("".join(f"{card_id}\n" for card_id in deck), encoding="utf-8")
    shutil.copy2(args.vocab, package / "card_vocab.json")
    shutil.copy2(args.contract, package / "action_semantics.json")
    shutil.copy2(ROOT / "decision_agent_v1/submission/main.py", package / "main.py")
    _copy_sources(package)
    _copy_runtime(package, args.runtime.resolve())

    metadata = {
        "submission_schema": "pokemon_tcg_ai_battle_v1",
        "checkpoint_source": str(args.checkpoint.resolve().relative_to(ROOT)),
        "checkpoint_source_sha256": _sha256(args.checkpoint.resolve()),
        "checkpoint_packaged_sha256": _sha256(package / "model.pt"),
        "epoch": checkpoint.get("epoch"),
        "global_step": checkpoint.get("global_step"),
        "model_config": checkpoint["model_config"],
        "data_schema_hash": checkpoint["data_schema_hash"],
        "action_contract_hash": checkpoint["action_contract_hash"],
        "card_vocabulary_hash": checkpoint["card_vocabulary_hash"],
        "adapter_hash": checkpoint.get("adapter_hash"),
        "deck": {
            "source": str(args.decks.resolve().relative_to(ROOT)),
            "index": args.deck_index,
            "name": selected_deck["name"],
            "replaced_total": selected_deck["replaced_total"],
        },
    }
    (package / "model_config.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    archive = output / "submission.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        for path in sorted(package.rglob("*")):
            if path.is_file():
                handle.add(path, arcname=path.relative_to(package))
    print(json.dumps({
        "archive": str(archive),
        "archive_bytes": archive.stat().st_size,
        "archive_sha256": _sha256(archive),
        "deck_index": args.deck_index,
        "deck_name": selected_deck["name"],
    }, ensure_ascii=False, indent=2))
    return archive


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the standalone V1 Kaggle submission")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--vocab", type=Path, default=DEFAULT_VOCAB)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--decks", type=Path, default=DEFAULT_DECKS)
    parser.add_argument("--deck-index", type=int, default=6)
    parser.add_argument("--allow-patched-deck", action="store_true")
    parser.add_argument("--runtime", type=Path, default=DEFAULT_RUNTIME)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    build(parser.parse_args())


if __name__ == "__main__":
    main()
