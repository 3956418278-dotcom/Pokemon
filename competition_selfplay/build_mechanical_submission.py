from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "outputs/competition_selfplay/submission_mechanical_v2"
EXPECTED_DECK_HASH = "1134964e85e85978ae1c6ea5a8234fee166ce787f38285bbbf2462d406a4bada"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build(output: Path) -> Path:
    output = output.resolve()
    package = output / "package"
    if package.exists():
        shutil.rmtree(package)
    package.mkdir(parents=True)
    shutil.copy2(ROOT / "competition_selfplay/submission/main.py", package / "main.py")
    shutil.copy2(ROOT / "competition_selfplay/mechanical_agent.py", package / "mechanical_agent.py")
    runtime_source = ROOT / "kaggle/datasets/cg_runtime/cg"
    runtime_target = package / "cg"
    runtime_target.mkdir()
    for name in ("__init__.py", "api.py", "game.py", "sim.py", "utils.py", "libcg.so"):
        shutil.copy2(runtime_source / name, runtime_target / name)

    decks = json.loads((ROOT / "decks/baseline_decks.json").read_text(encoding="utf-8"))["decks"]
    selected = decks[6]
    deck = [int(value) for value in selected["patched_deck_ids"]]
    deck_path = package / "deck.csv"
    deck_path.write_text("".join(f"{card_id}\n" for card_id in deck), encoding="utf-8")
    if len(deck) != 60 or _sha256(deck_path) != EXPECTED_DECK_HASH:
        raise RuntimeError("scored Raging Bolt Ogerpon deck identity changed")

    archive = output / "submission.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        for path in sorted(package.iterdir()):
            handle.add(path, arcname=path.name)
    print(json.dumps({
        "archive": str(archive),
        "archive_bytes": archive.stat().st_size,
        "archive_sha256": _sha256(archive),
        "files": sorted(path.name for path in package.iterdir()),
        "deck": selected["name"],
    }, ensure_ascii=False, indent=2))
    return archive


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    build(args.output)


if __name__ == "__main__":
    main()
