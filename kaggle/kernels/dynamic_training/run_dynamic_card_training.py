from __future__ import annotations

import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.static_card_adapter import (  # noqa: E402
    StaticArtifactContractNotConfigured,
    StaticCardAdapter,
)


PAUSE_REASON = (
    "dynamic training is paused until colleague static artifacts "
    "are integrated into StaticCardAdapter"
)
DEFAULT_OUTPUT_ROOT = (
    Path("/kaggle/working/outputs/dynamic_card_training")
    if Path("/kaggle/working").exists()
    else ROOT / "outputs/dynamic_card_training"
)
OUTPUT_ROOT = Path(os.environ.get("PTCG_DYNAMIC_OUTPUT_ROOT", DEFAULT_OUTPUT_ROOT))


def main() -> None:
    static_adapter = StaticCardAdapter()
    if not static_adapter.ready:
        payload = {
            "success": False,
            "completed_stage": "static_artifact_contract",
            "error_type": "StaticArtifactContractNotConfigured",
            "error": PAUSE_REASON,
        }
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        (OUTPUT_ROOT / "run_summary.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(PAUSE_REASON, file=sys.stderr)
        raise StaticArtifactContractNotConfigured(PAUSE_REASON)


if __name__ == "__main__":
    main()
