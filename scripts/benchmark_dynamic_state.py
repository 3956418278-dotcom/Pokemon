from __future__ import annotations

import sys
import warnings
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.benchmark_dynamic_card_fusion import main as benchmark_dynamic_card_fusion
from models.static_card_adapter import StaticCardAdapter
from training.train_dynamic_card_fusion import require_static_adapter_ready


def main() -> None:
    static_adapter = StaticCardAdapter()
    require_static_adapter_ready(static_adapter)
    warnings.warn(
        "scripts/benchmark_dynamic_state.py is deprecated; forwarding to "
        "scripts/benchmark_dynamic_card_fusion.py with the same arguments",
        FutureWarning,
        stacklevel=2,
    )
    benchmark_dynamic_card_fusion()


if __name__ == "__main__":
    main()
