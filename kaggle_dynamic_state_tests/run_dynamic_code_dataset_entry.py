from __future__ import annotations

import sys
from pathlib import Path

CODE_ROOT = Path("/kaggle/input/ptcg-dynamic-code-dataset")
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from scripts.train_dynamic_replay_features import main


if __name__ == "__main__":
    sys.argv = [
        sys.argv[0],
        "--episodes-index-dir",
        "/kaggle/input/pokemon-tcg-ai-battle-episodes-index",
        "--use-daily-manifest",
        "--daily-dataset-mount-root",
        "/kaggle/input",
        "--reserve-recent-days",
        "3",
        "--import-split",
        "train",
        "--max-days",
        "1",
        "--max-samples",
        "4096",
        "--epochs",
        "1",
        "--batch-size",
        "16",
        "--output-dir",
        "/kaggle/working/dynamic_replay_features",
    ]
    main()
