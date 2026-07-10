# Kaggle cloud workflow

Codex maintains code locally. Training, submission packaging, and competition-data extraction run as separate Kaggle kernels.

## Kernel folders

- `kaggle_extract/`: extracts recent public replay data and writes popular test decks.
- `kaggle_training/`: trains the agent and writes model artifacts.
- `kaggle_submission/`: packages a trained model into `submission.tar.gz`.

`kaggle_kernel/` is legacy from the earlier combined workflow.

## 1. Extract popular test decks

This kernel reads recent public replay data through the Kaggle competition API and groups decks by exact Pokemon + Energy counts. Trainer differences are ignored for grouping.

```bash
kaggle kernels push -p kaggle_extract
kaggle kernels status f7e6n5g4/ptcg-popular-deck-extract
kaggle kernels output f7e6n5g4/ptcg-popular-deck-extract -p outputs/popular_decks -o
```

Main outputs:

```text
popular_deck_outputs/popular_test_decks.json
popular_deck_outputs/popular_test_decks.md
popular_deck_outputs/popular_test_deck.csv
popular_deck_outputs/popular_deck_summary.csv
```

`popular_test_decks.json` is intentionally separate from the 8 curated baseline decks. Each popular deck group includes every distinct full 60-card variant observed inside that Pokemon + Energy signature group.

Useful environment variables:

```text
PTCG_RECENT_SUBMISSIONS_TO_USE=8
PTCG_MAX_REPLAYS_TO_DOWNLOAD=200
PTCG_MIN_POPULAR_DECK_GAMES=2
PTCG_MAX_POPULAR_DECKS=24
PTCG_SUBMISSION_IDS="54449821 54449681"
```

## 2. Train

Training uses the competition engine on Kaggle.

```bash
kaggle kernels push -p kaggle_training
kaggle kernels status f7e6n5g4/ptcg-agent-training
kaggle kernels output f7e6n5g4/ptcg-agent-training -p outputs -o
```

Expected training outputs:

```text
model.json
ppo_weights.json
training_summary.json
policy_state.pt
```

Default training still uses patched baseline decks from `baseline_decks.json`. The training kernel can also discover `popular_test_decks.json` if that dataset is attached later, but it does not replace baseline training by default.

Useful environment variables:

```text
PTCG_TRAINING_SCHEDULE=round_robin
PTCG_EPISODES_PER_MATCHUP=500
PTCG_PPO_BATCH_EPISODES=64
PTCG_MAX_STEPS=500
PTCG_SEED=20260710
```

`PTCG_BUILD_SUBMISSION=1` is no longer used in the split-kernel workflow. Use the submission kernel instead.

## 3. Build submission

The submission kernel reads training artifacts from the attached training kernel output or local working directory, then writes `submission.tar.gz`.

```bash
kaggle kernels push -p kaggle_submission
kaggle kernels status f7e6n5g4/ptcg-agent-submission
kaggle kernels output f7e6n5g4/ptcg-agent-submission -p outputs/submission -o
```

Expected submission outputs:

```text
main.py
deck.csv
model.json
weights.json
submission.tar.gz
```

Submit:

```bash
kaggle competitions submit -c pokemon-tcg-ai-battle -f outputs/submission/submission.tar.gz -m "message"
```

## Local helper scripts

Regenerate curated baseline decks:

```bash
python scripts/make_baseline_decks.py
```

This writes root-level `baseline_decks.json` / `baseline_decks.md` and syncs `baseline_decks.json` plus `baseline_deck.csv` into both `kaggle_training/` and `kaggle_submission/`.

## Static CardEncoder Pretraining

The static card pretraining module lives in root-level Python packages:

```text
data/
models/
training/
configs/card_pretrain.yaml
kaggle_card_pretrain/
```

Run it on Kaggle cloud through the packaged kernel:

```bash
kaggle datasets create -p kaggle_cg_runtime_dataset --dir-mode zip
# or, after the dataset already exists:
# kaggle datasets version -p kaggle_cg_runtime_dataset --dir-mode zip -m "update cg runtime"

kaggle kernels push -p kaggle_card_pretrain
kaggle kernels status f7e6n5g4/ptcg-card-pretrain
kaggle kernels output f7e6n5g4/ptcg-card-pretrain -p outputs/card_pretrain -o
```

`kaggle_cg_runtime_dataset/` publishes `cg/` as a Kaggle Dataset. `kaggle_card_pretrain/` mounts that dataset through `dataset_sources`, so preprocessing can call `cg.api.all_card_data()` and `cg.api.all_attack()` when available.

The card pretraining kernel is self-contained because Kaggle script kernels upload only `code_file`; `run_card_pretrain.py` embeds and extracts the training modules before execution.

For local smoke testing in an environment with PyTorch, the equivalent commands are:

```bash
python -m training.pretrain_card_encoder --config configs/card_pretrain.yaml --rebuild-cache
python -m training.export_card_embeddings \
  --checkpoint checkpoints/card_encoder_best.pt \
  --output artifacts/card_embeddings.pt
python -m training.evaluate_card_embeddings \
  --embeddings artifacts/card_embeddings.pt \
  --output-dir artifacts/card_embedding_analysis
```

Main outputs:

```text
artifacts/card_data/card_records.json
artifacts/card_data/card_id_to_index.json
artifacts/card_data/card_feature_schema.json
artifacts/card_data/card_preprocess_summary.json
checkpoints/card_encoder_best.pt
artifacts/card_embeddings.pt
artifacts/card_embeddings.npy
artifacts/card_id_to_index.json
artifacts/card_embedding_metadata.json
artifacts/card_embedding_analysis/report.md
```

Downstream usage:

```python
import json
import torch

card_embedding_table = torch.load("artifacts/card_embeddings.pt", map_location="cpu")
card_id_to_index = json.loads(open("artifacts/card_id_to_index.json", encoding="utf-8").read())
card_index = card_id_to_index[str(card_id)]
embedding = card_embedding_table[card_index]
```

The current module trains only static card embeddings. Board value learning, behavior cloning, self-play RL, and state-difference prediction remain separate future modules. Dynamic board models should consume the static embedding through `models.card_instance_encoder.CardInstanceEncoder`.
