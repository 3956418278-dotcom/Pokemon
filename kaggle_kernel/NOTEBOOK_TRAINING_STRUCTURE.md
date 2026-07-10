# Kaggle training notebook structure

## Previous structure

The first kernel version was a minimal submission-closure workflow:

1. Discover `/kaggle/input`.
2. Copy or synthesize `deck.csv`.
3. Run synthetic random search over rule-score weights.
4. Write `main.py`, `deck.csv`, `weights.json`, and `submission.tar.gz`.

That version did not call the battle engine during training.

## Current structure

The kernel is now structured as a cloud training script:

1. Discover competition input under either:
   - `/kaggle/input/pokemon-tcg-ai-battle`
   - `/kaggle/input/competitions/pokemon-tcg-ai-battle`
2. Copy or extract the sample submission `cg/` runtime.
3. Load patched baseline decks from `baseline_decks.json`.
4. Train a shared PPO policy with engine self-play:
   - for each baseline deck, each episode samples another baseline deck as the opponent;
   - both players use the same current policy;
   - terminal reward is assigned from the selected deck player's result.
5. Training export:
   - `model.json`
   - `ppo_weights.json`
   - `training_summary.json`
6. Submission export is handled separately by `submit_agent.py`:
   - `main.py`
   - `deck.csv`
   - `model.json`
   - `weights.json`
   - `submission.tar.gz`

## Practical defaults

The default training schedule is round-robin matchup training. With 8 decks, this means 56 directed matchups: A-as-player-0 vs B, and B-as-player-0 vs A are both trained. Each directed matchup runs 500 episodes by default.

```bash
PTCG_TRAINING_SCHEDULE=round_robin
PTCG_EPISODES_PER_MATCHUP=500
PTCG_PPO_BATCH_EPISODES=64
PTCG_MAX_STEPS=500
```

If `torch` is unavailable or PPO fails, the training script still writes fallback weights.
Set `PTCG_BUILD_SUBMISSION=1` only when you want `train_agent.py` to also call the submission builder.
