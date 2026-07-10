# Training ideas and concerns

## Privileged oracle teacher

Idea: during offline training, use a modified or wrapped engine that can see privileged global information, then train the normal competition agent from the points where the normal policy and privileged policy disagree.

This is a form of privileged teacher / oracle distillation:

- Teacher sees hidden state such as both decks, prize cards, opponent hand, and hidden zones.
- Student only receives the normal competition observation.
- Record decision points with normal observation, legal options, normal action, oracle action, and estimated oracle advantage.
- Train with behavior cloning or auxiliary imitation loss, then fine-tune with PPO from normal observations.

Important caveat: the teacher should not directly exploit future random outcomes such as future coin flips or shuffled order produced after a random event. If future randomness is used, it should be averaged over multiple rollouts and treated as value estimation, not as a direct action label.

Recommended data format:

```json
{
  "observation": "normal competition observation",
  "legal_options": [],
  "normal_action": [0],
  "oracle_action": [2],
  "normal_value": 0.12,
  "oracle_value": 0.48,
  "advantage": 0.36,
  "is_disagreement": true
}
```

Training should not only use disagreement points. Record all decision points, then up-weight disagreement points or high oracle-advantage points. Otherwise the model may miss common obvious actions.

## Current PPO opponent issue

Current implementation in `kaggle_training/train_agent.py` uses one shared policy for both players during self-play:

- player 0, the selected deck, uses the current policy;
- player 1, the opponent deck, also uses the same current policy;
- only player 0 transitions are recorded for PPO;
- player 1 decisions are not recorded, but they still change as the shared policy changes.

This means the opponent is non-stationary. The training target moves while the model learns. With 8 decks trained together, this is not strictly wrong, but it is noisy and can be unstable:

- improvements for one deck can change opponent behavior for every other deck;
- bad updates can make both home and away play worse at the same time;
- matchup results become hard to interpret because both sides changed;
- a shared policy may underfit deck-specific strategy.

## Better opponent setups

Preferred next designs:

1. Frozen opponent snapshots
   - Keep a copy of the policy at the start of an epoch.
   - Train player 0 against the frozen snapshot.
   - Refresh the snapshot only after a full round-robin cycle or after N updates.
   - This makes the PPO environment more stationary.

2. League / population self-play
   - Maintain several historical policy snapshots.
   - Sample opponents from recent and older snapshots.
   - Prevents overfitting to the current mirror policy.

3. Deck-conditioned policy
   - One shared network, but include deck ID or deck embedding in features.
   - The policy can learn different behavior for Dragapult, Crustle, Ogerpon, etc.

4. Per-deck heads
   - Shared feature trunk with one policy head per deck.
   - Train each deck head against frozen or league opponents.

5. Oracle distillation before PPO
   - Pretrain from privileged teacher decisions.
   - Then use PPO with frozen opponent snapshots.

## Practical next step

For the current codebase, the most practical improvement is:

1. Add frozen opponent policy snapshots.
2. Add `final_turn` and timeout diagnostics to `training_summary.json`.
3. Add deck ID as an input feature.
4. Later add oracle-generated behavior cloning data.

## Failed run: shared PPO round-robin

Status: stopped manually and considered failed.

Setup:

- Kaggle cloud run.
- Competition `cg` engine was used for self-play.
- 8 patched baseline decks.
- Round-robin directed matchups.
- Target was 500 games per matchup.
- One shared PPO policy controlled both home and away players.
- Only home/player-0 transitions were recorded for PPO.

Observed behavior:

- After roughly 50+ PPO update batches, loss oscillated around `0.1` to `0.5`.
- No clear evidence of continuing improvement.
- Training was paused before completion.

Why this is considered a failed direction:

- The opponent was non-stationary because away/player-1 used the same policy being updated.
- A single shared policy was expected to cover 8 different decks without a deck identity feature.
- Reward was sparse terminal win/loss, so the learning signal was weak and noisy.
- Loss oscillation alone is not a reliable performance metric, but combined with the setup issues it suggests that simply scaling this PPO loop is unlikely to be efficient.

Conclusion:

- Do not keep spending compute on the current shared-policy PPO design as-is.
- Before another long run, change the training structure.

Preferred replacement direction:

1. Split behavior into two layers:
   - behavior plausibility: whether an action is strategically reasonable;
   - game-optimal selection: which reasonable action is best in this matchup/state.
2. Use frozen opponent snapshots or a league of historical opponents.
3. Add deck identity or deck-specific policy heads.
4. Add oracle/privileged teacher data for behavior cloning and action-value labels.

## First/second turn selection bias

Observation from online match data: the player going first appears to have roughly a 10 percentage point higher win rate than the player going second.

Competition-specific exploit/risk idea: if our submission can bias the game setup toward choosing to go second, it may almost guarantee that our own agent is always the second player. This is counterintuitive because first player has the higher raw win rate, but it may be strategically useful if:

- the deck or policy is explicitly trained for going second;
- opponents are optimized for the general first-player advantage rather than the forced second-player distribution;
- the submission/game setup has an asymmetry that makes turn-order preference controllable;
- we can design a second-player strategy that recovers more than the observed 10 point global disadvantage.

Next checks before using this in training:

1. Verify the online statistic by deck archetype, not only globally.
2. Measure whether the turn-order bias is actually controllable by our agent/submission path.
3. Train/evaluate separate first-player and second-player policies or policy heads.
4. Add turn-order conditioning to offline evaluation and report win rate separately for first/second.
5. Avoid assuming second is better; treat this as a distribution-control opportunity that needs empirical validation.

