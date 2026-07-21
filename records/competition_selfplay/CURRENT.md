# Current state — competition self-play

Updated: 2026-07-21 (Asia/Shanghai)

## Transactional semantic PPO implementation

- The obsolete `RewardVector(outcome, prize_progress, setup_tempo)`, `VectorReward`, actor
  scalarization weights, setup-potential weights, and three-value critic contract have been removed.
  Config schema is now `transactional_semantic_selfplay_v2`; the critic is scalar.
- Rollout choices are assembled into causal transactions. Nested resolutions share one summed
  non-forced log probability and one transaction advantage; unique forced choices are recorded but
  excluded from the actor ratio.
- Phase A is locked to 20,000 completed games of pure terminal `+1/-1/0` transaction-level PPO.
  Concept, semantic-potential, residual, and full-value losses remain active while shaping alpha is
  exactly zero.
- The fixed nine-position semantic concept ensemble, deterministic applicability masks, automatic
  completed-trajectory labels, serial-aware survival, delayed causal links, calibrated confidence,
  bounded semantic potential, and reconstructable explanation interface are connected to training.
- Phase B can begin only after the fixed holdout simultaneously passes the Brier-improvement, ECE,
  seat-swap antisymmetry, and ranking gates. Alpha then ramps to 0.15 over 50,000 games.
- During a rollout batch only the learner policy is updated. Learner seats alternate P0/P1; actor
  loss excludes opponent transactions. The frozen opponent and the complete target semantic path
  remain unchanged until, respectively, a league promotion or the explicit post-update EMA step.
  Stored `target_phi_before/after` values make the collected batch reward invariant to online
  encoder updates.
- Checkpoints include learner/optimizer, scalar full-critic components, online semantic heads,
  complete target semantic module, residual head, frozen opponent and revision, phase/game/alpha,
  calibration metrics, league state, and config snapshot. Required metrics are emitted by the PPO
  update/runner.
- The implementation is code-ready for formal Phase A training. This means the full collection →
  labeling → transaction reward/GAE → PPO → post-batch EMA → checkpoint path exists and is tested;
  it does not claim that the 20,000-game Phase A run or the Phase B calibration gate has already
  completed.
- Verification on 2026-07-21: the maintained branch suite passes 61/61 tests. A no-output smoke
  game against the official local `cg` runtime completed with 123 transactions, 164 non-forced
  selects and 22 forced selects, preserved exact terminal reason `2`, produced labels and a compact
  two-perspective holdout, and completed
  one PPO update with all four critic losses. Opponent and target parameters stayed bitwise fixed
  during that update; the target changed only after the explicit EMA boundary.

## Confirmed intake package

- Intent: replace replay imitation as the main route with fixed-deck iterative self-play.
- Target: Kaggle submission `54815037`, `submission.tar.gz`, submitted 2026-07-18 19:43:59 UTC,
  status `COMPLETE`, public score `357.8`, description `V1 best_joint | Raging Bolt Ogerpon |
  isolated dependency check passed`.
- Deck: `decks/baseline_decks.json` index 6, `Raging Bolt Ogerpon`, 60/60 present, zero patched
  replacements, all 19 Pokemon are Basic Pokemon.
- Training contract: learner changes while opponent is frozen; promote and copy only above the
  threshold; freeze after repeated failed evaluations or the promotion cap.
- Reward contract: scalar terminal outcome plus a calibration-gated frozen semantic potential
  difference. The earlier three-dimensional/setup prototype is historical and no longer exists in
  the executable path.
- Feature boundary: Card ID plus deck-copy/live-instance identity; no opponent deck/hand/discard
  counts; public opponent field counts remain allowed.

## Safety and scope

The separately authorized mechanical-v2 upload has now happened exactly once. No additional Kaggle
submission, long training, dependency installation, deletion, or overwrite of replay work is
authorized by the current stage. Generated data belongs under `outputs/competition_selfplay/`;
durable collaboration state belongs in this record.

## Mechanical fallback status

- Mechanical v1 was uploaded to Kaggle exactly once on 2026-07-19 as submission `54821848`,
  description `Mechanical if-else v1 | validated isolated package`. Its resulting replay behavior
  was rejected as the design baseline after human inspection; the v1 output is preserved rather
  than overwritten.
- Four returned replays are stored under `replays/` and were used for counterfactual action checks:
  `86823089`, `86825257`, `86825818`, and `86826378`.
- Mechanical v2 is implemented in `competition_selfplay/mechanical_agent.py`; its explicit rule
  contract is in `competition_selfplay/MECHANICAL_POLICY.md`.
- The policy assigns every card one state-dependent importance order, then uses the exact inverse
  for discard/deck-return choices. The state branch is intentionally limited to key-card state:
  Latias ex (`1`), Teal Mask Ogerpon ex (`2`), and Wellspring Mask Ogerpon ex (`3`) positions,
  health, attached Energy, counts, and immediate attack/switch conditions. It does not use a broad
  learned or hand-authored board-potential score.
- Teal Mask roles are stable by card serial: the first healthy `2` is filled to three Grass before
  the second, and both Teal Dance and manual attachment use the same order.
- A strict pre-End gate moves `1`/`2`/`3` out of Active whenever the documented exception does not
  apply and a legal move exists.
- The `audit-20260719-001` run exposed that the first attempted "all-card order" still grouped
  Trainers/search engines into broad fixed tiers. It is retained only as a rejected comparison run.
- The rewritten ordering chooses the first unmet formation requirement, then evaluates every one of
  the fixed deck's 27 concrete card IDs separately. Search uses the order forward and discard/deck
  return uses it in reverse; Supporter/draw category sets now participate only in deck-out legality
  gates, never ranking.
- Runs `audit-20260719-002` through `audit-20260719-005` are retained as rejected intermediate
  evidence. Self-review found and fixed: search below the first deck boundary, reversed
  Ciphermaniac top-deck selection, stale-state multi-card acquisition, removal of both Latias
  copies, spreading Teal Dance Grass before the primary reached three, excess core copies occupying
  formation slots, and Burst Roar bypassing its deck-out prohibition.
- `audit-20260719-006` is rejected. Its checker proved legality and agreement with the then-current
  implementation, but failed to prove agreement with the user's policy. The implementation gave
  attacks top priority, did not preserve `first 2 > Grass > 1 > second 2` as a stable base order,
  used a coarse mixed draw/search ID set, and did not calculate the first `2`'s damage threshold
  separately against every opposing Pokémon.
- Future `run_mechanical_selfplay` invocations export one replay per episode by default into a new
  timestamped output run instead of printing only a summary.
- Isolated `decks` branch suite: 37 tests passed.
- The standalone v2 archive is
  `outputs/competition_selfplay/submission_mechanical_v2/submission.tar.gz`, 510082 bytes, SHA-256
  `32e4277798e6a3eb03bfe530e2d2d669444ef079fd1ada836122b1285e01c953`.
- That exact archive was uploaded to Kaggle once on 2026-07-19 as submission `54825132`, description
  `Mechanical v2 | per-card state order | audit-006`. It contains the rejected implementation and
  is permanently barred from training material regardless of score or completion status.
- `audit-20260719-007` is a superseded intermediate run because the non-core Pokémon placement and
  Meowth Supporter routing rules changed afterward.
- The corrected candidate source is represented by
  `outputs/competition_selfplay/mechanical_v2_selfplay/audit-20260719-008/`: 20/20 completed, zero
  invalid actions, crashes, timeouts, or fallbacks. Across 91 attacks, 90 followed at least one
  same-turn action. The sole turn-action-count-1 attack had only currently unusable/redundant
  alternatives; there were no selected search actions below 10, multi-card draws at or below 10,
  voluntary draws at or below 6, or attacks with a manual attachment still available. This is
  runtime evidence only and remains unaccepted until human replay inspection.
- Fully isolated packaged v2 self-play: 20/20 completed, zero fallbacks or forbidden dependencies;
  mean action latency about 0.19 ms.
- The earlier 39/1 legal-baseline report predates the per-card ordering rewrite and is no longer
  evidence for the current archive. It has not been reused as a performance claim.
- A local-only replay animator is available at
  `competition_selfplay/replay_viewer/index.html`. It loads the Kaggle JSON visualization frames,
  supports play/pause, stepping, timeline seeking, and never uploads the selected file.

## Next gate

For the independent mechanical fallback, human-inspect `audit-20260719-008` with the local animator
against the mechanical contract. Do not
use submission `54825132`, `audit-006`, or their games as training material. Do not rebuild/upload a
new archive until the corrected local replay decisions pass that inspection. Any further policy
revision should be driven by a concrete failed decision and its key-card state. After the mechanical
backup stabilizes, keep that acceptance decision separate from semantic self-play: formal Phase A
can use the frozen learned snapshot path and must not ingest rejected mechanical games.
