# Transactional semantic competition self-play

This package is the fixed-deck reinforcement-learning route for the exact ordered Raging Bolt
Ogerpon list from submission `54815037`. It does not consume replay imitation data, alter deck
contents, use the mechanical policy as a reward oracle, or define rewards for concrete Card IDs.

## Locked training contract

Simulator selections are assembled into causal transactions. A transaction begins at a player's
root choice and contains every nested choice until that player reaches another root choice, control
passes, the turn ends, or the game ends. A forced choice is retained in its event history but has
no actor log probability. PPO therefore uses one advantage and the summed non-forced log
probability for the whole transaction.

The formal reward is scalar:

```text
terminal outcome + alpha * (gamma * frozen_target_phi_after - frozen_target_phi_before)
```

Terminal outcomes are exactly `+1/-1/0`, terminal potential is zero, `gamma=0.997`, and the
potential is clipped to `[-0.8, 0.8]`. There are no prize, attachment, attack, evolution, hand,
board-preservation, self-KO, or Card-ID bonuses.

The shared encoder predicts nine fixed seat-relative concepts. Applicability comes only from
deterministic state/rule facts. Confidence comes from semantic-head ensemble disagreement and is
detached before the potential. The semantic potential uses learned one-dimensional piecewise
linear functions and six named interactions. The full scalar critic is always:

```text
full_value = semantic_value + residual_value
```

Concept, semantic-value, stop-gradient residual, and full-value losses are all consumed by the PPO
update and reported separately.

The compact global feature prefix remains stable. The shared input additionally carries the
learner's own deck count (needed by `self_deckout_risk_h6`), deterministic semantic facts, and
visible physical card identity. Opponent hidden hand/deck/discard contents and counts are excluded.
Legal option features resolve area/index references back to Card ID and serial without any
Card-ID-specific reward rule.

## Phase A and Phase B

The first 20,000 completed training games are Phase A. `alpha=0`, so the formal reward is strictly
terminal outcome, while the actor, full critic, concept ensemble, semantic potential, and residual
head all train at transaction granularity.

A fixed, optimizer-excluded trajectory holdout controls the one-way transition to Phase B. It must
simultaneously achieve at least 15% Brier improvement over a constant prior, ECE at most 0.10,
seat-swap antisymmetry error at most 0.08, and value-ranking accuracy at least 0.60. Failure leaves
the system in Phase A. After the first pass, alpha ramps from zero to 0.15 over 50,000 completed
games.

Each rollout batch alternates the learner between P0 and P1. The opponent snapshot and complete
target semantic path are frozen during collection and learner optimization; each transaction stores
the target potential values used by its reward. After the learner update, the target encoder,
concept ensemble, confidence buffers, and potential head receive one EMA update. The opponent moves
only when the existing league controller promotes the learner.

## Running

Validate configuration and unit/integration wiring without writing outputs:

```bash
python -m competition_selfplay.cli --dry-run
python -m pytest -q tests/test_competition_selfplay.py
```

Start a training run by explicitly choosing the batch size. The runtime root must contain the
official local `cg` package when it is not installed on `PYTHONPATH`:

```bash
python -m competition_selfplay.train_selfplay \
  --runtime-root kaggle/datasets/cg_runtime \
  --games-per-batch 512
```

The runner first fixes a holdout, then collects on-policy training games, updates the learner,
updates the target semantic EMA at the post-batch boundary, evaluates calibration when eligible,
and periodically routes seat-balanced results through league promotion. Metrics and checkpoints go
under a new run directory in `outputs/competition_selfplay/`; existing runs are not overwritten.

The independent deterministic fallback remains in `mechanical_agent.py`. Its build/run commands and
human replay-inspection contract are documented in `MECHANICAL_POLICY.md`; its rejected games are
not training material.

## Partitions

- Project body: `competition_selfplay/`, `decks/`, and maintained tests.
- Durable collaboration record: `records/competition_selfplay/CURRENT.md`.
- Generated rollouts, holdouts, metrics, checkpoints, and replays: `outputs/competition_selfplay/`.
