# Competition self-play — fixed Raging Bolt Ogerpon deck

This package is the competition-oriented fixed-deck route. It is pinned to the exact ordered
Raging Bolt Ogerpon list from Kaggle submission `54815037`; deck name, source index, patched-card
count, Card IDs, copy ordinals and the 60-card hash are validated before use.

The card vocabulary is generated deterministically from this deck alone. It does not consume the
repository's earlier global static-card features, replay-imitation pipeline, or the intermediate
8-prompt decision-agent route.

## Self-play contract

The intended loop is asymmetric self-play: learner and frozen opponent start from the same policy;
only the learner updates; after seat-balanced evaluation clears the promotion threshold, the
learner is copied to the frozen side. Repeated promotion continues until the stopping criterion is
met.

The checked-in promotion and training numbers are provisional wiring values, not approved final
hyperparameters. In particular, the current `0.58` promotion threshold and six failed evaluations
must not be treated as settled design decisions.

## Reward status

The checked-in `reward.py` still exposes the obsolete
`[outcome, prize_progress, setup_tempo]` prototype so its interfaces and tests remain executable.
The `setup_tempo` shaping and its YAML weights were rejected as too coarse and must be replaced
before reinforcement learning starts.

The confirmed direction is a three-dimensional terminal-reason critic:

- the opponent cannot maintain an Active Pokémon;
- this policy takes all Prize cards;
- this policy decks itself out.

The first two are different win mechanisms produced mainly by improving this deck's own board;
self deck-out remains an explicit controllable loss. Opponent deck, hand and discard counts are not
model features. No generic board-potential shaping has been approved.

## Mechanical fallback and replay inspection

The deterministic mechanical agent and its rule contract are in `mechanical_agent.py` and
`MECHANICAL_POLICY.md`. Build and exercise the isolated submission with:

```bash
python -m competition_selfplay.build_mechanical_submission
python -m competition_selfplay.run_mechanical_selfplay --episodes 20
```

Each run writes replay JSON files and a manifest to a timestamped directory under
`outputs/competition_selfplay/mechanical_v2_selfplay/`, unless `--output-dir` is supplied. These
commands do not submit anything to Kaggle.

For manual inspection, open `replay_viewer/index.html` and load a replay JSON file. The viewer runs
locally and does not upload the replay.

## Validation and partitions

```bash
python -m competition_selfplay.cli --dry-run
python -m pytest -q
```

- Project body: `competition_selfplay/`, `decks/`, runtime builder and tests.
- Durable collaboration record: `records/competition_selfplay/CURRENT.md`.
- Generated replay, checkpoint, rollout and metric data: `outputs/competition_selfplay/`.

The rejected submission `54825132` and its `audit-006` games are prohibited as training material.
The current mechanical candidate still requires human replay inspection; see the durable record for
the exact gate.
