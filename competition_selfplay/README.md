# Competition self-play — fixed Raging Bolt Ogerpon deck

This is the new competition-oriented route. It does not train from replay imitation and does not
modify the existing replay pipeline. Its target is pinned to Kaggle submission `54815037`, the
latest scored submission (`COMPLETE`, public score `258.4`) when this route was created. The deck
name, source index, absence of patched cards, and exact ordered 60-card hash are validated before
training.

The loop is asymmetric self-play: learner and frozen opponent start from the same parameters; only
the learner updates; after a seat-balanced evaluation exceeds the configured score-rate threshold,
the learner checkpoint is copied to the frozen side. Repeated promotion ends at the promotion cap,
or the current frozen policy is selected after several consecutive failed promotion evaluations.

## Three-dimensional reward

The critic target is `[outcome, prize_progress, setup_tempo]`, never a single `win = +1` target.
The actor may scalarize this vector using configurable weights.

- `outcome`: ordinary terminal result. A win caused only by the opponent decking out is zero by
  default; losing because this agent decks out receives an explicit stronger penalty.
- `prize_progress`: change in the prize-card race, normalized by six.
- `setup_tempo`: potential change based on active/bench setup and attached Energy, with only public
  on-field opponent counts included.

Opponent deck/hand/discard counts are absent from both `BattleSnapshot` and the compact global
feature schema. Deck-out is consumed only as terminal reason `2`; it is not inferred from an input
deck counter. Card inputs are Card-ID embedding indices plus `copy_ordinal`, `copies_in_deck`, and
the live instance serial/copy embedding supplied by the simulator adapter.

## Partitions

- Project body: `competition_selfplay/` and `tests/test_competition_selfplay.py`.
- Durable collaboration record: `records/competition_selfplay/CURRENT.md`.
- Generated checkpoints, rollouts, and metrics: `outputs/competition_selfplay/` (created by the
  eventual trainer, not by config validation).

Validate wiring without starting training:

```bash
python -m competition_selfplay.cli --dry-run
pytest -q tests/test_competition_selfplay.py
```

This stage intentionally defines the reward, feature boundary, target deck, training parameters,
and promotion/convergence controller. It now also contains an independently uploadable deterministic
mechanical agent. See `MECHANICAL_POLICY.md` for its rules.

Build the standalone rules submission locally:

```bash
python -m competition_selfplay.build_mechanical_submission
python -m competition_selfplay.run_mechanical_selfplay --episodes 20
```

The generated archive is written under `outputs/competition_selfplay/submission_mechanical_v2/`.
Every mechanical self-play run also writes one replay JSON per episode plus a `manifest.json` under
a new timestamped directory in `outputs/competition_selfplay/mechanical_v2_selfplay/`. Use
`--output-dir PATH` to choose an explicit replay directory. Building or running does not submit
anything to Kaggle. Simulator rollout collection and vector-value PPO updates remain the next
training stage.

For human replay inspection, open `competition_selfplay/replay_viewer/index.html` and drag in a
Kaggle replay JSON file. The standalone viewer provides frame-by-frame and timed playback without
uploading the replay. See `replay_viewer/README.md` for optional URL loading through a local static
server.
