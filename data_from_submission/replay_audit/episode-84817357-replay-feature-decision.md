# Replay Feature Decision

Replay: `data_from_submission/replays/episode-84817357-replay.json`

Episode: `84817357`

Teams: `['Lapra5', 'Jiachen Li']`

## Data Read

- Steps: `110`
- Agent observations: `220`
- Missing current/select observations: `4` / `4`
- Parser errors: `0`
- Unique visible card IDs: `30`

## Dynamic Feature Implications

- Card instance tokens must be variable length. This replay has instance count mean `48.68`, p90 `77`, p99 `88`, max `91`.
- A fixed board-token budget below 128 is risky. Token estimate max is `112` in this single replay; use padding/mask and start with a budget around 128-160 if batching needs a hard cap.
- Recent event tokens should stay capped, but raw logs can burst. This replay has event count max `49` while recent event memory caps at `16`. Match/state features therefore include current log count, reverse log count, and public-card log count.
- Hidden zone handling is required. Hidden instances mean `10.06`, max `12`; opponent hand should remain count-only.
- Static detail aggregation is justified. Observed visible/static card instances have detail count p99 `4`, max `4` in this replay; summary-only would discard useful attack/ability/effect separation.
- Action/options must be variable length. Options max `26`; old fixed candidate features are not enough for the next policy head.

## Current Feature Decision

- Keep: static `card_summary` plus explicit `detail_tokens` attention aggregation.
- Keep: per-card dynamic board state and appearance/memory groups.
- Keep: state, decision, match, ledger, and recent-event board tokens.
- Add now: current observation log summary features in match token.
- Do not assume fixed game length. Train samples should be decision-point rows with per-sample masks; full games can be grouped only by metadata.

## Needs More Data

This is one public replay. Before freezing dimensions, run the same audit over many replays or self-play games and inspect p99/max for instance count, option count, raw logs, and token estimate.
