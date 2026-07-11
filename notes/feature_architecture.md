# Feature Architecture Notes

Current decision snapshot for card-aware model feature construction.

## Core Feature Groups

The card-aware model should treat features as four separate groups with clear
ownership boundaries:

1. Static card features
   - Stable facts from card data.
   - Examples: card type, Pokemon or provided energy type, HP, retreat cost,
     attacks, attack costs, rule flags, text-derived embedding.
   - Basic Energy type separation belongs here.
   - Current static card embedding output is 128 dimensions.

2. Per-card board-state dynamic features
   - Current match state for a visible card instance.
   - Examples: current HP/damage, attached energy, active/bench/hand/discard
     zone, owner, special conditions, turn-local state.
   - The previous hard-coded 37-dimensional dynamic construction was removed
     because its concrete assumptions were premature.

3. Card appearance features
   - Metagame or dataset-derived priors about card usage.
   - Reserved as a separate feature group from static card facts because these
     values can change with the meta or training data window.
   - Planned split:
     - appearance-combination features: whether this card appears with a known
       friendly/opponent card or deck pattern;
     - popularity features: appearance rate, usage rate, or recent popularity.

4. Global state features
   - Whole-game context that is not owned by one card.
   - Examples: turn number, first player, supporter/stadium/energy flags,
     hand/deck/prize/discard counts, action/select context, board summaries.

## Current Encoder Boundary

`models.card_instance_encoder.CardInstanceEncoder` now reserves the per-card
fusion interface without hard-coding dynamic feature semantics:

```text
card_static_embedding: 128 dimensions by default
board_state_features: 32 reserved dimensions by default
appearance_features: 32 reserved dimensions by default
output: 128 dimensions by default
```

The encoder accepts missing board-state or appearance tensors and fills them
with zeros. This keeps static-only experiments compatible while leaving stable
slots for future dynamic and appearance features.

## Design Constraints

- Do not mix static card identity facts with metagame appearance priors.
- Version appearance features separately from static card data because they are
  derived from training/replay populations and may drift.
- Keep global state features outside per-card instance embeddings unless a
  later model explicitly conditions card fusion on global context.
- Avoid adding new fixed dynamic dimensions until the observation schema and
  downstream model usage are clear.
