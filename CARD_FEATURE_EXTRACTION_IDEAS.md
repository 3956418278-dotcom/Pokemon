# Card feature extraction ideas

This note is for the next training redesign. It focuses only on card-aware feature extraction and does not change the current Kaggle training code.

## Goal

Replace the current weak 48-dimensional hand-written features with features that let the model understand cards, board state, and legal action quality.

The first implementation should stay simple and Kaggle-safe:

- build static card metadata from `EN_Card_Data.csv` or `cg.api.all_card_data()`;
- encode visible observation state from `cg.api.to_observation_class`;
- encode each legal option as an action candidate;
- keep training and generated submission inference using the exact same encoder code.

## Current problem

Current features mostly encode:

- turn/action counters;
- coarse zone counts;
- select type/context;
- option type and a few IDs.

This loses the main TCG information:

- active Pokemon identity, HP, damage, energy, retreat cost, attacks;
- hand contents;
- bench threats and damaged targets;
- discard contents;
- energy type compatibility;
- evolution chains;
- prize tempo;
- deck identity;
- whether a legal action is actually sensible.

## Recommended feature layers

### 1. Static card registry

Create a `CardRegistry` loaded once from `EN_Card_Data.csv` if available, falling back to `cg.api.all_card_data()`.

Per card ID, store compact numeric metadata:

- card type one-hot: Pokemon, Item, Tool, Supporter, Stadium, Basic Energy, Special Energy;
- Pokemon stage: Basic, Stage 1, Stage 2, other;
- rule flags: ex, Mega ex, ACE SPEC, Ancient, Future, Team Rocket, named trainer families if exposed;
- HP normalized;
- Pokemon/energy type one-hot;
- weakness/resistance one-hot;
- retreat cost normalized;
- attack count;
- max printed attack damage;
- mean printed attack damage;
- total printed attack energy cost;
- attack cost type counts;
- has ability;
- has play effect;
- text keyword flags from card/effect text.

Useful first keyword flags:

- draw/search;
- discard;
- switch/gust;
- attach energy/accelerate energy;
- recover from discard;
- evolve/devolve;
- damage counter/place counters;
- heal;
- coin flip;
- stadium bump;
- hand disruption;
- prize interaction.

Do not try to fully parse card text at first. Keyword flags are enough for a first useful jump.

### 2. Visible card instance encoder

For each visible card or Pokemon instance from the observation, combine static card metadata with dynamic state:

- card ID embedding or hashed ID bucket;
- zone: active, bench, hand, discard, prize visible, stadium, looking;
- owner: self or opponent;
- current HP fraction;
- damage fraction;
- attached energy count;
- attached energy type counts;
- attached tool count;
- evolved depth / pre-evolution count;
- appeared this turn;
- special conditions for active Pokemon;
- can attack now if legal attack option exists;
- can retreat now if legal retreat option exists.

For normal competition observations, opponent hand and face-down prizes are hidden. Encode hidden zones only as counts and unknown masks. Do not leak hidden card IDs into the student model.

### 3. Board summary encoder

Build a fixed-size summary for each player:

- active Pokemon vector;
- aggregate bench vector: mean/max/sum over bench Pokemon features;
- best bench attacker estimate;
- lowest remaining HP target;
- total attached energy;
- energy type distribution;
- hand size;
- deck count;
- prize count;
- discard summary by type and important keywords;
- support/stadium/energy/retreat used flags;
- first-player and turn parity.

Important derived features:

- prize race: self prizes remaining minus opponent prizes remaining;
- board development: self Pokemon in play, opponent Pokemon in play;
- energy tempo: self attached energy minus opponent attached energy;
- threat estimate: opponent active max attack damage vs self active remaining HP;
- KO opportunity estimate: self active max attack damage vs opponent active remaining HP.

### 4. Hand and visible-zone set encoders

For self hand, use permutation-invariant pooling:

- mean card feature vector;
- sum card type counts;
- keyword counts;
- counts of Basic Pokemon, evolution Pokemon, energy, draw/search cards, switch cards, gust cards;
- duplicate counts for important card IDs.

For discard, use similar pooling but emphasize:

- energy in discard;
- Pokemon lines in discard;
- recoverable resources;
- discarded key attackers/support cards.

For looking/deck-search selections, encode the visible `select.deck` cards as a temporary candidate pool. This is important for search cards where the best action depends on the revealed deck subset.

### 5. Action candidate encoder

Each legal candidate should be encoded as:

- option type one-hot;
- select context one-hot;
- selected card metadata if the option references a card;
- target Pokemon dynamic metadata if the option targets a Pokemon;
- source and target zones;
- source owner and target owner;
- attack metadata if `attackId` is present;
- attach/evolve/retreat/end flags;
- candidate size for multi-select actions;
- whether candidate is empty.

Derived action-quality features:

- `is_attack_for_ko`: selected attack appears to KO opponent active;
- `is_attack_low_damage`: attack damage is much lower than another legal attack;
- `is_energy_to_active`: attaching energy to active;
- `is_energy_to_bench_attacker`: attaching energy to a bench Pokemon with strong attacks;
- `is_evolve_active` / `is_evolve_bench`;
- `is_switch_to_damaged_target` / `is_switch_to_fresh_target`;
- `is_discard_energy`, `is_discard_pokemon`, `is_discard_dead_card`;
- `is_end_turn_with_playable_actions`: ending while useful actions remain;
- `is_setup_basic`: setup action puts a Basic Pokemon into play.

This action encoder is the best place for the behavior plausibility layer.

## Model-facing structure

Prefer a two-tower candidate scorer:

```text
state_vector = board_encoder(observation, deck_id)
action_vector = action_encoder(observation, candidate)
score = MLP([state_vector, action_vector, state_vector * action_vector])
```

The policy scores every legal candidate and softmaxes over candidates. The value head should use only `state_vector`.

Add deck conditioning early:

- deck ID one-hot for the 8 baseline decks;
- or learned deck embedding;
- plus static decklist summary: Pokemon/type/energy/trainer counts and keyword counts.

Deck conditioning matters because a single shared policy should not treat Dragapult, Zoroark, Crustle, etc. as the same strategy.

## Implementation phases

### Phase 1: deterministic metadata features

Lowest-risk first step:

- parse `EN_Card_Data.csv`;
- build `card_features(card_id)`;
- enrich current `option_features` with selected card static metadata;
- enrich current `state_features` with active Pokemon and hand/discard summaries;
- increase `FEATURE_DIM` only after train and submission encoders are unified.

This should already outperform ID-only features.

### Phase 2: dynamic Pokemon and action target features

Add helpers to resolve an `Option` into:

- referenced card;
- referenced Pokemon;
- source zone;
- target Pokemon for attach/evolve/damage/switch.

Then encode HP, damage, energy, tool, special condition, and attack feasibility.

### Phase 3: plausibility labels

Before expensive PPO:

- collect legal action candidates from self-play;
- label obvious bad actions with heuristics;
- label reasonable actions from rule baseline, oracle search, or stronger rollouts;
- train a plausibility model to down-weight nonsensical legal choices.

Examples of obvious bad labels:

- end turn while attack or high-value setup action is available;
- discard a key attacker/evolution when discard has clear filler;
- attach off-type energy when same-type attacker needs energy elsewhere;
- switch into a Pokemon that is immediately KO-able without benefit.

### Phase 4: oracle or search-augmented labels

Use `cg.api.search_begin` or a modified engine only for teacher data generation. The student encoder must still use normal observation data.

Record:

- normal observation;
- legal candidates;
- candidate features;
- oracle action/value;
- rollout value by candidate where affordable;
- hidden-state metadata only in teacher records, not student input.

## Cautions

- Do not use hidden opponent hand/deck/prize card IDs in the student features.
- Do not let future random outcomes become direct labels. Average over rollouts if randomness is involved.
- Keep card registry deterministic and versioned because model weights depend on feature order.
- Unify train-time and submission-time feature code before growing the feature vector. Duplicated encoders are a high regression risk.
- Avoid overfitting to only 8 decks. Include generic card metadata and deck summaries so the model generalizes if decks change.

## First concrete design choice

For the next code pass, I would implement:

1. `card_registry.py` with stable `card_feature_names` and `card_features(card_id)`.
2. `feature_encoder.py` shared by training and submission generation.
3. A candidate scorer using:
   - board summary;
   - self hand summary;
   - self/opponent active Pokemon vectors;
   - selected option card vector;
   - selected target Pokemon vector;
   - deck ID one-hot.
4. A compatibility check that writes `feature_schema.json` beside `model.json`.

This gives us a strong foundation before touching frozen opponents, behavior cloning, or oracle teacher training.
