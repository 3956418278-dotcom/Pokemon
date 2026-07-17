# Replay decision data contract

This is the versioned, model-facing Replay contract. The current versions are:

```text
decision contract: replay_decision_contract_v2
compact reference: replay_decision_reference_v2
observation parser: replay_observation_parser_v2
```

## Identity, alignment, and compact references

The formal primary key is:

```text
replay_key + episode_id? + decision_step_index + action_step_index + player_index
```

`replay_key` is non-empty and is selected in this order: `EpisodeId`, replay
`id`, ZIP member/JSONL record identity, normalized source path. `episode_id`
remains nullable metadata. A JSONL line is part of source identity, so two
records without Episode IDs cannot collide.

Only actual `select` decision points enter behavior cloning. The former
`include_no_select` option was removed because it never produced a defined
training sample. A 60-card deck action without a pending decision is
configuration, not policy supervision. Actions accept integers and integer
strings; invalid values (including a list element of `None`) produce an
`ActionAlignmentError`, consume no label, and are never replaced with option 0.

A compact reference contains the key, source kind/path or archive, member or
JSONL line, source content hash, observation fingerprint, parser version,
decision schema version, action target, and supervision. It does not copy the
observation. `rebuild_replay_decision_from_reference` validates the content
hash, fingerprint, key, and both versions and fails explicitly on any mismatch.

## Missingness

Every optional model-facing value is paired with one of:

```text
PRESENT
MISSING
UNKNOWN
NOT_APPLICABLE
EXPLICIT_NULL
```

Sentinels are field-specific. `firstPlayer=-1` and `result=-1` are `UNKNOWN`;
an unrelated negative numeric value remains `PRESENT`. Missing fields and
explicit nulls remain distinct. Placeholder values are consumed only together
with their state/mask. A derived field is `PRESENT` only when every required
input is valid; for example, Bench free slots require both `benchMax` and
`bench`.

`effect_reference` and `context_card_reference` retain independent states and
never overwrite each other.

## Card instances, zones, and memory

Every visible serial remains a separate Card Instance. `select.deck` represents
a currently exposed choice list, so both its Card Instance and legal-option
source zone are `LOOKING` (12); `source="select.deck"` preserves origin. Tool
and Energy child options point their target zone to the actual parent entity's
zone.

The current snapshot has priority for exact serial positions. An anonymous log
updates cumulative anonymous zone flow only; it never clears or guesses the
position of every known serial in `fromArea`. `quantity` changes aggregate flow,
not serial identity.

Anonymous pools separate current snapshot-derived quantities from cumulative
flow:

```text
self_unknown_deck_count
self_unknown_prize_count
opponent_unknown_hand_count
opponent_unknown_deck_count
opponent_unknown_prize_count
cumulative_anonymous_zone_transitions_by_side
```

Card ID memory is derived from the serial registry and uses names matching its
calculation:

```text
currently_exact_zone_counts
ambiguous_serial_count
known_serial_count
visible_observation_count
movement_event_count
first_known_turn
last_known_turn
```

It does not call observation count a reveal count and does not duplicate belief
predictions.

## Temporal events

Replay logs do not establish their original occurrence timestamp. Events use
observation-relative names:

```text
observed_at_turn
observed_at_turn_action_count
batch_position
observation_age
turn_age
```

Batch order is preserved and reverse logs keep an independent flag. Event
features contain batch position, observation age, turn age, observed turn, and
observed turn action count; the version-2 event dimension is 19.

## Legal-option equivalence and action semantics

Each option key includes select type/context, option type, source and target
entity/owner/zone, all visible dynamic state, detail/effect/context-card
references, every settlement field, and field missingness. Explicit empty lists
are not replaced by fallback data. Missing Card ID does not permit name removal.
Serial is retained unless an audited engine rule proves different visible
copies interchangeable.

Resolution status is one of:

```text
FULLY_RESOLVED
PARTIALLY_RESOLVED
UNRESOLVED
```

Only `FULLY_RESOLVED` options may merge automatically. Class IDs are local to a
sample and have no cross-sample meaning.

The action target retains raw indices, class capacities, chosen class counts,
selected count, ordered class sequence when applicable, and count-value mapping.
The conservative context table is:

| select.type | context | semantic source | result |
|---:|---:|---|---|
| 1 | 2 | `RULE_CONFIRMED` | `UNORDERED_UNIQUE_SUBSET` |
| 1 | 5 | `UNRESOLVED` | `ORDERED_INDEX_SEQUENCE` |
| 1 | 7 | `UNRESOLVED` | `ORDERED_INDEX_SEQUENCE` |
| 1 | 8 | `UNRESOLVED` | `ORDERED_INDEX_SEQUENCE` |
| 1 | 9 | `UNRESOLVED` | `ORDERED_INDEX_SEQUENCE` |
| 1 | 15 | `RULE_CONFIRMED` | `UNORDERED_UNIQUE_SUBSET` |
| 1 | 21 | `RULE_CONFIRMED` | `UNORDERED_UNIQUE_SUBSET` |
| 1 | 22 | `REPLAY_SUPPORTED` sequential settlement | `ORDERED_INDEX_SEQUENCE` |
| 2 | 26 | `RULE_CONFIRMED` | `UNORDERED_UNIQUE_SUBSET` |
| 2 | 27 | `RULE_CONFIRMED` | `UNORDERED_UNIQUE_SUBSET` |
| 5 | 34 | `RULE_CONFIRMED` | `UNORDERED_UNIQUE_SUBSET` |

The audit reports manual rule status, Replay sample support, and whether Replay
can prove invariance separately. Replay acceptance alone is supporting evidence,
not automatic rule confirmation. Contexts 5, 7, 8, and 9 remain ordered until
card/effect rules and settlement evidence satisfy the unordered proof burden.
In the numeric Replay API these map (after the engine's `enum - 1`
serialization) to `ToBench`, `ToHand`, `Discard`, and `ToDeck`. The engine's
`setSelectedCardTarget` copies action indices into `targetList` in action order;
destination/log order therefore cannot be erased by a context-wide assumption.

## Policy-loss mask

The mask is computed after semantics and equivalence resolution. Every row has
a `policy_mask_reason`, including:

```text
SINGLE_LEGAL_OPTION
ONE_COUNT_VALUE
ONE_FEASIBLE_CLASS_TARGET
FORCED_FULL_SUBSET
UNRESOLVED_EQUIVALENCE
REAL_POLICY_CHOICE
```

Unresolved equivalence, variable counts, or ordered sequences retain policy
loss. Masking is allowed only for one engine option, one count value, or a
fully-resolved action space with one feasible semantic result.

## Turn ownership

Turn owner uses an explicit engine field when present. Otherwise, for a
two-player non-setup state with valid `firstPlayer` and `turn`, the engine rule
`((turn + 1) ^ firstPlayer) & 1` infers the owner and records
`INFERRED_AUDITED_TURN_RULE`. Setup, missing, or conflicting cases remain
`UNKNOWN`; select presence alone is never treated as ownership because forced
effects may ask the non-turn player to act. The audit reports explicit,
deterministically inferred, ambiguous, and conflicting counts.

## Deferred training systems

Group-aware subset target capacities/counts are present:

```text
TARGET CONTRACT IMPLEMENTED
TRAINING LOSS NOT IMPLEMENTED
```

No training loss module is added in this revision.

Residual IPF remains an independent mathematical utility. Replay datasets do
not run it by default. `hidden_belief_state=NOT_APPLICABLE` means disabled;
`PRESENT` requires a real computation, and `UNKNOWN` is reserved for an enabled
module with insufficient inputs. Expected zone counts are not presence
probabilities or calibrated posteriors.

`used_detail` is inactive. Non-Pokémon/no-detail rows are `NOT_APPLICABLE`;
unmapped Pokémon details are `UNKNOWN`. Only an explicit audited mapping may
emit `TRUE`/`FALSE` with an inference source. No Ability mapping or auxiliary
training target is claimed.
