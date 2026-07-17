# Replay decision data contract

This document freezes the model-facing data boundary before the staged model
changes. It is intentionally independent from any one encoder implementation.

## Decision identity and action alignment

The primary key is:

```text
episode_id + decision_step_index + action_step_index + player_index
```

For each player, extraction performs these operations in order:

```text
bind current agent_step.action to the previous pending decision
close that pending decision
deduplicate the current policy-visible observation
apply the new log batch and current snapshot to that player's memory
create a pending decision when observation.select is present
```

The fingerprint covers canonical `current`, `logs`, and `select`. It is an audit
field, not the primary key. A 60-card initial deck submission without a pending
decision is configuration input and is excluded from behavior cloning.

Formal exports are reference datasets. They retain the primary key,
`replay_member`/source path, action targets, supervision, and audit fields. They
do not copy observations, logs, Card Instances, or Memory snapshots. Those are
rebuilt from the source Replay at load time.

## Presence states

Every nullable, missing, unknown, or inapplicable scalar/reference carries one
of the following states independently from its placeholder value:

```text
PRESENT
MISSING
UNKNOWN
NOT_APPLICABLE
EXPLICIT_NULL
```

The placeholder value must never be interpreted without this state. Detail use
uses the narrower `TRUE/FALSE/UNKNOWN/NOT_APPLICABLE` contract.

Both decision references are retained:

```text
effect_reference + effect_presence
context_card_reference + context_card_presence
```

Neither globally overrides the other.

## Serial instances and Card ID memory

Every currently visible card that can be referenced by an option remains a
serial-level instance. This includes hand, discard, board, attachments,
evolution components, `looking`, and `select.deck`. The parser may expose a
`copy_count` feature, but it does not collapse those instances.

The serial registry is the only mutable historical identity store. Ledger
counts are derived views. Ledger and belief are merged per side and Card ID into
`[CARD_ID_MEMORY]`; the contract materializes instance-to-memory edge indices
for the bidirectional `INSTANCE_OF_CARD_ID` / `CARD_ID_HAS_INSTANCE` relations.
The static summary is joined by `card_id` during tokenization rather than copied
into every data record.

## Legal action target

Every sample records exactly one action semantic:

```text
SINGLE_INDEX
UNORDERED_UNIQUE_SUBSET
ORDERED_INDEX_SEQUENCE
INDEX_MULTISET
COUNT_VALUE
```

The target contains raw chosen option indices, an equivalence-class ID for each
option, selected counts per class, an ordered class sequence only when order is
semantic, and a resolved count value for `COUNT_VALUE` when the option exposes
`number`.

Equivalence resolution substitutes referenced source/target state for raw
indices and removes interchangeable serial identity. If a reference cannot be
resolved, its raw index remains in the signature, preventing an unsafe merge.

The complete equality key is:

```text
select type and context
option type
source and target zone
source and target card identity plus all visible dynamic state, without serial/name
detail / attack / skill / ability reference
effect reference and context-card reference, kept separately
missingness state for decision references and every option field
energy, quantity, number, special-condition, and other resolution fields
source and target owner
unresolved raw indices or direct cardId+serial as conservative fallback
```

Card ID alone is never sufficient. Direct `cardId+serial` options are first
resolved against current visible instances; serial is removed only after
successful resolution.

The final key produces 1,272,363 equivalence groups in 555,201 decisions. Group
size has median 2, P99 6, and maximum 21. Replay actions select a non-first
member of an equivalent group 46,686 times. Since every member is explicitly
listed by the engine as a legal option, class members are individually accepted;
the observed non-first selections additionally confirm that the environment is
not tied to the first serial. This supports class marginalization without
claiming that Card-ID equality alone establishes equivalence.

Unordered subset training predicts the selected class counts/cardinality and
uses fixed-cardinality subset dynamic programming. It does not enumerate label
permutations.

For option weights `w_j = exp(score_j)`, cardinality `k`, equivalence groups
`G_g`, and demonstrated group counts `n_g`, the exact group-aware subset term is:

```text
L_subset = -sum_g log e_{n_g}({w_j : j in G_g})
           + log e_k({w_j : all j})
```

where `e_r` is the degree-`r` elementary symmetric polynomial computed by the
standard descending fixed-cardinality DP. The full target also includes the
Count Head loss for `k`. This sums over every legal member subset with the same
group-count target, costs `O(Kk)`, and never penalizes an arbitrary serial choice
inside an equivalence group. For `SINGLE_INDEX`, probability is summed over the
demonstrated equivalence class. For `COUNT_VALUE`, supervision is
`select.option[action[0]].number`, with value-to-index mapping retained only for
engine inference.

## Residual hidden-zone allocation

Exact hidden copies are fixed before probabilistic allocation:

```text
u_c = expected_hidden_count_c - sum_z exact_hidden_count[c,z]
q_z = observed_zone_count_z - sum_c exact_hidden_count[c,z]
```

Both marginals must be non-negative and have equal total mass. IPF receives
only `u_c` and `q_z` and outputs `expected_zone_count`. It does not output or
imply presence probability. `probability_present` is a separately supervised
head. `zone_entropy` is the entropy of the unresolved-copy zone allocation.

The deck feature is named
`deck_archetype_compatibility_distribution`; it is not claimed to be a
calibrated Bayesian posterior. Archetypes, priors, and co-occurrence statistics
are fitted from the training time window only.

## Belief recall and rollout

Opponent Card ID recall reserves slots for presence/count, uncertainty,
decision-conditioned tactical relevance, and archetype representatives. The
tail token retains omitted expected mass, count summaries, and uncertainty.

The frozen schema supports staged activation:

```text
Facts
Events/Ledger
simple Belief
Deck Compatibility
Residual IPF
joint policy/Belief/Value training
```

Inactive modules emit masked placeholders; field positions and meanings do not
change between stages.

## Frozen eight-class record

`build_replay_decision_contract()` materializes the following in memory. It is
not another persisted copy of the Replay.

1. **Match Context** — `turn`, visible `turnActionCount`, starting-player flag,
   and turn-owner state. The Replay has no explicit turn-owner field, so that
   field remains `UNKNOWN`; parity is not guessed.
2. **Resource Context** — self/opponent deck, hand, prize, discard and free
   Bench counts, plus current-turn usage flags, each with its own field state.
3. **Card Instances** — every visible and option-referenceable serial remains a
   separate instance, including hand, discard, board, attachments, evolution
   components, `looking`, and `select.deck`.
4. **Temporal Memory** — the last 32 visible events, the serial registry as the
   only mutable identity truth, anonymous HAND/DECK/PRIZE flow counters, and
   Card ID memory derived from that registry.
5. **Hidden Belief** — staged fields for deck-archetype compatibility,
   independent presence prediction, and residual-IPF expected zone counts.
6. **Decision Context** — select type/context/count constraints and separate
   effect/context-card references.
7. **Legal Options** — raw legal options in memory plus equivalence classes and
   semantic action targets.
8. **Training Targets** — BC target, policy-loss mask, final outcome, and a
   reference to hidden truth in the source Replay.

Unknown sentinels such as `firstPlayer=-1` are encoded as `UNKNOWN` and receive
a neutral placeholder value. They never enter a normal binary embedding as
`-1`.

## Audited multi-select contexts

The full 7,500-Replay audit contains 1,128,724 aligned decisions. The seven
previously conservative contexts resolve as follows:

| select.type | context | option.type | samples | max options | max selected | candidate/effect | frozen semantics |
|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 2 | 3 | 1,806 | 5 | 4 | setup Basic Pokémon placed on Bench | `UNORDERED_UNIQUE_SUBSET` |
| 1 | 15 | 3 | 53 | 6 | 3 | Slowking copies Kyurem Trifrost; three equal-damage targets | `UNORDERED_UNIQUE_SUBSET` |
| 1 | 21 | 3 | 128 | 4 | 2 | Glass Trumpet/Janine target Pokémon set | `UNORDERED_UNIQUE_SUBSET` |
| 1 | 22 | 3 | 7,789 | 15 | 5 | Energy sources, each followed by a target decision | `ORDERED_INDEX_SEQUENCE` |
| 2 | 26 | 5 | 326 | 11 | 7 | attached Energy subset discarded for attack damage | `UNORDERED_UNIQUE_SUBSET` |
| 2 | 27 | 4 | 107 | 4 | 2 | Tool Scrapper target set | `UNORDERED_UNIQUE_SUBSET` |
| 5 | 34 | 15 | 1,238 | 4 | 4 | forced equivalent Risky Ruins triggers | `UNORDERED_UNIQUE_SUBSET` |

Context 22 is genuinely sequential: the engine consumes chosen Energy indices
in action order and opens one subsequent target select for each Energy, so
changing order can change Energy-to-target pairing. For the other six, the rule
and observed logs depend on membership/count rather than order.

No two samples had identical normalized pre-state, selected set, and decision
references. Cross-episode normalized comparisons are therefore retained as
coarse supporting evidence, not presented as controlled proof. The audit CSV
contains both exact and coarse comparison counts. Context 34 has no policy
freedom when all equivalent triggers are forced, so its policy loss is masked.

## Audit and length invariants

The frozen audit must satisfy:

```text
decision_count                 1,128,724
unpaired_pending               0
illegal_action_index           0
deck_configuration_action      15,000
duplicate_old_observation      1,143,724
duplicate action index         0
Ability detail log mapping     0 / 93,237 selected Ability actions
```

Ability detail use therefore defaults to `UNKNOWN`; only a versioned rule
resolver may emit `TRUE` or `FALSE` with an inference source.

`effect` and `contextCard` are simultaneously non-null in 37,789 decisions, so
neither can replace the other. `turnActionCount` is present and non-null in all
1,128,724 selects; within the same engine turn it strictly increases in 946,757
transitions, never stays equal or decreases, and 166,967 transitions cross a
turn boundary. It is therefore stored directly with validity rather than
reconstructed as a dataset-global decision position.

Board batching keeps all visible instances. The observed conservative upper
bound is 225 tokens (P99 about 213), with padding/relation masks and
length-bucketed batches. If a later budget is needed, Events or recalled Card ID
Memory are reduced before visible Card Instances.
