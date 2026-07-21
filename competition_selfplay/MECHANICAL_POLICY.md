# Raging Bolt Ogerpon mechanical policy v2

This document is the durable human-readable contract for the uploadable rules agent in
`mechanical_agent.py`.

## Fixed names

- `1`: Latias ex.
- `2`: Teal Mask Ogerpon ex.
- `3`: Wellspring Mask Ogerpon ex.

## Core board

The target field contains two copies of `2`, one copy of `1`, and at least two additional Pokémon
with at least 210 HP. Before selecting End, the policy applies a hard position gate: `2`, `1`, and
`3` must leave the Active Spot whenever a legal retreat exists. The only exceptions are an attack by
`2`/`3`, which itself ends the turn, and `1` when it can survive while every alternative cannot.
This is a decision layer, not a combat-value score. Area Zero Underdepths is played immediately when
a Tera Pokémon is in play. Setup and main-phase placement cap the protected core at two `2`, one `1`,
and one `3`; additional copies remain available as recovery rather than occupying formation slots.

If Area Zero or the last Tera Pokémon leaves play, forced Bench reduction follows inverse
importance: discard Chien-Pao first, then spent draw/search engines, then unpowered situational
Pokémon. Powered attackers and the `2/2/1` core are protected.

## Acquisition and Energy

The core acquisition order is first `2`, Grass Energy, `1`, second `2`. This base ordering remains
true outside the one branch currently being filled; in particular, `1` does not fall below the
second `2` during discard or deck-return decisions. `3` is normally a variant of `2`, but becomes
the highest Pokémon-search target when Water Energy is already in hand.

The first and second `2` are stable live-instance roles identified by card serial, not whichever
legal option happens to appear first. Grass Energy is the highest-value Energy, followed by Water,
then all other types. Manual attachment and Teal Dance share the same order: concentrate at least
three Grass on the first `2`, then keep concentrating Energy on it until its attack covers every
opposing Pokémon. Only then begin the second `2`. Damage is calculated separately for every target,
using that target's own attached Energy as the Energy it would have after becoming Active.
Non-Grass never counts toward the mandatory `{G}{G}{G}` attack cost. Myriad Leaf Shower deals:

`30 + 30 * (Energy on both Active Pokémon)`.

Crispin normally puts Grass into hand for Teal Dance and directly attaches the second Energy to
`2` or `3`. Immediate Knock Out overrides this destination: the Energy may go to the current
attacker, or to a `2`/`3` that can switch Active and Knock Out the target immediately.

Once attack costs are satisfied, spare Energy prepares retreat only when `1` is absent; Skyliner
makes every Basic Pokémon in this deck retreat for free.

## Tactical actions

- Attack is a terminal action and is below every currently useful play, attachment, Ability, and
  tactical retreat. Knock Out value chooses between attacks; it does not let an attack skip the
  remaining productive main-phase actions.

- If an opposing Benched Pokémon has more than two attached Energy and `2` can attack, Boss's
  Orders promotes the highest-Energy target. With three Energy on both Active Pokémon, `2` deals
  210 damage.
- `3` uses Torrential Pump when available and sends the 120 Bench damage to the opposing Pokémon
  with the most attached Energy.
- If one attachment lets the current Active Knock Out its opponent, that attachment overrides
  long-term setup.
- Iron Leaves ex is held for an immediate Rapid Vernier switch, Energy transfer, and 180-damage
  Knock Out.
- Raging Bolt ex uses Bellowing Thunder only as a confirmed finisher because it consumes the
  board's Energy. Burst Roar is disabled near deck-out.
- Mega Kangaskhan ex draws two while Active whenever the multi-card draw gate permits, then may
  retreat for free through `1`; it is not
  treated as a routine tank because its Knock Out gives up three Prize cards.
- Meowth ex searches a needed Supporter when played, then becomes an early discard candidate.
- Fezandipiti ex draws three only after an allied Knock Out and only while the deck is safely above
  the draw cutoff.
- Lillie's Clefairy ex gains priority against Dragon Pokémon or wide Benches. Passimian is a
  one-Prize wide-board attacker. Chien-Pao is played for Stadium removal only when it will not
  destroy our Area Zero board.

## Deck-out gates

- Deck above 10: use productive draw effects aggressively before attacking. This includes Pokémon
  effects, not only Trainer cards: Teal Dance draws 1, Run Errand 2, Flip the Script 3, Unfair Stamp
  5, Burst Roar 6, and Lillie's Determination 6 or 8. At exactly 10, deck search remains legal, but
  only one-card voluntary draw is enabled.
- Deck from 7 through 9: deck search is forbidden and only one-card draw effects are allowed. Recheck
  the deck count after each Teal Dance; stop as soon as it reaches 6.
- Deck at or below 6: no voluntary draw and no deck search. This includes Teal Dance, Run Errand,
  Flip the Script, Last-Ditch Catch, Ultra Ball, Crispin, Cyrano, Ciphermaniac's Codebreaking,
  Lillie's Determination, Unfair Stamp, and Burst Roar.
- One dynamic all-card importance ordering is generated without an aggregate board-potential
  estimate. It first selects the earliest unmet formation requirement: first `2`, the first `2`'s
  Grass, `1`, second `2`, the second `2`'s Grass, then the two non-core 210+ HP positions. A Water in
  hand creates the documented immediate `3` route. Each of the fixed deck's 27 concrete card IDs
  then has its own rule for that state; Trainer and Pokémon categories are never used as shared
  ranking buckets. For example, Crispin rises specifically for a Grass/Energy gap, Cyrano and Ultra
  Ball rise for a missing Pokémon component, and Meowth rises only when its Supporter search reaches
  the current missing component. Every search takes this ordering forward; every hand/field discard
  and shuffle-back uses it in reverse. Attached-Energy disposal derives from the same ordering while
  protecting a key attack cost.
- Multi-card acquisition advances the virtual requirement after every selected card: for example,
  Ciphermaniac chooses first `2`, then Grass, rather than choosing two copies of `2` from one stale
  state. Ciphermaniac's simulator `TO_DECK` context means "place chosen cards on top" and therefore
  uses acquisition order, not the disposal inverse.
- Disposal buckets start with spent Stadium-removal tools, then spent or redundant draw/search
  engines. Current lethal/setup resources such as Grass recovery and Energy movement remain above
  those engines. Attached Grass raises the current importance of its holder.
- Night Stretcher, Glass Trumpet, Energy Switch, manual attachment, retreat, and attack remain
  available near deck-out because they do not search or draw from the deck.
- Other Pokémon enter play only when they fill one of the two required non-core 210+ HP positions
  or their concrete effect is currently useful. Meowth requires a specific missing Supporter route;
  an unrelated Supporter in hand does not block its Crispin/Cyrano search. Chien-Pao requires a
  removable Stadium, and an unpowered `3` without Water does not consume a formation slot.

## Submission boundary

The standalone archive contains `main.py`, `mechanical_agent.py`, `deck.csv`, and the `cg` runtime.
It has no Torch or checkpoint dependency. Building and validating the archive are local operations;
uploading it to Kaggle remains a separately approved action.

## Human replay inspection

`replay_viewer/index.html` loads Kaggle replay JSON locally and animates the recorded
`visualize.current` frames. It supports drag-and-drop, a timeline, playback speed, keyboard stepping,
both boards, HP/Energy state, selection details, and logs. The JSON stays inside the browser.
