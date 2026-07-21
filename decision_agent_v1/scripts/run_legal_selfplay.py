from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from decision_agent_v1.adapters.card_vocab_adapter import CardVocabulary
from decision_agent_v1.adapters.observation_adapter import ObservationAdapter
from decision_agent_v1.adapters.simulator_adapter import SimulatorAdapter
from decision_agent_v1.baseline.deterministic_legal_agent import DeterministicLegalAgent
from decision_agent_v1.contracts.action_contract import ActionSemanticsContract

from ._common import DEFAULT_CONTRACT, DEFAULT_VOCAB, OUTPUT_ROOT, ROOT, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--max-decisions", type=int, default=2000)
    parser.add_argument("--deck-index", type=int, default=0)
    parser.add_argument("--card-vocab", type=Path, default=DEFAULT_VOCAB)
    parser.add_argument("--action-contract", type=Path, default=DEFAULT_CONTRACT)
    args = parser.parse_args()

    runtime_parent = ROOT / "kaggle/datasets/cg_runtime"
    sys.path.insert(0, str(runtime_parent))
    from cg.game import battle_finish, battle_select, battle_start

    decks = json.loads((ROOT / "decks/baseline_decks.json").read_text(encoding="utf-8"))["decks"]
    deck = [int(value) for value in decks[args.deck_index]["patched_deck_ids"]]
    if len(deck) != 60:
        raise RuntimeError("selected baseline deck is not 60 cards")
    vocabulary = CardVocabulary.from_json(args.card_vocab)
    contract = ActionSemanticsContract.load(args.action_contract)
    adapter = SimulatorAdapter(ObservationAdapter(vocabulary), contract)
    agent = DeterministicLegalAgent()
    completed = 0
    crashes = 0
    invalid = 0
    timeouts = 0
    lengths = []
    errors = []
    for episode in range(args.episodes):
        decisions = 0
        started = False
        adapter.reset()
        try:
            observation, start_data = battle_start(deck, deck)
            started = observation is not None
            if observation is None:
                crashes += 1
                errors.append(
                    {
                        "episode": episode,
                        "stage": "battle_start",
                        "error_player": int(start_data.errorPlayer),
                        "error_type": int(start_data.errorType),
                    }
                )
                continue
            while decisions < args.max_decisions:
                select = observation.get("select")
                if select is None:
                    completed += 1
                    lengths.append(decisions)
                    break
                raw_options = select.get("option") or []
                if not raw_options:
                    result = (observation.get("current") or {}).get("result", -1)
                    if isinstance(result, int) and result >= 0:
                        completed += 1
                        lengths.append(decisions)
                        break
                    raise RuntimeError(
                        f"simulator emitted a non-terminal empty option list: result={result}"
                    )
                view = adapter.adapt(observation)
                action = agent.act(view)
                option_count = len(raw_options)
                if any(index < 0 or index >= option_count for index in action):
                    invalid += 1
                    raise AssertionError("baseline emitted an out-of-range option index")
                if not (view.global_state.min_count <= len(action) <= view.global_state.max_count):
                    invalid += 1
                    raise AssertionError(
                        "baseline emitted an invalid action count: "
                        f"type={view.global_state.select_type}, "
                        f"context={view.global_state.select_context}, "
                        f"mode={view.selection_mode.value}, "
                        f"min={view.global_state.min_count}, max={view.global_state.max_count}, "
                        f"options={len(view.options)}, action={action}"
                    )
                try:
                    observation = battle_select(action)
                except IndexError:
                    invalid += 1
                    raise
                decisions += 1
            else:
                timeouts += 1
                lengths.append(decisions)
        except Exception as exc:
            crashes += 1
            errors.append({"episode": episode, "stage": "loop", "error": f"{type(exc).__name__}: {exc}"})
        finally:
            if started:
                try:
                    battle_finish()
                except Exception as exc:
                    crashes += 1
                    errors.append({"episode": episode, "stage": "battle_finish", "error": str(exc)})
    payload = {
        "deck": decks[args.deck_index]["name"],
        "requested_episodes": args.episodes,
        "completed_episodes": completed,
        "crash_count": crashes,
        "invalid_action_count": invalid,
        "timeout_count": timeouts,
        "unknown_context_count": sum(agent.statistics.values()),
        "unknown_contexts": dict(agent.statistics),
        "episode_length": lengths,
        "errors": errors,
        "passed": completed == args.episodes and crashes == 0 and invalid == 0 and timeouts == 0,
    }
    output = OUTPUT_ROOT / "selfplay/deterministic_legal_selfplay.json"
    write_json(output, payload)
    if not payload["passed"]:
        raise RuntimeError(f"legal selfplay gate failed; see {output}")
    print(output)


if __name__ == "__main__":
    main()
