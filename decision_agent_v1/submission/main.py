from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path
from typing import Any


_SOURCE_PATH = globals().get("__file__")
if _SOURCE_PATH is None:
    _SOURCE_PATH = sys._getframe().f_code.co_filename
ROOT = Path(str(_SOURCE_PATH)).resolve().parent
if not (ROOT / "deck.csv").exists():
    for _candidate in (Path("/kaggle_simulations/agent"), Path.cwd().resolve()):
        if (_candidate / "deck.csv").exists():
            ROOT = _candidate
            break
# Kaggle temporarily appends this directory only while exec() runs main.py and
# pops that appended entry before the first lazy model load. Keep our own copy
# at the front; the loader will remove its trailing copy, not this one.
sys.path.insert(0, str(ROOT))

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

_DECK: list[int] | None = None
_MODEL_AGENT: Any = None
_SIMULATOR_ADAPTER: Any = None
_LOAD_ATTEMPTED = False
_LOAD_ERROR: str | None = None
_MAX_TURN_ACTIONS = 100
_STATS = {
    "deck_requests": 0,
    "model_actions": 0,
    "fallback_actions": 0,
    "load_failures": 0,
    "inference_failures": 0,
    "loop_break_actions": 0,
}


def _read_deck() -> list[int]:
    global _DECK
    if _DECK is None:
        with (ROOT / "deck.csv").open(encoding="utf-8", newline="") as handle:
            _DECK = [int(row[0]) for row in csv.reader(handle) if row]
        if len(_DECK) != 60:
            raise RuntimeError(f"deck.csv must contain 60 rows, got {len(_DECK)}")
    return list(_DECK)


def _load_model() -> None:
    global _LOAD_ATTEMPTED, _LOAD_ERROR, _MODEL_AGENT, _SIMULATOR_ADAPTER
    if _LOAD_ATTEMPTED:
        return
    _LOAD_ATTEMPTED = True
    try:
        import torch

        from decision_agent_v1.adapters.card_vocab_adapter import CardVocabulary
        from decision_agent_v1.adapters.observation_adapter import ObservationAdapter
        from decision_agent_v1.adapters.simulator_adapter import SimulatorAdapter
        from decision_agent_v1.contracts.action_contract import ActionSemanticsContract
        from decision_agent_v1.inference.policy_value_agent import PolicyValueAgent

        torch.set_num_threads(2)
        metadata = json.loads((ROOT / "model_config.json").read_text(encoding="utf-8"))
        expected_hashes = {
            key: str(metadata[key])
            for key in ("data_schema_hash", "action_contract_hash", "card_vocabulary_hash")
        }
        vocabulary = CardVocabulary.from_json(ROOT / "card_vocab.json")
        contract = ActionSemanticsContract.load(ROOT / "action_semantics.json")
        _SIMULATOR_ADAPTER = SimulatorAdapter(ObservationAdapter(vocabulary), contract)
        _MODEL_AGENT = PolicyValueAgent.from_checkpoint(
            str(ROOT / "model.pt"), device="cpu", expected_hashes=expected_hashes
        )
    except Exception as exc:  # Kaggle must retain a legal agent if model loading fails.
        _LOAD_ERROR = f"{type(exc).__name__}: {exc}"
        _STATS["load_failures"] += 1
        print(f"V1 model disabled; using legal fallback ({_LOAD_ERROR})", file=sys.stderr)


def _fallback_action(observation: dict[str, Any]) -> list[int]:
    select = observation.get("select") or {}
    options = select.get("option") or []
    if not options:
        return []
    minimum = max(0, int(select.get("minCount", 0)))
    maximum = max(minimum, int(select.get("maxCount", minimum)))
    select_type = int(select.get("type", -1))
    option_types = {int(option.get("type", -1)) for option in options}
    count = 1 if select_type == 8 or 0 in option_types or maximum <= 1 else minimum
    count = min(max(count, minimum), maximum, len(options))
    return list(range(count))


def _is_legal_shape(observation: dict[str, Any], action: object) -> bool:
    if not isinstance(action, list) or not all(type(index) is int for index in action):
        return False
    select = observation.get("select") or {}
    options = select.get("option") or []
    minimum = max(0, int(select.get("minCount", 0)))
    maximum = max(minimum, int(select.get("maxCount", minimum)))
    return (
        minimum <= len(action) <= maximum
        and len(action) == len(set(action))
        and all(0 <= index < len(options) for index in action)
    )


def _loop_break_action(observation: dict[str, Any]) -> list[int] | None:
    current = observation.get("current") or {}
    if int(current.get("turnActionCount", 0)) < _MAX_TURN_ACTIONS:
        return None
    select = observation.get("select") or {}
    minimum = max(0, int(select.get("minCount", 0)))
    maximum = max(minimum, int(select.get("maxCount", minimum)))
    if not minimum <= 1 <= maximum:
        return None
    for index, option in enumerate(select.get("option") or []):
        if int(option.get("type", -1)) == 14:  # OptionType.END
            return [index]
    return None


def _stats_snapshot() -> dict[str, Any]:
    return {**_STATS, "load_error": _LOAD_ERROR}


# Kaggle's loader selects the last callable created by main.py. Keep the public
# agent function last in this file; adding a callable below it will change the
# competition entrypoint.
def agent(obs_dict: dict[str, Any]) -> list[int]:
    if obs_dict.get("select") is None:
        _STATS["deck_requests"] += 1
        if _SIMULATOR_ADAPTER is not None:
            _SIMULATOR_ADAPTER.reset()
        return _read_deck()

    loop_break = _loop_break_action(obs_dict)
    if loop_break is not None:
        _STATS["loop_break_actions"] += 1
        return loop_break

    _load_model()
    if _MODEL_AGENT is not None and _SIMULATOR_ADAPTER is not None:
        try:
            action = _MODEL_AGENT.act(_SIMULATOR_ADAPTER.adapt(obs_dict))
            if _is_legal_shape(obs_dict, action):
                _STATS["model_actions"] += 1
                return action
            raise RuntimeError(f"model returned an invalid action shape: {action!r}")
        except Exception as exc:
            _STATS["inference_failures"] += 1
            if _STATS["inference_failures"] <= 3:
                print(f"V1 inference fallback ({type(exc).__name__}: {exc})", file=sys.stderr)

    _STATS["fallback_actions"] += 1
    return _fallback_action(obs_dict)
