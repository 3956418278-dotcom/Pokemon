from __future__ import annotations

import csv
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
sys.path.insert(0, str(ROOT))

from mechanical_agent import MechanicalAgent, is_legal_action


_DECK: list[int] | None = None
_MECHANICAL_AGENT = MechanicalAgent()
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


def _fallback_action(observation: dict[str, Any]) -> list[int]:
    select = observation.get("select") or {}
    options = select.get("option") or []
    minimum = max(0, int(select.get("minCount", 0)))
    maximum = max(minimum, int(select.get("maxCount", minimum)))
    count = 1 if maximum <= 1 else minimum
    return list(range(min(max(count, minimum), maximum, len(options))))


def _loop_break_action(observation: dict[str, Any]) -> list[int] | None:
    current = observation.get("current") or {}
    if int(current.get("turnActionCount", 0)) < _MAX_TURN_ACTIONS:
        return None
    select = observation.get("select") or {}
    if not int(select.get("minCount", 0)) <= 1 <= int(select.get("maxCount", 0)):
        return None
    for index, option in enumerate(select.get("option") or []):
        if int(option.get("type", -1)) == 14:
            return [index]
    return None


def _stats_snapshot() -> dict[str, Any]:
    return {**_STATS, "mechanical_contexts": dict(_MECHANICAL_AGENT.statistics)}


# Kaggle selects the last callable created by main.py; keep agent last.
def agent(obs_dict: dict[str, Any]) -> list[int]:
    if obs_dict.get("select") is None:
        _STATS["deck_requests"] += 1
        _MECHANICAL_AGENT.reset()
        return _read_deck()
    loop_break = _loop_break_action(obs_dict)
    if loop_break is not None:
        _STATS["loop_break_actions"] += 1
        return loop_break
    try:
        action = _MECHANICAL_AGENT.act(obs_dict)
        if is_legal_action(obs_dict, action):
            _STATS["model_actions"] += 1
            return action
    except Exception:
        _STATS["inference_failures"] += 1
    _STATS["fallback_actions"] += 1
    return _fallback_action(obs_dict)
