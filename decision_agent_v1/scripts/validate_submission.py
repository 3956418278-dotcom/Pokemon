from __future__ import annotations

import argparse
import json
import os
import resource
import statistics
import sys
import tarfile
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARCHIVE = ROOT / "outputs/decision_agent_v1/submission_v1/submission.tar.gz"
DEFAULT_OUTPUT = ROOT / "outputs/decision_agent_v1/submission_v1/submission_validation.json"


class Struct(dict[str, Any]):
    """Minimal copy of kaggle_environments.utils.Struct."""

    def __init__(self, **entries: Any) -> None:
        super().__init__(entries)
        self.__dict__.update(entries)


def _structify(value: Any) -> Any:
    if isinstance(value, list):
        return [_structify(item) for item in value]
    if isinstance(value, dict):
        return Struct(**{key: _structify(item) for key, item in value.items()})
    return value


def _baseline_action(observation: dict[str, Any]) -> list[int]:
    select = observation.get("select") or {}
    options = select.get("option") or []
    if not options:
        return []
    minimum = max(0, int(select.get("minCount", 0)))
    maximum = max(minimum, int(select.get("maxCount", minimum)))
    count = 1 if maximum <= 1 else minimum
    return list(range(min(max(count, minimum), maximum, len(options))))


def _import_submission(agent_root: Path) -> Any:
    sys.path.insert(0, str(agent_root))
    # This validator itself is imported from the repository's decision_agent_v1
    # package. Purge those names so the submission proves it can import only its
    # own packaged copy, as it would under Kaggle.
    for name in tuple(sys.modules):
        if name == "data" or name.startswith("data.") or name == "cg" or name.startswith("cg."):
            del sys.modules[name]
        elif name == "decision_agent_v1" or name.startswith("decision_agent_v1."):
            del sys.modules[name]
    # Kaggle executes main.py as raw source and does not define __file__.
    # Mirror that behavior rather than importing the file conventionally.
    source = (agent_root / "main.py").read_text(encoding="utf-8")
    namespace: dict[str, Any] = {"__name__": "kaggle_v1_submission_main"}
    previous_cwd = Path.cwd()
    try:
        os.chdir(agent_root.parent.parent.parent)
        exec(compile(source, str(agent_root / "main.py"), "exec"), namespace)
    finally:
        os.chdir(previous_cwd)
    callables = [value for value in namespace.values() if callable(value)]
    if not callables or getattr(callables[-1], "__name__", None) != "agent":
        raise RuntimeError(
            "Kaggle raw loader would not select agent as the last callable: "
            f"{getattr(callables[-1], '__name__', None) if callables else None}"
        )
    return SimpleNamespace(**namespace, _kaggle_selected_callable=callables[-1])


def validate(args: argparse.Namespace) -> dict[str, Any]:
    archive = args.archive.resolve()
    with tarfile.open(archive, "r:gz") as handle:
        names = set(handle.getnames())
        if not {"main.py", "deck.csv"}.issubset(names):
            raise RuntimeError("submission archive lacks top-level main.py or deck.csv")
        if any(Path(name).is_absolute() or ".." in Path(name).parts for name in names):
            raise RuntimeError("submission archive contains an unsafe path")
        with tempfile.TemporaryDirectory(prefix="v1_submission_", dir=args.temp_root) as temp:
            agent_root = Path(temp) / "empty" / "kaggle_simulations" / "agent"
            agent_root.mkdir(parents=True)
            handle.extractall(agent_root)

            module = _import_submission(agent_root)
            if module._kaggle_selected_callable is not module.agent:
                raise RuntimeError("Kaggle callable selection does not resolve to agent")
            opponent_module = (
                _import_submission(agent_root)
                if args.agent_mode in {"self", "fallback", "load-failure"}
                else None
            )
            if args.agent_mode == "fallback":
                for current_module in (module, opponent_module):
                    current_module.agent.__globals__["_LOAD_ATTEMPTED"] = True
                    current_module.agent.__globals__["_MODEL_AGENT"] = None
                    current_module.agent.__globals__["_SIMULATOR_ADAPTER"] = None
            deck = module._kaggle_selected_callable(_structify({"select": None}))
            if not isinstance(deck, list) or len(deck) != 60 or not all(type(i) is int for i in deck):
                raise RuntimeError("submission did not return a valid 60-row deck")
            if args.agent_mode == "load-failure":
                (agent_root / "model.pt").unlink()

            from cg.game import battle_finish, battle_select, battle_start

            completed = invalid = crashes = timeouts = wins = losses = draws = 0
            decisions_per_game: list[int] = []
            model_latencies_ms: list[float] = []
            errors: list[dict[str, Any]] = []
            for episode in range(args.episodes):
                v1_player = episode % 2
                module.agent(_structify({"select": None}))
                if opponent_module is not None:
                    opponent_module.agent(_structify({"select": None}))
                observation = None
                started = False
                decisions = 0
                try:
                    observation, start_data = battle_start(deck, deck)
                    started = observation is not None
                    if observation is None:
                        raise RuntimeError(
                            f"battle_start failed: player={start_data.errorPlayer}, type={start_data.errorType}"
                        )
                    while decisions < args.max_decisions:
                        select = observation.get("select")
                        options = (select or {}).get("option") or []
                        result = int((observation.get("current") or {}).get("result", -1))
                        if select is None or (not options and result >= 0):
                            result = 2 if result < 0 else result
                            wins += int(result == v1_player)
                            losses += int(result in (0, 1) and result != v1_player)
                            draws += int(result == 2)
                            completed += 1
                            decisions_per_game.append(decisions)
                            break
                        if not options:
                            raise RuntimeError(f"non-terminal state has no legal options: result={result}")
                        acting_player = int((observation.get("current") or {}).get("yourIndex", -1))
                        use_submission = (
                            args.agent_mode in {"self", "fallback", "load-failure"}
                            or acting_player == v1_player
                        )
                        if use_submission:
                            started_at = time.perf_counter()
                            acting_module = (
                                module
                                if acting_player == v1_player or opponent_module is None
                                else opponent_module
                            )
                            action = acting_module.agent(_structify(observation))
                            model_latencies_ms.append((time.perf_counter() - started_at) * 1000.0)
                        else:
                            action = _baseline_action(observation)
                        minimum = int(select.get("minCount", 0))
                        maximum = int(select.get("maxCount", minimum))
                        if (
                            not isinstance(action, list)
                            or not all(type(index) is int and 0 <= index < len(options) for index in action)
                            or not minimum <= len(action) <= maximum
                        ):
                            invalid += 1
                            raise RuntimeError(f"invalid action shape: {action!r}")
                        try:
                            observation = battle_select(action)
                        except IndexError:
                            invalid += 1
                            raise
                        decisions += 1
                    else:
                        timeouts += 1
                        decisions_per_game.append(decisions)
                except Exception as exc:
                    crashes += 1
                    errors.append({
                        "episode": episode,
                        "decision": decisions,
                        "error": f"{type(exc).__name__}: {exc}",
                    })
                finally:
                    if started:
                        try:
                            battle_finish()
                        except Exception as exc:
                            crashes += 1
                            errors.append({"episode": episode, "stage": "finish", "error": str(exc)})

            stats = module._stats_snapshot()
            opponent_stats = (
                opponent_module._stats_snapshot() if opponent_module is not None else None
            )
            load_state_ok = (
                stats["load_failures"] == 1
                and opponent_stats is not None
                and opponent_stats["load_failures"] == 1
                if args.agent_mode == "load-failure"
                else stats["load_failures"] == 0
                and (opponent_stats is None or opponent_stats["load_failures"] == 0)
            )
            fallback_probe = module._fallback_action({
                "select": {"type": 0, "minCount": 1, "maxCount": 1, "option": [{"type": 14}]}
            })
            payload = {
                "archive": str(archive),
                "archive_bytes": archive.stat().st_size,
                "top_level_main": "main.py" in names,
                "top_level_deck": "deck.csv" in names,
                "simulated_agent_root": str(agent_root),
                "agent_mode": args.agent_mode,
                "requested_episodes": args.episodes,
                "completed_episodes": completed,
                "wins": wins,
                "losses": losses,
                "draws": draws,
                "invalid_action_count": invalid,
                "crash_count": crashes,
                "timeout_count": timeouts,
                "decisions_per_game": decisions_per_game,
                "cold_action_ms": model_latencies_ms[0] if model_latencies_ms else None,
                "mean_action_ms": statistics.fmean(model_latencies_ms) if model_latencies_ms else None,
                "p95_action_ms": (
                    sorted(model_latencies_ms)[max(0, int(len(model_latencies_ms) * 0.95) - 1)]
                    if model_latencies_ms else None
                ),
                "max_action_ms": max(model_latencies_ms) if model_latencies_ms else None,
                "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                "agent_stats": stats,
                "opponent_agent_stats": opponent_stats,
                "fallback_probe": fallback_probe,
                "passed": (
                    completed == args.episodes
                    and invalid == 0
                    and crashes == 0
                    and timeouts == 0
                    and load_state_ok
                    and fallback_probe == [0]
                ),
                "errors": errors,
            }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if not payload["passed"]:
        raise RuntimeError(f"submission validation failed; see {args.output}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the standalone V1 Kaggle submission")
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--max-decisions", type=int, default=2000)
    parser.add_argument(
        "--agent-mode",
        choices=("baseline", "self", "fallback", "load-failure"),
        default="baseline",
    )
    parser.add_argument("--temp-root", type=Path, default=ROOT / ".tmp")
    validate(parser.parse_args())


if __name__ == "__main__":
    main()
