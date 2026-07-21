from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import tarfile
import tempfile
import time
from pathlib import Path
from typing import Any, Callable


class Struct(dict[str, Any]):
    """Equivalent input shape to kaggle_environments.utils.Struct."""

    def __init__(self, **entries: Any) -> None:
        super().__init__(entries)
        self.__dict__.update(entries)


def structify(value: Any) -> Any:
    if isinstance(value, list):
        return [structify(item) for item in value]
    if isinstance(value, dict):
        return Struct(**{key: structify(item) for key, item in value.items()})
    return value


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def kaggle_raw_load(main_path: Path) -> tuple[Callable[..., Any], dict[str, Any]]:
    """Mirror kaggle_environments.agent.get_last_callable."""

    raw = main_path.read_text(encoding="utf-8")
    code_object = compile(raw, str(main_path), "exec")
    env: dict[str, Any] = {}
    sys.path.append(str(main_path.parent))
    try:
        exec(code_object, env)
    finally:
        sys.path.pop()
    callables = [value for value in env.values() if callable(value)]
    if not callables:
        raise RuntimeError("main.py did not create a callable")
    selected = callables[-1]
    if getattr(selected, "__name__", None) != "agent":
        raise RuntimeError(
            "Kaggle would select the wrong callable: "
            f"{getattr(selected, '__name__', type(selected).__name__)}"
        )
    return selected, env


def safe_extract(archive: Path, destination: Path) -> set[str]:
    with tarfile.open(archive, "r:gz") as handle:
        members = handle.getmembers()
        names = {member.name for member in members}
        for member in members:
            path = Path(member.name)
            if path.is_absolute() or ".." in path.parts:
                raise RuntimeError(f"unsafe archive path: {member.name}")
            if not (member.isfile() or member.isdir()):
                raise RuntimeError(f"unsupported archive member type: {member.name}")
        handle.extractall(destination)
    return names


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--forbidden-root", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-decisions", type=int, default=2000)
    args = parser.parse_args()

    archive = args.archive.resolve()
    output = args.output.resolve()
    forbidden_root = args.forbidden_root.resolve()
    checker_path = Path(__file__).resolve()
    initial_sys_path = tuple(sys.path)
    forbidden_sys_path = [
        entry
        for entry in initial_sys_path
        if entry and is_relative_to(Path(entry).resolve(), forbidden_root)
    ]
    if forbidden_sys_path:
        raise RuntimeError(f"isolated interpreter retained repository paths: {forbidden_sys_path}")

    opened_paths: set[str] = set()

    def audit_hook(event: str, event_args: tuple[Any, ...]) -> None:
        if event != "open" or not event_args:
            return
        value = event_args[0]
        if isinstance(value, (str, bytes, os.PathLike)):
            try:
                opened_paths.add(str(Path(value).resolve()))
            except (OSError, TypeError, ValueError):
                pass

    with tempfile.TemporaryDirectory(prefix="pokemon_v1_isolated_") as temp:
        temp_root = Path(temp).resolve()
        agent_root = temp_root / "kaggle_simulations" / "agent"
        unrelated_cwd = temp_root / "unrelated_working_directory"
        agent_root.mkdir(parents=True)
        unrelated_cwd.mkdir()
        names = safe_extract(archive, agent_root)
        if not {"main.py", "deck.csv"}.issubset(names):
            raise RuntimeError("archive lacks top-level main.py or deck.csv")
        forbidden_literal = str(forbidden_root).encode()
        embedded_paths = []
        for path in agent_root.rglob("*"):
            if path.is_file() and path.stat().st_size <= 2 * 1024 * 1024:
                if forbidden_literal in path.read_bytes():
                    embedded_paths.append(str(path.relative_to(agent_root)))
        if embedded_paths:
            raise RuntimeError(f"submission embeds repository paths: {embedded_paths}")

        os.chdir(unrelated_cwd)
        sys.addaudithook(audit_hook)
        agent0, env0 = kaggle_raw_load(agent_root / "main.py")
        agent1, env1 = kaggle_raw_load(agent_root / "main.py")
        deck0 = agent0(structify({"select": None}))
        deck1 = agent1(structify({"select": None}))
        if len(deck0) != 60 or len(deck1) != 60:
            raise RuntimeError("agent did not return a 60-card deck")

        sys.path.append(str(agent_root))
        try:
            from cg.game import battle_finish, battle_select, battle_start
        finally:
            sys.path.pop()

        completed = invalid = crashes = timeouts = 0
        results = [0, 0, 0]
        decision_counts: list[int] = []
        latencies_ms: list[float] = []
        errors: list[dict[str, Any]] = []
        for episode in range(args.episodes):
            agents = (agent0, agent1) if episode % 2 == 0 else (agent1, agent0)
            agents[0](structify({"select": None}))
            agents[1](structify({"select": None}))
            started = False
            decisions = 0
            try:
                observation, start_data = battle_start(deck0, deck1)
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
                        completed += 1
                        results[2 if result < 0 else result] += 1
                        decision_counts.append(decisions)
                        break
                    if not options:
                        raise RuntimeError(f"non-terminal state has no options: result={result}")
                    acting_player = int((observation.get("current") or {}).get("yourIndex", -1))
                    started_at = time.perf_counter()
                    action = agents[acting_player](structify(observation))
                    latencies_ms.append((time.perf_counter() - started_at) * 1000.0)
                    minimum = int(select.get("minCount", 0))
                    maximum = int(select.get("maxCount", minimum))
                    if (
                        not isinstance(action, list)
                        or not all(type(index) is int and 0 <= index < len(options) for index in action)
                        or not minimum <= len(action) <= maximum
                    ):
                        invalid += 1
                        raise RuntimeError(f"invalid action: {action!r}")
                    try:
                        observation = battle_select(action)
                    except IndexError:
                        invalid += 1
                        raise
                    decisions += 1
                else:
                    timeouts += 1
                    decision_counts.append(decisions)
            except Exception as exc:
                crashes += 1
                errors.append(
                    {"episode": episode, "decision": decisions, "error": f"{type(exc).__name__}: {exc}"}
                )
            finally:
                if started:
                    battle_finish()

        project_module_origins: dict[str, str] = {}
        bad_module_origins: dict[str, str] = {}
        for name, module in sorted(sys.modules.items()):
            if not (
                name == "data"
                or name.startswith("data.")
                or name == "decision_agent_v1"
                or name.startswith("decision_agent_v1.")
                or name == "cg"
                or name.startswith("cg.")
            ):
                continue
            origin = getattr(module, "__file__", None)
            if origin is None:
                continue
            resolved = Path(origin).resolve()
            project_module_origins[name] = str(resolved.relative_to(agent_root)) if is_relative_to(resolved, agent_root) else str(resolved)
            if not is_relative_to(resolved, agent_root):
                bad_module_origins[name] = str(resolved)

        forbidden_file_accesses = sorted(
            path
            for path in opened_paths
            if is_relative_to(Path(path), forbidden_root) and Path(path) != checker_path
        )
        stats0 = env0["_stats_snapshot"]()
        stats1 = env1["_stats_snapshot"]()
        payload = {
            "archive": str(archive),
            "archive_sha256": sha256(archive),
            "archive_bytes": archive.stat().st_size,
            "isolated_flag": bool(sys.flags.isolated),
            "initial_sys_path": initial_sys_path,
            "forbidden_sys_path": forbidden_sys_path,
            "temporary_agent_root": str(agent_root),
            "working_directory": str(unrelated_cwd),
            "kaggle_selected_callable": [agent0.__name__, agent1.__name__],
            "project_module_origins": project_module_origins,
            "bad_module_origins": bad_module_origins,
            "forbidden_file_accesses": forbidden_file_accesses,
            "episodes": args.episodes,
            "completed": completed,
            "results_by_player": results,
            "invalid": invalid,
            "crashes": crashes,
            "timeouts": timeouts,
            "decision_counts": decision_counts,
            "cold_action_ms": latencies_ms[0] if latencies_ms else None,
            "mean_action_ms": statistics.fmean(latencies_ms) if latencies_ms else None,
            "p95_action_ms": (
                sorted(latencies_ms)[max(0, int(len(latencies_ms) * 0.95) - 1)]
                if latencies_ms
                else None
            ),
            "agent_stats": [stats0, stats1],
            "passed": (
                bool(sys.flags.isolated)
                and not forbidden_sys_path
                and not bad_module_origins
                and not forbidden_file_accesses
                and completed == args.episodes
                and invalid == 0
                and crashes == 0
                and timeouts == 0
                and stats0["load_failures"] == 0
                and stats1["load_failures"] == 0
                and stats0["inference_failures"] == 0
                and stats1["inference_failures"] == 0
                and stats0["fallback_actions"] == 0
                and stats1["fallback_actions"] == 0
            ),
            "errors": errors,
        }
        os.chdir("/")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if not payload["passed"]:
        raise RuntimeError(f"isolated submission check failed; see {output}")


if __name__ == "__main__":
    main()
