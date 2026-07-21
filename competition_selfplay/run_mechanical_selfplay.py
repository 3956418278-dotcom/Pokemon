from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from competition_selfplay.mechanical_agent import MechanicalAgent, is_legal_action


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPLAY_ROOT = ROOT / "outputs/competition_selfplay/mechanical_v2_selfplay"


def _rewards(result: int) -> list[int]:
    if result == 0:
        return [1, -1]
    if result == 1:
        return [-1, 1]
    return [0, 0]


def _build_replay(
    *,
    episode: int,
    deck: list[int],
    initial_observation: dict[str, Any],
    decisions: list[dict[str, Any]],
    visualize: list[dict[str, Any]],
    result: int,
    completed: bool,
    error: str | None,
) -> dict[str, Any]:
    """Build a Kaggle-shaped local replay with the engine's real visual frames."""

    rewards = _rewards(result)
    statuses = ["DONE", "DONE"] if completed else ["ERROR", "ERROR"]
    steps: list[list[dict[str, Any]]] = [
        [
            {
                "action": deck,
                "info": {"policy": "mechanical_v2", "episode": episode},
                "observation": initial_observation,
                "reward": 0,
                "status": "ACTIVE",
                "visualize": visualize,
            },
            {
                "action": deck,
                "info": {"policy": "mechanical_v2", "episode": episode},
                "observation": initial_observation,
                "reward": 0,
                "status": "INACTIVE",
            },
        ]
    ]
    for decision_index, decision in enumerate(decisions):
        actor = int(decision["player"])
        final_step = completed and decision_index == len(decisions) - 1
        entries: list[dict[str, Any]] = []
        for player in (0, 1):
            is_actor = player == actor
            entries.append(
                {
                    "action": decision["action"] if is_actor else [],
                    "info": {"decision": decision_index},
                    "observation": decision["observation"] if is_actor else {},
                    "reward": rewards[player] if final_step else 0,
                    "status": "DONE" if final_step else ("ACTIVE" if is_actor else "INACTIVE"),
                }
            )
        steps.append(entries)
    return {
        "configuration": {"source": "local_cg_runtime"},
        "description": "Mechanical v2 mirror self-play replay for human audit.",
        "id": f"mechanical-v2-selfplay-{episode:04d}",
        "info": {
            "Agents": [{"Name": "Mechanical v2 P0"}, {"Name": "Mechanical v2 P1"}],
            "EpisodeId": episode,
            "TeamNames": ["Mechanical v2 P0", "Mechanical v2 P1"],
        },
        "name": "cabt-local",
        "rewards": rewards,
        "schema_version": "mechanical_selfplay_replay_v1",
        "statuses": statuses,
        "steps": steps,
        "title": "Mechanical v2 self-play",
        "version": "1.0.0",
        "local": {
            "completed": completed,
            "decision_count": len(decisions),
            "result": result,
            "error": error,
            "visual_frame_count": len(visualize),
        },
    }


def run(episodes: int, max_decisions: int, output_dir: Path) -> dict[str, object]:
    sys.path.insert(0, str(ROOT / "kaggle/datasets/cg_runtime"))
    from cg.game import battle_finish, battle_select, battle_start, visualize_data

    decks = json.loads((ROOT / "decks/baseline_decks.json").read_text(encoding="utf-8"))["decks"]
    deck = [int(value) for value in decks[6]["patched_deck_ids"]]
    agents = [MechanicalAgent(), MechanicalAgent()]
    completed = invalid = crashes = timeouts = 0
    results = [0, 0, 0]
    errors: list[str] = []
    fallback_contexts: dict[str, int] = {}
    lengths: list[int] = []
    replay_paths: list[str] = []
    replay_errors: list[str] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    for episode in range(episodes):
        started = False
        decisions = 0
        decision_records: list[dict[str, Any]] = []
        initial_observation: dict[str, Any] = {}
        episode_result = 2
        episode_completed = False
        episode_error: str | None = None
        for agent in agents:
            agent.reset()
        try:
            observation, start_data = battle_start(deck, deck)
            started = observation is not None
            if observation is None:
                raise RuntimeError(
                    f"battle_start failed: player={start_data.errorPlayer}, type={start_data.errorType}"
                )
            initial_observation = observation
            while decisions < max_decisions:
                select = observation.get("select")
                current = observation.get("current") or {}
                result = int(current.get("result", -1))
                if select is None or (not (select.get("option") or []) and result >= 0):
                    result = 2 if result < 0 else result
                    results[result] += 1
                    completed += 1
                    lengths.append(decisions)
                    episode_result = result
                    episode_completed = True
                    break
                player = int(current.get("yourIndex", 0))
                action = agents[player].act(observation)
                if not is_legal_action(observation, action):
                    invalid += 1
                    raise RuntimeError(
                        f"invalid action at episode={episode}, decision={decisions}: {action}"
                    )
                decision_records.append(
                    {"player": player, "observation": observation, "action": action}
                )
                observation = battle_select(action)
                decisions += 1
            else:
                timeouts += 1
                lengths.append(decisions)
        except Exception as exc:
            crashes += 1
            episode_error = f"{type(exc).__name__}: {exc}"
            errors.append(f"episode {episode}: {episode_error}")
        finally:
            for agent in agents:
                for key, value in agent.statistics.items():
                    fallback_contexts[key] = fallback_contexts.get(key, 0) + value
            if started:
                try:
                    visual_frames = json.loads(visualize_data())
                    replay = _build_replay(
                        episode=episode,
                        deck=deck,
                        initial_observation=initial_observation,
                        decisions=decision_records,
                        visualize=visual_frames,
                        result=episode_result,
                        completed=episode_completed,
                        error=episode_error,
                    )
                    replay_path = output_dir / f"episode-{episode:04d}-replay.json"
                    replay_path.write_text(
                        json.dumps(replay, ensure_ascii=False), encoding="utf-8"
                    )
                    replay_paths.append(str(replay_path.resolve()))
                except Exception as exc:
                    message = f"episode {episode} replay export: {type(exc).__name__}: {exc}"
                    replay_errors.append(message)
                    errors.append(message)
                finally:
                    battle_finish()
    result_payload: dict[str, object] = {
        "episodes": episodes,
        "completed": completed,
        "invalid": invalid,
        "crashes": crashes,
        "timeouts": timeouts,
        "results_by_player": results,
        "decision_counts": lengths,
        "fallback_contexts": fallback_contexts,
        "replay_output_dir": str(output_dir.resolve()),
        "replay_count": len(replay_paths),
        "replay_paths": replay_paths,
        "replay_errors": replay_errors,
        "errors": errors,
        "passed": (
            completed == episodes
            and invalid == 0
            and crashes == 0
            and timeouts == 0
            and len(replay_paths) == episodes
            and not replay_errors
        ),
    }
    (output_dir.parent / "manifest.json").write_text(
        json.dumps(result_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result_payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--max-decisions", type=int, default=2000)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Replay JSON directory; defaults to a new timestamped run under outputs/.",
    )
    args = parser.parse_args()
    output_dir = args.output_dir or (
        DEFAULT_REPLAY_ROOT / datetime.now().strftime("run-%Y%m%d-%H%M%S") / "replays"
    )
    result = run(args.episodes, args.max_decisions, output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
