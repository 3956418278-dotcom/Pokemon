from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.game_memory import GameMemoryState
from data.observation_parser import parse_observation


def pct(values: list[int], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * q)))
    return float(ordered[index])


def load_static_detail_counts(path: Path) -> dict[str, dict[str, int]]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    counts = {}
    for card in data.get("cards", []):
        details = card.get("details", [])
        by_type = Counter(item.get("detail_type") for item in details)
        counts[str(card.get("card_id"))] = {
            "details": len(details),
            "attacks": int(by_type.get("attack", 0)),
            "abilities": int(by_type.get("ability", 0)),
            "effects": int(by_type.get("special_effect", 0)),
        }
    return counts


def iter_agent_observations(replay: dict[str, Any]):
    for step_index, step in enumerate(replay.get("steps", []) or []):
        for agent_index, agent_step in enumerate(step or []):
            obs = (agent_step or {}).get("observation")
            if obs is not None:
                yield step_index, agent_index, agent_step, obs


def summarize(replay_path: Path, detail_metadata_path: Path) -> dict[str, Any]:
    replay = json.loads(replay_path.read_text(encoding="utf-8"))
    detail_counts = load_static_detail_counts(detail_metadata_path)
    memory_by_agent: dict[int, GameMemoryState] = defaultdict(GameMemoryState)

    counters: dict[str, Counter] = defaultdict(Counter)
    values: dict[str, list[int]] = defaultdict(list)
    observed_card_ids: Counter[str] = Counter()
    visible_card_ids: Counter[str] = Counter()
    unknown_card_instances = 0
    no_current = 0
    no_select = 0
    parser_errors: list[dict[str, Any]] = []

    for step_index, agent_index, agent_step, obs in iter_agent_observations(replay):
        if obs.get("current") is None:
            no_current += 1
        if obs.get("select") is None:
            no_select += 1
        try:
            parsed = parse_observation(obs)
            memory = memory_by_agent[agent_index]
            memory.update_from_parsed(parsed)
        except Exception as exc:
            parser_errors.append(
                {
                    "step": step_index,
                    "agent": agent_index,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue

        counters["agent"].update([agent_index])
        counters["select_type"].update([parsed.global_snapshot.select_type])
        counters["select_context"].update([parsed.global_snapshot.select_context])
        counters["result"].update([parsed.global_snapshot.result])
        values["instances"].append(len(parsed.card_instances))
        values["events"].append(len(parsed.events))
        values["options"].append(len(parsed.select_options))
        values["token_estimate"].append(
            1
            + len(parsed.card_instances)
            + 1
            + 1
            + 2
            + min(len(memory.recent_events), memory.max_recent_events)
        )
        values["recent_events"].append(len(memory.recent_events))
        values["visible_instances"].append(sum(1 for item in parsed.card_instances if item.is_visible))
        values["hidden_instances"].append(sum(1 for item in parsed.card_instances if not item.is_visible))
        values["pokemon_instances"].append(sum(1 for item in parsed.card_instances if item.is_pokemon))
        values["attached_instances"].append(sum(1 for item in parsed.card_instances if item.attached_to_serial is not None))
        for item in parsed.card_instances:
            counters["zone"].update([item.zone])
            counters["area"].update([item.area])
            if item.card_id is None:
                unknown_card_instances += 1
            else:
                observed_card_ids.update([str(item.card_id)])
                if item.is_visible:
                    visible_card_ids.update([str(item.card_id)])
                counts = detail_counts.get(str(item.card_id), {})
                values["static_detail_count"].append(int(counts.get("details", 0)))
                values["static_attack_count"].append(int(counts.get("attacks", 0)))
                values["static_ability_count"].append(int(counts.get("abilities", 0)))
                values["static_effect_count"].append(int(counts.get("effects", 0)))
        for event in parsed.events:
            counters["event_type"].update([event.event_type])
            counters["event_reverse"].update([int(event.is_reverse)])
            counters["event_has_card_id"].update([int(event.card_id is not None)])
        for option in parsed.select_options:
            counters["option_type"].update([option.get("type", -1)])
            if option.get("attackId") is not None:
                counters["option_has_attack_id"].update([1])
            if option.get("cardId") is not None:
                counters["option_has_card_id"].update([1])

    def value_summary(name: str) -> dict[str, float]:
        xs = values.get(name, [])
        return {
            "count": len(xs),
            "mean": float(mean(xs)) if xs else 0.0,
            "p50": pct(xs, 0.50),
            "p90": pct(xs, 0.90),
            "p99": pct(xs, 0.99),
            "max": float(max(xs)) if xs else 0.0,
        }

    observations = list(iter_agent_observations(replay))
    report = {
        "replay_path": str(replay_path),
        "episode_id": replay.get("info", {}).get("EpisodeId"),
        "teams": replay.get("info", {}).get("TeamNames"),
        "rewards": replay.get("rewards"),
        "steps": len(replay.get("steps", []) or []),
        "agent_observations": len(observations),
        "no_current": no_current,
        "no_select": no_select,
        "parser_errors": parser_errors[:20],
        "unknown_card_instances": unknown_card_instances,
        "unique_observed_card_ids": len(observed_card_ids),
        "unique_visible_card_ids": len(visible_card_ids),
        "value_summaries": {name: value_summary(name) for name in sorted(values)},
        "counters": {name: counter.most_common(30) for name, counter in sorted(counters.items())},
        "top_visible_card_ids": visible_card_ids.most_common(50),
        "top_observed_card_ids": observed_card_ids.most_common(50),
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("replay", type=Path)
    parser.add_argument("--detail-metadata", type=Path, default=Path("outputs/card_pretrain/artifacts/card_detail_metadata.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("data_from_submission/replay_audit"))
    args = parser.parse_args()

    report = summarize(args.replay, args.detail_metadata)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / (args.replay.stem + "-feature-audit.json")
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary = report["value_summaries"]
    md_path = args.output_dir / (args.replay.stem + "-feature-decision.md")
    md = f"""# Replay Feature Decision

Replay: `{report["replay_path"]}`

Episode: `{report["episode_id"]}`

Teams: `{report["teams"]}`

## Data Read

- Steps: `{report["steps"]}`
- Agent observations: `{report["agent_observations"]}`
- Missing current/select observations: `{report["no_current"]}` / `{report["no_select"]}`
- Parser errors: `{len(report["parser_errors"])}`
- Unique visible card IDs: `{report["unique_visible_card_ids"]}`

## Dynamic Feature Implications

- Card instance tokens must be variable length. This replay has instance count mean `{summary["instances"]["mean"]:.2f}`, p90 `{summary["instances"]["p90"]:.0f}`, p99 `{summary["instances"]["p99"]:.0f}`, max `{summary["instances"]["max"]:.0f}`.
- A fixed board-token budget below 128 is risky. Token estimate max is `{summary["token_estimate"]["max"]:.0f}` in this single replay; use padding/mask and start with a budget around 128-160 if batching needs a hard cap.
- Recent event tokens should stay capped, but raw logs can burst. This replay has event count max `{summary["events"]["max"]:.0f}` while recent event memory caps at `{summary["recent_events"]["max"]:.0f}`. Match/state features therefore include current log count, reverse log count, and public-card log count.
- Hidden zone handling is required. Hidden instances mean `{summary["hidden_instances"]["mean"]:.2f}`, max `{summary["hidden_instances"]["max"]:.0f}`; opponent hand should remain count-only.
- Static detail aggregation is justified. Observed visible/static card instances have detail count p99 `{summary["static_detail_count"]["p99"]:.0f}`, max `{summary["static_detail_count"]["max"]:.0f}` in this replay; summary-only would discard useful attack/ability/effect separation.
- Action/options must be variable length. Options max `{summary["options"]["max"]:.0f}`; old fixed candidate features are not enough for the next policy head.

## Current Feature Decision

- Keep: static `card_summary` plus explicit `detail_tokens` attention aggregation.
- Keep: per-card dynamic board state and appearance/memory groups.
- Keep: state, decision, match, ledger, and recent-event board tokens.
- Add now: current observation log summary features in match token.
- Do not assume fixed game length. Train samples should be decision-point rows with per-sample masks; full games can be grouped only by metadata.

## Needs More Data

This is one public replay. Before freezing dimensions, run the same audit over many replays or self-play games and inspect p99/max for instance count, option count, raw logs, and token estimate.
"""
    md_path.write_text(md, encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False)[:6000])
    print("wrote", output_path)
    print("wrote", md_path)


if __name__ == "__main__":
    main()
