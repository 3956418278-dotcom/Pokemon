#!/usr/bin/env python3
"""Kaggle cloud PPO training script for Pokemon TCG AI Battle.

This script keeps the submission loop simple but uses the competition engine for
self-play. It trains a small shared policy with PPO when PyTorch is available.
If training cannot run, it still exports a legal rule-score fallback submission.
"""

from __future__ import annotations

import json
import math
import os
import random
import shutil
import sys
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


KAGGLE_INPUT = Path("/kaggle/input")
KAGGLE_WORKING = Path("/kaggle/working")
SCRIPT_DIR = Path(__file__).resolve().parent
FEATURE_DIM = 48
DEFAULT_SEED = 20260710


RULE_WEIGHTS = {
    "attack": 2.0,
    "damage": 1.4,
    "draw": 1.0,
    "energy": 1.0,
    "evolve": 0.8,
    "retreat": -0.1,
    "end": -0.3,
    "random": 0.02,
}


@dataclass
class Transition:
    features: list[list[float]]
    action_index: int
    old_log_prob: float
    reward: float = 0.0


def kaggle_paths() -> tuple[Path, Path]:
    input_dir = KAGGLE_INPUT if KAGGLE_INPUT.exists() else Path.cwd()
    working_dir = KAGGLE_WORKING if KAGGLE_WORKING.exists() else Path.cwd() / "outputs"
    working_dir.mkdir(parents=True, exist_ok=True)
    return input_dir, working_dir


def discover_competition_root(input_dir: Path) -> Path:
    candidates = [
        input_dir / "pokemon-tcg-ai-battle",
        input_dir / "competitions" / "pokemon-tcg-ai-battle",
        input_dir / "input" / "competitions" / "pokemon-tcg-ai-battle",
        input_dir,
    ]
    for candidate in candidates:
        if (candidate / "sample_submission").exists() or (candidate / "EN_Card_Data.csv").exists():
            return candidate
    return input_dir


def local_competition_zip(input_dir: Path) -> Path | None:
    for root in (input_dir, Path.cwd()):
        path = root / "pokemon-tcg-ai-battle.zip"
        if path.exists():
            return path
    return None


def extract_directory(zip_path: Path, member_prefix: str, output_dir: Path) -> bool:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    copied = False
    with zipfile.ZipFile(zip_path) as zf:
        for member_name in zf.namelist():
            if not member_name.startswith(member_prefix) or member_name.endswith("/"):
                continue
            relative = Path(member_name).relative_to(member_prefix)
            target = output_dir / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member_name) as source, target.open("wb") as dest:
                shutil.copyfileobj(source, dest)
            copied = True
    return copied


def copy_cg_runtime(working_dir: Path, competition_root: Path, zip_path: Path | None) -> Path:
    output_dir = working_dir / "cg"
    source_dir = competition_root / "sample_submission" / "sample_submission" / "cg"
    if source_dir.exists():
        if output_dir.exists():
            shutil.rmtree(output_dir)
        shutil.copytree(source_dir, output_dir)
        print(f"Copied cg runtime from {source_dir}")
        return output_dir

    if zip_path is not None and extract_directory(
        zip_path, "sample_submission/sample_submission/cg", output_dir
    ):
        print(f"Extracted cg runtime from {zip_path}")
        return output_dir

    raise FileNotFoundError("Could not find sample_submission cg runtime")


def find_baseline_decks_json() -> Path:
    candidates = [
        SCRIPT_DIR / "baseline_decks.json",
        Path.cwd() / "baseline_decks.json",
        KAGGLE_INPUT / "baseline_decks.json",
    ]
    for path in candidates:
        if path.exists():
            return path

    if KAGGLE_INPUT.exists():
        matches = sorted(KAGGLE_INPUT.rglob("baseline_decks.json"))
        if matches:
            return matches[0]

    print("baseline_decks.json was not found. Current /kaggle/input files:")
    if KAGGLE_INPUT.exists():
        for path in sorted(KAGGLE_INPUT.rglob("*"))[:200]:
            if path.is_file():
                print(f"  - {path}")
    else:
        print("  - /kaggle/input does not exist")

    raise FileNotFoundError(
        "baseline_decks.json not found. Add a Kaggle Dataset containing "
        "baseline_decks.json to this kernel input. The script searches "
        "/kaggle/input/**/baseline_decks.json automatically."
    )


def load_baseline_decks() -> list[dict[str, Any]]:
    path = find_baseline_decks_json()
    print(f"Loading baseline decks from {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    decks = []
    for deck in data["decks"]:
        ids = deck.get("patched_deck_ids") or deck.get("deck_ids_if_all_present") or []
        if len(ids) == 60:
            decks.append(
                {
                    "name": deck["name"],
                    "deck_ids": [int(card_id) for card_id in ids],
                    "missing_total": deck.get("missing_total", 0),
                    "replaced_total": deck.get("replaced_total", 0),
                    "replacement_cards": deck.get("replacement_cards", []),
                }
            )
    if not decks:
        raise ValueError("No 60-card baseline decks found in baseline_decks.json")
    return decks


def write_deck_csv(path: Path, deck_ids: list[int]) -> None:
    path.write_text("\n".join(str(card_id) for card_id in deck_ids) + "\n", encoding="utf-8")


def option_value(option: Any, name: str, default: Any = None) -> Any:
    if isinstance(option, dict):
        return option.get(name, default)
    return getattr(option, name, default)


def option_text(option: Any) -> str:
    if option is None:
        return ""
    values = []
    for name in ("type", "number", "area", "index", "playerIndex", "attackId", "cardId", "inPlayArea"):
        value = option_value(option, name)
        if value is not None:
            values.append(f"{name}:{value}")
    return " ".join(values)


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if hasattr(value, "value"):
            return int(value.value)
        return int(value)
    except Exception:
        return default


def select_obj_from_observation(obs: Any) -> Any:
    if isinstance(obs, dict):
        return obs.get("select")
    return getattr(obs, "select", None)


def state_obj_from_observation(obs: Any) -> Any:
    if isinstance(obs, dict):
        return obs.get("current")
    return getattr(obs, "current", None)


def select_counts(select_obj: Any) -> tuple[int, int]:
    if select_obj is None:
        return 0, 0
    if isinstance(select_obj, dict):
        return int(select_obj.get("minCount", 0)), int(select_obj.get("maxCount", 0))
    return int(getattr(select_obj, "minCount", 0)), int(getattr(select_obj, "maxCount", 0))


def select_options(select_obj: Any) -> list[Any]:
    if select_obj is None:
        return []
    if isinstance(select_obj, dict):
        return list(select_obj.get("option", []))
    return list(getattr(select_obj, "option", []))


def make_candidates(select_obj: Any) -> list[list[int]]:
    options = select_options(select_obj)
    min_count, max_count = select_counts(select_obj)
    if not options and min_count == 0:
        return [[]]
    candidates: list[list[int]] = []
    if min_count == 0:
        candidates.append([])
    if max_count <= 1:
        candidates.extend([[i] for i in range(len(options))])
        return candidates
    if min_count <= 1:
        candidates.extend([[i] for i in range(len(options))])
    if min_count > 1:
        candidates.append(list(range(min_count)))
    return candidates or [list(range(min_count))]


def option_features(option: Any) -> list[float]:
    features = [0.0] * 24
    option_type = safe_int(option_value(option, "type"), 0)
    if 0 <= option_type < 17:
        features[option_type] = 1.0
    features[17] = safe_int(option_value(option, "number"), 0) / 10.0
    features[18] = safe_int(option_value(option, "index"), 0) / 10.0
    features[19] = safe_int(option_value(option, "inPlayIndex"), 0) / 10.0
    features[20] = safe_int(option_value(option, "attackId"), 0) / 500.0
    features[21] = safe_int(option_value(option, "cardId"), 0) / 1300.0
    text = option_text(option).lower()
    features[22] = 1.0 if "13" in text or "attack" in text else 0.0
    features[23] = 1.0 if "14" in text or "end" in text else 0.0
    return features


def state_features(obs: Any, select_obj: Any) -> list[float]:
    state = state_obj_from_observation(obs)
    features = [0.0] * 24
    if state is None:
        return features
    get = state.get if isinstance(state, dict) else lambda name, default=None: getattr(state, name, default)
    features[0] = safe_int(get("turn"), 0) / 100.0
    features[1] = safe_int(get("turnActionCount"), 0) / 30.0
    features[2] = float(bool(get("supporterPlayed", False)))
    features[3] = float(bool(get("stadiumPlayed", False)))
    features[4] = float(bool(get("energyAttached", False)))
    features[5] = float(bool(get("retreated", False)))
    features[6] = safe_int(get("yourIndex"), 0)
    features[7] = safe_int(get("firstPlayer"), -1) / 2.0
    players = get("players", []) or []
    your_index = safe_int(get("yourIndex"), 0)
    for offset, player_index in enumerate([your_index, 1 - your_index]):
        if player_index >= len(players):
            continue
        player = players[player_index]
        pget = player.get if isinstance(player, dict) else lambda name, default=None: getattr(player, name, default)
        base = 8 + offset * 6
        features[base] = len(pget("active", []) or []) / 1.0
        features[base + 1] = len(pget("bench", []) or []) / 5.0
        features[base + 2] = safe_int(pget("deckCount"), 0) / 60.0
        features[base + 3] = len(pget("prize", []) or []) / 6.0
        features[base + 4] = safe_int(pget("handCount"), 0) / 20.0
        features[base + 5] = len(pget("discard", []) or []) / 60.0
    stype = safe_int(select_obj.get("type") if isinstance(select_obj, dict) else getattr(select_obj, "type", 0), 0)
    sctx = safe_int(select_obj.get("context") if isinstance(select_obj, dict) else getattr(select_obj, "context", 0), 0)
    features[20] = stype / 12.0
    features[21] = sctx / 50.0
    features[22], features[23] = select_counts(select_obj)
    features[22] /= 5.0
    features[23] /= 5.0
    return features


def candidate_features(obs: Any, candidate: list[int]) -> list[float]:
    select_obj = select_obj_from_observation(obs)
    options = select_options(select_obj)
    sfeatures = state_features(obs, select_obj)
    if not candidate:
        ofeatures = [0.0] * 24
        ofeatures[23] = 1.0
    else:
        selected = [option_features(options[i]) for i in candidate if i < len(options)]
        ofeatures = [sum(values) / len(selected) for values in zip(*selected)]
    return (sfeatures + ofeatures)[:FEATURE_DIM]


def rule_score_candidate(obs: Any, candidate: list[int], rng: random.Random) -> float:
    select_obj = select_obj_from_observation(obs)
    options = select_options(select_obj)
    if not candidate:
        return -0.05
    score = 0.0
    for index in candidate:
        text = option_text(options[index]).lower() if index < len(options) else ""
        option_type = safe_int(option_value(options[index], "type"), -1) if index < len(options) else -1
        if option_type == 13:
            score += RULE_WEIGHTS["attack"]
        if option_type == 8:
            score += RULE_WEIGHTS["energy"]
        if option_type == 9:
            score += RULE_WEIGHTS["evolve"]
        if option_type == 12:
            score += RULE_WEIGHTS["retreat"]
        if option_type == 14:
            score += RULE_WEIGHTS["end"]
        if "draw" in text:
            score += RULE_WEIGHTS["draw"]
        if "damage" in text:
            score += RULE_WEIGHTS["damage"]
    return score + RULE_WEIGHTS["random"] * rng.random()


def rule_select(obs: Any, rng: random.Random) -> list[int]:
    select_obj = select_obj_from_observation(obs)
    candidates = make_candidates(select_obj)
    best = max(candidates, key=lambda candidate: rule_score_candidate(obs, candidate, rng))
    return best


def try_import_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        return torch, nn, F
    except Exception as exc:
        print(f"PyTorch unavailable; using fallback policy only: {exc}")
        return None, None, None


def build_policy(torch: Any, nn: Any):
    class PolicyNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.body = nn.Sequential(
                nn.Linear(FEATURE_DIM, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, 1),
            )

        def forward(self, x):
            return self.body(x).squeeze(-1)

    return PolicyNet()


def policy_select(policy: Any, torch: Any, obs: Any, rng: random.Random, train: bool) -> tuple[list[int], Transition | None]:
    select_obj = select_obj_from_observation(obs)
    candidates = make_candidates(select_obj)
    features = [candidate_features(obs, candidate) for candidate in candidates]
    if policy is None or torch is None:
        return rule_select(obs, rng), None

    with torch.no_grad():
        x = torch.tensor(features, dtype=torch.float32)
        logits = policy(x)
        probs = torch.softmax(logits, dim=0)
        if train:
            action_index = int(torch.multinomial(probs, 1).item())
        else:
            action_index = int(torch.argmax(probs).item())
        old_log_prob = float(torch.log(probs[action_index].clamp_min(1e-8)).item())
    return candidates[action_index], Transition(features, action_index, old_log_prob)


def result_reward(obs_class: Any, controlled_player: int) -> float:
    current = getattr(obs_class, "current", None)
    if current is None or getattr(current, "result", -1) == -1:
        return 0.0
    result = int(current.result)
    if result == controlled_player:
        return 1.0
    if result == 2:
        return 0.0
    return -1.0


def run_episode(
    deck: dict[str, Any],
    opponent: dict[str, Any],
    policy: Any,
    torch: Any,
    rng: random.Random,
    max_steps: int,
) -> tuple[list[Transition], float, int]:
    from cg.api import to_observation_class
    from cg.game import battle_finish, battle_select, battle_start

    obs, start_data = battle_start(deck["deck_ids"], opponent["deck_ids"])
    if obs is None:
        print(f"battle_start failed errorPlayer={start_data.errorPlayer} errorType={start_data.errorType}")
        return [], -1.0, 0

    transitions: list[Transition] = []
    reward = 0.0
    controlled_player = 0
    steps = 0
    try:
        for steps in range(1, max_steps + 1):
            obs_class = to_observation_class(obs)
            if obs_class.current is not None and obs_class.current.result != -1:
                reward = result_reward(obs_class, controlled_player)
                break
            if obs_class.select is None:
                break

            select_player = int(obs_class.current.yourIndex) if obs_class.current is not None else 0
            selected, transition = policy_select(policy, torch, obs_class, rng, train=select_player == controlled_player)
            if transition is not None and select_player == controlled_player:
                transitions.append(transition)
            obs = battle_select(selected)
        else:
            reward = -0.1
    except Exception as exc:
        print(f"Episode failed for {deck['name']} vs {opponent['name']}: {exc}")
        reward = -1.0
    finally:
        battle_finish()

    for transition in transitions:
        transition.reward = reward
    return transitions, reward, steps


def ppo_update(policy: Any, optimizer: Any, torch: Any, F: Any, transitions: list[Transition], epochs: int, clip_eps: float) -> float:
    if not transitions:
        return 0.0
    losses = []
    for _ in range(epochs):
        total_loss = None
        for transition in transitions:
            x = torch.tensor(transition.features, dtype=torch.float32)
            logits = policy(x)
            log_probs = F.log_softmax(logits, dim=0)
            new_log_prob = log_probs[transition.action_index]
            old_log_prob = torch.tensor(transition.old_log_prob, dtype=torch.float32)
            advantage = torch.tensor(float(transition.reward), dtype=torch.float32)
            ratio = torch.exp(new_log_prob - old_log_prob)
            clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantage
            policy_loss = -torch.min(ratio * advantage, clipped)
            entropy = -(torch.exp(log_probs) * log_probs).sum()
            loss = policy_loss - 0.01 * entropy
            total_loss = loss if total_loss is None else total_loss + loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        losses.append(float(total_loss.detach().item()) / len(transitions))
    return sum(losses) / len(losses)


def random_training_schedule(
    decks: list[dict[str, Any]],
    updates: int,
    episodes_per_deck: int,
    rng: random.Random,
) -> list[list[tuple[dict[str, Any], dict[str, Any]]]]:
    schedule = []
    for _ in range(updates):
        batch = []
        for deck in decks:
            opponents = [candidate for candidate in decks if candidate["name"] != deck["name"]]
            for _ in range(episodes_per_deck):
                batch.append((deck, rng.choice(opponents or decks)))
        schedule.append(batch)
    return schedule


def round_robin_training_schedule(
    decks: list[dict[str, Any]],
    episodes_per_matchup: int,
    batch_episodes: int,
    rng: random.Random,
) -> list[list[tuple[dict[str, Any], dict[str, Any]]]]:
    episodes = []
    for deck in decks:
        for opponent in decks:
            if opponent["name"] == deck["name"]:
                continue
            for _ in range(episodes_per_matchup):
                episodes.append((deck, opponent))
    rng.shuffle(episodes)
    return [episodes[i : i + batch_episodes] for i in range(0, len(episodes), batch_episodes)]


def train_ppo(decks: list[dict[str, Any]], working_dir: Path, seed: int) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    torch, nn, F = try_import_torch()
    summary = {
        "method": "ppo_self_play",
        "enabled": torch is not None,
        "episodes": [],
        "updates": 0,
        "note": "",
    }
    if torch is None:
        summary["note"] = "PyTorch unavailable; PPO skipped."
        return None, summary

    random.seed(seed)
    torch.manual_seed(seed)
    rng = random.Random(seed)
    policy = build_policy(torch, nn)
    optimizer = torch.optim.Adam(policy.parameters(), lr=float(os.environ.get("PTCG_PPO_LR", "0.0005")))
    schedule_mode = os.environ.get("PTCG_TRAINING_SCHEDULE", "round_robin")
    updates = int(os.environ.get("PTCG_PPO_UPDATES", "4"))
    episodes_per_deck = int(os.environ.get("PTCG_PPO_EPISODES_PER_DECK", "1"))
    episodes_per_matchup = int(os.environ.get("PTCG_EPISODES_PER_MATCHUP", "500"))
    batch_episodes = int(os.environ.get("PTCG_PPO_BATCH_EPISODES", "64"))
    max_steps = int(os.environ.get("PTCG_MAX_STEPS", "500"))
    ppo_epochs = int(os.environ.get("PTCG_PPO_EPOCHS", "2"))

    if schedule_mode == "round_robin":
        schedule = round_robin_training_schedule(decks, episodes_per_matchup, batch_episodes, rng)
    else:
        schedule = random_training_schedule(decks, updates, episodes_per_deck, rng)
    summary["schedule"] = {
        "mode": schedule_mode,
        "decks": len(decks),
        "directed_matchups": len(decks) * max(0, len(decks) - 1),
        "episodes_per_matchup": episodes_per_matchup if schedule_mode == "round_robin" else None,
        "batch_episodes": batch_episodes if schedule_mode == "round_robin" else None,
        "planned_episodes": sum(len(batch) for batch in schedule),
        "max_steps": max_steps,
    }

    for update, episode_batch in enumerate(schedule):
        batch: list[Transition] = []
        rewards = []
        for deck, opponent in episode_batch:
            transitions, reward, steps = run_episode(deck, opponent, policy, torch, rng, max_steps)
            batch.extend(transitions)
            rewards.append(reward)
            summary["episodes"].append(
                {
                    "update": update,
                    "deck": deck["name"],
                    "opponent": opponent["name"],
                    "reward": reward,
                    "steps": steps,
                    "transitions": len(transitions),
                }
            )
        loss = ppo_update(policy, optimizer, torch, F, batch, ppo_epochs, clip_eps=0.2)
        summary["updates"] += 1
        print(
            (
                f"PPO update {update + 1}/{len(schedule)}: episodes={len(episode_batch)} "
                f"transitions={len(batch)} mean_reward={sum(rewards)/max(1,len(rewards)):.3f} "
                f"loss={loss:.4f}"
            )
        )
        if (update + 1) % 10 == 0:
            model = export_policy(policy)
            (working_dir / "model.json").write_text(json.dumps(model, indent=2) + "\n", encoding="utf-8")
            (working_dir / "ppo_weights.json").write_text(json.dumps(model, indent=2) + "\n", encoding="utf-8")
            (working_dir / "training_summary.json").write_text(
                json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

    model = export_policy(policy)
    (working_dir / "policy_state.pt").write_bytes(b"")
    try:
        torch.save(policy.state_dict(), working_dir / "policy_state.pt")
    except Exception as exc:
        print(f"Could not save torch state_dict: {exc}")
    return model, summary


def export_policy(policy: Any) -> dict[str, Any]:
    layers = []
    for module in policy.body:
        if hasattr(module, "weight") and hasattr(module, "bias"):
            layers.append(
                {
                    "weight": module.weight.detach().cpu().tolist(),
                    "bias": module.bias.detach().cpu().tolist(),
                    "activation": "tanh",
                }
            )
    if layers:
        layers[-1]["activation"] = "linear"
    return {"feature_dim": FEATURE_DIM, "layers": layers}


def fallback_model() -> dict[str, Any]:
    return {"feature_dim": FEATURE_DIM, "layers": [], "fallback": "rule_score"}


def write_model_json(path: Path, model: dict[str, Any] | None) -> None:
    path.write_text(json.dumps(model or fallback_model(), indent=2) + "\n", encoding="utf-8")


def main_py_source() -> str:
    return r'''
import json
import math
import random
from pathlib import Path

try:
    from cg.api import to_observation_class
except Exception:
    to_observation_class = None

FEATURE_DIM = 48


def _read_json(name, default):
    for path in (Path(name), Path("/kaggle_simulations/agent") / name):
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)
    return default


MODEL = _read_json("model.json", {"layers": [], "fallback": "rule_score"})


def read_deck_csv():
    for path in (Path("deck.csv"), Path("/kaggle_simulations/agent/deck.csv")):
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                return [int(line.strip()) for line in f if line.strip()][:60]
    return []


def _safe_int(value, default=0):
    try:
        if hasattr(value, "value"):
            return int(value.value)
        return int(value)
    except Exception:
        return default


def _value(obj, name, default=None):
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _select_obj(observation):
    if to_observation_class is not None:
        try:
            return to_observation_class(observation).select
        except Exception:
            pass
    return observation.get("select") if isinstance(observation, dict) else None


def _state_obj(observation):
    if to_observation_class is not None:
        try:
            return to_observation_class(observation).current
        except Exception:
            pass
    return observation.get("current") if isinstance(observation, dict) else None


def _options(select_obj):
    if select_obj is None:
        return []
    return list(_value(select_obj, "option", []) or [])


def _counts(select_obj):
    if select_obj is None:
        return 0, 0
    return int(_value(select_obj, "minCount", 0) or 0), int(_value(select_obj, "maxCount", 0) or 0)


def _candidates(select_obj):
    options = _options(select_obj)
    min_count, max_count = _counts(select_obj)
    if not options and min_count == 0:
        return [[]]
    candidates = []
    if min_count == 0:
        candidates.append([])
    if max_count <= 1:
        candidates.extend([[i] for i in range(len(options))])
        return candidates
    if min_count <= 1:
        candidates.extend([[i] for i in range(len(options))])
    if min_count > 1:
        candidates.append(list(range(min_count)))
    return candidates or [list(range(min_count))]


def _option_features(option):
    features = [0.0] * 24
    option_type = _safe_int(_value(option, "type"), 0)
    if 0 <= option_type < 17:
        features[option_type] = 1.0
    features[17] = _safe_int(_value(option, "number"), 0) / 10.0
    features[18] = _safe_int(_value(option, "index"), 0) / 10.0
    features[19] = _safe_int(_value(option, "inPlayIndex"), 0) / 10.0
    features[20] = _safe_int(_value(option, "attackId"), 0) / 500.0
    features[21] = _safe_int(_value(option, "cardId"), 0) / 1300.0
    features[22] = 1.0 if option_type == 13 else 0.0
    features[23] = 1.0 if option_type == 14 else 0.0
    return features


def _state_features(observation, select_obj):
    state = _state_obj(observation)
    features = [0.0] * 24
    if state is None:
        return features
    features[0] = _safe_int(_value(state, "turn"), 0) / 100.0
    features[1] = _safe_int(_value(state, "turnActionCount"), 0) / 30.0
    features[2] = float(bool(_value(state, "supporterPlayed", False)))
    features[3] = float(bool(_value(state, "stadiumPlayed", False)))
    features[4] = float(bool(_value(state, "energyAttached", False)))
    features[5] = float(bool(_value(state, "retreated", False)))
    features[6] = _safe_int(_value(state, "yourIndex"), 0)
    features[7] = _safe_int(_value(state, "firstPlayer"), -1) / 2.0
    players = _value(state, "players", []) or []
    your_index = _safe_int(_value(state, "yourIndex"), 0)
    for offset, player_index in enumerate([your_index, 1 - your_index]):
        if player_index >= len(players):
            continue
        player = players[player_index]
        base = 8 + offset * 6
        features[base] = len(_value(player, "active", []) or [])
        features[base + 1] = len(_value(player, "bench", []) or []) / 5.0
        features[base + 2] = _safe_int(_value(player, "deckCount"), 0) / 60.0
        features[base + 3] = len(_value(player, "prize", []) or []) / 6.0
        features[base + 4] = _safe_int(_value(player, "handCount"), 0) / 20.0
        features[base + 5] = len(_value(player, "discard", []) or []) / 60.0
    features[20] = _safe_int(_value(select_obj, "type"), 0) / 12.0
    features[21] = _safe_int(_value(select_obj, "context"), 0) / 50.0
    min_count, max_count = _counts(select_obj)
    features[22] = min_count / 5.0
    features[23] = max_count / 5.0
    return features


def _candidate_features(observation, candidate):
    select_obj = _select_obj(observation)
    options = _options(select_obj)
    sfeatures = _state_features(observation, select_obj)
    if not candidate:
        ofeatures = [0.0] * 24
        ofeatures[23] = 1.0
    else:
        selected = [_option_features(options[i]) for i in candidate if i < len(options)]
        ofeatures = [sum(values) / len(selected) for values in zip(*selected)]
    return (sfeatures + ofeatures)[:FEATURE_DIM]


def _linear(x, weight, bias):
    return [sum(wi * xi for wi, xi in zip(row, x)) + b for row, b in zip(weight, bias)]


def _score(features):
    if not MODEL.get("layers"):
        return _rule_score(features)
    x = features
    for layer in MODEL["layers"]:
        x = _linear(x, layer["weight"], layer["bias"])
        if layer.get("activation") == "tanh":
            x = [math.tanh(v) for v in x]
    return x[0] if isinstance(x, list) else float(x)


def _rule_score(features):
    # Option one-hot begins at feature offset 24.
    option = features[24:48]
    return option[13] * 2.0 + option[8] + option[9] * 0.8 - option[14] * 0.2


def act(observation, configuration=None):
    if isinstance(observation, dict) and observation.get("select") is None:
        return read_deck_csv()
    select_obj = _select_obj(observation)
    if select_obj is None:
        return read_deck_csv()
    candidates = _candidates(select_obj)
    if not candidates:
        return []
    scored = [(_score(_candidate_features(observation, candidate)), candidate) for candidate in candidates]
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[0][1]


def agent(observation, configuration=None):
    return act(observation, configuration)
'''.lstrip()


def write_submission_tar(output_path: Path, files: list[Path], dirs: list[Path]) -> None:
    with tarfile.open(output_path, "w:gz") as tar:
        for path in files:
            tar.add(path, arcname=path.name)
        for directory in dirs:
            if directory.exists():
                tar.add(directory, arcname=directory.name)


def choose_submission_deck(decks: list[dict[str, Any]], summary: dict[str, Any]) -> dict[str, Any]:
    rewards: dict[str, list[float]] = {deck["name"]: [] for deck in decks}
    for episode in summary.get("episodes", []):
        rewards.setdefault(episode["deck"], []).append(float(episode["reward"]))
    scored = []
    for deck in decks:
        values = rewards.get(deck["name"], [])
        mean_reward = sum(values) / len(values) if values else -999.0
        scored.append((mean_reward, -deck["missing_total"], deck["name"], deck))
    scored.sort(reverse=True)
    return scored[0][3]


def run() -> None:
    input_dir, working_dir = kaggle_paths()
    competition_root = discover_competition_root(input_dir)
    zip_path = local_competition_zip(input_dir)
    print(f"Input directory: {input_dir}")
    print(f"Competition root: {competition_root}")
    print(f"Working directory: {working_dir}")

    cg_dir = copy_cg_runtime(working_dir, competition_root, zip_path)
    sys.path.insert(0, str(working_dir))

    decks = load_baseline_decks()
    print(f"Loaded {len(decks)} patched baseline decks")

    seed = int(os.environ.get("PTCG_SEED", str(DEFAULT_SEED)))
    model, summary = train_ppo(decks, working_dir, seed)
    model_path = working_dir / "model.json"
    ppo_weights_path = working_dir / "ppo_weights.json"
    summary_path = working_dir / "training_summary.json"

    write_model_json(model_path, model)
    write_model_json(ppo_weights_path, model)
    summary["baseline_decks"] = [
        {
            "name": deck["name"],
            "missing_total": deck["missing_total"],
            "replaced_total": deck["replaced_total"],
            "replacement_cards": deck["replacement_cards"],
        }
        for deck in decks
    ]
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("Wrote training artifacts:")
    for path in (model_path, ppo_weights_path, summary_path):
        print(f"  - {path} ({path.stat().st_size} bytes)")

    if os.environ.get("PTCG_BUILD_SUBMISSION", "0") == "1":
        from submit_agent import build_submission

        selected_deck = choose_submission_deck(decks, summary)
        build_submission(working_dir, cg_dir, selected_deck, model_path)


if __name__ == "__main__":
    run()
