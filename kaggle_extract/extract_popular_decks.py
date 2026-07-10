#!/usr/bin/env python3
"""Extract popular public replay decklists for Pokemon TCG AI Battle.

Decks are grouped by Pokemon + Energy only. Trainer differences do not create a
new group. Each popular group records every distinct full 60-card variant found
inside that group, while the most common variant is kept as the representative
test deck.
"""

from __future__ import annotations

import json
import math
import os
import time
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


COMPETITION = "pokemon-tcg-ai-battle"
KAGGLE_INPUT = Path("/kaggle/input")
KAGGLE_WORKING = Path("/kaggle/working")
SCRIPT_DIR = Path(__file__).resolve().parent

RECENT_SUBMISSIONS_TO_USE = int(os.environ.get("PTCG_RECENT_SUBMISSIONS_TO_USE", "8"))
SUBMISSION_PAGE_SIZE = int(os.environ.get("PTCG_SUBMISSION_PAGE_SIZE", "50"))
MAX_REPLAYS_TO_DOWNLOAD = int(os.environ.get("PTCG_MAX_REPLAYS_TO_DOWNLOAD", "200"))
MAX_DOWNLOAD_RETRIES = int(os.environ.get("PTCG_MAX_DOWNLOAD_RETRIES", "5"))
DOWNLOAD_SLEEP_SECONDS = float(os.environ.get("PTCG_DOWNLOAD_SLEEP_SECONDS", "0.5"))
MIN_GROUP_GAMES = int(os.environ.get("PTCG_MIN_POPULAR_DECK_GAMES", "2"))
MAX_POPULAR_DECKS = int(os.environ.get("PTCG_MAX_POPULAR_DECKS", "24"))
TEAM_NAME = os.environ.get("PTCG_TEAM_NAME", "")
SUBMISSION_IDS = [
    int(value)
    for value in os.environ.get("PTCG_SUBMISSION_IDS", "").replace(",", " ").split()
    if value.strip()
]


def kaggle_paths() -> tuple[Path, Path, Path]:
    working_dir = KAGGLE_WORKING if KAGGLE_WORKING.exists() else Path.cwd() / "popular_deck_outputs"
    temp_dir = Path("/kaggle/temp") if Path("/kaggle/temp").exists() else Path("/tmp")
    if not KAGGLE_WORKING.exists():
        temp_dir = working_dir / "tmp"
    replay_dir = temp_dir / "ptcg_popular_deck_replays"
    working_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)
    replay_dir.mkdir(parents=True, exist_ok=True)
    return working_dir, temp_dir, replay_dir


def normalize_value(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return [normalize_value(v) for v in value]
    if isinstance(value, dict):
        return {k: normalize_value(v) for k, v in value.items()}
    if isinstance(value, datetime):
        return value.isoformat()
    if hasattr(value, "name") and hasattr(value, "value"):
        return value.name
    return value


def as_plain_dict(obj: Any) -> dict[str, Any]:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return {k: normalize_value(v) for k, v in obj.items()}
    if hasattr(obj, "to_dict"):
        return {k: normalize_value(v) for k, v in obj.to_dict().items()}
    raw = getattr(obj, "__dict__", {})
    return {k.lstrip("_"): normalize_value(v) for k, v in raw.items() if not k.startswith("__")}


def first_present_column(df: pd.DataFrame, names: list[str]) -> str | None:
    for name in names:
        if name in df.columns:
            return name
    return None


def find_card_data_csv() -> Path | None:
    candidates = [
        KAGGLE_INPUT / "pokemon-tcg-ai-battle" / "EN_Card_Data.csv",
        KAGGLE_INPUT / "competitions" / "pokemon-tcg-ai-battle" / "EN_Card_Data.csv",
        KAGGLE_INPUT / "datasets" / "competitions" / "pokemon-tcg-ai-battle" / "EN_Card_Data.csv",
        SCRIPT_DIR / "EN_Card_Data.csv",
        SCRIPT_DIR.parent / "EN_Card_Data.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    if KAGGLE_INPUT.exists():
        matches = sorted(KAGGLE_INPUT.rglob("EN_Card_Data.csv"))
        if matches:
            return matches[0]
    zip_path = SCRIPT_DIR.parent / "pokemon-tcg-ai-battle.zip"
    if zip_path.exists():
        output = SCRIPT_DIR / "EN_Card_Data.csv"
        with zipfile.ZipFile(zip_path) as zf:
            with zf.open("EN_Card_Data.csv") as source:
                output.write_bytes(source.read())
        return output
    return None


def load_card_table() -> pd.DataFrame:
    card_path = find_card_data_csv()
    if card_path is None:
        print("EN_Card_Data.csv not found; signatures will treat all IDs as non-trainer unknown cards.")
        return pd.DataFrame(columns=["card_id", "card_name", "card_kind"])
    raw = pd.read_csv(card_path, encoding="utf-8-sig")
    table = pd.DataFrame(
        {
            "card_id": raw.iloc[:, 0].astype(int),
            "card_name": raw.iloc[:, 1].astype(str),
            "card_kind": raw.iloc[:, 4].astype(str),
        }
    )
    print(f"loaded card table: {card_path} rows={len(table)}")
    return table


CARDS_DF = load_card_table()
CARD_NAME = dict(zip(CARDS_DF["card_id"], CARDS_DF["card_name"]))
CARD_KIND = dict(zip(CARDS_DF["card_id"], CARDS_DF["card_kind"]))


def card_name(card_id: int) -> str:
    return CARD_NAME.get(int(card_id), str(card_id))


def card_kind(card_id: int) -> str:
    return CARD_KIND.get(int(card_id), "")


def is_pokemon_or_energy(card_id: int) -> bool:
    kind = card_kind(card_id)
    return "Pok" in kind or "Energy" in kind


def deck_signature(deck: list[int]) -> str:
    counts = Counter(card_id for card_id in deck if is_pokemon_or_energy(card_id))
    return "|".join(f"{card_id}:{counts[card_id]}" for card_id in sorted(counts))


def deck_counts_rows(counts: Counter[int]) -> list[dict[str, Any]]:
    return [
        {
            "card_id": int(card_id),
            "card_name": card_name(card_id),
            "card_kind": card_kind(card_id),
            "count": int(count),
        }
        for card_id, count in sorted(counts.items(), key=lambda item: (card_kind(item[0]), card_name(item[0]), item[0]))
    ]


def decklist_text(deck: list[int], only_pokemon_energy: bool = False, limit: int = 80) -> str:
    ids = [card_id for card_id in deck if is_pokemon_or_energy(card_id)] if only_pokemon_energy else list(deck)
    counts = Counter(ids)
    parts = []
    for card_id, count in counts.most_common(limit):
        parts.append(f"{card_name(card_id)} x{count}")
    return "; ".join(parts)


def archetype_name(deck: list[int]) -> str:
    pokemon_counts = Counter(card_id for card_id in deck if "Pok" in card_kind(card_id))
    if not pokemon_counts:
        return "Unknown"
    names = [card_name(card_id).replace(" ex", "") for card_id, _ in pokemon_counts.most_common(3)]
    return " / ".join(names)


def winner_from_rewards(team_names: list[str], rewards: list[float | int | None]) -> str:
    numeric = [-math.inf if reward is None else float(reward) for reward in rewards]
    if not numeric or len(set(numeric)) == 1:
        return "draw"
    return team_names[max(range(len(numeric)), key=lambda i: numeric[i])]


def extract_decks(steps: list[Any]) -> list[list[int]]:
    if len(steps) > 1:
        decks = []
        for seat in range(2):
            action = steps[1][seat].get("action", [])
            if isinstance(action, list) and len(action) == 60 and all(isinstance(x, int) for x in action):
                decks.append([int(x) for x in action])
        if len(decks) == 2:
            return decks

    visualize = steps[0][0].get("visualize", []) if steps else []
    if visualize and isinstance(visualize[0].get("action"), list):
        decks = visualize[0]["action"]
        if len(decks) == 2:
            return [[int(x) for x in deck] for deck in decks]
    return [[], []]


def parse_replay(path: Path, episode_meta: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    replay = json.loads(path.read_text(encoding="utf-8"))
    info = replay.get("info", {})
    team_names = list(info.get("TeamNames") or ["seat0", "seat1"])
    rewards = list(replay.get("rewards") or [None, None])
    steps = replay.get("steps") or []
    decks = extract_decks(steps)
    winner = winner_from_rewards(team_names, rewards)
    episode_id = int(info.get("EpisodeId") or path.stem.replace("episode-", "").replace("-replay", ""))

    rows = []
    for seat, deck in enumerate(decks):
        if len(deck) != 60:
            continue
        rows.append(
            {
                "episode_id": episode_id,
                "seat": seat,
                "team": team_names[seat] if seat < len(team_names) else "",
                "winner": winner,
                "won": winner == (team_names[seat] if seat < len(team_names) else ""),
                "reward": rewards[seat] if seat < len(rewards) else None,
                "source_submission_id": episode_meta.get("source_submission_id") if episode_meta else None,
                "deck": deck,
                "signature": deck_signature(deck),
                "archetype": archetype_name(deck),
            }
        )
    return rows


def replay_path_for_episode(replay_dir: Path, episode_id: int) -> Path:
    return replay_dir / f"episode-{episode_id}-replay.json"


def download_replay_json(api: Any, episode_id: int, destination: Path) -> None:
    from kaggle.api.kaggle_api_extended import ApiGetEpisodeReplayRequest

    last_error: Exception | None = None
    for attempt in range(MAX_DOWNLOAD_RETRIES):
        request = ApiGetEpisodeReplayRequest()
        request.episode_id = int(episode_id)
        try:
            with api.build_kaggle_client() as kaggle:
                response = kaggle.competitions.competition_api_client.get_episode_replay(request)
                response.raise_for_status()
                destination.write_bytes(response.content)
            if DOWNLOAD_SLEEP_SECONDS:
                time.sleep(DOWNLOAD_SLEEP_SECONDS)
            return
        except Exception as exc:
            last_error = exc
            status_code = getattr(getattr(exc, "response", None), "status_code", None)
            if status_code == 429:
                wait_seconds = min(60, 5 * (2**attempt))
            else:
                wait_seconds = min(20, 2 * (attempt + 1))
            if attempt < MAX_DOWNLOAD_RETRIES - 1:
                print(f"episode {episode_id}: {type(exc).__name__}, retrying in {wait_seconds}s")
                time.sleep(wait_seconds)
                continue
            break
    raise RuntimeError(f"failed to download episode {episode_id}: {last_error}")


def select_submission_ids(api: Any) -> tuple[list[int], pd.DataFrame, str]:
    team_name = TEAM_NAME
    if SUBMISSION_IDS:
        return SUBMISSION_IDS, pd.DataFrame({"ref": SUBMISSION_IDS}), team_name

    submissions = api.competition_submissions(COMPETITION, page_size=SUBMISSION_PAGE_SIZE) or []
    submissions_df = pd.DataFrame([as_plain_dict(submission) for submission in submissions])
    if submissions_df.empty:
        raise RuntimeError("No submissions were returned. Set PTCG_SUBMISSION_IDS manually.")

    team_col = first_present_column(submissions_df, ["team_name", "teamName"])
    if not team_name and team_col and submissions_df[team_col].notna().any():
        team_name = str(submissions_df[team_col].dropna().iloc[0])
        print("inferred PTCG_TEAM_NAME:", team_name)

    if "status" in submissions_df.columns:
        submissions_df = submissions_df[
            submissions_df["status"].astype(str).str.contains("COMPLETE", case=False, na=False)
        ]
    selected = submissions_df.head(RECENT_SUBMISSIONS_TO_USE)["ref"].astype(int).tolist()
    return selected, submissions_df, team_name


def collect_episode_rows(api: Any, submission_ids: list[int]) -> pd.DataFrame:
    rows = []
    seen_episode_ids: set[int] = set()
    for submission_id in submission_ids:
        episodes = api.competition_list_episodes(int(submission_id)) or []
        print(f"submission {submission_id}: {len(episodes)} episodes")
        for episode in episodes:
            row = as_plain_dict(episode)
            row["source_submission_id"] = int(submission_id)
            episode_id = int(row["id"])
            if episode_id in seen_episode_ids:
                continue
            seen_episode_ids.add(episode_id)
            agents = row.get("agents") or []
            for seat in range(2):
                agent = agents[seat] if seat < len(agents) else {}
                row[f"team_{seat}"] = agent.get("teamName") or agent.get("team_name")
                row[f"submission_{seat}"] = agent.get("submissionId") or agent.get("submission_id")
                row[f"agent_reward_{seat}"] = agent.get("reward")
            rows.append(row)

    episodes_df = pd.DataFrame(rows)
    if episodes_df.empty:
        raise RuntimeError("No public episodes were found for the selected submissions.")
    if "type" in episodes_df.columns:
        episodes_df = episodes_df[episodes_df["type"].astype(str).str.contains("PUBLIC", case=False, na=False)]
    if "state" in episodes_df.columns:
        episodes_df = episodes_df[episodes_df["state"].astype(str).str.contains("COMPLETED|COMPLETE", case=False, na=False)]
    return episodes_df.sort_values("id", ascending=False).head(MAX_REPLAYS_TO_DOWNLOAD).reset_index(drop=True)


def build_popular_decks(deck_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_signature: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in deck_rows:
        by_signature[row["signature"]].append(row)

    groups = []
    total_deck_observations = len(deck_rows)
    for signature, rows in by_signature.items():
        if len(rows) < MIN_GROUP_GAMES:
            continue

        full_deck_counts = Counter(tuple(row["deck"]) for row in rows)
        representative_deck_tuple, representative_count = full_deck_counts.most_common(1)[0]
        representative_deck = [int(card_id) for card_id in representative_deck_tuple]
        pe_counts = Counter(card_id for card_id in representative_deck if is_pokemon_or_energy(card_id))
        trainer_variant_count = len(full_deck_counts)
        wins = sum(1 for row in rows if row["won"])
        reward_values = [float(row["reward"]) for row in rows if row.get("reward") is not None]
        variants = []
        for variant_rank, (deck_tuple, variant_games) in enumerate(full_deck_counts.most_common(), start=1):
            variant_deck = [int(card_id) for card_id in deck_tuple]
            variants.append(
                {
                    "variant_rank": variant_rank,
                    "games": int(variant_games),
                    "share_within_group": round(variant_games / len(rows), 6),
                    "deck_ids": variant_deck,
                    "deck_counts": deck_counts_rows(Counter(variant_deck)),
                    "deck_text": decklist_text(variant_deck, only_pokemon_energy=False),
                }
            )

        groups.append(
            {
                "name": archetype_name(representative_deck),
                "signature": signature,
                "games": len(rows),
                "share": round(len(rows) / total_deck_observations, 6) if total_deck_observations else 0.0,
                "wins": wins,
                "win_rate": round(wins / len(rows), 4) if rows else 0.0,
                "mean_reward": round(sum(reward_values) / len(reward_values), 4) if reward_values else None,
                "representative_count": int(representative_count),
                "trainer_variant_count": int(trainer_variant_count),
                "deck_ids": representative_deck,
                "patched_deck_ids": representative_deck,
                "pokemon_energy_counts": deck_counts_rows(pe_counts),
                "pokemon_energy_signature_text": decklist_text(representative_deck, only_pokemon_energy=True),
                "representative_deck_text": decklist_text(representative_deck, only_pokemon_energy=False),
                "variants": variants,
            }
        )

    groups.sort(key=lambda item: (item["games"], item["representative_count"], item["win_rate"]), reverse=True)
    for index, group in enumerate(groups[:MAX_POPULAR_DECKS], start=1):
        group["rank"] = index
        group["name"] = f"Popular {index:02d}: {group['name']}"
    return groups[:MAX_POPULAR_DECKS]


def write_outputs(
    output_dir: Path,
    selected_submission_ids: list[int],
    team_name: str,
    episodes_df: pd.DataFrame,
    deck_rows: list[dict[str, Any]],
    popular_decks: list[dict[str, Any]],
    download_errors: list[dict[str, Any]],
    parse_errors: list[dict[str, Any]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = [
        {
            "rank": deck["rank"],
            "name": deck["name"],
            "games": deck["games"],
            "share": deck["share"],
            "win_rate": deck["win_rate"],
            "representative_count": deck["representative_count"],
            "trainer_variant_count": deck["trainer_variant_count"],
            "pokemon_energy": deck["pokemon_energy_signature_text"],
            "representative_deck": deck["representative_deck_text"],
        }
        for deck in popular_decks
    ]
    pd.DataFrame(summary_rows).to_csv(output_dir / "popular_deck_summary.csv", index=False, encoding="utf-8-sig")

    output_json = {
        "source": {
            "competition": COMPETITION,
            "selected_submission_ids": selected_submission_ids,
            "team_name": team_name,
            "episodes": int(len(episodes_df)),
            "parsed_decks": int(len(deck_rows)),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        },
        "signature_rule": "Decks are grouped by exact Pokemon + Energy card ID counts; Trainer cards are ignored for grouping.",
        "min_group_games": MIN_GROUP_GAMES,
        "max_popular_decks": MAX_POPULAR_DECKS,
        "decks": popular_decks,
    }
    (output_dir / "popular_test_decks.json").write_text(
        json.dumps(output_json, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    markdown_lines = [
        "# Popular test decks",
        "",
        "Grouped by exact Pokemon + Energy counts. Trainer differences are ignored for popularity grouping.",
        "",
    ]
    for deck in popular_decks:
        markdown_lines.extend(
            [
                f"## {deck['rank']}. {deck['name']}",
                "",
                f"- Games: {deck['games']}",
                f"- Share: {deck['share']}",
                f"- Win rate: {deck['win_rate']}",
                f"- Trainer variants: {deck['trainer_variant_count']}",
                f"- Representative full-deck count: {deck['representative_count']}",
                "",
                "Pokemon + Energy signature:",
                "",
                deck["pokemon_energy_signature_text"] or "(empty)",
                "",
                "Representative 60-card deck:",
                "",
                deck["representative_deck_text"] or "(empty)",
                "",
                "Full 60-card variants are recorded in `popular_test_decks.json`.",
                "",
            ]
        )
    (output_dir / "popular_test_decks.md").write_text("\n".join(markdown_lines), encoding="utf-8")

    if popular_decks:
        (output_dir / "popular_test_deck.csv").write_text(
            "\n".join(str(card_id) for card_id in popular_decks[0]["deck_ids"]) + "\n",
            encoding="utf-8",
        )

    print(
        json.dumps(
            {
                "popular_decks": int(len(popular_decks)),
                "parsed_decks": int(len(deck_rows)),
                "download_errors": int(len(download_errors)),
                "parse_errors": int(len(parse_errors)),
                "outputs": [
                    str(output_dir / "popular_test_decks.json"),
                    str(output_dir / "popular_test_decks.md"),
                    str(output_dir / "popular_test_deck.csv"),
                    str(output_dir / "popular_deck_summary.csv"),
                ],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


def run() -> None:
    from kaggle.api.kaggle_api_extended import KaggleApi

    working_dir, _temp_dir, replay_dir = kaggle_paths()
    output_dir = working_dir / "popular_deck_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    print("working dir:", working_dir.resolve())
    print("replay dir:", replay_dir.resolve())
    print("output dir:", output_dir.resolve())

    api = KaggleApi()
    api.authenticate()

    selected_submission_ids, submissions_df, team_name = select_submission_ids(api)
    print("using submission ids:", selected_submission_ids)

    episodes_df = collect_episode_rows(api, selected_submission_ids)
    if episodes_df.empty:
        raise RuntimeError("No completed public episodes remained after filtering.")
    print(f"selected {len(episodes_df)} public completed episodes")

    episode_meta_by_id = {int(row["id"]): row for row in episodes_df.to_dict("records")}
    downloaded_paths: list[Path] = []
    download_errors: list[dict[str, Any]] = []
    for episode_id in episodes_df["id"].astype(int).tolist():
        path = replay_path_for_episode(replay_dir, int(episode_id))
        if path.exists() and path.stat().st_size > 1000:
            downloaded_paths.append(path)
            continue
        try:
            download_replay_json(api, int(episode_id), path)
            downloaded_paths.append(path)
        except Exception as exc:
            download_errors.append({"episode_id": int(episode_id), "error": f"{type(exc).__name__}: {exc}"})
            print(f"episode {episode_id}: skipped after download error: {exc}")

    deck_rows: list[dict[str, Any]] = []
    parse_errors: list[dict[str, Any]] = []
    for path in downloaded_paths:
        try:
            episode_id = int(path.stem.replace("episode-", "").replace("-replay", ""))
            deck_rows.extend(parse_replay(path, episode_meta_by_id.get(episode_id)))
        except Exception as exc:
            parse_errors.append({"file": str(path), "error": f"{type(exc).__name__}: {exc}"})

    if not deck_rows:
        raise RuntimeError("No 60-card decklists were parsed from downloaded replays.")

    popular_decks = build_popular_decks(deck_rows)
    if not popular_decks:
        print(f"No deck groups reached PTCG_MIN_POPULAR_DECK_GAMES={MIN_GROUP_GAMES}; writing empty outputs.")

    write_outputs(
        output_dir,
        selected_submission_ids,
        team_name,
        episodes_df,
        deck_rows,
        popular_decks,
        download_errors,
        parse_errors,
    )


if __name__ == "__main__":
    run()
