from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Protocol

from .replay_dataset import ReplayDecisionDataset


ROOT = Path(__file__).resolve().parents[1]


COMPETITION = "pokemon-tcg-ai-battle"


@dataclass
class ReplayEpisodeRef:
    episode_id: int
    source_submission_id: int | None = None
    state: str | None = None
    episode_type: str | None = None
    created_at: str | None = None
    split: str = "train"
    source_index_file: str | None = None
    team_names: list[str] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class DailyReplayDatasetRef:
    date: str
    daily_dataset_slug: str
    daily_dataset_url: str | None = None
    episode_count: int | None = None
    total_bytes: int | None = None
    split: str = "train"
    mount_path: Path | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class OnlineReplayImportConfig:
    competition: str = COMPETITION
    submission_ids: list[int] = field(default_factory=list)
    recent_submissions_to_use: int = 4
    submission_page_size: int = 20
    max_replays: int = 40
    download_sleep_seconds: float = 0.5
    max_download_retries: int = 5
    cache_replays: bool = True
    include_private_episodes: bool = False
    episodes_index_dir: Path | None = None
    daily_replay_dirs: list[Path] = field(default_factory=list)
    daily_dataset_mount_root: Path = Path("/kaggle/input")
    reserve_recent_days: int = 0
    import_split: str = "train"
    output_dir: Path = ROOT / "outputs/replay_extract/online_replays"


class ReplayApiClient(Protocol):
    def list_recent_submission_ids(self, competition: str, page_size: int, limit: int) -> list[int]:
        ...

    def list_episodes(self, competition: str, submission_id: int) -> list[ReplayEpisodeRef]:
        ...

    def download_replay(self, competition: str, episode_id: int) -> bytes:
        ...


def _normalize_value(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return [_normalize_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _normalize_value(item) for key, item in value.items()}
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            return value
    if hasattr(value, "name") and hasattr(value, "value"):
        return value.name
    return value


def _plain_dict(obj: Any) -> dict[str, Any]:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return {key: _normalize_value(value) for key, value in obj.items()}
    if hasattr(obj, "to_dict"):
        return {key: _normalize_value(value) for key, value in obj.to_dict().items()}
    raw = getattr(obj, "__dict__", {})
    return {key.lstrip("_"): _normalize_value(value) for key, value in raw.items() if not key.startswith("__")}


def _first_present(row: dict[str, Any], names: list[str]) -> Any:
    lower = {str(key).lower(): value for key, value in row.items()}
    for name in names:
        if name in row and row[name] not in (None, ""):
            return row[name]
        value = lower.get(name.lower())
        if value not in (None, ""):
            return value
    return None


def _parse_timestamp(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    text = str(value).strip()
    if not text:
        return None
    if text.isdigit():
        number = int(text)
        if number > 10_000_000_000:
            number = number // 1000
        return datetime.fromtimestamp(number, tz=timezone.utc)
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        try:
            return datetime.strptime(text[:10], "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except ValueError:
            return None


def _iter_table_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".csv":
        import csv

        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return [dict(row) for row in payload if isinstance(row, dict)]
        if isinstance(payload, dict):
            for key in ["episodes", "data", "rows"]:
                rows = payload.get(key)
                if isinstance(rows, list):
                    return [dict(row) for row in rows if isinstance(row, dict)]
        return []
    if path.suffix.lower() == ".parquet":
        try:
            import pandas as pd
        except ImportError:
            return []
        return pd.read_parquet(path).to_dict("records")
    return []


def iter_episode_index_files(index_dir: Path) -> list[Path]:
    suffixes = {".csv", ".jsonl", ".json", ".parquet"}
    return sorted(path for path in index_dir.rglob("*") if path.is_file() and path.suffix.lower() in suffixes)


def load_episode_refs_from_index(index_dir: Path) -> list[ReplayEpisodeRef]:
    refs: list[ReplayEpisodeRef] = []
    seen: set[int] = set()
    for path in iter_episode_index_files(index_dir):
        for row in _iter_table_rows(path):
            episode_id = _first_present(row, ["episode_id", "EpisodeId", "episodeId", "id"])
            if episode_id is None:
                continue
            try:
                episode_id_int = int(episode_id)
            except (TypeError, ValueError):
                continue
            if episode_id_int in seen:
                continue
            seen.add(episode_id_int)
            submission_id = _first_present(row, ["submission_id", "submissionId", "source_submission_id", "SubmissionId"])
            created_at = _first_present(row, ["created_at", "createTime", "create_time", "date", "updated_at", "endTime"])
            agents = _first_present(row, ["team_names", "TeamNames", "teams", "agents"]) or []
            team_names = agents if isinstance(agents, list) else [str(agents)] if agents else []
            refs.append(
                ReplayEpisodeRef(
                    episode_id=episode_id_int,
                    source_submission_id=int(submission_id) if submission_id not in (None, "") else None,
                    state=str(_first_present(row, ["state", "status"])) if _first_present(row, ["state", "status"]) is not None else None,
                    episode_type=str(_first_present(row, ["type", "episode_type", "episodeType"]))
                    if _first_present(row, ["type", "episode_type", "episodeType"]) is not None
                    else None,
                    created_at=str(created_at) if created_at not in (None, "") else None,
                    source_index_file=str(path),
                    team_names=[str(name) for name in team_names],
                    raw=row,
                )
            )
    return refs


def assign_time_splits(refs: list[ReplayEpisodeRef], reserve_recent_days: int) -> list[ReplayEpisodeRef]:
    if reserve_recent_days <= 0:
        return refs
    timestamps = [_parse_timestamp(ref.created_at) for ref in refs]
    known = [ts for ts in timestamps if ts is not None]
    if not known:
        return refs
    cutoff = max(known) - timedelta(days=reserve_recent_days)
    for ref, ts in zip(refs, timestamps):
        if ts is not None and ts > cutoff:
            ref.split = "reserved"
        else:
            ref.split = "train"
    return refs


def load_daily_dataset_refs_from_manifest(index_dir: Path) -> list[DailyReplayDatasetRef]:
    manifest_path = index_dir / "manifest.csv"
    if not manifest_path.exists():
        return []
    refs: list[DailyReplayDatasetRef] = []
    for row in _iter_table_rows(manifest_path):
        date = _first_present(row, ["date"])
        slug = _first_present(row, ["daily_dataset_slug", "slug"])
        if not date or not slug:
            continue
        episode_count = _first_present(row, ["episode_count"])
        total_bytes = _first_present(row, ["total_bytes"])
        refs.append(
            DailyReplayDatasetRef(
                date=str(date),
                daily_dataset_slug=str(slug),
                daily_dataset_url=str(_first_present(row, ["daily_dataset_url", "url"]) or ""),
                episode_count=int(episode_count) if episode_count not in (None, "") else None,
                total_bytes=int(total_bytes) if total_bytes not in (None, "") else None,
                raw=row,
            )
        )
    return refs


def assign_daily_dataset_splits(refs: list[DailyReplayDatasetRef], reserve_recent_days: int) -> list[DailyReplayDatasetRef]:
    if reserve_recent_days <= 0:
        return refs
    timestamps = [_parse_timestamp(ref.date) for ref in refs]
    known = [ts for ts in timestamps if ts is not None]
    if not known:
        return refs
    cutoff = max(known) - timedelta(days=reserve_recent_days)
    for ref, ts in zip(refs, timestamps):
        if ts is not None and ts > cutoff:
            ref.split = "reserved"
        else:
            ref.split = "train"
    return refs


def select_daily_dataset_refs(
    index_dir: Path,
    mount_root: Path = Path("/kaggle/input"),
    reserve_recent_days: int = 0,
    import_split: str = "train",
    max_days: int | None = None,
) -> list[DailyReplayDatasetRef]:
    refs = assign_daily_dataset_splits(load_daily_dataset_refs_from_manifest(index_dir), reserve_recent_days)
    refs = [ref for ref in refs if ref.split == import_split]
    if import_split == "train":
        refs = sorted(refs, key=lambda ref: ref.date)
    else:
        refs = sorted(refs, key=lambda ref: ref.date)
    if max_days is not None:
        refs = refs[:max_days]
    for ref in refs:
        ref.mount_path = mount_root / ref.daily_dataset_slug
    return refs


def import_mounted_daily_replay_dataset(
    daily_replay_dirs: list[Path],
    output_dir: Path,
    include_no_select: bool = False,
    controlled_agents: set[int] | None = None,
    max_samples: int | None = None,
) -> tuple[ReplayDecisionDataset, dict[str, Any]]:
    replay_paths, metadata = prepare_mounted_daily_replays(daily_replay_dirs, output_dir)
    dataset = ReplayDecisionDataset(
        replay_paths,
        include_no_select=include_no_select,
        controlled_agents=controlled_agents,
        max_samples=max_samples,
    )
    metadata.update(
        {
            "include_no_select": include_no_select,
            "controlled_agents": sorted(controlled_agents) if controlled_agents is not None else None,
            "max_samples": max_samples,
        }
    )
    return dataset, metadata


def prepare_mounted_daily_replays(
    daily_replay_dirs: list[Path],
    output_dir: Path,
) -> tuple[list[Path], dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    from .replay_dataset import iter_replay_paths

    replay_paths = iter_replay_paths(daily_replay_dirs)
    metadata = {
        "source": "mounted_daily_replay_dirs",
        "daily_replay_dirs": [str(path) for path in daily_replay_dirs],
        "replay_paths": [str(path) for path in replay_paths],
    }
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "online_import_manifest.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return replay_paths, metadata


class KaggleReplayApiClient:
    def __init__(self) -> None:
        from kaggle.api.kaggle_api_extended import KaggleApi

        self.api = KaggleApi()
        self.api.authenticate()

    def list_recent_submission_ids(self, competition: str, page_size: int, limit: int) -> list[int]:
        submissions = self.api.competition_submissions(competition, page_size=page_size) or []
        ids: list[int] = []
        for submission in submissions:
            row = _plain_dict(submission)
            status = str(row.get("status", ""))
            if status and "COMPLETE" not in status.upper():
                continue
            ref = row.get("ref") or row.get("id")
            if ref is not None:
                ids.append(int(ref))
            if len(ids) >= limit:
                break
        return ids

    def list_episodes(self, competition: str, submission_id: int) -> list[ReplayEpisodeRef]:
        episodes = self.api.competition_list_episodes(int(submission_id)) or []
        refs: list[ReplayEpisodeRef] = []
        for episode in episodes:
            row = _plain_dict(episode)
            episode_id = row.get("id")
            if episode_id is None:
                continue
            agents = row.get("agents") or []
            team_names = []
            for agent in agents:
                agent_row = _plain_dict(agent)
                name = agent_row.get("teamName") or agent_row.get("team_name")
                if name is not None:
                    team_names.append(str(name))
            refs.append(
                ReplayEpisodeRef(
                    episode_id=int(episode_id),
                    source_submission_id=int(submission_id),
                    state=str(row.get("state")) if row.get("state") is not None else None,
                    episode_type=str(row.get("type")) if row.get("type") is not None else None,
                    team_names=team_names,
                    raw=row,
                )
            )
        return refs

    def download_replay(self, competition: str, episode_id: int) -> bytes:
        from kaggle.api.kaggle_api_extended import ApiGetEpisodeReplayRequest

        request = ApiGetEpisodeReplayRequest()
        request.episode_id = int(episode_id)
        with self.api.build_kaggle_client() as kaggle:
            response = kaggle.competitions.competition_api_client.get_episode_replay(request)
            response.raise_for_status()
            return bytes(response.content)


def select_episode_refs(
    client: ReplayApiClient,
    config: OnlineReplayImportConfig,
) -> list[ReplayEpisodeRef]:
    if config.episodes_index_dir is not None:
        refs = load_episode_refs_from_index(config.episodes_index_dir)
        refs = assign_time_splits(refs, config.reserve_recent_days)
        refs = [ref for ref in refs if ref.split == config.import_split]
        if not config.include_private_episodes:
            refs = [
                ref
                for ref in refs
                if (not ref.episode_type or "PUBLIC" in ref.episode_type.upper())
                and (not ref.state or "COMPLETE" in ref.state.upper())
            ]
        return refs[: config.max_replays]

    submission_ids = list(config.submission_ids)
    if not submission_ids:
        submission_ids = client.list_recent_submission_ids(
            config.competition,
            page_size=config.submission_page_size,
            limit=config.recent_submissions_to_use,
        )
    refs: list[ReplayEpisodeRef] = []
    seen: set[int] = set()
    for submission_id in submission_ids:
        for ref in client.list_episodes(config.competition, int(submission_id)):
            if ref.episode_id in seen:
                continue
            seen.add(ref.episode_id)
            if not config.include_private_episodes:
                if ref.episode_type and "PUBLIC" not in ref.episode_type.upper():
                    continue
                if ref.state and "COMPLETE" not in ref.state.upper():
                    continue
            refs.append(ref)
            if len(refs) >= config.max_replays:
                return refs
    return refs


def replay_cache_path(output_dir: Path, episode_id: int) -> Path:
    return output_dir / "replays" / f"episode-{int(episode_id)}-replay.json"


def download_episode_replays(
    client: ReplayApiClient,
    refs: list[ReplayEpisodeRef],
    config: OnlineReplayImportConfig,
) -> tuple[list[Path], list[dict[str, Any]]]:
    replay_dir = config.output_dir / "replays"
    replay_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    errors: list[dict[str, Any]] = []
    for ref in refs:
        path = replay_cache_path(config.output_dir, ref.episode_id)
        if config.cache_replays and path.exists() and path.stat().st_size > 1000:
            paths.append(path)
            continue
        last_error: Exception | None = None
        for attempt in range(config.max_download_retries):
            try:
                path.write_bytes(client.download_replay(config.competition, ref.episode_id))
                paths.append(path)
                if config.download_sleep_seconds:
                    time.sleep(config.download_sleep_seconds)
                break
            except Exception as exc:
                last_error = exc
                status_code = getattr(getattr(exc, "response", None), "status_code", None)
                if status_code == 429 and attempt < config.max_download_retries - 1:
                    time.sleep(min(60.0, 5.0 * (2**attempt)))
                    continue
                if attempt < config.max_download_retries - 1:
                    time.sleep(min(20.0, 2.0 * (attempt + 1)))
                    continue
                errors.append(
                    {
                        "episode_id": ref.episode_id,
                        "source_submission_id": ref.source_submission_id,
                        "error": f"{type(last_error).__name__}: {last_error}",
                    }
                )
    return paths, errors


def import_online_replay_dataset(
    config: OnlineReplayImportConfig,
    client: ReplayApiClient | None = None,
    include_no_select: bool = False,
    controlled_agents: set[int] | None = None,
    max_samples: int | None = None,
) -> tuple[ReplayDecisionDataset, dict[str, Any]]:
    replay_paths, metadata = prepare_online_replays(config, client=client)
    dataset = ReplayDecisionDataset(
        replay_paths,
        include_no_select=include_no_select,
        controlled_agents=controlled_agents,
        max_samples=max_samples,
    )
    metadata["config"].update(
        {
            "include_no_select": include_no_select,
            "controlled_agents": sorted(controlled_agents) if controlled_agents is not None else None,
            "max_samples": max_samples,
        }
    )
    return dataset, metadata


def prepare_online_replays(
    config: OnlineReplayImportConfig,
    client: ReplayApiClient | None = None,
) -> tuple[list[Path], dict[str, Any]]:
    client = client or KaggleReplayApiClient()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    refs = select_episode_refs(client, config)
    replay_paths, download_errors = download_episode_replays(client, refs, config)
    metadata = {
        "competition": config.competition,
        "submission_ids": sorted({ref.source_submission_id for ref in refs if ref.source_submission_id is not None}),
        "selected_episodes": [ref.__dict__ for ref in refs],
        "downloaded_replay_paths": [str(path) for path in replay_paths],
        "download_errors": download_errors,
        "config": {
            "max_replays": config.max_replays,
            "recent_submissions_to_use": config.recent_submissions_to_use,
            "include_private_episodes": config.include_private_episodes,
            "episodes_index_dir": str(config.episodes_index_dir) if config.episodes_index_dir is not None else None,
            "reserve_recent_days": config.reserve_recent_days,
            "import_split": config.import_split,
        },
    }
    reports_dir = config.output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "online_import_manifest.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return replay_paths, metadata
