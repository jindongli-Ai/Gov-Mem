from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DOMAINS = ("education", "household", "medical", "office")


JsonDict = Dict[str, Any]


def _discover_dataset_root() -> Path:
    dataset_root = PROJECT_ROOT / "dataset"
    candidates = []
    if dataset_root.exists():
        for path in dataset_root.rglob("episodes.jsonl"):
            parent = path.parent
            if (parent / "checkpoints.jsonl").exists():
                candidates.append(parent)
    domain_dirs = [path for path in candidates if path.name in DOMAINS]
    if domain_dirs:
        return domain_dirs[0].parent
    if candidates:
        return candidates[0]
    raise FileNotFoundError("Could not discover checkpoint benchmark dataset root under dataset/.")


DATASET_ROOT = _discover_dataset_root()


@dataclass(frozen=True)
class DomainStats:
    domain: str
    episodes: int
    checkpoints: int


def _validate_domain(domain: str) -> str:
    if domain not in DOMAINS:
        raise ValueError(f"Unsupported checkpoint benchmark domain: {domain!r}. Expected one of {DOMAINS}.")
    return domain


def _domain_dir(domain: str) -> Path:
    return DATASET_ROOT / _validate_domain(domain)


def _jsonl_path(domain: str, split_name: str) -> Path:
    path = _domain_dir(domain) / f"{split_name}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint benchmark file not found: {path}")
    return path


def _iter_jsonl(path: Path) -> Iterator[JsonDict]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL in {path} at line {line_number}") from exc


def iter_episodes(domain: str) -> Iterator[JsonDict]:
    yield from _iter_jsonl(_jsonl_path(domain, "episodes"))


def iter_checkpoints(domain: str) -> Iterator[JsonDict]:
    yield from _iter_jsonl(_jsonl_path(domain, "checkpoints"))


def load_episodes(domain: str) -> List[JsonDict]:
    return list(iter_episodes(domain))


def load_checkpoints(domain: str) -> List[JsonDict]:
    return list(iter_checkpoints(domain))


def load_episode_index(domain: str) -> Dict[str, JsonDict]:
    return {episode["episode_id"]: episode for episode in iter_episodes(domain)}


def join_checkpoints_with_episodes(domain: str) -> List[JsonDict]:
    episodes = load_episode_index(domain)
    joined: List[JsonDict] = []
    for checkpoint in iter_checkpoints(domain):
        episode = episodes.get(checkpoint["episode_id"])
        if episode is None:
            raise KeyError(
                f"Checkpoint {checkpoint['checkpoint_id']} references missing episode "
                f"{checkpoint['episode_id']!r} in domain {domain!r}."
            )
        joined.append(
            {
                **checkpoint,
                "episode": episode,
            }
        )
    return joined


def dataset_stats(domains: Iterable[str] = DOMAINS) -> List[DomainStats]:
    stats: List[DomainStats] = []
    for domain in domains:
        domain = _validate_domain(domain)
        stats.append(
            DomainStats(
                domain=domain,
                episodes=sum(1 for _ in iter_episodes(domain)),
                checkpoints=sum(1 for _ in iter_checkpoints(domain)),
            )
        )
    return stats


def dataset_overview(domains: Iterable[str] = DOMAINS) -> Mapping[str, Mapping[str, int]]:
    return {
        stat.domain: {
            "episodes": stat.episodes,
            "checkpoints": stat.checkpoints,
        }
        for stat in dataset_stats(domains)
    }


def load_all_episode_indexes(domains: Iterable[str] = DOMAINS) -> Dict[str, Dict[str, JsonDict]]:
    return {domain: load_episode_index(domain) for domain in domains}
