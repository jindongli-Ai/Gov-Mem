from __future__ import annotations

import argparse
import json
import os
import re
from queue import Queue
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gov_mem.eval.benchmark_official import load_and_validate_predictions, run_official_scorer
from gov_mem.llm.model_registry import resolve_llm_model
from gov_mem.utils.config import load_yaml_config


ROOT = Path(__file__).resolve().parents[1]
RUN_GOVMEM = ROOT / "run_govmem.py"
DEFAULT_DATA_ROOT = ROOT / "dataset" / "GateMem" / "gatemem" / "data"


def _load_manifest(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "entries" not in payload:
        raise ValueError(f"Suite manifest must be an object with entries: {path}")
    return payload


def _group_entries_by_domain(entries: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for entry in entries:
        domain = str(entry["domain"])
        grouped[domain].append(entry)
    return dict(grouped)


def _write_domain_manifest(
    *,
    suite_name: str,
    version: int,
    domain: str,
    entries: list[dict],
    output_dir: Path,
) -> Path:
    manifest = {
        "suite_name": f"{suite_name}:{domain}",
        "dataset_name": "checkpoint_benchmark",
        "version": version,
        "domain": domain,
        "entries": entries,
    }
    path = output_dir / "suite_manifests" / f"{domain}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def _read_summary(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _discover_api_keys(*, provider: str) -> list[str]:
    """Load the configured pool without exposing key values in manifests or logs."""
    provider = provider.lower().strip()
    if provider in {"jellyfishp", "openai-compatible-jellyfishp"}:
        pool_env = "JELLYFISHP_API_KEYS"
        single_env = "JELLYFISHP_API_KEY"
        readme_name = "README_API_jellyfishp.md"
    elif provider in {"openlux", "openai-compatible-openlux"}:
        pool_env = "OPENLUX_API_KEYS"
        single_env = "OPENLUX_API_KEY"
        # Keep the new provider's credentials in the sibling project as
        # requested; do not copy the secret-bearing README into this repo.
        readme = ROOT / "README_API_OpenLux"
        if not readme.exists():
            readme = Path("/data_nvme/user/jli/codes/2027_ICLR_MARC/README_API_OpenLux.md")
        keys: list[str] = []
        values = [value.strip() for value in os.environ.get(pool_env, "").split(",") if value.strip()]
        if values:
            return list(dict.fromkeys(values))
        if readme.exists():
            keys.extend(re.findall(r"sk-[A-Za-z0-9]+", readme.read_text(encoding="utf-8")))
        if keys:
            return list(dict.fromkeys(keys))
        single_key = os.environ.get(single_env, "").strip()
        return [single_key] if single_key else []
    else:
        pool_env = "YUNWU_API_KEYS"
        single_env = "YUNWU_API_KEY"
        readme_name = "README_API_Yunwu.md"
    values = [
        value.strip()
        for value in os.environ.get(pool_env, "").split(",")
        if value.strip()
    ]
    if values:
        return list(dict.fromkeys(values))

    readme = ROOT / readme_name
    keys: list[str] = []
    if readme.exists():
        # Accept Markdown code spans and quoted list entries.  The key pool
        # must be complete before parallel episodes are started; silently
        # reducing it to the backtick-formatted entries causes avoidable API
        # contention and invalidates runtime comparisons.
        text = readme.read_text(encoding="utf-8")
        keys.extend(re.findall(r"sk-[A-Za-z0-9]+", text))
    if keys:
        return list(dict.fromkeys(keys))

    single_key = os.environ.get(single_env, "").strip()
    if single_key:
        return [single_key]
    return list(dict.fromkeys(keys))


def _default_api_key_env(provider: str) -> str:
    provider = provider.lower().strip()
    if provider in {"jellyfishp", "openai-compatible-jellyfishp"}:
        return "JELLYFISHP_API_KEY"
    if provider in {"openlux", "openai-compatible-openlux"}:
        return "OPENLUX_API_KEY"
    return "YUNWU_API_KEY"


def _pool_env(api_key_env: str) -> str:
    return f"{api_key_env[:-3]}KEYS" if api_key_env.endswith("KEY") else f"{api_key_env}_POOL"


def _set_child_provider_key(
    child_env: dict[str, str],
    *,
    provider_config: dict,
    provider: str,
    keys: list[str],
    key_index: int | None,
) -> None:
    if key_index is None or not keys:
        return
    key_env = str(provider_config.get("api_key_env") or _default_api_key_env(provider))
    child_env[key_env] = keys[key_index]
    # Restrict retries to the episode's leased key. Otherwise a failed request
    # could rotate into a key belonging to another concurrent episode.
    child_env[_pool_env(key_env)] = keys[key_index]


def _episode_groups(entries: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for entry in entries:
        grouped[str(entry["episode_id"])].append(entry)
    return dict(grouped)


def _write_manifest(*, suite_name: str, version: int, domain: str, entries: list[dict], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "suite_name": f"{suite_name}:{domain}:{entries[0]['episode_id']}",
                "dataset_name": "checkpoint_benchmark",
                "version": version,
                "domain": domain,
                "episode_id": entries[0]["episode_id"],
                "entries": entries,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _merge_episode_predictions(*, domain: str, episode_dirs: list[Path], domain_output_dir: Path) -> Path:
    rows_by_id: dict[str, dict] = {}
    for episode_dir in episode_dirs:
        prediction_path = episode_dir / "predictions" / "checkpoint_benchmark" / "predictions.jsonl"
        for row in load_and_validate_predictions(prediction_path):
            checkpoint_id = str(row["checkpoint_id"])
            if checkpoint_id in rows_by_id:
                raise ValueError(f"Duplicate checkpoint_id across episode shards: {checkpoint_id}")
            rows_by_id[checkpoint_id] = row
    merged_path = domain_output_dir / "predictions" / "checkpoint_benchmark" / "predictions.jsonl"
    merged_path.parent.mkdir(parents=True, exist_ok=True)
    with merged_path.open("w", encoding="utf-8") as handle:
        for checkpoint_id in sorted(rows_by_id):
            handle.write(json.dumps(rows_by_id[checkpoint_id], ensure_ascii=False) + "\n")
    return merged_path


def _write_paper_metrics(out_dir: Path) -> dict:
    summary = _read_summary(out_dir / "summary.json")
    utility = float(summary.get("utility_accuracy", 0.0))
    access = float(summary.get("privacy_leakage_rate", 0.0))
    forgetting = float(summary.get("deletion_leakage_rate", 0.0))
    metrics = {
        "U": utility,
        "A": access,
        "F": forgetting,
        "OR": float(summary.get("over_refusal_rate", 0.0)),
        "MGS": utility * (1.0 - access) * (1.0 - forgetting),
        "source_summary": str(out_dir / "summary.json"),
    }
    (out_dir / "paper_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a fixed checkpoint benchmark suite across all domains.")
    parser.add_argument("--suite_manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--experiment_mode", required=True)
    parser.add_argument("--stage", default="all", choices=["all", "ingest", "retrieve", "answer", "evaluate"])
    parser.add_argument("--data_root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--skip_official_eval", action="store_true")
    parser.add_argument(
        "--base_model",
        default=None,
        help="Override the configured base model for every domain subprocess.",
    )
    parser.add_argument(
        "--embedding_model",
        default=None,
        help="Override the configured embedding model for every domain subprocess.",
    )
    parser.add_argument("--resume", action="store_true", help="Strictly resume compatible interrupted domain runs.")
    parser.add_argument("--parallel_domains", type=int, default=4, help="Number of domain subprocesses to run concurrently.")
    parser.add_argument(
        "--parallel_episodes",
        type=int,
        default=4,
        help="Run episode shards concurrently with one stable Yunwu key per episode.",
    )
    args = parser.parse_args()

    suite_manifest = Path(args.suite_manifest)
    output_dir = Path(args.output_dir)
    payload = _load_manifest(suite_manifest)
    config = load_yaml_config(args.config)
    suite_name = str(payload.get("suite_name") or suite_manifest.stem)
    version = int(payload.get("version") or 1)
    grouped = _group_entries_by_domain(payload.get("entries", []))
    llm_cfg = dict(config.get("llm") or {})
    embedding_cfg = dict(config.get("embedding") or {})
    llm_provider = str(llm_cfg.get("provider") or "yunwu")
    embedding_provider = str(embedding_cfg.get("provider") or llm_provider)
    llm_keys = _discover_api_keys(provider=llm_provider)
    embedding_keys = _discover_api_keys(provider=embedding_provider)
    judge_cfg = dict((config.get("evaluation") or {}).get("official_judge") or {})
    judge_provider = str(judge_cfg.get("provider") or "yunwu")
    judge_keys = _discover_api_keys(provider=judge_provider)
    memory_system_base_llm = str(
        args.base_model
        or llm_cfg.get("base_model")
        or resolve_llm_model(config, "answering")
    )
    official_evaluation_llm = str(judge_cfg.get("model") or "gpt-4o")
    embedding_model = str(args.embedding_model or embedding_cfg.get("model") or "")
    available_key_indices: Queue[int] | None = None
    if llm_keys:
        available_key_indices = Queue()
        for index in range(len(llm_keys)):
            available_key_indices.put(index)
    available_embedding_key_indices: Queue[int] | None = None
    if embedding_keys:
        available_embedding_key_indices = Queue()
        for index in range(len(embedding_keys)):
            available_embedding_key_indices.put(index)
    available_judge_key_indices: Queue[int] | None = None
    if judge_keys:
        available_judge_key_indices = Queue()
        for index in range(len(judge_keys)):
            available_judge_key_indices.put(index)

    suite_summary: dict[str, dict] = {
        "suite_name": suite_name,
        "version": version,
        "execution": {
            "parallel_domains": max(1, int(args.parallel_domains)),
            "parallel_episodes": max(1, int(args.parallel_episodes)),
            "memory_system_key_pool_size": len(llm_keys),
            "memory_system_key_isolation": bool(llm_keys),
            "memory_system_provider": llm_provider,
            "memory_system_base_llm": memory_system_base_llm,
            "embedding_provider": embedding_provider,
            "embedding_model": embedding_model,
            "embedding_key_pool_size": len(embedding_keys),
            "embedding_key_isolation": bool(embedding_keys),
            "official_evaluation_llm": official_evaluation_llm,
            "official_evaluation_provider": judge_provider,
            "official_evaluation_key_pool_size": len(judge_keys),
        },
        "domains": {},
    }

    def run_episode(domain: str, episode_id: str, entries: list[dict]) -> Path:
        domain_output_dir = output_dir / domain
        episode_output_dir = domain_output_dir / "episodes" / episode_id
        episode_manifest = _write_manifest(
            suite_name=suite_name,
            version=version,
            domain=domain,
            entries=entries,
            path=output_dir / "suite_manifests" / domain / f"{episode_id}.json",
        )
        cmd = [
            sys.executable,
            str(RUN_GOVMEM),
            "--dataset_name",
            "checkpoint_benchmark",
            "--data_path",
            str(Path(args.data_root) / domain),
            "--output_dir",
            str(episode_output_dir),
            "--config",
            args.config,
            "--experiment_mode",
            args.experiment_mode,
            "--stage",
            args.stage,
            "--checkpoint_manifest",
            str(episode_manifest),
            "--skip_official_eval",
        ]
        if args.base_model:
            cmd.extend(["--base_model", args.base_model])
        if args.embedding_model:
            cmd.extend(["--embedding_model", args.embedding_model])
        if args.resume:
            cmd.append("--resume")
        key_index: int | None = None
        embedding_key_index: int | None = None
        if available_key_indices is not None:
            # A lease is held for the complete child process lifetime. This
            # guarantees that concurrent episodes never share a Yunwu key;
            # when the pool is smaller than the requested parallelism, later
            # episodes wait and reuse keys only after an earlier episode ends.
            key_index = available_key_indices.get()
        if available_embedding_key_indices is not None:
            embedding_key_index = available_embedding_key_indices.get()
        try:
            child_env = os.environ.copy()
            _set_child_provider_key(
                child_env, provider_config=llm_cfg, provider=llm_provider,
                keys=llm_keys, key_index=key_index,
            )
            _set_child_provider_key(
                child_env, provider_config=embedding_cfg, provider=embedding_provider,
                keys=embedding_keys, key_index=embedding_key_index,
            )
            subprocess.run(cmd, check=True, cwd=str(ROOT), env=child_env)
            return episode_output_dir
        finally:
            if key_index is not None:
                available_key_indices.put(key_index)
            if embedding_key_index is not None:
                available_embedding_key_indices.put(embedding_key_index)

    def run_domain(domain: str, entries: list[dict]) -> tuple[str, dict]:
        domain_output_dir = output_dir / domain
        episodes = _episode_groups(entries)
        indexed_episodes = list(sorted(episodes.items()))
        episode_dirs: list[Path] = []
        max_workers = max(1, min(int(args.parallel_episodes), len(indexed_episodes)))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    run_episode,
                    domain,
                    episode_id,
                    episode_entries,
                )
                for episode_id, episode_entries in indexed_episodes
            ]
            for future in as_completed(futures):
                episode_dirs.append(future.result())
        merged_predictions = _merge_episode_predictions(
            domain=domain, episode_dirs=episode_dirs, domain_output_dir=domain_output_dir
        )
        official_out_dir = domain_output_dir / "official_eval" / "checkpoint_benchmark" / domain
        judge_config = dict((load_yaml_config(args.config).get("evaluation") or {}).get("official_judge") or {})
        if not args.skip_official_eval:
            judge_key_index: int | None = None
            if available_judge_key_indices is not None:
                judge_key_index = available_judge_key_indices.get()
            try:
                judge_key = judge_keys[judge_key_index] if judge_key_index is not None else None
                run_official_scorer(
                    domain=domain,
                    data_dir=Path(args.data_root) / domain,
                    predictions_path=merged_predictions,
                    out_dir=official_out_dir,
                    use_llm_judge=bool(judge_config.get("enabled", True)),
                    judge_provider=str(judge_config.get("provider") or "yunwu"),
                    judge_model=str(judge_config.get("model") or "gpt-4o"),
                    judge_temperature=float(judge_config.get("temperature", 0.0)),
                    judge_max_output_tokens=int(judge_config.get("max_output_tokens", 4096)),
                    judge_api_base=str(judge_config.get("api_base") or "https://yunwu.ai/v1"),
                    judge_api_key_env=str(judge_config.get("api_key_env") or "YUNWU_API_KEY"),
                    judge_api_key=judge_key,
                    judge_concurrency=int(judge_config.get("concurrency", 4)),
                )
            finally:
                if judge_key_index is not None:
                    available_judge_key_indices.put(judge_key_index)
        return domain, {
            "n_entries": len(entries),
            "n_episodes": len(indexed_episodes),
            "summary": _read_summary(official_out_dir / "summary.json"),
            "paper_metrics": _write_paper_metrics(official_out_dir) if not args.skip_official_eval else {},
        }

    workers = max(1, min(int(args.parallel_domains), len(grouped)))
    if workers == 1:
        results = [run_domain(domain, entries) for domain, entries in grouped.items()]
    else:
        results = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(run_domain, domain, entries): domain
                for domain, entries in grouped.items()
            }
            for future in as_completed(futures):
                results.append(future.result())
    for domain, result in sorted(results):
        suite_summary["domains"][domain] = result

    (output_dir / "suite_summary.json").write_text(
        json.dumps(suite_summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
