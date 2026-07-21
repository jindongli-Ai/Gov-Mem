from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gov_mem.eval.benchmark_official import load_and_validate_predictions, run_official_scorer
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


def _discover_api_keys() -> list[str]:
    """Load the configured pool without exposing key values in manifests or logs."""
    values = [
        value.strip()
        for value in os.environ.get("YUNWU_API_KEYS", "").split(",")
        if value.strip()
    ]
    if values:
        return list(dict.fromkeys(values))

    readme = ROOT / "README_API_Yunwu.md"
    if not readme.exists():
        return []
    keys: list[str] = []
    for line in readme.read_text(encoding="utf-8").splitlines():
        if "sk-" not in line:
            continue
        candidate = line.split("`")[1] if "`" in line else line.strip()
        if candidate.startswith("sk-"):
            keys.append(candidate)
    return list(dict.fromkeys(keys))


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
    parser.add_argument("--resume", action="store_true", help="Strictly resume compatible interrupted domain runs.")
    parser.add_argument("--parallel_domains", type=int, default=1, help="Number of domain subprocesses to run concurrently.")
    parser.add_argument(
        "--parallel_episodes",
        type=int,
        default=1,
        help="Run episode shards concurrently with one stable Yunwu key per episode.",
    )
    args = parser.parse_args()

    suite_manifest = Path(args.suite_manifest)
    output_dir = Path(args.output_dir)
    payload = _load_manifest(suite_manifest)
    suite_name = str(payload.get("suite_name") or suite_manifest.stem)
    version = int(payload.get("version") or 1)
    grouped = _group_entries_by_domain(payload.get("entries", []))
    api_keys = _discover_api_keys()
    episode_key_indices: dict[tuple[str, str], int] = {}
    next_key_index = 0
    for domain in sorted(grouped):
        for episode_id in sorted(_episode_groups(grouped[domain])):
            episode_key_indices[(domain, episode_id)] = next_key_index
            next_key_index += 1

    suite_summary: dict[str, dict] = {
        "suite_name": suite_name,
        "version": version,
        "domains": {},
    }

    def run_episode(domain: str, episode_id: str, entries: list[dict], key_index: int) -> Path:
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
        if args.resume:
            cmd.append("--resume")
        child_env = os.environ.copy()
        if api_keys:
            child_env["YUNWU_API_KEY"] = api_keys[key_index % len(api_keys)]
            child_env["YUNWU_API_KEYS"] = ",".join(api_keys)
        subprocess.run(cmd, check=True, cwd=str(ROOT), env=child_env)
        return episode_output_dir

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
                    episode_key_indices[(domain, episode_id)],
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
            judge_key = api_keys[0] if api_keys else None
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
