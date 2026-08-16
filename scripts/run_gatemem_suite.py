from __future__ import annotations

import argparse
import atexit
import json
import os
import re
import shutil
import time
from queue import Empty, Queue
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gov_mem.eval.benchmark_official import load_and_validate_predictions, run_official_scorer
from gov_mem.llm.model_registry import resolve_llm_model
from gov_mem.utils.config import load_yaml_config
from gov_mem.utils.storage import (
    configure_local_environment,
    runtime_root,
    stage_explicit_dataset,
    stage_runtime_code,
    stage_tracked_tree,
    storage_audit,
)


ROOT = Path(__file__).resolve().parents[1]
RUN_GOVMEM = ROOT / "run_govmem.py"
DEFAULT_DATA_ROOT = ROOT / "dataset" / "GateMem" / "gatemem" / "data"
# Keep one request in flight per episode worker; five is the approved bounded
# validation level for the current OpenLux experiments.
MAX_SAFE_EPISODE_WORKERS = 5


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


def _validate_paper_protocol(*, config: dict, experiment_mode: str) -> None:
    """Reject suite runs that cannot be compared with GateMem's paper table."""

    if experiment_mode != "rag_naive_v3_typed_rerank":
        return
    evaluation = dict(config.get("evaluation") or {})
    if str(evaluation.get("protocol") or "") != "gatemem_paper_main":
        raise ValueError(
            "Formal RAG-Naive suite requires evaluation.protocol=gatemem_paper_main."
        )
    llm = dict(config.get("llm") or {})
    embedding = dict(config.get("embedding") or {})
    rag = dict(config.get("rag") or {})
    stage2 = dict(config.get("stage2") or {})
    ledger = dict(stage2.get("long_context_field_ledger") or {})
    judge = dict(evaluation.get("official_judge") or {})
    checks = {
        "evaluation.clean_benchmark": bool(evaluation.get("clean_benchmark")) is True,
        "evaluation.allow_gold_feedback": bool(evaluation.get("allow_gold_feedback")) is False,
        "llm.temperature": float(llm.get("temperature", -1)) == 0.2,
        "llm.max_output_tokens": int(llm.get("max_output_tokens", -1)) == 4096,
        "embedding.model": str(embedding.get("model") or "") == "text-embedding-3-small",
        "rag.naive_top_k": int(rag.get("naive_top_k", -1)) == 20,
        "stage2.long_context_field_ledger.enabled": bool(ledger.get("enabled")) is False,
        "official_judge.model": str(judge.get("model") or "") == "gpt-4o",
        "official_judge.temperature": float(judge.get("temperature", -1)) == 0.0,
        "official_judge.max_output_tokens": int(judge.get("max_output_tokens", -1)) == 4096,
        "official_judge.gate_by_action": bool(judge.get("gate_by_action")) is False,
    }
    invalid = [name for name, ok in checks.items() if not ok]
    if invalid:
        raise ValueError("GateMem paper protocol mismatch: " + ", ".join(invalid))


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
        readme_candidates = (
            ROOT / "API-Key_OpenLux.md",
            ROOT / "README_API_OpenLux",
            Path("/data_nvme/user/jli/codes/2027_ICLR_MARC/README_API_OpenLux.md"),
        )
        readme = next((candidate for candidate in readme_candidates if candidate.exists()), readme_candidates[0])
        keys: list[str] = []
        values = [value.strip() for value in os.environ.get(pool_env, "").split(",") if value.strip()]
        if values:
            return list(dict.fromkeys(values))
        if readme.exists():
            keys.extend(re.findall(r"sk-[A-Za-z0-9]+", readme.read_text(encoding="utf-8")))
        if keys:
            unique_keys = list(dict.fromkeys(keys))
            offset = int(os.environ.get("OPENLUX_API_KEY_START_INDEX", "0") or 0)
            if unique_keys:
                offset %= len(unique_keys)
                unique_keys = unique_keys[offset:] + unique_keys[:offset]
            return unique_keys
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
            if not isinstance(row.get("output", {}).get("memory_audit"), dict):
                audit_path = episode_dir / "prompt_audit" / "checkpoint_benchmark" / f"{checkpoint_id}.json"
                if audit_path.exists():
                    try:
                        audit = json.loads(audit_path.read_text(encoding="utf-8"))
                    except (OSError, json.JSONDecodeError):
                        audit = {}
                    if isinstance(audit, dict):
                        answer_prompt = audit.get("answer_prompt")
                        runtime_prompt_present = isinstance(answer_prompt, dict) and "context_text" in answer_prompt
                        row["output"]["memory_audit"] = {
                            "schema_version": 1,
                            "audit_status": str(audit.get("audit_status") or "runtime_answer_prompt"),
                            "prompt_context": {
                                "source": "answer_prompt.context_text" if runtime_prompt_present else "no_runtime_answer_prompt",
                                "text": str(answer_prompt.get("context_text") or "") if runtime_prompt_present else "",
                            },
                        }
            rows_by_id[checkpoint_id] = row
    merged_path = domain_output_dir / "predictions" / "checkpoint_benchmark" / "predictions.jsonl"
    merged_path.parent.mkdir(parents=True, exist_ok=True)
    with merged_path.open("w", encoding="utf-8") as handle:
        for checkpoint_id in sorted(rows_by_id):
            handle.write(json.dumps(rows_by_id[checkpoint_id], ensure_ascii=False) + "\n")
    return merged_path


def _validate_complete_domain_predictions(*, entries: list[dict], predictions_path: Path) -> None:
    """Make partial or unaudited shards impossible to score as a full domain."""

    expected_ids = {str(entry["checkpoint_id"]) for entry in entries}
    rows = load_and_validate_predictions(predictions_path)
    actual_ids = {str(row["checkpoint_id"]) for row in rows}
    missing = sorted(expected_ids - actual_ids)
    unexpected = sorted(actual_ids - expected_ids)
    if missing or unexpected or len(rows) != len(expected_ids):
        details = []
        if missing:
            details.append(f"missing={missing[:5]}")
        if unexpected:
            details.append(f"unexpected={unexpected[:5]}")
        details.append(f"expected_count={len(expected_ids)} actual_count={len(rows)}")
        raise RuntimeError("Incomplete domain prediction set; " + "; ".join(details))

    unaudited = []
    for row in rows:
        output = row.get("output") if isinstance(row.get("output"), dict) else {}
        audit = output.get("memory_audit") if isinstance(output, dict) else None
        context = audit.get("prompt_context") if isinstance(audit, dict) else None
        if not isinstance(context, dict) or "text" not in context:
            unaudited.append(str(row["checkpoint_id"]))
    if unaudited:
        raise RuntimeError(
            "Official context audit is incomplete; refusing to score domain. "
            f"unaudited={unaudited[:5]} count={len(unaudited)}"
        )


def _prediction_shard_is_complete(*, entries: list[dict], predictions_path: Path) -> bool:
    """Recognize a finished episode shard without rerunning model calls."""

    if not predictions_path.exists():
        return False
    try:
        rows = load_and_validate_predictions(predictions_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    expected_ids = {str(entry["checkpoint_id"]) for entry in entries}
    return len(rows) == len(expected_ids) and {
        str(row["checkpoint_id"]) for row in rows
    } == expected_ids


def _write_paper_metrics(out_dir: Path) -> dict:
    summary = _read_summary(out_dir / "summary.json")
    utility = float(summary.get("utility_accuracy", 0.0))
    access = float(summary.get("privacy_leakage_rate", 0.0))
    forgetting = float(summary.get("deletion_leakage_rate", 0.0))
    context_coverage = float(summary.get("context_audit_coverage_rate", 0.0))
    context_complete = context_coverage >= 1.0
    metrics = {
        "U": utility,
        "A": access,
        "F": forgetting,
        "OR": float(summary.get("over_refusal_rate", 0.0)),
        "MGS": utility * (1.0 - access) * (1.0 - forgetting),
        "gated_by_action": bool(summary.get("gated_by_action", False)),
        "privacy_context_leakage": (
            float(summary.get("privacy_context_leakage_rate", 0.0)) if context_complete else None
        ),
        "deletion_context_leakage": (
            float(summary.get("deletion_context_leakage_rate", 0.0)) if context_complete else None
        ),
        "context_audit_coverage": context_coverage,
        "context_audit_status": "complete" if context_complete else "incomplete_or_unknown",
        "source_summary": str(out_dir / "summary.json"),
    }
    (out_dir / "paper_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return metrics


def _publish_tree(source: Path, target: Path) -> None:
    """Publish only after completion; source is local scratch, target may be NFS."""
    for root, dirs, files in os.walk(source):
        relative = Path(root).relative_to(source)
        destination = target / relative
        destination.mkdir(parents=True, exist_ok=True)
        for name in files:
            shutil.copy2(Path(root) / name, destination / name)


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
    parser.add_argument(
        "--parallel_domains",
        type=int,
        default=4,
        help="Legacy domain scheduling setting; episode workers are globally bounded.",
    )
    parser.add_argument(
        "--parallel_episodes",
        type=int,
        default=4,
        help="Global number of episode subprocesses to run concurrently.",
    )
    args = parser.parse_args()

    requested_episode_workers = max(1, int(args.parallel_episodes))
    if requested_episode_workers > MAX_SAFE_EPISODE_WORKERS:
        raise ValueError(
            "Refusing unsafe episode concurrency: "
            f"requested={requested_episode_workers}, "
            f"maximum={MAX_SAFE_EPISODE_WORKERS}. "
            "A larger API-key pool does not justify more active workers."
        )

    suite_manifest = Path(args.suite_manifest).resolve()
    remote_output_dir = Path(args.output_dir).resolve()
    run_id = f"{suite_manifest.stem}-{remote_output_dir}"
    local_root = runtime_root(run_id)
    configure_local_environment(local_root)
    output_dir = local_root / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    source_data_root = Path(args.data_root).resolve()
    local_data_root = stage_explicit_dataset(
        source_data_root,
        local_root / "dataset",
        ("education", "household", "medical", "office"),
    )
    local_code_root = stage_runtime_code(ROOT, local_root / "code")
    local_run_govmem = local_code_root / "run_govmem.py"
    local_official_root = stage_tracked_tree(
        ROOT / "third_party" / "GateMem-official", local_root / "official"
    )
    os.environ["GOVMEM_OFFICIAL_BENCHMARK_ROOT"] = str(local_official_root)
    os.environ["GOVMEM_DATASET_ROOT"] = str(local_data_root)
    local_config = local_root / "config.yaml"
    shutil.copy2(Path(args.config).resolve(), local_config)
    runtime_fingerprint = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip() or "unverified-runtime"
    storage_audit({
        "project_dir": ROOT,
        "dataset_source": source_data_root,
        "dataset_runtime": local_data_root,
        "runtime_dir": local_root,
        "result_dir": output_dir,
        "final_output_dir": remote_output_dir,
    })
    payload = _load_manifest(suite_manifest)
    config = load_yaml_config(args.config)
    suite_name = str(payload.get("suite_name") or suite_manifest.stem)
    version = int(payload.get("version") or 1)
    grouped = _group_entries_by_domain(payload.get("entries", []))
    _validate_paper_protocol(config=config, experiment_mode=args.experiment_mode)
    llm_cfg = dict(config.get("llm") or {})
    embedding_cfg = dict(config.get("embedding") or {})
    llm_provider = str(llm_cfg.get("provider") or "yunwu")
    embedding_provider = str(embedding_cfg.get("provider") or llm_provider)
    llm_keys = _discover_api_keys(provider=llm_provider)
    embedding_keys = _discover_api_keys(provider=embedding_provider)
    judge_cfg = dict((config.get("evaluation") or {}).get("official_judge") or {})
    judge_provider = str(judge_cfg.get("provider") or "yunwu")
    judge_keys = _discover_api_keys(provider=judge_provider)
    if llm_provider in {"openlux", "openai-compatible-openlux"} and not llm_keys:
        raise RuntimeError(
            "No OpenLux memory-system API key found. Refusing to start a real run. "
            "Set OPENLUX_API_KEYS/OPENLUX_API_KEY or provide API-Key_OpenLux.md."
        )
    if embedding_provider in {"openlux", "openai-compatible-openlux"} and not embedding_keys:
        raise RuntimeError(
            "No OpenLux embedding API key found. Refusing to start a real run. "
            "Set OPENLUX_API_KEYS/OPENLUX_API_KEY or provide API-Key_OpenLux.md."
        )
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
    shared_memory_key_leasing = (
        llm_provider == embedding_provider
        and str(llm_cfg.get("api_base") or "") == str(embedding_cfg.get("api_base") or "")
        and llm_keys == embedding_keys
        and available_key_indices is not None
    )
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
            "parallel_episodes": requested_episode_workers,
            "episode_worker_scope": "global",
            "runtime_storage": "local_scratch",
            "filesystem_scheduler_threads": 0,
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
            "official_gate_by_action": bool(judge_cfg.get("gate_by_action", False)),
        },
        "domains": {},
    }

    def run_episode(domain: str, episode_id: str, entries: list[dict]) -> Path:
        domain_output_dir = output_dir / domain
        episode_output_dir = domain_output_dir / "episodes" / episode_id
        resume_existing_run = args.resume and (episode_output_dir / "run_metadata.json").exists()
        episode_manifest = _write_manifest(
            suite_name=suite_name,
            version=version,
            domain=domain,
            entries=entries,
            path=output_dir / "suite_manifests" / domain / f"{episode_id}.json",
        )
        cmd = [
            sys.executable,
            str(local_run_govmem),
            "--dataset_name",
            "checkpoint_benchmark",
            "--data_path",
            str(local_data_root / domain),
            "--output_dir",
            str(episode_output_dir),
            "--config",
            str(local_config),
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
        if resume_existing_run:
            cmd.append("--resume")
        prediction_path = episode_output_dir / "predictions" / "checkpoint_benchmark" / "predictions.jsonl"
        remote_prediction_path = (
            remote_output_dir
            / domain
            / "episodes"
            / episode_id
            / "predictions"
            / "checkpoint_benchmark"
            / "predictions.jsonl"
        )
        if args.resume:
            # Results are published only after a run completes, while every
            # resume attempt gets a fresh local scratch tree. Rehydrate the
            # small prediction shard from the published tree so the
            # coordinator can merge/evaluate it without starting an empty
            # child with --resume.
            source_prediction_path = (
                prediction_path
                if prediction_path.exists()
                else remote_prediction_path
            )
            if _prediction_shard_is_complete(
                entries=entries,
                predictions_path=source_prediction_path,
            ):
                if source_prediction_path != prediction_path:
                    prediction_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source_prediction_path, prediction_path)
                return episode_output_dir
        key_index: int | None = None
        embedding_key_index: int | None = None
        if available_key_indices is not None:
            # A lease is held for the complete child process lifetime. This
            # guarantees that concurrent episodes never share a Yunwu key;
            # when the pool is smaller than the requested parallelism, later
            # episodes wait and reuse keys only after an earlier episode ends.
            try:
                key_index = available_key_indices.get_nowait()
            except Empty:
                return None
        if available_embedding_key_indices is not None:
            if shared_memory_key_leasing:
                embedding_key_index = key_index
            else:
                try:
                    embedding_key_index = available_embedding_key_indices.get_nowait()
                except Empty:
                    if key_index is not None:
                        available_key_indices.put(key_index)
                    return None
        try:
            child_env = os.environ.copy()
            child_env["GOVMEM_RUNTIME_FINGERPRINT"] = runtime_fingerprint
            child_env["GOVMEM_DATASET_ROOT"] = str(local_data_root)
            child_env["GOVMEM_OFFICIAL_BENCHMARK_ROOT"] = str(local_official_root)
            _set_child_provider_key(
                child_env, provider_config=llm_cfg, provider=llm_provider,
                keys=llm_keys, key_index=key_index,
            )
            _set_child_provider_key(
                child_env, provider_config=embedding_cfg, provider=embedding_provider,
                keys=embedding_keys, key_index=embedding_key_index,
            )
            return subprocess.Popen(cmd, cwd=str(local_code_root), env=child_env), key_index, embedding_key_index, episode_output_dir

        except Exception:
            if key_index is not None:
                available_key_indices.put(key_index)
            if embedding_key_index is not None and not shared_memory_key_leasing:
                available_embedding_key_indices.put(embedding_key_index)
            raise

    def finalize_domain(domain: str, entries: list[dict], episode_dirs: list[Path]) -> tuple[str, dict]:
        domain_output_dir = output_dir / domain
        merged_predictions = _merge_episode_predictions(
            domain=domain, episode_dirs=episode_dirs, domain_output_dir=domain_output_dir
        )
        _validate_complete_domain_predictions(entries=entries, predictions_path=merged_predictions)
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
                    data_dir=local_data_root / domain,
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
                    resume_judge=bool(args.resume),
                    gate_by_action=bool(judge_config.get("gate_by_action", False)),
                )
            finally:
                if judge_key_index is not None:
                    available_judge_key_indices.put(judge_key_index)
        return domain, {
            "n_entries": len(entries),
            "n_episodes": len(episode_dirs),
            "summary": _read_summary(official_out_dir / "summary.json"),
            "paper_metrics": _write_paper_metrics(official_out_dir) if not args.skip_official_eval else {},
        }

    # A single coordinator owns all subprocesses. There are no scheduler
    # threads: one API key is leased to one child process for its lifetime.
    episode_jobs = [
        (domain, episode_id, episode_entries)
        for domain, entries in grouped.items()
        for episode_id, episode_entries in sorted(_episode_groups(entries).items())
    ]
    episode_dirs_by_domain: dict[str, list[Path]] = defaultdict(list)
    pending_jobs = list(episode_jobs)
    active: list[tuple[subprocess.Popen, str, int | None, int | None, Path]] = []

    def stop_active_children() -> None:
        for process, _domain, _key_index, _embedding_key_index, _episode_dir in active:
            if process.poll() is None:
                process.terminate()
        for process, _domain, _key_index, _embedding_key_index, _episode_dir in active:
            if process.poll() is None:
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()

    atexit.register(stop_active_children)
    while pending_jobs or active:
        launched = False
        while pending_jobs and len(active) < requested_episode_workers:
            domain, episode_id, episode_entries = pending_jobs[0]
            prepared = run_episode(domain, episode_id, episode_entries)
            if prepared is None:
                break
            if isinstance(prepared, Path):
                pending_jobs.pop(0)
                episode_dirs_by_domain[domain].append(prepared)
                launched = True
                continue
            process, key_index, embedding_key_index, episode_dir = prepared
            pending_jobs.pop(0)
            active.append((process, domain, key_index, embedding_key_index, episode_dir))
            launched = True

        still_active: list[tuple[subprocess.Popen, str, int | None, int | None, Path]] = []
        for process, domain, key_index, embedding_key_index, episode_dir in active:
            return_code = process.poll()
            if return_code is None:
                still_active.append((process, domain, key_index, embedding_key_index, episode_dir))
                continue
            if key_index is not None:
                available_key_indices.put(key_index)
            if embedding_key_index is not None and not shared_memory_key_leasing:
                available_embedding_key_indices.put(embedding_key_index)
            if return_code != 0:
                raise subprocess.CalledProcessError(return_code, process.args)
            episode_dirs_by_domain[domain].append(episode_dir)
        active = still_active
        if active and not launched:
            time.sleep(0.2)
        elif pending_jobs and not active and not launched:
            raise RuntimeError(
                "No API key lease is available and no episode is running; "
                "check the configured provider key pool."
            )

    results = [
        finalize_domain(domain, entries, episode_dirs_by_domain.get(domain, []))
        for domain, entries in grouped.items()
    ]
    for domain, result in sorted(results):
        suite_summary["domains"][domain] = result

    (output_dir / "suite_summary.json").write_text(
        json.dumps(suite_summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _publish_tree(output_dir, remote_output_dir)
    shutil.rmtree(local_root, ignore_errors=True)


if __name__ == "__main__":
    main()
