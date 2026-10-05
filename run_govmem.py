"""Run the canonical Gov-Mem V8 Late Governance pipeline.

The historical experiment modes remain accepted so frozen artifacts can be
reproduced, but new runs should use ``govmem_v8_late_governance`` and the V8
configuration documented in the project README.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gov_mem.pipeline import GovMemRunner
from gov_mem.utils.config import deep_update, load_yaml_config, set_random_seed


def _parse_key_value_pairs(items: list[str] | None) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"Expected KEY=VALUE format, got: {item}")
        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            raise ValueError(f"Empty key in KEY=VALUE item: {item}")
        out[key] = value
    return out


def _checkpoint_file(data_path: str) -> Path:
    source = Path(data_path)
    return source / "checkpoints.jsonl" if source.is_dir() else source


def _load_checkpoint_manifest(path: str | None, *, data_path: str) -> list[str] | None:
    if not path:
        return None
    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        if "entries" in payload:
            checkpoint_ids = [str(item["checkpoint_id"]) for item in payload.get("entries", [])]
        elif "checkpoint_ids" in payload:
            checkpoint_ids = [str(item) for item in payload.get("checkpoint_ids", [])]
        else:
            checkpoint_ids = None
        if checkpoint_ids is not None:
            if payload.get("selection_unit") == "complete_episode":
                episode_id = str(payload.get("episode_id") or "").strip()
                if not episode_id:
                    raise ValueError("complete_episode manifest requires episode_id")
                checkpoint_path = _checkpoint_file(data_path)
                expected = []
                with checkpoint_path.open("r", encoding="utf-8") as handle:
                    for line in handle:
                        row = json.loads(line)
                        if str(row.get("episode_id") or "") == episode_id:
                            expected.append(str(row["checkpoint_id"]))
                if not expected:
                    raise ValueError(f"Episode not found in checkpoint source: {episode_id}")
                if checkpoint_ids != expected:
                    missing = [value for value in expected if value not in checkpoint_ids]
                    extra = [value for value in checkpoint_ids if value not in expected]
                    raise ValueError(
                        "Incomplete or reordered complete_episode manifest: "
                        f"episode={episode_id} expected={len(expected)} actual={len(checkpoint_ids)} "
                        f"missing={missing} extra={extra}"
                    )
            return checkpoint_ids
    if isinstance(payload, list):
        return [str(item) for item in payload]
    raise ValueError(f"Unsupported checkpoint manifest format: {manifest_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the canonical Gov-Mem V8 pipeline.")
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--max_instances", type=int, default=None)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--checkpoint_manifest", default=None)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume only an interrupted run with identical config and source fingerprint.",
    )
    parser.add_argument(
        "--stage",
        default="all",
        choices=["all", "ingest", "retrieve", "answer", "evaluate"],
    )
    parser.add_argument("--skip_official_eval", action="store_true")
    parser.add_argument(
        "--experiment_mode",
        default="govmem_v8_late_governance",
        choices=[
            "rag_naive",
            "rag_naive_v3_typed_rerank",
            "govmem_v4_symbolic",
            "govmem_v8_late_governance",
            "govmem_v8_dev1_late_governance",
            "rag_policy",
            "govmem_symbolic",
            # Legacy name retained so historical configs and outputs remain reproducible.
            "rag_policy_amem",
            "govmem_rag_policy_incremental",
            "govmem_structured_old",
            "stateful_policy_reasoning",
        ],
    )
    parser.add_argument("--llm_provider", default=None)
    parser.add_argument("--llm_api_base", default=None)
    parser.add_argument("--llm_api_key_env", default=None)
    parser.add_argument("--base_model", default=None)
    parser.add_argument(
        "--memory_governed_slot_graph",
        choices=["true", "false"],
        default=None,
        help="Override the optional ingestion-time governed slot graph channel.",
    )
    parser.add_argument(
        "--embedding_model",
        default=None,
        help="Override the configured embedding model for retrieval experiments.",
    )
    parser.add_argument(
        "--policy_ablation",
        default=None,
        choices=["full", "wo_temporal_transition", "wo_policy_conflict_resolution", "wo_explicit_memory_status", "llm_only", "rule_only"],
    )
    parser.add_argument(
        "--role_model",
        action="append",
        default=[],
        help="Override a role-specific model with ROLE=MODEL, e.g. action_decision=gpt-5",
    )
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    if args.experiment_mode is not None:
        config = deep_update(config, {"experiment": {"mode": args.experiment_mode}})
    llm_updates = {}
    if args.llm_provider is not None:
        llm_updates["provider"] = args.llm_provider
    if args.llm_api_base is not None:
        llm_updates["api_base"] = args.llm_api_base
    if args.llm_api_key_env is not None:
        llm_updates["api_key_env"] = args.llm_api_key_env
    if args.base_model is not None:
        llm_updates["base_model"] = args.base_model
    role_models = _parse_key_value_pairs(args.role_model)
    if role_models:
        llm_updates["role_models"] = role_models
    if llm_updates:
        config = deep_update(config, {"llm": llm_updates})
    if args.embedding_model is not None:
        config = deep_update(config, {"embedding": {"model": args.embedding_model}})
    if args.memory_governed_slot_graph is not None:
        config = deep_update(config, {
            "memory_governed_slot_graph": {
                "enabled": args.memory_governed_slot_graph == "true",
            }
        })
    if args.policy_ablation:
        ablation_updates = {
            "full": {"mode": "full"},
            "wo_temporal_transition": {"temporal_transition": False},
            "wo_policy_conflict_resolution": {"conflict_resolution": False},
            "wo_explicit_memory_status": {"explicit_memory_status": False},
            "llm_only": {"mode": "llm_only"},
            "rule_only": {"mode": "rule_only"},
        }[args.policy_ablation]
        config = deep_update(config, {"policy_reasoning": ablation_updates})
    set_random_seed(int(config["project"]["seed"]))

    runner = GovMemRunner(
        dataset_name=args.dataset_name,
        data_path=args.data_path,
        output_dir=args.output_dir,
        config=config,
        stage=args.stage,
        max_instances=args.max_instances,
        start_index=args.start_index,
        checkpoint_ids=_load_checkpoint_manifest(args.checkpoint_manifest, data_path=args.data_path),
        resume=args.resume,
        run_official_benchmark_eval=not args.skip_official_eval,
        experiment_mode=str((config.get("experiment") or {}).get("mode") or "govmem_v8_late_governance"),
    )
    runner.run()


if __name__ == "__main__":
    main()
