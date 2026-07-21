#!/usr/bin/env python3
"""Materialize fair, dev-only provenance-chain ablation configurations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

try:
    import yaml
except ModuleNotFoundError:
    deps_root = ROOT / "third_party" / "python_deps"
    fallback = next((path for path in deps_root.iterdir() if path.is_dir()), None) if deps_root.exists() else None
    if fallback is None:
        raise
    sys.path.insert(0, str(fallback))
    import yaml

from gov_mem.evolution.dev_guard import load_dev_attestation
from gov_mem.utils.config import deep_update, load_yaml_config


def _validate_variant(name: str, config: dict) -> None:
    graph = bool(config.get("use_governed_graph", False))
    dual = bool(config.get("use_dual_channel_retrieval", False))
    renderer = bool(config.get("enable_graph_typed_slot_realization", False))
    policy_compiler = bool((config.get("governance_runtime") or {}).get("use_llm_policy_frame_compiler", False))
    if renderer and not (graph and dual and policy_compiler):
        raise ValueError(f"{name}: graph typed-slot realization requires graph, dual retrieval, and cited policy compilation")
    if not graph and (dual or renderer):
        raise ValueError(f"{name}: graph-disabled ablation cannot leave graph-dependent modules enabled")
    if name == "no_governance_channel" and renderer:
        raise ValueError("no_governance_channel must not retain graph typed-slot realization")
    if name == "no_policy_frame_compiler" and renderer:
        raise ValueError("no_policy_frame_compiler must not retain graph typed-slot realization")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare declared dev-only Gov-Mem ablations.")
    parser.add_argument("--manifest", default="configs/ablations/gov_mem_provenance_dev.yaml")
    parser.add_argument("--dev_attestation", required=True)
    parser.add_argument("--output_root", default=None)
    parser.add_argument("--parallel_domains", type=int, default=4)
    args = parser.parse_args()
    if args.parallel_domains < 1:
        raise ValueError("--parallel_domains must be positive")

    manifest_path = (ROOT / args.manifest).resolve()
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    attestation = load_dev_attestation(args.dev_attestation)
    if str(attestation.get("manifest") or "") != str(manifest.get("suite_manifest") or ""):
        raise ValueError("Ablation suite manifest must match the attested ID-only development manifest")
    base_config = load_yaml_config(ROOT / str(manifest["base_config"]))
    output_root = Path(args.output_root).resolve() if args.output_root else ROOT / str(manifest["output_root"])
    declared_runs = {str(Path(path).resolve()) for path in attestation["source_runs"]}
    commands = []
    for variant in list(manifest.get("variants") or []):
        name = str(variant["name"])
        run_dir = output_root / name
        if str(run_dir.resolve()) not in declared_runs:
            raise ValueError(f"Variant output must be predeclared in attestation.source_runs: {run_dir}")
        resolved = deep_update(base_config, dict(variant.get("overrides") or {}))
        resolved = deep_update(resolved, {
            "runner": {
                "suite_manifest": str(manifest["suite_manifest"]),
                "output_dir": str(run_dir.relative_to(ROOT)),
                "stage": "all",
                "skip_official_eval": False,
            }
        })
        _validate_variant(name, resolved)
        config_path = output_root / "configs" / f"{name}.yaml"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(yaml.safe_dump(resolved, sort_keys=False, allow_unicode=True), encoding="utf-8")
        commands.append({
            "variant": name,
            "label": str(variant.get("label") or name),
            "config": str(config_path.relative_to(ROOT)),
            "output_dir": str(run_dir.relative_to(ROOT)),
            "command": (
                "python scripts/run_attested_dev_evaluation.py "
                f"--manifest {manifest['suite_manifest']} --dev_attestation {args.dev_attestation} "
                f"--config {config_path.relative_to(ROOT)} --output_dir {run_dir.relative_to(ROOT)} "
                f"--parallel_domains {args.parallel_domains}"
            ),
        })
    command_path = output_root / "commands.json"
    command_path.write_text(json.dumps({"manifest": str(manifest_path), "commands": commands}, indent=2) + "\n", encoding="utf-8")
    print(f"prepared_variants={len(commands)}")
    print(f"commands={command_path}")


if __name__ == "__main__":
    main()
