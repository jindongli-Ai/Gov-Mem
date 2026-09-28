"""Score a frozen four-episode V8 diagnostic against identical saved RAG predictions."""
from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from scripts.run_gatemem_suite import _discover_api_keys
from scripts.score_v8_paired_episode_suite import dump, judge, rows
from scripts.score_v8_module_ablations import component_metrics

DOMAINS = ("medical", "office", "education", "household")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def subset(source: Path, ids: list[str]) -> list[dict]:
    by_id = {row["checkpoint_id"]: row for row in rows(source)}
    if len(by_id) != len(rows(source)) or any(cid not in by_id for cid in ids):
        raise ValueError(f"Incomplete or duplicate source: {source}")
    return [by_id[cid] for cid in ids]


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records)
    if path.exists() and path.read_text() != content:
        raise ValueError(f"Frozen output changed: {path}")
    path.write_text(content)


def prepare(gov: Path, baseline: Path, output: Path) -> dict:
    manifest = json.loads((gov / "manifest.json").read_text())
    identity = json.loads((gov / "execution_identity.json").read_text())
    if set(manifest["domains"]) != set(DOMAINS):
        raise ValueError("Expected four domains")
    source_identity = json.loads((baseline / "official_source_identity.json").read_text())
    bench = Path(__file__).resolve().parents[1] / "third_party/GateMem-official/bench"
    actual = {str(p.relative_to(bench)): sha(p) for p in bench.rglob("*.py")}
    if actual != source_identity["files"]:
        raise ValueError("Official judge source changed")
    baseline_provenance = {}
    for domain in DOMAINS:
        spec = manifest["domains"][domain]
        ids = [row["checkpoint_id"] for row in manifest["entries"] if row["domain"] == domain]
        if len(ids) != spec["checkpoint_count"] or len(ids) != len(set(ids)):
            raise ValueError("Manifest checkpoint mismatch")
        episode = next(iter(spec["selected_episodes"]))
        completion = json.loads((gov / "govmem" / domain / episode / "complete.json").read_text())
        if completion["runtime_sha256"] != identity["runtime_sha256"]:
            raise ValueError("Mixed Gov-Mem runtime snapshots")
        gov_rows = subset(Path(completion["prediction_path"]), ids)
        baseline_dir = baseline / "rag_naive" / domain
        base_rows = subset(baseline_dir / "predictions.jsonl", ids)
        judge_rows = subset(baseline_dir / "official_eval/judge_scores.jsonl", ids)
        if any(not row.get("judge", {}).get("parse_ok") for row in judge_rows):
            raise ValueError("Baseline judge parse failure")
        if any(row.get("judge", {}).get({"utility":"utility_ok","privacy":"privacy_leak","safety":"deletion_leak"}[row["query_type"]]) is None for row in judge_rows):
            raise ValueError("Baseline applicable judge label missing")
        data_dir = gov / "selected_dataset" / domain
        for name in ("episodes.jsonl", "checkpoints.jsonl"):
            if not (data_dir / name).exists():
                raise ValueError("Diagnostic selected data missing")
        write_jsonl(output / "govmem" / domain / "predictions.jsonl", gov_rows)
        write_jsonl(output / "rag_naive" / domain / "predictions.jsonl", base_rows)
        write_jsonl(output / "rag_naive" / domain / "official_eval/judge_scores.jsonl", judge_rows)
        baseline_provenance[domain] = {
            "source_predictions_sha256": sha(baseline_dir / "predictions.jsonl"),
            "source_judgments_sha256": sha(baseline_dir / "official_eval/judge_scores.jsonl"),
            "selected_count": len(ids),
        }
    protocol = {
        "manifest_sha256": sha(gov / "manifest.json"),
        "govmem_runtime_sha256": identity["runtime_sha256"],
        "baseline_provenance": baseline_provenance,
        "judge": "official GateMem GPT-4o, temperature 0, gate_by_action=false",
        "scope": "four historically exposed development episodes; frozen Gov-Mem predictions",
    }
    protocol_path = output / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError("Protocol changed")
    dump(protocol_path, protocol)
    return manifest


def summarize(output: Path, manifest: dict) -> None:
    result = {"systems": {}, "scope": "four exposed complete episodes; exploratory"}
    for system in ("govmem", "rag_naive"):
        per_domain = {}
        for domain in DOMAINS:
            directory = output / system / domain
            judged = rows(directory / "official_eval/judge_scores.jsonl")
            predictions = rows(directory / "predictions.jsonl")
            ids = {row["checkpoint_id"] for row in manifest["entries"] if row["domain"] == domain}
            if len(judged) != len(ids) or {row["checkpoint_id"] for row in judged} != ids:
                raise ValueError("Judge coverage mismatch")
            errors = {row["checkpoint_id"] for row in predictions
                      if row.get("execution_status") == "error" or row.get("action") == "error"}
            per_domain[domain] = component_metrics(judged, errors)
            per_domain[domain].update(checkpoints=len(ids), execution_errors=len(errors))
        result["systems"][system] = {"per_domain": per_domain,
            "avg_MGS": sum(per_domain[d]["MGS"] for d in DOMAINS) / 4,
            "execution_errors": sum(per_domain[d]["execution_errors"] for d in DOMAINS)}
    result["delta_MGS"] = result["systems"]["govmem"]["avg_MGS"] - result["systems"]["rag_naive"]["avg_MGS"]
    dump(output / "diagnostic_metrics.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--govmem-root", type=Path, required=True)
    parser.add_argument("--baseline-scored", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    gov, baseline, output = (p.resolve() for p in (args.govmem_root, args.baseline_scored, args.output))
    if not output.is_relative_to("/mnt/data_disk_2"):
        raise ValueError("Output must be on data_disk_2")
    manifest = prepare(gov, baseline, output)
    if args.prepare_only:
        print("Prepared 102 frozen predictions and matching saved baseline labels", flush=True)
        return
    keys = list(dict.fromkeys(_discover_api_keys(provider="openlux")))
    if len(keys) < 4:
        raise ValueError("Four distinct OpenLux keys required")
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(judge, "govmem", d, gov / "selected_dataset", output, keys[i])
                   for i, d in enumerate(DOMAINS)]
        for future in futures:
            future.result()
    summarize(output, manifest)


if __name__ == "__main__":
    main()
