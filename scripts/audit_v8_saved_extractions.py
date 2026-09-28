"""Offline replay of captured Stage-2 extraction; no model, gold or scorer."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import inspect
import json
from pathlib import Path

from gov_mem.extraction.v8_event_validator import validate_v8_extraction
from gov_mem.extraction.v8_principal_registry import V8PrincipalRegistry, V8Principal
from gov_mem.llm.json_parser import parse_json_response


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-scope", choices=["ingestion", "prompt"], default="ingestion")
    args = parser.parse_args()
    counts, reasons, rows = Counter(), Counter(), []
    digest = hashlib.sha256()
    for path in sorted(args.root.glob("*/*/raw_chat_responses/*.json")):
        data = json.loads(path.read_text())
        messages = data["request_payload"]["messages"]
        if "Stage-2 governance reasoner" not in messages[0]["content"]:
            continue
        counts["stage2_requests"] += 1
        request = json.loads(messages[-1]["content"])
        content = data["response"]["choices"][0]["message"]["content"]
        digest.update(str(path.relative_to(args.root)).encode() + b"\0" + content.encode())
        try:
            raw = parse_json_response(content)
        except (ValueError, TypeError):
            counts["unparseable_responses"] += 1
            continue
        ingestion = request.get("ingestion")
        if ingestion is None:
            counts["requests_without_ingestion"] += 1
            continue
        events = raw.get("events") if isinstance(raw, dict) else None
        fields = ("scenes", "entities", "facts", "relations", "governance_events")
        if not isinstance(events, dict) or any(not isinstance(events.get(k), list) for k in fields):
            counts["invalid_events_shape"] += 1
            continue
        registry = V8PrincipalRegistry(
            principals={r["principal_id"]: V8Principal(**r) for r in ingestion["principal_registry"]},
            static_relationships=[])
        graph = request["graph_context"]
        sources = {r["turn_id"]: r["text"] for r in ingestion["new_turns"] + ingestion["context_turns"]}
        if args.source_scope == "prompt":
            from gov_mem.governance_runtime.v8_prompt_sources import collect_prompt_sources
            sources = collect_prompt_sources(request["rag_candidates"], graph, ingestion)
        batch = validate_v8_extraction(events,
            turn_text=sources,
            registry=registry, allowed_output_turn_ids={r["turn_id"] for r in ingestion["new_turns"]},
            existing_entity_ids={r["entity_id"] for r in graph.get("entities", [])}
                | {r["scene_id"] for r in graph.get("scene_hints", [])})
        critical = [r for r in batch.rejected
                    if r.get("reason") not in {"context_only_event", "context_only_record", "context_only_scene"}
                    and (r.get("kind") == "governance_event"
                         or (r.get("kind") == "fact" and r.get("lifecycle", "assert") != "assert"))]
        counts["validated_batches"] += 1
        counts["critical_failure_batches" if critical else "accepted_batches"] += 1
        reasons.update(r["reason"] for r in critical)
        rows.append({"capture": str(path.relative_to(args.root)), "critical": critical,
                     "source_repairs": batch.audit.get("source_repairs", []),
                     "accepted_counts": batch.audit["accepted_counts"]})
    result = {"kind": "saved_stage2_extraction_replay", "paid_api_calls": 0,
              "capture_sha256": digest.hexdigest(),
              "validator_sha256": hashlib.sha256(Path(inspect.getfile(validate_v8_extraction)).read_bytes()).hexdigest(),
              "source_scope": args.source_scope, "counts": dict(counts),
              "critical_reasons": dict(reasons), "batches": rows,
              "limitations": "Fixed historical responses, including repairs. Counts are requests, not checkpoints. No MGS or semantic correctness claim."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "batches"}, indent=2))


if __name__ == "__main__":
    main()
