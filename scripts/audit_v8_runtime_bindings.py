"""Audit actual V8 decisions and their original model proposals, offline."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from gov_mem.llm.json_parser import parse_json_response
from gov_mem.governance_runtime.v8_prompt_sources import collect_prompt_sources


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    captures = {}
    counts, errors, models = Counter(), Counter(), Counter()
    for path in sorted((args.root/"govmem").glob("*/*/raw_chat_responses/*.json")):
        data = json.loads(path.read_text())
        request = data["request_payload"]
        models[request.get("model")] += 1
        messages = request.get("messages", [])
        if not messages or "Stage-2 governance reasoner" not in messages[0]["content"]:
            continue
        counts["stage2_responses"] += 1
        try:
            raw = parse_json_response(data["response"]["choices"][0]["message"]["content"])
        except (ValueError, TypeError, KeyError):
            counts["unparseable_stage2_captures"] += 1
            continue
        key = hashlib.sha256((messages[0]["content"]+"\0"+messages[-1]["content"]).encode()).hexdigest()
        captures[key] = raw
    for path in (args.root/"govmem").glob("*/*/predictions/checkpoint_benchmark/predictions.jsonl"):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            counts["checkpoints"] += 1
            if row.get("execution_status") == "error":
                counts["execution_errors"] += 1
    per_domain = {}
    for path in (args.root/"govmem").glob("*/*/predictions/checkpoint_benchmark/*.json"):
        record = json.loads(path.read_text())
        raw_response = record.get("raw_response") or {}
        ledger = raw_response.get("v8_claim_ledger") or {}
        if not ledger:
            continue
        counts["normal_checkpoints"] += 1
        domain = path.relative_to(args.root/"govmem").parts[0]
        local = per_domain.setdefault(domain, Counter())
        local["normal_checkpoints"] += 1
        audit = ledger.get("audit") or {}
        counts["contract_repair_calls"] += audit.get("contract_repair_calls", 0)
        prompt = ledger.get("prompt_audit") or {}
        key = hashlib.sha256((prompt.get("system_prompt", "")+"\0"+prompt.get("user_prompt", "")).encode()).hexdigest()
        proposal = captures.get(key, {})
        payload = json.loads(prompt.get("user_prompt", "{}"))
        candidates = {c["candidate_id"]: c for c in payload.get("rag_candidates", [])}
        if payload.get("candidate_id_style") == "source":
            # Source-only selections can refer to exact quotations already in
            # graph/ingestion. Match the same unambiguous visible view as the
            # runtime, without loading a fuller bank record for this audit.
            retrieved_turns = {t for c in candidates.values() for t in c.get("source_message_ids", [])}
            sources = collect_prompt_sources(list(candidates.values()), payload.get("graph_context", {}), payload.get("ingestion"))
            for turn_id, snippets in sources.items():
                if turn_id in candidates or turn_id in retrieved_turns or not snippets:
                    continue
                text = max(snippets, key=len)
                if all(s in text for s in snippets):
                    candidates[turn_id] = {"text": text}
        has_binding = False
        for claim in ledger.get("claims") or []:
            counts["claims"] += 1
            if claim.get("permission_event_ids"):
                counts["bound_claims"] += 1
                local["bound_claims"] += 1
                has_binding = True
            reasons = claim.get("symbolic_veto_reasons") or []
            permission_veto = "bound_explicit_permission_veto" in reasons
            deletion_veto = "bound_lifecycle_deletion_veto" in reasons
            deletion_classification = "source_grounded_deletion_tombstone" in reasons
            if not (permission_veto or deletion_veto or deletion_classification):
                continue
            if permission_veto:
                counts["bound_permission_vetoes"] += 1
                local["bound_permission_vetoes"] += 1
            if deletion_veto:
                counts["bound_deletion_vetoes"] += 1
                local["bound_deletion_vetoes"] += 1
            originals = []
            for item in proposal.get("claims") or []:
                if item.get("candidate_id") != claim.get("candidate_id"):
                    continue
                value = candidates.get(item.get("candidate_id"), {}).get("text") if item.get("keep") is True else item.get("value") or item.get("quote")
                if not value and item.get("delivery") == "block":
                    value = candidates.get(item.get("candidate_id"), {}).get("text")
                if isinstance(value, str) and value.strip() == claim.get("value"):
                    originals.append("exact" if item.get("keep") is True else item.get("delivery"))
            if originals and all(v in {"exact", "summary"} for v in originals):
                counts["independent_symbolic_release_to_block"] += 1
                local["independent_symbolic_release_to_block"] += 1
                if deletion_veto or deletion_classification:
                    counts["independent_deletion_release_to_block"] += 1
            elif originals and all(v == "block" for v in originals):
                counts["symbolic_confirms_language_block"] += 1
                if deletion_classification:
                    counts["language_deletion_confirmations"] += 1
            else:
                counts["symbolic_origin_unresolved"] += 1
        counts["checkpoints_with_binding"] += int(has_binding)
    for path in (args.root/"govmem").glob("*/*/execution_errors.jsonl"):
        errors.update(json.loads(line)["error"] for line in path.read_text().splitlines())
    result = {"runtime_sha256": json.loads((args.root/"execution_identity.json").read_text())["runtime_sha256"],
              "counts": dict(counts), "domains": {d: dict(c) for d,c in per_domain.items()},
              "errors": dict(errors), "captured_memory_models": dict(models),
              "interpretation": "Independent veto counts require matching captured model proposals. No MGS or semantic-correctness claim."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
