#!/usr/bin/env python3
"""Audit structural extraction quality and call counts for a Gov-Mem v8 run."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def audit_v8_run(output_dir: Path) -> dict[str, Any]:
    debug_files = sorted(output_dir.glob("debug_cases/*/*.json"))
    cache_files = sorted(output_dir.glob("v8_event_cache/*/*.json"))
    accepted = Counter()
    rejected = Counter()
    new_turns = 0
    extraction_calls = 0
    joint_extraction_calls = 0
    logical_chat_calls = 0
    safe_released = 0
    safe_blocked = 0
    future_excluded = 0
    for path in debug_files:
        row = _read(path)
        audit = dict(row.get("v8_extraction_audit") or {})
        new_turns += int(audit.get("new_turn_count") or 0)
        extraction_calls += int(audit.get("extraction_call_count") or 0)
        joint_extraction_calls += int(bool(audit.get("joint_extraction")))
        logical_chat_calls += int((row.get("v8_cost") or {}).get("logical_chat_calls") or 0)
        for batch in audit.get("batches") or []:
            accepted.update(dict(batch.get("accepted_counts") or {}))
            rejected.update(dict(batch.get("rejected_reasons") or {}))
        safe = dict(row.get("v8_safe_evidence") or {})
        safe_released += int(safe.get("released_claim_count") or 0)
        safe_blocked += int(safe.get("blocked_claim_count") or 0)
        projection = dict(row.get("v8_state_projection") or {})
        future_excluded += int(projection.get("future_event_count_excluded") or 0)

    latest_stores = [_read(path) for path in cache_files]
    store_totals = {
        key: sum(len(store.get(key) or []) for store in latest_stores)
        for key in ("processed_turn_ids", "scenes", "entities", "facts", "relations", "governance_events")
    }
    cross_turn_events = sum(
        len({str(span.get("turn_id") or "") for span in event.get("source_spans") or []}) > 1
        for store in latest_stores
        for event in store.get("governance_events") or []
    )
    metadata_path = output_dir / "run_metadata.json"
    metadata = _read(metadata_path) if metadata_path.exists() else {}
    telemetry = dict(metadata.get("llm_telemetry") or {})
    chat = dict(telemetry.get("chat/completions") or {})
    retries = int(chat.get("retries") or 0)
    provider_chat_calls = int(chat.get("calls") or 0) + retries
    checkpoint_count = len(debug_files)
    return {
        "schema_version": "govmem-v8-extraction-audit-1",
        "output_dir": str(output_dir),
        "checkpoint_count": checkpoint_count,
        "conversation_cache_count": len(cache_files),
        "incremental_extraction": {
            "new_turns_processed_across_checkpoints": new_turns,
            "logical_extraction_calls": extraction_calls + joint_extraction_calls,
            "standalone_prefill_calls": extraction_calls,
            "shared_stage2_extractions": joint_extraction_calls,
            "turns_per_logical_extraction_call": (
                new_turns / (extraction_calls + joint_extraction_calls) if extraction_calls + joint_extraction_calls else None
            ),
        },
        "accepted_records": dict(accepted),
        "rejected_records_by_reason": dict(rejected),
        "event_store_totals": store_totals,
        "cross_turn_governance_event_count": cross_turn_events,
        "late_governance": {
            "released_claim_count": safe_released,
            "blocked_claim_count": safe_blocked,
            "future_events_excluded_across_projections": future_excluded,
        },
        "calls": {
            "provider_chat_requests": provider_chat_calls,
            "http_retries": retries,
            "json_parse_failures": (telemetry.get("chat_json_parse_failure") or {}).get("calls", 0),
            "chat_latency_s": chat.get("elapsed_s", 0),
            "prompt_tokens": chat.get("prompt_tokens"),
            "completion_tokens": chat.get("completion_tokens"),
            "total_tokens": chat.get("total_tokens"),
            "responses_with_usage": chat.get("responses_with_usage", 0),
            "logical_chat_calls_completed_checkpoints": logical_chat_calls,
            "provider_chat_requests_per_checkpoint": (
                provider_chat_calls / checkpoint_count if checkpoint_count else None
            ),
        },
        "interpretation": {
            "scope": "structural/source-grounding audit",
            "not_measured": [
                "semantic entity recall", "relation tuple recall",
                "permission event recall", "permission scope recall",
            ],
            "required_for_semantic_f1": "episode-disjoint human extraction annotations",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--write", type=Path, default=None)
    args = parser.parse_args()
    result = audit_v8_run(args.output_dir)
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    print(rendered)
    if args.write:
        args.write.parent.mkdir(parents=True, exist_ok=True)
        args.write.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
