"""Only explicitly released excerpts cross the final-answer boundary."""
from __future__ import annotations
from typing import Any
from gov_mem.data.schema import RetrievedEvidence


def build_v8_safe_evidence(claim_ledger: dict[str, Any]) -> tuple[list[RetrievedEvidence], dict[str, Any]]:
    safe, blocked = [], []
    slots = {s.get("slot") for s in claim_ledger.get("query_slots", []) if isinstance(s, dict)}
    blocked_values = [str(c.get("value")) for c in claim_ledger.get("claims", [])
                      if c.get("delivery") == "block" and c.get("value")]
    denied = [c for c in claim_ledger.get("claims", []) if c.get("delivery") == "block"]
    requested_released = 0
    for i, claim in enumerate(claim_ledger.get("claims", [])):
        requested = claim.get("requested", claim.get("slot") in slots if slots else True)
        if claim.get("delivery") == "block":
            # Neither free-text reasons nor denied values/source quotes go to
            # Stage 3. Even a model's slot label could contain the denied value.
            slot = str(claim.get("slot") or "requested information")
            for value in blocked_values:
                slot = slot.replace(value, "[withheld]")
            blocked.append({"claim_id": claim.get("claim_id"), "slot": slot,
                            "requested": bool(requested), "reason": "withheld",
                            "symbolic_veto_reasons": claim.get("symbolic_veto_reasons", [])})
            continue
        value = str(claim.get("value") or "").strip()
        if not value:
            continue
        # A released excerpt must not wrap a denied span of the SAME source.
        # Offsets, not subject identity or token overlap, establish containment.
        start, end = claim.get("value_start"), claim.get("value_end")
        ranges = []
        if isinstance(start, int) and isinstance(end, int):
            for other in denied:
                left, right = other.get("value_start"), other.get("value_end")
                if (other.get("source_memory_id") == claim.get("source_memory_id")
                        and isinstance(left, int) and isinstance(right, int)
                        and max(start, left) < min(end, right)):
                    ranges.append((max(start, left) - start, min(end, right) - start))
        if ranges:
            merged = []
            for left, right in sorted(ranges):
                if merged and left <= merged[-1][1]:
                    merged[-1] = (merged[-1][0], max(right, merged[-1][1]))
                else:
                    merged.append((left, right))
            for left, right in reversed(merged):
                value = value[:left] + "[withheld]" + value[right:]
            slot = str(claim.get("slot") or "requested information")
            for denied_value in blocked_values:
                slot = slot.replace(denied_value, "[withheld]")
            blocked.append({"claim_id": claim.get("claim_id"), "slot": slot,
                            "requested": bool(requested), "reason": "withheld",
                            "symbolic_veto_reasons": ["overlapping_restricted_source_span"]})
            if not value.replace("[withheld]", "").strip():
                continue
        requested_released += int(bool(requested))
        safe_slot = str(claim.get("slot") or "supporting context")
        for denied_value in blocked_values:
            safe_slot = safe_slot.replace(denied_value, "[withheld]")
        safe.append(RetrievedEvidence(
            memory_id=f"v8_safe_claim::{claim.get('claim_id') or i}",
            content=f"{safe_slot}: {value}", score=1.0,
            retrieval_source="v8_safe_claim_ledger", reason="language-reviewed release",
            source_message_ids=list(claim.get("source_message_ids") or []),
            metadata={"approved_value": value, "claim_id": claim.get("claim_id"), "source_memory_id": claim.get("source_memory_id"),
                      "slot": safe_slot, "delivery": claim.get("delivery"), "requested": bool(requested),
                      "source_time": claim.get("source_time"), "temporal_state": claim.get("temporal_state"), "authorization": claim.get("authorization"),
                      "symbolic_advisories": claim.get("symbolic_advisories", [])},
        ))
    blocked_requested = sum(b["requested"] for b in blocked)
    # Background context is not a partial answer if every requested field is blocked.
    if blocked_requested and not requested_released:
        safe = []
    return safe, {"schema_version": "govmem-v8-safe-evidence-2", "released_claim_count": len(safe),
                  "released_requested_count": requested_released, "blocked_requested_count": blocked_requested,
                  "blocked_claim_count": len(blocked), "blocked_claims": [b for b in blocked if b["requested"]],
                  "released_claim_ids": [r.metadata["claim_id"] for r in safe],
                  "claim_ledger_action": claim_ledger.get("answer_action")}
