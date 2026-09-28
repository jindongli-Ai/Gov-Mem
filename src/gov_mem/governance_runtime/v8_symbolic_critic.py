"""Late symbolic verification after LLM scope matching, without lexical vetoes."""
from __future__ import annotations
from typing import Any


def criticize_v8_claims(claim_ledger: dict[str, Any], *, state_projection: dict[str, Any],
                       requester_principal_id: str | None,
                       graph_context: dict[str, Any] | None = None) -> dict[str, Any]:
    permissions = {p.get("event_id"): p for p in state_projection.get("current_permissions", [])
                   if p.get("event_id") and p.get("grantee_principal_id") == requester_principal_id}
    lifecycle = {p.get("event_id"): p for p in state_projection.get("lifecycle_state", [])
                 if p.get("event_id")}
    reviewed, vetoes = [], []
    for raw in claim_ledger.get("claims", []):
        item = dict(raw)
        reasons = []
        # The LLM resolves natural-language scope and cites the applicable event.
        # Symbolic enforcement requires an exact resource/action/scene binding
        # and an active, unconditional whole-resource denial. Ambiguous scopes,
        # expiry conditions and competing issuers stay with language reasoning.
        matched = [permissions[e] for e in item.get("permission_event_ids", []) if e in permissions]
        for policy in matched:
            conflict = any(p.get("decision") == "allow" and p.get("activation") != "expired"
                           and p.get("resource_id") == policy.get("resource_id")
                           and p.get("action") == policy.get("action")
                           and p.get("scene_id") == policy.get("scene_id")
                           for p in permissions.values())
            if (policy.get("decision") == "deny" and policy.get("activation") == "active"
                    and policy.get("source_spans") and not policy.get("condition")
                    and not policy.get("included_scopes") and not policy.get("excluded_scopes")
                    and item.get("resource_id") and item["resource_id"] == policy.get("resource_id")
                    and item.get("action") == policy.get("action")
                    and item.get("scene_id") == policy.get("scene_id") and not conflict):
                reasons.append("bound_explicit_permission_veto")
        for event_id in item.get("permission_event_ids", []):
            event = lifecycle.get(event_id)
            if (event and event.get("effect") == "delete" and event.get("activation") == "active"
                    and event.get("source_spans") and not event.get("condition")
                    and not event.get("included_scopes") and not event.get("excluded_scopes")
                    and event.get("grantee_principal_id") in {None, requester_principal_id}
                    and item.get("resource_id") and item["resource_id"] == event.get("resource_id")
                    and item.get("scene_id") == event.get("scene_id")
                    and item.get("action") == (event.get("action") or "access")):
                # The explicit binding says this exact claim falls under this
                # deletion event, not merely the same topic. A newer value may
                # use the same resource ID but must not select the old delete.
                # Updates/cancellations, text overlap and unbound neighbors do
                # not create deletions.
                reasons.append("bound_lifecycle_deletion_veto")
        if (item.get("delivery") == "block" or item.get("temporal_state") == "deleted") and item.get("restriction_kind") == "deleted" and item.get("restriction_evidence"):
            reasons.append("source_grounded_deletion_tombstone")
        # No value-substring tombstone matching, scene-membership authorization,
        # absent-edge veto, or propagation to other fields of the same subject.
        if reasons:
            item["delivery"] = "block"
            item["symbolic_veto_reasons"] = list(dict.fromkeys(reasons))
            vetoes.append({"claim_id": item.get("claim_id"), "reasons": item["symbolic_veto_reasons"]})
        reviewed.append(item)
    return {**claim_ledger, "claims": reviewed, "symbolic_critic": {
        "mode": "language_scope_symbolic_binding", "veto_count": len(vetoes),
        "vetoes": vetoes, "unknown_is_denial": False}}
