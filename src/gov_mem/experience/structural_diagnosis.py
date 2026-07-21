"""Post-hoc, dev-only diagnosis from runtime traces rather than case wording."""

from __future__ import annotations

from typing import Any


def diagnose_runtime_trace(trace: dict[str, Any]) -> dict[str, Any]:
    """Classify an observed failure boundary without reading scorer labels.

    The caller may use an official score to decide *which* dev cases failed,
    but this function only consumes runtime observability emitted before any
    evaluation. Its categories are deliberately structural and reusable.
    """
    trace = dict(trace or {})
    semantic_spec = dict((trace.get("query_plan") or {}).get("semantic_spec") or {})
    requested = _requested_attributes(semantic_spec)
    coverage = dict(trace.get("slot_coverage") or {})
    selected_evidence = list(trace.get("selected_evidence") or [])
    dual = trace.get("dual_channel_retrieval")
    graph_certificate = dict(trace.get("graph_authorization_certificate") or {})
    final_realization = dict(trace.get("final_realization") or {})

    category = "unclassified_runtime_trace"
    evidence: list[str] = []
    if not semantic_spec:
        category = "semantic_contract_missing"
        evidence.append("planner_emitted_no_semantic_spec")
    elif not selected_evidence:
        category = "utility_retrieval_miss"
        evidence.append("selected_evidence_empty")
    elif requested and list(coverage.get("missing_slots") or []):
        category = "typed_evidence_coverage_miss"
        evidence.append("missing_slots=" + ",".join(sorted(map(str, coverage.get("missing_slots") or []))))
    elif dual is not None:
        utility = dict((dual or {}).get("utility_evidence") or {})
        governance = dict((dual or {}).get("governance_evidence") or {})
        if requested and not list(utility.get("selected_atom_ids") or []):
            category = "utility_channel_selection_miss"
            evidence.append("utility_selected_atom_ids_empty")
        elif requested and not list(governance.get("selected_policy_atom_ids") or []):
            category = "governance_binding_selection_miss"
            evidence.append("governance_selected_policy_atom_ids_empty")
        elif requested and not bool(graph_certificate.get("authorized")):
            category = "cross_channel_certificate_miss"
            evidence.append("certificate_reason=" + str(graph_certificate.get("reason") or "unknown"))
    if (
        category == "unclassified_runtime_trace"
        and requested
        and not bool(graph_certificate.get("authorized"))
    ):
        # All temporal modes now use the same graph certificate; retain its
        # auditable reason instead of collapsing this into retrieval failure.
        category = "graph_certificate_unavailable"
        evidence.append("certificate_reason=" + str(graph_certificate.get("reason") or "unknown"))
    if category == "unclassified_runtime_trace" and final_realization:
        verifier = dict(final_realization.get("verifier_result") or final_realization.get("post_verifier") or {})
        if verifier and not bool(verifier.get("passed", True)):
            category = "realization_verifier_rejection"
            evidence.append("realization_verifier_failed")
    return {
        "category": category,
        "requested_attributes": requested,
        "selected_evidence_count": len(selected_evidence),
        "dual_channel_observed": dual is not None,
        "graph_certificate_reason": str(graph_certificate.get("reason") or ""),
        "evidence": evidence,
    }


def _requested_attributes(semantic_spec: dict[str, Any]) -> list[str]:
    return list(dict.fromkeys(
        str(item).strip()
        for item in (
            semantic_spec.get("requested_attributes")
            or semantic_spec.get("requested_slots")
            or []
        )
        if str(item).strip()
    ))
