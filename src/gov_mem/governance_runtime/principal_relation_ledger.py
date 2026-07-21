"""Episode-local, source-grounded principal relationship resolution.

The ledger deliberately separates a person's organizational role from their
relationship to the owner of a requested record.  Roles are useful context for
policy evaluation, but never establish a relationship or access right by
themselves.  A relation is usable only when an LLM proposal is anchored to an
exact visible source span and passes closed-world validation.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from gov_mem.llm.client import LLMClient, LLMClientUnavailableError


_ACCESS_RELATIONS = {"owner", "family", "delegate", "authorized_staff"}
_KNOWN_RELATIONS = _ACCESS_RELATIONS | {"peer", "guest", "service_provider", "unknown"}
_KNOWN_STATUSES = {"proven", "unknown", "revoked", "contradicted"}


def build_principal_relation_ledger(
    *,
    messages: list[dict[str, Any]],
    requester_id: str | None,
    requester_role: str | None,
    principal_catalog: list[dict[str, Any]] | None,
    question: str,
    llm_client: LLMClient | None,
    model_name: str,
    fallback_owner_id: str | None = None,
    candidate_owner_ids: set[str] | None = None,
    relation_evidence_message_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Resolve the requester-to-owner relation using only visible episode data.

    The returned dictionary is an in-memory per-instance artifact.  It may be
    written to debug output for auditability, but it is never used across an
    episode boundary.  A missing LLM or an invalid proposal fails closed.
    """
    requester = str(requester_id or "").strip()
    roster = _build_roster(messages, principal_catalog, requester, requester_role)
    source_by_id = {
        str(message.get("message_id") or "").strip(): str(message.get("text") or "")
        for message in messages
        if str(message.get("message_id") or "").strip()
    }
    allowed_owner_ids = {
        owner_id for owner_id in set(candidate_owner_ids or set())
        if owner_id in roster
    }
    relation_evidence_ids = {
        message_id for message_id in set(relation_evidence_message_ids or set())
        if message_id in source_by_id
    }
    base = {
        "version": 1,
        "requester_id": requester or None,
        "question": str(question or ""),
        "principals": roster,
        "records": [],
        "owner_id": None,
        "effective_relation": "unknown",
        "effective_status": "unknown",
        "reason": "no_grounded_relation",
        "resolution_trace": {},
    }
    if not requester or requester not in roster:
        base["reason"] = "requester_not_in_episode_principal_roster"
        return base
    if llm_client is None or not llm_client.is_available():
        return _structural_fallback(base, requester=requester, fallback_owner_id=fallback_owner_id)

    try:
        raw = llm_client.chat_json(
            model=model_name,
            system_prompt=_SYSTEM_PROMPT,
            user_prompt=_relation_prompt(
                requester_id=requester,
                question=question,
                roster=roster,
                messages=messages,
                candidate_owner_ids=allowed_owner_ids or None,
            ),
        )
    except (LLMClientUnavailableError, Exception) as exc:
        base["reason"] = f"relation_resolution_unavailable:{type(exc).__name__}"
        return _structural_fallback(base, requester=requester, fallback_owner_id=fallback_owner_id)

    rows = _relation_rows(raw)
    validated, rejection_counts = _validate_rows(
        rows, requester, roster, source_by_id, allowed_owner_ids=allowed_owner_ids or None
    )
    repair_trace: dict[str, Any] = {"attempted": False}
    targeted_recovery_trace: dict[str, Any] = {"attempted": False}
    if (
        len(allowed_owner_ids) == 1
        and relation_evidence_ids
        and not _has_proven_access_relation(validated)
    ):
        targeted_recovery_trace["attempted"] = True
        recovery_raw = _request_targeted_relation_recovery(
            llm_client=llm_client,
            model_name=model_name,
            requester_id=requester,
            owner_id=next(iter(allowed_owner_ids)),
            question=question,
            roster=roster,
            source_by_id=source_by_id,
            evidence_message_ids=relation_evidence_ids,
        )
        recovery_rows = _relation_rows(recovery_raw)
        recovered, recovery_rejections = _validate_rows(
            recovery_rows, requester, roster, source_by_id, allowed_owner_ids=allowed_owner_ids
        )
        targeted_recovery_trace.update({
            "raw_top_level_keys": sorted(str(key) for key in recovery_raw.keys())
            if isinstance(recovery_raw, dict) else [],
            "candidate_record_count": len(recovery_rows),
            "validated_record_count": len(recovered),
            "rejection_counts": dict(sorted(recovery_rejections.items())),
            "evidence_message_ids": sorted(relation_evidence_ids),
        })
        if _has_proven_access_relation(recovered):
            validated = recovered
            rejection_counts = recovery_rejections
    if not validated and (
        rejection_counts.get("proven_relation_has_no_relation_specific_source_support")
        or rejection_counts.get("proven_relation_has_no_valid_source_support")
    ):
        repair_trace["attempted"] = True
        repair_raw = _request_relation_anchor_repair(
            llm_client=llm_client,
            model_name=model_name,
            requester_id=requester,
            question=question,
            roster=roster,
            messages=messages,
            prior_rows=rows,
            candidate_owner_ids=allowed_owner_ids or None,
        )
        repair_rows = _relation_rows(repair_raw)
        repaired, repair_rejections = _validate_rows(
            repair_rows, requester, roster, source_by_id, allowed_owner_ids=allowed_owner_ids or None
        )
        repair_trace.update({
            "raw_top_level_keys": sorted(str(key) for key in repair_raw.keys()) if isinstance(repair_raw, dict) else [],
            "candidate_record_count": len(repair_rows),
            "validated_record_count": len(repaired),
            "rejection_counts": dict(sorted(repair_rejections.items())),
        })
        if repaired:
            validated = repaired
            rejection_counts = repair_rejections
        elif not _relation_identity_candidates(rows, requester, roster):
            repair_trace["support_verification_skipped"] = "no_single_closed_relation_candidate"
        else:
            # Keep semantic resolution and evidence verification deliberately
            # separate.  The first proposal already fixed the closed IDs and
            # relation label; this pass may return only exact support spans for
            # that same tuple, never a new owner or access relationship.
            candidates = _relation_identity_candidates(
                rows, requester, roster, allowed_owner_ids=allowed_owner_ids or None
            )
            if len(candidates) == 1:
                candidate = candidates[0]
                support_raw = _request_relation_support_verification(
                    llm_client=llm_client,
                    model_name=model_name,
                    requester_id=requester,
                    question=question,
                    roster=roster,
                    messages=messages,
                    candidate=candidate,
                    candidate_owner_ids=allowed_owner_ids or None,
                )
                supports = _support_rows_for_candidate(support_raw, candidate)
                repaired_row = dict(candidate)
                repaired_row["supports"] = supports
                verified, verification_rejections = _validate_rows(
                    [repaired_row], requester, roster, source_by_id,
                    allowed_owner_ids=allowed_owner_ids or None,
                )
                repair_trace["support_verification"] = {
                    "attempted": True,
                    "raw_top_level_keys": sorted(str(key) for key in support_raw.keys())
                    if isinstance(support_raw, dict) else [],
                    "candidate_support_count": len(supports),
                    "validated_record_count": len(verified),
                    "rejection_counts": dict(sorted(verification_rejections.items())),
                }
                if verified:
                    validated = verified
                    rejection_counts = verification_rejections
    base["resolution_trace"] = {
        "raw_top_level_keys": sorted(str(key) for key in raw.keys()) if isinstance(raw, dict) else [],
        "candidate_record_count": len(rows),
        "validated_record_count": len(validated),
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "relation_anchor_repair": repair_trace,
        "targeted_relation_recovery": targeted_recovery_trace,
        "candidate_owner_ids": sorted(allowed_owner_ids),
    }
    base["records"] = validated
    resolution = _resolve_effective_relation(validated, requester)
    base.update(resolution)
    return base


def ledger_relation_for_principal(ledger: dict[str, Any]) -> str:
    """Return an authorization-relevant relation, failing closed by default."""
    relation = str(ledger.get("effective_relation") or "unknown")
    status = str(ledger.get("effective_status") or "unknown")
    return relation if status == "proven" and relation in _ACCESS_RELATIONS else "unknown"


def _has_proven_access_relation(records: list[dict[str, Any]]) -> bool:
    return any(
        str(record.get("status") or "") == "proven"
        and str(record.get("relation") or "") in _ACCESS_RELATIONS
        for record in records
        if isinstance(record, dict)
    )


def _build_roster(
    messages: list[dict[str, Any]],
    principal_catalog: list[dict[str, Any]] | None,
    requester_id: str,
    requester_role: str | None,
) -> dict[str, dict[str, str]]:
    roster: dict[str, dict[str, str]] = {}
    for row in list(principal_catalog or []):
        principal_id = str(row.get("principal_id") or row.get("id") or "").strip()
        if principal_id:
            roster[principal_id] = {
                "principal_id": principal_id,
                "display_name": str(row.get("display_name") or "").strip(),
                "role": str(row.get("role") or "").strip(),
            }
    for message in messages:
        principal_id = str(message.get("speaker_id") or "").strip()
        if not principal_id:
            continue
        current = roster.setdefault(principal_id, {"principal_id": principal_id, "display_name": "", "role": ""})
        if not current["role"]:
            current["role"] = str(message.get("speaker_role") or "").strip()
    if requester_id:
        current = roster.setdefault(requester_id, {"principal_id": requester_id, "display_name": "", "role": ""})
        if not current["role"]:
            current["role"] = str(requester_role or "").strip()
    return roster


def _relation_rows(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, dict):
        return []
    for key in (
        "relations",
        "requester_owner_relations",
        "relationship_records",
        "relation_records",
        "records",
    ):
        rows = raw.get(key)
        if isinstance(rows, list):
            return [dict(row) for row in rows if isinstance(row, dict)]
    for key in ("requester_owner_relation", "relationship", "relation_record"):
        row = raw.get(key)
        if isinstance(row, dict):
            return [dict(row)]
    return []


def _validate_rows(
    rows: list[dict[str, Any]],
    requester: str,
    roster: dict[str, dict[str, str]],
    source_by_id: dict[str, str],
    *,
    allowed_owner_ids: set[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    validated: list[dict[str, Any]] = []
    rejection_counts: dict[str, int] = defaultdict(int)
    for row in rows:
        record, rejection = _validate_record(
            row, requester, roster, source_by_id, allowed_owner_ids=allowed_owner_ids
        )
        if record is not None:
            validated.append(record)
        elif rejection:
            rejection_counts[rejection] += 1
    return validated, dict(rejection_counts)


def _request_targeted_relation_recovery(
    *,
    llm_client: LLMClient,
    model_name: str,
    requester_id: str,
    owner_id: str,
    question: str,
    roster: dict[str, dict[str, str]],
    source_by_id: dict[str, str],
    evidence_message_ids: set[str],
) -> dict[str, Any]:
    """Resolve one fixed requester-owner pair from selected factual evidence."""
    evidence = [
        {"message_id": message_id, "text": source_by_id[message_id]}
        for message_id in sorted(evidence_message_ids)
    ]
    try:
        raw = llm_client.chat_json(
            model=model_name,
            system_prompt=(
                "You verify an episode-local requester-to-owner relation for governed reranking. The requester "
                "and owner IDs are fixed closed-set values. The supplied source turns were selected as factual "
                "evidence about that owner's record; this does not itself grant access. Return a proven relation "
                "only if an exact supplied span establishes the requester's family relationship, delegation, or "
                "case assignment/authorization for that owner. Do not infer from a job title, identifier, or the "
                "fact value. Return JSON only."
            ),
            user_prompt=(
                "Return {\"relations\":[{\"subject_id\":string,\"owner_id\":string,\"relation\":"
                "\"family|delegate|authorized_staff|unknown\",\"relation_label\":string,\"status\":"
                "\"proven|unknown\",\"authorization_status\":string,\"direction\":"
                "\"requester_to_owner\",\"supports\":list}]}. Use exactly requester_id and owner_id below. "
                "Every support must copy a message_id and source_span from Selected evidence. Return one unknown "
                "row with supports=[] when no relation is explicit.\n"
                f"Requester ID: {requester_id}\nOwner ID: {owner_id}\nQuestion: {question}\n"
                f"Principal roster: {list(roster.values())}\nSelected evidence: {evidence}"
            ),
        )
        return raw if isinstance(raw, dict) else {}
    except (LLMClientUnavailableError, Exception):
        return {}


def _request_relation_anchor_repair(
    *,
    llm_client: LLMClient,
    model_name: str,
    requester_id: str,
    question: str,
    roster: dict[str, dict[str, str]],
    messages: list[dict[str, Any]],
    prior_rows: list[dict[str, Any]],
    candidate_owner_ids: set[str] | None,
) -> dict[str, Any]:
    try:
        raw = llm_client.chat_json(
            model=model_name,
            system_prompt=(
                "You repair source-grounded requester-to-owner relationship evidence. Return JSON only. "
                "Do not infer a family, staff, or delegate relationship from a title, identifier, name, or "
                "authorization sentence alone."
            ),
            user_prompt=(
                _relation_prompt(
                    requester_id=requester_id,
                    question=question,
                    roster=roster,
                    messages=messages,
                    candidate_owner_ids=candidate_owner_ids,
                )
                + "\nA previous proposal cited authorization but not the relationship itself. For family, cite an "
                "explicit_relationship span describing the family relationship. For authorized_staff, cite an "
                "explicit_assignment or explicit_authorization tying that exact person to the owner. If such a "
                "span is absent, return relation=unknown with status=unknown."
                + f"\nKeep requester_id, owner_id, relation, and direction from this prior closed-set proposal; "
                f"only repair source supports: {prior_rows}"
            ),
        )
        return raw if isinstance(raw, dict) else {}
    except (LLMClientUnavailableError, Exception):
        return {}


def _relation_identity_candidates(
    rows: list[dict[str, Any]],
    requester: str,
    roster: dict[str, dict[str, str]],
    *,
    allowed_owner_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Return one closed relation tuple eligible for support-only verification.

    This intentionally does not inspect transcript language.  It merely
    validates the identifiers and relation category that the first model has
    already proposed, so a verifier cannot broaden access by changing them.
    """
    candidates: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    for row in rows:
        subject_id = str(
            row.get("subject_id") or row.get("requester_id") or row.get("subject_principal_id")
            or row.get("requester_principal_id") or requester
        ).strip()
        owner_id = str(
            row.get("owner_id") or row.get("target_owner_id") or row.get("owner_principal_id")
            or row.get("target_principal_id") or row.get("object_id") or row.get("target_id") or ""
        ).strip()
        relation = _canonical_relation(
            row.get("relation") or row.get("relation_type") or row.get("canonical_relation")
        )
        status = str(row.get("status") or row.get("relation_status") or "unknown").strip().lower()
        direction = _canonical_direction(row.get("direction"))
        if (
            subject_id != requester
            or subject_id not in roster
            or owner_id not in roster
            or (allowed_owner_ids is not None and owner_id not in allowed_owner_ids)
            or relation not in _ACCESS_RELATIONS - {"owner"}
            or status != "proven"
            or direction != "requester_to_owner"
        ):
            continue
        key = (subject_id, owner_id, relation, status, direction)
        candidates.setdefault(key, {
            "subject_id": subject_id,
            "owner_id": owner_id,
            "relation": relation,
            "relation_label": str(row.get("relation_label") or row.get("relationship") or relation).strip(),
            "status": status,
            "authorization_status": str(
                row.get("authorization_status") or row.get("access_status") or "unknown"
            ).strip().lower(),
            "direction": direction,
        })
    return list(candidates.values())


def _request_relation_support_verification(
    *,
    llm_client: LLMClient,
    model_name: str,
    requester_id: str,
    question: str,
    roster: dict[str, dict[str, str]],
    messages: list[dict[str, Any]],
    candidate: dict[str, Any],
    candidate_owner_ids: set[str] | None,
) -> dict[str, Any]:
    try:
        raw = llm_client.chat_json(
            model=model_name,
            system_prompt=(
                "You are an evidence verifier. Return JSON only. You may not change or infer people, "
                "relationship category, direction, permission, or owner. Select an exact visible transcript "
                "span only when it explicitly supports the supplied fixed relationship tuple."
            ),
            user_prompt=(
                _relation_prompt(
                    requester_id=requester_id,
                    question=question,
                    roster=roster,
                    messages=messages,
                    candidate_owner_ids=candidate_owner_ids,
                )
                + "\nFixed candidate (do not repeat or alter its fields): " + repr(candidate)
                + "\nReturn only {\"supports\":[{\"message_id\":...,\"source_span\":...,"
                "\"evidence_kind\":...}]}. evidence_kind is exactly explicit_relationship, "
                "explicit_assignment, explicit_authorization, or explicit_revocation. Return supports=[] "
                "if no exact relation-specific span exists."
            ),
        )
        return raw if isinstance(raw, dict) else {}
    except (LLMClientUnavailableError, Exception):
        return {}


def _support_rows_for_candidate(raw: Any, candidate: dict[str, Any]) -> list[dict[str, Any]]:
    """Accept support-only output plus provider envelopes that repeat the fixed tuple."""
    if not isinstance(raw, dict):
        return []
    direct = raw.get("supports") or raw.get("evidence") or raw.get("evidence_spans")
    if direct is not None:
        return direct if isinstance(direct, list) else [direct] if isinstance(direct, dict) else []
    for row in _relation_rows(raw):
        if (
            str(row.get("subject_id") or row.get("requester_id") or "").strip() == candidate["subject_id"]
            and str(row.get("owner_id") or row.get("target_owner_id") or "").strip() == candidate["owner_id"]
            and _canonical_relation(row.get("relation") or row.get("relation_type")) == candidate["relation"]
        ):
            value = row.get("supports") or row.get("evidence") or row.get("evidence_spans")
            return value if isinstance(value, list) else [value] if isinstance(value, dict) else []
    return []


def _validate_record(
    row: dict[str, Any],
    requester: str,
    roster: dict[str, dict[str, str]],
    source_by_id: dict[str, str],
    *,
    allowed_owner_ids: set[str] | None = None,
) -> tuple[dict[str, Any] | None, str | None]:
    subject_id = str(
        row.get("subject_id") or row.get("requester_id") or row.get("subject_principal_id")
        or row.get("requester_principal_id") or requester
    ).strip()
    owner_id = str(
        row.get("owner_id") or row.get("target_owner_id") or row.get("owner_principal_id")
        or row.get("target_principal_id") or row.get("object_id") or row.get("target_id") or ""
    ).strip()
    relation = _canonical_relation(row.get("relation") or row.get("relation_type") or row.get("canonical_relation"))
    status = str(row.get("status") or row.get("relation_status") or "unknown").strip().lower()
    direction = _canonical_direction(row.get("direction"))
    if subject_id != requester:
        return None, "requester_id_not_closed_or_mismatched"
    if not owner_id or owner_id not in roster:
        return None, "owner_id_not_in_closed_roster"
    if allowed_owner_ids is not None and owner_id not in allowed_owner_ids:
        return None, "owner_id_not_in_fact_owner_closure"
    if subject_id not in roster:
        return None, "requester_id_not_in_closed_roster"
    if relation not in _KNOWN_RELATIONS:
        return None, "relation_not_in_closed_schema"
    if status not in _KNOWN_STATUSES:
        return None, "status_not_in_closed_schema"
    if direction != "requester_to_owner":
        return None, "direction_not_requester_to_owner"

    supports = _validated_supports(
        row.get("supports") or row.get("evidence") or row.get("evidence_spans"), source_by_id
    )
    if relation == "owner" and subject_id == owner_id:
        # Equality of two closed-world principal IDs is structural evidence.
        supports = supports or [{"message_id": None, "source_span": "", "evidence_kind": "self_identity"}]
    if status == "proven" and not supports:
        return None, "proven_relation_has_no_valid_source_support"
    if status == "proven" and relation != "owner" and not _has_relation_anchor(relation, supports):
        return None, "proven_relation_has_no_relation_specific_source_support"
    if status in {"revoked", "contradicted"} and not supports:
        return None, "terminal_relation_has_no_valid_source_support"
    return {
        "requester_id": requester,
        "owner_id": owner_id,
        "relation": relation,
        "relation_label": str(row.get("relation_label") or row.get("relationship") or relation).strip(),
        "status": status,
        "authorization_status": str(row.get("authorization_status") or row.get("access_status") or "unknown").strip().lower(),
        "direction": "requester_to_owner",
        "supports": supports,
    }, None


def _validated_supports(value: Any, source_by_id: dict[str, str]) -> list[dict[str, Any]]:
    supports = value if isinstance(value, list) else [value] if isinstance(value, dict) else []
    valid: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for support in supports:
        if not isinstance(support, dict):
            continue
        message_id = str(
            support.get("message_id") or support.get("turn_id") or support.get("source_message_id") or ""
        ).strip()
        span = str(
            support.get("source_span") or support.get("support_span") or support.get("span")
            or support.get("evidence_span") or support.get("verbatim_span") or support.get("quote") or ""
        ).strip()
        source = source_by_id.get(message_id, "")
        if not message_id or not span or span not in source:
            continue
        key = (message_id, span)
        if key in seen:
            continue
        seen.add(key)
        valid.append({
            "message_id": message_id,
            "source_span": span,
            "evidence_kind": _canonical_evidence_kind(str(
                support.get("evidence_kind") or support.get("evidence_type") or support.get("kind") or "unspecified"
            )),
        })
    return valid


def _canonical_evidence_kind(value: str) -> str:
    """Normalize JSON schema labels without inspecting source vocabulary."""
    normalized = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "relationship": "explicit_relationship",
        "family_relationship": "explicit_relationship",
        "familial_relationship": "explicit_relationship",
        "relation_statement": "explicit_relationship",
        "authorization": "explicit_authorization",
        "permission": "explicit_authorization",
        "consent": "explicit_authorization",
        "assignment": "explicit_assignment",
        "staff_assignment": "explicit_assignment",
        "revocation": "explicit_revocation",
    }
    return aliases.get(normalized, normalized or "unspecified")


def _has_relation_anchor(relation: str, supports: list[dict[str, Any]]) -> bool:
    """Require a relation-specific evidence type, not merely any nearby fact."""
    allowed_kinds = {
        "family": {"explicit_relationship"},
        "delegate": {"explicit_relationship", "explicit_authorization"},
        "authorized_staff": {"explicit_assignment", "explicit_authorization"},
        "peer": {"explicit_relationship"},
        "guest": {"explicit_relationship", "explicit_authorization"},
        "service_provider": {"explicit_assignment", "explicit_relationship"},
    }
    return any(
        str(support.get("evidence_kind") or "").strip().lower() in allowed_kinds.get(relation, set())
        for support in supports
    )


def _canonical_relation(value: Any) -> str:
    """Normalize only schema-level aliases emitted by JSON-capable providers.

    This adapter never examines transcript words and therefore cannot create a
    relationship from a role/name trigger. The semantic model still has to
    select the category from source-grounded evidence.
    """
    normalized = str(value or "unknown").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "self": "owner",
        "self_owner": "owner",
        "family_member": "family",
        "relative": "family",
        "authorized_staff_member": "authorized_staff",
        "staff": "authorized_staff",
        "delegated": "delegate",
    }
    return aliases.get(normalized, normalized)


def _canonical_direction(value: Any) -> str:
    normalized = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "requester_to_owner": "requester_to_owner",
        "requester__to__owner": "requester_to_owner",
        "from_requester_to_owner": "requester_to_owner",
        "requester_to_target_owner": "requester_to_owner",
        "subject_to_owner": "requester_to_owner",
    }
    return aliases.get(normalized, normalized)


def _resolve_effective_relation(records: list[dict[str, Any]], requester: str) -> dict[str, Any]:
    by_owner: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_owner[str(record["owner_id"])].append(record)
    candidates: list[tuple[tuple[int, int, str], str, dict[str, Any]]] = []
    priority = {"owner": 0, "family": 1, "delegate": 2, "authorized_staff": 3}
    for owner_id, rows in by_owner.items():
        proven = [row for row in rows if row["status"] == "proven"]
        blocked = [row for row in rows if row["status"] in {"revoked", "contradicted"}]
        relations = {str(row["relation"]) for row in proven}
        access_relations = relations & _ACCESS_RELATIONS
        if blocked and not proven:
            continue
        if len(access_relations) != 1:
            # Competing active access identities are not disambiguated by a
            # role priority: an explicit source-grounded resolution is needed.
            continue
        relation = next(iter(access_relations))
        support_count = sum(len(row["supports"]) for row in proven)
        candidates.append(((priority[relation], -support_count, owner_id), owner_id, proven[-1]))
    if not candidates:
        return {
            "owner_id": None,
            "effective_relation": "unknown",
            "effective_status": "unknown",
            "reason": "no_unambiguous_proven_requester_owner_relation",
        }
    # Self identity is structurally true for every requester, but it is not
    # evidence that the requested record belongs to that requester. When the
    # relation resolver also provides a single source-grounded non-self owner,
    # that latter tuple is the only candidate capable of identifying the
    # protected record. Keeping self at higher priority otherwise collapses a
    # staff/delegate request into an unrelated personal record.
    non_self_candidates = [
        candidate for candidate in candidates
        if candidate[1] != requester and str(candidate[2].get("relation") or "") != "owner"
    ]
    if non_self_candidates:
        candidates = non_self_candidates
    candidates.sort(key=lambda item: item[0])
    if len(candidates) > 1 and candidates[0][0][:2] == candidates[1][0][:2]:
        return {
            "owner_id": None,
            "effective_relation": "unknown",
            "effective_status": "contradicted",
            "reason": "multiple_equally_supported_owner_candidates",
        }
    _, owner_id, record = candidates[0]
    return {
        "owner_id": owner_id,
        "effective_relation": record["relation"],
        "effective_status": "proven",
        "reason": "source_grounded_requester_owner_relation",
    }


def _structural_fallback(base: dict[str, Any], *, requester: str, fallback_owner_id: str | None) -> dict[str, Any]:
    """No source-grounded resolver means no authorization relation."""
    # `fallback_owner_id` is retained for call compatibility, but the legacy
    # owner candidate may be derived from an identifier convention. It is not
    # a verified owner assertion and must never establish self access.
    del requester, fallback_owner_id
    base["reason"] = base["reason"] + ";fail_closed_without_relation_llm"
    return base


_SYSTEM_PROMPT = """You resolve identity and relationship evidence for a privacy-preserving memory system.
Use only the supplied visible episode transcript and closed principal roster. Organizational titles, ID prefixes, surnames, and a person merely appearing in a conversation do NOT prove a relationship or permission. Resolve the requester only against the person whose protected information the question targets. Extract an exact verbatim support span from a supplied message for each non-self relation. Keep relation direction requester_to_owner.

Allowed relation values: owner, family, delegate, authorized_staff, peer, guest, service_provider, unknown. Use owner only for the requester themself. Use authorized_staff only when transcript evidence ties that specific person to that specific owner's care/case/work; do not infer it from a job title. A relationship can remain proven while its permission is revoked; record that under authorization_status. Do not invent people, IDs, spans, or policy rights.

For every support, evidence_kind MUST be exactly one of: explicit_relationship, explicit_assignment, explicit_authorization, explicit_revocation. A family relation requires an explicit_relationship span describing that family relationship. An authorized_staff relation requires explicit_assignment or explicit_authorization for that exact owner. Appointment content, a requester asking a question, a title, or a name is not relation evidence. Return JSON only."""


def _relation_prompt(
    *,
    requester_id: str,
    question: str,
    roster: dict[str, dict[str, str]],
    messages: list[dict[str, Any]],
    candidate_owner_ids: set[str] | None = None,
) -> str:
    roster_rows = list(roster.values())
    transcript = [
        {
            "message_id": str(message.get("message_id") or ""),
            "speaker_id": str(message.get("speaker_id") or ""),
            "text": str(message.get("text") or ""),
        }
        for message in messages
    ]
    return (
        "Requester ID: " + requester_id + "\n"
        "Question: " + str(question or "") + "\n"
        "Fact-owner candidate IDs: " + repr(sorted(candidate_owner_ids or set())) + "\n"
        "Principal roster: " + repr(roster_rows) + "\n"
        "Visible transcript: " + repr(transcript) + "\n\n"
        "Return {\"relations\":[...]}. Each relation must contain subject_id, owner_id, relation, relation_label, status, authorization_status, direction, and supports. supports is a list of {message_id, source_span, evidence_kind}. Include only requester_to_owner records relevant to the question. When Fact-owner candidate IDs is nonempty, owner_id must be one of those IDs. If the requester is the owner, emit owner with an empty supports list. When no relation is established, emit one unknown record with status unknown and supports []."
    )
