"""Episode-local attribution of a fact's protected-information owner.

The speaker who reports a fact and the principal the fact is about are often
different. This ledger is deliberately independent from access authorization:
it only assigns a closed-set information owner when an LLM supplies exact
visible evidence. Unknown attribution remains unusable for owner-scoped graph
certification.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from gov_mem.governance_runtime.principal_relation_ledger import _build_roster, _validated_supports
from gov_mem.llm.client import LLMClient, LLMClientUnavailableError


def build_information_owner_ledger(
    *,
    messages: list[dict[str, Any]],
    principal_catalog: list[dict[str, Any]] | None,
    candidate_message_ids: set[str],
    required_message_ids: set[str] | None = None,
    target_owner_id: str | None = None,
    llm_client: LLMClient | None,
    model_name: str,
) -> dict[str, Any]:
    """Attribute selected source turns to principals without role heuristics."""
    roster = _build_roster(messages, principal_catalog, requester_id="", requester_role=None)
    source_by_id = {
        str(message.get("message_id") or "").strip(): str(message.get("text") or "")
        for message in messages if str(message.get("message_id") or "").strip()
    }
    candidate_ids = {message_id for message_id in candidate_message_ids if message_id in source_by_id}
    base: dict[str, Any] = {
        "version": 1,
        "candidate_message_ids": sorted(candidate_ids),
        "records": [],
        "owner_by_message_id": {},
        "resolution_trace": {},
    }
    if not candidate_ids or not roster or llm_client is None or not llm_client.is_available():
        base["resolution_trace"] = {"reason": "no_candidates_or_llm_unavailable"}
        return base
    initial_error = ""
    try:
        raw = llm_client.chat_json(
            model=model_name,
            system_prompt=(
                "You attribute the protected information in an observed source turn to a principal in a "
                "privacy-preserving memory system. The speaker is provenance, not automatically the information "
                "owner. Select only a supplied closed-roster principal and copy exact support spans from supplied "
                "visible messages. Information owner means the principal whose record, schedule, state, or private "
                "attribute is being described, never the staff member merely reporting it. Do not use job titles, "
                "identifier patterns, surnames, or unstated assumptions. "
                "Return unknown when the data subject cannot be grounded. This task does not grant access. Return JSON only."
            ),
            user_prompt=_owner_prompt(
                messages=messages, source_by_id=source_by_id, roster=roster,
                candidate_ids=candidate_ids, target_owner_id=target_owner_id,
            ),
        )
    except (LLMClientUnavailableError, Exception) as exc:
        # A batch attribution request can fail transiently while the same
        # provider remains available for a smaller closed-source request. Do
        # not turn that transport failure into an empty owner ledger before
        # attempting the already fail-closed per-message repair below.
        raw = {}
        initial_error = f"information_owner_unavailable:{type(exc).__name__}"

    rows = _owner_rows(raw)
    validated: list[dict[str, Any]] = []
    rejection_counts: dict[str, int] = defaultdict(int)
    for row in rows:
        record, reason = _validate_owner_row(row, candidate_ids, roster, source_by_id)
        if record is None:
            rejection_counts[reason] += 1
        else:
            validated.append(record)
    required_ids = {str(value) for value in list(required_message_ids or []) if str(value)} & candidate_ids
    unresolved_required_ids = required_ids - {str(record["message_id"]) for record in validated}
    repair_trace: dict[str, Any] = {"attempted": False}
    if unresolved_required_ids:
        repair_trace["attempted"] = True
        repair_rejections: dict[str, int] = defaultdict(int)
        repair_attempts: list[dict[str, Any]] = []
        # Source ownership is a per-turn claim. Requesting a large batch lets
        # a model omit precisely the factual turn that later carries a graph
        # slot. Each repair is closed to one already selected source ID and
        # still requires transcript support for its chosen owner. A second
        # independent attempt handles an empty structured response without
        # ever falling back to speaker, role, or target-owner guessing.
        for message_id in sorted(unresolved_required_ids):
            accepted_count = 0
            attempt_count = 0
            candidate_count = 0
            for _ in range(2):
                attempt_count += 1
                repair_raw = _request_single_owner_repair(
                    llm_client=llm_client,
                    model_name=model_name,
                    messages=messages,
                    source_by_id=source_by_id,
                    roster=roster,
                    message_id=message_id,
                    target_owner_id=target_owner_id,
                )
                repair_rows = _owner_rows(repair_raw)
                candidate_count += len(repair_rows)
                for row in repair_rows:
                    record, reason = _validate_owner_row(row, {message_id}, roster, source_by_id)
                    if record is None:
                        repair_rejections[reason] += 1
                    else:
                        validated.append(record)
                        accepted_count += 1
                if accepted_count:
                    break
            repair_attempts.append({
                "message_id": message_id,
                "attempt_count": attempt_count,
                "candidate_record_count": candidate_count,
                "validated_record_count": accepted_count,
            })
        repair_trace.update({
            "per_message_attempts": repair_attempts,
            "candidate_record_count": sum(item["candidate_record_count"] for item in repair_attempts),
            "validated_record_count": len(validated),
            "rejection_counts": dict(sorted(repair_rejections.items())),
        })
    owner_by_message_id = _resolve_unique_owners(validated)
    base.update({
        "records": validated,
        "owner_by_message_id": owner_by_message_id,
        "resolution_trace": {
            **({"initial_request_error": initial_error} if initial_error else {}),
            "raw_top_level_keys": sorted(str(key) for key in raw.keys()) if isinstance(raw, dict) else [],
            "candidate_record_count": len(rows),
            "validated_record_count": len(validated),
            "resolved_message_count": len(owner_by_message_id),
            "rejection_counts": dict(sorted(rejection_counts.items())),
            "priority_owner_repair": repair_trace,
        },
    })
    return base


def _owner_prompt(
    *,
    messages: list[dict[str, Any]],
    source_by_id: dict[str, str],
    roster: dict[str, dict[str, str]],
    candidate_ids: set[str],
    target_owner_id: str | None,
) -> str:
    transcript = [
        {"message_id": message_id, "speaker_id": str(message.get("speaker_id") or ""), "text": source_by_id[message_id]}
        for message_id, message in [(str(row.get("message_id") or "").strip(), row) for row in messages]
        if message_id
    ]
    return (
        "Return {\"information_owners\":[{\"message_id\":string,\"information_owner_id\":string,"
        "\"status\":\"proven|unknown\",\"supports\":[{\"message_id\":string,\"source_span\":string,"
        "\"evidence_kind\":\"explicit_data_subject|contextual_case_assignment|self_assertion\"}]}]}. "
        "Return one row only for each listed candidate source message whose information owner is grounded. "
        "A support may cite another visible turn only when it explicitly establishes the case/data subject "
        "for the candidate turn.\n"
        f"Closed principal roster: {list(roster.values())}\n"
        f"Question-target protected owner (may be used only with source support): {str(target_owner_id or '')}\n"
        f"Candidate source message IDs: {sorted(candidate_ids)}\n"
        f"Visible transcript: {transcript}"
    )


def _request_single_owner_repair(
    *,
    llm_client: LLMClient,
    model_name: str,
    messages: list[dict[str, Any]],
    source_by_id: dict[str, str],
    roster: dict[str, dict[str, str]],
    message_id: str,
    target_owner_id: str | None,
) -> dict[str, Any]:
    try:
        raw = llm_client.chat_json(
            model=model_name,
            system_prompt=(
                "You repair one source-grounded information-owner attribution. Return JSON only. The candidate "
                "message ID is fixed; do not emit any other message ID. Select a closed-roster owner only when an "
                "exact visible span establishes whose protected record the message describes. The question target, "
                "a speaker, title, or identifier is not sufficient by itself. Return an empty list when unsupported."
            ),
            user_prompt=_owner_prompt(
                messages=messages,
                source_by_id=source_by_id,
                roster=roster,
                candidate_ids={message_id},
                target_owner_id=target_owner_id,
            ) + "\nReturn either one proven row for this exact candidate or information_owners=[].",
        )
        return raw if isinstance(raw, dict) else {}
    except (LLMClientUnavailableError, Exception):
        return {}


def _owner_rows(raw: object) -> list[dict[str, Any]]:
    if not isinstance(raw, dict):
        return []
    for container in (raw, raw.get("result"), raw.get("data")):
        if not isinstance(container, dict):
            continue
        for key in ("information_owners", "fact_owners", "ownership_records", "records"):
            rows = container.get(key)
            if isinstance(rows, list):
                return [dict(row) for row in rows if isinstance(row, dict)]
    return []


def _validate_owner_row(
    row: dict[str, Any],
    candidate_ids: set[str],
    roster: dict[str, dict[str, str]],
    source_by_id: dict[str, str],
) -> tuple[dict[str, Any] | None, str]:
    message_id = str(row.get("message_id") or row.get("source_message_id") or "").strip()
    owner_id = str(row.get("information_owner_id") or row.get("owner_id") or row.get("principal_id") or "").strip()
    status = str(row.get("status") or "unknown").strip().lower()
    if message_id not in candidate_ids:
        return None, "message_id_not_in_closed_candidate_set"
    if owner_id not in roster:
        return None, "owner_id_not_in_closed_roster"
    if status != "proven":
        return None, "owner_status_not_proven"
    supports = _validated_supports(row.get("supports") or row.get("evidence"), source_by_id)
    allowed_kinds = {"explicit_data_subject", "contextual_case_assignment", "self_assertion"}
    if not supports or not any(str(item.get("evidence_kind") or "").lower() in allowed_kinds for item in supports):
        return None, "missing_information_owner_support"
    return {
        "message_id": message_id,
        "information_owner_id": owner_id,
        "status": "proven",
        "supports": supports,
    }, ""


def _resolve_unique_owners(records: list[dict[str, Any]]) -> dict[str, str]:
    by_message: dict[str, set[str]] = defaultdict(set)
    for record in records:
        by_message[str(record["message_id"])].add(str(record["information_owner_id"]))
    return {
        message_id: next(iter(owner_ids))
        for message_id, owner_ids in by_message.items()
        if len(owner_ids) == 1
    }
