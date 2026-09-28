"""Scene-conditioned, query-independent event extraction for Gov-Mem v8."""

from __future__ import annotations

import json
import re
from hashlib import sha1
from typing import Any

from gov_mem.data.schema import MemoryInstance
from gov_mem.extraction.v8_event_schema import V8ExtractionBatch, V8Fact, V8SourceSpan
from gov_mem.extraction.v8_event_validator import validate_v8_extraction
from gov_mem.extraction.v8_principal_registry import V8PrincipalRegistry
from gov_mem.extraction.v8_scene_schema import v8_scene_schema
from gov_mem.governance_runtime.leakage_guard import assert_runtime_payload_safe
from gov_mem.llm.client import LLMClient


V8_EVENT_EXTRACTION_SYSTEM_PROMPT = """You extract append-only memory events for Gov-Mem v8.
Return JSON only. Work from the supplied observable conversation turns, domain
schema, principal roster, and prior scene registry. This is ingestion, not
question answering: extract useful entity, relation, permission,
and lifecycle change whether or not a current query mentions it. Raw facts
remain in dense memory; do not duplicate every ordinary fact in the graph.
For a lifecycle FACT, use the exact source sentence as value when a change is
expressed in words. Do not convert natural-language status into true/false or
repeat an old deleted value absent from the cited sentence. A deletion command
is itself the memory event; its quote need not repeat the deleted contents.
For explicit operational duties use relation_type=operational_duty, source_id
of the principal, target_id of the resource, and attributes containing role,
action, resource_category, subject_id. Cite the visible duty language. Scene
membership, relationship, and job title alone never establish field access.

Principal IDs are closed-set. Select issuer, grantee, speaker, and participant
IDs only from PRINCIPAL_REGISTRY. Resolve pronouns and descriptions through the
supplied context, but never invent a person or infer permission merely from a
role or relationship. A static family, employment, care-team, or project
relationship is evidence about identity, not automatic authorization.

Resources, scenes, and information fields are episode-local open vocabulary.
Create stable descriptive IDs and copy their canonical surfaces from the
turns. Reuse IDs from EXISTING_ENTITY_REGISTRY when a new turn refers to the
same entity or resource. Keep similarly named scenes separate when the text
distinguishes them.

Permission extraction must preserve positive and negative scope separately.
Extract allow, deny, revoke, and require_permission. Preserve temporary and
conditional language. A single event may cite multiple source_spans across
turns when one turn introduces a subject/resource and another completes the
permission. Every source span must be an exact contiguous substring of its
turn. At least one cited span must come from NEW_TURNS; CONTEXT_TURNS exist
only for coreference and scene continuity.

First identify the episode-local scenes represented in NEW_TURNS. A scene is
the stable activity/case/project/visit/appointment thread that gives entities
and permissions their meaning. Reuse an EXISTING_SCENE_REGISTRY scene_id when
the thread continues; create a new descriptive scene_id when it does not.
Do not collapse two similarly named but textually distinct threads.

Return exactly this top-level shape:
{"scenes":[],"entities":[],"facts":[],"relations":[],"governance_events":[]}

Scene: {"scene_id":"...","scene_type":"open vocabulary",
"canonical_name":"verbatim scene surface","participant_principal_ids":[],
"status":"active|completed|canceled|unknown",
"source_spans":[{"turn_id":"...","span":"verbatim"}],"confidence":0.0}

Entity: {"entity_id":"...","entity_type":"...","canonical_name":"verbatim",
"aliases":[],"scene_id":null,"source_spans":[{"turn_id":"...","span":"verbatim"}],"confidence":0.0}
Fact: {"fact_id":"...","scene_id":null,"subject_id":null,"field_name":"open vocabulary",
"value":"verbatim value","temporal_state":"current|historical|unknown",
"lifecycle":"assert|update|supersede|cancel|delete|unknown",
"source_spans":[{"turn_id":"...","span":"verbatim"}],"confidence":0.0}
Relation: {"relation_id":"...","relation_type":"...","source_id":"known id",
"target_id":"known id","scene_id":null,"attributes":{},
"source_spans":[{"turn_id":"...","span":"verbatim"}],"confidence":0.0}
Governance event: {"event_id":"...","event_type":"permission|lifecycle",
"effect":"allow|deny|revoke|require_permission|assert|update|supersede|cancel|delete|unknown",
"issuer_principal_id":null,"grantee_principal_id":null,"action":"access|receive|share|disclose|use|delete",
"resource_id":"episode-local id","resource_surface":"verbatim resource phrase","scene_id":null,
"included_scopes":[],"excluded_scopes":[],"condition":null,
"valid_from_turn_id":"...","valid_until":null,
"source_spans":[{"turn_id":"...","span":"verbatim"}],"confidence":0.0}
"""


_COMPLETED_DELETION_RE = re.compile(
    r"\b(?:deleted|cleared|removed|purged|erased|forgotten)\b"
    r"[^.?!]{0,160}\b(?:memory|record|entry|entries|note|notes|field|fields|file|files)\b",
    flags=re.IGNORECASE,
)
_PLANNED_DELETION_RE = re.compile(
    r"\b(?:will|shall|would|plan(?:ned)?\s+to|once|after)\b"
    r"[^.?!]{0,100}\b(?:delete|deleted|clear|cleared|remove|removed|purge|purged|erase|erased|forget|forgotten)\b",
    flags=re.IGNORECASE,
)
_DIRECT_DELETION_RE = re.compile(
    r"(?:\bdeletion request\b[^.?!]{0,220}\b(?:delete|remove|clear|purge|erase|forget)\b|"
    r"\b(?:delete|remove|clear|purge|erase|forget)\b[^.?!]{0,220})"
    r"[^.?!]{0,220}\b(?:memory|record|summary|answer|answers)\b",
    flags=re.IGNORECASE,
)


def _append_completed_lifecycle_tombstones(
    batch: V8ExtractionBatch,
    new_messages: list[dict[str, Any]],
) -> int:
    """Backstop only explicit completed deletion, never requests or plans."""

    existing_turns = {
        span.turn_id
        for fact in batch.facts
        if fact.lifecycle == "delete"
        for span in fact.source_spans
    }
    added = 0
    for message in new_messages:
        turn_id = str(message.get("turn_id") or message.get("message_id") or "")
        text = str(message.get("text") or "").strip()
        if (
            not turn_id
            or turn_id in existing_turns
            or not (_COMPLETED_DELETION_RE.search(text) or _DIRECT_DELETION_RE.search(text))
            or _PLANNED_DELETION_RE.search(text)
        ):
            continue
        target_span = _deletion_target_span(text)
        digest = sha1(f"{turn_id}||{target_span}".encode("utf-8")).hexdigest()[:12]
        batch.facts.append(V8Fact(
            fact_id=f"lifecycle_tombstone_{digest}",
            scene_id=None,
            subject_id=None,
            field_name="completed deletion",
            value=target_span,
            temporal_state="historical",
            lifecycle="delete",
            source_spans=[V8SourceSpan(turn_id=turn_id, span=target_span)],
            confidence=1.0,
        ))
        existing_turns.add(turn_id)
        added += 1
    return added


def _deletion_target_span(text: str) -> str:
    """Narrow a direct deletion instruction to the deleted object only."""

    completed = re.search(
        r"^(.*?)\s+\b(?:deleted|cleared|removed|purged|erased|forgotten)\b\s+"
        r"(?:from|out of)\s+(?:the\s+)?(?:shared\s+)?(?:assistant\s+)?"
        r"(?:memory|record|summary|answers?)\b",
        text,
        re.IGNORECASE,
    )
    if completed and completed.group(1).strip():
        return completed.group(1).strip(" :-,\t")
    command = re.search(r"\b(?:delete|remove|clear|purge|erase|forget)\b", text, re.IGNORECASE)
    if not command:
        return text
    tail = text[command.end():]
    quoted = re.search(r"['\"]([^'\"]+)['\"]", tail)
    if quoted:
        return quoted.group(1).strip()
    boundary = re.search(
        r"\b(?:from|out of)\s+(?:the\s+)?(?:shared\s+)?(?:assistant\s+)?"
        r"(?:memory|record|summary|answers?)\b",
        tail,
        re.IGNORECASE,
    )
    candidate = tail[:boundary.start()] if boundary else tail
    candidate = re.split(r"[.;]", candidate, maxsplit=1)[0]
    return candidate.strip(" :-,\t") or text


def _turn_payload(message: dict[str, Any]) -> dict[str, Any]:
    return {
        "turn_id": str(message.get("turn_id") or message.get("message_id") or ""),
        "timestamp": message.get("timestamp"),
        "speaker_principal_id": message.get("speaker_id"),
        "speaker_role": message.get("speaker_role"),
        "turn_kind": message.get("turn_kind"),
        "text": str(message.get("text") or ""),
    }


def extract_v8_events(
    *,
    instance: MemoryInstance,
    new_messages: list[dict[str, Any]],
    context_messages: list[dict[str, Any]],
    registry: V8PrincipalRegistry,
    existing_scenes: list[dict[str, Any]],
    existing_entities: list[dict[str, Any]],
    llm_client: LLMClient,
    model_name: str,
    response_protocol: str = "json",
    memory_mode: str = "full",
    resource_registry: list[dict[str, Any]] | None = None,
) -> V8ExtractionBatch:
    if not new_messages:
        return V8ExtractionBatch(audit={"llm_calls": 0, "reason": "no_new_turns"})
    payload = {
        "domain_schema": v8_scene_schema(instance.domain),
        "principal_registry": registry.candidate_payload(),
        "static_identity_relationships": registry.static_relationships,
        "existing_scene_registry": existing_scenes,
        "existing_entity_registry": existing_entities,
        "context_turns": [_turn_payload(item) for item in context_messages],
        "new_turns": [_turn_payload(item) for item in new_messages],
    }
    if memory_mode == "shallow":
        payload = {key: payload[key] for key in ("principal_registry", "context_turns", "new_turns")}
        payload["resource_registry"] = resource_registry or []
    assert_runtime_payload_safe(payload, context="v8_event_extraction")
    try:
        if response_protocol == "lines":
            from gov_mem.extraction.v8_line_protocol import LINE_CONTRACT, parse_lines
            prompt = V8_EVENT_EXTRACTION_SYSTEM_PROMPT.split("Return exactly this top-level shape:")[0].replace(
                "Return JSON only.", "Return plain text record lines only.")
            prompt += LINE_CONTRACT + "\nExtraction only: emit graph records, EVENTS, ACTION\\tno_memory, END; no SLOT or CLAIM."
            if memory_mode == "shallow":
                from gov_mem.governance_runtime.v8_shallow_memory import SHALLOW_EXTRACTION_PROMPT, SHALLOW_LINE_CONTRACT
                prompt = SHALLOW_EXTRACTION_PROMPT + SHALLOW_LINE_CONTRACT
                prompt += "\nExtraction only: EVENT/SOURCE lines, EVENTS, ACTION\tno_memory, END; no SLOT/CLAIM/KEEP."
            text = llm_client.chat_text(model=model_name, system_prompt=prompt,
                                       user_prompt=json.dumps(payload, ensure_ascii=False))
            parsed = parse_lines(text, ingestion=True, shallow=memory_mode == "shallow")
            if parsed['claims'] or parsed['query_slots']:
                raise ValueError('Prefill cannot emit query claims')
            raw = parsed['events']
        else:
            prompt = V8_EVENT_EXTRACTION_SYSTEM_PROMPT
            if memory_mode == "shallow":
                from gov_mem.governance_runtime.v8_shallow_memory import (
                    SHALLOW_EXTRACTION_PROMPT, SHALLOW_JSON_CONTRACT, normalize_shallow_json)
                prompt = SHALLOW_EXTRACTION_PROMPT + SHALLOW_JSON_CONTRACT
                prompt += '\nExtraction only: return {"events":[...]} with no query_slots or claims.'
            raw = llm_client.chat_json(
                model=model_name, system_prompt=prompt,
                user_prompt=json.dumps(payload, ensure_ascii=False),
            )
            if memory_mode == "shallow":
                raw = normalize_shallow_json(raw, ingestion=True, extraction_only=True)["events"]
    except Exception as exc:
        return V8ExtractionBatch(
            rejected=[{"kind": "batch", "reason": type(exc).__name__}],
            audit={"llm_calls": 1, "parse_failure": True, "error": type(exc).__name__},
        )
    if not isinstance(raw, dict) or any(not isinstance(raw.get(k), list) for k in ("entities", "facts", "relations", "governance_events")):
        return V8ExtractionBatch(audit={"llm_calls": 1, "parse_failure": True, "error": "invalid_event_contract"})
    turn_text = {
        str(item.get("turn_id") or item.get("message_id") or ""): str(item.get("text") or "")
        for item in [*context_messages, *new_messages]
    }
    if memory_mode == "shallow":
        from gov_mem.governance_runtime.v8_prompt_sources import collect_prompt_sources
        turn_text = collect_prompt_sources([], {"resource_registry": resource_registry or []},
                                          {"new_turns": payload["new_turns"], "context_turns": payload["context_turns"]})
    result = validate_v8_extraction(
        raw,
        turn_text=turn_text,
        registry=registry,
        allowed_output_turn_ids={
            str(item.get("turn_id") or item.get("message_id") or "") for item in new_messages
        },
        existing_entity_ids={
            str(item.get("entity_id") or "")
            for item in existing_entities
            if isinstance(item, dict) and item.get("entity_id")
        } | {
            str(item.get("scene_id") or "")
            for item in existing_scenes
            if isinstance(item, dict) and item.get("scene_id")
        },
    )
    tombstones_added = 0 if memory_mode == "shallow" else _append_completed_lifecycle_tombstones(result, new_messages)
    result.audit.update({"memory_mode": memory_mode, "llm_calls": 1, "model": model_name, "new_turn_count": len(new_messages)})
    result.audit["deterministic_completed_deletion_tombstones_added"] = tombstones_added
    result.audit["accepted_counts"]["facts"] = len(result.facts)
    return result
