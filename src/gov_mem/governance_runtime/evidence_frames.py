"""Source-grounded evidence-frame adapter.

The paper-facing Gov-Mem-v4 path does not classify evidence with a natural
language keyword/regex vocabulary. Query-conditioned semantic atoms are the
only semantic input. This module keeps the small ``EvidenceFrame`` adapter
used by older callers and by the governed-slot graph, while refusing to infer
fields from raw evidence text when no semantic contract is attached.
"""

from __future__ import annotations

from dataclasses import asdict
from hashlib import md5
from typing import Any

from gov_mem.data.schema import EvidenceFrame, RetrievedEvidence
from gov_mem.llm.client import LLMClient, LLMClientUnavailableError
from gov_mem.llm.model_registry import resolve_llm_model
from gov_mem.llm.prompts import build_answering_user_prompt


# Gov-Mem-v4 deliberately uses one content-neutral evidence-frame kind. The
# governed slot graph obtains content semantics only from verified,
# open-vocabulary compiler atoms below, rather than an event/domain taxonomy.
FRAME_TYPES = frozenset({"general_fact"})


def _record(row: RetrievedEvidence) -> dict[str, Any]:
    value = (row.metadata or {}).get("structured_record")
    return dict(value) if isinstance(value, dict) else {}


def _compiler_atoms(row: RetrievedEvidence) -> list[dict[str, Any]]:
    return [
        dict(atom)
        for atom in ((row.metadata or {}).get("semantic_compiler_atoms") or [])
        if isinstance(atom, dict)
    ]


def _grounded_surface(text: str, value: Any) -> str | None:
    candidate = " ".join(str(value or "").split()).strip()
    if not candidate:
        return None
    if candidate.casefold() in " ".join(text.split()).casefold():
        return candidate
    return None


def compile_evidence_frames(evidence: list[RetrievedEvidence]) -> list[EvidenceFrame]:
    return [compile_evidence_frame(row) for row in evidence]


def compile_evidence_frame(row: RetrievedEvidence) -> EvidenceFrame:
    metadata = dict(row.metadata or {})
    record = _record(row)
    atoms = _compiler_atoms(row)
    compiler_contract = bool(metadata.get("semantic_compiler_contract"))

    # A semantic compiler contract makes the raw text opaque. No fallback
    # parser may recover a field, sensitivity class, lifecycle cue, or domain
    # phrase from the source sentence.
    frame_type = str(metadata.get("frame_type") or "general_fact")
    if frame_type not in FRAME_TYPES:
        frame_type = "general_fact"

    slots: dict[str, Any] = {}
    semantic_attributes: dict[str, Any] = {}
    surface_spans: dict[str, str] = {}
    if atoms:
        frame_type = "general_fact" if compiler_contract else frame_type
        source_text = str(record.get("text") or row.content or "")
        for atom in atoms:
            name = str(atom.get("slot_name") or "").strip()
            if not name:
                continue
            key = name.casefold().replace(" ", "_")
            value = atom.get("value")
            if value not in (None, ""):
                slots.setdefault(key, value)
            source = dict(atom.get("source") or {})
            span = _grounded_surface(source_text, source.get("span"))
            semantic_attributes[key] = {
                "value": value,
                "slot_id": atom.get("slot_id"),
                "source": source,
                "temporal": dict(atom.get("temporal") or {}),
                "authorization_semantics": dict(atom.get("authorization_semantics") or {}),
                "lifecycle_semantics": dict(atom.get("lifecycle_semantics") or {}),
                "sensitivity_semantics": dict(atom.get("sensitivity_semantics") or {}),
                "confidence": atom.get("confidence"),
            }
            if span:
                surface_spans[key] = span

    # Observable structured annotations may be carried by a non-v4 caller.
    # They are accepted as already-typed metadata, never inferred here.
    semantic_tags = dict(metadata.get("semantic_tags") or {})
    event_identity = dict(semantic_tags.get("event_identity") or {})
    typed_attributes = semantic_tags.get("attributes")
    if not compiler_contract and isinstance(typed_attributes, dict):
        source_text = str(record.get("text") or row.content or "")
        for key, value in typed_attributes.items():
            if value in (None, "", []):
                continue
            name = str(key)
            semantic_attributes.setdefault(name, value)
            slots.setdefault(name, value)
            surface = dict(semantic_tags.get("surface_values") or {}).get(name)
            grounded = _grounded_surface(source_text, surface)
            if grounded:
                surface_spans[name] = grounded

    sensitivity = {
        "privacy_level": metadata.get("privacy_level"),
        "redaction_required": bool(metadata.get("redaction_required") or metadata.get("requires_redaction")),
        "sensitive_entities": list(metadata.get("sensitive_entities") or []),
    }
    access_scope = {
        "authorized_users": list(metadata.get("authorized_users") or []),
        "forbidden_users": list(metadata.get("forbidden_users") or []),
        "access_scope": metadata.get("access_scope"),
    }
    state_delta = dict(semantic_tags.get("state_delta") or {})
    return EvidenceFrame(
        frame_id=md5(f"{row.memory_id}:{frame_type}:{row.content}".encode("utf-8")).hexdigest()[:12],
        memory_id=row.memory_id,
        source_message_ids=list(row.source_message_ids),
        frame_type=frame_type,
        owner_user=row.user_id,
        subject_entity=str(event_identity.get("entity_key") or "") or None,
        lifecycle_status=str(metadata.get("memory_status") or "active"),
        effective_time=row.time,
        event_time=(slots.get("date") or row.time),
        slots=slots,
        access_scope=access_scope,
        sensitivity=sensitivity,
        surface_spans=surface_spans,
        confidence=float(row.score),
        discourse_act=str(semantic_tags.get("discourse_act") or "unknown"),
        assertion_confidence=float(semantic_tags.get("assertion_confidence") or 0.0),
        event_identity=event_identity,
        state_delta=state_delta,
        semantic_attributes=semantic_attributes,
        provenance={
            "memory_id": row.memory_id,
            "source_message_ids": list(row.source_message_ids),
            "retrieval_source": row.retrieval_source,
        },
    )


def normalize_frames_with_llm(
    memory_item,
    initial_frames: list[EvidenceFrame],
    llm_client: LLMClient | None,
    config: dict | None,
) -> list[EvidenceFrame]:
    """Optionally add source-grounded typed frames for legacy callers.

    This is an explicit LLM path, not a deterministic parser. The active v4
    backbone uses semantic_compiler atoms before this adapter and does not need
    this compatibility normalizer.
    """
    if llm_client is None or not llm_client.is_available():
        return initial_frames
    runtime_cfg = dict((config or {}).get("governance_runtime") or {})
    if not bool(runtime_cfg.get("use_llm_frame_normalizer", True)):
        return initial_frames
    text = str(getattr(memory_item, "content", "") or "")
    system_prompt = (
        "Extract typed source-grounded event frames from the supplied memory text. "
        "Return JSON only. Do not answer a user question, infer hidden labels, "
        "authorize disclosure, or invent values. Preserve exact source spans."
    )
    user_prompt = build_answering_user_prompt(
        question=text,
        asking_user_id=None,
        choices=None,
        selected_evidence=[{
            "memory_id": getattr(memory_item, "memory_id", None),
            "content": text,
        }],
        reasoning_trace=[f"Initial frames: {len(initial_frames)}"],
        conclusion_hint="Frame normalization only.",
        skill_text="",
        retrieved_lessons=[],
    )
    try:
        raw = llm_client.chat_json(
            model=resolve_llm_model(config or {}, "memory_ingestion"),
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
    except (LLMClientUnavailableError, Exception):
        return initial_frames
    if not isinstance(raw, dict) or not isinstance(raw.get("frames"), list):
        return initial_frames
    extra: list[EvidenceFrame] = []
    for index, item in enumerate(raw["frames"]):
        if not isinstance(item, dict):
            continue
        frame_type = str(item.get("frame_type") or "general_fact")
        if frame_type not in FRAME_TYPES:
            frame_type = "general_fact"
        source_text = str(item.get("source_text") or "")
        if source_text and _grounded_surface(text, source_text) is None:
            continue
        try:
            extra.append(EvidenceFrame(
                frame_id=str(item.get("frame_id") or f"llm_{index}"),
                memory_id=str(getattr(memory_item, "memory_id", "")),
                source_message_ids=list(getattr(memory_item, "source_message_ids", []) or []),
                frame_type=frame_type,
                owner_user=getattr(memory_item, "user_id", None),
                subject_entity=str(item.get("subject_entity") or "") or None,
                lifecycle_status=str(item.get("lifecycle_status") or "active"),
                effective_time=item.get("effective_time"),
                event_time=item.get("event_time"),
                slots=dict(item.get("slots") or {}),
                access_scope=dict(item.get("access_scope") or {}),
                sensitivity=dict(item.get("sensitivity") or {}),
                surface_spans=dict(item.get("surface_spans") or {}),
                confidence=float(item.get("confidence", 0.5)),
                discourse_act=str(item.get("discourse_act") or "unknown"),
                assertion_confidence=float(item.get("assertion_confidence") or 0.0),
                event_identity=dict(item.get("event_identity") or {}),
                state_delta=dict(item.get("state_delta") or {}),
                semantic_attributes=dict(item.get("semantic_attributes") or item.get("attributes") or {}),
                provenance=dict(item.get("provenance") or {}),
            ))
        except (TypeError, ValueError):
            continue
    return initial_frames + extra


def frame_to_dict(frame: EvidenceFrame) -> dict[str, Any]:
    return asdict(frame)
