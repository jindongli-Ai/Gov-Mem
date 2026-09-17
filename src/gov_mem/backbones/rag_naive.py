from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

# Reuse the released GateMem prompt/domain helpers without requiring callers to
# set PYTHONPATH manually. The suite runner stages this tree separately.
_configured_bench_root = os.environ.get("GOVMEM_OFFICIAL_BENCHMARK_ROOT", "").strip()
OFFICIAL_BENCH_ROOT = (
    Path(_configured_bench_root)
    if _configured_bench_root
    else Path(__file__).resolve().parents[3] / "third_party" / "GateMem-official"
)
if str(OFFICIAL_BENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(OFFICIAL_BENCH_ROOT))

from gov_mem.backbones.v4_support import (
    BackboneRunResult,
    RAGChunk,
    build_reasoning_state,
    chunk_to_memory_item as _chunk_to_memory_item,
    save_rag_chunks,
)
from gov_mem.data.schema import (
    AnswerResult,
    GovernedActionDecision,
    MemoryInstance,
    QueryPlan,
    RetrievedEvidence,
)
from gov_mem.policy_schema import PolicyAction, PolicyDecision
from gov_mem.policy_verifier import verify_policy_delivery
from gov_mem.data.timestamps import normalize_message_timestamp, normalize_timestamp
from gov_mem.llm.client import LLMClient, LLMClientUnavailableError
from gov_mem.llm.model_registry import resolve_llm_model
from gov_mem.reasoning.operators import build_required_slot_plan
from gov_mem.memory.dense_index import DenseMemoryIndex
from gov_mem.memory.governed_slot_graph import (
    build_memory_governed_slot_graph,
    graph_retrieval_evidence,
)
from gov_mem.backbones.stage2_typed_rerank import (
    Stage2Decision,
    deletion_gate_reason,
    explicit_sensitive_boundary_reason,
    project_mixed_current_state_evidence,
    reason_mixed_evidence_with_llm,
    rerank_typed_scalar_evidence,
    build_summary_only_evidence,
    summary_only_boundary_reason,
    _query_safety_profile,
    _safety_requests_protected_fact,
    analyze_query_with_llm,
    _generic_fallback_query_analysis,
)
from gov_mem.backbones.symbolic_evidence import (
    build_symbolic_evidence,
)
from gov_mem.backbones.semantic_compiler import compile_semantics
from gov_mem.governance_runtime.evidence_frames import compile_evidence_frames
from bench.domains import format_relationship_fact


OFFICIAL_QUERY_PROMPT = OFFICIAL_BENCH_ROOT / "bench" / "prompts" / "query_prompt.txt"

# The released benchmark prompt contains placeholders for its domain-specific
# policy appendix. The paper-facing Gov-Mem path deliberately supplies only a
# neutral policy contract; benchmark domain labels and policy blocks are not
# runtime inputs to the model.
NEUTRAL_DOMAIN_LABEL = "unspecified"
NEUTRAL_ACCESS_POLICY = (
    "- Use only source-grounded evidence and the symbolic governance context.\n"
    "- Apply explicit allow, deny, revoke, lifecycle, and scope constraints.\n"
    "- Do not infer authorization from a role alone.\n"
    "- An unknown authorization state is not a denial: when the requested value is "
    "source-grounded and no explicit deny, revoke, conflict, deletion, or scope "
    "constraint applies, answer from the evidence.\n"
    "- Do not disclose a value only when the symbolic governance layer gives an "
    "explicit deny, revoke, conflict, deletion, or out-of-scope decision."
)


def _structured_message_record(
    *,
    instance: MemoryInstance,
    message: dict[str, Any],
    turn_index: int,
) -> dict[str, Any]:
    """Preserve the observable GateMem message as typed retrieval metadata."""
    message = normalize_message_timestamp(message)
    speaker_id = str(message.get("speaker_id") or "") or None
    speaker_role = str(message.get("speaker_role") or "") or None
    message_id = str(message.get("message_id") or "")
    turn_id = str(message.get("turn_id") or message_id)
    return {
        "record_type": "message",
        # Keep both Gov-Mem's normalized name and GateMem's original name.
        "message_id": message_id,
        "turn_id": turn_id,
        "turn_index": int(turn_index),
        "timestamp": normalize_timestamp(message.get("timestamp")),
        "speaker": {
            "principal_id": speaker_id,
            "role": speaker_role,
        },
        "turn_kind": message.get("turn_kind"),
        # The paper-facing path obtains authorization candidates from the
        # source-grounded semantic compiler. No lexical policy annotation is
        # added during ingestion.
        "authorization_assertions": [],
        "text": str(message.get("text") or ""),
        "checkpoint": {
            "as_of_turn_id": str(
                (instance.metadata.get("observable") or {}).get("as_of_turn_id") or ""
            ),
        },
        # The adapter keeps the visible source turn verbatim. This prevents
        # future GateMem fields from being lost at the RAG boundary.
        "source_turn": dict(message.get("source_turn") or {}),
    }


def _build_turn_chunks(instance: MemoryInstance) -> list[RAGChunk]:
    """Match GateMem's RAG-Naive V0: one retrievable chunk per visible turn."""
    chunks: list[RAGChunk] = []
    for index, message in enumerate(instance.messages):
        message = normalize_message_timestamp(message)
        speaker_id = str(message.get("speaker_id") or "unknown")
        role = str(message.get("speaker_role") or "")
        message_id = str(message.get("message_id") or "")
        text = str(message.get("text") or "").strip()
        prefix = f"[{role}:{speaker_id}]" if role else f"[{speaker_id}]"
        chunks.append(
            RAGChunk(
                # Match the released GateMem Chunker. The identifier is local
                # to this visible turn sequence and must not expose the
                # benchmark checkpoint ID to a later Stage 2 model.
                chunk_id=f"chunk_{index + 1:04d}_{message_id}_{message_id}",
                instance_id=instance.instance_id,
                text=f"{prefix} {text}".strip(),
                source_message_ids=[str(message.get("message_id") or "")],
                speaker_ids=[speaker_id],
                timestamp_range=(
                    normalize_timestamp(message.get("timestamp")),
                    normalize_timestamp(message.get("timestamp")),
                ),
                metadata={
                    "chunk_type": "turn",
                    "structured_record": _structured_message_record(
                        instance=instance,
                        message=message,
                        turn_index=index,
                    ),
                },
            )
        )
    return chunks


def _is_current_query_item(item: Any, instance: MemoryInstance) -> bool:
    """Identify the visible checkpoint question, which is not memory evidence.

    GateMem checkpoints can expose a normalized ``query_text`` that differs
    from the question text stored in the final visible turn (for example, a
    name may be expanded).  The adapter's observable ``as_of_turn_id`` is the
    structural checkpoint boundary and therefore the only reliable identity
    for excluding that turn.  Text equality remains a compatibility fallback
    for non-checkpoint callers that do not provide the boundary metadata.
    """
    metadata = dict(getattr(item, "metadata", {}) or {})
    record = dict(metadata.get("structured_record") or {})
    checkpoint_turn_id = str(
        ((instance.metadata.get("observable") or {}).get("as_of_turn_id")) or ""
    ).strip()
    record_turn_id = str(record.get("turn_id") or record.get("message_id") or "").strip()
    if checkpoint_turn_id and record_turn_id:
        return record_turn_id == checkpoint_turn_id
    record_text = " ".join(str(record.get("text") or "").split()).casefold()
    question_text = " ".join(str(instance.question or "").split()).casefold()
    return bool(question_text) and record_text == question_text


def _relationship_block(instance: MemoryInstance) -> str:
    raw_episode = dict((instance.metadata.get("raw_sample") or {}).get("episode") or {})
    entities = dict(raw_episode.get("entities") or {})
    relationships = list(entities.get("relationships") or [])
    requester = str(instance.asking_user_id or "")
    relevant = []
    for relationship in relationships:
        if not isinstance(relationship, dict):
            continue
        if any(
            str(value) == requester
            for key, value in relationship.items()
            if str(key).lower().endswith("_id")
        ):
            relevant.append(format_relationship_fact(relationship))
    return "\n".join(relevant) if relevant else "(none)"


def _source_grounded_access_context(
    *,
    instance: MemoryInstance,
    symbolic_trace: dict[str, Any],
    semantic_contract: Any,
    semantic_atoms: list[dict[str, Any]] | None = None,
    query_analysis: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Project query-relevant graph assignments without making an access decision."""
    requester = str(instance.asking_user_id or "")
    requester_node = f"principal::{requester}" if requester else ""
    graph_nodes = [item for item in symbolic_trace.get("graph_nodes") or [] if isinstance(item, dict)]
    graph_edges = [item for item in symbolic_trace.get("graph_edges") or [] if isinstance(item, dict)]
    target_phrases: list[str] = []
    for item in getattr(semantic_contract, "target_entities", []) or []:
        if isinstance(item, dict):
            # Keep each entity's references together.  Combining tokens from
            # multiple entities makes a graph node appear to match a target
            # that was never named as a complete phrase in the query.
            for value in (item.get("canonical_reference"), item.get("surface_form")):
                phrase = str(value or "").strip()
                if phrase:
                    target_phrases.append(phrase)

    def _tokens(value: Any) -> set[str]:
        return set(re.findall(r"[a-z0-9]+", str(value or "").casefold()))

    target_token_sets = [tokens for phrase in target_phrases if (tokens := _tokens(phrase))]
    target_ids: set[str] = set()
    for node in graph_nodes:
        node_id = str(node.get("node_id") or "")
        if not node_id or not target_token_sets:
            continue
        node_tokens = _tokens(node.get("value") or node_id.split("::", 1)[-1])
        # Partial overlap is unsafe: a shared token may refer to a distinct
        # entity in a nearby track. The entire query target phrase must match
        # the runtime graph node before it can participate in an access path.
        if any(
            phrase_tokens == node_tokens or phrase_tokens.issubset(node_tokens)
            for phrase_tokens in target_token_sets
        ) and node_id.startswith(("entity::", "principal::")):
            target_ids.add(node_id)

    adjacency: dict[str, list[dict[str, Any]]] = {}
    for edge in graph_edges:
        source, target = str(edge.get("source") or ""), str(edge.get("target") or "")
        # Exclude evidence/role/about edges: only declared principal/entity
        # relations are eligible as assignment context.
        if source.startswith(("principal::", "entity::")) and target.startswith(("principal::", "entity::")):
            adjacency.setdefault(source, []).append(edge)

    paths: list[list[dict[str, Any]]] = []
    if requester_node:
        frontier: list[tuple[str, list[dict[str, Any]]]] = [(requester_node, [])]
        visited: set[tuple[str, int]] = {(requester_node, 0)}
        while frontier:
            current, path = frontier.pop(0)
            if path and current in target_ids:
                paths.append(path)
                continue
            if len(path) >= 3:
                continue
            for edge in adjacency.get(current, []):
                next_node = str(edge.get("target") or "")
                state = (next_node, len(path) + 1)
                if state not in visited:
                    visited.add(state)
                    frontier.append((next_node, [*path, edge]))

    temporal = dict(symbolic_trace.get("temporal_authorization") or {})
    policy = dict(symbolic_trace.get("policy_consistency") or {})
    compact_paths = [
        [
            {
                "relation": str(edge.get("edge_type") or ""),
                "source": str(edge.get("source") or ""),
                "target": str(edge.get("target") or ""),
                "attributes": dict(edge.get("attributes") or {}),
            }
            for edge in path
        ]
        for path in paths[:8]
    ]
    safety = dict((query_analysis or {}).get("safety") or {})
    # Exactness increases sensitivity but is not itself a denial signal. An
    # explicit confirmation/existence probe remains excluded; an ordinary
    # source-grounded operational request may still be delivered when the
    # graph proves the assignment and no negative policy is present.
    ordinary_operational_request = (
        str(safety.get("delivery_mode") or "").casefold() == "ordinary_operational"
        and not bool(safety.get("confirmation", False))
        and not bool(safety.get("existence", False))
    )
    path_has_scoped_limit = any(
        bool(dict(edge.get("attributes") or {}).get("access_scope"))
        for path in paths
        for edge in path
    )
    policy_decision = str(
        temporal.get("decision") or policy.get("decision") or "unknown"
    ).casefold()
    # A complete set of verified, source-grounded query slots is sufficient to
    # deliver an ordinary operational answer even when the relation graph does
    # not contain an explicit assignment path.  This is evidence delivery, not
    # an authorization grant: explicit deny/revoke/conflict remains blocking.
    required_slot_ids = {
        str(slot.slot_id)
        for slot in getattr(semantic_contract, "requested_slots", []) or []
        if bool(getattr(slot, "required_for_answer", True))
    }
    grounded_slot_ids = {
        str(atom.get("slot_id") or "")
        for atom in semantic_atoms or []
        if isinstance(atom, dict)
        and str(atom.get("slot_id") or "")
        and isinstance(atom.get("source"), dict)
        and str(dict(atom.get("source") or {}).get("chunk_id") or "")
    }
    semantic_delivery_support = bool(required_slot_ids) and required_slot_ids.issubset(
        grounded_slot_ids
    )
    # A declared assignment path is evidence that the requester/target binding
    # is observable.  A scoped path is not itself a denial: only an explicit
    # negative symbolic certificate blocks delivery.  Field-level scope is
    # still carried in requested_slot_delivery for the answer model.
    operational_assignment_in_scope = (
        (bool(compact_paths) or semantic_delivery_support)
        and ordinary_operational_request
        and policy_decision not in {"deny", "revoke", "conflict"}
    )
    requested_slot_delivery = []
    for slot in getattr(semantic_contract, "requested_slots", []) or []:
        if not hasattr(slot, "slot_id"):
            continue
        requested_slot_delivery.append({
            "slot_id": str(slot.slot_id),
            "slot_name": str(slot.slot_name),
            "query_sensitive": bool(slot.sensitivity_possible or slot.authorization_relevant),
            "graph_scope": (
                "unscoped_assignment"
                if operational_assignment_in_scope
                else (
                    "scoped_assignment"
                    if compact_paths and path_has_scoped_limit
                    else "not_certified_for_unscoped_delivery"
                )
            ),
        })
    return {
        "source_grounded": bool(compact_paths) or semantic_delivery_support,
        "semantic_delivery_support": semantic_delivery_support,
        "required_slot_ids": sorted(required_slot_ids),
        "grounded_slot_ids": sorted(grounded_slot_ids),
        "requester": requester,
        "target_nodes": sorted(target_ids),
        "assignment_paths": compact_paths,
        "explicit_policy_decision": policy_decision,
        "operational_assignment_in_scope": operational_assignment_in_scope,
        "requested_slot_delivery": requested_slot_delivery,
        "scope_note": (
            "Assignment paths are observable governance context, not a blanket grant. "
            "Explicit deny/revoke/conflict remains authoritative; apply scope to the "
            "requested field rather than refusing unrelated fields."
        ),
    }


def _merge_question_safety_into_contract(
    *,
    semantic_contract: Any,
    query_analysis: dict[str, Any] | None,
    question: str,
) -> bool:
    """Lift an independently compiled question safety profile to slot scope.

    Query induction and Stage-2 safety analysis are separate neural views. If
    the question-only safety contract identifies a protected fact, every
    requested slot is marked for field-level governance before symbolic
    projection. This does not grant or deny access and does not inspect
    evidence text or use a domain lexicon.
    """
    profile = _query_safety_profile(
        query_analysis or {}, question=question, use_lexical_backstop=False,
    )
    category = str(profile.get("category") or "none").casefold()
    delivery_mode = str(profile.get("delivery_mode") or "unknown").casefold()
    protected = bool(profile.get("sensitive")) and (
        delivery_mode in {"protected_fact", "permission", "historical"}
        or bool(profile.get("exact"))
        or bool(profile.get("confirmation"))
        or bool(profile.get("existence"))
        or category in {"credential", "identity", "restricted"}
    )
    if not protected:
        return False
    changed = False
    for slot in getattr(semantic_contract, "requested_slots", []) or []:
        if not getattr(slot, "sensitivity_possible", False):
            slot.sensitivity_possible = True
            changed = True
        if not getattr(slot, "authorization_relevant", False):
            slot.authorization_relevant = True
            changed = True
    return changed


def _format_retrieved_memory(evidence: list[RetrievedEvidence]) -> str:
    if not evidence:
        return "(none)"
    lines = []
    # The full state ledger is an internal symbolic artifact and may contain
    # exact values for fields that were not released to the answer model.
    # Keep it out of the answer prompt unconditionally; the governed source
    # rows and the value-free certificates below are sufficient for delivery.
    def _safe_certificate(value: Any, *, kind: str) -> dict[str, Any] | None:
        """Expose only value-free governance status to answer-side prompts.

        The symbolic graph has already been consumed before answer realization.
        Its resource names, principal assignments, graph edges, and current
        authorization entries can contain protected source values, so they
        must never be serialized into the neural answer context.  Keep only
        aggregate certificate status and counts needed to interpret a boundary.
        """
        if not isinstance(value, dict):
            return None
        keep = {
            "version", "mode", "decision", "enforcement_applied",
            "conflict_count", "event_count", "unknown_event_count",
            "semantic_candidate_event_count", "ignored_future_event_count",
            "fact_count", "new_llm_calls",
        }
        safe = {key: value[key] for key in keep if key in value}
        safe["redacted_for_answer_context"] = True
        safe["certificate_kind"] = kind
        return safe

    policy_certificate = next(
        (
            (row.metadata or {}).get("symbolic_policy_certificate")
            for row in evidence
            if isinstance((row.metadata or {}).get("symbolic_policy_certificate"), dict)
        ),
        None,
    )
    if (safe_policy_certificate := _safe_certificate(policy_certificate, kind="policy")):
        lines.append(
            "[SYMBOLIC_POLICY_CERTIFICATE] "
            + json.dumps(safe_policy_certificate, ensure_ascii=False, sort_keys=True)
        )
    temporal_authorization_certificate = next(
        (
            (row.metadata or {}).get("symbolic_temporal_authorization_certificate")
            for row in evidence
            if isinstance((row.metadata or {}).get("symbolic_temporal_authorization_certificate"), dict)
        ),
        None,
    )
    if (safe_temporal_certificate := _safe_certificate(
        temporal_authorization_certificate, kind="temporal_authorization"
    )):
        lines.append(
            "[SYMBOLIC_TEMPORAL_AUTHORIZATION_CERTIFICATE] "
            + json.dumps(safe_temporal_certificate, ensure_ascii=False, sort_keys=True)
        )
    for index, row in enumerate(evidence, 1):
        metadata = dict(row.metadata or {})
        record = metadata.get("structured_record")
        if isinstance(record, dict):
            # Keep the typed provenance visible to Stage 2 without asking it
            # to recover identity, role, or time from natural language.
            typed = {
                "record_type": record.get("record_type"),
                "message_id": record.get("message_id"),
                "turn_id": record.get("turn_id"),
                "turn_index": record.get("turn_index"),
                "timestamp": record.get("timestamp"),
                "speaker": record.get("speaker"),
                "turn_kind": record.get("turn_kind"),
                "checkpoint": record.get("checkpoint"),
                "authorization_assertions": record.get("authorization_assertions"),
                "text": record.get("text"),
                "source_turn": record.get("source_turn"),
                "symbolic_provenance": metadata.get("symbolic_provenance"),
                "symbolic_consistency": metadata.get("symbolic_consistency"),
                # Value-bearing symbolic claims/events are intentionally not
                # copied into the answer prompt.  Symbolic reasoning has
                # already consumed them; only source text and certificates
                # (which are sanitized before this function) remain visible.
                "symbolic_policy_certificate": _safe_certificate(
                    metadata.get("symbolic_policy_certificate"), kind="policy"
                ),
                "symbolic_temporal_authorization_certificate": _safe_certificate(
                    metadata.get("symbolic_temporal_authorization_certificate"),
                    kind="temporal_authorization",
                ),
            }
            lines.append(
                f"Memory {index} [STRUCTURED_RECORD] "
                f"{json.dumps(typed, ensure_ascii=False, sort_keys=True)}"
            )
        else:
            speaker = str(metadata.get("speaker_id") or row.user_id or "unknown")
            lines.append(f"Memory {index} (speaker={speaker}): {row.content}")
    return "\n".join(lines)


def _semantic_context_boundary(
    *,
    evidence: list[RetrievedEvidence],
    semantic_contract: Any,
    semantic_atoms: list[dict[str, Any]] | None,
    symbolic_trace: dict[str, Any],
    query_analysis: dict[str, Any] | None = None,
) -> tuple[list[RetrievedEvidence], dict[str, Any]]:
    """Project protected query atoms before exposing context to the answer LLM.

    The projection is driven only by the open-vocabulary query contract and
    verified atom source spans. Authorization remains symbolic: a protected
    span is withheld only when the temporal authorization certificate gives an
    explicit ``deny``, ``revoke``, or ``conflict`` decision. An unknown state is
    not a denial and leaves source-grounded evidence available. No lexical
    field table participates in this boundary.
    """
    requested_slot_ids = {
        str(slot.slot_id)
        for slot in getattr(semantic_contract, "requested_slots", []) or []
        if getattr(slot, "slot_id", None)
    }
    protected_slots = {
        str(slot.slot_id)
        for slot in getattr(semantic_contract, "requested_slots", []) or []
        if bool(getattr(slot, "sensitivity_possible", False))
        or bool(getattr(slot, "authorization_relevant", False))
    }
    # The slot-level compiler is intentionally high-recall, but a provider can
    # under-mark one slot in a multi-field protected request.  The independent
    # question-only safety contract is therefore allowed to lift the boundary
    # to every requested slot when it explicitly requests protected delivery.
    # This is still neural query interpretation plus symbolic authorization;
    # it does not inspect field names or source text.
    safety = dict((query_analysis or {}).get("safety") or {})
    protected_delivery = (
        bool(safety.get("sensitive"))
        and str(safety.get("delivery_mode") or "").casefold()
        in {"protected_fact", "historical", "permission"}
    ) or (
        bool(safety.get("sensitive"))
        and any(bool(safety.get(key)) for key in ("exact", "confirmation", "existence"))
    )
    if protected_delivery:
        protected_slots.update(
            str(slot.slot_id)
            for slot in getattr(semantic_contract, "requested_slots", []) or []
            if getattr(slot, "slot_id", None)
        )
    relevance_boundary_enabled = bool(requested_slot_ids and semantic_atoms)
    audit: dict[str, Any] = {
        "enabled": bool(protected_slots or relevance_boundary_enabled),
        "relevance_slot_ids": sorted(requested_slot_ids),
        "protected_slot_ids": sorted(protected_slots),
        "query_level_protected_delivery": bool(protected_delivery),
        "authorization_decision": "unknown",
        "projected_memory_ids": [],
        "filtered_memory_ids": [],
        "redacted_atom_ids": [],
        "evidence_boundary": "stage1_top_k_closed_set",
    }
    if not protected_slots and not relevance_boundary_enabled:
        audit["enforcement_applied"] = False
        return list(evidence), audit

    temporal = dict(symbolic_trace.get("temporal_authorization") or {})
    policy = dict(symbolic_trace.get("policy_consistency") or {})
    decision = str(temporal.get("decision") or policy.get("decision") or "unknown").casefold()
    audit["authorization_decision"] = decision
    # Unknown authorization is not a reason to shrink ordinary utility
    # evidence. Keep the closed Stage-1 set intact so semantic extraction and
    # reranking can recover every requested field. Explicitly protected or
    # confirmation-style requests still use the source-grounded projection
    # below; explicit deny/revoke/conflict remains blocking.
    if (
        decision == "unknown"
        and not protected_slots
        and not protected_delivery
    ):
        audit["mode"] = "unknown_authorization_passthrough"
        audit["enforcement_applied"] = False
        return list(evidence), audit
    # Missing authorization evidence is not itself a policy conflict. The
    # boundary is fail-closed only for an explicit negative symbolic state.
    release_protected = decision not in {"deny", "revoke", "conflict"}
    if release_protected:
        # Unknown and allow do not deny any grounded value.  We still keep the
        # answer context closed over query-conditioned, source-grounded atoms:
        # this is a relevance/provenance boundary, not an authorization rule.
        # Without this distinction, every unrelated Top-20 private row reaches
        # the answer model and is counted as context leakage.
        atoms_by_memory: dict[str, list[dict[str, Any]]] = {}
        temporal_spans_by_memory: dict[str, list[str]] = {}
        for atom in semantic_atoms or []:
            if not isinstance(atom, dict):
                continue
            source = dict(atom.get("source") or {})
            memory_id = str(source.get("chunk_id") or source.get("memory_id") or "")
            span = str(source.get("span") or "").strip()
            if memory_id and span:
                atoms_by_memory.setdefault(memory_id, []).append(atom)
            slot_id = str(atom.get("slot_id") or "")
            if slot_id in protected_slots:
                continue
            temporal = dict(atom.get("temporal") or {})
            anchor_span = str(temporal.get("anchor_span") or "").strip()
            anchor_source = dict(temporal.get("anchor_source") or {})
            anchor_id = str(
                anchor_source.get("chunk_id")
                or anchor_source.get("memory_id")
                or memory_id
                or ""
            )
            if anchor_span and anchor_id:
                temporal_spans_by_memory.setdefault(anchor_id, []).append(anchor_span)
        if not atoms_by_memory and not temporal_spans_by_memory:
            audit["enforcement_applied"] = False
            return list(evidence), audit
        grounded_rows: list[RetrievedEvidence] = []
        for row in evidence:
            row_atoms = atoms_by_memory.get(row.memory_id, [])
            anchor_spans = temporal_spans_by_memory.get(row.memory_id, [])
            if not row_atoms and not anchor_spans:
                continue
            record = dict((row.metadata or {}).get("structured_record") or {})
            source_text = str(record.get("text") or row.content or "")
            source_normalized = " ".join(source_text.split()).casefold()
            spans: list[str] = []
            for atom in row_atoms:
                span = str(dict(atom.get("source") or {}).get("span") or "").strip()
                if span and " ".join(span.split()).casefold() in source_normalized:
                    if span not in spans:
                        spans.append(span)
            for anchor_span in anchor_spans:
                if (
                    anchor_span
                    and " ".join(anchor_span.split()).casefold() in source_normalized
                    and anchor_span not in spans
                ):
                    spans.append(anchor_span)
            if not spans:
                continue
            projected_text = " ".join(spans)
            metadata = dict(row.metadata or {})
            if record:
                projected_record = dict(record)
                projected_record["text"] = projected_text
                if isinstance(projected_record.get("source_turn"), dict):
                    projected_record["source_turn"] = {
                        **projected_record["source_turn"], "text": projected_text,
                    }
                metadata["structured_record"] = projected_record
            metadata["semantic_context_boundary"] = {
                "mode": "source_grounded_span_relevance",
                "projected": projected_text != source_text,
                "authorization_decision": decision,
            }
            if projected_text != source_text:
                audit["projected_memory_ids"].append(row.memory_id)
            grounded_rows.append(replace(row, content=projected_text, metadata=metadata))
        audit["filtered_memory_ids"] = [
            row.memory_id for row in evidence if row.memory_id not in atoms_by_memory
        ]
        audit["enforcement_applied"] = False
        return grounded_rows, audit
    atoms_by_memory: dict[str, list[dict[str, Any]]] = {}
    temporal_spans_by_memory: dict[str, list[str]] = {}
    for atom in semantic_atoms or []:
        if not isinstance(atom, dict):
            continue
        source = dict(atom.get("source") or {})
        memory_id = str(source.get("chunk_id") or source.get("memory_id") or "")
        span = str(source.get("span") or "").strip()
        if memory_id and span:
            atoms_by_memory.setdefault(memory_id, []).append(atom)
        # Temporal anchors are source-grounded auxiliary spans. They may be
        # carried to the answer boundary only for an unprotected slot.
        slot_id = str(atom.get("slot_id") or "")
        if slot_id in protected_slots:
            continue
        temporal = dict(atom.get("temporal") or {})
        anchor_span = str(temporal.get("anchor_span") or "").strip()
        anchor_source = dict(temporal.get("anchor_source") or {})
        anchor_id = str(
            anchor_source.get("chunk_id")
            or anchor_source.get("memory_id")
            or memory_id
            or ""
        )
        if anchor_span and anchor_id:
            temporal_spans_by_memory.setdefault(anchor_id, []).append(anchor_span)

    def _safe_certificate(value: Any, *, kind: str) -> dict[str, Any] | None:
        """Keep governance state while removing value/resource-bearing detail.

        The symbolic graph is still consumed before this projection.  Once a
        protected value is denied or unresolved, the answer-side neural
        prompts only need the certificate outcome and aggregate diagnostics;
        carrying the full ledger/policy graph would reintroduce the very value
        that the source span projection removed.
        """
        if not isinstance(value, dict):
            return None
        keep = {
            "version", "mode", "decision", "enforcement_applied",
            "conflict_count", "event_count", "unknown_event_count",
            "semantic_candidate_event_count", "ignored_future_event_count",
            "fact_count", "new_llm_calls",
        }
        safe = {key: value[key] for key in keep if key in value}
        safe["redacted_for_answer_context"] = True
        safe["certificate_kind"] = kind
        return safe

    def _strip_value_bearing_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
        """Remove semantic values from every carrier visible to answer prompts."""
        sanitized = dict(metadata)
        for key in (
            "semantic_compiler_atoms",
            "symbolic_state_claims",
            "symbolic_state_ledger",
            "symbolic_policy_facts",
            "symbolic_permission_claim",
            "symbolic_lifecycle_claim",
            "symbolic_lifecycle_target_binding",
            "symbolic_temporal_authorization_events",
        ):
            sanitized.pop(key, None)
        policy = _safe_certificate(sanitized.get("symbolic_policy_certificate"), kind="policy")
        if policy is not None:
            sanitized["symbolic_policy_certificate"] = policy
        else:
            sanitized.pop("symbolic_policy_certificate", None)
        temporal = _safe_certificate(
            sanitized.get("symbolic_temporal_authorization_certificate"),
            kind="temporal_authorization",
        )
        if temporal is not None:
            sanitized["symbolic_temporal_authorization_certificate"] = temporal
        else:
            sanitized.pop("symbolic_temporal_authorization_certificate", None)
        record = sanitized.get("structured_record")
        if isinstance(record, dict):
            record = dict(record)
            for key in (
                "semantic_compiler_atoms",
                "symbolic_state_claims",
                "symbolic_state_ledger",
                "symbolic_policy_facts",
                "symbolic_permission_claim",
                "symbolic_lifecycle_claim",
                "symbolic_lifecycle_target_binding",
                "symbolic_temporal_authorization_events",
            ):
                record.pop(key, None)
            sanitized["structured_record"] = record
        return sanitized

    projected: list[RetrievedEvidence] = []
    for row in evidence:
        row_atoms = atoms_by_memory.get(row.memory_id, [])
        anchor_spans = temporal_spans_by_memory.get(row.memory_id, [])
        if not row_atoms and not anchor_spans:
            # A protected query must not expose an ungrounded full source row.
            audit["filtered_memory_ids"].append(row.memory_id)
            continue
        record = dict((row.metadata or {}).get("structured_record") or {})
        source_text = str(record.get("text") or row.content or "")
        pieces: list[str] = []
        seen: set[tuple[str, str]] = set()
        for atom in row_atoms:
            source = dict(atom.get("source") or {})
            span = str(source.get("span") or "").strip()
            if not span or " ".join(span.split()).casefold() not in " ".join(source_text.split()).casefold():
                continue
            slot_id = str(atom.get("slot_id") or "")
            protected = slot_id in protected_slots
            piece = span
            if protected and not release_protected:
                piece = "[REDACTED GOVERNED CLAIM]"
                audit["redacted_atom_ids"].append(str(atom.get("atom_id") or ""))
            key = (piece, slot_id)
            if key not in seen:
                seen.add(key)
                pieces.append(piece)
        for anchor_span in anchor_spans:
            if (
                anchor_span
                and " ".join(anchor_span.split()).casefold()
                in " ".join(source_text.split()).casefold()
            ):
                key = (anchor_span, "__temporal_anchor__")
                if key not in seen:
                    seen.add(key)
                    pieces.append(anchor_span)
        if not pieces:
            audit["filtered_memory_ids"].append(row.memory_id)
            continue
        projected_text = " ".join(pieces)
        metadata = dict(row.metadata or {})
        if record:
            projected_record = dict(record)
            projected_record["text"] = projected_text
            if protected_slots and not release_protected:
                # The state ledger and typed semantic annotations are useful to
                # symbolic reasoning, but they may contain the exact value
                # outside the projected source text.  Do not pass those
                # value-bearing carriers to the answer LLM on an unknown/deny
                # path.  Authorization certificates remain available through
                # the non-value-bearing fields and the governance context.
                for key in (
                    "semantic_compiler_atoms",
                    "symbolic_state_claims",
                    "symbolic_state_ledger",
                    "symbolic_policy_facts",
                    "symbolic_permission_claim",
                    "symbolic_lifecycle_claim",
                    "symbolic_lifecycle_target_binding",
                    "symbolic_temporal_authorization_events",
                ):
                    projected_record.pop(key, None)
            if isinstance(projected_record.get("source_turn"), dict):
                projected_record["source_turn"] = {
                    **projected_record["source_turn"],
                    "text": projected_text,
                }
            metadata["structured_record"] = projected_record
        if protected_slots and not release_protected:
            # The top-level metadata is what _format_retrieved_memory uses to
            # populate its typed symbolic fields.  Sanitizing only the nested
            # record is insufficient and previously leaked exact atom values.
            metadata = _strip_value_bearing_metadata(metadata)
        metadata["semantic_context_boundary"] = {
            "protected_slot_ids": sorted(protected_slots),
            "authorization_decision": decision,
            "projected": projected_text != source_text,
        }
        if projected_text != source_text:
            audit["projected_memory_ids"].append(row.memory_id)
        projected.append(replace(row, content=projected_text, metadata=metadata))
    audit["enforcement_applied"] = bool(
        audit["projected_memory_ids"] or audit["filtered_memory_ids"]
    )
    return projected, audit


def _preserve_symbolic_state_carriers(
    *,
    selected: list[RetrievedEvidence],
    available: list[RetrievedEvidence],
    symbolic_trace: dict[str, Any],
) -> list[RetrievedEvidence]:
    """Keep source rows for resolved semantic slots and reattach certificates.

    Stage 2 is allowed to reorder/compact evidence, but it must not discard
    the only closed-set source carrier for a verified query slot.  Reattached
    rows are projected to the atom's verified source spans; the original full
    Top-20 row is never reintroduced into the answer prompt.  This keeps the
    carrier boundary source-grounded while making the operation domain
    agnostic.
    """

    rows = list(selected)
    available_by_id = {row.memory_id: row for row in available}
    selected_ids = {row.memory_id for row in rows}
    ledger = dict(symbolic_trace.get("state_ledger") or {})
    authorization = dict(symbolic_trace.get("temporal_authorization") or {})
    authorization_decision = str(authorization.get("decision") or "unknown").casefold()
    # Explicit negative governance remains authoritative.  In particular, do
    # not use carrier preservation to route around a deny/revoke/conflict
    # boundary applied by the symbolic layer.
    carrier_recovery_allowed = authorization_decision not in {"deny", "revoke", "conflict"}

    semantic_audit = dict(symbolic_trace.get("semantic_compiler") or {})
    contract = dict(semantic_audit.get("query_contract") or {})
    required_slot_ids = {
        str(slot.get("slot_id") or "")
        for slot in contract.get("requested_slots") or []
        if isinstance(slot, dict)
        and bool(slot.get("required_for_answer", True))
        and str(slot.get("slot_id") or "")
    }
    protected_slot_ids = {
        str(slot.get("slot_id") or "")
        for slot in contract.get("requested_slots") or []
        if isinstance(slot, dict)
        and (
            bool(slot.get("sensitivity_possible"))
            or bool(slot.get("authorization_relevant"))
        )
    }
    atoms = [
        atom for atom in semantic_audit.get("final_grounded_atoms") or []
        if isinstance(atom, dict)
        and str(atom.get("slot_id") or "") in required_slot_ids
        and isinstance(atom.get("source"), dict)
    ]
    atoms_by_slot: dict[str, list[dict[str, Any]]] = {}
    for atom in atoms:
        atoms_by_slot.setdefault(str(atom.get("slot_id") or ""), []).append(atom)

    def projected_carrier(row: RetrievedEvidence, row_atoms: list[dict[str, Any]]) -> RetrievedEvidence | None:
        record = dict((row.metadata or {}).get("structured_record") or {})
        source_text = str(record.get("text") or row.content or "")
        source_normalized = " ".join(source_text.split()).casefold()
        spans: list[str] = []
        for atom in row_atoms:
            span = str(dict(atom.get("source") or {}).get("span") or "").strip()
            if span and " ".join(span.split()).casefold() in source_normalized and span not in spans:
                spans.append(span)
        if not spans:
            return None
        projected_text = " ".join(spans)
        metadata = dict(row.metadata or {})
        if record:
            projected_record = dict(record)
            projected_record["text"] = projected_text
            if isinstance(projected_record.get("source_turn"), dict):
                projected_record["source_turn"] = {
                    **projected_record["source_turn"], "text": projected_text,
                }
            metadata["structured_record"] = projected_record
        metadata["semantic_carrier_preservation"] = {
            "source_grounded": True,
            "projected": projected_text != source_text,
            "atom_ids": [str(atom.get("atom_id") or "") for atom in row_atoms],
        }
        return replace(row, content=projected_text, metadata=metadata)

    def projected_temporal_carrier(
        row: RetrievedEvidence,
        anchor_spans: list[str],
    ) -> RetrievedEvidence | None:
        record = dict((row.metadata or {}).get("structured_record") or {})
        source_text = str(record.get("text") or row.content or "")
        normalized = " ".join(source_text.split()).casefold()
        spans = [
            span for span in dict.fromkeys(str(value).strip() for value in anchor_spans)
            if span and " ".join(span.split()).casefold() in normalized
        ]
        if not spans:
            return None
        projected_text = " ".join(spans)
        metadata = dict(row.metadata or {})
        if record:
            projected_record = dict(record)
            projected_record["text"] = projected_text
            if isinstance(projected_record.get("source_turn"), dict):
                projected_record["source_turn"] = {
                    **projected_record["source_turn"], "text": projected_text,
                }
            metadata["structured_record"] = projected_record
        metadata["semantic_carrier_preservation"] = {
            "source_grounded": True,
            "temporal_anchor_only": True,
            "projected": projected_text != source_text,
        }
        return replace(row, content=projected_text, metadata=metadata)

    for field in (ledger.get("fields") or {}).values():
        if not isinstance(field, dict) or field.get("status") != "resolved":
            continue
        if str(field.get("slot_id") or "") in protected_slot_ids:
            continue
        source_id = str(field.get("source_memory_id") or "")
        if source_id and source_id not in selected_ids and source_id in available_by_id:
            if carrier_recovery_allowed:
                source_atoms = [
                    atom for atom in atoms
                    if str(dict(atom.get("source") or {}).get("chunk_id") or "") == source_id
                ]
                carrier = projected_carrier(available_by_id[source_id], source_atoms)
                if carrier is not None:
                    rows.append(carrier)
                    selected_ids.add(source_id)

    # A field can be absent from the post-rerank ledger because the carrier
    # was dropped before symbolic state projection. Recover it only when all
    # verified candidates for that slot agree on one source row. Multiple
    # source rows remain a symbolic conflict/choice and are not reintroduced.
    if carrier_recovery_allowed:
        for slot_id, slot_atoms in atoms_by_slot.items():
            if not slot_atoms:
                continue
            if slot_id in protected_slot_ids:
                continue
            source_ids = {
                str(dict(atom.get("source") or {}).get("chunk_id") or "")
                for atom in slot_atoms
                if str(dict(atom.get("source") or {}).get("chunk_id") or "")
            }
            if len(source_ids) != 1:
                continue
            source_id = next(iter(source_ids))
            if source_id in selected_ids or source_id not in available_by_id:
                continue
            carrier = projected_carrier(available_by_id[source_id], slot_atoms)
            if carrier is not None:
                rows.append(carrier)
                selected_ids.add(source_id)

        # Recover a closed-set calendar anchor that was separated from the
        # latest value update. Only the exact anchor span is reattached; the
        # full historical source row is never exposed to the answer model.
        anchor_by_source: dict[str, list[str]] = {}
        for atom in atoms:
            if str(atom.get("slot_id") or "") in protected_slot_ids:
                continue
            temporal = dict(atom.get("temporal") or {})
            anchor_span = str(temporal.get("anchor_span") or "").strip()
            anchor_source = dict(temporal.get("anchor_source") or {})
            anchor_id = str(
                anchor_source.get("chunk_id")
                or anchor_source.get("memory_id")
                or ""
            )
            if anchor_span and anchor_id:
                anchor_by_source.setdefault(anchor_id, []).append(anchor_span)
        for field in (ledger.get("fields") or {}).values():
            if not isinstance(field, dict) or str(field.get("slot_id") or "") in protected_slot_ids:
                continue
            anchor = dict(field.get("temporal_anchor") or {})
            anchor_span = str(anchor.get("span") or "").strip()
            anchor_source = dict(anchor.get("source") or {})
            anchor_id = str(
                anchor_source.get("chunk_id")
                or anchor_source.get("memory_id")
                or ""
            )
            if anchor_span and anchor_id:
                anchor_by_source.setdefault(anchor_id, []).append(anchor_span)
        for source_id, anchor_spans in anchor_by_source.items():
            if source_id in selected_ids or source_id not in available_by_id:
                continue
            carrier = projected_temporal_carrier(available_by_id[source_id], anchor_spans)
            if carrier is not None:
                rows.append(carrier)
                selected_ids.add(source_id)
    if not rows:
        return rows
    metadata = dict(rows[0].metadata or {})
    if ledger:
        metadata["symbolic_state_ledger"] = ledger
    policy = symbolic_trace.get("policy_consistency")
    if isinstance(policy, dict) and policy.get("enabled") is not False:
        metadata["symbolic_policy_certificate"] = policy
    if isinstance(authorization, dict) and authorization.get("enabled") is not False:
        metadata["symbolic_temporal_authorization_certificate"] = authorization
    rows[0] = replace(rows[0], metadata=metadata)
    return rows


def _answer_redacted_confirmation_boundary_reason(
    *,
    question: str,
    answer: str,
    action: str,
    query_analysis: dict[str, Any] | None = None,
) -> str | None:
    """Reject confirmation text that contradicts an answer-redacted action."""

    if action != "answer_redacted":
        return None
    answer_text = str(answer or "").strip()
    contract_sensitive = bool((query_analysis or {}).get("semantic_contract_sensitive")) or bool(
        (query_analysis or {}).get("semantic_contract_has_sensitive_slot")
    )
    safety_profile = _query_safety_profile(
        query_analysis, question=question, use_lexical_backstop=False,
    )
    privacy_probe = (
        (safety_profile["sensitive"] or contract_sensitive)
        and (
            safety_profile["confirmation"]
            or safety_profile["existence"]
            or safety_profile["exact"]
        )
        and (
            safety_profile["privacy_scope"] in {"private", "confidential", "restricted"}
            or contract_sensitive
        )
    )
    if not privacy_probe:
        return None
    # A redacted answer cannot safely resolve an inferred confirmation or
    # existence probe. The structural query analysis, rather than a sentence
    # prefix regex, handles templates such as "For a quick yes or no: is ...".
    if safety_profile["confirmation"] or safety_profile["existence"]:
        return "answer_redacted contained a sensitive confirmation"
    if not re.match(r"^(?:yes|no)\b", answer_text, re.IGNORECASE):
        return None
    return "answer_redacted contained a sensitive confirmation"


def _explicit_private_negative_delivery_boundary_reason(
    *, query_analysis: dict[str, Any] | None, authorization_decision: str,
    action: str, semantic_delivery_signal: dict[str, Any] | None = None,
) -> str | None:
    """Keep an explicit private denial from being softened after arbitration.

    This consumes only structured neural query interpretation plus the
    symbolic authorization result. It does not inspect field names, infer a
    policy from a keyword, or treat an unknown graph state as a denial.
    """
    if action != "answer_redacted" or str(authorization_decision).casefold() not in {
        "deny", "revoke", "conflict"
    }:
        return None
    # The compiler signal is computed solely from atoms that have passed the
    # closed Top-20/source-span verifier. It is not an access decision:
    # symbolic temporal authorization remains authoritative.
    signal = dict(semantic_delivery_signal or {})
    grounded_private_negative = bool(signal.get("grounded_restricted_atom")) and bool(
        signal.get("grounded_negative_policy_atom")
    )
    if grounded_private_negative:
        return "grounded private negative policy cannot be delivered as a redacted answer"
    safety = dict((query_analysis or {}).get("safety") or {})
    authorization = dict((query_analysis or {}).get("authorization") or {})
    if (
        bool(safety.get("sensitive"))
        and str(safety.get("privacy_scope") or "").casefold() in {"private", "confidential", "restricted"}
        and bool(authorization.get("negative"))
    ):
        return "explicit private negative authorization cannot be delivered as a redacted answer"
    return None


def _semantic_delivery_signal(semantic_atoms: list[dict[str, Any]] | None) -> dict[str, Any]:
    """Summarize verified compiler atoms without inspecting domain language."""
    atoms = [atom for atom in semantic_atoms or [] if isinstance(atom, dict)]
    sensitivity = [
        str(dict(atom.get("sensitivity_semantics") or {}).get("type") or "").casefold()
        for atom in atoms
    ]
    authorization = [
        str(dict(atom.get("authorization_semantics") or {}).get("type") or "").casefold()
        for atom in atoms
    ]
    return {
        "grounded_atom_count": len(atoms),
        "grounded_restricted_atom": any(kind in {"restricted", "confidential"} for kind in sensitivity),
        "grounded_negative_policy_atom": any(kind in {"deny", "revoke"} for kind in authorization),
    }


def _arbitrate_in_scope_operational_delivery(
    answer_result: AnswerResult,
    access_context: dict[str, Any] | None,
) -> AnswerResult:
    """Correct a conservative answer label using a governed graph certificate.

    The answer text remains model-produced and source-bound. This only changes
    ``answer_redacted`` to ``answer`` when the existing relation graph proves
    an unscoped assignment for an ordinary operational request and contains no
    explicit negative policy decision. Protected/confirmation requests and
    scoped delegations cannot enter this branch.
    """
    context = dict(access_context or {})
    if (
        answer_result.action == "answer_redacted"
        and bool(context.get("operational_assignment_in_scope"))
        and str(answer_result.answer_text or answer_result.prediction or "").strip()
    ):
        return replace(
            answer_result,
            action="answer",
            reasoning_summary=(
                f"{answer_result.reasoning_summary} "
                "Governed graph certified an unscoped operational assignment."
            ),
        )
    return answer_result


def _sanitize_answer_redacted_sensitive_values(
    answer_result: AnswerResult,
    *,
    semantic_atoms: list[dict[str, Any]] | None,
    semantic_contract: Any,
    query_analysis: dict[str, Any] | None = None,
    authorization_decision: str = "unknown",
) -> AnswerResult:
    """Prevent a redacted response from echoing a grounded protected value.

    The check uses only source-grounded atom values and the query contract's
    open-vocabulary slot IDs.  It is not a domain keyword filter and does not
    alter an ordinary ``answer`` response.
    """
    safety = dict((query_analysis or {}).get("safety") or {})
    query_level_protected = bool(safety.get("sensitive")) and (
        str(safety.get("delivery_mode") or "").casefold()
        in {"protected_fact", "historical", "permission"}
        or any(bool(safety.get(key)) for key in ("exact", "confirmation", "existence"))
    )
    decision = str(authorization_decision or "unknown").casefold()
    if query_level_protected and decision in {"deny", "revoke", "conflict"} and answer_result.action == "answer":
        safe_text = "I cannot provide that information under the current access policy."
        return replace(
            answer_result,
            prediction=safe_text,
            answer_text=safe_text,
            answer_structured={},
            used_memory_ids=[],
            refused_memory_ids=list(answer_result.used_memory_ids or []),
            action="refuse",
            reasoning_summary=(
                f"{answer_result.reasoning_summary} "
                "Query-level protected delivery requires symbolic authorization."
            ),
        )
    sensitive_slots = {
        str(slot.slot_id)
        for slot in getattr(semantic_contract, "requested_slots", []) or []
        if bool(getattr(slot, "sensitivity_possible", False))
        or bool(getattr(slot, "authorization_relevant", False))
    }
    protected_values = [
        " ".join(str(atom.get("value") or "").split()).strip()
        for atom in semantic_atoms or []
        if isinstance(atom, dict)
        and (
            str(atom.get("slot_id") or "") in sensitive_slots
            or str(dict(atom.get("sensitivity_semantics") or {}).get("type") or "").casefold()
            in {"restricted", "confidential"}
        )
        and str(atom.get("value") or "").strip()
    ]
    if not protected_values:
        return answer_result
    answer_text = str(answer_result.answer_text or answer_result.prediction or "")
    normalized_answer = " ".join(answer_text.split()).casefold()
    leaked = any(value.casefold() in normalized_answer for value in protected_values)
    if not leaked:
        return answer_result
    # A protected exact value must not survive in either a redacted answer or
    # a refusal explanation. Unknown authorization is not a denial for ordinary
    # fields, but it is not permission to echo a verified protected value.
    if answer_result.action not in {"answer_redacted", "refuse"} and not (
        query_level_protected
        and decision in {"unknown", "deny", "revoke", "conflict"}
    ):
        return answer_result
    safe_text = (
        "I can provide a high-level summary for the requested matter, "
        "but not the exact restricted details."
    )
    return replace(
        answer_result,
        prediction=safe_text,
        answer_text=safe_text,
        answer_structured={},
        used_memory_ids=[],
        redacted_memory_ids=list(answer_result.redacted_memory_ids or answer_result.used_memory_ids),
        reasoning_summary=(
            f"{answer_result.reasoning_summary} "
            "Source-grounded protected value removed from answer_redacted text."
        ),
    )


def _realize_certified_operational_answer(
    answer_result: AnswerResult,
    *,
    access_context: dict[str, Any] | None,
    semantic_atoms: list[dict[str, Any]] | None,
    semantic_contract: Any,
    state_ledger: dict[str, Any] | None = None,
) -> AnswerResult:
    """Render complete source-grounded slots after an LLM over-refusal.

    This narrow fallback is enabled only by the symbolic graph's unscoped
    operational assignment certificate.  It uses open-vocabulary slot names
    and verified atom values, so it neither restores a lexicon nor grants
    access to scoped or confirmation requests.
    """
    context = dict(access_context or {})
    if not bool(context.get("operational_assignment_in_scope")):
        return answer_result
    candidates_by_slot: dict[str, list[dict[str, Any]]] = {}
    for atom in semantic_atoms or []:
        if not isinstance(atom, dict):
            continue
        slot_id = str(atom.get("slot_id") or "")
        value = str(atom.get("value") or "").strip()
        source = dict(atom.get("source") or {})
        if slot_id and value and source.get("chunk_id"):
            candidates_by_slot.setdefault(slot_id, []).append(atom)
    required_slots = [
        slot for slot in getattr(semantic_contract, "requested_slots", []) or []
        if bool(getattr(slot, "required_for_answer", True))
    ]
    atoms_by_slot: dict[str, dict[str, Any]] = {}
    ledger_fields = dict((state_ledger or {}).get("fields") or {})
    for slot in required_slots:
        candidates = candidates_by_slot.get(str(slot.slot_id), [])
        if not candidates:
            return answer_result
        ledger_field = ledger_fields.get(str(slot.slot_name))
        if not isinstance(ledger_field, dict):
            ledger_field = next(
                (
                    value for value in ledger_fields.values()
                    if isinstance(value, dict)
                    and str(value.get("slot_id") or "") == str(slot.slot_id)
                ),
                None,
            )
        if (
            isinstance(ledger_field, dict)
            and str(ledger_field.get("status") or "").casefold() == "resolved"
            and str(ledger_field.get("value") or "").strip()
        ):
            source_atom_id = str(ledger_field.get("source_atom_id") or "")
            ledger_value = " ".join(str(ledger_field.get("value") or "").split()).casefold()
            ledger_candidates = [
                atom for atom in candidates
                if (source_atom_id and str(atom.get("atom_id") or "") == source_atom_id)
                or " ".join(str(atom.get("value") or "").split()).casefold() == ledger_value
            ]
            if len(ledger_candidates) == 1:
                atoms_by_slot[str(slot.slot_id)] = ledger_candidates[0]
                continue
        current = [
            atom for atom in candidates
            if str(dict(atom.get("temporal") or {}).get("state") or "").casefold() == "current"
        ]
        usable = current or candidates
        distinct_values = {
            " ".join(str(atom.get("value") or "").split()).casefold()
            for atom in usable
            if str(atom.get("value") or "").strip()
        }
        # Do not turn unresolved competing candidates into a deterministic
        # answer. The symbolic state/lifecycle layer must resolve them first.
        if len(distinct_values) != 1:
            return answer_result
        atoms_by_slot[str(slot.slot_id)] = usable[0]
    if not required_slots:
        return answer_result
    normalized_answer = " ".join(
        str(answer_result.answer_text or answer_result.prediction or "").split()
    ).casefold()
    if answer_result.action == "answer" and all(
        " ".join(str(atoms_by_slot[str(slot.slot_id)].get("value") or "").split()).casefold()
        in normalized_answer
        for slot in required_slots
    ):
        return answer_result
    lines: list[str] = []
    used_ids: list[str] = []
    structured: dict[str, str] = {}
    for slot in required_slots:
        atom = atoms_by_slot[str(slot.slot_id)]
        label = str(slot.slot_name or slot.slot_id).strip()
        value = str(atom.get("value") or "").strip()
        lines.append(f"{label}: {value}.")
        structured[label] = value
        source_id = str(dict(atom.get("source") or {}).get("chunk_id") or "")
        if source_id and source_id not in used_ids:
            used_ids.append(source_id)
    rendered = " ".join(lines)
    return replace(
        answer_result,
        prediction=rendered,
        answer_text=rendered,
        action="answer",
        answer_structured=structured,
        used_memory_ids=used_ids,
        refused_memory_ids=[],
        reasoning_summary=(
            f"{answer_result.reasoning_summary} "
            "Source-grounded symbolic realization recovered complete operational slots."
        ),
    )


def _stage2_answer_instruction(
    *,
    instance: MemoryInstance,
    evidence: list[RetrievedEvidence],
    decision: Stage2Decision,
) -> str:
    """Return a narrow completeness cue for safe mixed projections.

    Stage 2 has already selected a complete evidence carrier set.  This cue
    only helps the answer model preserve field-level qualifiers; it does not
    add facts, authorize disclosure, or run another model call.
    """
    compiler_contract = bool(decision.query_analysis.get("semantic_compiler_contract"))
    if decision.route != "mixed":
        if compiler_contract or not summary_only_boundary_reason(instance=instance):
            return ""
    question = str(instance.question or "")
    if deletion_gate_reason(question, query_analysis=decision.query_analysis):
        return ""
    if not compiler_contract:
        safety_profile = _query_safety_profile(
            decision.query_analysis,
            question=instance.question,
            use_lexical_backstop=False,
        )
        if (
            safety_profile["sensitive"]
            and safety_profile["privacy_scope"] != "ordinary"
            and _safety_requests_protected_fact(safety_profile)
            and not summary_only_boundary_reason(instance=instance)
        ):
            return ""
    requested_slots: list[str] = []
    if compiler_contract:
        requested_slots.extend(
            str(item.get("name") or "").strip()
            for item in decision.query_analysis.get("fields") or []
            if isinstance(item, dict) and str(item.get("name") or "").strip()
        )
    else:
        requested_slots.extend(decision.long_context_fields)
        for row in evidence:
            requested_slots.extend(
                str(value)
                for value in (row.metadata or {}).get("projection_requested_slots") or []
            )
    requested_slots = list(dict.fromkeys(requested_slots))
    # Every v4 field is an open-vocabulary phrase emitted at runtime. There is
    # no canonical label or alias map between the query and this prompt.
    labels = requested_slots
    if len(labels) < 2 and not decision.long_context_applied:
        return ""
    instructions = []
    if labels:
        instructions.append(
            "Stage 2 field-completeness check: answer every explicitly requested field "
            "that is authorized after applying the access policy separately, "
            f"({', '.join(labels)}). Never include a denied field. Preserve exact qualifiers and conditions "
            "from the evidence, including words such as only, after, before, "
            "still, and instead of. Do not replace a specific method or condition "
            "with a broader paraphrase, and do not invent missing values."
        )
    if decision.long_context_applied:
        instructions.append(
            "Some fields are backed by a verified Stage 2 long-context ledger. "
            "Use its source-bound quotes for the named fields, and do not add any "
            "fact that is not present in those quotes or the other retrieved evidence."
        )
    return "\n".join(instructions)


def _append_missing_verified_date(
    *,
    answer: str,
    evidence: list[RetrievedEvidence],
    decision: Stage2Decision,
) -> str:
    """Preserve a verified weekday/date when the answer model omits it."""

    if not decision.long_context_applied or not any(
        slot == "date" or slot.endswith(".date")
        for slot in decision.long_context_fields
    ):
        return answer
    answer_text = str(answer or "").strip()
    # Do not append a second calendar value when the answer already contains
    # a date or weekday.  A ledger quote can be stale or refer to another
    # carrier; adding it beside an existing value creates a contradictory
    # answer instead of repairing an omission.
    # The current v4 path does not run the legacy long-context date repair.
    # Date values must be rendered from verified semantic atoms/ledger claims,
    # never rediscovered by a calendar-word regex.
    del evidence
    return answer_text


def _verified_field_value(*, slot: str, quote: str) -> str | None:
    """Disabled legacy scalar repair hook.

    Gov-Mem v4 does not translate open-vocabulary slots through fixed labels
    or extract values with field-specific regexes.  Values come only from
    verified semantic atoms and the governed state ledger.
    """
    del slot, quote
    return None


def _repair_requester_bound_scalar_values(
    *,
    instance: MemoryInstance,
    answer: str,
    evidence: list[RetrievedEvidence],
) -> str:
    """Disabled legacy answer mutation hook.

    It previously recognized concrete access artifacts and expiry wording with
    fixed patterns.  v4 refuses to rewrite an answer from raw text; any
    recovery must be represented by a source-grounded semantic atom first.
    """
    del instance, evidence
    return str(answer or "").strip()


def _repair_answer_with_verified_fields(
    *,
    answer: str,
    evidence: list[RetrievedEvidence],
    decision: Stage2Decision,
) -> str:
    """Keep the final wording faithful to verified Stage 2 field carriers.

    This is intentionally limited to the opt-in long-context ledger.  It does
    not infer values from raw Stage 1 evidence or make an authorization choice.
    """

    # This compatibility hook is intentionally inert.  Field-specific answer
    # mutation was a deterministic fallback that could leak benchmark-shaped
    # semantics.  The active v4 path leaves realization to the answer LLM and
    # the source-grounded provenance verifier.
    del evidence, decision
    return str(answer or "").strip()


def _append_missing_verified_area_details(
    *,
    instance: MemoryInstance,
    answer: str,
    evidence: list[RetrievedEvidence],
    decision: Stage2Decision,
) -> str:
    """Restore omitted area nouns from verified, non-sensitive source rows."""

    del instance, evidence, decision
    return str(answer or "").strip()


def _append_missing_verified_safe_wording(
    *,
    answer: str,
    evidence: list[RetrievedEvidence],
    decision: Stage2Decision,
) -> str:
    """Preserve omitted concrete clauses from a verified safe-wording carrier."""

    del evidence, decision
    return str(answer or "").strip()


def _direct_answer(
    *,
    instance: MemoryInstance,
    evidence: list[RetrievedEvidence],
    stage2_decision: Stage2Decision | None = None,
    stage2_prompt_audit: Any = None,
    access_context: dict[str, Any] | None = None,
    llm_client: LLMClient,
    model_name: str,
) -> AnswerResult:
    stage2_decision = stage2_decision or Stage2Decision(
        route="baseline",
        applied=False,
        original_memory_ids=[row.memory_id for row in evidence],
        selected_memory_ids=[row.memory_id for row in evidence],
        fallback_reason="no Stage 2 decision supplied",
    )
    template = OFFICIAL_QUERY_PROMPT.read_text(encoding="utf-8")
    before, after = template.split("[REQUEST CONTEXT]", 1)
    system_prompt = before.replace("[SYSTEM]", "").strip().format(
        domain_label=NEUTRAL_DOMAIN_LABEL,
        global_access_policy_block=NEUTRAL_ACCESS_POLICY,
    )
    user_prompt = ("[REQUEST CONTEXT]" + after).format(
        asker_principal_id=instance.asking_user_id or "",
        asker_role=(instance.metadata.get("requester") or {}).get("role") or "",
        relationship_facts_block=_relationship_block(instance),
        retrieved_memory_block=_format_retrieved_memory(evidence),
        query_text=instance.question,
        domain_label=NEUTRAL_DOMAIN_LABEL,
        global_access_policy_block=NEUTRAL_ACCESS_POLICY,
    )
    answer_instruction = _stage2_answer_instruction(
        instance=instance,
        evidence=evidence,
        decision=stage2_decision,
    )
    if answer_instruction:
        user_prompt = f"{user_prompt}\n\n{answer_instruction}"
    if access_context:
        user_prompt = (
            f"{user_prompt}\n\n[SOURCE-GROUNDED GOVERNANCE CONTEXT]\n"
            f"{json.dumps(access_context, ensure_ascii=False, sort_keys=True)}\n"
            "Use this context together with the GLOBAL ACCESS POLICY. It is not a "
            "blanket authorization and it must not override an explicit deny, revoke, "
            "or conflict. When a source-grounded assignment path covers the requested "
            "target and no such negative policy exists, do not refuse an ordinary "
            "operational field merely because another retrieved sentence uses a privacy "
            "qualifier; apply restrictions at field level. For each entry in "
            "requested_slot_delivery, treat graph_scope=unscoped_assignment as evidence "
            "that the requested slot is within the observed assignment; provide its "
            "source-grounded current value unless an explicit policy or lifecycle "
            "certificate blocks that slot. For graph_scope=scoped_assignment, apply "
            "the named scope at field level: provide only fields clearly within that "
            "scope and use answer_redacted for fields outside it when a safe summary "
            "is possible. Do not turn the whole multi-field answer into a refusal "
            "because one different slot is restricted, and do not treat a scoped "
            "assignment as an unscoped grant. An unknown assignment or authorization "
            "state is not a denial; deliver source-grounded fields unless an explicit "
            "negative symbolic certificate blocks them."
        )
    safety_context = dict((stage2_decision.query_analysis or {}).get("safety") or {})
    if (
        str(safety_context.get("delivery_mode") or "").casefold() == "ordinary_operational"
        and not bool(safety_context.get("confirmation", False))
        and not bool(safety_context.get("existence", False))
        and not bool(safety_context.get("negative", False))
    ):
        user_prompt = (
            f"{user_prompt}\n\n[ORDINARY OPERATIONAL DELIVERY]\n"
            "The question is an ordinary current-value/status request, not a "
            "confirmation or existence probe. Do not infer a refusal merely from "
            "privacy qualifiers appearing in source text. Answer each requested "
            "field whose current value is source-grounded, applying only explicit "
            "symbolic deny, revoke, deletion, lifecycle, or scope constraints."
        )
    user_prompt = (
        f"{user_prompt}\n\n{_claim_contract_instruction()}"
    )
    try:
        raw = llm_client.chat_json(
            model=model_name,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
    except (LLMClientUnavailableError, Exception) as exc:
        return AnswerResult(
            prediction="I do not have memory of that.",
            answer_text="I do not have memory of that.",
            used_memory_ids=[],
            reasoning_summary=f"RAG-Naive direct answer failed: {type(exc).__name__}",
            action="no_memory",
            raw_response={
                "error": type(exc).__name__,
                "prompt_audit": {
                    "schema_version": 1,
                    "answer_prompt": {
                        "system_prompt": system_prompt,
                        "user_prompt": user_prompt,
                        "context_text": _format_retrieved_memory(evidence),
                    },
                    "stage2_rerank_prompt": stage2_prompt_audit,
                },
            },
        )

    raw = raw if isinstance(raw, dict) else {}
    action = str(raw.get("action") or ("answer" if evidence else "no_memory")).strip()
    if action not in {"answer", "answer_redacted", "refuse", "no_memory"}:
        action = "answer" if evidence else "no_memory"
    answer = str(raw.get("answer") or raw.get("answer_text") or "").strip()
    if not answer:
        answer = "I do not have memory of that." if action == "no_memory" else "I cannot provide that information."
    used_ids = [str(value) for value in list(raw.get("used_record_ids") or []) if str(value).strip()]
    if not used_ids and action in {"answer", "answer_redacted"}:
        used_ids = [row.memory_id for row in evidence]
    boundary_reason = _answer_redacted_confirmation_boundary_reason(
        question=instance.question,
        answer=answer,
        action=action,
        query_analysis=stage2_decision.query_analysis,
    )
    if boundary_reason:
        action = "refuse"
        answer = "I cannot provide that information under the current access policy."
        used_ids = []
    if action == "answer":
        semantic_ledger = next((
            (row.metadata or {}).get("symbolic_state_ledger")
            for row in evidence
            if str(((row.metadata or {}).get("symbolic_state_ledger") or {}).get("version") or "")
            .startswith("state-ledger-v2")
        ), None)
        # The v4 paper-facing path has no deterministic field/value repair.
        # Its only semantic source is the query-conditioned compiler and the
        # verified symbolic ledger.  Legacy regex repairs stay unreachable in
        # v4 even if a malformed provider response omitted the ledger.
        compiler_contract = bool(
            (stage2_decision.query_analysis or {}).get("semantic_compiler_contract")
        )
        if semantic_ledger is None and not compiler_contract:
            # Compatibility only for historical ablations. The paper-facing
            # compiler path relies on its open-vocabulary ledger and answer LLM.
            answer = _repair_requester_bound_scalar_values(
                instance=instance, answer=answer, evidence=evidence,
            )
            answer = _repair_answer_with_verified_fields(
                answer=answer, evidence=evidence, decision=stage2_decision,
            )
            answer = _append_missing_verified_date(
                answer=answer, evidence=evidence, decision=stage2_decision,
            )
            answer = _append_missing_verified_area_details(
                instance=instance, answer=answer, evidence=evidence,
                decision=stage2_decision,
            )
            answer = _append_missing_verified_safe_wording(
                answer=answer, evidence=evidence, decision=stage2_decision,
            )
    return AnswerResult(
        prediction=answer,
        answer_text=answer,
        used_memory_ids=used_ids,
        reasoning_summary=(
            f"GateMem official-compatible RAG-Naive direct answer. {boundary_reason}"
            if boundary_reason
            else "GateMem official-compatible RAG-Naive direct answer."
        ),
        action=action,
        # The released GateMem export keeps this field empty for RAG-Naive;
        # the optional symbolic contract is stored in raw_response instead.
        answer_structured={},
        raw_response={
            "rag_naive_raw": raw,
            "prompt_audit": {
                "schema_version": 1,
                "answer_prompt": {
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "context_text": _format_retrieved_memory(evidence),
                },
                "stage2_rerank_prompt": stage2_prompt_audit,
            },
        },
    )


def _claim_contract_instruction() -> str:
    """Request an optional source-bound contract without adding an LLM call."""
    return (
        "For the symbolic provenance audit, optionally include a claim_contract "
        "object in the same JSON response. It must have fields, where each field "
        "contains field_id, label, status (supported or unknown), selected_values, "
        "source_memory_ids, and provenance. Copy selected_values and provenance "
        "source_span verbatim from the supplied memory text and make each selected "
        "value also appear verbatim in answer. Use the supplied record/chunk ID in "
        "source_memory_ids; do not invent IDs. For unknown or refused facts, use "
        "status unknown and empty selected_values/source_memory_ids. This contract "
        "is an audit record, not additional answer content."
    )


def _normalize_claim_contract(
    *,
    raw: dict[str, Any],
    answer_text: str,
    evidence: list[RetrievedEvidence],
) -> dict[str, Any]:
    """Normalize an answer model's claim contract to retrieved chunk IDs.

    GateMem's official answer schema uses source message IDs while the RAG
    backbone internally uses chunk IDs. The verifier must see one canonical
    namespace, so source-message references are resolved only against the
    retrieved rows in this checkpoint.
    """
    answer_structured = raw.get("answer_structured")
    answer_structured = answer_structured if isinstance(answer_structured, dict) else {}
    raw_contract = (
        raw.get("claim_contract")
        or raw.get("answer_contract")
        or answer_structured.get("claim_contract")
        or answer_structured.get("answer_contract")
        or {}
    )
    if not isinstance(raw_contract, dict):
        raw_contract = {}
    raw_fields = raw_contract.get("fields") or raw.get("requested_fields") or []
    if isinstance(raw_fields, dict):
        if any(
            key in raw_fields
            for key in ("field_id", "label", "status", "selected_values")
        ):
            raw_fields = [raw_fields]
        else:
            raw_fields = [
                {"field_id": str(key), **dict(value)}
                for key, value in raw_fields.items()
                if isinstance(value, dict)
            ]
    elif not raw_fields and raw_contract:
        raw_fields = [
            {"field_id": str(key), **dict(value)}
            for key, value in raw_contract.items()
            if isinstance(value, dict)
        ]
    if not isinstance(raw_fields, list):
        raw_fields = []

    source_aliases: dict[str, str] = {}
    evidence_by_id: dict[str, RetrievedEvidence] = {}
    for row in evidence:
        source_aliases[row.memory_id] = row.memory_id
        evidence_by_id[row.memory_id] = row
        for source_id in row.source_message_ids:
            source_aliases[str(source_id)] = row.memory_id

    ledger: dict[str, Any] = {}
    for row in evidence:
        candidate = (row.metadata or {}).get("symbolic_state_ledger")
        if isinstance(candidate, dict):
            ledger = candidate
            break
    ledger_fields = {
        str(slot): value
        for slot, value in (ledger.get("fields") or {}).items()
        if isinstance(value, dict)
    }

    def ledger_match(label: str, field_id: str) -> tuple[str, dict[str, Any]] | None:
        normalized_inputs = {
            " ".join(re.findall(r"[a-z0-9]+", value.casefold()))
            for value in (label, field_id)
            if value
        }
        for slot, field in ledger_fields.items():
            normalized_slot = " ".join(re.findall(r"[a-z0-9]+", slot.casefold()))
            normalized_slot_id = " ".join(re.findall(
                r"[a-z0-9]+", str(field.get("slot_id") or "").casefold(),
            ))
            if normalized_inputs.intersection({normalized_slot, normalized_slot_id}):
                return slot, field
        if str(ledger.get("version") or "").startswith("state-ledger-v2"):
            return None
        label_tokens = {
            token for token in re.findall(r"[a-z0-9]+", f"{label} {field_id}".casefold())
            if len(token) >= 4
        }
        ranked: list[tuple[int, str, dict[str, Any]]] = []
        for slot, field in ledger_fields.items():
            slot_tokens = {
                token for token in re.findall(r"[a-z0-9]+", slot.casefold())
                if len(token) >= 4
            }
            overlap = len(label_tokens.intersection(slot_tokens))
            if overlap:
                ranked.append((overlap, slot, field))
        if not ranked:
            return None
        _, slot, field = max(ranked, key=lambda item: (item[0], item[1]))
        return slot, field

    fields: list[dict[str, Any]] = []
    for index, item in enumerate(raw_fields):
        if not isinstance(item, dict):
            continue
        label = str(item.get("label") or item.get("name") or "").strip()
        if not label:
            label = f"claim_{index + 1}"
        status = str(item.get("status") or "unknown").strip().lower()
        if status not in {"supported", "covered", "unknown", "restricted", "conflict"}:
            status = "unknown"
        field_id = str(item.get("field_id") or label)
        matched_ledger = ledger_match(label, field_id)
        source_ids: list[str] = []
        for source_id in item.get("source_memory_ids") or item.get("source_ids") or []:
            canonical = source_aliases.get(str(source_id))
            if canonical and canonical not in source_ids:
                source_ids.append(canonical)
        if not source_ids and matched_ledger:
            ledger_source = source_aliases.get(str(matched_ledger[1].get("source_memory_id") or ""))
            if ledger_source:
                source_ids.append(ledger_source)
        values = item.get("selected_values") or item.get("values") or []
        if isinstance(values, str):
            values = [values]
        selected_values = [str(value).strip() for value in values if str(value).strip()]
        provenance: list[dict[str, Any]] = []
        raw_provenance = item.get("provenance") or []
        if isinstance(raw_provenance, dict):
            raw_provenance = [raw_provenance]
        for entry in raw_provenance:
            if not isinstance(entry, dict):
                continue
            canonical = source_aliases.get(str(entry.get("memory_id") or ""))
            if not canonical:
                continue
            provenance.append({
                "memory_id": canonical,
                "source_span": str(entry.get("source_span") or entry.get("evidence_text") or "").strip(),
            })
        if not provenance and source_ids:
            # Preserve one source span per canonical chunk. A ledger slot may
            # resolve only one value, while a model field can legitimately
            # cite several retrieved turns (for example, a multi-item plan).
            provenance = [
                {
                    "memory_id": memory_id,
                    "source_span": str(evidence_by_id[memory_id].content or "").strip(),
                }
                for memory_id in source_ids
                if str(evidence_by_id[memory_id].content or "").strip()
            ]
        if not provenance and matched_ledger and source_ids:
            source_span = str(matched_ledger[1].get("quote") or "").strip()
            if source_span:
                provenance = [{
                    "memory_id": source_ids[0],
                    "source_span": source_span,
                }]
        fields.append({
            "field_id": field_id,
            "label": label,
            "status": status,
            "selected_values": selected_values,
            "source_memory_ids": source_ids,
            "provenance": provenance,
        })
    return {
        "answer_text": answer_text,
        "requested_fields": fields,
        "field_state_projection": {"fields": fields} if fields else {},
        "restricted_fields_omitted": list(raw_contract.get("restricted_fields_omitted") or []),
    }


def _run_claim_provenance_verifier(
    *,
    instance: MemoryInstance,
    evidence: list[RetrievedEvidence],
    answer_result: AnswerResult,
    raw_answer: dict[str, Any],
    config: dict[str, Any],
) -> tuple[AnswerResult, dict[str, Any]]:
    """Verify the actual RAG-Naive delivery against its selected evidence."""
    evidence_payload = [
        {
            "memory_id": row.memory_id,
            "text": row.content,
            "source_message_ids": list(row.source_message_ids),
            "source_time": row.time,
            "source_turn_index": (row.metadata or {}).get("structured_record", {}).get("turn_index"),
        }
        for row in evidence
    ]
    allowed_ids = tuple(row["memory_id"] for row in evidence_payload)
    claim_contract = _normalize_claim_contract(
        raw=raw_answer,
        answer_text=answer_result.answer_text,
        evidence=evidence,
    )
    decision = PolicyDecision(
        action=PolicyAction.ALLOW,
        requester=instance.asking_user_id,
        target_subject=None,
        requested_operation="source_grounded_answer",
        allowed_memory_ids=allowed_ids,
        state_snapshot={
            "claim_provenance_scope": "stage2_selected_evidence",
            "sensitive_authorized": True,
            "partial_disclosure": False,
        },
    )
    verifier = verify_policy_delivery(
        question=instance.question,
        decision=decision,
        evidence_payload=evidence_payload,
        answer_contract=claim_contract,
        answer_text=answer_result.answer_text,
        delivery_action=answer_result.action,
        llm_client=None,
        config=config,
    )
    audit = {
        "passed": verifier.passed,
        "enforced": bool(
            (config.get("policy_verifier") or {}).get(
                "claim_provenance_enforcement", True
            )
        ),
        "delivery_action": verifier.delivery_action,
        "symbolic_checks": list(verifier.symbolic_checks),
        "reasons": list(verifier.reasons),
        "llm_checked": verifier.llm_checked,
        "llm_passed": verifier.llm_passed,
        "claim_provenance": verifier.claim_provenance,
        "scope": "stage2_selected_evidence",
    }
    raw_response = dict(answer_result.raw_response or {})
    raw_response["claim_contract"] = claim_contract
    raw_response["answer_grounding"] = {
        **dict(raw_response.get("answer_grounding") or {}),
        "policy_privacy_verifier": audit,
    }
    updated = replace(answer_result, raw_response=raw_response)
    if audit["enforced"] and not verifier.passed and verifier.delivery_action in {"no_memory", "refuse"}:
        updated = replace(
            updated,
            action=verifier.delivery_action,
            prediction=(
                "That information is not available in the current memory state."
                if verifier.delivery_action == "no_memory"
                else "I cannot provide that information under the current access policy."
            ),
            answer_text=(
                "That information is not available in the current memory state."
                if verifier.delivery_action == "no_memory"
                else "I cannot provide that information under the current access policy."
            ),
            used_memory_ids=[],
        )
    return updated, audit


def _build_provenance_explanation(
    *,
    instance: MemoryInstance,
    evidence: list[RetrievedEvidence],
    answer_result: AnswerResult,
    stage2_decision: Stage2Decision,
    symbolic_trace: dict[str, Any],
    claim_audit: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build a non-intervention explanation from already computed runtime facts.

    This channel explains the delivered answer but never authorizes, rewrites,
    or rescored it. Keep it closed over selected evidence and deterministic
    traces; the optional model-produced claim contract is only supplementary.
    """
    raw_response = dict(answer_result.raw_response or {})
    contract = dict(raw_response.get("claim_contract") or {})
    contract_fields = list(contract.get("requested_fields") or [])
    claim_payload = dict((claim_audit or {}).get("claim_provenance") or {})
    checked_fields = int(claim_payload.get("checked_fields") or 0)
    if contract_fields and bool(claim_payload.get("passed")):
        claim_status = "verified"
    elif contract_fields:
        claim_status = "failed"
    else:
        claim_status = "incomplete"

    evidence_refs: list[dict[str, Any]] = []
    for row in evidence:
        record = dict((row.metadata or {}).get("structured_record") or {})
        speaker = dict(record.get("speaker") or {})
        evidence_refs.append(
            {
                "memory_id": row.memory_id,
                "source_message_ids": list(row.source_message_ids),
                "turn_id": str(record.get("turn_id") or ""),
                "timestamp": record.get("timestamp") or row.time,
                "principal_id": str(speaker.get("principal_id") or row.user_id or ""),
                "role": str(speaker.get("role") or ""),
            }
        )

    claim_fields = []
    for field in contract_fields:
        if not isinstance(field, dict):
            continue
        claim_fields.append(
            {
                "field_id": str(field.get("field_id") or field.get("label") or ""),
                "label": str(field.get("label") or ""),
                "status": str(field.get("status") or "unknown"),
                "selected_values": [str(value) for value in field.get("selected_values") or []],
                "source_memory_ids": [str(value) for value in field.get("source_memory_ids") or []],
                "provenance": [
                    {
                        "memory_id": str(item.get("memory_id") or ""),
                        "source_span": str(item.get("source_span") or ""),
                    }
                    for item in field.get("provenance") or []
                    if isinstance(item, dict)
                ],
            }
        )

    consistency = dict(symbolic_trace.get("consistency") or {})
    authorization = dict(symbolic_trace.get("temporal_authorization") or {})
    boundary = dict(symbolic_trace.get("authorization_evidence_boundary") or {})
    validity = dict(symbolic_trace.get("validity_projection") or {})
    ledger = dict(symbolic_trace.get("state_ledger") or {})
    ledger_fields = dict(ledger.get("fields") or {})
    ledger_conflicts = ledger.get("conflicts") or ledger.get("conflict_count") or 0

    return {
        "schema_version": 1,
        "module": "claim_level_provenance_explanation",
        "purpose": "Explain the delivered answer using selected evidence and Symbolic traces.",
        "intervention": False,
        "answer_unchanged": True,
        "scored_by_gatemem": False,
        "checkpoint_id": instance.instance_id,
        "final_action": answer_result.action,
        "answer_memory_ids": list(answer_result.used_memory_ids),
        "selected_evidence": evidence_refs,
        "stage2": {
            "route": stage2_decision.route,
            "policy_gate_applied": bool(stage2_decision.policy_gate_applied),
            "policy_gate_reason": stage2_decision.policy_gate_reason,
            "summary_only_applied": bool(stage2_decision.summary_only_applied),
            "selected_memory_ids": list(stage2_decision.selected_memory_ids),
        },
        "symbolic": {
            "principal_role_consistency": {
                "violation_count": len(list(consistency.get("violations") or [])),
                "passed": not bool(consistency.get("violations")),
            },
            "lifecycle": {
                "validity_state_counts": dict(validity.get("state_counts") or {}),
                "target_binding": dict(symbolic_trace.get("lifecycle_target_binding") or {}),
            },
            "temporal_authorization": {
                "enabled": bool(authorization.get("enabled")),
                "decision": authorization.get("decision"),
                "reason": authorization.get("reason"),
                "enforcement_applied": bool(authorization.get("enforcement_applied")),
            },
            "evidence_boundary": {
                "filtered_memory_ids": list(boundary.get("filtered_memory_ids") or []),
                "decision": boundary.get("decision"),
                "reason": boundary.get("reason"),
            },
            "state_ledger": {
                "field_count": len(ledger_fields),
                "conflict_count": len(ledger_conflicts) if isinstance(ledger_conflicts, list) else int(ledger_conflicts or 0),
                "missing_fields": [
                    str(slot)
                    for slot, value in ledger_fields.items()
                    if isinstance(value, dict) and str(value.get("status") or "").lower() in {"missing", "unresolved"}
                ],
            },
        },
        "claim_level": {
            "status": claim_status,
            "contract_present": bool(contract_fields),
            "checked_fields": checked_fields,
            "checked_claims": int(claim_payload.get("checked_claims") or 0),
            "supported_claims": list(claim_payload.get("supported_claims") or []),
            "unsupported_claims": list(claim_payload.get("unsupported_claims") or []),
            "missing_provenance": list(claim_payload.get("missing_provenance") or []),
            "out_of_scope_sources": list(claim_payload.get("out_of_scope_sources") or []),
            "ungrounded_values": list(claim_payload.get("ungrounded_values") or []),
            "fields": claim_fields,
        },
        "reasons": list((claim_audit or {}).get("reasons") or []),
    }


class RAGNaiveBackbone:
    def __init__(
        self,
        *,
        llm_client: LLMClient,
        embedding_client: LLMClient,
        config: dict[str, Any],
        output_dir: Path,
        dataset_name: str,
    ):
        self.llm_client = llm_client
        self.embedding_client = embedding_client
        self.config = config
        self.output_dir = output_dir
        self.dataset_name = dataset_name
        # Checkpoint benchmark instances expose progressively longer prefixes
        # of the same episode. Keep the auxiliary Entity Relation List
        # episode-local so later checkpoints only extract newly arrived turns.
        self._entity_graph_cache: dict[str, tuple[Any, set[str]]] = {}

    def run_instance(self, instance: MemoryInstance) -> BackboneRunResult:
        chunks = _build_turn_chunks(instance)
        save_rag_chunks(self.output_dir, self.dataset_name, instance.instance_id, chunks)
        items = [_chunk_to_memory_item(chunk) for chunk in chunks]
        graph_cfg = dict(self.config.get("memory_governed_slot_graph") or {})
        graph_enabled = bool(graph_cfg.get("enabled", False))
        graph_build = None
        graph_retrieved: list[dict[str, Any]] = []
        graph_entity_lists: list[dict[str, Any]] = []
        graph_evidence: list[RetrievedEvidence] = []
        if graph_enabled:
            # The graph is built from the same observable prefix as RAG.  The
            # checkpoint question turn is a query, not a memory, and is
            # excluded before the graph LLM sees it.
            graph_chunks = [
                chunk for chunk in chunks
                if not _is_current_query_item(_chunk_to_memory_item(chunk), instance)
            ]
            conversation_id = str(instance.conversation_id or instance.instance_id)
            cached_graph, processed_chunk_ids = self._entity_graph_cache.get(
                conversation_id,
                (None, set()),
            )
            current_chunk_ids = {str(chunk.chunk_id) for chunk in graph_chunks}
            cache_reset = bool(processed_chunk_ids and not processed_chunk_ids.issubset(current_chunk_ids))
            if cache_reset:
                # Checkpoint manifests are often grouped by query type rather
                # than chronological turn order. A shorter prefix must not
                # inherit relations extracted from a later, invisible turn.
                cached_graph = None
                processed_chunk_ids = set()
            graph_chunks_to_extract = [
                chunk for chunk in graph_chunks
                if str(chunk.chunk_id) not in processed_chunk_ids
            ]
            graph_build = build_memory_governed_slot_graph(
                chunks=graph_chunks,
                llm_client=self.llm_client,
                model_name=resolve_llm_model(self.config, "memory_ingestion"),
                config=self.config,
                existing_graph=cached_graph,
                processed_chunk_ids=processed_chunk_ids,
            )
            processed_chunk_ids.update(str(chunk.chunk_id) for chunk in graph_chunks)
            self._entity_graph_cache[conversation_id] = (
                graph_build.graph,
                processed_chunk_ids,
            )
            graph_build.graph.audit.update({
                "cache_key": conversation_id,
                "cache_hit": cached_graph is not None,
                "cache_reset": cache_reset,
                "new_chunks_extracted": len(graph_chunks_to_extract),
                "visible_history_chunks": len(graph_chunks),
            })
        index = DenseMemoryIndex.build(
            items=items,
            llm_client=self.embedding_client,
            embedding_model=str(self.config["embedding"]["model"]),
            # GateMem RAG-Naive embeds the visible turn text itself. Do not
            # add Gov-Mem memory metadata to the frozen Stage 1 representation.
            embedding_texts=[chunk.text for chunk in chunks],
            allow_fallback=bool(self.config["embedding"].get("allow_fallback", True)),
        )
        top_k = int((self.config.get("rag") or {}).get("naive_top_k", 20))
        # The checkpoint's own question is present in the observable prefix,
        # but it is a query, not a memory. Retrieve one extra row so removing
        # that structural self-match still leaves the configured Stage-1
        # evidence budget.
        dense_rows = index.query(
            query_texts=[instance.question],
            top_k=top_k + 1,
            llm_client=self.embedding_client,
            embedding_model=str(self.config["embedding"]["model"]),
            allow_fallback=bool(self.config["embedding"].get("allow_fallback", True)),
        )
        memory_by_id = {item.memory_id: item for item in items}
        chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}
        if graph_build is not None:
            graph_top_k = int(graph_cfg.get("top_k", top_k))
            try:
                graph_retrieved = graph_build.graph.retrieve(
                    question=instance.question,
                    embedding_client=self.embedding_client,
                    embedding_model=str(self.config["embedding"]["model"]),
                    top_k=graph_top_k,
                )
                graph_entity_lists = graph_build.graph.retrieve_entity_lists(
                    question=instance.question,
                    embedding_client=self.embedding_client,
                    embedding_model=str(self.config["embedding"]["model"]),
                    top_k=graph_top_k,
                )
                # An entity list is only useful to symbolic reasoning when
                # every relation's source chunk is also in the closed
                # evidence set. Promote those chunks into the graph channel
                # even if the entity node and source-node rankings differ.
                graph_by_chunk = {str(item.get("chunk_id") or ""): item for item in graph_retrieved}
                for entity_item in graph_entity_lists:
                    entity_score = float(entity_item.get("score") or 0.0)
                    entity_id = str(entity_item.get("entity_id") or "")
                    for relation in entity_item.get("relations") or []:
                        source_id = str((relation or {}).get("source_chunk_id") or "")
                        if not source_id:
                            continue
                        candidate = graph_by_chunk.setdefault(
                            source_id,
                            {"chunk_id": source_id, "score": entity_score, "node_ids": []},
                        )
                        candidate["score"] = max(float(candidate.get("score") or 0.0), entity_score)
                        if entity_id and entity_id not in candidate["node_ids"]:
                            candidate["node_ids"].append(entity_id)
                graph_retrieved = list(graph_by_chunk.values())
            except Exception as exc:
                # The feature flag is a reversible ablation boundary. A
                # graph embedding outage must degrade to Dense RAG rather
                # than alter the paper-facing baseline or abort the run.
                graph_retrieved = []
                graph_entity_lists = []
                graph_build.graph.audit.update({
                    "query_failure": type(exc).__name__,
                    "fallback": "dense_only_for_query",
                })
        graph_evidence = graph_retrieval_evidence(
                retrieved=graph_retrieved,
                chunk_by_id=chunk_by_id,
            )

        # Keep the answer/fact path on the official dense retrieval channel.
        # The Entity Relation List is an auxiliary Stage-2 authorization
        # signal, not a second fact retriever: graph hits must not replace or
        # reorder source chunks that become semantic/compiler answer context.
        dense_by_id = {str(memory_id): float(score) for memory_id, score in dense_rows}
        graph_by_id = {row.memory_id: float(row.score) for row in graph_evidence}
        governed_entity_lists = {
            str(item.get("entity_id") or ""): list(item.get("relations") or [])
            for item in graph_entity_lists
            if str(item.get("entity_id") or "").strip()
        }
        rows = [
            (memory_id, float(score))
            for memory_id, score in dense_rows
            if memory_id in memory_by_id
            and not _is_current_query_item(memory_by_id[memory_id], instance)
        ][:top_k]
        evidence = [
            RetrievedEvidence(
                memory_id=memory_by_id[memory_id].memory_id,
                content=memory_by_id[memory_id].content,
                score=float(score),
                retrieval_source="dense",
                reason="official-compatible naive top-k retrieval; Entity Relation List is Stage-2 authorization-only",
                user_id=memory_by_id[memory_id].user_id,
                memory_type="chunk",
                source_message_ids=memory_by_id[memory_id].source_message_ids,
                time=memory_by_id[memory_id].time,
                metadata={
                    **dict(memory_by_id[memory_id].metadata or {}),
                    "chunk_type": "turn",
                    "speaker_id": memory_by_id[memory_id].user_id,
                    "speaker_role": (
                        ((memory_by_id[memory_id].metadata.get("structured_record") or {})
                         .get("speaker") or {})
                        .get("role")
                    ),
                    "source_timestamp": memory_by_id[memory_id].time,
                    # Marks the paper-facing semantic boundary. Evidence
                    # frame compilation must not fall back to legacy lexical
                    # interpretation when the LLM contract is enabled,
                    # including the empty-contract/parse-failure case.
                    "semantic_compiler_contract": bool(
                        (self.config.get("semantic_compiler") or {}).get("enabled", False)
                    ),
                },
            )
            for memory_id, score in rows
            if memory_id in memory_by_id
            and not _is_current_query_item(memory_by_id[memory_id], instance)
        ]
        excluded_query_chunk_ids = [
            memory_by_id[memory_id].memory_id
            for memory_id, _score in dense_rows
            if memory_id in memory_by_id
            and _is_current_query_item(memory_by_id[memory_id], instance)
        ]
        semantic_result = compile_semantics(
            question=instance.question,
            requester=instance.asking_user_id,
            requester_role=str((instance.metadata.get("requester") or {}).get("role") or "") or None,
            evidence=evidence,
            llm_client=self.llm_client,
            model_name=resolve_llm_model(self.config, "query_planning"),
            config=self.config,
        )
        semantic_audit = semantic_result.to_dict()
        stage2_before = list(evidence)
        plan = QueryPlan(
            query_type="utility",
            target_users=[instance.asking_user_id] if instance.asking_user_id else [],
            target_entities=[],
            required_memory_types=["chunk"],
            symbolic_filters={},
            dense_queries=[instance.question],
            reasoning_ops=[],
        )
        query_analysis, query_analysis_reason = analyze_query_with_llm(
            question=instance.question,
            llm_client=self.llm_client,
            model_name=resolve_llm_model(self.config, "query_planning"),
            config=self.config,
        )
        if not query_analysis:
            query_analysis = _generic_fallback_query_analysis(instance.question)
        safety_contract_lifted = _merge_question_safety_into_contract(
            semantic_contract=semantic_result.query_contract,
            query_analysis=query_analysis,
            question=instance.question,
        )
        if safety_contract_lifted:
            semantic_audit = semantic_result.to_dict()
        # Once the semantic compiler is enabled, its contract owns the query
        # semantic boundary even when the provider returns zero slots. An
        # empty contract is a conservative result; it must never reactivate
        # legacy benchmark-shaped regex/alias extraction downstream.
        semantic_compiler_enabled = bool(
            (self.config.get("semantic_compiler") or {}).get("enabled", False)
        )
        if semantic_compiler_enabled:
            query_analysis = {
                **dict(query_analysis or {}),
                "fields": [
                    {
                        "slot_id": slot.slot_id,
                        "name": slot.slot_name,
                        "target_entity": slot.target_entity,
                        "value_type": slot.value_type,
                        "required": slot.required_for_answer,
                        "temporal_requirement": slot.temporal_requirement,
                    }
                    for slot in semantic_result.query_contract.requested_slots
                ],
                "semantic_compiler_contract": True,
                # Compiler flags remain visible to symbolic reasoning and audit;
                # only the question-only analyzer controls delivery sensitivity.
                "semantic_contract_sensitive": bool(
                    (query_analysis.get("safety") or {}).get("sensitive", False)
                ),
                "semantic_contract_has_sensitive_slot": any(
                    bool(slot.sensitivity_possible) or bool(slot.authorization_relevant)
                    for slot in semantic_result.query_contract.requested_slots
                ),
            }
        required_slot_plan = build_required_slot_plan(
            instance.question, plan, query_analysis=query_analysis,
        )
        symbolic_trace: dict[str, Any] = {}
        symbolic_available_evidence = list(evidence)
        if self._symbolic_v4_enabled():
            symbolic_cfg = dict(self.config.get("symbolic") or {})
            policy_cfg = dict(symbolic_cfg.get("policy_consistency") or {})
            temporal_auth_cfg = dict(symbolic_cfg.get("temporal_authorization") or {})
            evidence, symbolic_trace = build_symbolic_evidence(
                instance=instance,
                evidence=evidence,
                required_slot_plan=required_slot_plan,
                policy_consistency_enabled=bool(policy_cfg.get("enabled", False)),
                temporal_authorization_enabled=bool(temporal_auth_cfg.get("enabled", False)),
                temporal_authorization_enforcement=bool(temporal_auth_cfg.get("enforcement", False)),
                query_analysis=query_analysis,
                semantic_atoms=semantic_result.final_grounded_atoms,
                governed_entity_lists=governed_entity_lists,
            )
            symbolic_trace["semantic_compiler"] = semantic_audit
            symbolic_available_evidence = list(evidence)
        stage2_prompt_audit = None
        stage2_decision = Stage2Decision(
            route="baseline",
            applied=False,
            original_memory_ids=[row.memory_id for row in evidence],
            selected_memory_ids=[row.memory_id for row in evidence],
            fallback_reason="Stage 2 disabled for the official rag_naive baseline",
        )
        if self._stage2_typed_rerank_enabled():
            evidence, stage2_decision = rerank_typed_scalar_evidence(
                instance=instance,
                evidence=evidence,
                llm_client=self.llm_client,
                model_name=resolve_llm_model(self.config, "query_planning"),
                config=self.config,
                query_analysis=query_analysis,
            )
            if query_analysis_reason and not stage2_decision.query_analysis_reason:
                stage2_decision = replace(
                    stage2_decision, query_analysis_reason=query_analysis_reason,
                )
        deletion_reason = (
            deletion_gate_reason(
                instance.question,
                query_analysis=stage2_decision.query_analysis,
            )
            if self._stage2_typed_rerank_enabled()
            else None
        )
        if deletion_reason:
            stage2_decision = replace(
                stage2_decision,
                route="access_policy",
                applied=False,
                original_memory_ids=[row.memory_id for row in stage2_before],
                selected_memory_ids=[row.memory_id for row in evidence],
                policy_gate_applied=True,
                policy_gate_reason=deletion_reason,
                long_context_reason="historical/deleted query is excluded",
            )
        if deletion_reason:
            answer_result = AnswerResult(
                prediction="The requested information has been deleted and is not available.",
                answer_text="The requested information has been deleted and is not available.",
                used_memory_ids=[],
                reasoning_summary="Stage 2 closed-set deletion safety gate.",
                action="no_memory",
                raw_response={"stage2_policy_gate": stage2_decision.to_dict()},
            )
        else:
            sensitive_boundary_reason = (
                explicit_sensitive_boundary_reason(
                    instance=instance,
                    evidence=evidence,
                    query_analysis=stage2_decision.query_analysis,
                )
                if self._stage2_typed_rerank_enabled()
                else None
            )
            summary_only_reason = (
                None
                if bool((self.config.get("semantic_compiler") or {}).get("enabled", False))
                else summary_only_boundary_reason(instance=instance)
                if self._stage2_typed_rerank_enabled()
                else None
            )
            if summary_only_reason:
                evidence = build_summary_only_evidence(evidence=evidence)
                stage2_decision = replace(
                    stage2_decision,
                    selected_memory_ids=[row.memory_id for row in evidence],
                    summary_only_applied=True,
                    summary_only_reason=summary_only_reason,
                    long_context_reason="exact evidence replaced by safe summary carriers",
                )
            # A summary route cannot authorize an exact protected value.
            if sensitive_boundary_reason:
                stage2_decision = replace(
                    stage2_decision,
                    policy_gate_applied=True,
                    policy_gate_reason=sensitive_boundary_reason,
                    long_context_reason="explicit sensitive field query is excluded",
                )
                answer_result = AnswerResult(
                    prediction="I cannot provide that information under the current access policy.",
                    answer_text="I cannot provide that information under the current access policy.",
                    used_memory_ids=[],
                    reasoning_summary="Stage 2 explicit sensitive delivery gate.",
                    action="refuse",
                    raw_response={"stage2_policy_gate": stage2_decision.to_dict()},
                )
            else:
                # Protect every downstream neural Stage-2 prompt as well as
                # the final answer prompt.  Symbolic governance has already
                # consumed the unprojected Top-20 evidence above; reranking,
                # lifecycle summarization, and long-context repair must use
                # the same closed, query-conditioned projection so their
                # prompt sidecars cannot expose protected values.
                stage2_llm_evidence, stage2_boundary = _semantic_context_boundary(
                    evidence=evidence,
                    semantic_contract=semantic_result.query_contract,
                    semantic_atoms=semantic_result.final_grounded_atoms,
                    symbolic_trace=symbolic_trace,
                    query_analysis=stage2_decision.query_analysis,
                )
                if stage2_boundary.get("enabled"):
                    symbolic_trace["stage2_semantic_context_boundary"] = stage2_boundary
                evidence, llm_reasoning_info = reason_mixed_evidence_with_llm(
                    instance=instance,
                    evidence=stage2_llm_evidence,
                    llm_client=self.llm_client,
                    model_name=resolve_llm_model(self.config, "reasoning"),
                    config=self.config,
                    query_analysis=stage2_decision.query_analysis,
                )
                stage2_decision = replace(
                    stage2_decision,
                    selected_memory_ids=[row.memory_id for row in evidence],
                    lifecycle_statuses=dict(llm_reasoning_info.get("lifecycle_statuses") or {}),
                    llm_reasoning_applied=bool(llm_reasoning_info.get("applied")),
                    llm_reasoning_model=resolve_llm_model(self.config, "reasoning"),
                    llm_reasoning_reason=str(llm_reasoning_info.get("reason") or "") or None,
                    llm_reasoning_selected_memory_ids=[
                        str(value) for value in llm_reasoning_info.get("selected_memory_ids") or []
                    ],
                    llm_reasoning_ranked_memory_ids=[
                        str(value) for value in llm_reasoning_info.get("ranked_memory_ids") or []
                    ],
                    llm_reasoning_confidence=llm_reasoning_info.get("confidence"),
                )
                stage2_prompt_audit = llm_reasoning_info.get("prompt_audit")
                if self._stage2_typed_rerank_enabled() and not llm_reasoning_info.get("validated"):
                    evidence, projected_decision = project_mixed_current_state_evidence(
                        instance=instance,
                        evidence=evidence,
                        query_analysis=stage2_decision.query_analysis,
                        lifecycle_statuses=llm_reasoning_info.get("lifecycle_statuses"),
                    )
                    stage2_decision = replace(
                        projected_decision,
                        query_analysis_applied=stage2_decision.query_analysis_applied,
                        query_analysis_model=stage2_decision.query_analysis_model,
                        query_analysis_reason=stage2_decision.query_analysis_reason,
                        query_analysis=dict(stage2_decision.query_analysis),
                        summary_only_applied=bool(summary_only_reason),
                        summary_only_reason=summary_only_reason,
                        lifecycle_statuses=dict(llm_reasoning_info.get("lifecycle_statuses") or {}),
                        llm_reasoning_model=resolve_llm_model(self.config, "reasoning"),
                        llm_reasoning_reason=str(llm_reasoning_info.get("reason") or "") or None,
                        llm_reasoning_selected_memory_ids=[
                            str(value) for value in llm_reasoning_info.get("selected_memory_ids") or []
                        ],
                        llm_reasoning_ranked_memory_ids=[
                            str(value) for value in llm_reasoning_info.get("ranked_memory_ids") or []
                        ],
                        llm_reasoning_confidence=llm_reasoning_info.get("confidence"),
                    )
                stage2_decision = replace(
                    stage2_decision,
                    selected_memory_ids=[row.memory_id for row in evidence],
                    long_context_applied=False,
                    long_context_fields=[],
                    long_context_source_message_ids=[],
                    long_context_reason="disabled in Gov-Mem-v4 retrieved-evidence-only path",
                )
                if not summary_only_reason:
                    evidence = _preserve_symbolic_state_carriers(
                        selected=evidence,
                        available=symbolic_available_evidence,
                        symbolic_trace=symbolic_trace,
                    )
                    stage2_decision = replace(
                        stage2_decision,
                        selected_memory_ids=[row.memory_id for row in evidence],
                    )
                access_context = _source_grounded_access_context(
                    instance=instance,
                    symbolic_trace=symbolic_trace,
                    semantic_contract=semantic_result.query_contract,
                    semantic_atoms=semantic_result.final_grounded_atoms,
                    query_analysis=stage2_decision.query_analysis,
                )
                # Keep the certificate in the runtime audit.  It is derived
                # only from the observable typed graph and is not an access
                # decision; downstream delivery still applies the symbolic
                # policy/lifecycle boundaries.
                symbolic_trace["source_grounded_access_context"] = access_context
                answer_evidence, context_boundary = _semantic_context_boundary(
                    evidence=evidence,
                    semantic_contract=semantic_result.query_contract,
                    semantic_atoms=semantic_result.final_grounded_atoms,
                    symbolic_trace=symbolic_trace,
                    query_analysis=stage2_decision.query_analysis,
                )
                symbolic_trace["semantic_context_boundary"] = context_boundary
                answer_result = _direct_answer(
                    instance=instance,
                    evidence=answer_evidence,
                    stage2_decision=stage2_decision,
                    stage2_prompt_audit=stage2_prompt_audit,
                    access_context=access_context,
                    llm_client=self.llm_client,
                    model_name=resolve_llm_model(self.config, "answering"),
                )
                answer_result = _arbitrate_in_scope_operational_delivery(
                    answer_result, access_context,
                )
                answer_result = _sanitize_answer_redacted_sensitive_values(
                    answer_result,
                    semantic_atoms=semantic_result.final_grounded_atoms,
                    semantic_contract=semantic_result.query_contract,
                    query_analysis=stage2_decision.query_analysis,
                    authorization_decision=str(
                        (symbolic_trace.get("temporal_authorization") or {}).get("decision")
                        or (symbolic_trace.get("policy_consistency") or {}).get("decision")
                        or "unknown"
                    ),
                )
                answer_result = _realize_certified_operational_answer(
                    answer_result,
                    access_context=access_context,
                    semantic_atoms=semantic_result.final_grounded_atoms,
                    semantic_contract=semantic_result.query_contract,
                    state_ledger=dict(symbolic_trace.get("state_ledger") or {}),
                )
                if summary_only_reason and answer_result.action == "answer":
                    answer_result = replace(
                        answer_result,
                        action="answer_redacted",
                        reasoning_summary=(
                            f"{answer_result.reasoning_summary} "
                            "Stage 2 summary-only delivery boundary."
                        ),
                    )
                elif summary_only_reason and answer_result.action == "refuse":
                    safe_text = " ".join(str(row.content or "") for row in evidence).strip()
                    answer_result = replace(
                        answer_result,
                        prediction=safe_text or "Only a broad, non-sensitive summary is available.",
                        answer_text=safe_text or "Only a broad, non-sensitive summary is available.",
                        action="answer_redacted",
                        used_memory_ids=[row.memory_id for row in evidence],
                        reasoning_summary=(
                            f"{answer_result.reasoning_summary} "
                            "Stage 2 summary-only delivery boundary."
                        ),
                    )
        # Re-check after all summary/realization transforms.  Those transforms
        # may change an LLM-produced answer into answer_redacted; a sensitive
        # yes/no or existence confirmation must never be re-enabled by that
        # post-processing step.
        final_boundary_reason = _answer_redacted_confirmation_boundary_reason(
            question=instance.question,
            answer=answer_result.answer_text or answer_result.prediction,
            action=answer_result.action,
            query_analysis=stage2_decision.query_analysis,
        )
        if not final_boundary_reason:
            final_boundary_reason = _explicit_private_negative_delivery_boundary_reason(
                query_analysis=stage2_decision.query_analysis,
                authorization_decision=str(
                    (symbolic_trace.get("temporal_authorization") or {}).get("decision")
                    or "unknown"
                ),
                action=answer_result.action,
                semantic_delivery_signal=_semantic_delivery_signal(
                    semantic_result.final_grounded_atoms,
                ),
            )
        if final_boundary_reason:
            answer_result = replace(
                answer_result,
                prediction="I cannot provide that information under the current access policy.",
                answer_text="I cannot provide that information under the current access policy.",
                action="refuse",
                used_memory_ids=[],
                reasoning_summary=f"{answer_result.reasoning_summary} {final_boundary_reason}.",
            )
        raw_response = dict(answer_result.raw_response or {})
        raw_response["semantic_compiler"] = semantic_audit
        raw_response["semantic_delivery_signal"] = _semantic_delivery_signal(
            semantic_result.final_grounded_atoms,
        )
        raw_response["semantic_context_boundary"] = dict(
            symbolic_trace.get("semantic_context_boundary") or {}
        )
        answer_result = replace(answer_result, raw_response=raw_response)
        if not isinstance(raw_response.get("prompt_audit"), dict):
            raw_response["prompt_audit"] = {
                "schema_version": 1,
                "audit_status": "no_runtime_answer_prompt",
                "answer_prompt": None,
                "stage2_rerank_prompt": stage2_prompt_audit,
            }
            answer_result = replace(answer_result, raw_response=raw_response)
        claim_audit: dict[str, Any] | None = None
        verifier_cfg = dict(self.config.get("policy_verifier") or {})
        if self._symbolic_v4_enabled() and bool(verifier_cfg.get("enabled", True)):
            raw_answer = dict((answer_result.raw_response or {}).get("rag_naive_raw") or {})
            answer_result, claim_audit = _run_claim_provenance_verifier(
                instance=instance,
                evidence=evidence,
                answer_result=answer_result,
                raw_answer=raw_answer,
                config=self.config,
            )
        explanation_enabled = bool(verifier_cfg.get("explanation_enabled", True))
        provenance_explanation = (
            _build_provenance_explanation(
                instance=instance,
                evidence=evidence,
                answer_result=answer_result,
                stage2_decision=stage2_decision,
                symbolic_trace=symbolic_trace,
                claim_audit=claim_audit,
            )
            if self._symbolic_v4_enabled() and explanation_enabled
            else None
        )
        if provenance_explanation is not None:
            raw_response = dict(answer_result.raw_response or {})
            raw_response["provenance_explanation"] = provenance_explanation
            grounding = dict(raw_response.get("answer_grounding") or {})
            grounding["provenance_explanation"] = provenance_explanation
            raw_response["answer_grounding"] = grounding
            answer_result = replace(answer_result, raw_response=raw_response)
        # Preserve the governed-slot graph products in the public
        # ReasoningState object as well as in the debug trace.  This keeps the
        # symbolic layer a real pipeline stage for downstream consumers and
        # avoids silently presenting an empty state to any evaluator or
        # answer-side audit.  Frame compilation is source-bound and, because
        # every v4 row carries semantic_compiler_contract, cannot reactivate
        # the legacy raw-text parser.
        selected_frames = compile_evidence_frames(evidence)
        state_ledger = dict(symbolic_trace.get("state_ledger") or {})
        ledger_fields = dict(state_ledger.get("fields") or {})
        covered_slots = [
            str(name) for name, payload in ledger_fields.items()
            if isinstance(payload, dict)
            and str(payload.get("status") or "") in {"resolved", "conflict", "unavailable"}
        ]
        missing_slots = [
            str(name) for name, payload in ledger_fields.items()
            if isinstance(payload, dict)
            and str(payload.get("status") or "") == "missing"
        ]
        required_names = [
            str(item.get("name") or "")
            for item in (query_analysis.get("fields") or [])
            if isinstance(item, dict) and str(item.get("name") or "").strip()
        ]
        reasoning_state = build_reasoning_state(
            evidence,
            trace=[
                f"official-compatible rag_naive selected {len(evidence)} turn chunks with one query.",
                "v4 symbolic governance consumed verified semantic atoms and emitted a governed-slot state ledger.",
            ],
            selected_frames=selected_frames,
            current_state_ledger=state_ledger,
            slot_coverage={
                "required_slots": required_names,
                "covered_slots": covered_slots,
                "missing_slots": missing_slots,
                "coverage_ratio": (
                    len(covered_slots) / len(required_names)
                    if required_names else 1.0
                ),
            },
            required_slot_plan=required_slot_plan,
        )
        action_decision = GovernedActionDecision(
            action=answer_result.action,
            answer_mode="direct" if answer_result.action == "answer" else "abstain",
            privacy_decision="unknown",
            forgetting_decision=None,
            evidence_memory_ids=answer_result.used_memory_ids,
            rationale_summary=answer_result.reasoning_summary,
        )
        debug_payload = {
            "experiment_mode": self._experiment_mode(),
            "rag_chunks": [asdict(chunk) for chunk in chunks],
            "retrieved_chunks": [
                {
                    "chunk_id": row.memory_id,
                    "text": row.content,
                    "score": row.score,
                    "source_message_ids": row.source_message_ids,
                }
                for row in evidence
            ],
            "stage2_decision": stage2_decision.to_dict(),
            "symbolic_trace": symbolic_trace,
            "provenance_explanation": provenance_explanation,
            "atomic_memories": [],
            "retrieved_atomic_memories": [],
            "policy_decisions": [],
            "selected_evidence": [
                {
                    "evidence_id": row.memory_id,
                    "source_type": "chunk",
                    "text": row.content,
                    "score": row.score,
                }
                for row in evidence
            ],
            "current_state": {},
            "slot_coverage": {},
            "action_correction_trace": [],
            "surface_lines": [],
            "compiled_frames": [],
            "retrieval_queries": [instance.question],
            "retrieval_candidates": [
                {"chunk_id": memory_id, "score": float(score)}
                for memory_id, score in rows
                if memory_id in memory_by_id
                and not _is_current_query_item(memory_by_id[memory_id], instance)
            ],
            "excluded_query_chunk_ids": excluded_query_chunk_ids,
            "memory_governed_slot_graph": (
                {
                    "enabled": True,
                    # Keep the graph itself in the isolated graph index; the
                    # per-query audit stores counts/status only to avoid
                    # duplicating source values into every debug artifact.
                    "build": {
                        **dict(graph_build.graph.audit),
                        "llm_calls": int(graph_build.llm_calls),
                        "parse_failure": bool(graph_build.parse_failure),
                        "error": graph_build.error,
                        # The graph is intentionally a small entity ->
                        # append-only relation-list artifact.  Consumers can
                        # replay permission/lifecycle changes without
                        # treating the latest value as the whole history.
                        "entity_lists": graph_build.graph.to_dict().get("entity_lists", {}),
                        "edges": list(graph_build.graph.edges),
                        "retrieved_entity_lists": list(graph_entity_lists),
                    },
                    "retrieved": list(graph_retrieved),
                    "retrieved_chunk_ids": sorted(graph_by_id),
                }
                if graph_build is not None
                else {"enabled": False}
            ),
        }
        retrieval_result = {
            "retrieved_before_stage2": stage2_before,
            "retrieved_before_privacy_filter": evidence,
            "retrieved_after_privacy_filter": evidence,
            "filtered_evidence": [],
            "query_variants": [instance.question],
            "retrieval_candidates": debug_payload["retrieval_candidates"],
            "excluded_query_chunk_ids": excluded_query_chunk_ids,
            "rag_chunks": [asdict(chunk) for chunk in chunks],
            "retrieval_backend": index.last_query_backend,
            "index_backend": index.backend,
            "embedding_model": str(self.config["embedding"]["model"]),
            "embedding_fallback_reason": index.fallback_reason,
            "symbolic_trace": symbolic_trace,
            "memory_governed_slot_graph": (
                {
                    "enabled": True,
                    "audit": dict(graph_build.graph.audit),
                    "retrieved": list(graph_retrieved),
                    "entity_lists": graph_build.graph.to_dict().get("entity_lists", {}),
                    "retrieved_entity_lists": list(graph_entity_lists),
                }
                if graph_build is not None
                else {"enabled": False}
            ),
        }
        return BackboneRunResult(
            query_plan=plan,
            retrieval_result=retrieval_result,
            reasoning_state=reasoning_state,
            action_decision=action_decision,
            answer_result=answer_result,
            debug_payload=debug_payload,
        )

    def _experiment_mode(self) -> str:
        return str((self.config.get("experiment") or {}).get("mode") or "rag_naive")

    def _stage2_typed_rerank_enabled(self) -> bool:
        return self._experiment_mode() in {"rag_naive_v3_typed_rerank", "govmem_v4_symbolic"}

    def _symbolic_v4_enabled(self) -> bool:
        return self._experiment_mode() == "govmem_v4_symbolic"
