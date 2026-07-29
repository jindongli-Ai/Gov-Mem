from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from hashlib import md5, sha256
import json
import logging
import os
from pathlib import Path
import re
from typing import Any

from gov_mem.data.schema import MemoryInstance, RetrievedEvidence
from gov_mem.governance_runtime.evidence_frames import compile_evidence_frame
from gov_mem.governance_runtime.leakage_guard import assert_runtime_payload_safe
from gov_mem.llm.client import LLMClient, LLMClientUnavailableError


LOGGER = logging.getLogger(__name__)
ANNOTATION_CACHE_CONTRACT_VERSION = "grounded-semantic-annotation-v7-complete-source-records"
SEMANTIC_ANNOTATION_BATCH_SIZE = 3
SEMANTIC_ANNOTATION_REPAIR_BATCH_SIZE = 2


@dataclass
class AtomicMemory:
    memory_id: str
    instance_id: str
    owner_user: str | None
    memory_type: str
    content: str
    entities: list[str]
    slots: dict[str, Any]
    source_message_ids: list[str]
    timestamp: str | None
    lifecycle_status: str
    access_tags: dict[str, Any]
    confidence: float


class AtomicMemoryExtractor:
    def __init__(self, *, llm_client: LLMClient, model_name: str, annotation_cache_dir: str | Path | None = None):
        self.llm_client = llm_client
        self.model_name = model_name
        self.annotation_cache_dir = Path(annotation_cache_dir) if annotation_cache_dir else None

    def extract(
        self,
        instance: MemoryInstance,
        *,
        annotation_source_message_ids: set[str] | None = None,
        required_annotation_source_message_ids: set[str] | None = None,
    ) -> list[AtomicMemory]:
        heuristic_items = self._heuristic_extract(instance)
        selected_items = heuristic_items
        if annotation_source_message_ids is not None:
            selected_items = [
                item
                for item in heuristic_items
                if set(str(value) for value in item.source_message_ids) & annotation_source_message_ids
            ]
        annotated_items = self._annotate_semantics_with_llm(
            instance,
            selected_items,
            required_source_message_ids=required_annotation_source_message_ids,
        )
        annotated_by_id = {item.memory_id: item for item in annotated_items}
        if annotation_source_message_ids is not None:
            # Dynamic record decomposition creates new atom IDs. Preserve
            # those source-grounded records and suppress only their raw
            # heuristic source fragments; otherwise the ID-based merge would
            # silently discard the very records required for certification.
            decomposed_source_keys = {
                tuple(str(value) for value in item.source_message_ids)
                for item in annotated_items
                if "::record::" in item.memory_id
            }
            merged_items = [
                annotated_by_id.get(item.memory_id, item)
                for item in heuristic_items
                if tuple(str(value) for value in item.source_message_ids) not in decomposed_source_keys
            ]
            merged_items.extend(
                item for item in annotated_items
                if "::record::" in item.memory_id
            )
            annotated_items = _merge_atomic_memories(merged_items, [])
        annotated_count = sum(
            bool((item.access_tags or {}).get("semantic_tags"))
            for item in annotated_items
        )
        if annotated_count:
            LOGGER.warning(
                "atomic_semantic_annotation instance=%s tagged=%d total=%d rate=%.4f",
                instance.instance_id,
                annotated_count,
                len(annotated_items),
                annotated_count / max(len(annotated_items), 1),
            )
            return annotated_items
        if annotation_source_message_ids is not None:
            LOGGER.warning(
                "atomic_semantic_annotation instance=%s selected=%d total=%d reason=no_grounded_selected_claims",
                instance.instance_id,
                len(selected_items),
                len(heuristic_items),
            )
            # A query-scoped annotation budget must never trigger a much
            # larger full-conversation generation fallback. Raw atoms remain
            # available for policy, lifecycle, and retrieval evidence.
            return heuristic_items
        llm_items = self._extract_with_llm(instance)
        normalized_llm_items = _normalize_llm_atomic_memories(instance, llm_items)
        LOGGER.info(
            "atomic_extraction instance=%s raw_llm=%d normalized_llm=%d heuristic=%d",
            instance.instance_id,
            len(llm_items),
            len(normalized_llm_items),
            len(heuristic_items),
        )
        if not normalized_llm_items:
            LOGGER.warning("atomic_extraction_fallback instance=%s reason=no_normalized_llm", instance.instance_id)
            return heuristic_items
        if _llm_atomic_quality_is_poor(
            llm_items=normalized_llm_items,
            raw_llm_count=len(llm_items),
            heuristic_items=heuristic_items,
        ):
            LOGGER.warning(
                "atomic_extraction_fallback instance=%s reason=quality_gate useful_llm=%d useful_heuristic=%d",
                instance.instance_id,
                _count_useful_atomic_memories(normalized_llm_items),
                _count_useful_atomic_memories(heuristic_items),
            )
            return heuristic_items
        if _needs_heuristic_backfill(
            llm_items=normalized_llm_items,
            heuristic_items=heuristic_items,
        ):
            LOGGER.info("atomic_extraction_backfill instance=%s", instance.instance_id)
            return _merge_atomic_memories(normalized_llm_items, heuristic_items)
        LOGGER.info("atomic_extraction_selected instance=%s source=llm", instance.instance_id)
        return normalized_llm_items

    def _annotate_semantics_with_llm(
        self,
        instance: MemoryInstance,
        candidates: list[AtomicMemory],
        *,
        required_source_message_ids: set[str] | None = None,
    ) -> list[AtomicMemory]:
        """Annotate grounded candidates without asking the LLM to rewrite evidence."""
        if not candidates:
            return []
        annotations: dict[str, dict[str, Any]] = {}
        record_annotations: dict[str, list[dict[str, Any]]] = {}
        retryable_representatives: set[str] = set()
        source_turn_text = {
            str(message.get("message_id") or message.get("id") or ""): str(message.get("text") or "")
            for message in list(instance.messages or [])
            if isinstance(message, dict)
        }

        def candidate_source_text(item: AtomicMemory) -> str:
            return "\n".join(
                source_turn_text.get(str(message_id), "")
                for message_id in list(item.source_message_ids or [])
                if source_turn_text.get(str(message_id), "")
            )

        for item in candidates:
            source_text = candidate_source_text(item)
            cached = self._load_cached_annotation(item=item, source_text=source_text)
            if cached is not None:
                tags, records = cached
                annotations[item.memory_id] = tags
                if records:
                    record_annotations[item.memory_id] = records
        system_prompt = (
            "Annotate each supplied evidence candidate without rewriting its text. "
            "Return JSON with field annotations. Each annotation must copy candidate_id "
            "and provide semantic_tags containing discourse_act (assertion, question, "
            "request, confirmation, update, revocation, or other), assertion_confidence "
            "from 0 to 1, event_identity with a stable entity_key and entity_surface_span. "
            "entity_surface_span must be copied verbatim from source_turn_text. For every factual "
            "assertion, also return claims: a list of objects with property_label, value_span, "
            "claim_span, and optional subject_span. property_label is an open canonical snake_case "
            "intrinsic property; value_span, claim_span, and subject_span must be copied verbatim "
            "from source_turn_text, and value_span must occur inside claim_span. Return one claim "
            "per independently asserted value or state. A factual assertion can state an empty set, "
            "absence, completion, clearance, or boolean status without naming an item; represent that "
            "as an intrinsic property and use the complete asserted state phrase as value_span (for "
            "example, a source-local statement that no items remain). Do not omit such a state merely "
            "because it has no number, entity name, or identifier. Entity names and lifecycle qualifiers such as "
            "current, old, initial, or updated are not property_label values; encode lifecycle in "
            "state_delta instead. Questions, requests, and unsupported assertions must return an "
            "empty claims list. attributes is optional compatibility output and must agree with claims. "
            "state_delta must contain operation (create, set, add, "
            "remove, supersede, revoke, or none) plus changed_fields as an object mapping "
            "each intrinsic property to its post-turn active value. For replacements, "
            "changed_fields must contain the new value under the base property key; prior "
            "values are provenance, not active changed fields. Questions and requests are "
            "not facts. source_turn_text is provenance context from the same observable "
            "turn. When candidate text is a fragment, annotate the complete asserted event "
            "state from source_turn_text, copying only values stated verbatim there. semantic_tags must also "
            "contain evidence_span: one continuous exact span from source_turn_text that supports every "
            "attribute, changed field, and entity identity in the annotation. Omit unsupported fields."
            " Also provide surface_values as an object from every attribute or changed-field "
            "key to the shortest exact value span copied from source_turn_text. Preserve "
            "punctuation, separators, units, currency, percent signs, and date formatting."
            " When a turn governs whether nearby claims may override an approved state, "
            "set state_delta.authority_effect to one of authoritative, non_authoritative, "
            "blocks_override, or none, and state_delta.authority_scope to a short structured "
            "description. Infer this from discourse meaning, not from a phrase list. "
            "When one source turn states multiple independent records or events, also return "
            "record_annotations: a list of one semantic_tags object per record. Each record annotation "
            "must have its own event_identity and one continuous evidence_span containing only that "
            "record's fields. Do not split a single record merely because it has several properties; do "
            "split distinct records even if they appear in one sentence. This is evidence decomposition, "
            "not summarization: every copied span must be verbatim from source_turn_text."
        )

        def annotate_batch(
            batch: list[AtomicMemory],
            *,
            repair: bool,
            members_by_representative: dict[str, list[AtomicMemory]],
        ) -> None:
            payload = [
                {
                    "candidate_id": item.memory_id,
                    "speaker_id": item.owner_user,
                    "timestamp": item.timestamp,
                    "source_message_ids": list(item.source_message_ids),
                    "text": item.content,
                    "source_turn_text": candidate_source_text(item),
                }
                for item in batch
            ]
            try:
                raw = self.llm_client.chat_json(
                    model=self.model_name,
                    system_prompt=system_prompt,
                    user_prompt=(
                        (
                            "Repair missing factual claim structures. For each factual assertion, "
                            "return one or more fully grounded claims; do not return an empty claims "
                            "list merely because its property is not in a fixed vocabulary. "
                            if repair else ""
                        )
                        + f"Candidates: {payload}"
                    ),
                )
            except Exception as exc:
                LOGGER.exception(
                    "atomic_semantic_annotation_error instance=%s batch_id=%s error_type=%s",
                    instance.instance_id,
                    batch[0].memory_id if batch else "empty",
                    type(exc).__name__,
                )
                return
            # Providers/models may name an evidence-grounding batch either
            # `annotations` or `evidence_annotations`; both carry the same
            # per-candidate closed schema and are validated identically.
            rows = (
                raw.get("annotations", raw.get("evidence_annotations"))
                if isinstance(raw, dict)
                else None
            )
            if not isinstance(rows, list):
                LOGGER.warning(
                    "atomic_semantic_annotation_schema instance=%s batch_id=%s response_type=%s keys=%s",
                    instance.instance_id,
                    batch[0].memory_id if batch else "empty",
                    type(raw).__name__,
                    sorted(str(key) for key in raw.keys()) if isinstance(raw, dict) else [],
                )
                return
            for row in rows:
                if not isinstance(row, dict):
                    continue
                candidate_id = str(row.get("candidate_id") or "")
                if candidate_id:
                    candidate = next((item for item in batch if item.memory_id == candidate_id), None)
                    source_text = candidate_source_text(candidate) if candidate is not None else ""
                    normalized = _ground_semantic_tags(
                        _annotation_semantic_payload(row), source_text=source_text
                    )
                    normalized_records = _ground_record_annotations(
                        row.get("record_annotations"), source_text=source_text
                    )
                    if _semantic_annotation_is_informative(normalized) or normalized_records:
                        cache_tags = normalized if _semantic_annotation_is_informative(normalized) else normalized_records[0]
                        for member in members_by_representative.get(candidate_id, [candidate] if candidate else []):
                            annotations[member.memory_id] = cache_tags
                            if normalized_records:
                                record_annotations[member.memory_id] = normalized_records
                            self._store_cached_annotation(
                                item=member,
                                source_text=candidate_source_text(member),
                                semantic_tags=cache_tags,
                                record_semantic_tags=normalized_records,
                            )
                    elif (
                        not repair
                        and str(normalized.get("discourse_act") or "").lower()
                        in {"assertion", "confirmation", "update", "revocation"}
                    ):
                        # Retry only a source turn that was recognized as a
                        # factual assertion but failed claim grounding. Pure
                        # questions/requests correctly need no second call.
                        retryable_representatives.add(candidate_id)

        def compile_required_record_batch(
            batch: list[AtomicMemory],
            *,
            members_by_representative: dict[str, list[AtomicMemory]],
        ) -> None:
            """Compile selected factual sources into bounded, attested records."""
            if not batch:
                return
            payload = [
                {
                    "candidate_id": item.memory_id,
                    "source_message_ids": list(item.source_message_ids),
                    "source_turn_text": candidate_source_text(item),
                    "existing_record_count": max(
                        (
                            len(record_annotations.get(member.memory_id) or [])
                            for member in members_by_representative.get(item.memory_id, [])
                        ),
                        default=0,
                    ),
                }
                for item in batch
            ]
            try:
                raw = self.llm_client.chat_json(
                    model=self.model_name,
                    system_prompt=(
                        "You compile source-grounded records for a governed memory reranker. Each supplied source "
                        "was already selected as possible factual utility evidence, but that is not authorization. "
                        "Return every independently stated event/state in that source. This is a completeness audit, "
                        "not a request to improve just one existing record. Count all independently resolvable records "
                        "before responding and emit one row for each. In an enumeration, recap, comparison, or compound "
                        "sentence, each separately stated event/state is a separate record even if they share a heading, "
                        "type, or some values. Each evidence_span must isolate that record and must not include the fields "
                        "of another independent record. Do not return "
                        "policy, permission, requester questions, inferred values, or information from another turn. "
                        "A selected operational instruction or logistics sentence is factual evidence when it states "
                        "the current plan, route, location, handoff, timing, or state to use; preserve each such "
                        "independently resolvable field even when the heuristic candidate text contains only one "
                        "fragment from the source turn. Treat the complete source_turn_text as the bounded record "
                        "closure, while copying every value and evidence span exactly from it. "
                        "Every event_identity.entity_surface_span, evidence_span, claim value_span, and claim_span "
                        "must be verbatim in the candidate's source_turn_text. evidence_span must contain only one "
                        "record's fields. Use open snake_case property labels. Return JSON only."
                    ),
                    user_prompt=(
                        "Return {\"records\":[{\"candidate_id\":string,\"discourse_act\":string,"
                        "\"assertion_confidence\":number,\"event_identity\":object,\"attributes\":object,"
                        "\"surface_values\":object,\"claims\":list,\"state_delta\":object,"
                        "\"evidence_span\":string}]}. Emit no row when a source does not contain a factual "
                        "record. existing_record_count is only an audit hint and never a cap: return the complete "
                        "source-grounded record set. Candidates: " + repr(payload)
                    ),
                )
            except Exception as exc:
                LOGGER.warning(
                    "required_record_compilation_error instance=%s batch_id=%s error_type=%s",
                    instance.instance_id,
                    batch[0].memory_id,
                    type(exc).__name__,
                )
                return
            rows = raw.get("records") if isinstance(raw, dict) else None
            if not isinstance(rows, list):
                return
            records_by_candidate: dict[str, list[dict[str, Any]]] = {}
            for row in rows:
                if not isinstance(row, dict):
                    continue
                candidate_id = str(row.get("candidate_id") or "")
                candidate = next((item for item in batch if item.memory_id == candidate_id), None)
                if candidate is None:
                    continue
                tags = _ground_semantic_tags(
                    _annotation_semantic_payload(row), source_text=candidate_source_text(candidate)
                )
                identity = dict(tags.get("event_identity") or {})
                if not _semantic_annotation_is_informative(tags) or not identity.get("entity_key"):
                    continue
                records_by_candidate.setdefault(candidate_id, []).append(tags)
            for candidate_id, records in records_by_candidate.items():
                deduped = _dedupe_record_semantic_tags(records)
                if not deduped:
                    continue
                existing_count = max(
                    (
                        len(record_annotations.get(member.memory_id) or [])
                        for member in members_by_representative.get(candidate_id, [])
                    ),
                    default=0,
                )
                # Completion can only increase record coverage. A weaker
                # retry must never collapse an already decomposed source.
                if existing_count and len(deduped) <= existing_count:
                    continue
                for member in members_by_representative.get(candidate_id, []):
                    annotations[member.memory_id] = deduped[0]
                    record_annotations[member.memory_id] = deduped
                    self._store_cached_annotation(
                        item=member,
                        source_text=candidate_source_text(member),
                        semantic_tags=deduped[0],
                        record_semantic_tags=deduped,
                    )

        uncached = [item for item in candidates if item.memory_id not in annotations]
        members_by_source: dict[tuple[tuple[str, ...], str], list[AtomicMemory]] = {}
        for item in uncached:
            key = (tuple(str(value) for value in item.source_message_ids), candidate_source_text(item))
            members_by_source.setdefault(key, []).append(item)
        representatives = [members[0] for members in members_by_source.values()]
        members_by_representative = {
            representative.memory_id: members
            for members in members_by_source.values()
            for representative in members[:1]
        }
        # Keep each LLM request focused on a few source turns. Long episodes
        # otherwise produce oversized generation requests and diffuse claim
        # boundaries across unrelated local events.
        for start in range(0, len(representatives), SEMANTIC_ANNOTATION_BATCH_SIZE):
            annotate_batch(
                representatives[start : start + SEMANTIC_ANNOTATION_BATCH_SIZE],
                repair=False,
                members_by_representative=members_by_representative,
            )
        repair_rows = [
            representative
            for representative in representatives
            if representative.memory_id in retryable_representatives
            and not any(member.memory_id in annotations for member in members_by_representative[representative.memory_id])
        ]
        for start in range(0, len(repair_rows), SEMANTIC_ANNOTATION_REPAIR_BATCH_SIZE):
            annotate_batch(
                repair_rows[start : start + SEMANTIC_ANNOTATION_REPAIR_BATCH_SIZE],
                repair=True,
                members_by_representative=members_by_representative,
            )
        # A utility locator's selected fact source is a closed, provenance
        # bearing proposal. If batch annotation omitted it entirely, give only
        # that source one narrow repair call. This does not invent a fact or
        # broaden retrieval; the repaired output still needs exact spans.
        required_ids = {str(value) for value in list(required_source_message_ids or []) if str(value)}
        omitted_required_rows = [
            representative
            for representative in representatives
            if set(str(value) for value in representative.source_message_ids) & required_ids
            and not any(
                member.memory_id in annotations
                for member in members_by_representative[representative.memory_id]
            )
        ]
        for start in range(0, len(omitted_required_rows), SEMANTIC_ANNOTATION_REPAIR_BATCH_SIZE):
            annotate_batch(
                omitted_required_rows[start : start + SEMANTIC_ANNOTATION_REPAIR_BATCH_SIZE],
                repair=True,
                members_by_representative=members_by_representative,
            )
        still_omitted_required_rows = [
            representative
            for representative in omitted_required_rows
            if not any(
                member.memory_id in annotations
                for member in members_by_representative[representative.memory_id]
            )
        ]
        for start in range(0, len(still_omitted_required_rows), SEMANTIC_ANNOTATION_REPAIR_BATCH_SIZE):
            compile_required_record_batch(
                still_omitted_required_rows[start : start + SEMANTIC_ANNOTATION_REPAIR_BATCH_SIZE],
                members_by_representative=members_by_representative,
            )
        # Normal annotation may emit one valid record from a source that
        # actually contains several independent records. Re-audit only the
        # selected factual-source closure and accept a retry solely when it
        # increases the number of source-grounded records.
        partially_decomposed_required_rows = [
            representative
            for representative in representatives
            if set(str(value) for value in representative.source_message_ids) & required_ids
            and any(
                member.memory_id in annotations
                for member in members_by_representative[representative.memory_id]
            )
        ]
        for start in range(0, len(partially_decomposed_required_rows), SEMANTIC_ANNOTATION_REPAIR_BATCH_SIZE):
            compile_required_record_batch(
                partially_decomposed_required_rows[start : start + SEMANTIC_ANNOTATION_REPAIR_BATCH_SIZE],
                members_by_representative=members_by_representative,
            )
        normalized_items = []
        emitted_record_sources: set[tuple[tuple[str, ...], str]] = set()
        for item in candidates:
            tags = annotations.get(item.memory_id)
            if tags is None:
                normalized_items.append(item)
                continue
            source_key = (
                tuple(str(value) for value in item.source_message_ids),
                candidate_source_text(item),
            )
            records = record_annotations.get(item.memory_id) or []
            if records:
                if source_key not in emitted_record_sources:
                    emitted_record_sources.add(source_key)
                    normalized_items.extend(
                        _materialize_record_atom(item=item, semantic_tags=record)
                        for record in records
                    )
                continue
            annotated = replace(
                item,
                lifecycle_status=_lifecycle_from_semantic_tags(item.lifecycle_status, tags),
                access_tags={
                    **dict(item.access_tags or {}),
                    "semantic_tags": tags,
                    "semantic_annotation_source": "llm_grounded_candidate",
                },
            )
            normalized_items.append(_apply_attested_evidence_span(annotated, tags))
        return normalized_items

    def _load_cached_annotation(
        self, *, item: AtomicMemory, source_text: str
    ) -> tuple[dict[str, Any], list[dict[str, Any]]] | None:
        path = self._annotation_cache_path(item=item, source_text=source_text)
        if path is None or not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(payload, dict) or payload.get("contract_version") != ANNOTATION_CACHE_CONTRACT_VERSION:
            return None
        if payload.get("model_name") != self.model_name:
            return None
        if payload.get("candidate_fingerprint") != self._annotation_fingerprint(item=item, source_text=source_text):
            return None
        tags = _ground_semantic_tags(payload.get("semantic_tags"), source_text=source_text)
        if not _semantic_annotation_is_informative(tags):
            return None
        return tags, _ground_record_annotations(
            payload.get("record_semantic_tags"), source_text=source_text
        )

    def _store_cached_annotation(
        self,
        *,
        item: AtomicMemory,
        source_text: str,
        semantic_tags: dict[str, Any],
        record_semantic_tags: list[dict[str, Any]] | None = None,
    ) -> None:
        path = self._annotation_cache_path(item=item, source_text=source_text)
        if path is None:
            return
        payload = {
            "contract_version": ANNOTATION_CACHE_CONTRACT_VERSION,
            "model_name": self.model_name,
            "candidate_fingerprint": self._annotation_fingerprint(item=item, source_text=source_text),
            "semantic_tags": semantic_tags,
            "record_semantic_tags": list(record_semantic_tags or []),
        }
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(payload, ensure_ascii=True, sort_keys=True) + "\n", encoding="utf-8")
            os.replace(temporary, path)
        except OSError:
            LOGGER.warning("atomic_semantic_annotation_cache_write_failed path=%s", path)

    def _annotation_cache_path(self, *, item: AtomicMemory, source_text: str) -> Path | None:
        if self.annotation_cache_dir is None:
            return None
        return self.annotation_cache_dir / f"{self._annotation_fingerprint(item=item, source_text=source_text)}.json"

    def _annotation_fingerprint(self, *, item: AtomicMemory, source_text: str) -> str:
        payload = {
            "contract_version": ANNOTATION_CACHE_CONTRACT_VERSION,
            "model_name": self.model_name,
            "owner_user": item.owner_user,
            "timestamp": item.timestamp,
            "source_message_ids": list(item.source_message_ids or []),
            "candidate_text": item.content,
            "source_turn_text": source_text,
        }
        encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        return sha256(encoded.encode("utf-8")).hexdigest()

    def _extract_with_llm(self, instance: MemoryInstance) -> list[dict[str, Any]]:
        messages = list(instance.messages)
        observable_payload = {
            "messages": [
                {
                    "message_id": message.get("message_id"),
                    "speaker_id": message.get("speaker_id"),
                    "speaker_role": message.get("speaker_role"),
                    "timestamp": message.get("timestamp"),
                    "text": message.get("text"),
                }
                for message in messages
            ]
        }
        assert_runtime_payload_safe(observable_payload, context=f"atomic_memory:{instance.instance_id}")
        try:
            raw = self.llm_client.chat_json(
                model=self.model_name,
                system_prompt=(
                    "Extract atomic memories from the observable conversation. "
                    "Preserve exact surface strings for dates, times, locations, providers, procedures, arrival times, instructions, allergies, medications, cancellations, updates, permissions, and forgetting/deletion instructions. "
                    "Each atomic memory should contain one useful fact or event. "
                    "For every item also return semantic_tags with: discourse_act "
                    "(assertion, question, request, confirmation, update, revocation, or other), "
                    "assertion_confidence from 0 to 1, event_identity with a stable entity_key and "
                    "entity_surface_span copied verbatim from the source message, "
                    "evidence_span as one continuous exact source-message span supporting every returned "
                    "attribute, changed field, and entity identity, "
                    "attributes as a flat object of canonical snake_case intrinsic property names to "
                    "grounded values. Omit entity names and lifecycle qualifiers such as current, old, "
                    "initial, or updated from attribute keys; encode lifecycle in state_delta. Also "
                    "return state_delta with operation "
                    "(create, set, add, remove, supersede, revoke, or none) plus changed fields. "
                    "Questions and requests are not factual assertions unless the same source turn "
                    "contains an explicit assertion. Do not infer facts from requested content. "
                    "Do not answer the user question."
                ),
                user_prompt=f"Return JSON with field atomic_memories. Messages: {observable_payload['messages']}",
            )
            LOGGER.warning(
                "atomic_extraction_llm_response instance=%s response_type=%s keys=%s item_type=%s item_count=%d first_item_keys=%s",
                instance.instance_id,
                type(raw).__name__,
                sorted(str(key) for key in raw.keys()) if isinstance(raw, dict) else [],
                type(raw.get("atomic_memories")).__name__ if isinstance(raw, dict) else "none",
                len(raw.get("atomic_memories") or []) if isinstance(raw, dict) and isinstance(raw.get("atomic_memories"), list) else 0,
                sorted(str(key) for key in raw["atomic_memories"][0].keys())
                if isinstance(raw, dict)
                and isinstance(raw.get("atomic_memories"), list)
                and raw["atomic_memories"]
                and isinstance(raw["atomic_memories"][0], dict)
                else [],
            )
            if isinstance(raw, dict) and isinstance(raw.get("atomic_memories"), list):
                return [dict(item or {}) for item in raw["atomic_memories"] if isinstance(item, dict)]
        except Exception as exc:
            LOGGER.exception(
                "atomic_extraction_llm_error instance=%s error_type=%s",
                instance.instance_id,
                type(exc).__name__,
            )
        return []

    def _heuristic_extract(self, instance: MemoryInstance) -> list[AtomicMemory]:
        items: list[AtomicMemory] = []
        for idx, message in enumerate(instance.messages):
            text = str(message.get("text") or "").strip()
            if not text:
                continue
            segments = _extract_atomic_segments(text)
            for seg_idx, segment in enumerate(segments):
                evidence = RetrievedEvidence(
                    memory_id=f"{instance.instance_id}_msg_{idx:04d}_{seg_idx:02d}",
                    content=segment,
                    score=1.0,
                    retrieval_source="message",
                    reason="atomic extraction source",
                    user_id=message.get("speaker_id"),
                    memory_type="message",
                    scope="message",
                    entities=[],
                    time=message.get("timestamp"),
                    source_message_ids=[str(message.get("message_id"))],
                    metadata={},
                )
                frame = compile_evidence_frame(evidence)
                items.append(
                    AtomicMemory(
                        memory_id=md5(f"{instance.instance_id}:{idx}:{seg_idx}:{segment}".encode("utf-8")).hexdigest()[:12],
                        instance_id=instance.instance_id,
                        owner_user=message.get("speaker_id"),
                        memory_type=frame.frame_type,
                        content=segment,
                        entities=_infer_entities_from_frame(frame),
                        slots=dict(frame.slots),
                        source_message_ids=[str(message.get("message_id"))],
                        timestamp=message.get("timestamp"),
                        lifecycle_status=_infer_atomic_lifecycle(frame, segment),
                        access_tags={
                            "surface_spans": dict(frame.surface_spans),
                            "frame_type": frame.frame_type,
                            "segment_index": seg_idx,
                            "extraction_source": "heuristic",
                        },
                        confidence=0.58 if seg_idx == 0 else 0.56,
                    )
                )
        return items


def atomic_memory_to_dict(memory: AtomicMemory) -> dict[str, Any]:
    return asdict(memory)


def _normalize_llm_atomic_memories(instance: MemoryInstance, raw_items: list[dict[str, Any]]) -> list[AtomicMemory]:
    message_by_id = {
        str(message.get("message_id")): message
        for message in instance.messages
        if message.get("message_id") is not None
    }
    items: list[AtomicMemory] = []
    for idx, item in enumerate(raw_items):
        content = _recover_atomic_content(item)
        if not content:
            continue
        source_message_ids = [str(x) for x in item.get("source_message_ids", []) if str(x).strip()]
        fallback_message = message_by_id.get(source_message_ids[0]) if source_message_ids else None
        evidence = RetrievedEvidence(
            memory_id=str(item.get("memory_id") or f"{instance.instance_id}_amem_{idx:04d}"),
            content=content,
            score=float(item.get("confidence", 0.66)),
            retrieval_source="atomic_memory",
            reason="llm atomic extraction",
            user_id=item.get("owner_user") or (fallback_message or {}).get("speaker_id"),
            memory_type=str(item.get("memory_type") or "general_fact"),
            scope="atomic_memory",
            entities=[str(x) for x in item.get("entities", []) if str(x).strip()],
            time=item.get("timestamp") or (fallback_message or {}).get("timestamp"),
            source_message_ids=source_message_ids,
            metadata={},
        )
        frame = compile_evidence_frame(evidence)
        source_turn_text = "\n".join(
            str(message_by_id.get(message_id, {}).get("text") or "")
            for message_id in source_message_ids
            if str(message_by_id.get(message_id, {}).get("text") or "")
        )
        semantic_tags = _ground_semantic_tags(
            item.get("semantic_tags"), source_text=source_turn_text
        )
        slots = dict(item.get("slots") or {})
        for key, value in frame.slots.items():
            # Deterministic extraction is grounded in the recovered source
            # sentence and repairs vague or malformed LLM slot values.
            slots[key] = value
        surface_spans = dict(item.get("surface_spans") or {})
        for key, value in frame.surface_spans.items():
            surface_spans[key] = value
        memory_type = str(item.get("memory_type") or "").strip() or frame.frame_type or "general_fact"
        if memory_type == "general_fact" and frame.frame_type != "general_fact":
            memory_type = frame.frame_type
        entities = [str(x) for x in item.get("entities", []) if str(x).strip()]
        if not entities:
            entities = _infer_entities_from_frame(frame)
        normalized_item = AtomicMemory(
                memory_id=str(item.get("memory_id") or evidence.memory_id),
                instance_id=instance.instance_id,
                owner_user=item.get("owner_user") or (fallback_message or {}).get("speaker_id"),
                memory_type=memory_type,
                content=content,
                entities=entities,
                slots=slots,
                source_message_ids=source_message_ids,
                timestamp=item.get("timestamp") or (fallback_message or {}).get("timestamp"),
                lifecycle_status=str(item.get("lifecycle_status") or _infer_atomic_lifecycle(frame, content) or "active"),
                access_tags={
                    **dict(item.get("access_tags") or {}),
                    "surface_spans": surface_spans,
                    "frame_type": frame.frame_type,
                    "extraction_source": "llm",
                    "semantic_tags": semantic_tags,
                },
                confidence=float(item.get("confidence", 0.66)),
            )
        items.append(_apply_attested_evidence_span(normalized_item, semantic_tags))
    return _merge_atomic_memories(items, [])


def _normalize_semantic_tags(raw: Any) -> dict[str, Any]:
    item = dict(raw or {}) if isinstance(raw, dict) else {}
    discourse_act = str(item.get("discourse_act") or "unknown").strip().lower()
    valid_acts = {"assertion", "question", "request", "confirmation", "update", "revocation", "other", "unknown"}
    if discourse_act not in valid_acts:
        discourse_act = "unknown"
    raw_state_delta = item.get("state_delta")
    state_delta_input = raw_state_delta if isinstance(raw_state_delta, dict) else {}
    operation = str(state_delta_input.get("operation") or "none").strip().lower()
    valid_operations = {"create", "set", "add", "remove", "supersede", "revoke", "none"}
    if operation not in valid_operations:
        operation = "none"
    try:
        assertion_confidence = min(1.0, max(0.0, float(item.get("assertion_confidence", 0.0))))
    except (TypeError, ValueError):
        assertion_confidence = 0.0
    state_delta = dict(state_delta_input)
    state_delta["operation"] = operation
    return {
        "discourse_act": discourse_act,
        "assertion_confidence": assertion_confidence,
        "event_identity": dict(item.get("event_identity")) if isinstance(item.get("event_identity"), dict) else {},
        "attributes": _normalize_attribute_map(item.get("attributes")),
        "claims": _normalize_grounded_claims(item.get("claims")),
        "surface_values": _normalize_attribute_map(item.get("surface_values")),
        "state_delta": state_delta,
        "evidence_span": str(item.get("evidence_span") or "").strip(),
    }


def _annotation_semantic_payload(row: dict[str, Any]) -> dict[str, Any]:
    """Accept equivalent nested and row-level annotation JSON schemas."""
    nested = dict(row.get("semantic_tags") or {}) if isinstance(row.get("semantic_tags"), dict) else {}
    # Some compatible models place the evidence-bearing fields alongside the
    # semantic_tags envelope. Nested keys win when both are supplied.
    for key in (
        "discourse_act",
        "assertion_confidence",
        "event_identity",
        "attributes",
        "claims",
        "surface_values",
        "state_delta",
        "evidence_span",
    ):
        if key not in nested and key in row:
            nested[key] = row[key]
    return nested


def _ground_semantic_tags(raw: Any, *, source_text: str) -> dict[str, Any]:
    """Retain semantic fields only when their value is cited in source text."""
    tags = _normalize_semantic_tags(raw)
    source = str(source_text or "")
    evidence_span = str(tags.get("evidence_span") or "").strip()
    if not evidence_span or evidence_span not in source:
        return {
            **tags,
            "event_identity": {},
            "attributes": {},
            "claims": [],
            "surface_values": {},
            "state_delta": {**dict(tags.get("state_delta") or {}), "changed_fields": {}},
            "evidence_span": "",
        }
    surface_values = dict(tags.get("surface_values") or {})
    raw_attributes = dict(tags.get("attributes") or {})
    state_delta = tags.get("state_delta")
    state_delta = state_delta if isinstance(state_delta, dict) else {}
    changed_fields = state_delta.get("changed_fields")
    # Models occasionally emit a list or scalar for this open-schema field.
    # Treat that malformed subfield as absent so one bad annotation cannot
    # discard the whole checkpoint.
    raw_changed_fields = changed_fields if isinstance(changed_fields, dict) else {}
    grounded_claims = [
        claim
        for claim in list(tags.get("claims") or [])
        if claim["value_span"] in evidence_span
        and claim["claim_span"] in evidence_span
        and claim["value_span"] in claim["claim_span"]
        and (not claim.get("subject_span") or claim["subject_span"] in evidence_span)
    ]

    def grounded_value(key: str) -> str | None:
        # A duplicate surface_values entry is helpful but not required when
        # the canonical attribute value itself is verbatim in the attested
        # span. Values that are not present remain fail-closed.
        for candidate in (
            surface_values.get(key), raw_attributes.get(key), raw_changed_fields.get(key)
        ):
            if isinstance(candidate, (str, int, float, bool)):
                value = str(candidate).strip()
                if value and value in evidence_span:
                    return value
        return None

    attributes = {
        key: value
        for key in dict(tags.get("attributes") or {})
        if (value := grounded_value(str(key))) is not None
    }
    # Claim values are independently span-validated and therefore provide a
    # usable open-schema fact representation even when the flat legacy map is
    # omitted by the annotating model.
    for claim in grounded_claims:
        attributes.setdefault(claim["property_label"], claim["value_span"])
    state_delta = dict(state_delta)
    changed_fields = state_delta.get("changed_fields")
    grounded_changed_fields = {}
    if isinstance(changed_fields, dict):
        for key in changed_fields:
            grounded_value_text = grounded_value(str(key))
            if grounded_value_text is not None:
                grounded_changed_fields[key] = grounded_value_text
    state_delta["changed_fields"] = grounded_changed_fields
    event_identity = dict(tags.get("event_identity") or {})
    entity_key = re.sub(r"[^a-z0-9]+", "_", str(event_identity.get("entity_key") or "").lower()).strip("_")
    entity_span = str(event_identity.get("entity_surface_span") or "").strip()
    if not entity_key or not entity_span or entity_span not in evidence_span:
        event_identity = {}
    else:
        event_identity = {"entity_key": entity_key, "entity_surface_span": entity_span}
    return {
        **tags,
        "event_identity": event_identity,
        "attributes": attributes,
        "claims": grounded_claims,
        "surface_values": {key: value for key, value in surface_values.items() if grounded_value(str(key))},
        "state_delta": state_delta,
        "evidence_span": evidence_span,
    }


def _ground_record_annotations(raw: Any, *, source_text: str) -> list[dict[str, Any]]:
    """Validate episode-local record decomposition proposed by the annotator."""
    if not isinstance(raw, list):
        return []
    records: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for item in raw:
        if not isinstance(item, dict):
            continue
        tags = _ground_semantic_tags(_annotation_semantic_payload(item), source_text=source_text)
        if not _semantic_annotation_is_informative(tags):
            continue
        identity = dict(tags.get("event_identity") or {})
        evidence_span = str(tags.get("evidence_span") or "").strip()
        # A record needs a locally attested identity. Otherwise a list could
        # silently turn one event into arbitrary field groupings.
        entity_key = str(identity.get("entity_key") or "").strip()
        if not entity_key or not evidence_span:
            continue
        key = (entity_key, evidence_span)
        if key not in seen:
            seen.add(key)
            records.append(tags)
    return records


def _dedupe_record_semantic_tags(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Preserve distinct source-local records while removing repeated LLM rows."""
    result: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for tags in records:
        identity = dict(tags.get("event_identity") or {})
        key = (
            str(identity.get("entity_key") or "").strip(),
            str(tags.get("evidence_span") or "").strip(),
        )
        if not all(key) or key in seen:
            continue
        seen.add(key)
        result.append(tags)
    return result


def _materialize_record_atom(*, item: AtomicMemory, semantic_tags: dict[str, Any]) -> AtomicMemory:
    """Create a source-grounded dynamic atom for one independently attested record."""
    evidence_span = str(semantic_tags.get("evidence_span") or "").strip()
    identity = dict(semantic_tags.get("event_identity") or {})
    record_key = f"{item.memory_id}:{identity.get('entity_key', '')}:{evidence_span}"
    record_id = md5(record_key.encode("utf-8")).hexdigest()[:12]
    annotated = replace(
        item,
        memory_id=f"{item.memory_id}::record::{record_id}",
        content=evidence_span,
        # Only the record's grounded semantic fields can reach graph slots.
        # The source candidate may have contained fields of neighboring records.
        slots={},
        entities=[str(identity.get("entity_surface_span") or "").strip()],
        lifecycle_status=_lifecycle_from_semantic_tags(item.lifecycle_status, semantic_tags),
        access_tags={
            **dict(item.access_tags or {}),
            "surface_spans": dict(semantic_tags.get("surface_values") or {}),
            "semantic_tags": semantic_tags,
            "semantic_annotation_source": "llm_grounded_record_decomposition",
        },
    )
    return _apply_attested_evidence_span(annotated, semantic_tags)


def _apply_attested_evidence_span(item: AtomicMemory, semantic_tags: dict[str, Any]) -> AtomicMemory:
    """Make retrieval and graph construction consume the same attested fact span."""
    evidence_span = str(semantic_tags.get("evidence_span") or "").strip()
    if not evidence_span:
        return item
    slots = {
        key: value
        for key, value in dict(item.slots or {}).items()
        if str(value).strip() and str(value).lower() in evidence_span.lower()
    }
    for key, value in dict(semantic_tags.get("attributes") or {}).items():
        if value not in (None, "", []) and str(value).lower() in evidence_span.lower():
            slots.setdefault(str(key), value)
    surface_spans = {
        **dict((item.access_tags or {}).get("surface_spans") or {}),
        **dict(semantic_tags.get("surface_values") or {}),
    }
    return replace(
        item,
        content=evidence_span,
        slots=slots,
        access_tags={**dict(item.access_tags or {}), "surface_spans": surface_spans},
    )


def _semantic_annotation_is_informative(tags: dict[str, Any]) -> bool:
    discourse_act = str(tags.get("discourse_act") or "unknown").lower()
    if discourse_act in {"question", "request"}:
        return True
    if discourse_act in {"unknown", "other"}:
        return False
    attributes = dict(tags.get("attributes") or {})
    state_delta = dict(tags.get("state_delta") or {})
    changed_fields = state_delta.get("changed_fields")
    return bool(attributes) or bool(changed_fields)


def _normalize_grounded_claims(raw: Any) -> list[dict[str, str]]:
    """Normalize an open claim schema before exact-span validation."""
    if not isinstance(raw, list):
        return []
    normalized: list[dict[str, str]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        property_label = re.sub(
            r"[^a-z0-9]+", "_", str(item.get("property_label") or "").strip().lower()
        ).strip("_")
        value_span = str(item.get("value_span") or "").strip()
        claim_span = str(item.get("claim_span") or "").strip()
        subject_span = str(item.get("subject_span") or "").strip()
        if not property_label or len(property_label) > 80 or not value_span or not claim_span:
            continue
        claim = {
            "property_label": property_label,
            "value_span": value_span,
            "claim_span": claim_span,
            "subject_span": subject_span,
        }
        if claim not in normalized:
            normalized.append(claim)
    return normalized


def _normalize_attribute_map(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {}
    normalized: dict[str, Any] = {}
    for raw_key, raw_value in raw.items():
        key = re.sub(r"[^a-z0-9]+", "_", str(raw_key).strip().lower()).strip("_")
        if not key or len(key) > 80:
            continue
        if isinstance(raw_value, (str, int, float, bool)):
            value: Any = raw_value.strip() if isinstance(raw_value, str) else raw_value
        elif isinstance(raw_value, list):
            value = [
                item.strip() if isinstance(item, str) else item
                for item in raw_value
                if isinstance(item, (str, int, float, bool))
                and (not isinstance(item, str) or item.strip())
            ]
        else:
            continue
        if value not in ("", []):
            normalized[key] = value
    return normalized


def _lifecycle_from_semantic_tags(current: str, semantic_tags: dict[str, Any]) -> str:
    lifecycle = str(current or "active").lower()
    discourse_act = str(semantic_tags.get("discourse_act") or "unknown").lower()
    state_delta = semantic_tags.get("state_delta")
    operation = str(state_delta.get("operation") or "none").lower() if isinstance(state_delta, dict) else "none"
    is_active_operation = (
        discourse_act in {"assertion", "update", "confirmation"}
        and operation in {"create", "set", "add", "supersede"}
    )
    if lifecycle == "superseded" and is_active_operation:
        return "active"
    if lifecycle == "deleted" and is_active_operation and _has_grounded_active_change(semantic_tags):
        return "active"
    return lifecycle


def _has_grounded_active_change(semantic_tags: dict[str, Any]) -> bool:
    """A decomposed record can outlive a deleted sibling from its source turn."""
    state_delta = semantic_tags.get("state_delta")
    changed = state_delta.get("changed_fields") if isinstance(state_delta, dict) else None
    if not isinstance(changed, dict):
        return False
    return any(
        value not in (None, "", [])
        and not str(key).lower().endswith(("_deleted", "_deletion", "_removed", "_unavailable"))
        and str(value).strip().lower() not in {"deleted", "removed", "unavailable"}
        for key, value in changed.items()
    )


def _recover_atomic_content(item: dict[str, Any]) -> str:
    content = " ".join(str(item.get("content") or "").strip().split())
    if content:
        return content
    surface_spans = dict(item.get("surface_spans") or {})
    slots = dict(item.get("slots") or {})
    ordered_keys = [
        "instruction",
        "condition",
        "timing",
        "medication",
        "dosage",
        "substance",
        "reaction",
        "allergy_substance",
        "allergy_reaction",
        "procedure",
        "visit_type",
        "date",
        "time",
        "arrival_time",
        "location",
        "provider",
        "status",
        "consent_scope",
        "contact_method",
        "phone",
        "portal",
        "backup_contact",
    ]
    values: list[str] = []
    for key in ordered_keys:
        value = surface_spans.get(key) or slots.get(key)
        if value:
            values.append(str(value).strip())
    if not values:
        return ""
    return " ".join(value for value in values if value)


def _atomic_usefulness(item: AtomicMemory) -> int:
    score = 0
    if item.content.strip():
        score += 2
    if item.memory_type != "general_fact":
        score += 2
    if item.slots:
        score += min(len(item.slots), 5)
    if item.entities:
        score += min(len(item.entities), 3)
    if item.source_message_ids:
        score += 1
    score += int(round(max(item.confidence, 0.0) * 2))
    return score


def _count_useful_atomic_memories(items: list[AtomicMemory]) -> int:
    return sum(1 for item in items if _atomic_usefulness(item) >= 4)


def _llm_atomic_quality_is_poor(
    *,
    llm_items: list[AtomicMemory],
    raw_llm_count: int,
    heuristic_items: list[AtomicMemory],
) -> bool:
    if not llm_items:
        return True
    valid_ratio = len(llm_items) / max(raw_llm_count, 1)
    useful_llm = _count_useful_atomic_memories(llm_items)
    useful_heuristic = _count_useful_atomic_memories(heuristic_items)
    if valid_ratio < 0.5:
        return True
    if useful_llm == 0:
        return True
    if useful_heuristic >= 8 and useful_llm <= max(2, useful_heuristic // 5):
        return True
    return False


def _needs_heuristic_backfill(
    *,
    llm_items: list[AtomicMemory],
    heuristic_items: list[AtomicMemory],
) -> bool:
    useful_llm = _count_useful_atomic_memories(llm_items)
    useful_heuristic = _count_useful_atomic_memories(heuristic_items)
    return useful_llm < 8 and useful_heuristic > useful_llm


def _merge_atomic_memories(primary: list[AtomicMemory], secondary: list[AtomicMemory]) -> list[AtomicMemory]:
    merged: dict[tuple[str, str, tuple[str, ...]], AtomicMemory] = {}
    order: list[tuple[str, str, tuple[str, ...]]] = []
    for item in [*primary, *secondary]:
        content = " ".join(str(item.content or "").strip().split())
        if not content:
            continue
        key = (
            content.lower(),
            str(item.memory_type or "general_fact"),
            tuple(str(x) for x in item.source_message_ids),
        )
        existing = merged.get(key)
        if existing is None:
            merged[key] = item
            order.append(key)
            continue
        if _atomic_usefulness(item) > _atomic_usefulness(existing):
            merged[key] = item
    return [merged[key] for key in order]


def _extract_atomic_segments(text: str) -> list[str]:
    cleaned = " ".join(str(text or "").strip().split())
    if not cleaned:
        return []
    protected = cleaned.replace("Dr. ", "Dr ")
    sentences = [part.strip().replace("Dr ", "Dr. ") for part in re.split(r"(?<=[.!?;])\s+", protected) if part.strip()]
    segments: list[str] = []
    seen: set[str] = set()
    for sentence in sentences:
        if _looks_atomic(sentence):
            norm = sentence.lower()
            if norm not in seen:
                seen.add(norm)
                segments.append(sentence)
        if any(token in sentence.lower() for token in [" and ", ";", " followed by ", " right after", " plus "]):
            for part in re.split(r";|\s+\band\b\s+|\s+followed by\s+|\s+right after\.?\s*|\s+\bplus\b\s+", sentence, flags=re.IGNORECASE):
                atom = part.strip(" .")
                if len(atom) < 8:
                    continue
                atom = atom.replace("Dr ", "Dr. ")
                if _looks_atomic(atom):
                    norm = atom.lower()
                    if norm not in seen:
                        seen.add(norm)
                        segments.append(atom)
    return segments or [cleaned]


def _looks_atomic(text: str) -> bool:
    lowered = str(text or "").lower()
    return any(
        token in lowered
        for token in [
            "appointment",
            "ultrasound",
            "scan",
            "visit",
            "follow-up",
            "follow up",
            "arrive by",
            "arrival",
            "dr.",
            "suite",
            "allergy",
            "rash",
            "reaction",
            "portal",
            "callback",
            "backup",
            "canceled",
            "cancelled",
            "moved to",
            "rescheduled",
            "take ",
            "continue ",
            "restart ",
            "hold ",
            "nothing by mouth",
            "unless symptoms worsen",
            "beta-hcg",
            "logistics only",
            "through friday",
            "through monday",
        ]
    )


def _infer_entities_from_frame(frame) -> list[str]:
    values: list[str] = []
    for key in ["procedure", "provider", "location", "date", "time", "arrival_time", "allergy", "reaction", "medication", "instruction"]:
        value = frame.slots.get(key) or frame.surface_spans.get(key)
        if value:
            values.append(str(value))
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        norm = value.lower()
        if norm in seen:
            continue
        seen.add(norm)
        deduped.append(value)
    return deduped[:6]


def _infer_atomic_lifecycle(frame, text: str) -> str:
    lowered = str(text or "").lower()
    if any(token in lowered for token in ["canceled", "cancelled", "no longer active", "no longer scheduled"]):
        return "canceled"
    if any(token in lowered for token in ["rescheduled", "moved to", "updated to", "replaced"]):
        return "superseded"
    return frame.lifecycle_status
