"""Optional ingestion-time, source-grounded governed slot graph.

This module is deliberately independent from the Dense RAG index.  A single
LLM pass proposes open-vocabulary slot and policy candidates for the observable
episode prefix; deterministic validation then turns only grounded candidates
into a directed graph index.  The graph proposes source chunks for retrieval,
but it never makes an authorization decision.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from gov_mem.backbones.v4_support import RAGChunk, chunk_to_memory_item
from gov_mem.data.schema import RetrievedEvidence
from gov_mem.governance_runtime.leakage_guard import assert_runtime_payload_safe
from gov_mem.llm.client import LLMClient
from gov_mem.memory.dense_index import DenseMemoryIndex


GRAPH_SYSTEM_PROMPT = """You build an episode-local governed slot graph for memory retrieval.
Return JSON only. Use only the supplied observable message prefix. Do not use
hidden labels, gold answers, future turns, benchmark vocabulary, or a fixed
domain ontology. Do not answer any question and do not decide whether a
requester should be allowed access.

For each supplied source-message batch, extract zero or more open-vocabulary factual slots
that are explicitly stated in that message. A slot name must be induced from
the source text, not selected from a predefined list. Keep exact values and
verbatim source spans. Identify the slightly abstract thing/event that the
record is about (for example an appointment, access arrangement, or meeting
room); return it in target_entities. A record may concern more than one such
entity. Also extract explicit policy candidates only when the source states an
allow, deny, revoke, or permission requirement; these are source-level
candidates, not authorization decisions. Preserve explicit lifecycle
candidates such as update, supersede, cancel, or delete. Every candidate must
cite one supplied source_chunk_id and an exact contiguous span.
Do not infer a policy from a role, a value, or a sensitive-looking phrase.

Return exactly:
{"records":[{"source_chunk_id":"...","source_message_id":"...",
"source_span":"verbatim source substring","target_entities":[],
"slots":[{"slot_name":"open vocabulary name","value":"verbatim value",
"value_span":"verbatim substring"}],
"policies":[{"effect":"allow|deny|revoke|require_permission",
"subject":null,"resource":null,"span":"verbatim substring"}],
"lifecycle":[{"type":"assert|update|supersede|cancel|delete|none",
"span":"verbatim substring"}]}]}
"""


def _norm(value: Any) -> str:
    return " ".join(re.findall(r"\w+", str(value or "").casefold()))


def _contains(text: Any, fragment: Any) -> bool:
    needle = _norm(fragment)
    haystack = _norm(text)
    return bool(needle and f" {needle} " in f" {haystack} ")


def _tokens(value: Any) -> set[str]:
    return set(re.findall(r"\w+", str(value or "").casefold()))


def _is_future_conditional_policy(effect: str, span: str) -> bool:
    """Reject future/conditional revocations from the current permission ledger.

    Phrases such as ``revocation expected after the repro window`` describe a
    planned transition, not a revocation that has already happened. Treating
    them as current state creates an artificial allow/revoke conflict during
    symbolic replay. The source remains available as ordinary evidence.
    """
    if str(effect).casefold() != "revoke":
        return False
    text = f" {str(span or '').casefold()} "
    return any(
        marker in text
        for marker in (
            " expected after ",
            " after the ",
            " after ",
            " once ",
            " when ",
            " should be revoked ",
            " will be revoked ",
        )
    )


@dataclass
class GraphSlotCandidate:
    slot_id: str
    source_chunk_id: str
    source_message_id: str
    slot_name: str
    value: str
    source_span: str
    value_span: str
    target_entity: str | None = None
    confidence: float = 0.0


@dataclass
class GraphPolicyCandidate:
    policy_id: str
    source_chunk_id: str
    source_message_id: str
    effect: str
    subject: str | None
    resource: str | None
    span: str


@dataclass
class GraphLifecycleCandidate:
    source_chunk_id: str
    source_message_id: str
    lifecycle_type: str
    span: str


@dataclass
class GraphRelationRecord:
    """One append-only relation event in an entity's ledger.

    The graph edges are useful for navigation, but this record is the
    paper-facing state boundary: permission updates are retained in order and
    are never overwritten by a later value.
    """

    relation_id: str
    entity_id: str
    relation_type: str
    target: str | None
    effect: str | None
    lifecycle_type: str | None
    source_chunk_id: str
    source_message_id: str
    source_span: str
    sequence: int
    subject: str | None = None


@dataclass
class MemoryGovernedSlotGraph:
    """Directed, episode-local entity-relation graph and retrieval index.

    ``entity_lists`` is intentionally a plain mapping of entity id to an
    ordered list.  It is the simple representation needed to answer questions
    such as "how did permission for this thing change?" without pretending the
    graph itself is an authorization engine.
    """

    nodes: list[dict[str, Any]] = field(default_factory=list)
    edges: list[dict[str, Any]] = field(default_factory=list)
    slots: list[GraphSlotCandidate] = field(default_factory=list)
    policies: list[GraphPolicyCandidate] = field(default_factory=list)
    lifecycle: list[GraphLifecycleCandidate] = field(default_factory=list)
    entity_lists: dict[str, list[GraphRelationRecord]] = field(default_factory=dict)
    audit: dict[str, Any] = field(default_factory=dict)
    _search_index: DenseMemoryIndex | None = field(default=None, repr=False)
    _node_sources: dict[str, set[str]] = field(default_factory=dict, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": list(self.nodes),
            "edges": list(self.edges),
            "slots": [asdict(item) for item in self.slots],
            "policies": [asdict(item) for item in self.policies],
            "lifecycle": [asdict(item) for item in self.lifecycle],
            "entity_lists": {
                entity_id: [asdict(item) for item in records]
                for entity_id, records in self.entity_lists.items()
            },
            "audit": dict(self.audit),
        }

    def relation_list(self, entity_id: str) -> list[dict[str, Any]]:
        """Return an entity's append-only relation list as plain dictionaries."""
        return [asdict(item) for item in self.entity_lists.get(str(entity_id), [])]

    def permission_history(
        self, entity_id: str, *, resource: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return only permission events, preserving source order."""
        rows = [
            item for item in self.entity_lists.get(str(entity_id), [])
            if item.relation_type == "permission"
            and (resource is None or item.target == resource)
        ]
        return [asdict(item) for item in rows]

    def current_permission(
        self, entity_id: str, *, resource: str | None = None,
    ) -> str | None:
        """Project the latest permission event without deleting its history."""
        history = self.permission_history(entity_id, resource=resource)
        if not history:
            return None
        effect = str(history[-1].get("effect") or "")
        return "deny" if effect == "revoke" else effect or None

    def append_entity_change(
        self,
        *,
        entity_id: str,
        relation_type: str,
        target: str | None = None,
        effect: str | None = None,
        lifecycle_type: str | None = None,
        subject: str | None = None,
        source_chunk_id: str = "",
        source_message_id: str = "",
        source_span: str = "",
    ) -> GraphRelationRecord:
        """Append one source-grounded change to an abstract entity's list.

        This small API is useful for structured callers that already performed
        extraction. It deliberately does not compute authorization or replace
        an earlier record.
        """
        return self._append_relation(
            entity_id=entity_id,
            relation_type=relation_type,
            target=target,
            effect=effect,
            lifecycle_type=lifecycle_type,
            subject=subject,
            source_chunk_id=source_chunk_id,
            source_message_id=source_message_id,
            source_span=source_span,
        )

    def _append_relation(
        self,
        *,
        entity_id: str,
        relation_type: str,
        target: str | None,
        effect: str | None,
        lifecycle_type: str | None,
        source_chunk_id: str,
        source_message_id: str,
        source_span: str,
        subject: str | None = None,
    ) -> GraphRelationRecord:
        entity_id = str(entity_id).strip()
        records = self.entity_lists.setdefault(entity_id, [])
        if records:
            previous = records[-1]
            # Entity lists are change logs, not assertion logs. Repeatedly
            # restating the same state should not make the list grow; the
            # first grounded source remains the representative provenance.
            if (
                previous.relation_type == relation_type
                and previous.target == target
                and previous.effect == effect
                and previous.lifecycle_type == lifecycle_type
                and previous.subject == subject
            ):
                return previous
        relation = GraphRelationRecord(
            relation_id=f"relation::{entity_id}::{len(records)}",
            entity_id=entity_id,
            relation_type=relation_type,
            target=target,
            effect=effect,
            lifecycle_type=lifecycle_type,
            subject=subject,
            source_chunk_id=source_chunk_id,
            source_message_id=source_message_id,
            source_span=source_span,
            sequence=sum(len(values) for values in self.entity_lists.values()),
        )
        records.append(relation)
        return relation

    def retrieve(
        self,
        *,
        question: str,
        embedding_client: LLMClient,
        embedding_model: str,
        top_k: int = 20,
    ) -> list[dict[str, Any]]:
        if self._search_index is None or not self._node_sources or top_k <= 0:
            return []
        rows = self._search_index.query(
            query_texts=[question],
            top_k=max(top_k, 1),
            llm_client=embedding_client,
            embedding_model=embedding_model,
            allow_fallback=False,
        )
        by_source: dict[str, dict[str, Any]] = {}
        for node_id, score in rows:
            for source_id in self._node_sources.get(str(node_id), set()):
                item = by_source.setdefault(
                    source_id,
                    {"chunk_id": source_id, "score": float(score), "node_ids": []},
                )
                item["score"] = max(float(item["score"]), float(score))
                item["node_ids"].append(str(node_id))
        return sorted(
            by_source.values(),
            key=lambda item: (float(item["score"]), str(item["chunk_id"])),
            reverse=True,
        )[:top_k]

    def retrieve_entity_lists(
        self,
        *,
        question: str,
        embedding_client: LLMClient | None = None,
        embedding_model: str | None = None,
        top_k: int,
        max_relations_per_entity: int = 12,
    ) -> list[dict[str, Any]]:
        """Retrieve compact, query-scoped change lists with lexical matching.

        Entity lists are deliberately small structured ledgers. They do not
        need a second embedding request: matching the observed entity/resource
        words is deterministic, cheap, and keeps this channel independent from
        the dense RAG score. Permission and lifecycle history is always kept
        for a matched entity because symbolic replay needs every transition;
        ordinary fact rows are included only when their slot/value matches the
        question and are capped to keep the retrieved list lightweight. The
        embedding arguments remain accepted for API compatibility with the
        optional graph runner.
        """
        del embedding_client, embedding_model
        if not self.entity_lists or top_k <= 0:
            return []
        query_tokens = _tokens(question)
        matched: list[dict[str, Any]] = []
        for entity_id, records in self.entity_lists.items():
            entity_tokens = _tokens(entity_id)
            entity_overlap = query_tokens.intersection(entity_tokens)
            relation_matches: list[tuple[int, int, GraphRelationRecord]] = []
            relation_overlap: set[str] = set()
            for index, record in enumerate(records):
                searchable = _tokens(
                    " ".join(
                        str(value or "")
                        for value in (
                            record.relation_type,
                            record.target,
                            record.effect,
                            record.subject,
                        )
                    )
                )
                overlap = query_tokens.intersection(searchable)
                relation_kind = str(record.relation_type or "").casefold()
                # Governance history is the reason this list exists: retain
                # every permission/lifecycle transition for symbolic replay,
                # but only after the entity or resource itself matches the
                # query. This prevents unrelated policy ledgers from leaking
                # into a compact retrieval result.
                if relation_kind in {"permission", "lifecycle"}:
                    if entity_overlap or overlap:
                        relation_matches.append((len(overlap), index, record))
                        relation_overlap.update(overlap)
                elif overlap:
                    relation_matches.append((len(overlap), index, record))
                    relation_overlap.update(overlap)
            if not relation_matches:
                continue
            governance = [
                item for item in relation_matches
                if str(item[2].relation_type or "").casefold()
                in {"permission", "lifecycle"}
            ]
            ordinary = [
                item for item in relation_matches
                if str(item[2].relation_type or "").casefold()
                not in {"permission", "lifecycle"}
            ]
            ordinary.sort(key=lambda item: (item[0], -item[1]), reverse=True)
            if max_relations_per_entity > 0:
                ordinary = ordinary[:max_relations_per_entity]
            selected = sorted(
                [*governance, *ordinary], key=lambda item: item[1],
            )
            score = len(entity_overlap.union(relation_overlap)) / max(1, len(query_tokens))
            matched.append({
                "entity_id": entity_id,
                "score": float(score),
                "retrieval_backend": "entity_list_lexical",
                "relations": [asdict(item[2]) for item in selected],
            })
        return sorted(
            matched,
            key=lambda item: (float(item["score"]), str(item["entity_id"])),
            reverse=True,
        )[:top_k]


@dataclass
class GovernedSlotGraphBuild:
    graph: MemoryGovernedSlotGraph
    llm_calls: int = 0
    parse_failure: bool = False
    error: str | None = None


def _visible_chunks(chunks: list[RAGChunk]) -> list[dict[str, Any]]:
    return [
        {
            "source_chunk_id": chunk.chunk_id,
            "source_message_id": (chunk.source_message_ids or [None])[0],
            "text": chunk.text,
            "turn_index": (chunk.metadata.get("structured_record") or {}).get("turn_index"),
            "speaker": (chunk.metadata.get("structured_record") or {}).get("speaker"),
        }
        for chunk in chunks
    ]


def _add_node(graph: MemoryGovernedSlotGraph, node_id: str, node_type: str, label: str, **attrs: Any) -> None:
    if any(str(node.get("node_id")) == node_id for node in graph.nodes):
        return
    graph.nodes.append({"node_id": node_id, "node_type": node_type, "label": label, "attributes": attrs})


def _add_edge(graph: MemoryGovernedSlotGraph, edge_type: str, source_id: str, target_id: str, **attrs: Any) -> None:
    edge = {"edge_type": edge_type, "source_id": source_id, "target_id": target_id, "attributes": attrs}
    if not any(
        item["edge_type"] == edge_type
        and item["source_id"] == source_id
        and item["target_id"] == target_id
        for item in graph.edges
    ):
        graph.edges.append(edge)


def build_memory_governed_slot_graph(
    *,
    chunks: list[RAGChunk],
    llm_client: LLMClient,
    model_name: str,
    config: dict[str, Any],
    existing_graph: MemoryGovernedSlotGraph | None = None,
    processed_chunk_ids: set[str] | None = None,
) -> GovernedSlotGraphBuild:
    """Build or extend a graph from observable history chunks.

    The default remains a fresh graph for callers that pass one checkpoint.
    Incremental callers pass ``existing_graph`` and only the newly arrived
    chunks; extracted relations are then appended to the existing entity
    ledgers instead of re-reading the whole episode prefix.
    """

    graph = existing_graph or MemoryGovernedSlotGraph(
        audit={"enabled": True, "evidence_boundary": "observable_prefix"},
    )
    already_processed = {str(value) for value in (processed_chunk_ids or set())}
    if already_processed:
        chunks = [chunk for chunk in chunks if str(chunk.chunk_id) not in already_processed]
    visible_chunks = _visible_chunks(chunks)
    batch_size = max(
        1,
        int((config.get("memory_governed_slot_graph") or {}).get("batch_size", 8)),
    )
    raw_records: list[dict[str, Any]] = []
    errors: list[str] = []
    llm_calls = 0
    for start in range(0, len(visible_chunks), batch_size):
        payload = {"messages": visible_chunks[start:start + batch_size]}
        assert_runtime_payload_safe(payload, context="memory_governed_slot_graph")
        try:
            raw = llm_client.chat_json(
                model=model_name,
                system_prompt=GRAPH_SYSTEM_PROMPT,
                user_prompt=json.dumps(payload, ensure_ascii=False),
            )
            llm_calls += 1
        except Exception as exc:
            errors.append(type(exc).__name__)
            continue
        records = raw.get("records") if isinstance(raw, dict) else None
        if not isinstance(records, list):
            errors.append("records_not_list")
            continue
        raw_records.extend(item for item in records if isinstance(item, dict))

    if not raw_records and errors and llm_calls == 0:
        graph.audit.update({
            "parse_failure": True,
            "fallback": "empty_graph",
            "batch_count": (len(visible_chunks) + batch_size - 1) // batch_size,
        })
        return GovernedSlotGraphBuild(
            graph=graph, llm_calls=llm_calls, parse_failure=True, error=errors[0],
        )
    if errors:
        graph.audit.update({"batch_failures": errors, "partial_graph": True})

    rows_by_chunk = {chunk.chunk_id: chunk for chunk in chunks}
    rows_by_message = {
        str(message_id): chunk
        for chunk in chunks
        for message_id in chunk.source_message_ids
        if str(message_id)
    }
    records = raw_records
    entity_fallback_count = 0
    grounding_fallback_count = 0
    for record_index, item in enumerate(records):
        if not isinstance(item, dict):
            continue
        chunk_id = str(item.get("source_chunk_id") or "")
        chunk = rows_by_chunk.get(chunk_id) or rows_by_message.get(str(item.get("source_message_id") or ""))
        if chunk is None:
            continue
        source_message_id = str((chunk.source_message_ids or [""])[0])
        source_span = str(item.get("source_span") or "").strip()
        if not source_span or not _contains(chunk.text, source_span):
            # Preserve a source-grounded record even when the extractor
            # paraphrased its span. This is intentionally not treated as an
            # extracted slot or policy: it only keeps the entity ledger
            # addressable until a later repair pass can recover exact spans.
            memory_id = f"memory::{chunk.chunk_id}"
            _add_node(graph, memory_id, "MemoryNode", chunk.text, source_chunk_id=chunk.chunk_id)
            graph._node_sources.setdefault(memory_id, set()).add(chunk.chunk_id)
            fallback_entity_id = f"entity::record::{_norm(chunk.chunk_id)}"
            _add_node(
                graph,
                fallback_entity_id,
                "EntityNode",
                f"source record {source_message_id or chunk.chunk_id}",
                fallback=True,
            )
            graph._node_sources.setdefault(fallback_entity_id, set()).add(chunk.chunk_id)
            graph._append_relation(
                entity_id=fallback_entity_id,
                relation_type="source_record",
                target=chunk.chunk_id,
                effect=None,
                lifecycle_type=None,
                source_chunk_id=chunk.chunk_id,
                source_message_id=source_message_id,
                source_span=chunk.text,
            )
            grounding_fallback_count += 1
            continue
        memory_id = f"memory::{chunk.chunk_id}"
        _add_node(graph, memory_id, "MemoryNode", chunk.text, source_chunk_id=chunk.chunk_id)
        graph._node_sources.setdefault(memory_id, set()).add(chunk.chunk_id)
        raw_entities = item.get("target_entities")
        if not isinstance(raw_entities, list):
            # ``target_entity`` is retained as a compatibility alias for old
            # graph extractor responses.
            raw_entities = [item.get("target_entity")]
        target_entities = []
        for raw_entity in raw_entities:
            entity = str(raw_entity or "").strip()
            if entity and _contains(source_span, entity) and entity not in target_entities:
                target_entities.append(entity)
        entity_ids = []
        for target_entity in target_entities:
            entity_id = f"entity::{_norm(target_entity)}"
            entity_ids.append(entity_id)
            _add_node(graph, entity_id, "EntityNode", target_entity)
            graph._node_sources.setdefault(entity_id, set()).add(chunk.chunk_id)
        if not entity_ids:
            # Missing ``target_entities`` must not erase an otherwise grounded
            # record. Use a source-record key rather than inventing a semantic
            # entity name; symbolic policy replay still requires explicit
            # subject/resource fields and will not infer them from this key.
            fallback_entity_id = f"entity::record::{_norm(chunk.chunk_id)}"
            entity_ids.append(fallback_entity_id)
            _add_node(
                graph,
                fallback_entity_id,
                "EntityNode",
                f"source record {source_message_id or chunk.chunk_id}",
                fallback=True,
            )
            graph._node_sources.setdefault(fallback_entity_id, set()).add(chunk.chunk_id)
            entity_fallback_count += 1

        record_relation_added = False
        for slot_index, raw_slot in enumerate(item.get("slots") or []):
            if not isinstance(raw_slot, dict):
                continue
            name = str(raw_slot.get("slot_name") or "").strip()
            value = str(raw_slot.get("value") or "").strip()
            value_span = str(raw_slot.get("value_span") or value).strip()
            if not name or not value or not value_span or not _contains(source_span, value_span):
                continue
            slot_id = f"slot::{chunk.chunk_id}::{record_index}_{slot_index}"
            candidate = GraphSlotCandidate(
                slot_id=slot_id,
                source_chunk_id=chunk.chunk_id,
                source_message_id=source_message_id,
                slot_name=name,
                value=value,
                source_span=source_span,
                value_span=value_span,
                target_entity=(target_entities[0] if target_entities else None),
                confidence=float(raw_slot.get("confidence") or 0.0),
            )
            graph.slots.append(candidate)
            _add_node(graph, slot_id, "SlotNode", f"{name}={value}", slot_name=name, slot_value=value)
            _add_edge(graph, "contains_slot", memory_id, slot_id, slot_name=name)
            graph._node_sources.setdefault(slot_id, set()).add(chunk.chunk_id)
            for entity_id in entity_ids:
                _add_edge(graph, "describes", entity_id, slot_id)
                graph._append_relation(
                    entity_id=entity_id,
                    relation_type="has_slot",
                    target=f"{name}={value}",
                    effect=None,
                    lifecycle_type=None,
                    source_chunk_id=chunk.chunk_id,
                    source_message_id=source_message_id,
                    source_span=source_span,
                )
                record_relation_added = True

        for policy_index, raw_policy in enumerate(item.get("policies") or []):
            if not isinstance(raw_policy, dict):
                continue
            effect = str(raw_policy.get("effect") or "").casefold().strip()
            span = str(raw_policy.get("span") or "").strip()
            if effect not in {"allow", "deny", "revoke", "require_permission"} or not _contains(source_span, span):
                continue
            if _is_future_conditional_policy(effect, span):
                # A planned transition is not a current permission event.
                # Keep it out of the append-only authorization history so it
                # cannot manufacture a symbolic conflict with an active allow.
                continue
            subject = str(raw_policy.get("subject") or "").strip() or None
            resource = str(raw_policy.get("resource") or "").strip() or None
            # The model may cite a short policy span (for example, just
            # "logs-only incident access") while naming the principal in the
            # surrounding source sentence.  Keep the candidate grounded to
            # the same source chunk without requiring the short span to repeat
            # every argument.
            if subject and not _contains(chunk.text, subject):
                subject = None
            if resource and not _contains(chunk.text, resource):
                resource = None
            policy_id = f"policy::{chunk.chunk_id}::{record_index}_{policy_index}"
            policy = GraphPolicyCandidate(
                policy_id=policy_id,
                source_chunk_id=chunk.chunk_id,
                source_message_id=source_message_id,
                effect=effect,
                subject=subject,
                resource=resource,
                span=span,
            )
            graph.policies.append(policy)
            _add_node(graph, policy_id, "PolicyNode", span, effect=effect)
            _add_edge(graph, "policy_in", memory_id, policy_id, effect=effect)
            graph._node_sources.setdefault(policy_id, set()).add(chunk.chunk_id)
            for slot in graph.slots:
                if slot.source_chunk_id == chunk.chunk_id and (
                    not policy.resource or _contains(slot.source_span, policy.resource)
                ):
                    _add_edge(graph, effect, policy_id, slot.slot_id, resource=slot.slot_name)
            permission_entities = list(target_entities)
            if not permission_entities:
                permission_entities = [value for value in (resource, subject) if value]
            if not permission_entities and entity_ids:
                permission_entities = list(entity_ids)
            seen_permission_ids: set[str] = set()
            for permission_entity in permission_entities:
                permission_entity_id = f"entity::{_norm(permission_entity)}"
                if permission_entity_id in seen_permission_ids:
                    continue
                seen_permission_ids.add(permission_entity_id)
                _add_node(graph, permission_entity_id, "EntityNode", permission_entity)
                graph._node_sources.setdefault(permission_entity_id, set()).add(chunk.chunk_id)
                _add_edge(graph, "permission_relation", permission_entity_id, policy_id, effect=effect)
                graph._append_relation(
                    entity_id=permission_entity_id,
                    relation_type="permission",
                    target=resource,
                    effect=effect,
                    lifecycle_type=None,
                    subject=subject,
                    source_chunk_id=chunk.chunk_id,
                    source_message_id=source_message_id,
                    source_span=span,
                )
                record_relation_added = True

        for raw_lifecycle in item.get("lifecycle") or []:
            if not isinstance(raw_lifecycle, dict):
                continue
            lifecycle_type = str(raw_lifecycle.get("type") or "none").casefold()
            span = str(raw_lifecycle.get("span") or "").strip()
            if lifecycle_type not in {"assert", "update", "supersede", "cancel", "delete", "none"} or not _contains(source_span, span):
                continue
            graph.lifecycle.append(GraphLifecycleCandidate(chunk.chunk_id, source_message_id, lifecycle_type, span))
            if lifecycle_type != "none":
                for entity_id in entity_ids:
                    graph._append_relation(
                        entity_id=entity_id,
                        relation_type="lifecycle",
                        target=None,
                        effect=None,
                        lifecycle_type=lifecycle_type,
                        source_chunk_id=chunk.chunk_id,
                        source_message_id=source_message_id,
                        source_span=span,
                    )
                    record_relation_added = True

        if not record_relation_added:
            # Keep a lightweight append-only anchor for records that contain
            # no validated slot/policy/lifecycle candidate.
            graph._append_relation(
                entity_id=entity_ids[0],
                relation_type="source_record",
                target=chunk.chunk_id,
                effect=None,
                lifecycle_type=None,
                source_chunk_id=chunk.chunk_id,
                source_message_id=source_message_id,
                source_span=source_span,
            )

    graph.audit.update({
        "record_count": len(records),
        "llm_calls": llm_calls,
        "new_chunk_count": len(visible_chunks),
        "processed_chunk_count": len(already_processed) + len(visible_chunks),
        "incremental": existing_graph is not None,
        "batch_count": (len(visible_chunks) + batch_size - 1) // batch_size,
        "batch_failures": list(errors),
        "slot_count": len(graph.slots),
        "policy_count": len(graph.policies),
        "lifecycle_count": len(graph.lifecycle),
        "entity_count": len(graph.entity_lists),
        "relation_count": sum(len(values) for values in graph.entity_lists.values()),
        "node_count": len(graph.nodes),
        "edge_count": len(graph.edges),
        "entity_fallback_count": entity_fallback_count,
        "grounding_fallback_count": grounding_fallback_count,
    })

    # Entity Relation List retrieval is intentionally deterministic and
    # lightweight. The dense graph-node index is retained only as an explicit
    # compatibility option because it adds an embedding request at every
    # incremental write without contributing factual answer evidence.
    build_dense_index = bool(
        (config.get("memory_governed_slot_graph") or {}).get("dense_index", False)
    )
    if graph._node_sources and build_dense_index:
        items = []
        texts = []
        for node_id, source_ids in graph._node_sources.items():
            node = next((item for item in graph.nodes if item.get("node_id") == node_id), {})
            text = f"{node.get('label', '')} {node.get('attributes', {})}"
            items.append(chunk_to_memory_item(RAGChunk(
                chunk_id=node_id,
                instance_id="governed_slot_graph",
                text=text,
                source_message_ids=sorted(source_ids),
                speaker_ids=[],
                timestamp_range=(None, None),
                metadata={},
            )))
            texts.append(text)
        try:
            graph._search_index = DenseMemoryIndex.build(
                items=items,
                llm_client=llm_client,
                embedding_model=str((config.get("embedding") or {}).get("model") or "text-embedding-3-small"),
                embedding_texts=texts,
                allow_fallback=False,
            )
            graph.audit["index_backend"] = graph._search_index.backend
        except Exception as exc:
            # The optional channel must never take down the RAG path when the
            # separate graph index cannot be embedded.
            graph._search_index = None
            graph.audit.update({"index_failure": type(exc).__name__, "fallback": "empty_graph_retrieval"})
    else:
        graph._search_index = None
        graph.audit["index_backend"] = "disabled_entity_list_only"
    return GovernedSlotGraphBuild(
        graph=graph,
        llm_calls=llm_calls,
        parse_failure=bool(errors and not raw_records),
        error=errors[0] if errors and not raw_records else None,
    )


def graph_retrieval_evidence(
    *,
    retrieved: list[dict[str, Any]],
    chunk_by_id: dict[str, RAGChunk],
) -> list[RetrievedEvidence]:
    """Convert graph source proposals to ordinary evidence rows only."""
    out: list[RetrievedEvidence] = []
    for item in retrieved:
        chunk = chunk_by_id.get(str(item.get("chunk_id") or ""))
        if chunk is None:
            continue
        out.append(RetrievedEvidence(
            memory_id=chunk.chunk_id,
            content=chunk.text,
            score=float(item.get("score") or 0.0),
            retrieval_source="governed_slot_graph",
            reason="ingestion-time source-grounded graph match",
            user_id=(chunk.speaker_ids or [None])[0],
            memory_type="chunk",
            source_message_ids=list(chunk.source_message_ids),
            time=chunk.timestamp_range[0],
            metadata={
                **dict(chunk.metadata or {}),
                "graph_node_ids": list(item.get("node_ids") or []),
                "graph_candidate": True,
            },
        ))
    return out


__all__ = [
    "MemoryGovernedSlotGraph",
    "GraphRelationRecord",
    "GovernedSlotGraphBuild",
    "build_memory_governed_slot_graph",
    "graph_retrieval_evidence",
]
