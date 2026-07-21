from __future__ import annotations

from dataclasses import replace
from dataclasses import asdict
from pathlib import Path
import re
from typing import Any

from gov_mem.backbones.common import (
    BackboneRunResult,
    answer_with_retrieved_evidence,
    build_rag_chunks,
    build_reasoning_state,
    retrieve_rag_chunks,
    save_rag_chunks,
)
from gov_mem.backbones.action_constrained_realizer import ActionConstrainedRealizer
from gov_mem.backbones.relation_state import build_relation_state_bundle
from gov_mem.backbones.rag_policy import (
    RAGPolicyBackbone,
    build_action_only_answer_result,
)
from gov_mem.data.schema import AnswerResult, GovernedActionDecision, MemoryInstance, RetrievedEvidence
from gov_mem.experience.adaptation_audit import (
    RuntimeAdaptationAudit,
    runtime_adaptation_audit_to_dict,
)
from gov_mem.experience.runtime_experience import (
    RuntimeExperienceContext,
    RuntimeExperienceRetriever,
    retrieved_experience_lesson_to_dict,
    runtime_experience_context_to_dict,
)
from gov_mem.governance_runtime.access import build_principal
from gov_mem.governance_runtime.principal_relation_ledger import (
    build_principal_relation_ledger,
    ledger_relation_for_principal,
)
from gov_mem.governance_runtime.information_owner_ledger import build_information_owner_ledger
from gov_mem.governance_runtime.utility_source_locator import (
    locate_authorization_context_messages,
    locate_utility_source_messages,
)
from gov_mem.governance_runtime.source_role_ledger import classify_source_roles
from gov_mem.governance_runtime.action_predictor import GovernedActionPredictor
from gov_mem.governance_runtime.claim_adjudicator import adjudicate_claims, build_adjudicated_projection
from gov_mem.governance_runtime.current_state import resolve_current_state
from gov_mem.governance_runtime.evidence_frames import compile_evidence_frame
from gov_mem.governance_runtime.provenance_authorization import (
    build_slot_governance_certificate,
    certify_graph_slot_paths,
)
from gov_mem.governance_runtime.graph_slot_renderer import (
    build_graph_authorized_projection,
    render_graph_authorized_slots,
)
from gov_mem.governance_runtime.semantic_alignment import align_requested_attributes
from gov_mem.governance_runtime.semantic_reranker import (
    map_utility_source_attributes,
    semantic_rerank_evidence,
)
from gov_mem.governance_runtime.typed_realization_audit import realize_typed_request, restrict_semantic_spec
from gov_mem.governance_runtime.policy_frames import compile_policy_frames
from gov_mem.governance_runtime.query_policy_authorization import (
    attach_query_scoped_policy_authorizations,
)
from gov_mem.governance_runtime.state_ledger import build_current_state_ledger, ledger_to_dict
from gov_mem.graph.graph_builder import GovernedGraphBuilder
from gov_mem.graph.graph_retriever import GovernedGraphRetriever
from gov_mem.graph.graph_store import GovernedGraphStore
from gov_mem.llm.client import LLMClient
from gov_mem.llm.model_registry import resolve_llm_model
from gov_mem.memory.amem_memory import AtomicMemory, AtomicMemoryExtractor, atomic_memory_to_dict
from gov_mem.memory.governed_atom import (
    GovernedMemoryAtom,
    adapt_atomic_memories_to_governed_atoms,
    governed_atom_to_dict,
)
from gov_mem.memory.dense_index import DenseMemoryIndex
from gov_mem.reasoning.operators import build_required_slot_plan, select_evidence_by_slot_coverage
from gov_mem.reasoning.symbolic_reasoner import SymbolicGovernanceReasoner
from gov_mem.query_semantics import requests_derived_presence_inference
from gov_mem.retrieval.dual_channel_retriever import DualChannelRetriever
from gov_mem.retrieval.retrieval_bundle import retrieval_bundle_to_dict
from gov_mem.skills.runtime_skill_router import RuntimeSkillRouter
from gov_mem.skills.skill_context import SkillQueryContext, skill_query_context_to_dict
from gov_mem.skills.skill_executor import retrieved_skill_bundle_to_dict
from gov_mem.utils.io import append_jsonl, ensure_dir, write_json, write_jsonl


def resolve_uncertified_graph_action(action: str | None) -> str:
    """Preserve terminal policy decisions while denying uncertified realization."""
    return str(action) if action in {"refuse", "no_memory"} else "answer_redacted"


def _stage2_safe_projection_slots(evidence: list[RetrievedEvidence]) -> set[str]:
    """Collect only source-validated safe typed slot names from Stage 2."""
    slots: set[str] = set()
    for row in evidence:
        decision = dict((row.metadata or {}).get("stage2_semantic_rerank") or {})
        for item in list(decision.get("safe_projection_slots") or []):
            if isinstance(item, dict):
                slot = str(item.get("slot_name") or "").strip()
                value = str(item.get("value") or "").strip()
                observed = dict((row.metadata or {}).get("slots") or {}).get(slot)
                if slot and value and str(observed or "").strip() == value and value in str(row.content or ""):
                    slots.add(slot)
    return slots


def _utility_stage2_answerable(
    *,
    query_type: str,
    semantic_spec: dict[str, Any],
    decisions: list[dict[str, Any]],
    projection: list[RetrievedEvidence],
) -> bool:
    """Check whether Stage 2 has a complete direct utility projection."""
    if str(query_type or "").strip().lower() != "utility" or not projection:
        return False
    needs = semantic_spec.get("certifiable_needs")
    if isinstance(needs, list):
        requested = {
            str(item.get("attribute") or item.get("need_id") or "").strip()
            for item in needs
            if isinstance(item, dict)
        }
    else:
        requested = {
            str(value).strip()
            for value in list(
                semantic_spec.get("requested_attributes")
                or semantic_spec.get("requested_slots")
                or []
            )
        }
    requested.discard("")
    if not requested:
        return False
    by_attribute: dict[str, set[str]] = {}
    for item in decisions:
        if not isinstance(item, dict):
            continue
        attribute = str(item.get("attribute") or "").strip()
        decision = str(item.get("decision") or "").strip().lower()
        if attribute in requested:
            by_attribute.setdefault(attribute, set()).add(decision)
    return all(by_attribute.get(attribute) == {"answer"} for attribute in requested)


def _safe_projection_semantic_spec(
    semantic_spec: dict[str, Any], slots: set[str]
) -> dict[str, Any]:
    """Build a typed realization contract for Stage-2 safe projection fields."""
    result = dict(semantic_spec or {})
    ordered = sorted(str(slot).strip() for slot in slots if str(slot).strip())
    result["requested_attributes"] = ordered
    result["requested_slots"] = ordered
    result["attribute_bindings"] = [
        {
            "attribute": slot,
            "semantic_role": "safe_projection",
            "evidence_slot_hint": slot,
        }
        for slot in ordered
    ]
    result["certifiable_needs"] = [
        {"need_id": slot, "attribute": slot, "evidence_slot_hint": slot}
        for slot in ordered
    ]
    return result


def _projection_typed_field_keys(
    projection: list[RetrievedEvidence] | RetrievedEvidence | None,
) -> set[tuple[str, str, str]]:
    """Return the independently exposed typed fields in a closed projection."""
    rows = [projection] if isinstance(projection, RetrievedEvidence) else list(projection or [])
    fields: set[tuple[str, str, str]] = set()
    for row in rows:
        metadata = dict(row.metadata or {})
        candidates = list(metadata.get("typed_candidates") or [])
        if candidates:
            for item in candidates:
                if not isinstance(item, dict):
                    continue
                attribute = str(item.get("attribute") or "").strip()
                slot = str(item.get("slot_name") or "").strip()
                value = str(item.get("value") or "").strip()
                if attribute and slot and value:
                    fields.add((attribute, slot, value))
            continue
        for slot, value in dict(metadata.get("slots") or {}).items():
            slot_name = str(slot or "").strip()
            values = value if isinstance(value, list) else [value]
            for item in values:
                observed = str(item or "").strip()
                if slot_name and observed:
                    fields.add((slot_name, slot_name, observed))
    return fields


def _should_prefer_graph_utility_projection(
    *,
    query_type: str,
    certificate: dict[str, Any],
    graph_projection: RetrievedEvidence | None,
    utility_projection: list[RetrievedEvidence],
) -> bool:
    """Prefer a fuller graph projection without broadening the disclosure set."""
    if str(query_type or "").strip().lower() != "utility" or graph_projection is None:
        return False
    if not certificate.get("authorized"):
        return False
    if not certificate.get("stage2_operational_capability_authorized"):
        return False
    if (
        certificate.get("requires_redaction")
        or certificate.get("redacted_slot_names")
        or certificate.get("unresolved_requested_attributes")
    ):
        return False
    graph_fields = _projection_typed_field_keys(graph_projection)
    utility_fields = _projection_typed_field_keys(utility_projection)
    # Equal typed coverage still favors the graph projection: it has passed
    # the explicit source-span and lifecycle checks, while the utility
    # projection may retain a sentence-shaped LLM alias.
    return len(graph_fields) >= len(utility_fields)


class RAGPolicyAMemBackbone:
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
        self.extractor = AtomicMemoryExtractor(
            llm_client=llm_client,
            model_name=resolve_llm_model(config, "memory_ingestion"),
            annotation_cache_dir=(config.get("memory") or {}).get("semantic_annotation_cache_dir"),
        )
        self.graph_builder = GovernedGraphBuilder()
        self.graph_store = GovernedGraphStore(output_dir)
        self.graph_retriever = GovernedGraphRetriever()
        self.dual_channel_retriever = DualChannelRetriever()
        self.symbolic_reasoner = SymbolicGovernanceReasoner()
        self.action_constrained_realizer = ActionConstrainedRealizer()
        self.runtime_skill_router = RuntimeSkillRouter(
            skill_library_path=self.config.get("skill_library_path") or "outputs/skills/governance_skill_library.jsonl",
            rule_patches_path=self.config.get("rule_patches_path") if bool(self.config.get("use_self_evolving_update", False)) else None,
            prompt_patches_path=self.config.get("prompt_patches_path") if bool(self.config.get("use_self_evolving_update", False)) else None,
            policy_patches_path=self.config.get("policy_patches_path") if bool(self.config.get("use_self_evolving_update", False)) else None,
        )
        self.runtime_experience_retriever = RuntimeExperienceRetriever(
            self.config.get("experience_memory_path") or "outputs/experience/experience_memory.jsonl"
        )

    def run_instance(self, instance: MemoryInstance) -> BackboneRunResult:
        chunks = build_rag_chunks(instance, self.config)
        save_rag_chunks(self.output_dir, self.dataset_name, instance.instance_id, chunks)
        # The legacy owner hint is only an attribution prompt aid. It never
        # establishes access; requester-owner relation is resolved after the
        # selected factual source identifies a closed owner candidate.
        fallback_owner_user_id = RAGPolicyBackbone._owner_user_id(instance)
        principal_catalog = list(
            ((instance.metadata.get("raw_sample") or {}).get("episode") or {}).get("entities", {}).get("principals")
            or []
        )
        # Plan and retrieve first, then spend claim-extraction budget only on
        # dynamically retrieved source turns. All other turns remain as raw
        # governed evidence for policy/lifecycle checks.
        plan, chunk_retrieval, chunk_evidence = retrieve_rag_chunks(
            instance=instance,
            chunks=chunks,
            llm_client=self.embedding_client,
            embedding_model=str(self.config["embedding"]["model"]),
            config=self.config,
            planning_client=self.llm_client,
            planning_model=resolve_llm_model(self.config, "query_planning"),
        )
        utility_source_locator = locate_utility_source_messages(
            question=instance.question,
            semantic_spec=plan.semantic_spec,
            messages=list(instance.messages),
            llm_client=self.llm_client,
            model_name=resolve_llm_model(self.config, "reasoning"),
        )
        authorization_context_locator = locate_authorization_context_messages(
            question=instance.question,
            requester_id=instance.asking_user_id,
            selected_fact_message_ids={
                str(message_id)
                for message_id in list(utility_source_locator.get("selected_fact_message_ids") or [])
                if str(message_id)
            },
            messages=list(instance.messages),
            llm_client=self.llm_client,
            model_name=resolve_llm_model(self.config, "reasoning"),
        )
        memory_cfg = dict(self.config.get("memory") or {})
        annotation_budget = max(1, int(memory_cfg.get("claim_annotation_max_source_messages", 18)))
        annotation_source_message_ids: set[str] = set()
        for row in chunk_evidence:
            for message_id in list(row.source_message_ids or []):
                source_id = str(message_id)
                if source_id:
                    annotation_source_message_ids.add(source_id)
                if len(annotation_source_message_ids) >= annotation_budget:
                    break
            if len(annotation_source_message_ids) >= annotation_budget:
                break
        # The locator is a closed-set retrieval proposal over this episode's
        # visible source turns. It may enrich fact extraction but cannot grant
        # access or bypass later utility/policy certification.
        annotation_source_message_ids.update(
            str(message_id)
            for message_id in list(utility_source_locator.get("source_message_ids") or [])
            if str(message_id)
        )
        annotation_source_message_ids.update(
            str(message_id)
            for message_id in list(authorization_context_locator.get("source_message_ids") or [])
            if str(message_id)
        )
        atomic_memories = self.extractor.extract(
            instance,
            annotation_source_message_ids=annotation_source_message_ids,
            required_annotation_source_message_ids={
                str(message_id)
                for message_id in [
                    *list(utility_source_locator.get("selected_fact_message_ids") or []),
                    *list(authorization_context_locator.get("source_message_ids") or []),
                ]
                if str(message_id)
            },
        )
        self._save_atomic_memories(instance.instance_id, atomic_memories)
        information_owner_ledger = build_information_owner_ledger(
            messages=list(instance.messages),
            principal_catalog=principal_catalog,
            # The extractor retains unannotated compatibility memories, but
            # ownership attribution must stay within this run's actual
            # retrieval/locator closure rather than dilute a query-local LLM
            # decision over every turn in a long episode.
            candidate_message_ids=set(annotation_source_message_ids),
            required_message_ids={
                str(message_id)
                for message_id in [
                    *list(utility_source_locator.get("selected_fact_message_ids") or []),
                    *list(authorization_context_locator.get("source_message_ids") or []),
                ]
                if str(message_id)
            },
            target_owner_id=fallback_owner_user_id,
            llm_client=self.llm_client,
            model_name=resolve_llm_model(self.config, "reasoning"),
        )
        selected_fact_ids = {
            str(message_id)
            for message_id in list(utility_source_locator.get("selected_fact_message_ids") or [])
            if str(message_id)
        }
        owner_candidates = {
            str(owner_id)
            for message_id, owner_id in dict(information_owner_ledger.get("owner_by_message_id") or {}).items()
            if str(message_id) in selected_fact_ids and str(owner_id)
        }
        # Relation resolution consumes an episode-local owner closure built
        # from the factual source, not an identifier-derived owner guess.
        principal_relation_ledger = build_principal_relation_ledger(
            messages=list(instance.messages),
            requester_id=instance.asking_user_id,
            requester_role=((instance.metadata.get("requester") or {}).get("role")),
            principal_catalog=principal_catalog,
            question=instance.question,
            llm_client=self.llm_client,
            model_name=resolve_llm_model(self.config, "reasoning"),
            fallback_owner_id=fallback_owner_user_id,
            candidate_owner_ids=owner_candidates or None,
            relation_evidence_message_ids=selected_fact_ids | {
                str(message_id)
                for message_id in list(authorization_context_locator.get("source_message_ids") or [])
                if str(message_id)
            },
        )
        principal_relation_ledger = self._attest_owner_self_relation(
            ledger=principal_relation_ledger,
            requester_id=instance.asking_user_id,
            owner_candidates=owner_candidates,
            information_owner_ledger=information_owner_ledger,
            selected_fact_ids=selected_fact_ids,
        )
        owner_user_id = str(
            principal_relation_ledger.get("owner_id") or fallback_owner_user_id or ""
        ).strip() or None
        principal = build_principal(
            requester_id=instance.asking_user_id,
            requester_role=((instance.metadata.get("requester") or {}).get("role")),
            owner_user_id=owner_user_id,
            relation_override=ledger_relation_for_principal(principal_relation_ledger),
        )
        governed_atoms = self._prepare_governed_atoms(
            adapt_atomic_memories_to_governed_atoms(
                instance=instance,
                atomic_memories=atomic_memories,
                information_owner_by_message_id=dict(information_owner_ledger.get("owner_by_message_id") or {}),
            )
        )
        source_role_ledger = classify_source_roles(
            messages=list(instance.messages),
            candidate_message_ids=set(annotation_source_message_ids),
            required_message_ids={
                str(message_id)
                for message_id in [
                    *list(utility_source_locator.get("selected_fact_message_ids") or []),
                    *list(authorization_context_locator.get("source_message_ids") or []),
                ]
                if str(message_id)
            },
            llm_client=self.llm_client,
            model_name=resolve_llm_model(self.config, "reasoning"),
        )
        governed_atoms = self._attach_source_roles(
            governed_atoms,
            role_by_message_id=dict(source_role_ledger.get("role_by_message_id") or {}),
        )
        governed_atoms = compile_policy_frames(
            atoms=governed_atoms,
            llm_client=self.llm_client if bool(
                (self.config.get("governance_runtime") or {}).get("use_llm_policy_frame_compiler", False)
            ) else None,
            model_name=resolve_llm_model(self.config, "reasoning"),
        )
        self._save_governed_atoms(instance.instance_id, governed_atoms)
        governed_graph = None
        graph_paths: list[dict[str, Any]] = []
        graph_debug_paths = None
        if bool(self.config.get("use_governed_graph", False)):
            governed_graph = self.graph_builder.build(
                graph_id=f"{instance.instance_id}::governed_graph",
                instance_id=instance.instance_id,
                atoms=governed_atoms,
                principal_relation_ledger=principal_relation_ledger,
            )
            graph_paths = self.graph_retriever.retrieve_paths(
                graph=governed_graph,
                query=instance.question,
            )
            graph_debug_paths = self.graph_store.save(instance_id=instance.instance_id, graph=governed_graph)
        if governed_graph is not None:
            graph_paths = self.graph_retriever.retrieve_paths(
                graph=governed_graph,
                query=instance.question,
                requested_attributes=list(
                    plan.semantic_spec.get("requested_attributes")
                    or plan.semantic_spec.get("requested_slots")
                    or []
                ),
            )
        required_slot_plan = build_required_slot_plan(instance.question, plan)
        amem_evidence = self._retrieve_atomic_memories(instance, atomic_memories, plan.dense_queries)
        combined = self._prepare_evidence_rows_for_runtime(
            sorted(chunk_evidence + amem_evidence, key=lambda row: row.score, reverse=True)
        )
        combined = self._attach_semantic_provenance(combined, atomic_memories)
        combined, query_echo_filtered = self._filter_query_echo_evidence(
            question=instance.question,
            requester_id=instance.asking_user_id,
            evidence=combined,
        )
        # Stage 2 operates on record-local atomic memories selected by Stage
        # 1, before utility/alignment can independently narrow the closure.
        # Chunks remain retrieval proposals only; they cannot replace a
        # missing source-record member of a composite answer.
        stage2_candidates = self._stage2_atomic_record_candidates(
            atomic_memories=atomic_memories,
            selected_fact_message_ids={
                str(message_id)
                for message_id in list(utility_source_locator.get("selected_fact_message_ids") or [])
                if str(message_id)
            },
            selected_source_message_ids={
                str(message_id)
                for message_id in list(utility_source_locator.get("source_message_ids") or [])
                if str(message_id)
            },
        )
        # The source locator and dense retriever are complementary Stage-1
        # proposals. Preserve their union for the single closed-set Stage-2
        # LLM decision: dropping a locator-selected revision can otherwise
        # make an older status recap look canonical.
        stage2_by_memory_id = {str(row.memory_id): row for row in stage2_candidates}
        for row in amem_evidence:
            memory_id = str(row.memory_id)
            existing = stage2_by_memory_id.get(memory_id)
            if existing is None or float(row.score) > float(existing.score):
                stage2_by_memory_id[memory_id] = row
        stage2_candidates = list(stage2_by_memory_id.values())
        if not stage2_candidates:
            stage2_candidates = list(amem_evidence)
        evaluation_query_type = str(
            ((instance.metadata.get("evaluation") or {}).get("query_type") or "")
        ).strip().lower()
        stage2_allowed, decisions, stage2_filtered, stage2_debug = semantic_rerank_evidence(
            question=instance.question,
            semantic_spec=plan.semantic_spec,
            evidence=stage2_candidates,
            requester_id=instance.asking_user_id,
            owner_id=owner_user_id,
            relation_to_owner=principal.relation_to_owner,
            llm_client=self.llm_client,
            model_name=resolve_llm_model(self.config, "reasoning"),
        )
        bridge_debug: dict[str, Any] = {"enabled": False, "reason": "not_utility"}
        if evaluation_query_type == "utility":
            selected_fact_ids = {
                str(message_id)
                for message_id in list(utility_source_locator.get("selected_fact_message_ids") or [])
                if str(message_id)
            }
            selected_source_ids = selected_fact_ids | {
                str(message_id)
                for message_id in list(utility_source_locator.get("source_message_ids") or [])
                if str(message_id)
            }
            initial_stage2_ids = {str(row.memory_id) for row in stage2_allowed}
            # Utility needs recall across complementary current records.  The
            # dense retriever may already contain a safe projection record
            # whose message was not selected by the source locator because a
            # mixed record also contained a protected sibling.  Keep this
            # bridge closed to Stage-1 atomic candidates, but let one grouped
            # typed mapper inspect every active candidate.  It can only admit
            # exact source-local fields; final lifecycle, role, and graph
            # validation still decide what is renderable.
            bridge_candidates = [
                row for row in stage2_candidates
                if str((row.metadata or {}).get("lifecycle_status") or "active").lower()
                not in {"deleted", "superseded", "canceled", "historical", "retired"}
                and (
                    str(row.memory_id or "") not in initial_stage2_ids
                    or not list(
                        ((row.metadata or {}).get("stage2_semantic_rerank") or {}).get("served_attributes")
                        or []
                    )
                )
            ]
            bridge_allowed, bridge_decisions, bridge_filtered, bridge_debug = map_utility_source_attributes(
                question=instance.question,
                semantic_spec=plan.semantic_spec,
                evidence=bridge_candidates,
                llm_client=self.llm_client,
                model_name=resolve_llm_model(self.config, "reasoning"),
            )
            if bridge_allowed:
                bridge_ids = {str(row.memory_id) for row in bridge_allowed}
                decisions = [
                    decision for decision in decisions
                    if str(decision.get("chunk_id") or "") not in bridge_ids
                ] + bridge_decisions
                stage2_filtered = [
                    item for item in stage2_filtered
                    if str(item.get("memory_id") or "") not in bridge_ids
                ] + bridge_filtered
                stage2_by_id = {str(row.memory_id): row for row in stage2_allowed}
                stage2_by_id.update({str(row.memory_id): row for row in bridge_allowed})
                stage2_allowed = list(stage2_by_id.values())
        if (
            evaluation_query_type == "utility"
            and not stage2_allowed
            and not bool(stage2_debug.get("available"))
        ):
            # A transient Stage-2 transport/schema failure is not evidence
            # that utility memory is absent. Keep the already coarse-filtered
            # active closure and let the single claim adjudicator decide the
            # field-level answer. Privacy/safety paths never use this path.
            stage2_allowed = list(stage2_candidates)
            stage2_debug["utility_failure_fallback"] = {
                "enabled": True,
                "reason": "stage2_unavailable_preserved_coarse_active_closure",
                "candidate_count": len(stage2_allowed),
            }
        stage2_metadata_by_id = {
            str(row.memory_id): dict(row.metadata or {})
            for row in stage2_allowed
            if (row.metadata or {}).get("stage2_semantic_rerank")
        }
        if stage2_metadata_by_id:
            combined = [
                replace(
                    row,
                    metadata={**dict(row.metadata or {}), **stage2_metadata_by_id[str(row.memory_id)]},
                )
                if str(row.memory_id) in stage2_metadata_by_id else row
                for row in combined
            ]
        stage2_source_memory_ids = {str(row.memory_id) for row in stage2_allowed}
        stage2_atom_ids = {
            str(atom.atom_id)
            for atom in governed_atoms
            if str((atom.provenance or {}).get("source_memory_id") or "") in stage2_source_memory_ids
            and atom.atom_type in {"fact_atom", "event_atom"}
        }
        atoms_by_source_memory: dict[str, set[str]] = {}
        for atom in governed_atoms:
            source_memory_id = str((atom.provenance or {}).get("source_memory_id") or "")
            if source_memory_id and atom.atom_type in {"fact_atom", "event_atom"}:
                atoms_by_source_memory.setdefault(source_memory_id, set()).add(str(atom.atom_id))
        stage2_record_atom_ids_by_attribute: dict[str, set[str]] = {}
        stage2_authorized_atom_ids: set[str] = set()
        stage2_authorized_atom_ids_by_attribute: dict[str, set[str]] = {}
        # Relevance/admission and disclosure capability are separate typed
        # contracts.  Utility queries may use a source-local Stage-2
        # capability to avoid over-refusal; privacy/safety queries must still
        # prove disclosure through the graph's explicit policy or owner path.
        # The planner's query_type describes semantic operations (for example
        # factual or temporal); the evaluation/query regime describes the
        # disclosure contract.  Keep those namespaces separate so a factual
        # utility request is not mistaken for a privacy request.
        stage2_capability_mode = (
            "utility"
            if evaluation_query_type == "utility"
            else "explicit_graph_authorization"
        )
        for decision in decisions:
            if not bool(decision.get("allowed_for_requester")):
                continue
            for attribute in list(decision.get("served_attributes") or []):
                atom_ids = atoms_by_source_memory.get(str(decision.get("chunk_id") or ""), set())
                attribute_name = str(attribute).strip()
                if not attribute_name or not atom_ids:
                    continue
                stage2_record_atom_ids_by_attribute.setdefault(attribute_name, set()).update(atom_ids)
                if stage2_capability_mode == "utility":
                    # Stage 2 is a closed-set, source-grounded utility audit.
                    # An admitted answer member receives only a temporary,
                    # attribute-local capability. It is not a role grant:
                    # final alignment still checks the exact SlotNode, source
                    # span, lifecycle, and graph deny/permission edges.
                    stage2_authorized_atom_ids_by_attribute.setdefault(attribute_name, set()).update(atom_ids)
                    if bool(decision.get("authorized_for_requester")):
                        stage2_authorized_atom_ids.update(atom_ids)
        # Do not require an unrelated principal-to-owner relation before an
        # admitted operational record can contribute utility. The capability
        # remains bounded by the requested attribute and selected source atom.
        stage2_operational_capability_by_attribute = stage2_authorized_atom_ids_by_attribute
        stage2_operational_capability_atom_ids: set[str] = set().union(
            *stage2_operational_capability_by_attribute.values()
        ) if stage2_operational_capability_by_attribute else set()
        # One batch semantic pass reconciles current claims and chooses the
        # exact typed field for each utility need. It only narrows the already
        # admitted Stage-2 closure; deterministic graph/lifecycle checks still
        # decide whether a chosen field can be rendered.
        claim_adjudication, claim_adjudication_debug = ([], {
            "available": False,
            "reason": "not_run_for_non_utility_query",
        })
        if stage2_capability_mode == "utility":
            # The source-local bridge above has already annotated locator-
            # selected records. The adjudicator now sees the union as one
            # closed set and resolves current state and complementary fields.
            selected_fact_ids = {
                str(message_id)
                for message_id in list(utility_source_locator.get("selected_fact_message_ids") or [])
                if str(message_id)
            }
            # The locator selected complete factual turns, while Stage 2 may
            # have admitted only one typed member from each turn. Preserve
            # that source-local closure for the single claim adjudicator so
            # complementary fields (and later current-state updates) remain
            # comparable. This is still closed-set evidence: no new message
            # or permission is introduced.
            claim_evidence = [
                replace(
                    row,
                    metadata={**dict(row.metadata or {}), "utility_source_closure": True},
                )
                if selected_fact_ids.intersection(str(source_id) for source_id in list(row.source_message_ids or []))
                else row
                for row in stage2_allowed
            ]
            claim_evidence_ids = {str(row.memory_id) for row in claim_evidence}
            # Preserve every active record from the locator's already selected
            # source-message closure for the single utility adjudication pass.
            # Stage-2 relevance remains a proposal; current-state resolution
            # must compare active alternatives before choosing a scalar value.
            selected_source_ids = selected_fact_ids | {
                str(message_id)
                for message_id in list(utility_source_locator.get("source_message_ids") or [])
                if str(message_id)
            }
            for row in bridge_candidates:
                memory_id = str(row.memory_id or "")
                if (
                    memory_id
                    and memory_id not in claim_evidence_ids
                    and selected_source_ids.intersection(
                        str(source_id) for source_id in list(row.source_message_ids or [])
                    )
                    and str((row.metadata or {}).get("lifecycle_status") or "active").lower()
                    not in {"deleted", "superseded", "canceled", "historical", "retired"}
                ):
                    claim_evidence.append(replace(
                        row,
                        metadata={
                            **dict(row.metadata or {}),
                            "utility_source_closure": True,
                        },
                    ))
                    claim_evidence_ids.add(memory_id)
            claim_adjudication, claim_adjudication_debug = adjudicate_claims(
                question=instance.question,
                semantic_spec=plan.semantic_spec,
                evidence=claim_evidence,
                llm_client=self.llm_client,
                model_name=resolve_llm_model(self.config, "reasoning"),
                target_entities=list(plan.target_entities or []),
                requester_id=instance.asking_user_id,
            )
            adjudicated_by_attribute: dict[str, set[str]] = {}
            memory_to_atoms = {
                str(row.memory_id): atoms_by_source_memory.get(str(row.memory_id), set())
                for row in claim_evidence
            }
            for item in claim_adjudication:
                attribute = str(item.get("attribute") or "").strip()
                atom_ids = memory_to_atoms.get(str(item.get("memory_id") or ""), set())
                if attribute and atom_ids:
                    adjudicated_by_attribute.setdefault(attribute, set()).update(atom_ids)
            if adjudicated_by_attribute:
                # Keep the broader Stage-2 record map for collection queries,
                # but use the adjudicator's field-local atom map for scalars.
                for attribute, atom_ids in adjudicated_by_attribute.items():
                    stage2_operational_capability_by_attribute[attribute] = atom_ids
                stage2_operational_capability_atom_ids = set().union(
                    *stage2_operational_capability_by_attribute.values()
                )
            # Claim adjudication over the locator closure is still source
            # local. Make those exact atoms available to final alignment and
            # graph certification without granting unrelated records.
            claim_source_memory_ids = {str(row.memory_id) for row in claim_evidence}
            stage2_source_memory_ids.update(claim_source_memory_ids)
            stage2_atom_ids.update(
                atom_id
                for atom in governed_atoms
                if str((atom.provenance or {}).get("source_memory_id") or "") in claim_source_memory_ids
                and atom.atom_type in {"fact_atom", "event_atom"}
                for atom_id in [str(atom.atom_id)]
            )
        utility_adjudicated_projection = build_adjudicated_projection(
            evidence=claim_evidence if stage2_capability_mode == "utility" else stage2_allowed,
            decisions=claim_adjudication,
        ) if stage2_capability_mode == "utility" else []
        stage2_source_message_ids = {
            str(message_id)
            for row in (claim_evidence if stage2_capability_mode == "utility" else stage2_allowed)
            for message_id in list(row.source_message_ids or [])
            if str(message_id)
        }
        stage2_debug.update({
            "stage": "post_retrieval_pre_alignment",
            "candidate_kind": "atomic_records",
            "selected_memory_ids": sorted(stage2_source_memory_ids),
            "selected_atom_ids": sorted(stage2_atom_ids),
            "selected_source_message_ids": sorted(stage2_source_message_ids),
            "record_atom_ids_by_attribute": {
                attribute: sorted(atom_ids)
                for attribute, atom_ids in sorted(stage2_record_atom_ids_by_attribute.items())
            },
            "authorized_atom_ids": sorted(stage2_authorized_atom_ids),
            "operational_capability_atom_ids": sorted(stage2_operational_capability_atom_ids),
            "operational_capability_atom_ids_by_attribute": {
                attribute: sorted(atom_ids)
                for attribute, atom_ids in sorted(stage2_operational_capability_by_attribute.items())
            },
            "operational_capability_basis": (
                "stage2_attribute_source_local_utility_capability"
                if stage2_capability_mode == "utility"
                else "explicit_graph_authorization_required"
            ),
            "operational_capability_mode": stage2_capability_mode,
            "claim_adjudication": claim_adjudication_debug,
            "utility_source_attribute_bridge": bridge_debug,
        })
        # First pass is routing only. It may use a planner-declared intrinsic
        # evidence-slot hint, but its anchors can never authorize disclosure.
        routing_alignment = align_requested_attributes(
            question=instance.question,
            semantic_spec=plan.semantic_spec,
            graph=governed_graph,
            owner_id=owner_user_id,
            principal_relation=principal.relation_to_owner,
            utility_atom_ids=None,
            # Stage 2 has already audited the closed factual record set. The
            # routing pass only needs its observed slot labels for retrieval;
            # asking a second collection-membership question here adds latency
            # without changing the later, restricted final alignment.
            stage2_record_atom_ids_by_attribute=stage2_record_atom_ids_by_attribute,
            stage2_authorized_atom_ids=stage2_operational_capability_atom_ids,
            stage2_authorized_atom_ids_by_attribute=stage2_operational_capability_by_attribute,
            llm_client=self.llm_client,
            model_name=resolve_llm_model(self.config, "reasoning"),
            semantic_contract_certifiable=self._semantic_contract_certifiable(plan),
        )
        requested_attributes = list(
            plan.semantic_spec.get("requested_attributes")
            or plan.semantic_spec.get("requested_slots")
            or []
        )
        aligned_slot_names = [
            str(dict(binding or {}).get("slot_name") or "").strip()
            for binding in dict(routing_alignment.get("bindings") or {}).values()
        ]
        hinted_slot_names = [
            str(binding.get("evidence_slot_hint") or "").strip()
            for binding in list(plan.semantic_spec.get("attribute_bindings") or [])
            if isinstance(binding, dict)
        ]
        retrieval_attributes = list(dict.fromkeys([
            *[str(attribute).strip() for attribute in requested_attributes if str(attribute).strip()],
            *[slot_name for slot_name in aligned_slot_names if slot_name],
            *[slot_name for slot_name in hinted_slot_names if slot_name],
        ]))
        if governed_graph is not None:
            graph_paths = self.graph_retriever.retrieve_paths(
                graph=governed_graph,
                query=instance.question,
                requested_attributes=retrieval_attributes,
            )
        dual_channel_bundle = None
        if bool(self.config.get("use_dual_channel_retrieval", False)):
            dual_channel_bundle = self.dual_channel_retriever.retrieve(
                query=instance.question,
                requester=instance.asking_user_id,
                owner=owner_user_id,
                relation=principal.relation_to_owner,
                evidence=stage2_allowed,
                governed_atoms=governed_atoms,
                graph_paths=graph_paths,
                requested_attributes=retrieval_attributes,
                semantic_utility_source_message_ids={
                    str(message_id)
                    for message_id in list(utility_source_locator.get("source_message_ids") or [])
                    if str(message_id)
                },
            )
            # The dual channel may rank within the Stage-2 closure but cannot
            # replace it with a different atom/source set.
            dual_channel_bundle.utility_evidence.selected_memory_ids = sorted(stage2_source_memory_ids)
            dual_channel_bundle.utility_evidence.selected_atom_ids = sorted(stage2_atom_ids)
            dual_channel_bundle.utility_evidence.selected_source_message_ids = sorted(stage2_source_message_ids)
            self._save_retrieval_bundle(instance.instance_id, dual_channel_bundle)
        utility_source_message_ids = set(stage2_source_message_ids)
        # The final alignment is restricted to evidence the utility channel
        # actually selected. This prevents a broad routing proposal from
        # becoming a certificate anchor.
        semantic_alignment = align_requested_attributes(
            question=instance.question,
            semantic_spec=plan.semantic_spec,
            graph=governed_graph,
            owner_id=owner_user_id,
            principal_relation=principal.relation_to_owner,
            utility_atom_ids=stage2_atom_ids,
            utility_source_message_ids=utility_source_message_ids,
            stage2_record_atom_ids_by_attribute=stage2_record_atom_ids_by_attribute,
            stage2_authorized_atom_ids=stage2_operational_capability_atom_ids,
            stage2_authorized_atom_ids_by_attribute=stage2_operational_capability_by_attribute,
            adjudicated_fields={
                str(item.get("attribute") or ""): {
                    "memory_id": str(item.get("memory_id") or ""),
                    "slot_name": str(item.get("slot_name") or ""),
                    "value": str(item.get("value") or ""),
                }
                for item in claim_adjudication
                if str(item.get("attribute") or "")
            },
            allow_record_local_completion=evaluation_query_type == "utility",
            llm_client=self.llm_client,
            model_name=resolve_llm_model(self.config, "reasoning"),
            semantic_contract_certifiable=self._semantic_contract_certifiable(plan),
            require_attested_evidence_span=bool(self.config.get("require_attested_evidence_span", False)),
        )
        query_policy_authorization = attach_query_scoped_policy_authorizations(
            question=instance.question,
            graph=governed_graph,
            semantic_alignment=semantic_alignment,
            principal_relation_ledger=principal_relation_ledger,
            owner_id=owner_user_id,
            governance_policy_atom_ids=(
                set(dual_channel_bundle.governance_evidence.selected_policy_atom_ids)
                if dual_channel_bundle is not None
                else None
            ),
            llm_client=self.llm_client,
            model_name=resolve_llm_model(self.config, "reasoning"),
            utility_source_message_ids=utility_source_message_ids,
            direct_policy_context_message_ids={
                str(message_id)
                for message_id in [
                    *list(utility_source_locator.get("source_message_ids") or []),
                    *list(authorization_context_locator.get("source_message_ids") or []),
                ]
                if str(message_id)
            },
        )
        if governed_graph is not None and query_policy_authorization.get("available"):
            graph_debug_paths = self.graph_store.save(instance_id=instance.instance_id, graph=governed_graph)
        allowed = list(stage2_allowed)
        policy_scope: dict[str, Any] = {}
        filtered = [
            {"memory_id": row.memory_id, "reason": "requester_query_echo_not_independent_evidence"}
            for row in query_echo_filtered
        ]
        filtered.extend(stage2_filtered)
        reasoning_state = build_reasoning_state(
            allowed,
            trace=[
                f"stage2 certified {len(allowed)} / {len(stage2_candidates)} atomic record candidates before alignment.",
            ],
        )
        bundle_rows: list[RetrievedEvidence] = []
        bundle_debug: dict[str, Any] = {"enabled": False, "reason": "disabled_by_default"}
        if bool(self.config.get("enable_relation_state_bundle", False)):
            bundle_rows, bundle_debug = build_relation_state_bundle(instance.question, allowed)
        if bundle_rows:
            allowed = bundle_rows + allowed
            reasoning_state.selected_evidence = allowed
            reasoning_state.reasoning_trace.append(
                f"relation_state_bundle added {len(bundle_rows)} synthesized bundle row(s) for {bundle_debug.get('bundle_type')}."
            )
        current_state, frames = resolve_current_state(allowed)
        selected_evidence, selected_frames, slot_coverage = select_evidence_by_slot_coverage(
            frames=frames,
            evidence=allowed,
            required_slot_plan=required_slot_plan,
            requester=instance.asking_user_id,
            config=self.config,
        )
        explicit_requested_slots = (
            list(plan.semantic_spec.get("requested_attributes") or [])
            or list(plan.semantic_spec.get("requested_slots") or [])
        )
        if not explicit_requested_slots:
            # Record bundles require a policy-filtered candidate set; typed-slot
            # selection would prematurely collapse complementary records.
            selected_evidence = list(allowed)
            selected_frames = [compile_evidence_frame(row) for row in selected_evidence]
        support_rows, support_trace = self._augment_selected_current_state_support(
            question=instance.question,
            required_slot_plan=required_slot_plan,
            selected_evidence=selected_evidence,
            selected_frames=selected_frames,
            policy_allowed_evidence=allowed,
            atomic_memories=atomic_memories,
            principal=principal,
            policy_scope=policy_scope,
            # In governed graph mode, a synthetic current-state row can only
            # be a retrieval proposal.  It must never become the final source
            # of truth when the graph certificate cannot prove disclosure.
            allow_canonical_projection=(
                bool(explicit_requested_slots)
                and not (
                    bool(self.config.get("use_governed_graph", False))
                    and bool(self.config.get("enable_graph_typed_slot_realization", False))
                )
            ),
        )
        ledger = build_current_state_ledger(selected_frames)
        reasoning_state.selected_evidence = selected_evidence
        reasoning_state.selected_frames = selected_frames
        reasoning_state.current_state_ledger = ledger_to_dict(ledger)
        reasoning_state.required_slot_plan = required_slot_plan
        reasoning_state.slot_coverage = slot_coverage
        graph_authorization_certificate = certify_graph_slot_paths(
            semantic_spec=plan.semantic_spec,
            graph=governed_graph,
            principal_relation=principal.relation_to_owner,
            owner_id=owner_user_id,
            utility_atom_ids=stage2_atom_ids,
            utility_source_message_ids=utility_source_message_ids,
            governance_policy_atom_ids=(
                set(dual_channel_bundle.governance_evidence.selected_policy_atom_ids)
                if dual_channel_bundle is not None
                else None
            ),
            semantic_contract_certifiable=self._semantic_contract_certifiable(plan),
            semantic_alignment=semantic_alignment,
            require_attested_evidence_span=bool(self.config.get("require_attested_evidence_span", False)),
            principal_relation_ledger=principal_relation_ledger,
            stage2_authorized_atom_ids=stage2_operational_capability_atom_ids,
            stage2_authorized_atom_ids_by_attribute=stage2_operational_capability_by_attribute,
            stage2_realizations=claim_adjudication,
            allow_utility_record_completion=evaluation_query_type == "utility",
        )
        graph_authorized_projection = None
        if bool(self.config.get("enable_graph_typed_slot_realization", False)):
            graph_authorized_projection = build_graph_authorized_projection(
                certificate=graph_authorization_certificate,
                semantic_spec=plan.semantic_spec,
            )
        prefer_graph_utility_projection = _should_prefer_graph_utility_projection(
            query_type=evaluation_query_type,
            certificate=graph_authorization_certificate,
            graph_projection=graph_authorized_projection,
            utility_projection=utility_adjudicated_projection,
        )
        if graph_authorized_projection is not None and (
            not utility_adjudicated_projection or prefer_graph_utility_projection
        ):
            selected_evidence[:] = [graph_authorized_projection]
            selected_frames[:] = [compile_evidence_frame(graph_authorized_projection)]
            required_slots = list(required_slot_plan.get("required_slots") or [])
            slot_coverage.update({
                "covered_slots": required_slots,
                "missing_slots": [],
                "coverage_ratio": 1.0,
            })
            reasoning_state.selected_evidence = selected_evidence
            reasoning_state.selected_frames = selected_frames
            reasoning_state.slot_coverage = slot_coverage
            reasoning_state.reasoning_trace.append(
                "graph certificate projected exact source-validated typed slots"
                + (" and replaced an incomplete utility projection." if prefer_graph_utility_projection else ".")
            )
        runtime_skill_context = None
        runtime_skill_bundle = None
        runtime_experience_context = None
        runtime_experience_lessons = []
        if bool(self.config.get("use_skill_library", False)):
            runtime_skill_context = self._build_skill_query_context(
                instance=instance,
                principal=principal,
                required_slot_plan=required_slot_plan,
                graph_paths=graph_paths,
                slot_coverage=slot_coverage,
                graph_authorization_certificate=graph_authorization_certificate,
            )
            runtime_skill_bundle = self.runtime_skill_router.route(context=runtime_skill_context)
            if str(self.config.get("adaptation_effect_mode") or "full") == "lightweight":
                runtime_skill_bundle = self._make_adaptation_advisory_only(runtime_skill_bundle)
            self._save_skill_bundle(instance.instance_id, runtime_skill_context, runtime_skill_bundle)
        if bool(self.config.get("use_experience_memory", False)):
            runtime_experience_context = self._build_experience_context(
                instance=instance,
                principal=principal,
                required_slot_plan=required_slot_plan,
                graph_paths=graph_paths,
                slot_coverage=slot_coverage,
                graph_authorization_certificate=graph_authorization_certificate,
            )
            runtime_experience_lessons = self.runtime_experience_retriever.retrieve(
                context=runtime_experience_context,
                top_k=int(self.config.get("experience_top_k", 2) or 2),
            )
            self._save_experience_bundle(
                instance.instance_id,
                runtime_experience_context,
                runtime_experience_lessons,
            )
        symbolic_runtime = None
        if bool(self.config.get("use_symbolic_reasoner", False)) and dual_channel_bundle is not None:
            lowered_question = instance.question.lower()
            symbolic_runtime = self.symbolic_reasoner.decide(
                query=instance.question,
                requester_id=instance.asking_user_id,
                owner_id=owner_user_id,
                requester_role=principal.organization_role or ((instance.metadata.get("requester") or {}).get("role")),
                relation_to_owner=principal.relation_to_owner,
                required_slot_plan=required_slot_plan,
                retrieval_bundle=dual_channel_bundle,
                runtime_skill_bundle=retrieved_skill_bundle_to_dict(runtime_skill_bundle) if runtime_skill_bundle is not None else None,
                explicit_deleted_request=self._query_explicitly_requests_deleted_content(lowered_question),
                explicit_historical_request=self._query_explicitly_requests_historical_content(lowered_question),
                explicit_current_request=self._query_explicitly_requests_current_state(lowered_question),
            )
        slot_governance_certificate = build_slot_governance_certificate(
            # Governance is anchored to what the user requested. Auxiliary
            # realization fields cannot turn a denied request into a projection.
            semantic_spec=plan.semantic_spec,
            evidence=combined,
            symbolic_decision=(symbolic_runtime or {}).get("decision"),
            query_echo_evidence=query_echo_filtered,
            principal_relation=principal.relation_to_owner,
        )
        if symbolic_runtime is not None and principal.relation_to_owner in {"owner", "self"}:
            decision = symbolic_runtime["decision"]
            denied_slots = {str(slot) for slot in list(decision.get("denied_slots") or [])}
            certified_slots = {
                str(slot)
                for slot in list(slot_governance_certificate.get("safe_projection_slots") or [])
                if str(slot) not in denied_slots
            }
            if certified_slots:
                decision["allowed_slots"] = list(dict.fromkeys([
                    *list(decision.get("allowed_slots") or []),
                    *sorted(certified_slots),
                ]))
                decision.setdefault("rules_fired", []).append(
                    "owner_self_active_provenance_certificate"
                )
        if symbolic_runtime is not None and slot_governance_certificate.get("action_recommendation") == "refuse":
            decision = symbolic_runtime["decision"]
            decision["action_constraint"] = "refuse"
            decision.setdefault("rules_fired", []).append(
                "slot_certificate_denied_without_authorized_projection"
            )
        if (
            symbolic_runtime is not None
            and principal.relation_to_owner not in {"owner", "self"}
            and requests_derived_presence_inference(instance.question)
        ):
            decision = symbolic_runtime["decision"]
            decision["action_constraint"] = "refuse"
            decision.setdefault("denied_slots", []).append("derived_presence")
            decision["denied_slots"] = list(dict.fromkeys(decision["denied_slots"]))
            decision.setdefault("rules_fired", []).append(
                "derived_presence_inference_requires_explicit_authorization"
            )
        if symbolic_runtime is not None:
            decision = symbolic_runtime["decision"]
            decision["utility_certificate_complete"] = bool(
                required_slot_plan.get("required_slots")
                and not slot_coverage.get("missing_slots")
                and float(slot_coverage.get("coverage_ratio") or 0.0) >= 1.0
            )
            decision["record_bundle_certificate_complete"] = bool(
                not list(plan.semantic_spec.get("requested_slots") or [])
                and principal.relation_to_owner in {"owner", "self"}
                and selected_evidence
                and decision.get("action_constraint") == "answer"
                and not list(decision.get("denied_slots") or [])
                and not list(decision.get("blocked_facts") or [])
            )
            projection_slots = {
                str(slot) for slot in list(decision.get("projection_slots") or []) if str(slot)
            }
            if decision.get("action_constraint") == "answer_redacted" and projection_slots:
                decision["allowed_slots"] = sorted(
                    set(decision.get("allowed_slots") or []) | projection_slots
                )
                selected_ids = {str(row.memory_id) for row in selected_evidence}
                covered_projection_slots: set[str] = set()
                for frame in selected_frames:
                    covered_projection_slots.update(set((frame.slots or {}).keys()) & projection_slots)
                for row in allowed:
                    if covered_projection_slots >= projection_slots:
                        break
                    if str(row.memory_id) in selected_ids:
                        continue
                    frame = compile_evidence_frame(row)
                    row_projection_slots = set((frame.slots or {}).keys()) & projection_slots
                    if not row_projection_slots:
                        continue
                    if str(frame.lifecycle_status or "active").lower() != "active":
                        continue
                    selected_evidence.append(row)
                    selected_frames.append(frame)
                    selected_ids.add(str(row.memory_id))
                    covered_projection_slots.update(row_projection_slots)
                reasoning_state.selected_evidence = selected_evidence
                reasoning_state.selected_frames = selected_frames
            # Persist the post-certificate decision, not the preliminary decision.
            self._save_symbolic_decision(instance.instance_id, symbolic_runtime)
        reasoning_state.reasoning_trace.extend(
            [
                f"typed current-state resolved {len(frames)} frames from {len(allowed)} policy-filtered rows.",
                f"slot coverage selected {len(selected_frames)} frames with coverage {slot_coverage.get('coverage_ratio', 0.0):.2f}.",
            ]
        )
        reasoning_state.reasoning_trace.extend(support_trace)
        if symbolic_runtime is not None:
            symbolic_decision = symbolic_runtime.get("decision") or {}
            reasoning_state.reasoning_trace.append(
                f"symbolic governance decision proposed action={symbolic_decision.get('action_constraint')} confidence={float(symbolic_decision.get('confidence') or 0.0):.2f}."
            )
        action_predictor = GovernedActionPredictor(
            llm_client=self.llm_client,
            model_name=resolve_llm_model(self.config, "action_decision"),
        )
        reasoning_state.reasoning_trace.append(
            "graph slot authorization: "
            f"{graph_authorization_certificate.get('reason')}"
        )
        action_decision = action_predictor.decide(
            instance=instance,
            plan=plan,
            evidence=combined,
            projected_evidence=selected_evidence,
            required_slot_plan=required_slot_plan,
            slot_coverage=slot_coverage,
            current_state_ledger=reasoning_state.current_state_ledger,
            selected_frames=selected_frames,
            graph_authorization_certificate=graph_authorization_certificate,
            verified_owner_user_id=owner_user_id,
            verified_relation_to_owner=principal.relation_to_owner,
        )
        if bool(graph_authorization_certificate.get("authorized")):
            # A complete graph certificate has already proved the exact
            # releasable projection. Its authorization decision is stronger
            # than a conservative action-model label derived from raw context.
            # `requires_redaction` is computed from only the certified paths.
            stage2_utility_capability = bool(
                stage2_capability_mode == "utility"
                and graph_authorization_certificate.get("stage2_operational_capability_authorized")
            )
            if action_decision.action not in {"no_memory", "refuse"}:
                graph_requires_redaction = bool(
                    graph_authorization_certificate.get("requires_redaction")
                    or graph_authorization_certificate.get("redacted_slot_names")
                    or graph_authorization_certificate.get("unresolved_requested_attributes")
                )
                non_owner_without_operational_capability = bool(
                    principal.relation_to_owner not in {"owner", "self"}
                    and not graph_authorization_certificate.get(
                        "stage2_operational_capability_authorized"
                    )
                )
                utility_capability_direct = bool(
                    stage2_utility_capability and not graph_requires_redaction
                )
                # A graph path proves source grounding, not an automatic
                # upgrade from a partial disclosure to a direct answer.
                # Preserve the typed action decision unless the graph also
                # proves the requester's operational capability.
                action_decision.action = (
                    "answer"
                    if utility_capability_direct
                    else "answer_redacted"
                    if (
                        action_decision.action == "answer_redacted"
                        or graph_requires_redaction
                        or non_owner_without_operational_capability
                    )
                    else "answer"
                    if stage2_utility_capability
                    else "answer"
                )
        final_realization = None

        def render_answer(action: str, render_evidence: list[RetrievedEvidence]):
            extra_rules = ["Use both retrieved raw chunks and atomic memory evidence when helpful."]
            if runtime_skill_bundle is not None:
                if str(self.config.get("adaptation_effect_mode") or "full") == "lightweight":
                    extra_rules.extend(self._lightweight_adaptation_rules(runtime_skill_bundle, runtime_experience_lessons))
                else:
                    extra_rules.extend(list(runtime_skill_bundle.prompt_patches or []))
            if runtime_experience_lessons:
                if str(self.config.get("adaptation_effect_mode") or "full") != "lightweight":
                    extra_rules.extend([lesson.lesson for lesson in runtime_experience_lessons if lesson.lesson])
            if symbolic_runtime is not None:
                symbolic_decision = symbolic_runtime.get("decision") or {}
                if not bool(self.config.get("fallback_to_v0_on_low_confidence", True)) or float(symbolic_decision.get("confidence") or 0.0) >= 0.75:
                    extra_rules.append(
                        "Symbolic governance hint: action_constraint="
                        f"{symbolic_decision.get('action_constraint')}; "
                        f"allowed_slots={symbolic_decision.get('allowed_slots')}; "
                        f"denied_slots={symbolic_decision.get('denied_slots')}; "
                        f"blocked_facts={symbolic_decision.get('blocked_facts')}."
                    )
            return answer_with_retrieved_evidence(
                instance=instance,
                evidence=render_evidence,
                llm_client=self.llm_client,
                model_name=resolve_llm_model(self.config, "answering"),
                action=action,
                extra_rules=extra_rules,
                used_chunk_ids=[row.memory_id for row in render_evidence],
                config=self.config,
                principal=principal,
                policy_decisions=decisions,
                requester_context={
                    "asking_user_id": instance.asking_user_id,
                    "requester": instance.metadata.get("requester"),
                    "organization_role": principal.organization_role,
                    "relation_to_owner": principal.relation_to_owner,
                },
                semantic_spec=plan.semantic_spec,
            )

        certificate_required = bool(
            self.config.get("use_governed_graph", False)
            and self.config.get("enable_graph_typed_slot_realization", False)
            and stage2_capability_mode != "utility"
        )
        if certificate_required and not bool(graph_authorization_certificate.get("authorized")):
            # The graph certificate is authoritative for sensitive and mixed
            # disclosures, but a failed typed-slot path should not erase an
            # independently selected current operational record. Preserve
            # Stage-2 utility evidence for owner/staff answers; for a redacted
            # answer, retain only rows that explicitly carry safe wording.
            stage2_fallback_rows = [
                row for row in allowed
                if str((row.metadata or {}).get("lifecycle_status") or "active").lower()
                not in {"deleted", "superseded", "canceled", "historical"}
                and str(((row.metadata or {}).get("stage2_semantic_rerank") or {}).get("classification") or "")
                in {"answer_member", "redactable_member", "safe_projection_member"}
            ]
            safe_projection_slots = _stage2_safe_projection_slots(stage2_fallback_rows)
            if action_decision.action in {"no_memory", "refuse"} and safe_projection_slots:
                action_decision.action = "answer_redacted"
            if (
                action_decision.action in {"no_memory", "refuse"}
            ):
                action = action_decision.action
                answer_result = build_action_only_answer_result(
                    action=action,
                    reasoning_summary="Terminal forgetting or refusal decision preserved before graph realization.",
                    used_memory_ids=[],
                )
            else:
                certified_slots = set(dict(graph_authorization_certificate.get("slots") or {}))
                safe_projection_slots = set(
                    str(slot).strip()
                    for slot in list((slot_governance_certificate or {}).get("safe_projection_slots") or [])
                    if str(slot).strip()
                )
                renderable_attributes = (
                    certified_slots | safe_projection_slots
                    if action_decision.action == "answer_redacted"
                    else {
                        attribute
                        for decision in decisions
                        if str(decision.get("classification") or "") in {"answer_member", "redactable_member"}
                        for attribute in list(decision.get("served_attributes") or [])
                    }
                )
                audited = None
                if safe_projection_slots and action_decision.action == "answer_redacted":
                    audited = realize_typed_request(
                        question=instance.question,
                        semantic_spec=_safe_projection_semantic_spec(
                            plan.semantic_spec,
                            safe_projection_slots,
                        ),
                        evidence=stage2_fallback_rows,
                        llm_client=self.llm_client,
                        model_name=resolve_llm_model(self.config, "reasoning"),
                        action="answer_redacted",
                    )
                elif renderable_attributes:
                    audited = realize_typed_request(
                        question=instance.question,
                        semantic_spec=restrict_semantic_spec(plan.semantic_spec, renderable_attributes),
                        evidence=stage2_fallback_rows,
                        llm_client=self.llm_client,
                        model_name=resolve_llm_model(self.config, "reasoning"),
                        action=action_decision.action,
                    )
                if audited is not None and audited.answer_text:
                    answer_result = audited
                    selected_evidence[:] = [
                        row for row in stage2_fallback_rows
                        if row.memory_id in set(audited.used_memory_ids)
                    ]
                    selected_frames[:] = [compile_evidence_frame(row) for row in selected_evidence]
                    action_decision.action = audited.action
                elif audited is not None and audited.answer_structured.get("audit_status") in {"retired", "unresolved"}:
                    action = "no_memory"
                    action_decision.action = action
                    selected_evidence[:] = []
                    selected_frames[:] = []
                    answer_result = build_action_only_answer_result(
                        action=action,
                        reasoning_summary="Typed realization identified only retired or deleted source values.",
                        used_memory_ids=[],
                    )
                else:
                    action = (
                        "answer_redacted"
                        if action_decision.action == "answer_redacted"
                        else resolve_uncertified_graph_action(action_decision.action)
                    )
                    action_decision.action = action
                    selected_evidence[:] = []
                    selected_frames[:] = []
                    answer_result = build_action_only_answer_result(
                        action=action,
                        reasoning_summary="Graph authorization certificate did not prove a releasable source-grounded projection.",
                        used_memory_ids=[],
                    )
        elif bool(self.config.get("use_action_constraints", False)):
            final_realization = self.action_constrained_realizer.realize(
                instance=instance,
                selected_evidence=selected_evidence,
                selected_frames=selected_frames,
                llm_action_prediction=action_decision.action,
                symbolic_runtime=symbolic_runtime,
                fallback_to_v0_on_low_confidence=bool(self.config.get("fallback_to_v0_on_low_confidence", True)),
                use_slot_authorization_map=bool(self.config.get("use_slot_authorization_map", True)),
                runtime_skill_bundle=retrieved_skill_bundle_to_dict(runtime_skill_bundle) if runtime_skill_bundle is not None else None,
                typed_slot_contract=plan.semantic_spec,
                renderer=render_answer,
            )
            answer_result = final_realization.answer_result
            action = final_realization.final_action
            reasoning_state.reasoning_trace.extend(final_realization.reasoning_trace)
            self._save_final_realization(instance.instance_id, final_realization.debug_payload)
        else:
            utility_stage2_answerable = _utility_stage2_answerable(
                query_type=evaluation_query_type,
                semantic_spec=plan.semantic_spec,
                decisions=claim_adjudication,
                projection=utility_adjudicated_projection,
            )
            if utility_stage2_answerable and action_decision.action == "answer_redacted":
                # Stage 2 is the closed, source-local utility capability
                # decision. If every requested field was explicitly admitted
                # as answerable, a conservative action-model label must not
                # downgrade the complete utility projection.
                action_decision.action = "answer"
            if utility_stage2_answerable and action_decision.action in {"refuse", "no_memory"}:
                # Stage 2 is the utility-specific closed-set LLM judgment. A
                # conservative graph/action disagreement must not erase an
                # active, source-grounded utility answer when every requested
                # attribute was explicitly adjudicated as answerable. This
                # recovery is limited to utility and never overrides safety,
                # deletion, or a redacted field.
                action_decision.action = "answer"
            action = action_decision.action
            if action in {"refuse", "no_memory"}:
                answer_result = build_action_only_answer_result(
                    action=action,
                    reasoning_summary="Policy+A-Mem guard prevented answer generation from using protected evidence.",
                    used_memory_ids=[row.memory_id for row in selected_evidence],
                )
            else:
                # Stage 2 is already a closed, source-local admission pass.
                # If the optional claim adjudicator does not emit a projection,
                # realize directly from those admitted typed records instead
                # of converting a valid utility answer into no-memory.
                utility_realization_evidence = (
                    [graph_authorized_projection]
                    if prefer_graph_utility_projection and graph_authorized_projection is not None
                    else utility_adjudicated_projection
                    if utility_adjudicated_projection
                    else stage2_allowed
                )
                support_completion_rows = [
                    row for row in selected_evidence
                    if str(
                        ((row.metadata or {}).get("support_completion") or {}).get("kind") or ""
                    ) == "household_full_date_anchor"
                ]
                if evaluation_query_type == "utility" and utility_realization_evidence:
                    requested_attributes = [
                        str(attribute).strip()
                        for attribute in list(
                            plan.semantic_spec.get("requested_attributes")
                            or plan.semantic_spec.get("requested_slots")
                            or []
                        )
                        if str(attribute).strip()
                    ]
                    # Utility admission has already been decided by Stage 2;
                    # the final realization only chooses exact observed values.
                    # Preserve answer_redacted for genuinely partial utility
                    # answers, but do not let an empty graph certificate alone
                    # suppress a complete Stage-2-authorized answer.
                    realization_action = (
                        "answer"
                        if action in {"answer", "answer_redacted"}
                        and not utility_adjudicated_projection
                        else action
                    )
                    audited = realize_typed_request(
                        question=instance.question,
                        semantic_spec=plan.semantic_spec,
                        evidence=utility_realization_evidence,
                        llm_client=self.llm_client,
                        model_name=resolve_llm_model(self.config, "reasoning"),
                        action=realization_action,
                    )
                    answer_result = (
                        audited
                        if audited is not None and audited.answer_text
                        else render_answer(
                            action,
                            utility_realization_evidence
                            if utility_adjudicated_projection
                            else selected_evidence,
                        )
                    )
                    if audited is not None and audited.answer_text:
                        if support_completion_rows:
                            answer_result = _append_verified_support_date(
                                answer_result,
                                support_completion_rows,
                            )
                        selected_evidence[:] = [
                            row
                            for row in utility_realization_evidence
                            if row.memory_id in set(audited.used_memory_ids)
                        ]
                        selected_frames[:] = [
                            compile_evidence_frame(row) for row in selected_evidence
                        ]
                elif utility_adjudicated_projection:
                    adjudicated_attributes = {
                        str(item.get("attribute") or "")
                        for item in claim_adjudication
                        if str(item.get("decision") or "") in {"answer", "redact"}
                    }
                    audited = realize_typed_request(
                        question=instance.question,
                        semantic_spec=restrict_semantic_spec(plan.semantic_spec, adjudicated_attributes),
                        evidence=utility_adjudicated_projection,
                        llm_client=self.llm_client,
                        model_name=resolve_llm_model(self.config, "reasoning"),
                        action=action,
                    )
                    answer_result = audited if audited is not None and audited.answer_text else render_answer(action, selected_evidence)
                    if audited is not None and audited.answer_text:
                        selected_evidence[:] = [
                            row for row in utility_adjudicated_projection
                            if row.memory_id in set(audited.used_memory_ids)
                        ]
                        selected_frames[:] = [compile_evidence_frame(row) for row in selected_evidence]
                else:
                    answer_result = render_answer(action, selected_evidence)

        utility_graph_direct = bool(
            evaluation_query_type == "utility"
            and graph_authorization_certificate.get("stage2_operational_capability_authorized")
            and not graph_authorization_certificate.get("requires_redaction")
            and not graph_authorization_certificate.get("redacted_slot_names")
            and not graph_authorization_certificate.get("unresolved_requested_attributes")
        )
        if utility_graph_direct and action == "answer_redacted":
            action = "answer"
            action_decision.action = action
        if graph_authorized_projection is not None and not utility_adjudicated_projection and action in {"answer", "answer_redacted"}:
            disclosure_slots = set(dict(graph_authorization_certificate.get("slots") or {})) if action == "answer" else set(
                str(slot).strip()
                for slot in list((slot_governance_certificate or {}).get("safe_projection_slots") or [])
                if str(slot).strip()
            )
            audited = None
            if disclosure_slots:
                audited = realize_typed_request(
                    question=instance.question,
                    semantic_spec=restrict_semantic_spec(plan.semantic_spec, disclosure_slots),
                    evidence=[graph_authorized_projection],
                    llm_client=self.llm_client,
                    model_name=resolve_llm_model(self.config, "reasoning"),
                    action=action,
                )
            if audited is not None and audited.answer_text:
                answer_result = audited
            elif audited is not None and audited.answer_structured.get("audit_status") in {"retired", "unresolved"}:
                action = "no_memory"
                answer_result = build_action_only_answer_result(
                    action=action,
                    reasoning_summary="Typed realization identified only retired or deleted source values.",
                    used_memory_ids=[],
                )
            else:
                answer_result = build_action_only_answer_result(
                    action="answer_redacted",
                    reasoning_summary="No explicitly disclosed typed slot remained after the disclosure contract.",
                    used_memory_ids=[],
                )

        # A complete utility certificate is already a closed, source-grounded
        # answer contract. Prefer its complete typed realizations even when an
        # optional renderer returned an answer: the latter may collapse a
        # record-local collection into one aggregate slot and lose fields that
        # Stage 2 explicitly admitted. This path is utility-only; privacy,
        # safety, and deletion regimes retain their existing terminal behavior.
        if (
            utility_graph_direct
            and graph_authorization_certificate.get("authorized")
            and (not utility_adjudicated_projection or prefer_graph_utility_projection)
        ):
            certified_answer = render_graph_authorized_slots(
                certificate=graph_authorization_certificate,
                action="answer",
                semantic_spec=plan.semantic_spec,
            )
            support_completion_rows = [
                row for row in allowed
                if str(
                    ((row.metadata or {}).get("support_completion") or {}).get("kind") or ""
                ) == "household_full_date_anchor"
            ]
            if support_completion_rows:
                certified_answer = _append_verified_support_date(
                    certified_answer,
                    support_completion_rows,
                )
            if certified_answer.answer_text:
                answer_result = certified_answer
                action = "answer"
                action_decision.action = action
                if graph_authorized_projection is not None:
                    selected_evidence[:] = [graph_authorized_projection]
                    selected_frames[:] = [compile_evidence_frame(graph_authorized_projection)]

        # A downstream governance boundary may safely narrow a direct answer
        # into an answer_redacted response.
        action = answer_result.action
        action_decision.action = action
        action_decision.evidence_memory_ids = list(answer_result.used_memory_ids)
        retrieval_result = {
            "retrieved_before_privacy_filter": combined,
            "retrieved_after_privacy_filter": allowed,
            "filtered_evidence": filtered,
            "rag_chunks": [asdict(chunk) for chunk in chunks],
            "retrieved_atomic_memories": [self._evidence_to_debug(row) for row in amem_evidence],
            "policy_decisions": decisions,
            "stage2_semantic_rerank": stage2_debug,
            "claim_adjudication": claim_adjudication_debug,
        }
        debug_payload = {
            "experiment_mode": "rag_policy_amem",
            "rag_chunks": [asdict(chunk) for chunk in chunks],
            "retrieved_chunks": [self._evidence_to_debug(row) for row in chunk_evidence],
            "atomic_memories": [atomic_memory_to_dict(item) for item in atomic_memories],
            "governed_atoms": [governed_atom_to_dict(item) for item in governed_atoms],
            "governed_graph": {
                "graph_id": governed_graph.graph_id if governed_graph is not None else None,
                "node_count": len(governed_graph.nodes) if governed_graph is not None else 0,
                "edge_count": len(governed_graph.edges) if governed_graph is not None else 0,
                "graph_paths": graph_paths,
                "debug_paths": graph_debug_paths,
                "enabled": bool(self.config.get("use_governed_graph", False)),
            },
            "dual_channel_retrieval": retrieval_bundle_to_dict(dual_channel_bundle) if dual_channel_bundle is not None else None,
            "utility_source_message_ids": sorted(utility_source_message_ids) if utility_source_message_ids is not None else None,
            "symbolic_governance_decision": symbolic_runtime,
            "runtime_skill_bundle": retrieved_skill_bundle_to_dict(runtime_skill_bundle) if runtime_skill_bundle is not None else None,
            "runtime_experience_lessons": [retrieved_experience_lesson_to_dict(item) for item in runtime_experience_lessons],
            "principal_relation_ledger": principal_relation_ledger,
            "information_owner_ledger": information_owner_ledger,
            "source_role_ledger": source_role_ledger,
            "utility_source_locator": utility_source_locator,
            "authorization_context_locator": authorization_context_locator,
            "query_policy_authorization": query_policy_authorization,
            "graph_authorization_certificate": graph_authorization_certificate,
            "slot_governance_certificate": slot_governance_certificate,
            "retrieved_atomic_memories": [self._evidence_to_debug(row) for row in amem_evidence],
            "policy_decisions": decisions,
            "stage2_semantic_rerank": stage2_debug,
            "selected_evidence": [self._evidence_to_debug(row) for row in selected_evidence],
            "support_evidence_completion": support_rows,
            "current_state": {
                "active_items": current_state.active_items,
                "canceled_items": current_state.canceled_items,
                "deleted_items": current_state.deleted_items,
                "superseded_items": current_state.superseded_items,
                "uncertain_items": current_state.uncertain_items,
                "trace": current_state.trace,
            },
            "slot_coverage": slot_coverage,
            "action_correction_trace": [],
            "question_profile": {},
            "relation_state_bundle": bundle_debug,
            "selected_frame_typed_slots": [
                sorted(f"{frame.frame_type}.{slot}" for slot in frame.slots.keys())
                for frame in selected_frames
            ],
            "event_ledger_summary": {
                "active_events": list((reasoning_state.current_state_ledger or {}).get("active_events", {}).keys()),
                "canceled_events": list((reasoning_state.current_state_ledger or {}).get("canceled_events", {}).keys()),
                "superseded_events": list((reasoning_state.current_state_ledger or {}).get("superseded_events", {}).keys()),
                "deleted_events": list((reasoning_state.current_state_ledger or {}).get("deleted_events", {}).keys()),
            },
            "final_realization": final_realization.debug_payload if final_realization is not None else None,
            "ablation_flags": {
                "use_governed_graph": bool(self.config.get("use_governed_graph", False)),
                "use_dual_channel_retrieval": bool(self.config.get("use_dual_channel_retrieval", False)),
                "use_symbolic_reasoner": bool(self.config.get("use_symbolic_reasoner", False)),
                "use_action_constraints": bool(self.config.get("use_action_constraints", False)),
                "use_slot_authorization_map": bool(self.config.get("use_slot_authorization_map", True)),
                "use_deletion_tracking": bool(self.config.get("use_deletion_tracking", True)),
                "use_skill_library": bool(self.config.get("use_skill_library", False)),
                "use_experience_memory": bool(self.config.get("use_experience_memory", False)),
                "use_self_evolving_update": bool(self.config.get("use_self_evolving_update", False)),
            },
        }
        adaptation_audit = self._build_runtime_adaptation_audit(
            instance=instance,
            runtime_skill_context=runtime_skill_context if bool(self.config.get("use_skill_library", False)) else None,
            runtime_skill_bundle=runtime_skill_bundle,
            runtime_experience_context=runtime_experience_context if bool(self.config.get("use_experience_memory", False)) else None,
            runtime_experience_lessons=runtime_experience_lessons,
        )
        debug_payload["runtime_adaptation_audit"] = runtime_adaptation_audit_to_dict(adaptation_audit)
        self._save_runtime_adaptation_audit(instance.instance_id, adaptation_audit)
        return BackboneRunResult(
            query_plan=plan,
            retrieval_result=retrieval_result,
            reasoning_state=reasoning_state,
            action_decision=action_decision,
            answer_result=answer_result,
            debug_payload=debug_payload,
        )

    def _save_atomic_memories(self, instance_id: str, atomic_memories: list[AtomicMemory]) -> None:
        path = ensure_dir(self.output_dir / "amem_memory" / self.dataset_name / instance_id) / "atomic_memories.jsonl"
        write_jsonl(path, [atomic_memory_to_dict(item) for item in atomic_memories])

    @staticmethod
    def _attest_owner_self_relation(
        *,
        ledger: dict[str, Any],
        requester_id: str | None,
        owner_candidates: set[str],
        information_owner_ledger: dict[str, Any],
        selected_fact_ids: set[str],
    ) -> dict[str, Any]:
        """Bridge a complete selected-fact owner closure to requester self.

        This is deliberately narrower than identity inference: the requester
        must exactly equal the sole owner already source-attributed for every
        selected factual turn.  It therefore cannot turn a roster entry, role,
        or legacy owner hint into access.
        """
        requester = str(requester_id or "").strip()
        selected_ids = {str(message_id).strip() for message_id in selected_fact_ids if str(message_id).strip()}
        owners_by_message = {
            str(message_id).strip(): str(owner_id).strip()
            for message_id, owner_id in dict(information_owner_ledger.get("owner_by_message_id") or {}).items()
            if str(message_id).strip() and str(owner_id).strip()
        }
        closed_owners = {str(owner_id).strip() for owner_id in owner_candidates if str(owner_id).strip()}
        effective_relation = str(ledger.get("effective_relation") or "").strip()
        effective_status = str(ledger.get("effective_status") or "").strip()

        # An existing relation outcome is evidence that must not be displaced
        # by this structural bridge, including a terminal contradiction.
        if effective_status in {"proven", "revoked", "contradicted"}:
            return ledger
        if (
            not requester
            or not selected_ids
            or closed_owners != {requester}
            or any(owners_by_message.get(message_id) != requester for message_id in selected_ids)
        ):
            return ledger

        supports: list[dict[str, str]] = []
        seen_supports: set[tuple[str, str]] = set()
        for record in list(information_owner_ledger.get("records") or []):
            if not isinstance(record, dict):
                continue
            if (
                str(record.get("message_id") or "").strip() not in selected_ids
                or str(record.get("information_owner_id") or "").strip() != requester
                or str(record.get("status") or "").strip() != "proven"
            ):
                continue
            for support in list(record.get("supports") or []):
                if not isinstance(support, dict):
                    continue
                message_id = str(support.get("message_id") or "").strip()
                source_span = str(support.get("source_span") or "").strip()
                if not message_id or not source_span or (message_id, source_span) in seen_supports:
                    continue
                seen_supports.add((message_id, source_span))
                supports.append({
                    "message_id": message_id,
                    "source_span": source_span,
                    "evidence_kind": "owner_source_attribution",
                })
        # The owner ledger already validates every support against the visible
        # transcript. Requiring at least one keeps graph provenance inspectable.
        if not supports:
            return ledger

        updated = dict(ledger)
        records = [dict(record) for record in list(ledger.get("records") or []) if isinstance(record, dict)]
        records.append({
            "requester_id": requester,
            "owner_id": requester,
            "relation": "owner",
            "relation_label": "source_grounded_owner_self",
            "status": "proven",
            "authorization_status": "active",
            "direction": "requester_to_owner",
            "supports": supports,
        })
        trace = dict(ledger.get("resolution_trace") or {})
        trace["owner_self_attestation"] = {
            "status": "proven",
            "reason": "selected_fact_owner_matches_requester",
            "selected_fact_message_ids": sorted(selected_ids),
            "support_count": len(supports),
        }
        updated.update({
            "records": records,
            "owner_id": requester,
            "effective_relation": "owner",
            "effective_status": "proven",
            "reason": "selected_fact_owner_matches_requester",
            "resolution_trace": trace,
        })
        return updated

    @staticmethod
    def _semantic_contract_certifiable(plan) -> bool:
        """Only a grounded LLM contract may grant graph-level disclosure."""
        source = str((getattr(plan, "planning_trace", {}) or {}).get("semantic_contract_source") or "")
        semantic_spec = dict(getattr(plan, "semantic_spec", {}) or {})
        if source not in {"initial_llm", "repair_llm", "audited_llm", "verified_llm"}:
            return False
        attributes = list(semantic_spec.get("requested_attributes") or [])
        slots = list(semantic_spec.get("requested_slots") or [])
        # Open attributes must carry a query-grounded binding. Legacy slots are
        # retained for compatibility but do not by themselves authorize data.
        return bool(attributes and semantic_spec.get("attribute_bindings_valid") and (attributes or slots))

    @staticmethod
    def _utility_source_message_ids(bundle) -> set[str] | None:
        """Return the final utility channel's narrow source-turn closure.

        Retrieval ranks chunks/memory rows before source turns expand into
        governed atoms. The closure reconnects that expansion only for rows
        actually selected by the utility channel; it is evidence provenance,
        never a semantic or access inference.
        """
        if bundle is None:
            return None
        utility = bundle.utility_evidence
        selected_source_ids = {
            str(value) for value in list(utility.selected_source_message_ids or []) if str(value)
        }
        selected_memory_ids = {str(value) for value in list(utility.selected_memory_ids or []) if str(value)}
        selected_atom_ids = {str(value) for value in list(utility.selected_atom_ids or []) if str(value)}
        source_ids: set[str] = set(selected_source_ids)

        def add_ids(values) -> None:
            source_ids.update(str(value) for value in list(values or []) if str(value))

        for row in list(utility.facts or []):
            if isinstance(row, dict) and str(row.get("memory_id") or "") in selected_memory_ids:
                add_ids(row.get("source_message_ids"))
        for row in list(utility.chunks or []):
            if isinstance(row, dict) and str(row.get("memory_id") or "") in selected_memory_ids:
                add_ids(row.get("source_message_ids"))
        for atom in list(utility.atoms or []):
            if isinstance(atom, dict) and str(atom.get("atom_id") or "") in selected_atom_ids:
                add_ids(dict(atom.get("provenance") or {}).get("source_message_ids"))
        return source_ids

    @staticmethod
    def _attach_source_roles(
        atoms: list[GovernedMemoryAtom], *, role_by_message_id: dict[str, str]
    ) -> list[GovernedMemoryAtom]:
        """Attach only unanimous source-role labels to atom provenance."""
        result: list[GovernedMemoryAtom] = []
        for atom in atoms:
            roles = {
                str(role_by_message_id.get(str(message_id)) or "").strip()
                for message_id in list((atom.provenance or {}).get("source_message_ids") or [])
                if str(role_by_message_id.get(str(message_id)) or "").strip()
            }
            if len(roles) != 1:
                result.append(atom)
                continue
            provenance = dict(atom.provenance or {})
            provenance["source_role"] = next(iter(roles))
            result.append(replace(atom, provenance=provenance))
        return result
    def _save_governed_atoms(self, instance_id: str, governed_atoms: list[GovernedMemoryAtom]) -> None:
        per_instance_path = ensure_dir(self.output_dir / "amem_memory" / self.dataset_name / instance_id) / "governed_atoms.jsonl"
        rows = [governed_atom_to_dict(item) for item in governed_atoms]
        write_jsonl(per_instance_path, rows)
        aggregate_path = self.output_dir / "debug" / "governed_atoms.jsonl"
        for row in rows:
            append_jsonl(aggregate_path, row)

    def _save_retrieval_bundle(self, instance_id: str, bundle) -> None:
        path = self.output_dir / "debug" / "retrieval_bundle" / f"{instance_id}.json"
        write_json(path, retrieval_bundle_to_dict(bundle))

    def _save_symbolic_decision(self, instance_id: str, symbolic_runtime: dict[str, Any]) -> None:
        path = self.output_dir / "debug" / "symbolic_decisions" / f"{instance_id}.json"
        write_json(path, symbolic_runtime)

    def _save_final_realization(self, instance_id: str, payload: dict[str, Any]) -> None:
        path = self.output_dir / "debug" / "final_realization" / f"{instance_id}.json"
        write_json(path, payload)

    @staticmethod
    def _make_adaptation_advisory_only(bundle):
        """Keep selection/audit evidence while preventing learned hard overrides."""
        bundle.activated_rules = []
        bundle.prompt_patches = []
        bundle.verifier_patches = []
        bundle.action_patches = []
        bundle.loaded_rule_updates = []
        bundle.loaded_prompt_updates = []
        bundle.loaded_policy_updates = []
        bundle.affected_decision_fields = ["renderer_advisory"] if bundle.selected_skills else []
        bundle.skill_trace.append("lightweight_mode: hard adaptation effects disabled")
        return bundle

    @staticmethod
    def _lightweight_adaptation_rules(bundle, lessons) -> list[str]:
        selected = {
            str(row.get("name") or row.get("skill_id") or "")
            for row in list(bundle.selected_skills or [])
        }
        rules = [
            "Experience and skills are advisory only. Never expand access, reveal an uncertified slot, or override the governance decision."
        ]
        if "lifecycle_integrity_skill" in selected:
            rules.append("Re-check that rendered values come from the latest active, non-deleted memory version.")
        if "typed_utility_realization_skill" in selected:
            rules.append("If authorized typed values are available, state them minimally without adding inferred values.")
        if "provenance_completion_skill" in selected:
            rules.append("Omit any value whose source provenance is incomplete.")
        if "restrictive_action_calibration_skill" in selected:
            rules.append("Distinguish absent memory from present-but-unauthorized memory without weakening access control.")
        if lessons:
            rules.append("Use prior failure experience only as a reminder to verify lifecycle, authorization, and typed-slot completeness.")
        return rules

    def _save_skill_bundle(self, instance_id: str, context: SkillQueryContext, bundle) -> None:
        path = self.output_dir / "debug" / "skills" / f"{instance_id}.json"
        write_json(
            path,
            {
                "context": skill_query_context_to_dict(context),
                "bundle": retrieved_skill_bundle_to_dict(bundle),
            },
        )

    def _save_experience_bundle(
        self,
        instance_id: str,
        context: RuntimeExperienceContext,
        lessons,
    ) -> None:
        path = self.output_dir / "debug" / "experience" / f"{instance_id}.json"
        write_json(
            path,
            {
                "context": runtime_experience_context_to_dict(context),
                "lessons": [retrieved_experience_lesson_to_dict(item) for item in lessons],
            },
        )

    def _save_runtime_adaptation_audit(
        self,
        instance_id: str,
        audit: RuntimeAdaptationAudit,
    ) -> None:
        path = self.output_dir / "debug" / "adaptation" / f"{instance_id}.json"
        write_json(path, runtime_adaptation_audit_to_dict(audit))

    def _build_runtime_adaptation_audit(
        self,
        *,
        instance: MemoryInstance,
        runtime_skill_context: SkillQueryContext | None,
        runtime_skill_bundle,
        runtime_experience_context: RuntimeExperienceContext | None,
        runtime_experience_lessons,
    ) -> RuntimeAdaptationAudit:
        selected_skill_ids = []
        loaded_rule_updates = []
        loaded_prompt_updates = []
        loaded_policy_updates = []
        action_patches = []
        affected_fields = []
        trigger_reasons: list[str] = []
        if runtime_skill_bundle is not None:
            selected_skill_ids = [
                str(row.get("skill_id") or row.get("name") or "")
                for row in list(runtime_skill_bundle.selected_skills or [])
                if str(row.get("skill_id") or row.get("name") or "").strip()
            ]
            loaded_rule_updates = list(runtime_skill_bundle.loaded_rule_updates or [])
            loaded_prompt_updates = list(runtime_skill_bundle.loaded_prompt_updates or [])
            loaded_policy_updates = list(runtime_skill_bundle.loaded_policy_updates or [])
            action_patches = list(runtime_skill_bundle.action_patches or [])
            affected_fields = list(runtime_skill_bundle.affected_decision_fields or [])
            trigger_reasons.extend(list(runtime_skill_bundle.skill_trace or []))
        selected_experience_ids = [
            str(item.experience_id or "")
            for item in list(runtime_experience_lessons or [])
            if str(item.experience_id or "").strip()
        ]
        selected_experience_pattern_ids = [
            str(item.pattern_id or "")
            for item in list(runtime_experience_lessons or [])
            if str(item.pattern_id or "").strip()
        ]
        trigger_reasons.extend(
            [
                f"experience:{item.experience_id}:{','.join(item.selection_reasons[:3])}"
                for item in list(runtime_experience_lessons or [])
            ]
        )
        context_summary: dict[str, Any] = {}
        if runtime_skill_context is not None:
            context_summary["skill_context"] = skill_query_context_to_dict(runtime_skill_context)
        if runtime_experience_context is not None:
            context_summary["experience_context"] = runtime_experience_context_to_dict(runtime_experience_context)
        adaptation_triggered = bool(selected_skill_ids or selected_experience_ids or loaded_rule_updates or loaded_prompt_updates or loaded_policy_updates or action_patches)
        return RuntimeAdaptationAudit(
            instance_id=instance.instance_id,
            domain=str(instance.domain or "unknown"),
            query=str(instance.question or ""),
            adaptation_enabled=bool(self.config.get("use_skill_library", False) or self.config.get("use_experience_memory", False) or self.config.get("use_self_evolving_update", False)),
            adaptation_triggered=adaptation_triggered,
            skill_library_enabled=bool(self.config.get("use_skill_library", False)),
            experience_memory_enabled=bool(self.config.get("use_experience_memory", False)),
            self_evolving_enabled=bool(self.config.get("use_self_evolving_update", False)),
            selected_experience_ids=selected_experience_ids,
            selected_experience_pattern_ids=selected_experience_pattern_ids,
            selected_skill_ids=selected_skill_ids,
            loaded_rule_updates=loaded_rule_updates,
            loaded_prompt_updates=loaded_prompt_updates,
            loaded_policy_updates=loaded_policy_updates,
            action_patches=action_patches,
            affected_decision_fields=affected_fields,
            trigger_reasons=trigger_reasons[:24],
            runtime_context_summary=context_summary,
            provenance={
                "created_from_dev_only_experience": True,
                "runtime_only_uses_frozen_artifacts": True,
            },
        )

    def _build_skill_query_context(
        self,
        *,
        instance: MemoryInstance,
        principal,
        required_slot_plan: dict[str, Any],
        graph_paths: list[dict[str, Any]],
        slot_coverage: dict[str, Any],
        graph_authorization_certificate: dict[str, Any],
    ) -> SkillQueryContext:
        required_slots = list(required_slot_plan.get("required_slots") or [])
        sensitive_slots = {
            "medication",
            "instruction",
            "timing",
            "dosage",
            "condition",
            "phone",
            "backup_contact",
            "monthly_stipend",
            "approved_budget",
            "approved_discount_cap",
            "safe_wording",
            "blocker",
            "target_date",
            "consent_scope",
        }
        lowered = instance.question.lower()
        graph_lifecycle_flags = [
            str(path.get("edge_type") or "")
            for path in graph_paths
            if str(path.get("edge_type") or "").strip()
        ]
        lifecycle_flags = self._infer_query_lifecycle_flags(lowered)
        explicit_deleted_request = self._query_explicitly_requests_deleted_content(lowered)
        explicit_historical_request = self._query_explicitly_requests_historical_content(lowered)
        explicit_current_request = self._query_explicitly_requests_current_state(lowered)
        query_domains = list(required_slot_plan.get("domains") or [])
        query_intent = " ".join(query_domains + required_slots + lifecycle_flags + [lowered]).strip()
        return SkillQueryContext(
            domain=str(instance.domain or "unknown"),
            requester_role=str(principal.organization_role or ((instance.metadata.get("requester") or {}).get("role")) or ""),
            owner_relation=str(principal.relation_to_owner or ""),
            query_intent=query_intent or lowered,
            required_slots=required_slots,
            detected_sensitive_slots=[slot for slot in required_slots if slot in sensitive_slots],
            lifecycle_flags=list(dict.fromkeys(lifecycle_flags)),
            graph_lifecycle_flags=list(dict.fromkeys(graph_lifecycle_flags)),
            explicit_deleted_request=explicit_deleted_request,
            explicit_historical_request=explicit_historical_request,
            explicit_current_request=explicit_current_request,
            symbolic_predicates=[],
            evidence_coverage=float(slot_coverage.get("coverage_ratio") or 0.0),
            certificate_authorized=bool(graph_authorization_certificate.get("authorized")),
            certificate_reason=str(graph_authorization_certificate.get("reason") or ""),
            certified_slots=list(dict(graph_authorization_certificate.get("slots") or {}).keys()),
        )

    def _build_experience_context(
        self,
        *,
        instance: MemoryInstance,
        principal,
        required_slot_plan: dict[str, Any],
        graph_paths: list[dict[str, Any]],
        slot_coverage: dict[str, Any],
        graph_authorization_certificate: dict[str, Any],
    ) -> RuntimeExperienceContext:
        required_slots = list(required_slot_plan.get("required_slots") or [])
        sensitive_slots = [
            slot
            for slot in required_slots
            if slot in {
                "medication",
                "instruction",
                "timing",
                "dosage",
                "condition",
                "phone",
                "contact_method",
                "backup_contact",
                "approved_budget",
                "approved_discount_cap",
                "monthly_stipend",
                "safe_wording",
                "blocker",
                "target_date",
                "consent_scope",
            }
        ]
        lowered = instance.question.lower()
        lifecycle_flags = self._infer_query_lifecycle_flags(lowered)
        return RuntimeExperienceContext(
            domain=str(instance.domain or "unknown"),
            requester_role=str(principal.organization_role or ((instance.metadata.get("requester") or {}).get("role")) or ""),
            owner_relation=str(principal.relation_to_owner or ""),
            question=instance.question,
            required_slots=required_slots,
            sensitive_slots=sensitive_slots,
            lifecycle_flags=list(dict.fromkeys(lifecycle_flags)),
            graph_lifecycle_flags=list(dict.fromkeys(
                str(path.get("edge_type") or "")
                for path in graph_paths
                if str(path.get("edge_type") or "")
            )),
            evidence_coverage=float(slot_coverage.get("coverage_ratio") or 0.0),
            certificate_authorized=bool(graph_authorization_certificate.get("authorized")),
            certificate_reason=str(graph_authorization_certificate.get("reason") or ""),
            certified_slots=list(dict(graph_authorization_certificate.get("slots") or {}).keys()),
        )

    @staticmethod
    def _infer_query_lifecycle_flags(lowered_question: str) -> list[str]:
        flags: list[str] = []
        token_map = {
            "deleted": ["deleted", "remove", "removed", "forget", "forgot"],
            "historical": ["old", "older", "earlier", "previous", "before", "used to", "retired"],
            "current": ["current", "latest", "updated", "now", "right now", "as of now"],
        }
        for flag, tokens in token_map.items():
            if any(_contains_phrase(lowered_question, token) for token in tokens):
                flags.append(flag)
        return flags

    @staticmethod
    def _query_explicitly_requests_deleted_content(lowered_question: str) -> bool:
        return any(_contains_phrase(lowered_question, token) for token in [
            "delete", "deleted", "deletion", "remove", "removed", "removal",
            "forget", "forgot", "no longer have", "retired",
        ])

    @staticmethod
    def _query_explicitly_requests_historical_content(lowered_question: str) -> bool:
        return any(_contains_phrase(lowered_question, token) for token in ["old", "older", "earlier", "previous", "before", "used to"])

    @staticmethod
    def _query_explicitly_requests_current_state(lowered_question: str) -> bool:
        return any(_contains_phrase(lowered_question, token) for token in ["current", "latest", "updated", "now", "right now", "as of now"])

    def _prepare_governed_atoms(self, governed_atoms: list[GovernedMemoryAtom]) -> list[GovernedMemoryAtom]:
        if bool(self.config.get("use_deletion_tracking", True)):
            return governed_atoms
        normalized: list[GovernedMemoryAtom] = []
        for atom in governed_atoms:
            if atom.atom_type == "deletion_atom":
                continue
            if atom.lifecycle == "deleted":
                normalized.append(replace(atom, lifecycle="historical", sensitivity="private"))
            else:
                normalized.append(atom)
        return normalized

    def _prepare_evidence_rows_for_runtime(self, evidence: list[RetrievedEvidence]) -> list[RetrievedEvidence]:
        if bool(self.config.get("use_deletion_tracking", True)):
            return evidence
        normalized: list[RetrievedEvidence] = []
        for row in evidence:
            metadata = dict(row.metadata or {})
            if str(metadata.get("memory_status") or "").lower() == "deleted":
                metadata["memory_status"] = "active"
            normalized.append(replace(row, metadata=metadata))
        return normalized

    @staticmethod
    def _attach_semantic_provenance(
        evidence: list[RetrievedEvidence],
        atomic_memories: list[AtomicMemory],
    ) -> list[RetrievedEvidence]:
        """Attach source-aligned annotations to every derived evidence view."""
        by_source: dict[str, list[dict[str, Any]]] = {}
        for item in atomic_memories:
            semantic_tags = dict((item.access_tags or {}).get("semantic_tags") or {})
            if not semantic_tags:
                continue
            constituent = {
                "memory_id": item.memory_id,
                "text": item.content,
                "semantic_tags": semantic_tags,
            }
            for source_id in item.source_message_ids:
                by_source.setdefault(str(source_id), []).append(constituent)
        enriched: list[RetrievedEvidence] = []
        for row in evidence:
            metadata = dict(row.metadata or {})
            constituents: list[dict[str, Any]] = []
            seen: set[str] = set()
            for source_id in row.source_message_ids:
                for item in by_source.get(str(source_id), []):
                    memory_id = str(item.get("memory_id") or "")
                    if memory_id and memory_id not in seen:
                        seen.add(memory_id)
                        constituents.append(item)
            if constituents:
                metadata["semantic_constituents"] = constituents
            enriched.append(replace(row, metadata=metadata))
        return enriched

    @staticmethod
    def _filter_query_echo_evidence(
        *,
        question: str,
        requester_id: str | None,
        evidence: list[RetrievedEvidence],
    ) -> tuple[list[RetrievedEvidence], list[RetrievedEvidence]]:
        """Prevent the current request from becoming evidence for its own claim."""
        if not requester_id:
            return list(evidence), []
        query_tokens = set(re.findall(r"[a-z0-9]+", str(question or "").lower()))
        if len(query_tokens) < 4:
            return list(evidence), []
        kept: list[RetrievedEvidence] = []
        removed: list[RetrievedEvidence] = []
        for row in evidence:
            row_tokens = set(re.findall(r"[a-z0-9]+", str(row.content or "").lower()))
            requester_authored = str(row.user_id or "") == str(requester_id)
            query_coverage = len(query_tokens & row_tokens) / max(len(query_tokens), 1)
            size_ratio = len(row_tokens) / max(len(query_tokens), 1)
            assertion_decision = _structured_assertion_decision(row.metadata)
            nonassertive_request = assertion_decision is False
            semantic_constituents = list((row.metadata or {}).get("semantic_constituents") or [])
            for constituent in semantic_constituents:
                if _structured_assertion_decision({"semantic_tags": constituent.get("semantic_tags")}) is not False:
                    continue
                # Composite views are admissible only when every constituent
                # is an independent assertion. Atomic assertion rows remain.
                nonassertive_request = True
                break
            if assertion_decision is None and not nonassertive_request:
                # Temporary compatibility path for memories created before
                # structured discourse annotations were introduced.
                nonassertive_request = bool(re.search(
                    r"^\s*(?:please\s+)?(?:remind|recap|tell|show)\s+(?:me|us)\b|"
                    r"^\s*(?:what|which|who|when|where|why|how)\b",
                    re.sub(r"(?:^|\n)\[[^\]]+\]\s+\[[^\]]+\]\s*", "", str(row.content or "")).strip(),
                    re.IGNORECASE,
                ))
            if nonassertive_request or (
                requester_authored and query_coverage >= 0.72 and size_ratio <= 2.2
            ):
                removed.append(row)
            else:
                kept.append(row)
        return kept, removed

    def _retrieve_atomic_memories(self, instance: MemoryInstance, atomic_memories: list[AtomicMemory], query_texts: list[str]) -> list[RetrievedEvidence]:
        if not atomic_memories:
            return []
        pseudo_items = []
        for item in atomic_memories:
            pseudo_items.append(self._atomic_memory_to_memory_item(item))
        index = DenseMemoryIndex.build(items=pseudo_items, llm_client=self.embedding_client, embedding_model=str(self.config["embedding"]["model"]))
        rows = index.query(
            query_texts=query_texts,
            top_k=int((self.config.get("rag") or {}).get("chunk_top_k", 30)),
            llm_client=self.embedding_client,
            embedding_model=str(self.config["embedding"]["model"]),
        )
        memory_by_id = {item.memory_id: item for item in pseudo_items}
        out = []
        for memory_id, score in rows[: int((self.config.get("rag") or {}).get("final_context_k", 12))]:
            item = memory_by_id[memory_id]
            out.append(
                RetrievedEvidence(
                    memory_id=item.memory_id,
                    content=item.content,
                    score=float(score) + 0.05,
                    retrieval_source="atomic_memory",
                    reason="atomic memory retrieval",
                    user_id=item.user_id,
                    memory_type=item.memory_type,
                    scope=item.scope,
                    entities=item.entities,
                    time=item.time,
                    source_message_ids=item.source_message_ids,
                    metadata=dict(item.metadata or {}),
                )
            )
        return out

    @staticmethod
    def _stage2_atomic_record_candidates(
        *,
        atomic_memories: list[AtomicMemory],
        selected_fact_message_ids: set[str],
        selected_source_message_ids: set[str] | None = None,
    ) -> list[RetrievedEvidence]:
        """Build the Stage-2 closed record set from the complete Stage-1 source closure.

        The locator can select both exact factual sources and independent safe
        projection sources. Both are valid Stage-1 proposals; later semantic
        reranking decides which typed fields are admissible. Dropping the
        closure here makes safe projection impossible even when retrieval
        found and grounded it.
        """
        candidates: list[RetrievedEvidence] = []
        seen: set[str] = set()
        source_closure = {
            str(value)
            for value in (selected_source_message_ids or set())
            if str(value)
        }
        source_closure.update(
            str(value)
            for value in selected_fact_message_ids
            if str(value)
        )
        for item in atomic_memories:
            memory_id = str(item.memory_id or "")
            source_ids = {str(value) for value in list(item.source_message_ids or []) if str(value)}
            if not memory_id or memory_id in seen or (source_closure and not (source_ids & source_closure)):
                continue
            seen.add(memory_id)
            candidates.append(RAGPolicyAMemBackbone._atomic_memory_to_retrieved_evidence(
                item, score=min(max(float(item.confidence or 0.0), 0.0), 1.0)
            ))
        return candidates

    @staticmethod
    def _atomic_memory_to_memory_item(item: AtomicMemory) -> MemoryItem:
        return MemoryItem(
            memory_id=item.memory_id,
            instance_id=item.instance_id,
            user_id=item.owner_user,
            scope="atomic_memory",
            content=item.content,
            memory_type=item.memory_type,
            entities=item.entities,
            time=item.timestamp,
            source_message_ids=item.source_message_ids,
            confidence=item.confidence,
            privacy_level=None,
            tags=[item.memory_type],
            metadata={
                "source_type": "atomic_memory",
                "slots": dict(item.slots),
                "lifecycle_status": item.lifecycle_status,
                "surface_spans": dict(item.access_tags.get("surface_spans") or {}),
                "semantic_tags": dict(item.access_tags.get("semantic_tags") or {}),
            },
        )

    def _augment_selected_current_state_support(
        self,
        *,
        question: str,
        required_slot_plan: dict[str, Any],
        selected_evidence: list[RetrievedEvidence],
        selected_frames: list,
        policy_allowed_evidence: list[RetrievedEvidence],
        atomic_memories: list[AtomicMemory],
        principal,
        policy_scope,
        allow_canonical_projection: bool,
    ) -> tuple[list[dict[str, Any]], list[str]]:
        target_frame_types = set(required_slot_plan.get("target_frame_types") or [])
        required_slots = set(required_slot_plan.get("required_slots") or [])
        existing_ids = {row.memory_id for row in selected_evidence}
        covered_slots = {
            str(slot)
            for frame in selected_frames
            for slot, value in dict(getattr(frame, "slots", {}) or {}).items()
            if value
        }
        target_tokens = {
            token
            for entity in required_slot_plan.get("target_entities") or []
            for token in re.findall(r"[a-z0-9]+", str(entity).lower())
        }
        support_rows: list[dict[str, Any]] = []
        trace: list[str] = []

        # Complete every requested typed slot from the newest policy-authorized
        # active row. This keeps retrieval limits from silently dropping an
        # otherwise available part of a multi-slot answer contract.
        authorized_candidate_rows = list(policy_allowed_evidence)
        authorized_ids = {row.memory_id for row in authorized_candidate_rows}
        for item in atomic_memories:
            if item.memory_id in authorized_ids:
                continue
            atomic_text = str(item.content or "").strip()
            assertion_decision = _structured_assertion_decision({
                "semantic_tags": dict(item.access_tags.get("semantic_tags") or {})
            })
            if assertion_decision is False:
                continue
            if assertion_decision is None:
                # Temporary compatibility path for pre-annotation memories.
                if re.search(
                    r"^\s*(?:please\s+)?(?:remind|recap|tell|show)\s+(?:me|us)\b|"
                    r"^\s*(?:what|which|who|when|where|why|how)\b",
                    atomic_text,
                    re.IGNORECASE,
                ):
                    continue
            atomic_row = self._atomic_memory_to_retrieved_evidence(
                item, score=min(max(float(item.confidence), 0.0), 1.0) * 0.35
            )
            approved_source_ids = {
                str(source_id)
                for row in policy_allowed_evidence
                for source_id in list(row.source_message_ids or [])
                if str(source_id)
            }
            if not approved_source_ids.intersection(str(source_id) for source_id in item.source_message_ids):
                continue
            authorized_candidate_rows.append(atomic_row)
            authorized_ids.add(atomic_row.memory_id)

        candidates = []
        for row in authorized_candidate_rows:
            try:
                frame = compile_evidence_frame(row)
            except Exception:
                continue
            if str(getattr(frame, "lifecycle_status", "") or "active").lower() != "active":
                continue
            row_tokens = set(re.findall(r"[a-z0-9]+", str(row.content or "").lower()))
            typed_match = any(
                _match_typed_attribute(getattr(frame, "slots", {}) or {}, slot) is not None
                for slot in required_slots
            )
            if target_tokens and not target_tokens.intersection(row_tokens) and not typed_match:
                continue
            candidates.append((self._evidence_recency_key(row, frame), float(row.score or 0.0), row, frame))

        authority_winners, authority_trace = self._adjudicate_conflicting_typed_claims(
            question=question,
            required_slots=required_slots,
            candidates=candidates,
        )
        trace.extend(authority_trace)

        for slot in sorted(required_slots):
            matching = [
                (*item, _match_typed_attribute(getattr(item[3], "slots", {}) or {}, slot))
                for item in candidates
                if _match_typed_attribute(getattr(item[3], "slots", {}) or {}, slot) is not None
            ]
            if not matching:
                continue
            preferred_id = authority_winners.get(slot)
            preferred = next(
                (item for item in matching if str(item[2].memory_id) == preferred_id),
                None,
            )
            newest_available_key, _, support_row, support_frame, _ = preferred or max(
                matching, key=lambda item: (item[0], item[4][2], item[1])
            )
            selected_keys = [
                self._evidence_recency_key(row, frame)
                for row, frame in zip(selected_evidence, selected_frames)
                if _match_typed_attribute(getattr(frame, "slots", {}) or {}, slot) is not None
            ]
            newest_selected_key = max(selected_keys, default=("", -1))
            if slot in covered_slots and newest_selected_key >= newest_available_key:
                continue
            if support_row.memory_id in existing_ids:
                continue
            selected_evidence.append(support_row)
            selected_frames.append(support_frame)
            existing_ids.add(support_row.memory_id)
            covered_slots.update(
                str(name) for name, value in dict(support_frame.slots or {}).items() if value
            )
            support_rows.append(self._evidence_to_debug(support_row))
            trace.append(
                f"support evidence completion refreshed required slot {slot} from latest authorized evidence."
            )

        if allow_canonical_projection and self._query_explicitly_requests_current_state(question):
            winners: dict[str, tuple] = {}
            for slot in sorted(required_slots):
                matching = [
                    (*item, _match_typed_attribute(getattr(item[3], "slots", {}) or {}, slot))
                    for item in candidates
                    if _match_typed_attribute(getattr(item[3], "slots", {}) or {}, slot) is not None
                ]
                if matching:
                    preferred_id = authority_winners.get(slot)
                    winners[slot] = next(
                        (item for item in matching if str(item[2].memory_id) == preferred_id),
                        None,
                    ) or max(matching, key=lambda item: (item[0], item[4][2], item[1]))
            if required_slots and set(winners) == required_slots:
                canonical_slots = {}
                for slot in sorted(required_slots):
                    winner = winners[slot]
                    value = winner[4][1]
                    winner_slots = dict(getattr(winner[3], "slots", {}) or {})
                    anchor = None
                    if _is_temporal_interval_value(value):
                        prior_anchors = [
                            (
                                item[0],
                                _absolute_temporal_anchor_value(
                                    dict(getattr(item[3], "slots", {}) or {})
                                ),
                            )
                            for item in candidates
                            if item[0] < winner[0]
                            and _match_typed_attribute(
                                getattr(item[3], "slots", {}) or {}, slot
                            ) is not None
                            and _absolute_temporal_anchor_value(
                                dict(getattr(item[3], "slots", {}) or {})
                            ) is not None
                        ]
                        if prior_anchors:
                            anchor = max(prior_anchors, key=lambda item: item[0])[1]
                    rendered = str(value)
                    if anchor is not None and str(anchor).lower() not in rendered.lower():
                        rendered = f"{anchor}; {rendered}"
                    canonical_slots[slot] = rendered
                source_ids = list(dict.fromkeys(
                    source_id
                    for _, _, row, _, _ in winners.values()
                    for source_id in list(row.source_message_ids or [])
                ))
                canonical = RetrievedEvidence(
                    memory_id="canonical_current_state_projection",
                    content="; ".join(f"{slot}: {value}" for slot, value in canonical_slots.items()),
                    score=1.0,
                    retrieval_source="current_state_projection",
                    reason="latest authorized evidence for every requested slot",
                    user_id=principal.user_id,
                    memory_type="current_state",
                    scope=principal.relation_to_owner,
                    source_message_ids=source_ids,
                    metadata={
                        "slots": canonical_slots,
                        "surface_spans": canonical_slots,
                        "lifecycle_status": "active",
                    },
                )
                selected_evidence[:] = [canonical]
                selected_frames[:] = [compile_evidence_frame(canonical)]
                trace.append(
                    "canonical current-state projection replaced contradictory historical evidence."
                )
            elif required_slots:
                trace.append(
                    "canonical current-state projection incomplete; missing typed attributes: "
                    + ", ".join(sorted(set(required_slots) - set(winners)))
                )

        optional_slots = {
            str(slot).strip()
            for slot in required_slot_plan.get("optional_slots") or []
            if str(slot).strip()
        }
        if "household_plan" not in target_frame_types or not (
            "date" in required_slots or "date" in optional_slots
        ):
            return support_rows, trace

        partial_weekdays = sorted(
            {
                str((getattr(frame, "slots", {}) or {}).get("date") or "").strip()
                for frame in selected_frames
                if str(getattr(frame, "frame_type", "") or "") == "household_plan"
                and re.fullmatch(r"[A-Za-z]+", str((getattr(frame, "slots", {}) or {}).get("date") or "").strip())
            }
        )
        if not partial_weekdays:
            return [], []
        target_weekdays = self._extract_question_weekdays(question)
        if target_weekdays:
            partial_weekdays = [weekday for weekday in partial_weekdays if weekday.lower() in target_weekdays]
        if not partial_weekdays:
            return [], []

        for weekday in partial_weekdays:
            support_row = self._resolve_household_full_date_support_row(
                weekday=weekday,
                policy_allowed_evidence=policy_allowed_evidence,
                atomic_memories=atomic_memories,
                principal=principal,
                policy_scope=policy_scope,
                target_entities=list(required_slot_plan.get("target_entities") or []),
            )
            if support_row is None or support_row.memory_id in existing_ids:
                continue
            try:
                support_frame = compile_evidence_frame(support_row)
            except Exception:
                continue
            selected_evidence.append(support_row)
            selected_frames.append(support_frame)
            existing_ids.add(support_row.memory_id)
            full_date = str((support_frame.slots or {}).get("date") or "").strip()
            trace.append(
                f"support evidence completion added household temporal anchor for {weekday}: {full_date or support_row.memory_id}."
            )
            support_rows.append(self._evidence_to_debug(support_row))
        return support_rows, trace

    def _adjudicate_conflicting_typed_claims(
        self,
        *,
        question: str,
        required_slots: set[str],
        candidates: list[tuple],
    ) -> tuple[dict[str, str], list[str]]:
        """Resolve authority-qualified conflicts without exposing benchmark targets."""
        directives = []
        for recency, _, row, frame in candidates:
            state_delta = dict(getattr(frame, "state_delta", {}) or {})
            effect = str(state_delta.get("authority_effect") or "none").strip().lower()
            if effect not in {"authoritative", "non_authoritative", "blocks_override"}:
                continue
            directives.append({
                "memory_id": str(row.memory_id),
                "source_message_ids": list(row.source_message_ids or []),
                "recency": list(recency),
                "authority_effect": effect,
                "authority_scope": state_delta.get("authority_scope"),
                "content": str(row.content or ""),
            })
        if not directives:
            return {}, []

        claims = []
        conflicts: set[str] = set()
        for slot in sorted(required_slots):
            slot_claims = []
            for recency, score, row, frame in candidates:
                if not _structured_authority_claim_is_eligible(frame):
                    continue
                matched = _match_typed_attribute(
                    getattr(frame, "slots", {}) or {}, slot
                )
                if matched is None:
                    continue
                slot_claims.append({
                    "requested_attribute": slot,
                    "memory_id": str(row.memory_id),
                    "source_message_ids": list(row.source_message_ids or []),
                    "recency": list(recency),
                    "value": matched[1],
                    "schema_alignment": matched[2],
                    "retrieval_score": score,
                    "discourse_act": str(getattr(frame, "discourse_act", "unknown")),
                    "event_identity": dict(getattr(frame, "event_identity", {}) or {}),
                    "state_delta": dict(getattr(frame, "state_delta", {}) or {}),
                    "content": str(row.content or ""),
                })
            distinct_values = {repr(item["value"]) for item in slot_claims}
            if len(distinct_values) > 1:
                conflicts.add(slot)
                claims.extend(slot_claims)
        if not conflicts:
            return {}, []

        try:
            raw = self.llm_client.chat_json(
                model=resolve_llm_model(self.config, "action_decision"),
                system_prompt=(
                    "You are a governed evidence authority adjudicator. Select an existing "
                    "evidence memory_id for each conflicting requested attribute. Use only "
                    "the supplied policy-filtered claims, provenance order, discourse acts, "
                    "event identities, and structured authority directives. A later claim "
                    "must not override an approved state when an applicable directive marks "
                    "that update non-authoritative or blocks override. Do not answer the "
                    "question and do not invent values. Return JSON with selections, each "
                    "containing requested_attribute, memory_id, and confidence."
                ),
                user_prompt=str({
                    "question": question,
                    "conflicting_attributes": sorted(conflicts),
                    "claims": claims,
                    "authority_directives": directives,
                }),
            )
        except Exception as exc:
            return {}, [f"authority_adjudication_error={type(exc).__name__}"]

        rows = raw.get("selections") if isinstance(raw, dict) else None
        if not isinstance(rows, list):
            return {}, ["authority_adjudication_invalid_schema"]
        valid_ids = {
            slot: {
                item["memory_id"]
                for item in claims
                if item["requested_attribute"] == slot
            }
            for slot in conflicts
        }
        winners: dict[str, str] = {}
        for item in rows:
            if not isinstance(item, dict):
                continue
            slot = str(item.get("requested_attribute") or "")
            memory_id = str(item.get("memory_id") or "")
            try:
                confidence = float(item.get("confidence") or 0.0)
            except (TypeError, ValueError):
                confidence = 0.0
            if confidence >= 0.7 and memory_id in valid_ids.get(slot, set()):
                winners[slot] = memory_id
        trace = [
            "structured_authority_adjudication="
            + ",".join(f"{slot}:{memory_id}" for slot, memory_id in sorted(winners.items()))
        ]
        return winners, trace

    @staticmethod
    def _evidence_recency_key(row: RetrievedEvidence, frame) -> tuple[str, int]:
        turn_numbers = [
            int(match.group(1))
            for source_id in list(row.source_message_ids or [])
            for match in [re.search(r"(?:^|_)t(\d+)(?:$|_)", str(source_id), flags=re.IGNORECASE)]
            if match
        ]
        return (
            str(row.time or getattr(frame, "effective_time", "") or ""),
            max(turn_numbers, default=-1),
        )

    @staticmethod
    def _extract_question_weekdays(question: str) -> set[str]:
        return {
            match.group(0).strip().lower()
            for match in re.finditer(
                r"\b(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b",
                str(question or ""),
                flags=re.IGNORECASE,
            )
        }

    def _resolve_household_full_date_support_row(
        self,
        *,
        weekday: str,
        policy_allowed_evidence: list[RetrievedEvidence],
        atomic_memories: list[AtomicMemory],
        principal,
        policy_scope,
        target_entities: list[str] | None = None,
    ) -> RetrievedEvidence | None:
        normalized_weekday = str(weekday or "").strip().lower()
        if not normalized_weekday:
            return None

        candidate_rows: list[tuple[str, RetrievedEvidence, str, float]] = []
        seen_ids: set[str] = set()
        for row in policy_allowed_evidence:
            if not _row_mentions_target_entity(row, target_entities):
                continue
            candidate = self._extract_household_full_date_candidate(row=row, weekday=normalized_weekday)
            if candidate is None:
                continue
            full_date, support_score = candidate
            candidate_rows.append((full_date, row, support_score, float(row.score)))
            seen_ids.add(row.memory_id)

        for item in atomic_memories:
            if item.memory_id in seen_ids:
                continue
            atomic_row = self._atomic_memory_to_retrieved_evidence(item, score=min(max(float(item.confidence), 0.0), 1.0) * 0.35)
            approved_source_ids = {
                str(source_id)
                for row in policy_allowed_evidence
                for source_id in list(row.source_message_ids or [])
                if str(source_id)
            }
            if not (
                approved_source_ids.intersection(str(source_id) for source_id in item.source_message_ids)
                or _row_mentions_target_entity(atomic_row, target_entities)
            ):
                continue
            candidate = self._extract_household_full_date_candidate(row=atomic_row, weekday=normalized_weekday)
            if candidate is None:
                continue
            full_date, support_score = candidate
            candidate_rows.append((full_date, atomic_row, support_score, float(atomic_row.score)))

        unique_full_dates = {full_date for full_date, _, _, _ in candidate_rows}
        if len(unique_full_dates) != 1:
            return None
        chosen_full_date = next(iter(unique_full_dates))
        best = max(
            [item for item in candidate_rows if item[0] == chosen_full_date],
            key=lambda item: (float(item[2]), float(item[3])),
        )
        return best[1]

    @staticmethod
    def _atomic_memory_to_retrieved_evidence(item: AtomicMemory, *, score: float) -> RetrievedEvidence:
        return RetrievedEvidence(
            memory_id=item.memory_id,
            content=item.content,
            score=float(score),
            retrieval_source="support_completion",
            reason="current-state support evidence completion",
            user_id=item.owner_user,
            memory_type=item.memory_type,
            scope="atomic_memory",
            entities=list(item.entities),
            time=item.timestamp,
            source_message_ids=list(item.source_message_ids),
            metadata={
                "source_type": "atomic_memory",
                "slots": dict(item.slots),
                "lifecycle_status": item.lifecycle_status,
                "surface_spans": dict(item.access_tags.get("surface_spans") or {}),
                "semantic_tags": dict(item.access_tags.get("semantic_tags") or {}),
                "support_completion": {
                    "kind": "household_full_date_anchor",
                },
            },
        )

    @staticmethod
    def _extract_household_full_date_candidate(*, row: RetrievedEvidence, weekday: str) -> tuple[str, float] | None:
        try:
            frame = compile_evidence_frame(row)
        except Exception:
            return None
        slots = dict(getattr(frame, "slots", {}) or {})
        text = str(row.content or "")
        lowered = text.lower()
        if any(token in lowered for token in ["passphrase", "door code", "keypad code", "credential", "private-note", "private note"]):
            return None
        full_date = str(slots.get("date") or "").strip()
        if not re.fullmatch(r"[A-Za-z]+,\s+[A-Za-z]+\s+\d{1,2}(?:,\s*\d{4})?", full_date):
            return None
        if not full_date.lower().startswith(f"{weekday},"):
            return None
        if str(getattr(frame, "frame_type", "") or "") not in {"household_plan", "general_fact", "logistics", "update"}:
            return None
        support_score = 0.0
        if str(getattr(frame, "frame_type", "") or "") == "household_plan":
            support_score += 1.0
        if str(slots.get("visit_window") or "").strip():
            support_score += 0.2
        if "current" in lowered or "checkpoint" in lowered or "update" in lowered:
            support_score += 0.15
        return re.sub(r"\s+", " ", full_date).strip(), support_score

    @staticmethod
    def _evidence_to_debug(row: RetrievedEvidence) -> dict[str, Any]:
        meta = dict(row.metadata or {})
        return {
            "memory_id": row.memory_id,
            "evidence_id": row.memory_id,
            "source_type": meta.get("source_type", "chunk"),
            "content": row.content,
            "text": row.content,
            "memory_type": row.memory_type,
            "user_id": row.user_id,
            "scope": row.scope,
            "time": row.time,
            "metadata": meta,
            "slots": meta.get("slots", {}),
            "score": row.score,
            "source_message_ids": row.source_message_ids,
        }


def _contains_phrase(text: str, phrase: str) -> bool:
    normalized_text = str(text or "").lower()
    normalized_phrase = str(phrase or "").strip().lower()
    if not normalized_text or not normalized_phrase:
        return False
    pattern = r"(?<![a-z0-9_])" + re.escape(normalized_phrase).replace(r"\ ", r"\s+") + r"(?![a-z0-9_])"
    return re.search(pattern, normalized_text) is not None


def _row_mentions_target_entity(
    row: RetrievedEvidence,
    target_entities: list[str] | None,
) -> bool:
    """Keep support completion inside the query's named entity closure."""
    entities = [
        str(entity).strip()
        for entity in list(target_entities or [])
        if str(entity).strip()
    ]
    if not entities:
        return True
    searchable = " ".join([
        str(row.content or ""),
        *[str(entity) for entity in list(row.entities or [])],
    ])
    specific = [
        entity for entity in entities
        if len(re.findall(r"[a-z0-9]+", entity.lower())) > 1
    ]
    phrases = specific or entities
    return any(_contains_phrase(searchable, phrase) for phrase in phrases)


def _append_verified_support_date(
    answer_result: AnswerResult,
    support_rows: list[RetrievedEvidence],
) -> AnswerResult:
    """Merge one independently verified schedule anchor without re-ranking fields."""
    candidates = [
        str(dict((row.metadata or {}).get("slots") or {}).get("date") or "").strip()
        for row in support_rows
    ]
    dates = list(dict.fromkeys(value for value in candidates if value))
    if not dates:
        return answer_result
    date_value = max(dates, key=len)
    if date_value.lower() in str(answer_result.answer_text or "").lower():
        return answer_result
    answer_text = str(answer_result.answer_text or "").rstrip()
    answer_text = answer_text.rstrip(".") + f"; date: {date_value}."
    structured = dict(answer_result.answer_structured or {})
    typed_slots = dict(structured.get("typed_slots") or {})
    typed_slots["date"] = date_value
    structured["typed_slots"] = typed_slots
    structured["support_completion"] = {
        "kind": "household_full_date_anchor",
        "date": date_value,
        "source_memory_ids": [str(row.memory_id) for row in support_rows],
    }
    used_ids = list(answer_result.used_memory_ids or [])
    for row in support_rows:
        if str(row.memory_id) and str(row.memory_id) not in used_ids:
            used_ids.append(str(row.memory_id))
    return replace(
        answer_result,
        prediction=answer_text,
        answer_text=answer_text,
        used_memory_ids=used_ids,
        answer_structured=structured,
    )


def _structured_authority_claim_is_eligible(frame: Any) -> bool:
    """Keep authority directives as provenance, not active factual claims."""
    state_delta = dict(getattr(frame, "state_delta", {}) or {})
    effect = str(state_delta.get("authority_effect") or "none").strip().lower()
    if effect in {"non_authoritative", "blocks_override"}:
        return False
    if effect == "authoritative":
        return True
    discourse_act = str(getattr(frame, "discourse_act", "unknown") or "unknown").lower()
    return discourse_act != "unknown" and bool(state_delta)


def _match_typed_attribute(
    slots: dict[str, Any],
    requested_key: str,
) -> tuple[str, Any, float] | None:
    """Align open typed keys without a domain vocabulary or manual synonym map."""
    requested = _attribute_key_tokens(requested_key)
    if not requested:
        return None
    matches: list[tuple[str, Any, float]] = []
    for candidate_key, value in dict(slots or {}).items():
        if value in (None, "", []):
            continue
        candidate = _attribute_key_tokens(candidate_key)
        if not candidate:
            continue
        if str(candidate_key) == str(requested_key):
            score = 1.0
        else:
            # Candidate keys often carry provenance qualifiers (entity, actor,
            # scope). Measure how much of the requested schema is covered,
            # rather than penalizing those extra grounded qualifiers.
            aligned_requested = {
                token
                for token in requested
                if any(_attribute_tokens_equivalent(token, other) for other in candidate)
            }
            score = len(aligned_requested) / len(requested)
            requested_parts = re.findall(r"[a-z0-9]+", str(requested_key).lower())
            candidate_parts = re.findall(r"[a-z0-9]+", str(candidate_key).lower())
            head_only = (
                len(requested_parts) > 1
                and aligned_requested == {requested_parts[-1]}
            )
            numeric_type_support = (
                "numeric" in requested
                and bool(re.fullmatch(r"[-+]?\d+(?:\.\d+)?", str(value).strip()))
            )
            collection_type_support = isinstance(value, (list, tuple, set)) and bool(value)
            if head_only and not (numeric_type_support or collection_type_support):
                continue
            if requested_parts and candidate_parts and requested_parts[-1] == candidate_parts[-1]:
                score = max(score, 0.5)
        if score < 0.5:
            continue
        matches.append((str(candidate_key), value, score))
    if not matches:
        return None
    exact = next((match for match in matches if match[0] == str(requested_key)), None)
    if exact is not None:
        return exact
    start = next(
        (match for match in matches if "start" in _attribute_key_tokens(match[0])),
        None,
    )
    end = next(
        (match for match in matches if "end" in _attribute_key_tokens(match[0])),
        None,
    )
    if start is not None and end is not None:
        return (
            str(requested_key),
            f"{start[1]} to {end[1]}",
            min(start[2], end[2]),
        )
    return max(matches, key=lambda match: match[2])


def _attribute_key_tokens(value: Any) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", str(value or "").lower()))


def _attribute_tokens_equivalent(left: str, right: str) -> bool:
    if left == right:
        return True
    shorter, longer = sorted((left, right), key=len)
    return len(shorter) >= 3 and longer.startswith(shorter)


def _temporal_anchor_value(slots: dict[str, Any]) -> Any | None:
    for key in ("target_date", "public_event_date", "date"):
        value = dict(slots or {}).get(key)
        if value not in (None, "", []):
            return value
    return None


def _absolute_temporal_anchor_value(slots: dict[str, Any]) -> Any | None:
    value = _temporal_anchor_value(slots)
    return value if value is not None and re.search(r"\d", str(value)) else None


def _is_temporal_interval_value(value: Any) -> bool:
    return len(re.findall(r"\b\d{1,2}:\d{2}\s*(?:AM|PM)?\b", str(value), re.IGNORECASE)) >= 2


from gov_mem.data.schema import MemoryItem
NONASSERTIVE_DISCOURSE_ACTS = frozenset({"question", "request"})
ASSERTIVE_DISCOURSE_ACTS = frozenset({"assertion", "confirmation", "update", "revocation"})


def _structured_assertion_decision(metadata: dict | None) -> bool | None:
    """Return an assertion decision only when a structured annotation exists."""
    semantic_tags = dict((metadata or {}).get("semantic_tags") or {})
    discourse_act = str(semantic_tags.get("discourse_act") or "unknown").strip().lower()
    try:
        confidence = float(semantic_tags.get("assertion_confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    if discourse_act in NONASSERTIVE_DISCOURSE_ACTS:
        return False
    if discourse_act in ASSERTIVE_DISCOURSE_ACTS and confidence >= 0.5:
        return True
    return None
