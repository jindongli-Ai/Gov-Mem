from __future__ import annotations

from dataclasses import asdict
from datetime import date
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
from gov_mem.data.schema import AnswerResult, GovernedActionDecision, MemoryInstance, RetrievedEvidence
from gov_mem.governance_runtime.access import build_principal, infer_owner_user_id, normalize_text
from gov_mem.governance_runtime.action_predictor import GovernedActionPredictor
from gov_mem.llm.client import LLMClient
from gov_mem.llm.model_registry import resolve_llm_model
from gov_mem.query_semantics import classify_state_slot_families, extract_state_slots, infer_state_record_type


LOGISTICS_TOKENS = {"appointment", "arrive", "arrival", "location", "suite", "clinic", "visit", "follow-up", "follow up", "schedule", "time", "parking", "dr.", "ultrasound", "scan", "imaging"}
CLINICAL_TOKENS = {
    "diagnosis", "result", "results", "symptom", "symptoms", "medication", "medications", "allergy", "pregnancy",
    "reaction", "lab numbers", "clinical interpretation", "assessment", "bleeding", "spotting", "cramping",
    "miscarriage", "yolk sac", "viable", "viability", "hcg", "beta-hcg", "beta hcg", "slower-rising",
    "slower rising", "impression", "atrial fibrillation", "afib", "cardioversion", "blood thinner",
    "recovery house", "cedar bridge", "hepatitis", "syphilis", "hiv",
}
POLICY_TOKENS = {"authorized", "permission", "privacy", "consent", "must not", "may receive", "logistics only", "do not share", "forbidden", "revoked", "nothing else", "time/location logistics"}
DELETION_TOKENS = {"deleted", "forget", "remove", "no longer remember", "old backup", "backup number", "deleted contact", "temporary number", "safe number"}
NONCLINICAL_STATE_TOKENS = {
    "current",
    "approved amount",
    "grant amount",
    "current amount",
    "tuition hold",
    "hold state",
    "review memo",
    "memo closure",
    "orientation date",
    "public date",
    "safe wording",
    "broad wording",
    "visit window",
    "arrival window",
    "entry method",
    "approved entrance",
    "approved areas",
    "approved rooms",
    "parking pass",
    "visitor pass",
}
NONCLINICAL_AUTHORIZED_STAFF = {
    "financial_aid",
    "advisor",
    "registrar",
    "professor",
    "department_admin",
    "campus_it",
    "resident",
    "primary_resident",
    "product_manager",
    "counselor",
    "ta",
    "ra",
    "security",
    "legal",
    "sre",
    "engineer",
    "employee",
}
DATE_RE = re.compile(r"\b(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday),?\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}\b", re.IGNORECASE)
TIME_RE = re.compile(r"\b\d{1,2}:\d{2}\s?(?:AM|PM)\b", re.IGNORECASE)
LOCATION_RE = re.compile(r"\b(?:(?!(?:AM|PM)\b)[A-Z][A-Za-z0-9&' -]{1,60}\s+(?:Suite|Clinic|Center|Office|Lab|Ward|Desk)\s*[A-Z0-9-]*|front desk)\b")
MONTHS = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}


def _is_policy_revocation_or_permission_change(text: str) -> bool:
    lowered = normalize_text(text)
    return any(
        token in lowered
        for token in {
            "removed from scheduling-contact",
            "removed from callback-contact",
            "family scheduling access revoked",
            "do not share future appointment details",
            "no longer release appointment times",
            "callback-contact permissions",
            "scheduling-contact permissions",
            "permission revoked",
        }
    )


def _is_operational_instruction_line(text: str) -> bool:
    lowered = normalize_text(text)
    return any(
        token in lowered
        for token in {
            "arrive by",
            "arrival",
            "check in by",
            "nothing by mouth",
            "no routine beta-hcg",
            "before tuesday",
            "unless symptoms worsen",
            "generic callback only",
            "no voicemail mentioning pregnancy",
            "do not mention pregnancy in voicemail",
            "portal only",
            "temporary callback",
            "safe phone changes after sunday",
            "starting sunday evening",
            "bring photo id",
        }
    )


class RAGPolicyBackbone:
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

    def run_instance(self, instance: MemoryInstance) -> BackboneRunResult:
        chunks = build_rag_chunks(instance, self.config)
        save_rag_chunks(self.output_dir, self.dataset_name, instance.instance_id, chunks)
        plan, retrieval_result, evidence = retrieve_rag_chunks(
            instance=instance,
            chunks=chunks,
            llm_client=self.embedding_client,
            embedding_model=str(self.config["embedding"]["model"]),
            config=self.config,
            planning_client=self.llm_client,
            planning_model=resolve_llm_model(self.config, "query_planning"),
        )
        principal = build_principal(
            requester_id=instance.asking_user_id,
            requester_role=((instance.metadata.get("requester") or {}).get("role")),
            owner_user_id=self._owner_user_id(instance),
        )
        decisions = [classify_chunk_policy(row, principal) for row in evidence]
        policy_scope = infer_policy_scope(evidence, principal)
        projected_evidence = []
        filtered_ids = []
        for row, decision in zip(evidence, decisions):
            if decision["allowed_for_requester"]:
                projected = project_evidence_for_principal(row, principal, decision, policy_scope)
                if projected is not None:
                    projected_evidence.append(projected)
            else:
                filtered_ids.append(row.memory_id)
        action_predictor = GovernedActionPredictor(
            llm_client=self.llm_client,
            model_name=resolve_llm_model(self.config, "action_decision"),
        )
        action_decision = action_predictor.decide(
            instance=instance,
            plan=plan,
            evidence=evidence,
            projected_evidence=projected_evidence,
        )
        action = action_decision.action
        if action in {"refuse", "no_memory"}:
            answer_result = build_action_only_answer_result(
                action=action,
                reasoning_summary="Policy/action guard prevented answer generation from using protected evidence.",
                used_memory_ids=[row.memory_id for row in projected_evidence],
            )
        else:
            extra_rules = []
            if principal.relation_to_owner == "family":
                extra_rules.append("Share logistics only unless broader authorization is explicitly observable.")
            answer_result = answer_with_retrieved_evidence(
                instance=instance,
                evidence=projected_evidence,
                llm_client=self.llm_client,
                model_name=resolve_llm_model(self.config, "answering"),
                action=action,
                extra_rules=extra_rules,
                used_chunk_ids=[row.memory_id for row in projected_evidence],
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
        reasoning_state = build_reasoning_state(
            projected_evidence,
            trace=[
                f"rag_policy kept {len(projected_evidence)} / {len(evidence)} chunks after policy filtering.",
            ],
        )
        answer_result.action = action
        action_decision.evidence_memory_ids = [row.memory_id for row in projected_evidence]
        debug_payload = {
            "experiment_mode": "rag_policy",
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
            "atomic_memories": [],
            "retrieved_atomic_memories": [],
            "policy_decisions": decisions,
            "selected_evidence": [
                {
                    "evidence_id": row.memory_id,
                    "source_type": "chunk",
                    "text": row.content,
                    "score": row.score,
                    "projection_reason": row.metadata.get("projection_reason"),
                    "projected_line_count": row.metadata.get("projected_line_count"),
                }
                for row in projected_evidence
            ],
            "current_state": {},
            "slot_coverage": {},
            "action_correction_trace": [],
            "policy_filtered_chunk_ids": filtered_ids,
            "question_profile": {},
        }
        retrieval_result["policy_decisions"] = decisions
        retrieval_result["retrieved_after_privacy_filter"] = projected_evidence
        retrieval_result["filtered_evidence"] = [{"memory_id": chunk_id, "reason": "policy_filter"} for chunk_id in filtered_ids]
        return BackboneRunResult(
            query_plan=plan,
            retrieval_result=retrieval_result,
            reasoning_state=reasoning_state,
            action_decision=action_decision,
            answer_result=answer_result,
            debug_payload=debug_payload,
        )

    @staticmethod
    def _owner_user_id(instance: MemoryInstance) -> str | None:
        return infer_owner_user_id(
            messages=list(instance.messages),
            requester_id=instance.asking_user_id,
        )


def classify_chunk_policy(row: RetrievedEvidence, principal) -> dict[str, Any]:
    lowered = normalize_text(row.content)
    meta_slots = dict((row.metadata or {}).get("slots") or {})
    inferred_state_record_type = infer_state_record_type(
        text=str(row.content or ""),
        slots=meta_slots,
        frame_type=str(getattr(row, "memory_type", "") or ""),
    )
    state_families = classify_state_slot_families(text=str(row.content or ""), slots=meta_slots)
    household_or_nonclinical_state = inferred_state_record_type in {"household_plan", "project_state", "research_state"} or bool(
        state_families & {"date", "visit_window", "entry_method", "package_rule", "approved_areas", "parking_pass", "arrival_contact_rule"}
    )
    contains_policy = any(token in lowered for token in POLICY_TOKENS)
    contains_schedule = any(token in lowered for token in LOGISTICS_TOKENS)
    contains_sensitive_clinical = any(token in lowered for token in CLINICAL_TOKENS)
    if household_or_nonclinical_state:
        contains_sensitive_clinical = False
    contains_nonclinical_state = any(token in lowered for token in NONCLINICAL_STATE_TOKENS)
    if household_or_nonclinical_state:
        contains_nonclinical_state = True
        contains_schedule = contains_schedule or bool(
            state_families & {"date", "visit_window", "entry_method", "approved_areas", "parking_pass", "arrival_contact_rule"}
        )
    safe_scope_summary = household_or_nonclinical_state and _is_scope_summary_text(str(row.content or ""))
    contains_deleted_or_forbidden = _has_deletion_or_forgetting_signal(row.content)
    contains_private_timing_note = _looks_like_private_timing_note(row.content)
    if _is_policy_revocation_or_permission_change(row.content):
        contains_deleted_or_forbidden = False
    if safe_scope_summary:
        contains_deleted_or_forbidden = False
    contains_utility = contains_schedule or contains_sensitive_clinical or contains_nonclinical_state
    allowed_slot_groups = []
    denied_slot_groups = []
    allowed = not contains_deleted_or_forbidden
    policy_reason = "source-grounded relation pending"
    if principal.relation_to_owner not in {"owner", "family", "delegate", "authorized_staff"}:
        return {
            "chunk_id": row.memory_id,
            "contains_policy": contains_policy,
            "contains_utility": contains_utility,
            "contains_schedule": contains_schedule,
            "contains_sensitive_clinical": contains_sensitive_clinical,
            "contains_deleted_or_forbidden": contains_deleted_or_forbidden,
            "allowed_for_requester": False,
            "allowed_slot_groups": [],
            "denied_slot_groups": ["all"],
            "policy_reason": "requester_owner_relation_unproven",
        }
    if principal.relation_to_owner == "owner":
        allowed_slot_groups = ["logistics", "clinical", "nonclinical_state"]
    elif principal.relation_to_owner == "authorized_staff" and principal.organization_role in {"scheduler", "front_desk"}:
        allowed_slot_groups = ["logistics"]
        denied_slot_groups = ["clinical"] if contains_sensitive_clinical else []
        if contains_sensitive_clinical and not _is_operational_instruction_line(row.content):
            policy_reason = "logistics staff restricted from clinical detail"
            if not contains_schedule:
                allowed = False
    elif principal.relation_to_owner == "authorized_staff" and principal.organization_role in NONCLINICAL_AUTHORIZED_STAFF:
        allowed_slot_groups = ["logistics", "nonclinical_state"]
        denied_slot_groups = ["clinical"] if contains_sensitive_clinical else []
        if contains_sensitive_clinical and not contains_schedule:
            allowed = False
            policy_reason = "nonclinical authorized staff denied clinical detail"
        else:
            policy_reason = "authorized nonclinical staff access"
    elif principal.relation_to_owner == "authorized_staff" and principal.organization_role in {"pharmacist"}:
        allowed_slot_groups = ["logistics", "clinical"]
        policy_reason = "pharmacist clinical medication access"
    elif principal.relation_to_owner == "authorized_staff" and principal.organization_role in {"billing", "social_worker", "labtech", "staff"}:
        allowed_slot_groups = ["logistics"] if contains_schedule and not contains_sensitive_clinical else []
        denied_slot_groups = ["clinical"] if contains_sensitive_clinical else []
        if contains_sensitive_clinical or not contains_schedule:
            allowed = False
            policy_reason = "non-care-team staff denied clinical or identity detail"
    elif principal.relation_to_owner == "authorized_staff" and principal.organization_role in {"clinician", "nurse"}:
        allowed_slot_groups = ["logistics", "clinical"]
        policy_reason = "clinical requester requires downstream authorization check"
    elif principal.relation_to_owner == "family":
        allowed_slot_groups = ["logistics", "nonclinical_state"] if contains_schedule or contains_policy or _looks_like_shareable_nonclinical_state(row.content) else []
        denied_slot_groups = ["clinical"] if contains_sensitive_clinical else []
        if contains_sensitive_clinical and not contains_schedule:
            allowed = False
            policy_reason = "family logistics-only restriction"
        elif contains_schedule or _looks_like_shareable_nonclinical_state(row.content):
            policy_reason = "family logistics-or-broad-state access"
    elif principal.organization_role == "delegate_assistant" or principal.relation_to_owner == "delegate":
        allowed_slot_groups = ["logistics", "nonclinical_state"] if contains_schedule or contains_policy or _looks_like_shareable_nonclinical_state(row.content) else []
        denied_slot_groups = ["clinical"] if contains_sensitive_clinical else []
        if not allowed_slot_groups or (contains_sensitive_clinical and not contains_schedule):
            allowed = False
            policy_reason = "delegate restricted to public or broadly safe state"
        else:
            policy_reason = "delegate public-or-broad-state access"
    else:
        allowed_slot_groups = ["logistics"] if contains_schedule else []
        if not allowed_slot_groups:
            allowed = False
            policy_reason = "unknown requester denied"
    if principal.relation_to_owner != "owner" and contains_private_timing_note:
        allowed = False
        policy_reason = "private timing note denied"
    return {
        "chunk_id": row.memory_id,
        "contains_policy": contains_policy,
        "contains_utility": contains_utility,
        "contains_schedule": contains_schedule,
        "contains_sensitive_clinical": contains_sensitive_clinical,
        "contains_deleted_or_forbidden": contains_deleted_or_forbidden,
        "allowed_for_requester": allowed,
        "allowed_slot_groups": allowed_slot_groups,
        "denied_slot_groups": denied_slot_groups,
        "policy_reason": policy_reason,
    }


def project_evidence_for_principal(row: RetrievedEvidence, principal, decision: dict[str, Any], policy_scope: dict[str, Any] | None = None) -> RetrievedEvidence | None:
    if principal.relation_to_owner not in {"family", "delegate"} and principal.organization_role not in {"scheduler", "front_desk", "delegate_assistant"}:
        return row
    kept_lines: list[str] = []
    for raw_line in str(row.content or "").splitlines():
        line_text = raw_line.strip()
        if not line_text:
            continue
        line_decision = classify_line_policy(line_text, principal, policy_scope or {})
        if line_decision["allowed"]:
            kept_lines.append(line_decision.get("sanitized_line") or raw_line)
    if not kept_lines:
        return None
    projected = RetrievedEvidence(
        memory_id=row.memory_id,
        content="\n".join(kept_lines),
        score=row.score,
        retrieval_source=row.retrieval_source,
        reason=row.reason,
        user_id=row.user_id,
        memory_type=row.memory_type,
        scope=row.scope,
        entities=list(row.entities),
        time=row.time,
        source_message_ids=list(row.source_message_ids),
        metadata=dict(row.metadata or {}),
    )
    projected.metadata["projection_reason"] = decision.get("policy_reason", "policy_projection")
    projected.metadata["projected_line_count"] = len(kept_lines)
    projected.metadata["projected_for_role"] = principal.relation_to_owner or principal.organization_role
    return projected


def classify_line_policy(line_text: str, principal, policy_scope: dict[str, Any] | None = None) -> dict[str, Any]:
    lowered = normalize_text(line_text)
    state_slots = {key: value for key, value in dict(extract_state_slots(line_text)).items() if value}
    inferred_state_record_type = infer_state_record_type(text=line_text, slots=state_slots, frame_type=None)
    state_families = classify_state_slot_families(text=line_text, slots=state_slots)
    household_or_nonclinical_state = inferred_state_record_type in {"household_plan", "project_state", "research_state"} or bool(
        state_families & {"date", "visit_window", "entry_method", "package_rule", "approved_areas", "parking_pass", "arrival_contact_rule"}
    )
    contains_policy = any(token in lowered for token in POLICY_TOKENS)
    contains_schedule = any(token in lowered for token in LOGISTICS_TOKENS)
    contains_sensitive_clinical = any(token in lowered for token in CLINICAL_TOKENS)
    if household_or_nonclinical_state:
        contains_sensitive_clinical = False
    contains_deleted_or_forbidden = any(token in lowered for token in DELETION_TOKENS)
    contains_private_timing_note = _looks_like_private_timing_note(line_text)
    safe_scope_summary = household_or_nonclinical_state and _is_scope_summary_text(line_text)
    if _is_policy_revocation_or_permission_change(line_text):
        contains_deleted_or_forbidden = False
    contains_concrete_logistics = _line_has_concrete_logistics(line_text)
    contains_operational_instruction = _is_operational_instruction_line(line_text)
    allowed = not contains_deleted_or_forbidden
    reason = "allowed"
    if principal.relation_to_owner == "family":
        if contains_policy and _is_policy_revocation_or_permission_change(line_text):
            allowed = True
            reason = "family policy-state line"
        elif contains_policy and not contains_concrete_logistics:
            allowed = False
            reason = "policy lines reserved for access reasoning"
        elif contains_sensitive_clinical and not contains_concrete_logistics:
            allowed = False
            reason = "family clinical line denied"
        elif contains_concrete_logistics:
            allowed = True
            reason = "family logistics line"
            if not _line_within_policy_scope(line_text, policy_scope or {}):
                allowed = False
                reason = "outside family policy time scope"
        elif _line_has_shareable_nonclinical_state(line_text):
            allowed = True
            reason = "family broad-state line"
        else:
            allowed = False
            reason = "family non-logistics line denied"
    elif principal.organization_role in {"scheduler", "front_desk"}:
        if contains_concrete_logistics:
            allowed = True
            reason = "logistics staff line"
        elif contains_operational_instruction:
            allowed = True
            reason = "operational instruction line"
        elif contains_sensitive_clinical:
            allowed = False
            reason = "logistics staff clinical line denied"
    elif principal.organization_role == "delegate_assistant" or principal.relation_to_owner == "delegate":
        if contains_concrete_logistics:
            allowed = True
            reason = "delegate logistics line"
        elif _line_has_shareable_nonclinical_state(line_text):
            allowed = True
            reason = "delegate broad-state line"
        elif contains_sensitive_clinical:
            allowed = False
            reason = "delegate clinical line denied"
    elif principal.relation_to_owner == "owner" and safe_scope_summary:
        allowed = True
        reason = "owner safe helper-scope summary"
    if principal.relation_to_owner != "owner" and contains_private_timing_note:
        allowed = False
        reason = "private timing note denied"
    sanitized_line = line_text
    if allowed and contains_concrete_logistics and contains_sensitive_clinical:
        sanitized_line = _sanitize_logistics_line(line_text)
    return {
        "allowed": allowed,
        "contains_schedule": contains_schedule,
        "contains_sensitive_clinical": contains_sensitive_clinical,
        "contains_policy": contains_policy,
        "reason": reason,
        "sanitized_line": sanitized_line,
    }


def build_action_only_answer_result(*, action: str, reasoning_summary: str, used_memory_ids: list[str]) -> AnswerResult:
    if action == "refuse":
        text = "I cannot share that information because the requester is not authorized to access it."
    else:
        text = "I do not have memory of that."
    return AnswerResult(
        prediction=text,
        answer_text=text,
        used_memory_ids=used_memory_ids,
        reasoning_summary=reasoning_summary,
        action=action,
        raw_response={"policy_guard": True},
    )
def infer_policy_scope(evidence: list[RetrievedEvidence], principal) -> dict[str, Any]:
    if principal.relation_to_owner != "family":
        return {}
    cutoff = None
    for row in evidence:
        for raw_line in str(row.content or "").splitlines():
            lowered = normalize_text(raw_line)
            if "through friday" not in lowered and "through march" not in lowered:
                continue
            parsed = _extract_first_date(raw_line)
            if parsed is not None:
                cutoff = parsed
                break
        if cutoff is not None:
            break
    return {"family_cutoff_date": cutoff.isoformat() if cutoff else None}


def _line_has_concrete_logistics(line_text: str) -> bool:
    has_time = bool(TIME_RE.search(line_text))
    has_location = bool(LOCATION_RE.search(line_text))
    has_date = bool(DATE_RE.search(line_text))
    state_slots = {key: value for key, value in dict(extract_state_slots(line_text)).items() if value}
    household_state_logistics = any(
        state_slots.get(key)
        for key in ["visit_window", "entry_method", "package_rule", "approved_areas", "parking_pass", "arrival_contact_rule"]
    )
    return bool(has_time or has_location or household_state_logistics or (has_date and (has_time or has_location or household_state_logistics)))


def _is_scope_summary_text(text: str) -> bool:
    lowered = normalize_text(text)
    boundary_cues = ("outside scope", "out of scope", "may not", "must not", "do not share", "restricted", "private", "resident-only")
    role_cues = ("helper", "delegate", "visitor", "guest", "staff", "contractor", "requester")
    return any(cue in lowered for cue in boundary_cues) and any(cue in lowered for cue in role_cues)


def _looks_like_shareable_nonclinical_state(text: str) -> bool:
    lowered = normalize_text(text)
    if any(token in lowered for token in ["exact award amount", "room code", "door code", "active code", "pin", "passphrase", "credential rotation"]):
        return False
    inferred_state_record_type = infer_state_record_type(text=text, slots={}, frame_type=None)
    if inferred_state_record_type in {"household_plan", "project_state", "research_state"}:
        return True
    broad_state_tokens = {
        "public",
        "broad wording",
        "safe wording",
        "calendar line",
        "orientation",
        "public event",
        "room reassignment",
    }
    return any(token in lowered for token in broad_state_tokens)


def _line_has_shareable_nonclinical_state(line_text: str) -> bool:
    lowered = normalize_text(line_text)
    state_slots = {key: value for key, value in dict(extract_state_slots(line_text)).items() if value}
    if any(state_slots.get(key) for key in ["visit_window", "date", "approved_areas", "arrival_contact_rule", "monthly_stipend", "blocker"]):
        return True
    if any(
        token in lowered
        for token in [
            "support amount",
            "approved amount",
            "current blocker",
            "active blocker",
            "tuition hold",
            "hold remains",
            "arrival-contact rule",
            "contact leo on arrival",
            "contact on arrival",
            "approved areas",
            "what i may do",
            "resident-only",
            "local-setup scope",
            "setup window",
        ]
    ):
        return True
    return _looks_like_shareable_nonclinical_state(line_text)


def _looks_like_private_timing_note(text: str) -> bool:
    lowered = normalize_text(text)
    private_scope_tokens = [
        "private note",
        "private-note",
        "private callback",
        "private counseling",
        "keep the exact timing private",
        "keep the exact time",
        "exact timing private",
        "remains private",
        "should remain private",
        "should not appear",
        "should not reveal",
        "should not be used to infer",
        "separate private-note thread",
        "upstairs study",
        "upstairs desk nook",
    ]
    if not any(token in lowered for token in private_scope_tokens):
        return False
    return bool(TIME_RE.search(text) or DATE_RE.search(text))


def _has_deletion_or_forgetting_signal(text: str) -> bool:
    lowered = normalize_text(text)
    if any(token in lowered for token in DELETION_TOKENS):
        return True
    if "cleared" in lowered and any(token in lowered for token in ["memory", "contact", "number", "backup", "deleted", "removed"]):
        return True
    return False


def _sanitize_logistics_line(line_text: str) -> str:
    sanitized = re.sub(r"\b(?:because|given|due to)\b.*", "", line_text, flags=re.IGNORECASE).strip(" .;,")
    sanitized = re.sub(
        r"\b(?:is|being)\s+(?:worked up|evaluated|seen|treated)\b.*",
        "",
        sanitized,
        flags=re.IGNORECASE,
    ).strip(" .;,")
    sanitized = re.sub(
        r"\b(?:for|about)\b\s+(?:miscarriage|routine ob|spotting|cramping|bleeding|pregnancy|results?)\b.*",
        "",
        sanitized,
        flags=re.IGNORECASE,
    ).strip(" .;,")
    if _line_has_concrete_logistics(sanitized):
        return sanitized
    fallback = _extract_logistics_surface(line_text)
    return fallback or sanitized or line_text


def _extract_logistics_surface(line_text: str) -> str:
    text = str(line_text or "").strip()
    matches: list[str] = []
    for pattern in [
        r"(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday),?\s+(?:March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}[^.;]*?(?:arrive by\s+\d{1,2}:\d{2}\s?(?:AM|PM))?[^.;]*?(?:Suite|suite|Radiology|radiology|clinic|Clinic)[^.;]*",
        r"(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)[^.;]*?\d{1,2}:\d{2}\s?(?:AM|PM)[^.;]*?(?:Suite|suite|Radiology|radiology|clinic|Clinic)[^.;]*",
        r"(?:arrive by\s+\d{1,2}:\d{2}\s?(?:AM|PM)[^.;]*?(?:Suite|suite|Radiology|radiology|clinic|Clinic)[^.;]*)",
        r"(?:[A-Z][A-Za-z0-9&' -]{1,60}\s+(?:Suite|Clinic|Center|Office|Lab|Ward|Desk)\s*[A-Z0-9-]*|front desk)[^.;]*",
    ]:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            candidate = match.group(0).strip(" .;,")
            if candidate:
                matches.append(candidate)
    if not matches:
        return ""
    return " ".join(dict.fromkeys(matches))


def _line_within_policy_scope(line_text: str, policy_scope: dict[str, Any]) -> bool:
    cutoff_raw = policy_scope.get("family_cutoff_date")
    if not cutoff_raw:
        return True
    line_date = _extract_first_date(line_text)
    if line_date is None:
        return True
    cutoff = date.fromisoformat(str(cutoff_raw))
    return line_date <= cutoff


def _extract_first_date(text: str) -> date | None:
    match = DATE_RE.search(text)
    if not match:
        return None
    value = match.group(0).replace(",", "")
    parts = value.split()
    if len(parts) < 3:
        return None
    month = MONTHS.get(parts[-2].lower())
    if month is None:
        return None
    try:
        day = int(parts[-1])
    except ValueError:
        return None
    return date(2026, month, day)
