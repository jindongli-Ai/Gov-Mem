from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from gov_mem.evolution.dev_guard import has_valid_embedded_dev_attestation
from gov_mem.utils.io import read_jsonl


@dataclass
class RuntimeExperienceContext:
    domain: str
    requester_role: str
    owner_relation: str
    question: str
    required_slots: list[str] = field(default_factory=list)
    sensitive_slots: list[str] = field(default_factory=list)
    lifecycle_flags: list[str] = field(default_factory=list)
    graph_lifecycle_flags: list[str] = field(default_factory=list)
    evidence_coverage: float = 0.0
    certificate_authorized: bool = False
    certificate_reason: str = ""
    certified_slots: list[str] = field(default_factory=list)


@dataclass
class RetrievedExperienceLesson:
    experience_id: str
    pattern_id: str
    reusable_skill_hypothesis: str
    failure_type: str
    lesson: str
    confidence: float
    selection_score: float
    selection_reasons: list[str] = field(default_factory=list)


class RuntimeExperienceRetriever:
    def __init__(self, experience_memory_path: str | Path | None) -> None:
        self.experience_memory_path = Path(experience_memory_path) if experience_memory_path else None
        self._cache: list[dict[str, Any]] | None = None

    def retrieve(
        self,
        *,
        context: RuntimeExperienceContext,
        top_k: int = 2,
    ) -> list[RetrievedExperienceLesson]:
        rows = self._load_rows()
        if not rows:
            return []

        scored: list[RetrievedExperienceLesson] = []
        for row in rows:
            score, reasons = _score_experience(row=row, context=context)
            if score < 1.2:
                continue
            scored.append(
                RetrievedExperienceLesson(
                    experience_id=str(row.get("exp_id") or ""),
                    pattern_id=str(row.get("pattern_id") or ""),
                    reusable_skill_hypothesis=str(row.get("reusable_skill_hypothesis") or ""),
                    failure_type=str(row.get("failure_type") or "unknown"),
                    lesson=str(row.get("correction_strategy") or "").strip(),
                    confidence=float(row.get("confidence") or 0.0),
                    selection_score=round(score, 4),
                    selection_reasons=reasons,
                )
            )

        scored.sort(key=lambda item: (item.selection_score, item.confidence), reverse=True)
        return scored[:top_k]

    def _load_rows(self) -> list[dict[str, Any]]:
        if self.experience_memory_path is None or not self.experience_memory_path.exists():
            return []
        if self._cache is None:
            # Runtime must not consume scorer-derived artifacts unless their
            # producing workflow explicitly attested to development-only data.
            self._cache = [
                dict(row)
                for row in read_jsonl(self.experience_memory_path)
                if has_valid_embedded_dev_attestation(dict(row))
            ]
        return self._cache


def runtime_experience_context_to_dict(context: RuntimeExperienceContext) -> dict[str, Any]:
    return asdict(context)


def retrieved_experience_lesson_to_dict(item: RetrievedExperienceLesson) -> dict[str, Any]:
    return asdict(item)


def _score_experience(*, row: dict[str, Any], context: RuntimeExperienceContext) -> tuple[float, list[str]]:
    score = 0.0
    reasons: list[str] = []
    failure_type = str(row.get("failure_type") or "").strip().lower()
    applicable_roles = {str(item).strip().lower() for item in row.get("applicable_roles") or [] if str(item).strip()}
    applicable_slots = {str(item).strip().lower() for item in row.get("applicable_slots") or [] if str(item).strip()}
    requester_role = context.requester_role.strip().lower()
    owner_relation = context.owner_relation.strip().lower()
    required_slots = {str(item).strip().lower() for item in context.required_slots if str(item).strip()}
    sensitive_slots = {str(item).strip().lower() for item in context.sensitive_slots if str(item).strip()}
    lifecycle_flags = {str(item).strip().lower() for item in context.lifecycle_flags if str(item).strip()}
    lifecycle_flags.update(str(item).strip().lower() for item in context.graph_lifecycle_flags if str(item).strip())
    lowered_question = context.question.lower()

    # Domain identity is intentionally excluded: reusable experience is
    # selected from governance state, not benchmark taxonomy.
    score += 0.25
    reasons.append("domain_invariant_experience")

    if applicable_roles and ({requester_role, owner_relation} & applicable_roles):
        score += 0.8
        reasons.append(f"role_match:{sorted(({requester_role, owner_relation} & applicable_roles))}")

    slot_overlap = (required_slots | sensitive_slots) & applicable_slots
    if slot_overlap:
        score += min(1.5, 0.8 + 0.3 * len(slot_overlap))
        reasons.append(f"slot_overlap:{sorted(slot_overlap)}")

    if failure_type == "deleted_reconstruction":
        if lifecycle_flags & {"deleted", "deletes", "forget", "forgetting", "remove", "removed"}:
            score += 1.0
            reasons.append("deleted_lifecycle_cue")
        if any(token in lowered_question for token in ["deleted", "removed", "forgot", "forget"]):
            score += 0.5
            reasons.append("deleted_question_cue")
    elif failure_type == "stale_state":
        if lifecycle_flags & {"supersedes", "superseded", "current", "latest", "updated"}:
            score += 1.0
            reasons.append("state_refresh_cue")
        if any(token in lowered_question for token in ["current", "latest", "updated", "now", "right now"]):
            score += 0.5
            reasons.append("current_state_question")
    elif failure_type in {"leakage", "under_redaction"}:
        if sensitive_slots and owner_relation not in {"owner", "self", "patient", "clinician", "authorized_staff"}:
            score += 1.0
            reasons.append("sensitive_non_owner_request")
        if "contact" in required_slots or "medication" in required_slots or "condition" in required_slots:
            score += 0.4
            reasons.append("sensitive_slot_requested")
    elif failure_type in {"missing_utility", "no_memory_collapse"}:
        if len(required_slots) >= 2:
            score += 0.9
            reasons.append("multi_slot_utility_request")
        if any(token in lowered_question for token in ["which", "what", "list", "details", "current"]):
            score += 0.3
            reasons.append("explicit_detail_request")
    elif failure_type == "over_refusal":
        if owner_relation in {"owner", "self", "patient", "clinician", "authorized_staff"}:
            score += 0.9
            reasons.append("authorized_requester")
    else:
        if required_slots:
            score += 0.25
            reasons.append("generic_runtime_alignment")

    if failure_type in {"missing_utility", "no_memory_collapse"}:
        if context.certificate_authorized and context.certified_slots:
            score += 1.0
            reasons.append("authorized_certificate_with_renderable_slots")
        if context.evidence_coverage < 1.0:
            score += 0.4
            reasons.append("incomplete_typed_slot_coverage")
    if failure_type in {"leakage", "under_redaction", "over_refusal"}:
        if context.certificate_reason:
            score += 0.5
            reasons.append(f"certificate_state:{context.certificate_reason}")

    return score, reasons
