from __future__ import annotations

from dataclasses import dataclass, field
import re

from gov_mem.backbones.coverage_packer import PackedUtilityEvidence
from gov_mem.backbones.coverage_packer import _required_units
from gov_mem.backbones.need_spec import AnswerNeedSpec
from gov_mem.backbones.utility_records import canonicalize_medication_surface, extract_medication_clauses


@dataclass
class CoverageVerification:
    required_units: list[str]
    rendered_units: list[str]
    missing_units: list[str]
    extra_units: list[str]
    denied_surface_leaks: list[str]
    pass_coverage: bool
    pass_minimality: bool
    pass_access_safety: bool
    trace: list[str] = field(default_factory=list)


def verify_canonical_answer(
    answer: str,
    packed: PackedUtilityEvidence,
    answer_need: AnswerNeedSpec,
    principal,
    config: dict,
) -> CoverageVerification:
    lower_answer = str(answer or "").lower()
    normalized_med_answer = canonicalize_medication_surface(answer).lower()
    required_units = sorted(_required_units(answer_need))
    rendered_units: list[str] = []
    denied_surface_leaks: list[str] = []
    for record in packed.selected_records:
        if record.record_type == "medication_status":
            rendered_units.extend(_rendered_medication_units(record, normalized_med_answer, answer_need))
            for slot in record.denied_slots:
                surface = record.surface_spans.get(slot) or record.slots.get(slot) or ""
                if surface and str(surface).lower() in lower_answer:
                    denied_surface_leaks.append(surface)
            continue
        for slot in record.allowed_slots:
            unit = f"{record.record_type}.{slot}"
            surface = record.surface_spans.get(slot) or record.slots.get(slot) or ""
            if unit in required_units and surface and _surface_is_rendered(unit=unit, surface=str(surface), lower_answer=lower_answer):
                rendered_units.append(unit)
        for slot in record.denied_slots:
            surface = record.surface_spans.get(slot) or record.slots.get(slot) or ""
            if surface and str(surface).lower() in lower_answer:
                denied_surface_leaks.append(surface)
    rendered_units = sorted(set(rendered_units))
    missing_units = sorted(set(required_units) - set(rendered_units))
    extra_units = sorted(set(rendered_units) - set(required_units))
    pass_coverage = not missing_units
    pass_access_safety = not denied_surface_leaks
    pass_minimality = True
    if answer_need.minimality_policy == "answer_only_requested_domains":
        pass_minimality = len(extra_units) <= max(2, len(required_units) // 2)
    return CoverageVerification(
        required_units=list(required_units),
        rendered_units=rendered_units,
        missing_units=missing_units,
        extra_units=extra_units,
        denied_surface_leaks=denied_surface_leaks,
        pass_coverage=pass_coverage,
        pass_minimality=pass_minimality,
        pass_access_safety=pass_access_safety,
        trace=[
            f"required_units={required_units}",
            f"rendered_units={rendered_units}",
            f"missing_units={missing_units}",
            f"extra_units={extra_units}",
            f"denied_surface_leaks={denied_surface_leaks}",
        ],
    )


def _surface_is_rendered(*, unit: str, surface: str, lower_answer: str) -> bool:
    normalized_surface = str(surface or "").lower().strip()
    if not normalized_surface:
        return False
    if normalized_surface in lower_answer:
        return True
    if unit == "household_plan.package_rule":
        informative_tokens = [token for token in re.findall(r"[a-z0-9]+", normalized_surface) if len(token) >= 4]
        if informative_tokens and sum(token in lower_answer for token in informative_tokens) >= min(2, len(informative_tokens)):
            return True
    return False


def _rendered_medication_units(record, normalized_answer: str, answer_need: AnswerNeedSpec) -> list[str]:
    rendered: list[str] = []
    required_units = _required_units(answer_need)
    clause_hits = [
        clause for clause in extract_medication_clauses(record.evidence_line)
        if canonicalize_medication_surface(clause).lower() in normalized_answer
    ]
    normalized_clause_hits = {
        canonicalize_medication_surface(clause).lower()
        for clause in clause_hits
        if canonicalize_medication_surface(clause).lower()
    }
    medication_surface = canonicalize_medication_surface(record.slots.get("medication") or record.slots.get("medication_name") or "").lower()
    dosage_surface = canonicalize_medication_surface(record.slots.get("dosage") or "").lower()
    timing_surface = canonicalize_medication_surface(record.slots.get("timing") or "").lower()
    condition_surface = canonicalize_medication_surface(record.slots.get("condition") or "").lower()
    medication_hit = bool(
        medication_surface
        and (
            medication_surface in normalized_answer
            or any(
                medication_surface in clause
                or clause in medication_surface
                or any(token and token in clause for token in medication_surface.split())
                for clause in normalized_clause_hits
            )
        )
    )
    if medication_hit and f"{record.record_type}.medication" in required_units:
        rendered.append(f"{record.record_type}.medication")
    if dosage_surface and f"{record.record_type}.dosage" in required_units and dosage_surface in normalized_answer:
        rendered.append(f"{record.record_type}.dosage")
    if clause_hits and f"{record.record_type}.instruction" in required_units:
        rendered.append(f"{record.record_type}.instruction")
    if timing_surface and f"{record.record_type}.timing" in required_units and timing_surface in normalized_answer:
        rendered.append(f"{record.record_type}.timing")
    if condition_surface and f"{record.record_type}.condition" in required_units and condition_surface in normalized_answer:
        rendered.append(f"{record.record_type}.condition")
    if medication_surface and f"{record.record_type}.medication_name" in required_units and medication_surface in normalized_answer:
        rendered.append(f"{record.record_type}.medication_name")
    return rendered
