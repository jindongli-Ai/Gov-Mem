from __future__ import annotations

from dataclasses import asdict, dataclass, field
import re
from typing import Any

from gov_mem.data.schema import EvidenceFrame, RetrievedEvidence
from gov_mem.governance_runtime.evidence_frames import compile_evidence_frame


@dataclass
class SlotAuthorizationResult:
    authorized_evidence: list[RetrievedEvidence]
    allowed_slots_used: list[str]
    denied_slots_checked: list[str]
    blocked_facts_checked: list[str]
    denied_surface_values: dict[str, list[str]] = field(default_factory=dict)
    blocked_surface_values: dict[str, list[str]] = field(default_factory=dict)
    authorization_trace: list[str] = field(default_factory=list)
    row_decisions: list[dict[str, Any]] = field(default_factory=list)


class SlotAuthorizer:
    def authorize(
        self,
        *,
        evidence: list[RetrievedEvidence],
        selected_frames: list[EvidenceFrame],
        symbolic_decision: dict[str, Any] | None,
        final_action: str,
        fallback_used: bool,
        enforce_allowed_slot_projection: bool = True,
    ) -> SlotAuthorizationResult:
        decision = dict(symbolic_decision or {})
        allowed_slots = {str(item) for item in decision.get("allowed_slots") or [] if str(item).strip()}
        explicit_denied_slots = {
            str(item)
            for item in decision.get("denied_slots") or []
            if str(item).strip()
        }
        redacted_slots = {
            str(item)
            for item in decision.get("redacted_slots") or []
            if str(item).strip()
        }
        denied_slots = explicit_denied_slots | (redacted_slots - allowed_slots)
        blocked_facts = [str(item) for item in decision.get("blocked_facts") or [] if str(item).strip()]
        trace = [
            f"final_action={final_action}",
            f"fallback_used={fallback_used}",
            f"allowed_slots={sorted(allowed_slots)}",
            f"denied_slots={sorted(denied_slots)}",
            f"blocked_facts={blocked_facts}",
        ]
        if final_action in {"refuse", "no_memory"}:
            trace.append("restrictive_action_no_utility_realization")
            return SlotAuthorizationResult(
                authorized_evidence=[],
                allowed_slots_used=[],
                denied_slots_checked=sorted(denied_slots),
                blocked_facts_checked=blocked_facts,
                authorization_trace=trace,
            )

        frames_by_memory_id: dict[str, list[EvidenceFrame]] = {}
        for frame in selected_frames:
            frames_by_memory_id.setdefault(str(frame.memory_id), []).append(frame)

        authorized_evidence: list[RetrievedEvidence] = []
        allowed_slots_used: set[str] = set()
        denied_surface_values: dict[str, list[str]] = {}
        blocked_surface_values: dict[str, list[str]] = {}
        row_decisions: list[dict[str, Any]] = []

        for row in evidence:
            frames = frames_by_memory_id.get(str(row.memory_id)) or [compile_evidence_frame(row)]
            row_slots = self._collect_row_slots(frames=frames, row=row)
            slot_names = set(row_slots.keys())
            denied_on_row = sorted(slot_names & denied_slots)
            allowed_on_row = sorted(slot_names & allowed_slots)
            lifecycle_states = {str(frame.lifecycle_status or "").lower() for frame in frames}
            memory_status = str((row.metadata or {}).get("memory_status") or "").lower()
            is_blocked = bool({"deleted", "superseded"} & lifecycle_states) or memory_status in {"deleted", "superseded"}

            if denied_on_row:
                for slot_name in denied_on_row:
                    denied_surface_values[slot_name] = row_slots.get(slot_name) or []
            if is_blocked:
                for slot_name, values in row_slots.items():
                    if values:
                        blocked_surface_values.setdefault(slot_name, [])
                        blocked_surface_values[slot_name].extend(values)

            keep_original = fallback_used or not decision
            keep_row = False
            sanitized_row = row
            decision_reason = "dropped"

            if is_blocked:
                decision_reason = "blocked_deleted_or_superseded"
            elif keep_original:
                keep_row = True
                decision_reason = "fallback_keep_original"
            elif final_action == "answer" and not denied_slots and not blocked_facts:
                if allowed_slots and enforce_allowed_slot_projection:
                    sanitized_text = self._render_allowed_surface_text(
                        allowed_slots=allowed_on_row,
                        row_slots=row_slots,
                    )
                    if allowed_on_row and sanitized_text:
                        keep_row = True
                        decision_reason = "full_answer_slot_projection"
                        sanitized_row = self._clone_row(
                            row=row,
                            content=sanitized_text,
                            allowed_slots=allowed_on_row,
                            denied_slots=sorted(slot_names - allowed_slots),
                        )
                        allowed_slots_used.update(allowed_on_row)
                    else:
                        decision_reason = "explicit_contract_without_authorized_slot"
                else:
                    keep_row = True
                    decision_reason = (
                        "record_bundle_keep_policy_filtered"
                        if allowed_slots else "full_answer_keep_original"
                    )
            elif final_action == "answer_redacted" and not allowed_slots:
                decision_reason = "redacted_action_without_explicit_safe_slot"
            elif allowed_slots:
                if allowed_on_row:
                    sanitized_text = self._render_allowed_surface_text(
                        allowed_slots=allowed_on_row,
                        row_slots=row_slots,
                    )
                    if sanitized_text:
                        keep_row = True
                        decision_reason = "slot_authorized_projection"
                        sanitized_row = self._clone_row(
                            row=row,
                            content=sanitized_text,
                            allowed_slots=allowed_on_row,
                            denied_slots=denied_on_row,
                        )
                        allowed_slots_used.update(allowed_on_row)
                    else:
                        decision_reason = "allowed_slots_without_renderable_surface"
                else:
                    decision_reason = "no_allowed_slots_on_row"
            elif not denied_on_row:
                keep_row = True
                decision_reason = "no_explicit_slot_constraints"

            if keep_row:
                authorized_evidence.append(sanitized_row)
            row_decisions.append(
                {
                    "memory_id": row.memory_id,
                    "slot_names": sorted(slot_names),
                    "allowed_on_row": allowed_on_row,
                    "denied_on_row": denied_on_row,
                    "lifecycle_states": sorted(state for state in lifecycle_states if state),
                    "memory_status": memory_status or None,
                    "retrieval_source": row.retrieval_source,
                    "kept": keep_row,
                    "decision_reason": decision_reason,
                }
            )

        if final_action == "answer_redacted" and not fallback_used and allowed_slots:
            atomic_rows = [row for row in authorized_evidence if row.retrieval_source == "atomic_memory"]
            if atomic_rows:
                row_debug_by_id = {str(row_debug["memory_id"]): row_debug for row_debug in row_decisions}
                covered_slots: set[str] = set()
                retained_rows: list[RetrievedEvidence] = []
                for row in authorized_evidence:
                    row_debug = row_debug_by_id.get(str(row.memory_id)) or {}
                    row_allowed = {str(slot) for slot in row_debug.get("allowed_on_row") or [] if str(slot).strip()}
                    if row.retrieval_source == "atomic_memory":
                        retained_rows.append(row)
                        covered_slots.update(row_allowed)
                        continue
                    if row_allowed - covered_slots:
                        retained_rows.append(row)
                        covered_slots.update(row_allowed)
                atomic_ids = {row.memory_id for row in retained_rows if row.retrieval_source == "atomic_memory"}
                retained_ids = {row.memory_id for row in retained_rows}
                authorized_evidence = retained_rows
                for row_debug in row_decisions:
                    if row_debug["memory_id"] in retained_ids or not row_debug["kept"]:
                        continue
                    if row_debug["memory_id"] not in atomic_ids:
                        row_debug["kept"] = False
                        row_debug["decision_reason"] = "dropped_in_favor_of_atomic_memory_projection"
                trace.append("authorized_evidence_prefers_atomic_memory_when_slot_coverage_preserved")

        trace.append(f"authorized_rows={len(authorized_evidence)}/{len(evidence)}")
        if allowed_slots_used:
            trace.append(f"allowed_slots_used={sorted(allowed_slots_used)}")
        return SlotAuthorizationResult(
            authorized_evidence=authorized_evidence,
            allowed_slots_used=sorted(allowed_slots_used),
            denied_slots_checked=sorted(denied_slots),
            blocked_facts_checked=blocked_facts,
            denied_surface_values=_dedupe_surface_map(denied_surface_values),
            blocked_surface_values=_dedupe_surface_map(blocked_surface_values),
            authorization_trace=trace,
            row_decisions=row_decisions,
        )

    @staticmethod
    def _collect_row_slots(*, frames: list[EvidenceFrame], row: RetrievedEvidence) -> dict[str, list[str]]:
        row_slots: dict[str, list[str]] = {}
        for frame in frames:
            for slot_name, slot_value in dict(frame.slots or {}).items():
                values = row_slots.setdefault(str(slot_name), [])
                surface = str((frame.surface_spans or {}).get(slot_name) or "").strip()
                if surface:
                    values.append(surface)
                elif _has_slot_value(slot_value):
                    values.append(str(slot_value))
        for slot_name, slot_value in dict((row.metadata or {}).get("slots") or {}).items():
            values = row_slots.setdefault(str(slot_name), [])
            if _has_slot_value(slot_value):
                values.append(str(slot_value))
        normalized = _dedupe_surface_map(row_slots)
        original_content = str((row.metadata or {}).get("original_content") or row.content or "")
        for slot_name, values in list(normalized.items()):
            normalized[slot_name] = [
                _normalize_slot_surface_value(
                    slot_name=slot_name,
                    value=value,
                    row_time=row.time,
                    original_content=original_content,
                )
                for value in values
            ]
        return _dedupe_surface_map(normalized)

    @staticmethod
    def _render_allowed_surface_text(*, allowed_slots: list[str], row_slots: dict[str, list[str]]) -> str:
        ordered_values: list[str] = []
        slot_labels = {
            "target_date": "target date",
            "public_event_date": "public event date",
            "approved_budget": "approved budget",
            "approved_discount_cap": "approved maximum discount",
            "monthly_stipend": "monthly stipend",
            "safe_wording": "safe wording",
            "blocker": "blocker",
            "date": "date",
            "visit_window": "visit window",
            "entry_method": "entry method",
            "package_rule": "package rule",
            "approved_areas": "approved areas",
            "parking_pass": "parking pass",
            "arrival_contact_rule": "arrival contact rule",
            "instruction": "instruction",
            "timing": "timing",
            "condition": "condition",
            "medication": "medication",
        }
        for slot_name in allowed_slots:
            for value in row_slots.get(slot_name) or []:
                cleaned = value.strip()
                labeled = f"{slot_labels.get(slot_name, slot_name.replace('_', ' '))}: {cleaned}" if cleaned else ""
                if labeled and labeled not in ordered_values:
                    ordered_values.append(labeled)
        return "; ".join(ordered_values).strip()

    @staticmethod
    def _clone_row(
        *,
        row: RetrievedEvidence,
        content: str,
        allowed_slots: list[str],
        denied_slots: list[str],
    ) -> RetrievedEvidence:
        metadata = dict(row.metadata or {})
        metadata["authorized_slots"] = list(allowed_slots)
        metadata["denied_slots_removed"] = list(denied_slots)
        metadata["authorization_mode"] = "slot_projected"
        metadata["original_content"] = row.content
        return RetrievedEvidence(
            memory_id=row.memory_id,
            content=content,
            score=row.score,
            retrieval_source=row.retrieval_source,
            reason=row.reason,
            user_id=row.user_id,
            memory_type=row.memory_type,
            scope=row.scope,
            entities=list(row.entities),
            time=row.time,
            source_message_ids=list(row.source_message_ids),
            metadata=metadata,
        )


def slot_authorization_to_dict(result: SlotAuthorizationResult) -> dict[str, Any]:
    payload = asdict(result)
    payload["authorized_evidence"] = [
        {
            "memory_id": row.memory_id,
            "content": row.content,
            "metadata": dict(row.metadata or {}),
            "score": row.score,
        }
        for row in result.authorized_evidence
    ]
    return payload


def _has_slot_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, set, dict)):
        return bool(value)
    return True


def _dedupe_surface_map(surface_map: dict[str, list[str]]) -> dict[str, list[str]]:
    return {
        key: list(dict.fromkeys(value.strip() for value in values if str(value).strip()))
        for key, values in surface_map.items()
    }


def _normalize_slot_surface_value(
    *,
    slot_name: str,
    value: str,
    row_time: str | None,
    original_content: str,
) -> str:
    cleaned = str(value or "").strip()
    if not cleaned:
        return cleaned
    if slot_name not in {"target_date", "date", "public_event_date"}:
        return cleaned
    if re.search(r"\b\d{4}\b", cleaned):
        return cleaned
    full_from_source = _recover_full_date_from_source(partial_date=cleaned, original_content=original_content)
    if full_from_source:
        return full_from_source
    inferred = _attach_year_from_row_time(partial_date=cleaned, row_time=row_time)
    if inferred:
        return inferred
    return cleaned


def _recover_full_date_from_source(*, partial_date: str, original_content: str) -> str | None:
    if not partial_date or not original_content:
        return None
    pattern = re.escape(partial_date)
    match = re.search(rf"\b{pattern},\s*\d{{4}}\b", original_content)
    if match:
        return match.group(0).strip()
    return None


def _attach_year_from_row_time(*, partial_date: str, row_time: str | None) -> str | None:
    if not partial_date or not row_time:
        return None
    if not re.fullmatch(r"[A-Za-z]+\s+\d{1,2}", partial_date.strip()):
        return None
    year_match = re.match(r"(\d{4})-", str(row_time).strip())
    if not year_match:
        return None
    return f"{partial_date}, {year_match.group(1)}"
