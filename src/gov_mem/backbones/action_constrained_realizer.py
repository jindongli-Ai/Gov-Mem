from __future__ import annotations

from dataclasses import asdict, dataclass, field
import re
from typing import Any, Callable

from gov_mem.backbones.action_arbitrator import (
    ActionArbitrationResult,
    ActionArbitrator,
    action_arbitration_to_dict,
)
from gov_mem.backbones.rag_policy import build_action_only_answer_result
from gov_mem.backbones.slot_authorization import (
    SlotAuthorizationResult,
    SlotAuthorizer,
    slot_authorization_to_dict,
)
from gov_mem.data.schema import AnswerResult, EvidenceFrame, MemoryInstance, RetrievedEvidence
from gov_mem.governance_runtime.evidence_frames import compile_evidence_frame


@dataclass
class RealizationVerifierResult:
    passed: bool
    allowed_slots_used: list[str]
    denied_slots_checked: list[str]
    blocked_facts_checked: list[str]
    denied_surface_hits: dict[str, list[str]] = field(default_factory=dict)
    blocked_surface_hits: dict[str, list[str]] = field(default_factory=dict)
    verifier_trace: list[str] = field(default_factory=list)


@dataclass
class ActionConstrainedRealization:
    final_action: str
    answer_result: AnswerResult
    arbitration_result: ActionArbitrationResult
    authorization_result: SlotAuthorizationResult
    verifier_result: RealizationVerifierResult
    debug_payload: dict[str, Any]
    reasoning_trace: list[str]


class ActionConstrainedRealizer:
    def __init__(self) -> None:
        self.arbitrator = ActionArbitrator()
        self.slot_authorizer = SlotAuthorizer()

    def realize(
        self,
        *,
        instance: MemoryInstance,
        selected_evidence: list[RetrievedEvidence],
        selected_frames: list[EvidenceFrame],
        llm_action_prediction: str,
        symbolic_runtime: dict[str, Any] | None,
        fallback_to_v0_on_low_confidence: bool,
        use_slot_authorization_map: bool,
        runtime_skill_bundle: dict[str, Any] | None,
        typed_slot_contract: dict[str, Any] | None,
        renderer: Callable[[str, list[RetrievedEvidence]], AnswerResult],
    ) -> ActionConstrainedRealization:
        symbolic_decision = dict((symbolic_runtime or {}).get("decision") or {})
        arbitration = self.arbitrator.arbitrate(
            llm_action_prediction=llm_action_prediction,
            symbolic_decision=symbolic_decision or None,
            fallback_to_v0_on_low_confidence=fallback_to_v0_on_low_confidence,
            runtime_skill_bundle=runtime_skill_bundle,
        )
        if use_slot_authorization_map:
            authorization = self.slot_authorizer.authorize(
                evidence=selected_evidence,
                selected_frames=selected_frames,
                symbolic_decision=symbolic_decision or None,
                final_action=arbitration.final_action,
                fallback_used=arbitration.fallback_used,
                enforce_allowed_slot_projection=bool(
                    (typed_slot_contract or {}).get("requested_slots")
                ),
            )
        else:
            authorization = SlotAuthorizationResult(
                authorized_evidence=list(selected_evidence),
                allowed_slots_used=[],
                denied_slots_checked=list(symbolic_decision.get("denied_slots") or []),
                blocked_facts_checked=list(symbolic_decision.get("blocked_facts") or []),
                authorization_trace=[
                    f"final_action={arbitration.final_action}",
                    f"fallback_used={arbitration.fallback_used}",
                    "slot_authorization_map_bypassed",
                ],
                row_decisions=[
                    {
                        "memory_id": row.memory_id,
                        "kept": True,
                        "decision_reason": "authorization_bypass_keep_original",
                        "retrieval_source": row.retrieval_source,
                    }
                    for row in selected_evidence
                ],
            )

        final_action = arbitration.final_action
        reasoning_trace = list(arbitration.arbitration_trace) + list(authorization.authorization_trace)
        if final_action in {"refuse", "no_memory"}:
            answer_result = build_action_only_answer_result(
                action=final_action,
                reasoning_summary="Action-constrained realization applied governance denial before answer generation.",
                used_memory_ids=[row.memory_id for row in authorization.authorized_evidence],
            )
        elif not authorization.authorized_evidence:
            if final_action == "answer_redacted":
                reasoning_trace.append("policy_backed_redaction_without_private_evidence")
            else:
                final_action = "no_memory"
                reasoning_trace.append("authorization_produced_no_renderable_evidence")
            answer_result = build_action_only_answer_result(
                action=final_action,
                reasoning_summary=(
                    "The requested detail is outside the authorized disclosure scope."
                    if final_action == "answer_redacted"
                    else "Action-constrained realization found no renderable authorized evidence."
                ),
                used_memory_ids=[],
            )
        else:
            answer_result = None
            if final_action == "answer" and not list(
                (typed_slot_contract or {}).get("requested_slots") or []
            ):
                answer_result = _render_authorized_record_bundle(
                    evidence=authorization.authorized_evidence,
                    question=instance.question,
                )
                if answer_result is not None:
                    reasoning_trace.append("authorized_record_bundle_rendered_from_recap")
            if answer_result is None:
                answer_result = _render_authorized_current_slots(
                    evidence=authorization.authorized_evidence,
                    contract=typed_slot_contract or {},
                )
            if answer_result is None:
                answer_result = renderer(final_action, authorization.authorized_evidence)
            else:
                reasoning_trace.append("authorized_current_slots_rendered_canonically")
            answer_result.action = final_action

        verifier = self._verify_answer(
            answer_result=answer_result,
            authorization_result=authorization,
            runtime_skill_bundle=runtime_skill_bundle,
        )
        if not verifier.passed and final_action not in {"refuse", "no_memory"}:
            repaired_action = "no_memory" if verifier.blocked_surface_hits else "refuse"
            reasoning_trace.append(f"post_realization_verifier_failed->{repaired_action}")
            answer_result = build_action_only_answer_result(
                action=repaired_action,
                reasoning_summary="Post-realization governance verifier blocked an unsafe answer.",
                used_memory_ids=[],
            )
            final_action = repaired_action
            verifier = self._verify_answer(
                answer_result=answer_result,
                authorization_result=authorization,
                runtime_skill_bundle=runtime_skill_bundle,
            )

        debug_payload = {
            "final_action": final_action,
            "symbolic_action_constraint": arbitration.symbolic_action_constraint,
            "llm_action_prediction": llm_action_prediction,
            "action_arbitration_result": action_arbitration_to_dict(arbitration),
            "slot_authorization": slot_authorization_to_dict(authorization),
            "post_verifier": asdict(verifier),
            "fallback_used": arbitration.fallback_used,
            "final_answer": answer_result.answer_text,
        }
        return ActionConstrainedRealization(
            final_action=final_action,
            answer_result=answer_result,
            arbitration_result=arbitration,
            authorization_result=authorization,
            verifier_result=verifier,
            debug_payload=debug_payload,
            reasoning_trace=reasoning_trace,
        )

    @staticmethod
    def _verify_answer(
        *,
        answer_result: AnswerResult,
        authorization_result: SlotAuthorizationResult,
        runtime_skill_bundle: dict[str, Any] | None,
    ) -> RealizationVerifierResult:
        answer_text = str(answer_result.answer_text or "").lower()
        denied_surface_hits = _scan_surface_hits(
            answer_text=answer_text,
            surface_map=authorization_result.denied_surface_values,
        )
        blocked_surface_hits = _scan_surface_hits(
            answer_text=answer_text,
            surface_map=authorization_result.blocked_surface_values,
        )
        trace = [
            f"action={answer_result.action}",
            f"used_memory_ids={answer_result.used_memory_ids}",
            f"denied_hits={sorted(denied_surface_hits)}",
            f"blocked_hits={sorted(blocked_surface_hits)}",
        ]
        verifier_patches = list((runtime_skill_bundle or {}).get("verifier_patches") or [])
        if verifier_patches:
            trace.append(f"verifier_patches={verifier_patches}")
        if answer_result.action in {"refuse", "no_memory"}:
            if "verify_canonical_restrictive_action_phrase" in verifier_patches:
                canonical_texts = {
                    "refuse": "i cannot share that information because the requester is not authorized to access it.",
                    "no_memory": "i do not have memory of that.",
                }
                expected = canonical_texts.get(answer_result.action)
                if expected and answer_text != expected:
                    return RealizationVerifierResult(
                        passed=False,
                        allowed_slots_used=authorization_result.allowed_slots_used,
                        denied_slots_checked=authorization_result.denied_slots_checked,
                        blocked_facts_checked=authorization_result.blocked_facts_checked,
                        verifier_trace=trace + ["restrictive_action_phrase_mismatch"],
                    )
            return RealizationVerifierResult(
                passed=True,
                allowed_slots_used=authorization_result.allowed_slots_used,
                denied_slots_checked=authorization_result.denied_slots_checked,
                blocked_facts_checked=authorization_result.blocked_facts_checked,
                verifier_trace=trace + ["restrictive_action_auto_pass"],
            )
        passed = not denied_surface_hits and not blocked_surface_hits
        if passed:
            trace.append("authorized_surface_subset_check_passed")
        else:
            trace.append("authorized_surface_subset_check_failed")
        return RealizationVerifierResult(
            passed=passed,
            allowed_slots_used=authorization_result.allowed_slots_used,
            denied_slots_checked=authorization_result.denied_slots_checked,
            blocked_facts_checked=authorization_result.blocked_facts_checked,
            denied_surface_hits=denied_surface_hits,
            blocked_surface_hits=blocked_surface_hits,
            verifier_trace=trace,
        )


def _scan_surface_hits(*, answer_text: str, surface_map: dict[str, list[str]]) -> dict[str, list[str]]:
    hits: dict[str, list[str]] = {}
    for slot_name, surfaces in surface_map.items():
        for surface in surfaces:
            normalized = str(surface or "").strip().lower()
            if len(normalized) < 3:
                continue
            if normalized in answer_text:
                hits.setdefault(slot_name, []).append(surface)
    return hits


def _render_authorized_record_bundle(
    *, evidence: list[RetrievedEvidence], question: str
) -> AnswerResult | None:
    if re.search(
        r"\b(?:including|include|covering|two|three|four|five|\d+|"
        r"plans?|routes?|triggers?|steps?|instructions?)\b",
        str(question or "").lower(),
    ):
        coverage_result = _render_coverage_aware_record_bundle(
            evidence=evidence, question=question
        )
        if coverage_result is not None:
            return coverage_result
    query_terms = {
        token for token in re.findall(r"[a-z0-9]+", str(question or "").lower())
        if len(token) > 2 and token not in {"what", "which", "the", "are", "for", "now", "right", "current"}
    }
    candidates: list[tuple[float, int, float, float, RetrievedEvidence, str]] = []
    for row in evidence:
        content = str(row.content or "").strip()
        if not content:
            continue
        lowered = content.lower()
        marker_score = 0.0
        if re.search(r"\b(?:current|active|updated|final)\b.{0,40}\b(?:recap|summary|plan|schedule)\b", lowered):
            marker_score += 4.0
        if re.search(r"\b(?:recap|summary|say (?:that )?back exactly)\b", lowered):
            marker_score += 3.0
        if re.search(
            r"\b(?:booked|scheduled|remaining|current|active|updated)\s+"
            r"(?:now|items?|routes?|plan|schedule)?\s*:",
            lowered,
        ):
            marker_score += 4.0
        if marker_score <= 0.0:
            continue
        content_terms = set(re.findall(r"[a-z0-9]+", lowered))
        overlap = len(query_terms & content_terms) / max(len(query_terms), 1)
        turn_numbers = [
            int(match.group(1))
            for value in [row.memory_id, *list(row.source_message_ids or [])]
            for match in re.finditer(r"(?:^|_)(?:t|msg_|event_)?(\d{1,6})(?:_|$)", str(value), re.IGNORECASE)
        ]
        recency_turn = max(turn_numbers, default=0)
        cleaned = re.sub(r"(?:^|\n)\[[^\]]+\]\s+\[[^\]]+\]\s*", " ", content).strip()
        # Prefer a complete authorized record over a fragment derived from the
        # same source turn. The bounded bonus cannot outrank relevance or time.
        completeness_bonus = min(len(content_terms), 50) / 10000.0
        candidates.append((marker_score, recency_turn, overlap, completeness_bonus, row, cleaned))
    if not candidates:
        return _render_coverage_aware_record_bundle(evidence=evidence, question=question)
    temporal_current = bool(re.search(r"\b(?:current|latest|now|remain|after)\b", str(question or "").lower()))
    if temporal_current:
        _, _, _, _, row, text = max(
            candidates, key=lambda item: (item[0], item[1], item[2], item[3])
        )
    else:
        _, _, _, _, row, text = max(
            candidates,
            key=lambda item: (item[0] + item[2] + min(item[1], 100000) / 100000.0 + item[3]),
        )
    return AnswerResult(
        prediction=text,
        answer_text=text,
        used_memory_ids=[row.memory_id],
        reasoning_summary="Surface-preserving realization from an authorized explicit recap record.",
        action="answer",
        answer_structured={"realization_mode": "authorized_record_bundle_recap"},
    )


def _render_coverage_aware_record_bundle(
    *, evidence: list[RetrievedEvidence], question: str
) -> AnswerResult | None:
    """Compose complementary authorized sentences for explicit list requests."""
    lowered_question = str(question or "").lower()
    list_match = re.search(r"\b(?:including|include|covering)\b(.+?)[?.]?$", lowered_question)
    counted_match = re.search(
        r"\b(?:what|which|list|give|show|tell)\b.*?\b(?:two|three|four|five|\d+)\b\s+(.+?)[?.]?$",
        lowered_question,
    )
    collection_match = re.search(
        r"\b(?:plans?|routes?|triggers?|steps?|instructions?)\b",
        lowered_question,
    )
    if not list_match and not counted_match and not collection_match:
        return None
    demand_text = (
        (list_match or counted_match).group(1)
        if (list_match or counted_match)
        else lowered_question
    )
    demand_phrases = [
        part.strip(" ,")
        for part in re.split(r",|\band\b", demand_text)
        if part.strip(" ,")
    ]
    if len(demand_phrases) < 2 and counted_match is None and collection_match is None:
        return None

    stop = {
        "the", "a", "an", "current", "now", "and", "or", "including",
        "should", "watch", "for", "remain", "what", "which",
    }
    demands = [
        _semantic_terms(phrase, stop=stop)
        for phrase in demand_phrases
    ]
    sentence_rows: list[tuple[str, RetrievedEvidence, set[str]]] = []
    seen_sentences: set[str] = set()
    for row in evidence:
        cleaned = re.sub(r"(?:^|\n)\[[^\]]+\]\s+\[[^\]]+\]\s*", "\n", str(row.content or ""))
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", cleaned):
            sentence = sentence.strip()
            normalized = re.sub(r"\s+", " ", sentence.lower()).strip(" .")
            if not normalized or normalized in seen_sentences:
                continue
            seen_sentences.add(normalized)
            terms = _semantic_terms(normalized)
            sentence_rows.append((sentence, row, terms))

    selected: list[tuple[str, RetrievedEvidence]] = []
    selected_text: set[str] = set()
    temporal_current = bool(re.search(
        r"\b(?:current|latest|now|remain|after)\b", lowered_question
    ))
    for demand in demands:
        if not demand:
            continue
        def rank_key(item: tuple[str, RetrievedEvidence, set[str]]) -> tuple:
            semantic_coverage = len(demand & item[2]) / len(demand)
            primary = (
                (_evidence_recency(item[1]), semantic_coverage)
                if temporal_current
                else (semantic_coverage, _evidence_recency(item[1]))
            )
            return (*primary, _value_bearing_score(item[0]),
                    len(item[2] & set().union(*demands)), min(len(item[0]), 500))
        ranked = sorted(sentence_rows, key=rank_key, reverse=True)
        if not ranked or not (demand & ranked[0][2]):
            continue
        sentence, row, _ = ranked[0]
        normalized = re.sub(r"\s+", " ", sentence.lower()).strip(" .")
        if normalized not in selected_text:
            selected.append((sentence, row))
            selected_text.add(normalized)
    # Partial updates often change a time window without repeating its stable
    # date. Add one relevant authorized temporal anchor when the bundle has
    # concrete times but no date, preserving unchanged state compositionally.
    selected_categories = {
        category for text, _ in selected for category in _value_categories(text)
    }
    if "clock" in selected_categories and "calendar_date" not in selected_categories:
        all_demands = set().union(*demands)
        dated = [
            item for item in sentence_rows
            if "calendar_date" in _value_categories(item[0])
        ]
        dated.sort(
            key=lambda item: (
                len(item[2] & all_demands),
                _evidence_recency(item[1]),
                _value_bearing_score(item[0]),
            ),
            reverse=True,
        )
        if dated and dated[0][2] & all_demands:
            sentence, row, _ = dated[0]
            normalized = re.sub(r"\s+", " ", sentence.lower()).strip(" .")
            if normalized not in selected_text:
                selected.insert(0, (sentence, row))
                selected_text.add(normalized)
    if not selected:
        return None
    text = " ".join(sentence for sentence, _ in selected)
    return AnswerResult(
        prediction=text,
        answer_text=text,
        used_memory_ids=list(dict.fromkeys(row.memory_id for _, row in selected)),
        reasoning_summary="Coverage-aware realization from complementary authorized records.",
        action="answer",
        answer_structured={"realization_mode": "authorized_record_bundle_coverage"},
    )


def _value_bearing_score(text: str) -> int:
    """Estimate whether a sentence carries concrete, verifiable field values."""
    lowered = str(text or "").lower()
    score = 0
    if re.search(r"\b\d{1,2}:\d{2}\s*(?:a\.?m\.?|p\.?m\.?)?\b", lowered):
        score += 3
    if re.search(
        r"\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
        r"january|february|march|april|may|june|july|august|september|october|november|december)\b",
        lowered,
    ):
        score += 2
    if re.search(r"\b\d{2,}\b", lowered):
        score += 2
    if re.search(r"['\"][^'\"]+['\"]", text):
        score += 1
    return score


def _semantic_terms(text: str, *, stop: set[str] | None = None) -> set[str]:
    aliases = {
        "callback": "call", "callbacks": "call",
        "booked": "book", "booking": "book", "bookings": "book",
    }
    blocked = stop or set()
    return {
        aliases.get(token, token)
        for token in re.findall(r"[a-z0-9]+", str(text or "").lower())
        if len(token) > 2 and token not in blocked
    }


def _value_categories(text: str) -> set[str]:
    lowered = str(text or "").lower()
    categories: set[str] = set()
    if re.search(r"\b\d{1,2}:\d{2}\s*(?:a\.?m\.?|p\.?m\.?)?\b", lowered):
        categories.add("clock")
    if re.search(r"\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", lowered):
        categories.add("date")
    if re.search(
        r"\b(?:january|february|march|april|may|june|july|august|"
        r"september|october|november|december)\b(?:\s+\d{1,2})?",
        lowered,
    ):
        categories.add("calendar_date")
    if re.search(r"['\"][^'\"]+['\"]", text):
        categories.add("quoted")
    if re.search(r"\b\d{2,}\b", lowered):
        categories.add("number")
    return categories


def _evidence_recency(row: RetrievedEvidence) -> tuple[int, str]:
    """Return a stable turn-first recency key for competing field values."""
    turn_numbers = [
        int(match.group(1))
        for value in [row.memory_id, *list(row.source_message_ids or [])]
        for match in re.finditer(r"(?:^|_)(?:t|msg_|event_)?(\d{1,6})(?:_|$)", str(value), re.IGNORECASE)
    ]
    timestamp = "".join(ch for ch in str(row.time or "") if ch.isdigit())
    return max(turn_numbers, default=0), timestamp


def _render_authorized_current_slots(
    *, evidence: list[RetrievedEvidence], contract: dict[str, Any]
) -> AnswerResult | None:
    requested = list(dict.fromkeys(
        str(slot).strip()
        for slot in (
            contract.get("requested_attributes")
            or contract.get("requested_slots")
            or []
        )
        if str(slot).strip()
    ))
    if not requested:
        return None
    if str(contract.get("temporal_scope") or "").lower() != "current":
        return None
    resolved: dict[str, tuple[str, str, str]] = {}
    for row in evidence:
        try:
            frame = compile_evidence_frame(row)
        except Exception:
            continue
        if str(frame.lifecycle_status or "active").lower() != "active":
            continue
        timestamp = "".join(ch for ch in str(frame.effective_time or row.time or "") if ch.isdigit())
        surfaces = dict(frame.surface_spans or {})
        slots = dict(frame.slots or {})
        certified_slots = dict((row.metadata or {}).get("slots") or {})
        certified_surfaces = dict((row.metadata or {}).get("surface_spans") or {})
        for slot in requested:
            value = str(
                certified_surfaces.get(slot)
                or certified_slots.get(slot)
                or surfaces.get(slot)
                or slots.get(slot)
                or ""
            ).strip()
            if not value:
                continue
            previous = resolved.get(slot)
            candidate = (timestamp, value, row.memory_id)
            if previous is None or candidate[0] > previous[0]:
                resolved[slot] = candidate
    if any(slot not in resolved for slot in requested):
        return None
    clauses = [f"{slot.replace('_', ' ')}: {resolved[slot][1]}" for slot in requested]
    used_ids = list(dict.fromkeys(resolved[slot][2] for slot in requested))
    text = "Current authorized state: " + "; ".join(clauses) + "."
    return AnswerResult(
        prediction=text,
        answer_text=text,
        used_memory_ids=used_ids,
        reasoning_summary="Canonical realization from latest active policy-authorized typed slots.",
        action="answer",
        answer_structured={
            "typed_slots": {slot: resolved[slot][1] for slot in requested},
            "realization_mode": "authorized_current_slots",
        },
    )
