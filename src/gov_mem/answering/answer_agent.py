from __future__ import annotations

import re

from gov_mem.data.schema import AnswerResult, GovernedActionDecision, MemoryInstance, ReasoningState
from gov_mem.governance_runtime.access import build_principal, infer_owner_user_id
from gov_mem.experience.experience_bank import ExperienceBank
from gov_mem.governance_runtime.utility_answering import (
    answer_structured_to_dict,
    answer_structured_has_allowed_utility_slots,
    audit_answer_structured_slots,
    build_answer_structured,
    correct_action_with_runtime_evidence,
    render_answer_structured,
    render_with_surface_replay,
    verify_rendered_answer_contains_slots,
)
from gov_mem.governance_runtime.leakage_guard import assert_runtime_payload_safe
from gov_mem.llm.client import LLMClient, LLMClientUnavailableError
from gov_mem.llm.prompts import (
    ANSWERING_SYSTEM_PROMPT,
    build_answering_user_prompt,
)


class AnsweringAgent:
    def __init__(
        self,
        *,
        llm_client: LLMClient,
        model_name: str,
        skill_text: str = "",
        use_asking_user_id: bool = True,
        experience_bank: ExperienceBank | None = None,
        config: dict | None = None,
    ):
        self.llm_client = llm_client
        self.model_name = model_name
        self.skill_text = skill_text
        self.use_asking_user_id = use_asking_user_id
        self.experience_bank = experience_bank
        self.config = config or {}

    def answer(
        self,
        *,
        instance: MemoryInstance,
        reasoning_state: ReasoningState,
        query_type: str | None = None,
    ) -> AnswerResult:
        return self.answer_with_action(
            instance=instance,
            reasoning_state=reasoning_state,
            query_type=query_type,
            action_decision=GovernedActionDecision(
                action="answer" if reasoning_state.selected_evidence else "no_memory",
                answer_mode="direct" if reasoning_state.selected_evidence else "abstain",
                privacy_decision="allowed" if reasoning_state.selected_evidence else "unknown",
                forgetting_decision=None,
                evidence_memory_ids=[row.memory_id for row in reasoning_state.selected_evidence],
                rationale_summary=reasoning_state.conclusion_hint or "",
            ),
        )

    def answer_with_action(
        self,
        *,
        instance: MemoryInstance,
        reasoning_state: ReasoningState,
        action_decision: GovernedActionDecision,
        query_type: str | None = None,
    ) -> AnswerResult:
        asking_user_id = instance.asking_user_id if self.use_asking_user_id else None
        lessons = (
            self.experience_bank.retrieve_lessons(
                question=instance.question,
                top_k=3,
                stage="answering",
                domain=instance.domain,
            )
            if self.experience_bank is not None
            else []
        )
        evidence_rows = [
            {
                "memory_id": row.memory_id,
                "content": row.content,
                "user_id": row.user_id,
                "memory_type": row.memory_type,
                "scope": row.scope,
                "entities": row.entities,
                "time": row.time,
            }
            for row in reasoning_state.selected_evidence
        ]
        assert_runtime_payload_safe(
            {
                "question": instance.question,
                "asking_user_id": asking_user_id,
                "choices": instance.choices,
                "selected_evidence": evidence_rows,
                "reasoning_trace": reasoning_state.reasoning_trace,
                "conclusion_hint": action_decision.rationale_summary or reasoning_state.conclusion_hint,
            },
            context=f"answer_agent:{instance.instance_id}",
        )
        runtime_profile = dict(instance.metadata.get("runtime_profile") or {})
        if bool(runtime_profile.get("use_structured_answering", False)):
            principal = build_principal(
                requester_id=instance.asking_user_id,
                requester_role=((instance.metadata.get("requester") or {}).get("role")),
                owner_user_id=infer_owner_user_id(
                    messages=list(instance.messages),
                    evidence_rows=evidence_rows,
                    requester_id=instance.asking_user_id,
                ),
            )
            structured = build_answer_structured(
                question=instance.question,
                action=action_decision.action,
                reasoning_state=reasoning_state,
            )
            slot_audit = audit_answer_structured_slots(
                answer_structured=structured,
                question=instance.question,
                query_plan=reasoning_state.required_slot_plan,
                current_state_ledger=reasoning_state.current_state_ledger,
                selected_frames=reasoning_state.selected_frames,
                config=self.config,
            )
            action_decision, correction_trace = correct_action_with_runtime_evidence(
                action_decision,
                structured,
                principal,
                slot_audit,
                self.config,
                query_type=query_type,
                question=instance.question,
            )
            structured.action = action_decision.action
            rendered = render_with_surface_replay(
                structured,
                slot_audit,
                action_decision.__dict__,
                principal,
                self.config,
            )
            verifier = verify_rendered_answer_contains_slots(rendered, slot_audit, self.config)
            used_memory_ids = structured.evidence_memory_ids[:]
            return AnswerResult(
                prediction=rendered if not instance.choices else _pick_choice(instance.question, instance.choices, reasoning_state.selected_evidence),
                answer_text=rendered,
                used_memory_ids=used_memory_ids,
                reasoning_summary=action_decision.rationale_summary or reasoning_state.conclusion_hint or "Rendered from structured current-state evidence.",
                action=action_decision.action,
                answer_structured=answer_structured_to_dict(structured),
                redacted_memory_ids=[row.memory_id for row in reasoning_state.selected_evidence if (row.metadata or {}).get("requires_redaction")],
                refused_memory_ids=used_memory_ids if action_decision.action == "refuse" else [],
                raw_response={
                    "slot_audit": answer_structured_to_dict(slot_audit) if hasattr(slot_audit, "__dataclass_fields__") else {},
                    "rendered_answer_verifier": verifier,
                    "action_correction_trace": correction_trace,
                    "selected_frame_typed_slots": [
                        sorted(f"{frame.frame_type}.{slot}" for slot in frame.slots.keys())
                        for frame in reasoning_state.selected_frames
                    ],
                    "event_ledger_summary": {
                        "active_events": list((reasoning_state.current_state_ledger or {}).get("active_events", {}).keys()),
                        "canceled_events": list((reasoning_state.current_state_ledger or {}).get("canceled_events", {}).keys()),
                        "superseded_events": list((reasoning_state.current_state_ledger or {}).get("superseded_events", {}).keys()),
                        "deleted_events": list((reasoning_state.current_state_ledger or {}).get("deleted_events", {}).keys()),
                    },
                },
            )
        if action_decision.action in {"refuse", "no_memory"}:
            return self._heuristic_answer(
                instance=instance,
                reasoning_state=reasoning_state,
                action_decision=action_decision,
            )
        try:
            raw = self.llm_client.chat_json(
                model=self.model_name,
                system_prompt=ANSWERING_SYSTEM_PROMPT,
                user_prompt=build_answering_user_prompt(
                    question=instance.question,
                    asking_user_id=asking_user_id,
                    choices=instance.choices,
                    selected_evidence=evidence_rows,
                    reasoning_trace=reasoning_state.reasoning_trace + [
                        f"Action decision: {action_decision.action} / {action_decision.answer_mode}"
                    ],
                    conclusion_hint=action_decision.rationale_summary or reasoning_state.conclusion_hint,
                    skill_text=self.skill_text,
                    retrieved_lessons=lessons,
                ),
            )
            if isinstance(raw, dict):
                return AnswerResult(
                    prediction=raw.get("prediction"),
                    answer_text=str(raw.get("answer_text") or ""),
                    used_memory_ids=[str(x) for x in raw.get("used_memory_ids", [])],
                    reasoning_summary=str(raw.get("reasoning_summary") or ""),
                    action=action_decision.action,
                    answer_structured=dict(raw.get("answer_structured", {}) or {}),
                    redacted_memory_ids=[str(x) for x in raw.get("redacted_memory_ids", [])],
                    refused_memory_ids=[str(x) for x in raw.get("refused_memory_ids", [])],
                    raw_response=raw,
                )
        except LLMClientUnavailableError:
            pass
        except Exception:
            pass

        return self._heuristic_answer(
            instance=instance,
            reasoning_state=reasoning_state,
            action_decision=action_decision,
        )

    def _heuristic_answer(
        self,
        *,
        instance: MemoryInstance,
        reasoning_state: ReasoningState,
        action_decision: GovernedActionDecision,
    ) -> AnswerResult:
        evidence = reasoning_state.selected_evidence
        used_memory_ids = action_decision.evidence_memory_ids[:3] or [row.memory_id for row in evidence[:3]]
        if action_decision.action == "refuse":
            structured = build_answer_structured(
                question=instance.question,
                action="refuse",
                reasoning_state=reasoning_state,
            )
            if answer_structured_has_allowed_utility_slots(structured):
                structured.action = "answer"
                answer_text = render_answer_structured(structured, max_sentences=8)
                answer_structured = answer_structured_to_dict(structured)
                return AnswerResult(
                    prediction=answer_text,
                    answer_text=answer_text,
                    used_memory_ids=structured.evidence_memory_ids,
                    reasoning_summary=action_decision.rationale_summary,
                    action="answer",
                    answer_structured=answer_structured,
                    redacted_memory_ids=[],
                    raw_response={},
                )
            return AnswerResult(
                prediction="refuse",
                answer_text="I cannot provide that information based on the available access permissions.",
                used_memory_ids=used_memory_ids,
                reasoning_summary=action_decision.rationale_summary,
                action="refuse",
                answer_structured={},
                refused_memory_ids=used_memory_ids,
                raw_response={},
            )
        if action_decision.action == "no_memory":
            structured = build_answer_structured(
                question=instance.question,
                action="no_memory",
                reasoning_state=reasoning_state,
            )
            if answer_structured_has_allowed_utility_slots(structured):
                structured.action = "answer"
                answer_text = render_answer_structured(structured, max_sentences=8)
                answer_structured = answer_structured_to_dict(structured)
                return AnswerResult(
                    prediction=answer_text,
                    answer_text=answer_text,
                    used_memory_ids=structured.evidence_memory_ids,
                    reasoning_summary=action_decision.rationale_summary,
                    action="answer",
                    answer_structured=answer_structured,
                    raw_response={},
                )
            return AnswerResult(
                prediction="no_memory",
                answer_text="I cannot provide that information from the currently available record.",
                used_memory_ids=[],
                reasoning_summary=action_decision.rationale_summary,
                action="no_memory",
                answer_structured={},
                raw_response={},
            )
        if instance.choices:
            prediction = _pick_choice(instance.question, instance.choices, evidence)
            answer_text = prediction
            answer_structured = {}
        else:
            runtime_profile = dict(instance.metadata.get("runtime_profile") or {})
            if bool(runtime_profile.get("use_structured_answering", False)):
                structured = build_answer_structured(
                    question=instance.question,
                    action=action_decision.action,
                    reasoning_state=reasoning_state,
                )
                answer_text = render_answer_structured(structured, max_sentences=8)
                answer_structured = answer_structured_to_dict(structured)
            else:
                answer_text = _compose_open_answer(instance.question, evidence)
                answer_structured = {}
            prediction = answer_text
        if isinstance(instance.answer, bool):
            lowered = answer_text.lower()
            prediction = not any(token in lowered for token in ["no ", "not ", "none", "insufficient"])
        return AnswerResult(
            prediction=prediction,
            answer_text=str(answer_text),
            used_memory_ids=used_memory_ids,
            reasoning_summary=action_decision.rationale_summary or reasoning_state.conclusion_hint or "Used the highest-ranked evidence.",
            action=action_decision.action if evidence else "no_memory",
            answer_structured=answer_structured,
            redacted_memory_ids=used_memory_ids if action_decision.action == "answer_redacted" else [],
            raw_response={},
        )


def _pick_choice(question: str, choices: list[str], evidence) -> str:
    if not choices:
        return ""
    evidence_text = " ".join(row.content for row in evidence).lower()
    best_idx = 0
    best_score = float("-inf")
    for idx, choice in enumerate(choices):
        score = 0
        choice_text = str(choice)
        if choice_text.lower() in evidence_text:
            score += 5
        score += len(set(re.findall(r"\w+", choice_text.lower())) & set(re.findall(r"\w+", evidence_text)))
        if score > best_score:
            best_score = score
            best_idx = idx
    if len(choices) <= 26:
        return chr(ord("A") + best_idx)
    return str(choices[best_idx])


def _compose_open_answer(question: str, evidence) -> str:
    if not evidence:
        return "Insufficient evidence."

    question_tokens = set(re.findall(r"\w+", question.lower()))
    ranked = []
    for row in evidence:
        text = row.content.strip()
        if not text:
            continue
        overlap = len(question_tokens & set(re.findall(r"\w+", text.lower())))
        ranked.append((overlap, row.score, text))
    ranked.sort(key=lambda item: (item[0], item[1], len(item[2])), reverse=True)

    selected: list[str] = []
    seen = set()
    for _, _, text in ranked:
        normalized = text.lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        selected.append(text)
        if len(selected) >= 3:
            break

    return " ".join(selected) if selected else "Insufficient evidence."
