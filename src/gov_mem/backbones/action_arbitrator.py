from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ActionArbitrationResult:
    llm_action_prediction: str
    symbolic_action_constraint: str | None
    final_action: str
    fallback_used: bool = False
    arbitration_trace: list[str] = field(default_factory=list)


class ActionArbitrator:
    def arbitrate(
        self,
        *,
        llm_action_prediction: str,
        symbolic_decision: dict[str, Any] | None,
        fallback_to_v0_on_low_confidence: bool,
        runtime_skill_bundle: dict[str, Any] | None = None,
    ) -> ActionArbitrationResult:
        trace: list[str] = [f"llm_action={llm_action_prediction}"]
        action_patches = list((runtime_skill_bundle or {}).get("action_patches") or [])
        if action_patches:
            trace.append(f"skill_action_patches={action_patches}")
        if not symbolic_decision:
            trace.append("no_symbolic_decision")
            return ActionArbitrationResult(
                llm_action_prediction=llm_action_prediction,
                symbolic_action_constraint=None,
                final_action=llm_action_prediction,
                fallback_used=True,
                arbitration_trace=trace,
            )

        symbolic_action = str(symbolic_decision.get("action_constraint") or "")
        confidence = float(symbolic_decision.get("confidence") or 0.0)
        trace.append(f"symbolic_action={symbolic_action}")
        trace.append(f"symbolic_confidence={confidence:.2f}")

        if fallback_to_v0_on_low_confidence and confidence < 0.75:
            trace.append("fallback_to_v0_low_confidence")
            return ActionArbitrationResult(
                llm_action_prediction=llm_action_prediction,
                symbolic_action_constraint=symbolic_action,
                final_action=llm_action_prediction,
                fallback_used=True,
                arbitration_trace=trace,
            )

        blocked_facts = list(symbolic_decision.get("blocked_facts") or [])
        explicit_deleted_request = bool(symbolic_decision.get("explicit_deleted_request"))
        explicit_historical_request = bool(symbolic_decision.get("explicit_historical_request"))
        if symbolic_action == "no_memory" and blocked_facts:
            trace.append("deleted_block_enforced")
            return ActionArbitrationResult(
                llm_action_prediction=llm_action_prediction,
                symbolic_action_constraint=symbolic_action,
                final_action="no_memory",
                arbitration_trace=trace,
            )

        if symbolic_action == "no_memory" and bool(symbolic_decision.get("explicit_deleted_request")) and bool(symbolic_decision.get("explicit_historical_request")):
            trace.append("explicit_deleted_history_constraint_overrides_llm")
            return ActionArbitrationResult(
                llm_action_prediction=llm_action_prediction,
                symbolic_action_constraint=symbolic_action,
                final_action="no_memory",
                arbitration_trace=trace,
            )

        if symbolic_action == "refuse" and llm_action_prediction in {"answer", "answer_redacted", "no_memory"}:
            trace.append("symbolic_denial_overrides_llm")
            return ActionArbitrationResult(
                llm_action_prediction=llm_action_prediction,
                symbolic_action_constraint=symbolic_action,
                final_action="refuse",
                arbitration_trace=trace,
            )

        if (
            symbolic_action == "answer_redacted"
            and llm_action_prediction in {"answer", "no_memory", "refuse"}
            and (
                llm_action_prediction != "refuse"
                or bool(symbolic_decision.get("allowed_slots"))
            )
        ):
            trace.append(
                "symbolic_redaction_overrides_full_answer"
                if llm_action_prediction == "answer"
                else "symbolic_policy_projection_overrides_false_restriction"
            )
            return ActionArbitrationResult(
                llm_action_prediction=llm_action_prediction,
                symbolic_action_constraint=symbolic_action,
                final_action="answer_redacted",
                arbitration_trace=trace,
            )

        if (
            symbolic_action == "answer"
            and llm_action_prediction in {"refuse", "no_memory", "answer_redacted"}
            and bool(
                symbolic_decision.get("utility_certificate_complete")
                or symbolic_decision.get("record_bundle_certificate_complete")
            )
            and not list(symbolic_decision.get("denied_slots") or [])
            and not explicit_deleted_request
            and not explicit_historical_request
        ):
            # A complete certificate is constructed only from latest active,
            # authorized provenance. Unrelated stale facts elsewhere in the
            # episode must not cause over-refusal of the current request.
            trace.append("complete_active_slot_certificate_overrides_false_restriction")
            return ActionArbitrationResult(
                llm_action_prediction=llm_action_prediction,
                symbolic_action_constraint=symbolic_action,
                final_action="answer",
                arbitration_trace=trace,
            )
            return ActionArbitrationResult(
                llm_action_prediction=llm_action_prediction,
                symbolic_action_constraint=symbolic_action,
                final_action="answer_redacted",
                arbitration_trace=trace,
            )

        if "force_no_memory_on_deleted_query" in action_patches and blocked_facts and (
            explicit_deleted_request or explicit_historical_request
        ):
            trace.append("skill_force_no_memory_on_deleted_query")
            return ActionArbitrationResult(
                llm_action_prediction=llm_action_prediction,
                symbolic_action_constraint=symbolic_action,
                final_action="no_memory",
                arbitration_trace=trace,
            )

        if (
            "prefer_answer_redacted_for_sensitive_medical_non_owner" in action_patches
            and llm_action_prediction == "answer"
            and symbolic_action in {"answer", "answer_redacted"}
        ):
            trace.append("skill_force_answer_redacted_medical")
            return ActionArbitrationResult(
                llm_action_prediction=llm_action_prediction,
                symbolic_action_constraint=symbolic_action,
                final_action="answer_redacted",
                arbitration_trace=trace,
            )

        if (
            "prefer_answer_redacted_for_education_partial_access" in action_patches
            and llm_action_prediction == "answer"
            and symbolic_action in {"answer", "answer_redacted"}
        ):
            trace.append("skill_force_answer_redacted_education")
            return ActionArbitrationResult(
                llm_action_prediction=llm_action_prediction,
                symbolic_action_constraint=symbolic_action,
                final_action="answer_redacted",
                arbitration_trace=trace,
            )

        trace.append("llm_action_retained")
        return ActionArbitrationResult(
            llm_action_prediction=llm_action_prediction,
            symbolic_action_constraint=symbolic_action,
            final_action=llm_action_prediction,
            arbitration_trace=trace,
        )


def action_arbitration_to_dict(result: ActionArbitrationResult) -> dict[str, Any]:
    return asdict(result)
