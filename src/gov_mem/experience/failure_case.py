from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from gov_mem.experience.structural_diagnosis import diagnose_runtime_trace
from gov_mem.utils.io import read_json, read_jsonl


@dataclass
class FailureCase:
    case_id: str
    domain: str
    backbone: str
    query: str
    predicted_action: str
    official_result: dict[str, Any]
    failure_type: str
    retrieved_utility_evidence: list[Any]
    retrieved_governance_evidence: list[Any]
    symbolic_decision: dict[str, Any]
    final_answer: str
    suspected_causes: list[str]
    structural_diagnosis: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)


class DevFailureCaseBuilder:
    def build_from_run(
        self,
        *,
        run_dir: str | Path,
        backbone: str = "rag_policy_amem",
    ) -> list[FailureCase]:
        root = Path(run_dir)
        case_result_paths = list(root.glob("*/eval/*/case_results.jsonl"))
        eval_case_path = root / "eval"
        if eval_case_path.exists():
            case_result_paths.extend(eval_case_path.glob("*/case_results.jsonl"))
        failure_cases: list[FailureCase] = []
        for case_result_path in sorted(dict.fromkeys(case_result_paths)):
            domain_root = case_result_path.parents[2]
            case_results = read_jsonl(case_result_path)
            domain = _infer_domain(case_results=case_results, domain_root=domain_root)
            official_scores = _load_official_scores(domain_root=domain_root, domain=domain)
            if not official_scores:
                continue
            for case_result in case_results:
                instance_id = str(case_result.get("instance_id") or "")
                official_row = official_scores.get(instance_id)
                if official_row is None or _is_official_success(official_row):
                    continue
                prediction_payload = _safe_read_json(_resolve_dataset_artifact(domain_root / "predictions", instance_id))
                runtime_trace = _safe_read_json(_resolve_runtime_trace(domain_root=domain_root, instance_id=instance_id))
                retrieval_bundle = _safe_read_json(domain_root / "debug" / "retrieval_bundle" / f"{instance_id}.json")
                symbolic_runtime = _safe_read_json(domain_root / "debug" / "symbolic_decisions" / f"{instance_id}.json")
                final_realization = _safe_read_json(domain_root / "debug" / "final_realization" / f"{instance_id}.json")
                failure_type = _map_failure_type(case_result=case_result, official_row=official_row)
                failure_cases.append(
                    FailureCase(
                        case_id=instance_id,
                        domain=domain,
                        backbone=backbone,
                        query=str(case_result.get("question") or ""),
                        predicted_action=str(
                            ((case_result.get("metadata") or {}).get("predicted_action"))
                            or ((case_result.get("metadata") or {}).get("action"))
                            or official_row.get("pred_action")
                            or ""
                        ),
                        official_result=official_row,
                        failure_type=failure_type,
                        retrieved_utility_evidence=list(((retrieval_bundle.get("utility_evidence") or {}).get("facts") or []))[:8],
                        retrieved_governance_evidence=_build_governance_snapshot(
                            retrieval_bundle=retrieval_bundle,
                            final_realization=final_realization,
                        ),
                        symbolic_decision=dict((symbolic_runtime.get("decision") or {})),
                        final_answer=str(
                            prediction_payload.get("answer_text")
                            or prediction_payload.get("prediction")
                            or ""
                        ),
                        suspected_causes=_suspected_causes(
                            case_result=case_result,
                            official_row=official_row,
                            final_realization=final_realization,
                        ),
                        structural_diagnosis=diagnose_runtime_trace(runtime_trace),
                        provenance={
                            "run_dir": str(root),
                            "domain_root": str(domain_root),
                            "case_result_path": str(case_result_path),
                            "created_from_dev_only": True,
                            "official_score_available": True,
                            "runtime_trace_path": str(_resolve_runtime_trace(domain_root=domain_root, instance_id=instance_id)),
                        },
                    )
                )
        return failure_cases


def failure_case_to_dict(case: FailureCase) -> dict[str, Any]:
    return asdict(case)


def _infer_domain(*, case_results: list[dict[str, Any]], domain_root: Path) -> str:
    if case_results:
        value = str(case_results[0].get("domain") or "").strip()
        if value:
            return value
    return domain_root.name


def _load_official_scores(*, domain_root: Path, domain: str) -> dict[str, dict[str, Any]]:
    score_path = _resolve_official_score_path(domain_root=domain_root, domain=domain)
    if not score_path.exists():
        return {}
    return {
        str(row.get("checkpoint_id") or ""): row
        for row in read_jsonl(score_path)
        if str(row.get("checkpoint_id") or "").strip()
    }


def _is_official_success(official_row: dict[str, Any]) -> bool:
    if not bool(official_row.get("action_correct")):
        return False
    if bool(official_row.get("privacy_e2e_leak")) or bool(official_row.get("deletion_e2e_leak")):
        return False
    query_type = str(official_row.get("query_type") or "")
    if query_type == "utility":
        return bool(official_row.get("utility_correct"))
    return not bool(official_row.get("over_refusal"))


def _map_failure_type(*, case_result: dict[str, Any], official_row: dict[str, Any]) -> str:
    metadata = dict(case_result.get("metadata") or {})
    expected_action = str(official_row.get("expected_action") or "")
    predicted_action = str(official_row.get("pred_action") or metadata.get("predicted_action") or "")
    if bool(official_row.get("privacy_e2e_leak")):
        return "leakage"
    if bool(official_row.get("deletion_e2e_leak")):
        return "deleted_reconstruction"
    if bool(official_row.get("over_refusal")):
        return "over_refusal"
    if expected_action == "answer_redacted" and predicted_action == "answer":
        return "under_redaction"
    if expected_action == "answer" and predicted_action in {"no_memory", "refuse"}:
        return "no_memory_collapse"
    if str(case_result.get("failure_type") or "") in {"state_tracking_error", "temporal_error"}:
        return "stale_state"
    return "missing_utility"


def _build_governance_snapshot(
    *,
    retrieval_bundle: dict[str, Any],
    final_realization: dict[str, Any],
) -> list[Any]:
    governance = dict(retrieval_bundle.get("governance_evidence") or {})
    snapshot = [
        {"roles": list(governance.get("roles") or [])[:4]},
        {"policies": list(governance.get("policies") or [])[:4]},
        {"relations": list(governance.get("relations") or [])[:4]},
        {"deletions": list(governance.get("deletions") or [])[:4]},
    ]
    if final_realization:
        snapshot.append({"final_realization": final_realization})
    return snapshot


def _suspected_causes(
    *,
    case_result: dict[str, Any],
    official_row: dict[str, Any],
    final_realization: dict[str, Any],
) -> list[str]:
    causes: list[str] = []
    local_failure = str(case_result.get("failure_type") or "").strip()
    if local_failure:
        causes.append(f"local_failure={local_failure}")
    if not bool(official_row.get("action_correct")):
        causes.append("action_arbitration_or_prediction_mismatch")
    if bool(official_row.get("over_refusal")):
        causes.append("excessive_conservatism")
    if str(official_row.get("query_type") or "") == "utility" and not bool(official_row.get("utility_correct")):
        causes.append("utility_surface_missing_or_incomplete")
    if bool(official_row.get("privacy_e2e_leak")):
        causes.append("privacy_scope_violation")
    if bool(official_row.get("deletion_e2e_leak")):
        causes.append("deleted_fact_reconstruction")
    if final_realization:
        arbitration = dict(final_realization.get("action_arbitration_result") or {})
        if bool(arbitration.get("fallback_used")):
            causes.append("fallback_to_v0_on_low_confidence")
        verifier = dict(final_realization.get("post_verifier") or {})
        if not bool(verifier.get("passed", True)):
            causes.append("post_realization_verifier_failure")
    return causes or ["official_failure_without_specific_local_tag"]


def _safe_read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return dict(read_json(path) or {})


def _resolve_dataset_artifact(root: Path, instance_id: str) -> Path:
    for dataset_dir in sorted(root.glob("*")):
        candidate = dataset_dir / f"{instance_id}.json"
        if candidate.exists():
            return candidate
    return root / "checkpoint_benchmark" / f"{instance_id}.json"


def _resolve_runtime_trace(*, domain_root: Path, instance_id: str) -> Path:
    candidates = [
        domain_root / "debug_cases" / "checkpoint_benchmark" / f"{instance_id}.json",
        domain_root / "debug" / "runtime" / f"{instance_id}.json",
    ]
    return next((path for path in candidates if path.exists()), candidates[0])


def _resolve_official_score_path(*, domain_root: Path, domain: str) -> Path:
    for dataset_dir in sorted((domain_root / "official_eval").glob("*")):
        candidate = dataset_dir / domain / "scores.jsonl"
        if candidate.exists():
            return candidate
    return domain_root / "official_eval" / "checkpoint_benchmark" / domain / "scores.jsonl"
