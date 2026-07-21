from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from gov_mem.utils.io import read_json, read_jsonl, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--case_ids", nargs="+", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    summary_path = output_dir / "official_eval" / args.dataset_name / args.domain / "summary.json"
    scores_path = output_dir / "official_eval" / args.dataset_name / args.domain / "scores.jsonl"
    debug_dir = output_dir / "debug_cases" / args.dataset_name

    summary = read_json(summary_path) if summary_path.exists() else {}
    scores = {}
    if scores_path.exists():
        for row in read_jsonl(scores_path):
            scores[str(row.get("checkpoint_id") or row.get("instance_id"))] = row

    report = {
        "summary": summary,
        "cases": [],
    }
    for case_id in args.case_ids:
        debug_path = debug_dir / f"med_episode_rewrite_en_001_early_pregnancy_confidential_contact_{case_id}.json"
        debug = read_json(debug_path) if debug_path.exists() else {}
        official = scores.get(f"med_episode_rewrite_en_001_early_pregnancy_confidential_contact_{case_id}", {})
        retrieval_before = _as_list(debug.get("retrieved_before_privacy_filter"))
        retrieval_after = _as_list(debug.get("retrieved_after_privacy_filter"))
        filtered = _as_list(debug.get("filtered_evidence"))
        selected_evidence = _as_list(debug.get("selected_evidence"))
        atomic_memories = _as_list(debug.get("atomic_memories"))
        retrieved_atomic_memories = _as_list(debug.get("retrieved_atomic_memories"))
        retrieval_count_before = len(retrieval_before)
        retrieval_count_after = len(retrieval_after)
        selected_frames = _extract_selected_frames(debug)
        structured = (debug.get("final_prediction") or {}).get("answer_structured", {})
        active_schedule_items = structured.get("active_schedule_items", [])
        canceled_items = structured.get("canceled_items", [])
        allergies = structured.get("allergies", [])
        medications = structured.get("medications", [])
        rendered = (debug.get("final_prediction") or {}).get("answer_text", "")
        atomic_quality = _summarize_atomic_quality(atomic_memories, retrieved_atomic_memories)
        stage_loss_type = _infer_stage_loss_type(
            official=official,
            debug=debug,
            retrieval_before=retrieval_before,
            retrieval_after=retrieval_after,
            filtered=filtered,
            selected_evidence=selected_evidence,
            selected_frames=selected_frames,
            structured=structured,
            rendered=rendered,
            atomic_quality=atomic_quality,
        )
        report["cases"].append(
            {
                "case_id": case_id,
                "predicted_action": (debug.get("final_prediction") or {}).get("action"),
                "official_utility_success": bool(official.get("utility_correct")),
                "over_refusal": bool(official.get("over_refusal")),
                "retrieval_count_before_filter": retrieval_count_before,
                "retrieval_count_after_filter": retrieval_count_after,
                "frame_count": len(debug.get("memory_items", [])),
                "selected_frame_count": len(selected_frames),
                "active_schedule_items_count": len(active_schedule_items),
                "canceled_items_count": len(canceled_items),
                "allergies_count": len(allergies),
                "medications_count": len(medications),
                "selected_evidence_count": len(selected_evidence),
                "atomic_memories_count": len(atomic_memories),
                "retrieved_atomic_memories_count": len(retrieved_atomic_memories),
                "empty_atomic_memories_count": atomic_quality["empty_atomic_memories_count"],
                "empty_retrieved_atomic_memories_count": atomic_quality["empty_retrieved_atomic_memories_count"],
                "rendered_answer": rendered,
                "stage_loss_type": stage_loss_type,
                "notes": _notes_for_case(
                    official=official,
                    debug=debug,
                    structured=structured,
                    filtered=filtered,
                    selected_evidence=selected_evidence,
                    atomic_quality=atomic_quality,
                ),
            }
        )

    analysis_dir = output_dir / "analysis"
    write_json(analysis_dir / "stage_loss_report.json", report)
    _write_markdown(analysis_dir / "stage_loss_report.md", report)


def _infer_stage_loss_type(
    *,
    official,
    debug,
    retrieval_before,
    retrieval_after,
    filtered,
    selected_evidence,
    selected_frames,
    structured,
    rendered,
    atomic_quality,
):
    pred_action = str(official.get("pred_action") or (debug.get("final_prediction") or {}).get("action") or "")
    if pred_action in {"refuse", "no_memory"} and official.get("expected_action") == "answer":
        return "action_loss"
    if not retrieval_before:
        return "retrieval_loss"
    if retrieval_before and not retrieval_after:
        return "access_filter_loss"
    if atomic_quality["retrieved_atomic_memories_count"] > 0 and atomic_quality["empty_retrieved_atomic_memories_count"] == atomic_quality["retrieved_atomic_memories_count"]:
        return "amem_extraction_loss"
    if filtered and not selected_evidence:
        return "policy_filter_loss"
    if retrieval_after and not selected_evidence:
        return "slot_selection_loss"
    if not selected_frames and not selected_evidence:
        return "slot_selection_loss"
    if not (structured.get("active_schedule_items") or structured.get("allergies") or structured.get("medications") or structured.get("instructions")):
        return "answer_structured_loss"
    if not rendered.strip():
        return "renderer_loss"
    verifier = debug.get("coverage_verification") or debug.get("rendered_answer_verifier") or {}
    if verifier and verifier.get("missing_units"):
        return "renderer_omission"
    if not official.get("utility_correct"):
        return "official_match_loss"
    return "ok"


def _notes_for_case(*, official, debug, structured, filtered, selected_evidence, atomic_quality):
    notes = []
    if official.get("utility_correct") is False:
        notes.append(f"pred_action={official.get('pred_action')}")
    if structured.get("unavailable_slots"):
        notes.append(f"missing_slots={structured.get('unavailable_slots')}")
    if filtered:
        notes.append(f"filtered_evidence={len(filtered)}")
    if selected_evidence:
        notes.append(f"selected_evidence={len(selected_evidence)}")
    if atomic_quality["empty_atomic_memories_count"]:
        notes.append(
            "empty_atomic_memories="
            f"{atomic_quality['empty_atomic_memories_count']}/{atomic_quality['atomic_memories_count']}"
        )
    if atomic_quality["empty_retrieved_atomic_memories_count"]:
        notes.append(
            "empty_retrieved_atomic_memories="
            f"{atomic_quality['empty_retrieved_atomic_memories_count']}/{atomic_quality['retrieved_atomic_memories_count']}"
        )
    used_renderer = debug.get("used_renderer")
    if used_renderer:
        notes.append(f"used_renderer={used_renderer}")
    failure_type = debug.get("failure_type")
    if failure_type:
        notes.append(f"local_failure_type={failure_type}")
    return notes


def _extract_selected_frames(debug: dict) -> list[dict]:
    final_prediction = debug.get("final_prediction") or {}
    structured = final_prediction.get("answer_structured") or {}
    frames = _as_list(structured.get("utility_frames"))
    if frames:
        return frames
    packed = debug.get("packed_utility_evidence") or {}
    selected_records = _as_list(packed.get("selected_records"))
    if selected_records:
        return selected_records
    utility_records = _as_list(debug.get("utility_records"))
    if utility_records:
        return utility_records
    return _as_list(debug.get("selected_evidence"))


def _summarize_atomic_quality(atomic_memories: list[dict], retrieved_atomic_memories: list[dict]) -> dict:
    empty_atomic_memories = sum(1 for row in atomic_memories if not str((row or {}).get("content") or "").strip())
    empty_retrieved_atomic_memories = sum(1 for row in retrieved_atomic_memories if not str((row or {}).get("text") or "").strip())
    return {
        "atomic_memories_count": len(atomic_memories),
        "retrieved_atomic_memories_count": len(retrieved_atomic_memories),
        "empty_atomic_memories_count": empty_atomic_memories,
        "empty_retrieved_atomic_memories_count": empty_retrieved_atomic_memories,
    }


def _as_list(value):
    return list(value) if isinstance(value, list) else []


def _write_markdown(path: Path, report: dict) -> None:
    lines = ["# GateMem Stage Loss Report", "", "## Summary", ""]
    summary = report.get("summary", {})
    for key in ["U", "A", "F", "OR", "MGS"]:
        if key in summary:
            lines.append(f"- {key}: {summary[key]}")
    lines.extend(["", "## Cases", ""])
    for case in report.get("cases", []):
        lines.append(f"- {case['case_id']}: {case['stage_loss_type']} | action={case['predicted_action']} | utility={case['official_utility_success']}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
