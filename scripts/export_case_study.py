from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gov_mem.data.gatemem import DOMAINS, load_checkpoints


@dataclass
class CaseStudyRow:
    category: str
    domain: str
    checkpoint_id: str
    question: str
    pred_action: str
    action_correct: bool
    answer: str
    justification: str


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _discover_domain_dirs(run_dir: Path) -> dict[str, Path]:
    domain_dirs: dict[str, Path] = {}
    for domain in DOMAINS:
        candidate = run_dir / domain
        if (candidate / "official_eval" / "gatemem" / domain / "scores.jsonl").exists():
            domain_dirs[domain] = candidate
    if domain_dirs:
        return domain_dirs
    for domain in DOMAINS:
        if (run_dir / "official_eval" / "gatemem" / domain / "scores.jsonl").exists():
            domain_dirs[domain] = run_dir
    return domain_dirs


def _checkpoint_index(domain: str) -> dict[str, dict[str, Any]]:
    return {row["checkpoint_id"]: row for row in load_checkpoints(domain)}


def _prediction_index(path: Path) -> dict[str, dict[str, Any]]:
    rows = _load_jsonl(path)
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        checkpoint_id = str(row.get("checkpoint_id") or "")
        output = dict(row.get("output") or row)
        out[checkpoint_id] = output
    return out


def _select_case_studies(run_dir: Path) -> list[CaseStudyRow]:
    domain_dirs = _discover_domain_dirs(run_dir)
    selected: dict[str, CaseStudyRow] = {}
    for domain, domain_root in domain_dirs.items():
        checkpoint_index = _checkpoint_index(domain)
        predictions = _prediction_index(domain_root / "predictions" / "gatemem" / "predictions.jsonl")
        scores_path = domain_root / "official_eval" / "gatemem" / domain / "scores.jsonl"
        for row in _load_jsonl(scores_path):
            checkpoint_id = str(row.get("checkpoint_id") or "")
            checkpoint = checkpoint_index.get(checkpoint_id) or {}
            prediction = predictions.get(checkpoint_id) or {}
            answer = str(prediction.get("answer") or "")
            pred_action = str(prediction.get("action") or row.get("pred_action") or "")
            query_type = str(row.get("query_type") or checkpoint.get("query_type") or "")
            attack_type = str(row.get("attack_type") or "")
            action_correct = bool(row.get("action_correct"))

            if (
                "medical_partial_disclosure" not in selected
                and domain == "medical"
                and pred_action == "answer_redacted"
                and action_correct
                and not bool(row.get("privacy_leak"))
                and not bool(row.get("privacy_e2e_leak"))
            ):
                selected["medical_partial_disclosure"] = CaseStudyRow(
                    category="medical_partial_disclosure",
                    domain=domain,
                    checkpoint_id=checkpoint_id,
                    question=str(checkpoint.get("query_text") or ""),
                    pred_action=pred_action,
                    action_correct=action_correct,
                    answer=answer,
                    justification="Medical non-owner request is answered through constrained partial disclosure rather than unrestricted replay.",
                )
            if (
                "deleted_memory_blocking" not in selected
                and query_type == "safety"
                and "delete" in attack_type
                and pred_action == "no_memory"
                and action_correct
                and not bool(row.get("deletion_e2e_leak"))
            ):
                selected["deleted_memory_blocking"] = CaseStudyRow(
                    category="deleted_memory_blocking",
                    domain=domain,
                    checkpoint_id=checkpoint_id,
                    question=str(checkpoint.get("query_text") or ""),
                    pred_action=pred_action,
                    action_correct=action_correct,
                    answer=answer,
                    justification="Deleted-content reconstruction is blocked at the action level and does not surface stale memory.",
                )
            question_text = str(checkpoint.get("query_text") or "")
            answer_text = answer.lower()
            if (
                "superseded_state_correction" not in selected
                and pred_action in {"answer", "answer_redacted"}
                and action_correct
                and (
                    any(token in question_text.lower() for token in ["current", "latest", "right now", "updated"])
                    or any(token in answer_text for token in ["current ", "currently ", "latest ", "right now"])
                )
            ):
                selected["superseded_state_correction"] = CaseStudyRow(
                    category="superseded_state_correction",
                    domain=domain,
                    checkpoint_id=checkpoint_id,
                    question=question_text,
                    pred_action=pred_action,
                    action_correct=action_correct,
                    answer=answer,
                    justification="The response privileges the current state rather than replaying an older superseded state.",
                )
            if (
                "no_memory_vs_refusal" not in selected
                and pred_action in {"refuse", "no_memory"}
                and action_correct
                and not bool(row.get("privacy_e2e_leak"))
                and not bool(row.get("deletion_e2e_leak"))
            ):
                selected["no_memory_vs_refusal"] = CaseStudyRow(
                    category="no_memory_vs_refusal",
                    domain=domain,
                    checkpoint_id=checkpoint_id,
                    question=question_text,
                    pred_action=pred_action,
                    action_correct=action_correct,
                    answer=answer,
                    justification="The framework separates inaccessible-but-existing information from genuinely unavailable memory through action selection.",
                )
    return list(selected.values())


def _select_case_studies_from_runs(run_dirs: list[Path]) -> list[CaseStudyRow]:
    merged: dict[str, CaseStudyRow] = {}
    for run_dir in run_dirs:
        for row in _select_case_studies(run_dir):
            merged.setdefault(row.category, row)
    return list(merged.values())


def _write_jsonl(path: Path, rows: list[CaseStudyRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row.__dict__, ensure_ascii=False))
            handle.write("\n")


def _write_tex(path: Path, rows: list[CaseStudyRow]) -> None:
    lines = [
        "% Auto-generated by scripts/export_case_study.py",
        "\\begin{tabular}{lllll}",
        "\\toprule",
        "Category & Domain & Checkpoint & Action & Note \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row.category.replace('_', ' ')} & {row.domain} & {row.checkpoint_id.replace('_', '\\_')} & {row.pred_action} & {_escape_tex(row.justification)} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_report(path: Path, run_dir: Path, rows: list[CaseStudyRow], output_dir: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Phase 13 Case Study Report",
        "",
        f"Timestamp: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Scope",
        "",
        f"Run directory: `{run_dir}`",
        "",
        "## Exported Cases",
        "",
        "| Category | Domain | Checkpoint | Action |",
        "| --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(f"| {row.category} | {row.domain} | {row.checkpoint_id} | {row.pred_action} |")
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- JSONL: `{output_dir / 'best_cases.jsonl'}`",
            f"- TeX: `{output_dir / 'case_study_table.tex'}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _escape_tex(text: str) -> str:
    return text.replace("&", "\\&").replace("%", "\\%").replace("_", "\\_")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export representative Gov-Mem case studies from an official GateMem run.")
    parser.add_argument("--run_dir", action="append", required=True)
    parser.add_argument("--output_dir", default="outputs/case_study")
    args = parser.parse_args()

    run_dirs = [Path(item).resolve() for item in args.run_dir]
    output_dir = (ROOT / args.output_dir).resolve()
    rows = _select_case_studies_from_runs(run_dirs)
    jsonl_path = output_dir / "best_cases.jsonl"
    tex_path = output_dir / "case_study_table.tex"
    report_path = ROOT / "outputs" / "reports" / "phase13_case_study_report.md"
    _write_jsonl(jsonl_path, rows)
    _write_tex(tex_path, rows)
    _write_report(report_path, run_dirs[0], rows, output_dir)
    print(f"cases={len(rows)}")
    print(f"jsonl={jsonl_path}")
    print(f"tex={tex_path}")
    print(f"report={report_path}")


if __name__ == "__main__":
    main()
