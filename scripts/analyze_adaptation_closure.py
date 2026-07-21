from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _iter_adaptation_rows(run_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    roots: list[Path] = []
    direct_root = run_dir / "debug" / "adaptation"
    if direct_root.exists():
        roots.append(direct_root)
    for root in sorted(run_dir.glob("*/debug/adaptation")):
        if root.exists():
            roots.append(root)
    seen: set[Path] = set()
    for root in roots:
        resolved = root.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        for path in sorted(root.glob("*.json")):
            payload = _read_json(path)
            payload["_path"] = str(path)
            payload["_audit_root"] = str(root)
            if not payload.get("domain"):
                try:
                    payload["domain"] = path.relative_to(run_dir).parts[0]
                except Exception:
                    payload["domain"] = "unknown"
            rows.append(payload)
    return rows


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    triggered = [row for row in rows if bool(row.get("adaptation_triggered"))]
    enabled = [row for row in rows if bool(row.get("adaptation_enabled"))]
    skill_counter: Counter[str] = Counter()
    pattern_counter: Counter[str] = Counter()
    field_counter: Counter[str] = Counter()
    domain_counter: Counter[str] = Counter()
    audit_roots = sorted({str(row.get("_audit_root") or "") for row in rows if str(row.get("_audit_root") or "").strip()})
    for row in triggered:
        domain_counter.update([str(row.get("domain") or "unknown")])
        skill_counter.update(str(item) for item in row.get("selected_skill_ids") or [] if str(item).strip())
        pattern_counter.update(str(item) for item in row.get("selected_experience_pattern_ids") or [] if str(item).strip())
        field_counter.update(str(item) for item in row.get("affected_decision_fields") or [] if str(item).strip())
    return {
        "n_instances": len(rows),
        "n_enabled": len(enabled),
        "n_triggered": len(triggered),
        "enabled_rate": round(len(enabled) / len(rows), 4) if rows else 0.0,
        "trigger_rate": round(len(triggered) / len(rows), 4) if rows else 0.0,
        "top_skills": skill_counter.most_common(10),
        "top_patterns": pattern_counter.most_common(10),
        "affected_fields": field_counter.most_common(10),
        "triggered_by_domain": domain_counter.most_common(),
        "audit_roots": audit_roots,
    }


def _write_md(path: Path, run_dir: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Runtime Adaptation Closure Report",
        "",
        f"- Run directory: `{run_dir}`",
        f"- Instances: `{summary['n_instances']}`",
        f"- Adaptation enabled: `{summary['n_enabled']}`",
        f"- Adaptation triggered: `{summary['n_triggered']}`",
        f"- Enable rate: `{summary['enabled_rate']:.4f}`",
        f"- Trigger rate: `{summary['trigger_rate']:.4f}`",
        "",
        "## Audit Roots",
        "",
    ]
    for root in summary["audit_roots"]:
        lines.append(f"- `{root}`")
    lines.extend([
        "",
        "## Top Skills",
        "",
        "| Skill | Count |",
        "| --- | ---: |",
    ])
    for name, count in summary["top_skills"]:
        lines.append(f"| {name} | {count} |")
    lines.extend([
        "",
        "## Top Experience Patterns",
        "",
        "| Pattern ID | Count |",
        "| --- | ---: |",
    ])
    for name, count in summary["top_patterns"]:
        lines.append(f"| {name} | {count} |")
    lines.extend([
        "",
        "## Affected Decision Fields",
        "",
        "| Field | Count |",
        "| --- | ---: |",
    ])
    for name, count in summary["affected_fields"]:
        lines.append(f"| {name} | {count} |")
    lines.extend([
        "",
        "## Triggered by Domain",
        "",
        "| Domain | Count |",
        "| --- | ---: |",
    ])
    for name, count in summary["triggered_by_domain"]:
        lines.append(f"| {name} | {count} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze the runtime adaptation closure of Gov-Mem.")
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_md", required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    rows = _iter_adaptation_rows(run_dir)
    summary = _summarize(rows)
    Path(args.output_json).write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_md(Path(args.output_md), run_dir, summary)
    print(f"rows={len(rows)}")
    print(f"triggered={summary['n_triggered']}")
    print(f"output_json={Path(args.output_json).resolve()}")
    print(f"output_md={Path(args.output_md).resolve()}")


if __name__ == "__main__":
    main()
