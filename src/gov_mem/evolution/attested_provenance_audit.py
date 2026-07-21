"""Offline validation for source-attested typed graph evidence."""

from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any

from gov_mem.utils.io import read_jsonl, write_json


def audit_run(*, run_dir: str | Path, output_path: str | Path | None = None) -> dict[str, Any]:
    """Audit only fact atoms actually selected for graph realization."""
    root = Path(run_dir)
    per_domain: dict[str, dict[str, Any]] = {}
    violations: list[dict[str, str]] = []
    for domain_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        atoms_path = domain_dir / "debug" / "governed_atoms.jsonl"
        cases_dir = domain_dir / "debug_cases" / "checkpoint_benchmark"
        if not atoms_path.exists() or not cases_dir.exists():
            continue
        atoms_by_id = {
            str(atom.get("atom_id") or ""): atom
            for atom in read_jsonl(atoms_path)
            if str(atom.get("atom_id") or "")
        }
        realized_values: dict[str, set[str]] = defaultdict(set)
        for case_path in cases_dir.glob("*.json"):
            try:
                case = json.loads(case_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            certificate = dict(case.get("graph_authorization_certificate") or {})
            if not certificate.get("authorized"):
                continue
            for realization in list(certificate.get("realizations") or []):
                if not isinstance(realization, dict):
                    continue
                atom_id = str(realization.get("source_atom_id") or "")
                value = str(realization.get("value") or "").strip()
                if atom_id and value:
                    realized_values[atom_id].add(value)
        domain_counts: Counter[str] = Counter()
        for atom_id, values in realized_values.items():
            atom = atoms_by_id.get(atom_id)
            if atom is None:
                violations.append({"domain": domain_dir.name, "atom_id": atom_id, "reason": "realization_atom_missing"})
                continue
            slots = dict(atom.get("slots") or {})
            domain_counts["realized_atoms"] += 1
            text = str(atom.get("text") or "")
            span = str((atom.get("provenance") or {}).get("evidence_span") or "")
            if not span:
                violations.append({"domain": domain_dir.name, "atom_id": atom_id, "reason": "missing_evidence_span"})
                continue
            if span != text:
                violations.append({"domain": domain_dir.name, "atom_id": atom_id, "reason": "atom_text_not_attested_span"})
                continue
            missing_values = [value for value in values if value.lower() not in span.lower()]
            if missing_values:
                violations.append({
                    "domain": domain_dir.name,
                    "atom_id": atom_id,
                    "reason": "slot_value_outside_evidence_span",
                })
                continue
            domain_counts["attested_realized_atoms"] += 1
        per_domain[domain_dir.name] = dict(domain_counts)
    payload = {
        "run_dir": str(root.resolve()),
        "passed": not violations,
        "per_domain": per_domain,
        "violations": violations,
        "realized_atom_count": sum(int(row.get("realized_atoms") or 0) for row in per_domain.values()),
        "attested_realized_atom_count": sum(int(row.get("attested_realized_atoms") or 0) for row in per_domain.values()),
    }
    if output_path is not None:
        write_json(output_path, payload)
    return payload
