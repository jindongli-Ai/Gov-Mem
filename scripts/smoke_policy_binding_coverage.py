#!/usr/bin/env python3
"""Structural policy-to-slot coverage checks without benchmark vocabulary."""

from __future__ import annotations

from gov_mem.governance_runtime.policy_frames import compile_policy_frames
from gov_mem.memory.governed_atom import GovernedMemoryAtom


def _atom(atom_id, atom_type, text, slots, owner="owner"):
    return GovernedMemoryAtom(
        atom_id=atom_id,
        atom_type=atom_type,
        text=text,
        slots=slots,
        owner_id=owner,
        subject_id="record",
        speaker_id=owner,
        source_turn=1,
        timestamp="2026-01-01T00:00:00",
        lifecycle="active",
        sensitivity="private",
        access_scope=[],
        related_entities=[],
        provenance={"source_memory_id": atom_id, "evidence_span": text},
        confidence=1.0,
    )


class _CoverageLLM:
    def __init__(self):
        self.calls = 0

    def is_available(self):
        return True

    def chat_json(self, **_kwargs):
        self.calls += 1
        if self.calls == 1:
            return {"bindings": [{
                "atom_id": "policy",
                "effect": "allow",
                "scopes": ["collaborator"],
                "slots": ["field_a"],
                "support_spans": ["The collaborator may receive this record's operational fields."],
            }]}
        return {"bindings": [{
            "atom_id": "policy",
            "effect": "allow",
            "scopes": ["collaborator"],
            "slots": ["field_a", "field_b"],
            "support_spans": ["The collaborator may receive this record's operational fields."],
        }]}


def main() -> None:
    policy = _atom(
        "policy", "permission_atom",
        "The collaborator may receive this record's operational fields.", {},
    )
    fact = _atom("fact", "fact_atom", "A record contains first and second fields.", {"field_a": "first", "field_b": "second"})
    result = compile_policy_frames(atoms=[policy, fact], llm_client=_CoverageLLM(), model_name="fake")
    compiled = next(atom for atom in result if atom.atom_id == "policy")
    binding = compiled.provenance.get("policy_binding") or {}
    assert binding.get("slots") == ["field_a", "field_b"], binding
    assert binding.get("source") == "llm_policy_coverage_audit", binding
    print("policy_binding_coverage_smoke=PASS")


if __name__ == "__main__":
    main()
