#!/usr/bin/env python3
"""Policy/mixed source roles cannot materialize answerable graph slots."""

from __future__ import annotations

from gov_mem.graph.graph_builder import GovernedGraphBuilder
from gov_mem.governance_runtime.source_role_ledger import classify_source_roles
from gov_mem.memory.governed_atom import GovernedMemoryAtom


class _RoleLLM:
    def is_available(self):
        return True

    def chat_json(self, **_kwargs):
        return {"source_roles": [{
            "message_id": "m-policy", "role": "policy", "support_span": "A delegate may receive the field."
        }]}


class _BroadAuditLLM:
    def __init__(self):
        self.calls = 0

    def is_available(self):
        return True

    def chat_json(self, **_kwargs):
        self.calls += 1
        if self.calls == 1:
            return {"source_roles": [{
                "message_id": "m-fact", "role": "factual",
                "support_span": "The current operational record reports a revised service value for the assigned case."
            }]}
        return {"source_roles": [{
            "message_id": "m-fact", "role": "policy",
            "support_span": "The current operational record reports a revised service value for the assigned case."
        }]}


class _SelectedFactVerifierLLM:
    def __init__(self):
        self.calls = 0

    def is_available(self):
        return True

    def chat_json(self, **_kwargs):
        self.calls += 1
        if self.calls == 1:
            return {"source_roles": [{
                "message_id": "m-fact", "role": "policy",
                "support_span": "The operational value is ready for the active record."
            }]}
        return {"source_roles": [{
            "message_id": "m-fact", "role": "factual",
            "support_span": "The operational value is ready for the active record."
        }]}


class _StateUpdateLLM:
    def is_available(self):
        return True

    def chat_json(self, **kwargs):
        assert "old value was deleted" in kwargs["system_prompt"], kwargs["system_prompt"]
        return {"source_roles": [{
            "message_id": "m-state", "role": "factual",
            "support_span": "The current record uses channel-A only and the former channel was deleted.",
        }]}


def _atom(atom_id, text, slots, message_id, source_role=""):
    return GovernedMemoryAtom(
        atom_id=atom_id, atom_type="event_atom", text=text, slots=slots,
        owner_id="owner", subject_id="record", speaker_id="speaker", source_turn=1,
        timestamp="2026-01-01T00:00:00", lifecycle="active", sensitivity="private",
        access_scope=[], related_entities=[], confidence=1.0,
        provenance={"source_memory_id": atom_id, "source_message_ids": [message_id], "source_role": source_role},
    )


def main() -> None:
    ledger = classify_source_roles(
        messages=[{"message_id": "m-policy", "speaker_id": "owner", "text": "A delegate may receive the field."}],
        candidate_message_ids={"m-policy"}, required_message_ids={"m-policy"}, llm_client=_RoleLLM(), model_name="fake",
    )
    assert ledger["role_by_message_id"] == {"m-policy": "policy"}, ledger
    factual_ledger = classify_source_roles(
        messages=[{
            "message_id": "m-fact", "speaker_id": "reporter",
            "text": "The current operational record reports a revised service value for the assigned case.",
        }],
        candidate_message_ids={"m-fact"}, required_message_ids={"m-fact"},
        llm_client=_BroadAuditLLM(), model_name="fake",
    )
    assert factual_ledger["role_by_message_id"] == {"m-fact": "factual"}, factual_ledger
    selected_fact_ledger = classify_source_roles(
        messages=[{
            "message_id": "m-fact", "speaker_id": "reporter",
            "text": "The operational value is ready for the active record.",
        }],
        candidate_message_ids={"m-fact"}, required_message_ids={"m-fact"},
        llm_client=_SelectedFactVerifierLLM(), model_name="fake",
    )
    assert selected_fact_ledger["role_by_message_id"] == {"m-fact": "factual"}, selected_fact_ledger
    assert selected_fact_ledger["resolution_trace"]["required_fact_verification"]["attempted"], selected_fact_ledger
    state_ledger = classify_source_roles(
        messages=[{
            "message_id": "m-state", "speaker_id": "reporter",
            "text": "The current record uses channel-A only and the former channel was deleted.",
        }],
        candidate_message_ids={"m-state"}, required_message_ids={"m-state"},
        llm_client=_StateUpdateLLM(), model_name="fake",
    )
    assert state_ledger["role_by_message_id"] == {"m-state": "factual"}, state_ledger
    graph = GovernedGraphBuilder().build(
        graph_id="role-separation", instance_id="role-separation",
        atoms=[
            _atom("policy-derived", "A delegate may receive the field.", {"field": "policy"}, "m-policy", "policy"),
            _atom("fact", "The field value is ready.", {"field": "ready"}, "m-fact", "factual"),
        ],
    )
    values = {node.attributes.get("slot_value") for node in graph.nodes if node.node_type == "SlotNode"}
    assert values == {"ready"}, values
    print("source_role_separation_smoke=PASS")


if __name__ == "__main__":
    main()
