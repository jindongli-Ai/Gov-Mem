#!/usr/bin/env python3
"""Closed-set source selection and information-owner attribution checks."""

from __future__ import annotations

from gov_mem.data.schema import MemoryInstance
from gov_mem.governance_runtime.information_owner_ledger import build_information_owner_ledger
from gov_mem.governance_runtime.utility_source_locator import (
    locate_authorization_context_messages,
    locate_utility_source_messages,
)
from gov_mem.memory.amem_memory import AtomicMemory
from gov_mem.memory.governed_atom import adapt_atomic_memories_to_governed_atoms


class _OwnerLLM:
    def is_available(self):
        return True

    def chat_json(self, **_kwargs):
        return {"information_owners": [{
            "message_id": "m-fact",
            "information_owner_id": "record-owner",
            "status": "proven",
            "supports": [{
                "message_id": "m-context",
                "source_span": "The record belongs to the named owner.",
                "evidence_kind": "contextual_case_assignment",
            }],
        }]}


class _LocatorLLM:
    def is_available(self):
        return True

    def chat_json(self, **_kwargs):
        return {"result": {"utility_sources": [{
            "source_message_id": "m-fact", "source_span": "The operational value is ready."
        }]}}


class _TransientLocatorLLM:
    def __init__(self):
        self.calls = 0

    def is_available(self):
        return True

    def chat_json(self, **_kwargs):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("transient")
        return _LocatorLLM().chat_json()


class _AuthorizationContextLLM:
    def is_available(self):
        return True

    def chat_json(self, **_kwargs):
        return {"sources": [{
            "message_id": "m-context", "support_span": "The record belongs to the named owner."
        }]}


class _AuthorizationContextRepairLLM:
    def __init__(self):
        self.calls = 0

    def is_available(self):
        return True

    def chat_json(self, **_kwargs):
        self.calls += 1
        if self.calls == 1:
            return {"sources": [{
                "message_id": "m-fact", "support_span": "The operational value is ready."
            }]}
        return {"sources": [{
            "message_id": "m-operation", "support_span": "I updated the operational value."
        }]}


class _OwnerRepairLLM:
    def __init__(self):
        self.calls = 0

    def is_available(self):
        return True

    def chat_json(self, **_kwargs):
        self.calls += 1
        if self.calls == 1:
            return {"information_owners": []}
        return _OwnerLLM().chat_json()


class _TransientOwnerLLM:
    def __init__(self):
        self.calls = 0

    def is_available(self):
        return True

    def chat_json(self, **_kwargs):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("transient")
        return _OwnerLLM().chat_json()


class _DoubleTransientOwnerLLM:
    def __init__(self):
        self.calls = 0

    def is_available(self):
        return True

    def chat_json(self, **_kwargs):
        self.calls += 1
        if self.calls < 3:
            return {"information_owners": []}
        return _OwnerLLM().chat_json()


def main() -> None:
    messages = [
        {"message_id": "m-context", "speaker_id": "coordinator", "text": "The record belongs to the named owner."},
        {"message_id": "m-operation", "speaker_id": "coordinator", "text": "I updated the operational value."},
        {"message_id": "m-fact", "speaker_id": "coordinator", "text": "The operational value is ready."},
    ]
    catalog = [
        {"principal_id": "record-owner", "display_name": "Named Owner", "role": "member"},
        {"principal_id": "coordinator", "display_name": "Coordinator", "role": "staff"},
    ]
    ledger = build_information_owner_ledger(
        messages=messages, principal_catalog=catalog, candidate_message_ids={"m-fact"}, required_message_ids={"m-fact"},
        llm_client=_OwnerLLM(), model_name="fake",
    )
    assert ledger["owner_by_message_id"] == {"m-fact": "record-owner"}, ledger
    repaired_ledger = build_information_owner_ledger(
        messages=messages, principal_catalog=catalog, candidate_message_ids={"m-fact"}, required_message_ids={"m-fact"},
        llm_client=_OwnerRepairLLM(), model_name="fake",
    )
    assert repaired_ledger["owner_by_message_id"] == {"m-fact": "record-owner"}, repaired_ledger
    assert repaired_ledger["resolution_trace"]["priority_owner_repair"]["per_message_attempts"][0]["validated_record_count"] == 1
    transient_ledger = build_information_owner_ledger(
        messages=messages, principal_catalog=catalog, candidate_message_ids={"m-fact"}, required_message_ids={"m-fact"},
        llm_client=_TransientOwnerLLM(), model_name="fake",
    )
    assert transient_ledger["owner_by_message_id"] == {"m-fact": "record-owner"}, transient_ledger
    assert transient_ledger["resolution_trace"]["initial_request_error"] == "information_owner_unavailable:RuntimeError"
    double_transient_ledger = build_information_owner_ledger(
        messages=messages, principal_catalog=catalog, candidate_message_ids={"m-fact"}, required_message_ids={"m-fact"},
        llm_client=_DoubleTransientOwnerLLM(), model_name="fake",
    )
    assert double_transient_ledger["owner_by_message_id"] == {"m-fact": "record-owner"}, double_transient_ledger
    assert double_transient_ledger["resolution_trace"]["priority_owner_repair"]["per_message_attempts"][0]["attempt_count"] == 2
    locator = locate_utility_source_messages(
        question="What is the operational value?", semantic_spec={"requested_attributes": ["operational_value"]},
        messages=messages, llm_client=_LocatorLLM(), model_name="fake",
    )
    assert locator["selected_fact_message_ids"] == ["m-fact"], locator
    assert locator["source_message_ids"] == ["m-operation", "m-fact"], locator
    recovered_locator = locate_utility_source_messages(
        question="What is the operational value?", semantic_spec={"requested_attributes": ["operational_value"]},
        messages=messages, llm_client=_TransientLocatorLLM(), model_name="fake",
    )
    assert recovered_locator["selected_fact_message_ids"] == ["m-fact"], recovered_locator
    authorization_context = locate_authorization_context_messages(
        question="What is the operational value?", requester_id="coordinator", selected_fact_message_ids={"m-fact"},
        messages=messages, llm_client=_AuthorizationContextLLM(), model_name="fake",
    )
    assert authorization_context["source_message_ids"] == ["m-context"], authorization_context
    recovered_authorization_context = locate_authorization_context_messages(
        question="What is the operational value?", requester_id="coordinator", selected_fact_message_ids={"m-fact"},
        messages=messages, llm_client=_AuthorizationContextRepairLLM(), model_name="fake",
    )
    assert recovered_authorization_context["source_message_ids"] == ["m-fact", "m-operation"], recovered_authorization_context
    assert recovered_authorization_context["diagnostics"]["operation_recovery"]["added_count"] == 1
    memory = AtomicMemory(
        memory_id="fact", instance_id="smoke", owner_user="coordinator", memory_type="fact",
        content="The operational value is ready.", entities=[], slots={"operational_value": "ready"},
        source_message_ids=["m-fact"], timestamp=None, lifecycle_status="active", access_tags={}, confidence=1.0,
    )
    instance = MemoryInstance("smoke", "synthetic", None, messages, "What is the operational value?", None, None, None)
    atoms = adapt_atomic_memories_to_governed_atoms(
        instance=instance, atomic_memories=[memory], information_owner_by_message_id=ledger["owner_by_message_id"],
    )
    assert atoms[0].owner_id == "record-owner" and atoms[0].speaker_id == "coordinator", atoms[0]
    unknown_atoms = adapt_atomic_memories_to_governed_atoms(
        instance=instance, atomic_memories=[memory], information_owner_by_message_id={},
    )
    assert unknown_atoms[0].owner_id is None, unknown_atoms[0]
    print("information_owner_and_utility_locator_smoke=PASS")


if __name__ == "__main__":
    main()
