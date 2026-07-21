#!/usr/bin/env python3
"""Structural checks for source-grounded, episode-local relationship ledgers."""

from __future__ import annotations

from copy import deepcopy

from gov_mem.data.schema import RetrievedEvidence
from gov_mem.backbones.rag_policy_amem import RAGPolicyAMemBackbone
from gov_mem.backbones.rag_policy import classify_chunk_policy
from gov_mem.governance_runtime.access import build_principal
from gov_mem.governance_runtime.principal_relation_ledger import (
    build_principal_relation_ledger,
    ledger_relation_for_principal,
)
from gov_mem.graph.graph_builder import GovernedGraphBuilder


class _FakeLLM:
    def __init__(self, response):
        self.response = response

    def is_available(self):
        return True

    def chat_json(self, **_kwargs):
        return deepcopy(self.response)


class _RepairLLM:
    def __init__(self):
        self.calls = 0

    def is_available(self):
        return True

    def chat_json(self, **_kwargs):
        self.calls += 1
        if self.calls == 1:
            return _record(span="Taylor is my parent", relation="family", evidence_kind="explicit_authorization")
        return _record(span="Taylor is my parent", relation="family", evidence_kind="explicit_relationship")


class _SupportVerifierLLM:
    def __init__(self):
        self.calls = 0

    def is_available(self):
        return True

    def chat_json(self, **_kwargs):
        self.calls += 1
        if self.calls == 1:
            return _record(span="Taylor is my parent", relation="family", evidence_kind="unspecified")
        if self.calls == 2:
            return {}
        return {"supports": [{
            "message_id": "m1", "source_span": "Taylor is my parent", "evidence_kind": "family_relationship",
        }]}


class _TargetedRecoveryLLM:
    def __init__(self):
        self.calls = 0

    def is_available(self):
        return True

    def chat_json(self, **kwargs):
        self.calls += 1
        if self.calls == 1:
            return _record(relation="unknown", status="unknown", span="")
        assert "Selected evidence" in kwargs["user_prompt"], kwargs["user_prompt"]
        return _record()


def _messages():
    return [
        {
            "message_id": "m1",
            "speaker_id": "person_owner",
            "speaker_role": "member",
            "text": "Taylor is my parent and may receive arrival logistics only.",
        },
        {
            "message_id": "m2",
            "speaker_id": "person_requester",
            "speaker_role": "visitor",
            "text": "I will only coordinate the arrival.",
        },
    ]


def _ledger(response, requester="person_requester"):
    return build_principal_relation_ledger(
        messages=_messages(),
        requester_id=requester,
        requester_role="visitor",
        principal_catalog=[
            {"principal_id": "person_owner", "display_name": "Jordan", "role": "member"},
            {"principal_id": "person_requester", "display_name": "Taylor", "role": "visitor"},
            {"principal_id": "person_other", "display_name": "Morgan", "role": "member"},
        ],
        question="What arrival details can Taylor receive?",
        llm_client=_FakeLLM(response),
        model_name="fake",
    )


def _record(*, relation="family", status="proven", span="Taylor is my parent", owner="person_owner", evidence_kind="explicit_relationship"):
    return {
        "relations": [{
            "subject_id": "person_requester",
            "owner_id": owner,
            "relation": relation,
            "relation_label": "parent",
            "status": status,
            "authorization_status": "active",
            "direction": "requester_to_owner",
            "supports": [{"message_id": "m1", "source_span": span, "evidence_kind": evidence_kind}],
        }]
    }


def main() -> None:
    owner_self_base = {
        "requester_id": "person_owner",
        "owner_id": None,
        "effective_relation": "unknown",
        "effective_status": "unknown",
        "records": [],
        "resolution_trace": {"prior": "preserved"},
    }
    owner_self_info = {
        "owner_by_message_id": {"m1": "person_owner"},
        "records": [{
            "message_id": "m1", "information_owner_id": "person_owner", "status": "proven",
            "supports": [{
                "message_id": "m1", "source_span": "Taylor is my parent",
                "evidence_kind": "explicit_data_subject",
            }],
        }],
    }
    owner_self = RAGPolicyAMemBackbone._attest_owner_self_relation(
        ledger=owner_self_base, requester_id="person_owner", owner_candidates={"person_owner"},
        information_owner_ledger=owner_self_info, selected_fact_ids={"m1"},
    )
    assert ledger_relation_for_principal(owner_self) == "owner", owner_self
    assert owner_self["owner_id"] == "person_owner", owner_self
    assert owner_self["records"][-1]["supports"][0]["source_span"] == "Taylor is my parent", owner_self
    assert owner_self["resolution_trace"]["prior"] == "preserved", owner_self

    ambiguous_self = RAGPolicyAMemBackbone._attest_owner_self_relation(
        ledger=owner_self_base, requester_id="person_owner",
        owner_candidates={"person_owner", "person_other"}, information_owner_ledger=owner_self_info,
        selected_fact_ids={"m1"},
    )
    assert ambiguous_self is owner_self_base, ambiguous_self
    mismatched_self = RAGPolicyAMemBackbone._attest_owner_self_relation(
        ledger=owner_self_base, requester_id="person_requester", owner_candidates={"person_owner"},
        information_owner_ledger=owner_self_info, selected_fact_ids={"m1"},
    )
    assert mismatched_self is owner_self_base, mismatched_self
    existing_non_owner = dict(owner_self_base, owner_id="person_owner", effective_relation="family", effective_status="proven")
    preserved_non_owner = RAGPolicyAMemBackbone._attest_owner_self_relation(
        ledger=existing_non_owner, requester_id="person_owner", owner_candidates={"person_owner"},
        information_owner_ledger=owner_self_info, selected_fact_ids={"m1"},
    )
    assert preserved_non_owner is existing_non_owner, preserved_non_owner

    family = _ledger(_record())
    assert family["owner_id"] == "person_owner", family
    assert ledger_relation_for_principal(family) == "family", family
    assert family["records"][0]["supports"][0]["source_span"] == "Taylor is my parent"
    owner_closed = build_principal_relation_ledger(
        messages=_messages(), requester_id="person_requester", requester_role="visitor",
        principal_catalog=[
            {"principal_id": "person_owner", "display_name": "Jordan", "role": "member"},
            {"principal_id": "person_requester", "display_name": "Taylor", "role": "visitor"},
            {"principal_id": "person_other", "display_name": "Morgan", "role": "member"},
        ],
        question="What arrival details can Taylor receive?", llm_client=_FakeLLM(_record()), model_name="fake",
        candidate_owner_ids={"person_other"},
    )
    assert ledger_relation_for_principal(owner_closed) == "unknown", owner_closed
    assert owner_closed["resolution_trace"]["candidate_owner_ids"] == ["person_other"], owner_closed
    targeted_recovery = build_principal_relation_ledger(
        messages=_messages(), requester_id="person_requester", requester_role="visitor",
        principal_catalog=None, question="What arrival details can Taylor receive?",
        llm_client=_TargetedRecoveryLLM(), model_name="fake", candidate_owner_ids={"person_owner"},
        relation_evidence_message_ids={"m1"},
    )
    assert ledger_relation_for_principal(targeted_recovery) == "family", targeted_recovery
    assert targeted_recovery["resolution_trace"]["targeted_relation_recovery"]["attempted"], targeted_recovery
    relation_graph = GovernedGraphBuilder().build(
        graph_id="relation-smoke", instance_id="relation-smoke", atoms=[], principal_relation_ledger=family
    )
    relation_node_id = "relation::person_requester::person_owner::family"
    assert any(node.node_id == relation_node_id for node in relation_graph.nodes)
    assert {
        ("has_relation", "principal::person_requester", relation_node_id),
        ("relation_owner", relation_node_id, "principal::person_owner"),
    }.issubset({(edge.edge_type, edge.source_id, edge.target_id) for edge in relation_graph.edges})

    provider_envelope = _ledger({
        "requester_owner_relation": {
            "requester_id": "person_requester",
            "target_owner_id": "person_owner",
            "relation_type": "family_member",
            "relation_label": "parent",
            "relation_status": "proven",
            "access_status": "active",
            "direction": "requester_to_owner",
            "evidence": [{"turn_id": "m1", "span": "Taylor is my parent", "evidence_kind": "explicit_relationship"}],
        }
    })
    assert ledger_relation_for_principal(provider_envelope) == "family", provider_envelope

    quote_envelope = _ledger({
        "relations": [{
            "subject_id": "person_requester", "owner_id": "person_owner", "relation": "family",
            "status": "proven", "direction": "requester_to_owner",
            "supports": {"source_message_id": "m1", "quote": "Taylor is my parent", "kind": "family_relationship"},
        }]
    })
    assert ledger_relation_for_principal(quote_envelope) == "family", quote_envelope

    unsupported_staff = _ledger(_record(relation="authorized_staff", span=""))
    assert ledger_relation_for_principal(unsupported_staff) == "unknown", unsupported_staff

    bad_span = _ledger(_record(span="Taylor is a secret sibling"))
    assert ledger_relation_for_principal(bad_span) == "unknown", bad_span

    repaired = build_principal_relation_ledger(
        messages=_messages(), requester_id="person_requester", requester_role="visitor",
        principal_catalog=None, question="What arrival details can Taylor receive?",
        llm_client=_RepairLLM(), model_name="fake",
    )
    assert ledger_relation_for_principal(repaired) == "family", repaired
    assert repaired["resolution_trace"]["relation_anchor_repair"]["attempted"], repaired

    verified = build_principal_relation_ledger(
        messages=_messages(), requester_id="person_requester", requester_role="visitor",
        principal_catalog=None, question="What arrival details can Taylor receive?",
        llm_client=_SupportVerifierLLM(), model_name="fake",
    )
    assert ledger_relation_for_principal(verified) == "family", verified
    verification = verified["resolution_trace"]["relation_anchor_repair"]["support_verification"]
    assert verification["attempted"] and verification["validated_record_count"] == 1, verified

    self_and_family = build_principal_relation_ledger(
        messages=_messages(), requester_id="person_requester", requester_role="visitor",
        principal_catalog=None, question="What arrival details can Taylor receive?",
        llm_client=_FakeLLM({"relations": [
            _record()["relations"][0],
            {
                "subject_id": "person_requester", "owner_id": "person_requester", "relation": "owner",
                "status": "proven", "direction": "requester_to_owner", "supports": [],
            },
        ]}), model_name="fake",
    )
    assert self_and_family["owner_id"] == "person_owner", self_and_family
    assert ledger_relation_for_principal(self_and_family) == "family", self_and_family

    revoked = _ledger(_record(status="revoked"))
    assert ledger_relation_for_principal(revoked) == "unknown", revoked

    self_identity = build_principal_relation_ledger(
        messages=_messages(),
        requester_id="person_owner",
        requester_role="member",
        principal_catalog=None,
        question="What is my current plan?",
        llm_client=_FakeLLM({"relations": [{
            "subject_id": "person_owner",
            "owner_id": "person_owner",
            "relation": "owner",
            "relation_label": "self",
            "status": "proven",
            "authorization_status": "active",
            "direction": "requester_to_owner",
            "supports": [],
        }]}),
        model_name="fake",
    )
    assert ledger_relation_for_principal(self_identity) == "owner", self_identity

    offline_self_candidate = build_principal_relation_ledger(
        messages=_messages(),
        requester_id="person_owner",
        requester_role="member",
        principal_catalog=None,
        question="What is my current plan?",
        llm_client=None,
        model_name="fake",
        fallback_owner_id="person_owner",
    )
    assert ledger_relation_for_principal(offline_self_candidate) == "unknown", offline_self_candidate

    renamed = _ledger(_record(span="Taylor is my parent"))
    assert ledger_relation_for_principal(renamed) == ledger_relation_for_principal(family)

    unproven_principal = build_principal(
        requester_id="person_requester",
        requester_role="clinician",
        owner_user_id="person_owner",
    )
    row = RetrievedEvidence(
        memory_id="e1", content="Arrival is at 09:00.", score=1.0,
        retrieval_source="smoke", reason="smoke",
    )
    decision = classify_chunk_policy(row, unproven_principal)
    assert not decision["allowed_for_requester"], decision
    assert decision["policy_reason"] == "requester_owner_relation_unproven", decision

    print("principal_relation_ledger_smoke=PASS")


if __name__ == "__main__":
    main()
