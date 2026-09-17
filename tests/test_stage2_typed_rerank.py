from gov_mem.backbones.stage2_typed_rerank import (
    _candidate_matches_request_slot,
    build_summary_only_evidence,
    deletion_gate_reason,
    route_query,
)
from gov_mem.data.schema import RetrievedEvidence


def _analysis(*, fields, temporal="current", safety=None, lifecycle=None):
    result = {
        "semantic_compiler_contract": True,
        "fields": fields,
        "temporal": {"orientation": temporal},
    }
    if safety is not None:
        result["safety"] = safety
    if lifecycle is not None:
        result["lifecycle"] = lifecycle
    return result


def test_route_is_driven_by_open_vocabulary_contract_value_types():
    analysis = _analysis(fields=[{
        "slot_id": "qslot_0", "name": "arrival point", "value_type": "location",
    }])
    assert route_query("Where should the delivery arrive?", query_analysis=analysis) == (
        "typed_scalar", ["location"],
    )


def test_multi_slot_contract_uses_mixed_route_without_aliases():
    analysis = _analysis(fields=[
        {"slot_id": "qslot_0", "name": "arrival point", "value_type": "location"},
        {"slot_id": "qslot_1", "name": "arrival time", "value_type": "time"},
    ])
    assert route_query("When and where is the delivery?", query_analysis=analysis) == (
        "mixed", ["date_time", "location"],
    )


def test_candidate_match_requires_verified_semantic_atom_slot():
    grounded = RetrievedEvidence(
        memory_id="m0",
        content="The arrival point is west gate.",
        score=1.0,
        retrieval_source="dense",
        reason="test",
        metadata={"semantic_compiler_atoms": [{"slot_id": "qslot_0"}]},
    )
    assert _candidate_matches_request_slot(
        grounded,
        "unused",
        field_spec={"name": "arrival point", "value_type": "location", "slot_id": "qslot_0"},
    )
    assert not _candidate_matches_request_slot(
        RetrievedEvidence(
            memory_id="m1",
            content="The arrival point is west gate.",
            score=1.0,
            retrieval_source="dense",
            reason="test",
        ),
        "unused",
        field_spec={"name": "arrival point", "value_type": "location", "slot_id": "qslot_0"},
    )


def test_deletion_gate_requires_structured_lifecycle_and_safety_contract():
    analysis = _analysis(
        fields=[{"slot_id": "qslot_0", "name": "previous identifier", "value_type": "identifier"}],
        temporal="historical",
        safety={
            "available": True,
            "sensitive": True,
            "exact": True,
            "delivery_mode": "protected_fact",
        },
        lifecycle={"explicit_deleted": True},
    )
    assert deletion_gate_reason("What was it?", query_analysis=analysis)
    assert deletion_gate_reason("What was it?", query_analysis=None) is None


def test_summary_helper_cannot_create_lexical_projection():
    evidence = [RetrievedEvidence(
        memory_id="m0", content="Observable source.", score=1.0,
        retrieval_source="dense", reason="test",
    )]
    assert build_summary_only_evidence(evidence=evidence) == evidence
