from gov_mem.backbones.rag_naive import _format_retrieved_memory, _semantic_context_boundary
from gov_mem.backbones.semantic_compiler import QuerySemanticContract, RequestedSlot
from gov_mem.data.schema import RetrievedEvidence
from gov_mem.backbones import symbolic_evidence, stage2_typed_rerank
from gov_mem.governance_runtime import access, evidence_frames
from gov_mem.query_semantics import (
    CURRENT_STATE_SLOT_ALIASES,
    CURRENT_STATE_DOMAIN_ALIASES,
    HOUSEHOLD_SLOT_ALIASES,
    HOUSEHOLD_DELIVERY_SLOT_ALIASES,
)


def _row(memory_id: str, text: str) -> RetrievedEvidence:
    return RetrievedEvidence(
        memory_id=memory_id,
        content=text,
        score=1.0,
        retrieval_source="dense",
        reason="test",
        source_message_ids=["t1"],
        metadata={
            "structured_record": {
                "record_type": "message",
                "message_id": "t1",
                "turn_id": "t1",
                "turn_index": 0,
                "text": text,
                "speaker": {"principal_id": "owner", "role": "owner"},
                "checkpoint": {"as_of_turn_id": "t1"},
            }
        },
    )


def _contract(*, protected: bool) -> QuerySemanticContract:
    return QuerySemanticContract(
        requested_slots=[
            RequestedSlot(
                slot_id="qslot_0",
                slot_name="requested value",
                sensitivity_possible=protected,
                authorization_relevant=protected,
            )
        ]
    )


def _atom(*, memory_id: str, span: str, slot_id: str = "qslot_0") -> dict:
    return {
        "atom_id": "atom_0",
        "slot_id": slot_id,
        "slot_name": "requested value",
        "value": "42",
        "source": {
            "chunk_id": memory_id,
            "turn_id": "t1",
            "message_id": "t1",
            "span": span,
        },
    }


def test_protected_unknown_auth_keeps_closed_evidence_available():
    rows = [
        _row("m1", "The requested value is 42."),
        _row("m2", "Unrelated private source text."),
    ]
    projected, audit = _semantic_context_boundary(
        evidence=rows,
        semantic_contract=_contract(protected=True),
        semantic_atoms=[_atom(memory_id="m1", span="The requested value is 42.")],
        symbolic_trace={"temporal_authorization": {"decision": "unknown"}},
    )
    assert [row.memory_id for row in projected] == ["m1"]
    assert "42" in projected[0].content
    assert audit["filtered_memory_ids"] == ["m2"]
    assert audit["enforcement_applied"] is not True


def test_protected_allow_keeps_grounded_source_span():
    row = _row("m1", "The requested value is 42.")
    projected, audit = _semantic_context_boundary(
        evidence=[row],
        semantic_contract=_contract(protected=True),
        semantic_atoms=[_atom(memory_id="m1", span="The requested value is 42.")],
        symbolic_trace={"temporal_authorization": {"decision": "allow"}},
    )
    assert projected[0].content == row.content
    assert audit["redacted_atom_ids"] == []


def test_ordinary_contract_does_not_project_context():
    row = _row("m1", "The requested value is 42.")
    projected, audit = _semantic_context_boundary(
        evidence=[row],
        semantic_contract=_contract(protected=False),
        semantic_atoms=[],
        symbolic_trace={},
    )
    assert projected == [row]
    assert audit["enabled"] is False


def test_unknown_authorization_keeps_closed_evidence_for_ordinary_slots():
    rows = [_row("m1", "The requested value is 42."), _row("m2", "Unrelated source text.")]
    projected, audit = _semantic_context_boundary(
        evidence=rows,
        semantic_contract=_contract(protected=False),
        semantic_atoms=[_atom(memory_id="m1", span="The requested value is 42")],
        symbolic_trace={"temporal_authorization": {"decision": "unknown"}},
    )
    assert [row.memory_id for row in projected] == ["m1", "m2"]
    assert audit["authorization_decision"] == "unknown"
    assert audit["filtered_memory_ids"] == []
    assert audit["mode"] == "unknown_authorization_passthrough"


def test_query_level_protected_safety_covers_under_marked_slots():
    rows = [_row("m1", "The requested value is 42.")]
    contract = QuerySemanticContract(
        requested_slots=[
            RequestedSlot(slot_id="qslot_0", slot_name="private value", sensitivity_possible=True),
            RequestedSlot(slot_id="qslot_1", slot_name="related value"),
        ]
    )
    projected, audit = _semantic_context_boundary(
        evidence=rows,
        semantic_contract=contract,
        semantic_atoms=[_atom(memory_id="m1", span="The requested value is 42.")],
        symbolic_trace={"temporal_authorization": {"decision": "unknown"}},
        query_analysis={
            "safety": {
                "sensitive": True,
                "exact": True,
                "delivery_mode": "protected_fact",
            }
        },
    )
    assert audit["query_level_protected_delivery"] is True
    assert audit["protected_slot_ids"] == ["qslot_0", "qslot_1"]
    assert "42" in projected[0].content


def test_unknown_protected_boundary_keeps_value_bearing_state_ledger():
    row = _row("m1", "The requested value is 42.")
    row = row.__class__(
        **{**row.__dict__, "metadata": {
            **row.metadata,
            "symbolic_state_ledger": {
                "fields": {"private value": {"value": "42"}},
                "version": "state-ledger-v2-semantic-atoms",
            },
        }}
    )
    projected, _ = _semantic_context_boundary(
        evidence=[row],
        semantic_contract=_contract(protected=True),
        semantic_atoms=[_atom(memory_id="m1", span="The requested value is 42.")],
        symbolic_trace={"temporal_authorization": {"decision": "unknown"}},
    )
    assert projected[0].metadata["symbolic_state_ledger"]["fields"]["private value"]["value"] == "42"


def test_unknown_protected_boundary_keeps_top_level_symbolic_claim_values():
    row = _row("m1", "The requested value is 42.")
    row = row.__class__(
        **{**row.__dict__, "metadata": {
            **row.metadata,
            "symbolic_state_claims": {"qslot_0": {"value": "42"}},
            "symbolic_policy_facts": [{"resource": "requested value"}],
            "symbolic_temporal_authorization_events": [{"resource": "requested value"}],
            "symbolic_temporal_authorization_certificate": {
                "version": "temporal-authorization-v1",
                "decision": "unknown",
                "current_authorization": [{"resource": "42", "decision": "unknown"}],
                "graph_edges": [{"target": "42"}],
            },
        }}
    )
    projected, _ = _semantic_context_boundary(
        evidence=[row],
        semantic_contract=_contract(protected=True),
        semantic_atoms=[_atom(memory_id="m1", span="The requested value is 42.")],
        symbolic_trace={"temporal_authorization": {"decision": "unknown"}},
    )
    metadata = projected[0].metadata
    assert metadata["symbolic_state_claims"]["qslot_0"]["value"] == "42"
    assert metadata["symbolic_policy_facts"][0]["resource"] == "requested value"
    assert metadata["symbolic_temporal_authorization_events"][0]["resource"] == "requested value"
    cert = metadata["symbolic_temporal_authorization_certificate"]
    assert cert["decision"] == "unknown"
    assert cert["current_authorization"][0]["resource"] == "42"
    assert cert["graph_edges"][0]["target"] == "42"


def test_answer_formatter_exposes_only_value_free_symbolic_certificates():
    row = _row("m1", "A source-grounded value is 42.")
    row = row.__class__(
        **{**row.__dict__, "metadata": {
            **row.metadata,
            "symbolic_policy_certificate": {
                "version": "policy-v1",
                "decision": "deny",
                "resource": "private value",
                "current_authorization": [{"resource": "42"}],
                "graph_edges": [{"target": "42"}],
                "event_count": 1,
            },
            "symbolic_temporal_authorization_certificate": {
                "version": "temporal-v1",
                "decision": "unknown",
                "query_resource_tokens": ["42"],
                "current_authorization": [{"resource": "42"}],
            },
        }}
    )
    formatted = _format_retrieved_memory([row])
    assert '"resource": "42"' not in formatted
    assert '"current_authorization"' not in formatted
    assert '"graph_edges"' not in formatted
    assert '"decision": "deny"' in formatted


def test_v4_semantic_path_has_no_fixed_query_or_policy_alias_tables():
    assert CURRENT_STATE_SLOT_ALIASES == {}
    assert CURRENT_STATE_DOMAIN_ALIASES == {}
    assert HOUSEHOLD_SLOT_ALIASES == {}
    assert HOUSEHOLD_DELIVERY_SLOT_ALIASES == {}
    assert stage2_typed_rerank._TYPED_QUERY_SLOTS == {}
    assert stage2_typed_rerank._SEMANTIC_QUERY_SLOTS == set()
    assert not hasattr(symbolic_evidence, "_POLICY_SCOPE_ALIASES")


def test_v4_runtime_access_has_no_domain_slot_or_role_tables():
    # Episode-local roles/relations and compiler sensitivity metadata are the
    # only semantic inputs. These names must not reappear as hand-written
    # GateMem/domain vocabularies in the active access boundary.
    for name in (
        "ROLE_TOKENS",
        "GROUP_RULES",
        "LOGISTICS_SLOTS",
        "CLINICAL_SENSITIVE_SLOTS",
        "POLICY_SLOTS",
        "FAMILY_LIKE_TOKENS",
        "DELEGATE_LIKE_TOKENS",
    ):
        assert not hasattr(access, name)


def test_evidence_frame_adapter_has_no_raw_text_semantic_parser_tables():
    # The active adapter must consume compiler atoms/typed metadata only. A
    # raw-text table here would silently become an alternative semantic path.
    for name in (
        "DATE_RE",
        "LOCATION_RE",
        "ALLERGY_RE",
        "MED_DOSAGE_RE",
        "ACCESS_ARTIFACT_TERM_RE",
        "UPDATED_TARGET_DATE_RE",
        "SAFE_WORDING_RE",
        "HOUSEHOLD_SLOT_NAMES",
    ):
        assert not hasattr(evidence_frames, name)
