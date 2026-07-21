"""A partial certificate must realize allowed fields without reopening denied ones."""

from gov_mem.governance_runtime.graph_slot_renderer import (
    build_graph_authorized_projection,
    render_graph_authorized_slots,
)


def main() -> None:
    certificate = {
        "authorized": True,
        "slots": {"time": {"value": "11:00 AM"}},
        "realizations": [{
            "attribute": "time", "slot_name": "start_time", "value": "11:00 AM",
            "source_text": "The appointment is Friday at 11:00 AM.",
            "source_memory_id": "m1", "record_complete": True,
        }],
    }
    spec = {"requested_attributes": ["time", "restricted_interpretation"]}
    projection = build_graph_authorized_projection(certificate=certificate, semantic_spec=spec)
    assert projection is not None, certificate
    assert projection.metadata["redacted_requested_attributes"] == ["restricted_interpretation"]
    answer = render_graph_authorized_slots(certificate=certificate, action="answer_redacted")
    assert answer.action == "answer_redacted" and "11:00 AM" in answer.answer_text, answer
    assert "restricted" not in answer.answer_text.lower(), answer
    assert "appointment is friday" not in answer.answer_text.lower(), answer
    print("graph_partial_redaction_smoke=PASS")


if __name__ == "__main__":
    main()
