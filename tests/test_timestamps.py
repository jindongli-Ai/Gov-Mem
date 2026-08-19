from gov_mem.data.timestamps import normalize_message_timestamp, normalize_timestamp


def test_normalize_timestamp_adds_zero_seconds_only_when_missing():
    assert normalize_timestamp("2026-05-01T09:00") == "2026-05-01T09:00:00"
    assert normalize_timestamp("2026-05-01 09:00+0800") == "2026-05-01 09:00:00+0800"


def test_normalize_timestamp_preserves_existing_precision_and_missing_values():
    assert normalize_timestamp("2026-05-01T09:00:07") == "2026-05-01T09:00:07"
    assert normalize_timestamp("2026-05-01T09:00:07.250Z") == "2026-05-01T09:00:07.250Z"
    assert normalize_timestamp(None) is None
    assert normalize_timestamp("2026-05-01") == "2026-05-01"


def test_normalize_message_timestamp_keeps_source_turn_aligned():
    message = {
        "timestamp": "2026-05-01T09:00",
        "source_turn": {"timestamp": "2026-05-01T09:00", "record_refs": ["r1"]},
    }

    normalized = normalize_message_timestamp(message)

    assert normalized["timestamp"] == "2026-05-01T09:00:00"
    assert normalized["source_turn"]["timestamp"] == "2026-05-01T09:00:00"
    assert message["timestamp"] == "2026-05-01T09:00"


def test_checkpoint_adapter_normalizes_visible_turn_and_retained_source_turn():
    from gov_mem.data.adapters import CheckpointBenchmarkAdapter

    visible = CheckpointBenchmarkAdapter._visible_messages_until_checkpoint(
        [
            {
                "turn_id": "t1",
                "timestamp": "2026-05-01T09:00",
                "speaker": {"principal_id": "p1", "role": "nurse"},
                "text": "A note.",
                "future_field": {"keep": True},
            }
        ],
        "t1",
    )

    assert visible[0]["timestamp"] == "2026-05-01T09:00:00"
    assert visible[0]["source_turn"]["timestamp"] == "2026-05-01T09:00:00"
    assert visible[0]["source_turn"]["future_field"] == {"keep": True}
