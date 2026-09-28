from gov_mem.llm.client import LLMClient, LLMConfig


def test_openai_compatible_chat_sends_gate_mem_output_budget():
    client = LLMClient(
        LLMConfig(provider="openlux", max_output_tokens=4096, allow_fallback=False)
    )
    captured = {}
    client.is_available = lambda: True

    def fake_post_json(*, endpoint, payload):
        captured["endpoint"] = endpoint
        captured["payload"] = payload
        return {"choices": [{"message": {"content": "{}"}}]}

    client._post_json = fake_post_json
    assert client.chat_json(model="gpt-4o-mini", system_prompt="", user_prompt="{}") == {}
    assert captured["endpoint"] == "chat/completions"
    assert captured["payload"]["temperature"] == 0.0
    assert captured["payload"]["max_tokens"] == 4096


def test_v8_does_not_pay_again_for_invalid_json():
    import pytest
    client = LLMClient(LLMConfig(provider="openlux", max_retries=3, json_max_attempts=1))
    client.is_available = lambda: True
    requests = []
    def post(**kwargs):
        requests.append(kwargs)
        return {"choices": [{"message": {"content": "not JSON"}}]}
    client._post_json = post
    with pytest.raises(Exception):
        client.chat_json(model="test", system_prompt="", user_prompt="")
    assert len(requests) == 1
    assert client.telemetry_snapshot()["chat_json_parse_failure"]["calls"] == 1


def test_successful_provider_response_accounts_tokens(monkeypatch):
    client = LLMClient(LLMConfig(provider="openlux", max_retries=1))
    class Response:
        def raise_for_status(self): pass
        def json(self): return {"usage": {"prompt_tokens": 70, "completion_tokens": 30, "total_tokens": 100}}
    monkeypatch.setattr(client._session, "post", lambda *a, **k: Response())
    client._post_json(endpoint="chat/completions", payload={"model": "test"})
    usage = client.telemetry_snapshot()["chat/completions"]
    assert usage["calls"] == 1 and usage["total_tokens"] == 100


def test_http_response_decode_retry_is_not_double_counted(monkeypatch):
    import requests
    client = LLMClient(LLMConfig(provider="openlux", max_retries=2))
    attempts = []
    class Response:
        def raise_for_status(self): pass
        def json(self):
            if len(attempts) == 1:
                raise requests.exceptions.JSONDecodeError("bad", "x", 0)
            return {"usage": {"total_tokens": 12}}
    def post(*a, **k):
        attempts.append(1)
        return Response()
    monkeypatch.setattr(client._session, "post", post)
    monkeypatch.setattr("gov_mem.llm.client.time.sleep", lambda seconds: None)
    client._post_json(endpoint="chat/completions", payload={"model": "test"})
    row = client.telemetry_snapshot()["chat/completions"]
    assert len(attempts) == row["calls"] + row["retries"] == 2
    assert row["responses_with_usage"] == 1
