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
