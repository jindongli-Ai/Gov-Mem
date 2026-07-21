from gov_mem.backbones.rag_policy_amem import _structured_assertion_decision


def main() -> None:
    request = {"semantic_tags": {"discourse_act": "request", "assertion_confidence": 0.9}}
    assertion = {"semantic_tags": {"discourse_act": "assertion", "assertion_confidence": 0.9}}
    assert _structured_assertion_decision(request) is False
    assert _structured_assertion_decision(assertion) is True
    assert _structured_assertion_decision({}) is None
    print("structured_assertion_smoke=PASS")


if __name__ == "__main__":
    main()
