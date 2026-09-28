import ast
import json
from pathlib import Path

import pytest

from gov_mem.governance_runtime.v8_prompt_input import application_policy, pack_prompt, unpack_prompt
from gov_mem.governance_runtime.v8_claim_reasoner import reason_v8_claims
from gov_mem.data.schema import MemoryInstance, RetrievedEvidence
from gov_mem.utils.config import load_yaml_config


def test_public_policies_are_verbatim_official_rules_not_episode_data():
    root = Path(__file__).resolve().parents[1]
    config = load_yaml_config(root / "configs/govmem_v8_late_governance_gemini25flashlite.yaml")
    tree = ast.parse((root / "third_party/GateMem-official/bench/domains.py").read_text())
    constants = {n.targets[0].id: ast.literal_eval(n.value) for n in tree.body
                 if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
                 and n.targets[0].id.endswith("_QUERY_POLICY")}
    for domain in ("medical", "office", "education", "household"):
        assert application_policy(config, domain)["text"].rstrip("\n") == constants[domain.upper() + "_QUERY_POLICY"]
    assert application_policy({}, "medical") is None
    with pytest.raises(ValueError, match="No configured"):
        application_policy(config, "unseen")


def test_lossless_repeated_text_preserves_all_sources_and_unicode():
    text = 'Literal "quoted" text \\ with tabs\tand newlines\n还有中文。' * 20
    original = {"question": "What?", "candidates": [text, text],
                "graph": [{"source_spans": [{"turn_id": "t1", "span": text}]}],
                "new_turns": [{"turn_id": "t2", "text": text}]}
    packed, audit = pack_prompt(original)
    assert unpack_prompt(packed) == original
    assert audit["saved_chars"] > 0
    assert original["candidates"][0] == text
    short = {"x": ["no", "no"]}
    assert pack_prompt(short)[0] == short


@pytest.mark.parametrize("configured", [False, True])
def test_policy_citations_are_valid_only_when_supplied(configured):
    policy = application_policy({"access_policy": {"enabled": True, "by_domain": {
        "synthetic": "Only an assigned operator may receive the private credential."
    }}}, "synthetic")
    class LLM:
        def chat_text(self, **kwargs):
            payload = unpack_prompt(json.loads(kwargs["user_prompt"]))
            assert ("access_policy" in payload) is configured
            return ("SLOT\ts1\tslot=credential\n"
                    "CLAIM\tc1\tslot=credential\tcandidate_id=candidate_0\tvalue=Z-19"
                    "\tdelivery=block\tbind=NONE\trestriction_kind=role_scope_mismatch\n"
                    "RESTRICTION\tc1\tturn_id=application_access_policy"
                    "\tspan=Only an assigned operator may receive the private credential.\n"
                    "ACTION\trefuse\nEND")
    instance = MemoryInstance(instance_id="synthetic", domain="synthetic", conversation_id="ep", messages=[], question="Credential?",
        asking_user_id="guest", choices=None, answer=None, metadata={})
    evidence = [RetrievedEvidence(memory_id="m1", content="Credential Z-19.", score=1,
        retrieval_source="dense", reason="test", source_message_ids=["t1"])]
    kwargs = dict(instance=instance, evidence=evidence, state_projection={}, graph_context={},
        llm_client=LLM(), model_name="fake", response_protocol="lines",
        access_policy=policy if configured else None, compact_input=True)
    if configured:
        result = reason_v8_claims(**kwargs)
        assert result["claims"][0]["delivery"] == "block"
        assert result["audit"]["access_policy_sha256"] == policy["sha256"]
    else:
        with pytest.raises(ValueError, match="grounded"):
            reason_v8_claims(**kwargs)
