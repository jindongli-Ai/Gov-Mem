"""Revalidate saved model outputs against their original visible request only."""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
from gov_mem.data.schema import MemoryInstance, RetrievedEvidence
from gov_mem.governance_runtime.v8_claim_reasoner import reason_v8_claims
from gov_mem.extraction.v8_event_validator import validate_v8_extraction
from gov_mem.extraction.v8_principal_registry import V8Principal, V8PrincipalRegistry
from gov_mem.llm.json_parser import parse_json_response


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    counts, failures, details = Counter(), Counter(), []
    for path in sorted((args.root/"govmem").glob("*/*/raw_chat_responses/*.json")):
        captured = json.loads(path.read_text())
        msgs = captured["request_payload"]["messages"]
        if "Stage-2 governance reasoner" not in msgs[0]["content"]:
            continue
        request = json.loads(msgs[-1]["content"])
        content = captured["response"]["choices"][0]["message"]["content"]
        domain = path.relative_to(args.root/"govmem").parts[0]
        row = MemoryInstance(instance_id="replay", conversation_id="replay", domain=domain,
            messages=[], question=request["question"], asking_user_id=request["requester"]["principal_id"],
            choices=None, answer=None, metadata={"requester": request["requester"]})
        evidence = [RetrievedEvidence(memory_id=c["source_memory_id"], content=c["text"],
            score=0, retrieval_source="captured", reason="offline replay",
            source_message_ids=c["source_message_ids"], time=c.get("timestamp"))
            for c in request["rag_candidates"]]
        class CapturedLLM:
            def chat_json(self, **kwargs):
                return parse_json_response(content)
        ingestion = request.get("ingestion")
        def accept_events(events, sources):
            fields = ("scenes", "entities", "facts", "relations", "governance_events")
            if not isinstance(events, dict) or any(not isinstance(events.get(k), list) for k in fields):
                raise ValueError("Invalid event arrays")
            registry = V8PrincipalRegistry({r["principal_id"]: V8Principal(**r) for r in ingestion["principal_registry"]}, [])
            batch = validate_v8_extraction(events, turn_text=sources, registry=registry,
                allowed_output_turn_ids={r["turn_id"] for r in ingestion["new_turns"]}, existing_entity_ids=set())
            critical = [r for r in batch.rejected
                if r.get("reason") not in {"context_only_event", "context_only_record", "context_only_scene"}
                and (r.get("kind") == "governance_event" or (r.get("kind") == "fact" and r.get("lifecycle", "assert") != "assert"))]
            if critical:
                raise ValueError("Invalid governance events: "+str(critical))
        counts["stage2_requests"] += 1
        try:
            ledger = reason_v8_claims(instance=row, evidence=evidence, state_projection={},
                graph_context=request["graph_context"], llm_client=CapturedLLM(), model_name="offline",
                ingestion=ingestion, response_protocol="json", memory_mode="shallow",
                **({"candidate_id_style": request["candidate_id_style"]} if request.get("candidate_id_style") else {}),
                access_policy=request.get("access_policy"), accept_events=accept_events)
            counts["passed"] += 1
            details.append({"capture":str(path.relative_to(args.root)), "passed":True,
                            "claims":len(ledger["claims"])})
        except (ValueError, TypeError, KeyError) as exc:
            counts["failed"] += 1
            failures[str(exc)] += 1
            details.append({"capture":str(path.relative_to(args.root)), "passed":False, "error":str(exc)})
    result = {"counts":dict(counts), "failures":dict(failures), "captures":details,
              "note":"Offline contract replay, including repair attempts. No new answering/judging, cache evolution, or MGS inference."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(result["counts"]))


if __name__ == "__main__":
    main()
