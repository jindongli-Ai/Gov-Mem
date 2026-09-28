"""Replay deterministic V8 projection/critic on saved ledgers, without judging."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import inspect
import json
from pathlib import Path

from gov_mem.governance_runtime.v8_event_store import V8EventStore
from gov_mem.governance_runtime.v8_state_projector import project_v8_state
from gov_mem.governance_runtime.v8_symbolic_critic import criticize_v8_claims


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    counts, changes = Counter(), []
    for path in sorted((args.root / "govmem").glob("*/*/debug_cases/checkpoint_benchmark/*.json")):
        saved = json.loads(path.read_text())
        ledger, cache = saved.get("v8_claim_ledger"), saved.get("v8_event_store")
        if not ledger or not cache:
            continue
        store = V8EventStore(**cache)
        original_state = saved["v8_state_projection"]
        turns = store.processed_turn_ids
        # Never replay a larger cached prefix than the original checkpoint.
        if (len(turns) != original_state["visible_turn_count"]
                or (turns[-1] if turns else None) != original_state["as_of_turn_id"]):
            raise ValueError(f"Cannot reconstruct original visible prefix: {path}")
        state = project_v8_state(store, visible_turn_ids=turns,
            requester_principal_id=saved["asking_user_id"], question=saved["question"],
            checkpoint_time=original_state.get("checkpoint_time"))
        reviewed = criticize_v8_claims(ledger, state_projection=state,
            requester_principal_id=saved["asking_user_id"])
        counts["normal_checkpoints_replayed"] += 1
        for before, after in zip(ledger["claims"], reviewed["claims"], strict=True):
            if before["delivery"] != "block" and after["delivery"] == "block":
                counts["additional_release_to_block"] += 1
                changes.append({"checkpoint_id": saved["instance_id"], "claim_id": after["claim_id"],
                    "domain": path.relative_to(args.root / "govmem").parts[0],
                    "reasons": after.get("symbolic_veto_reasons", [])})
    result = {"counts": dict(counts), "changes": changes,
        "source_runtime_sha256": json.loads((args.root / "execution_identity.json").read_text())["runtime_sha256"],
        "replayed_code_sha256": {function.__name__: hashlib.sha256(Path(inspect.getfile(function)).read_bytes()).hexdigest()
                                 for function in (project_v8_state, criticize_v8_claims)},
        "note": "Offline deterministic replay on finalized saved ledgers. No new LLM output, answer generation, judge labels or MGS inference."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["counts"]))


if __name__ == "__main__":
    main()
