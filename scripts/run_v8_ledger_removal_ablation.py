"""System-level control: raw RAG + language review, without persistent events.

Production V8 is unchanged. This control intentionally removes both accumulated
ledger information and its symbolic projection/bindings. It is NOT an
information-matched test of structured representation; flat_ledger is that test.
"""
from __future__ import annotations
import ast
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
from scripts import run_v8_pipeline_ablation_suite as suite


def remove_ledger(arm,files,old_shallow):
    if arm!='no_persistent_ledger':raise ValueError(arm)
    files=dict(files); name='src/gov_mem/backbones/govmem_v8_late_governance.py'
    source=files[name];tree=ast.parse(source)
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='GovMemV8LateGovernanceBackbone')
    fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_prepare_ingestion')
    replacement='''    def _prepare_ingestion(self, instance: MemoryInstance, registry):
        # Research-only system ablation. Same raw retrieval, neighbors, public
        # policy, requester, source grounding and answering contract. No event
        # extraction, accumulation or history quotations supplied by a ledger.
        store, path = self._store(str(instance.conversation_id or instance.instance_id))
        if store.processed_turn_ids:
            raise ValueError("No-ledger control cannot resume an event-bearing cache")
        turn_ids = [str(m.get("turn_id") or m.get("message_id") or "") for m in instance.messages]
        if not all(turn_ids) or len(set(turn_ids)) != len(turn_ids):
            raise ValueError("V8 requires unique visible turn IDs")
        state = project_v8_state(store, visible_turn_ids=turn_ids,
            requester_principal_id=instance.asking_user_id, question=instance.question,
            checkpoint_time=instance.messages[-1].get("timestamp") if instance.messages else None)
        return store, path, state, None, {"memory_mode": self.memory_mode,
            "new_turn_count": 0, "extraction_call_count": 0,
            "joint_extraction": False, "batches": [], "cache_path": str(path),
            "ablation": "no_persistent_ledger"}
'''
    lines=source.splitlines(keepends=True)
    files[name]=''.join(lines[:fn.lineno-1])+replacement+''.join(lines[fn.end_lineno:])
    compile(files[name],name,'exec')
    return files


if __name__=='__main__':
    suite.ARMS=('no_persistent_ledger',)
    suite.transform=remove_ledger
    suite.main()
