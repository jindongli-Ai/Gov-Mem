import ast
from pathlib import Path
from types import SimpleNamespace
import pytest
from scripts.run_v8_ledger_removal_ablation import remove_ledger
from gov_mem.data.schema import MemoryInstance
from gov_mem.governance_runtime.v8_event_store import V8EventStore
from gov_mem.governance_runtime.v8_state_projector import project_v8_state


def test_empty_ledger_control_preserves_time_and_never_ingests(tmp_path):
    root=Path(__file__).resolve().parents[1]
    name='src/gov_mem/backbones/govmem_v8_late_governance.py'
    original=(root/name).read_text()
    transformed=remove_ledger('no_persistent_ledger',{name:original},'')[name]
    tree=ast.parse(transformed)
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='GovMemV8LateGovernanceBackbone')
    fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_prepare_ingestion')
    namespace={'MemoryInstance':MemoryInstance,'project_v8_state':project_v8_state}
    exec(compile(ast.Module(body=[fn],type_ignores=[]),'ablation','exec'),namespace)
    store=V8EventStore('episode',memory_mode='shallow')
    obj=SimpleNamespace(_store=lambda _: (store,tmp_path/'unused.json'),memory_mode='shallow')
    instance=MemoryInstance('c','medical','episode',[{'turn_id':'t1','timestamp':'2026-01-01T00:00:00Z','text':'data'}],
                            'query','p',None,None)
    _,_,state,ingestion,audit=namespace['_prepare_ingestion'](obj,instance,None)
    assert ingestion is None and audit['extraction_call_count']==0
    assert state['as_of_turn_id']=='t1' and not state['current_permissions']
    assert not (tmp_path/'unused.json').exists()
    store.processed_turn_ids.append('t1')
    with pytest.raises(ValueError):namespace['_prepare_ingestion'](obj,instance,None)
