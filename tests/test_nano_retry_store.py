import sys, types
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'third_party/GateMem-official')]
from scripts.retry_nano_failed_episodes import repair
from bench.remem.types import MemoryNode
class Embed:
    fail=False
    def embed_texts(self, texts):
        if self.fail: raise RuntimeError('injected outage')
        return np.array([[1.,0.]],dtype=np.float32), {}
    def embed_query(self, text): return np.array([1.,0.],dtype=np.float32), {}
def node(i): return MemoryNode(i,'gists',i,'t1','p','user',None,{})
def load(relative, fixed):
    name='bench.remem.test_'+Path(relative).stem+str(fixed)
    mod=types.ModuleType(name); mod.__package__='bench.remem';sys.modules[name]=mod
    text=(ROOT/'third_party/GateMem-official'/relative).read_text()
    exec(repair(relative,text) if fixed else text,mod.__dict__)
    return mod

def test_outage_and_recovery():
    import pytest
    old=load('bench/remem/store.py',False).NodeStore
    new=load('bench/remem/store.py',True).NodeStore
    for cls in (old,new):
        e=Embed();s=cls(embed_router=e);s.add(node('a'));e.fail=True
        with pytest.raises(RuntimeError):s.add(node('b'))
        if cls is new:assert list(s.nodes)==['a'] and s._id_list==['a'] and len(s._emb)==1
        e.fail=False;s.add(node('c'))
        if cls is old:
            with pytest.raises(IndexError):s.similarity_to_existing('c')
        else:assert len(s.semantic_search('x',top_k=20)[0])==2
    e=Embed();a=old(embed_router=e);b=new(embed_router=e)
    for i in ('a','b','c'):a.add(node(i));b.add(node(i))
    assert a._id_list==b._id_list and np.array_equal(a._emb,b._emb)
    assert a.semantic_search('x',top_k=3)==b.semantic_search('x',top_k=3)

def test_outer_index_failed_write_can_retry():
    import pytest
    cls=load('bench/remem/retriever.py',True).ReMemIndex
    e=Embed();s=load('bench/remem/store.py',True).NodeStore(embed_router=e)
    idx=object.__new__(cls);idx.nodes={};idx.stores={'gists':s};e.fail=True
    with pytest.raises(RuntimeError):idx._add_node(node('a'))
    assert not idx.nodes and not s.nodes
    e.fail=False;idx._add_node(node('a'));idx._add_node(node('a'))
    assert len(s._id_list)==1
