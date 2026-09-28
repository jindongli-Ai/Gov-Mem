import copy
import pytest
from gov_mem.governance_runtime.v8_shallow_memory import normalize_shallow_json,expand_kept_candidates
from gov_mem.governance_runtime.v8_claim_reasoner import reason_v8_claims
from gov_mem.data.schema import MemoryInstance,RetrievedEvidence


def payload():
    return {'query_slots':[{'slot':'note'}], 'answer_action':'refuse',
        'claims':[{'slot':'note','candidate_id':'t1','block':{
            'restriction_kind':'explicit_restriction',
            'restriction_evidence':[{'turn_id':'t1','span':'Do not share the private note.'}]}}]}


def test_nested_denial_preserves_decision_sources_and_never_becomes_keep_release():
    raw=payload();expected=copy.deepcopy(raw['claims'][0]['block']['restriction_evidence'])
    normalized=normalize_shallow_json(raw,ingestion=False)
    expand_kept_candidates(normalized,[{'candidate_id':'t1','text':'Do not share the private note.'}])
    c=normalized['claims'][0]
    assert c['delivery']=='block' and c['restriction_evidence']==expected
    assert c['permission_event_ids']==[] and c['value']=='Do not share the private note.'


@pytest.mark.parametrize('extra',[{'keep':True},{'delivery':'exact'},{'delivery':'summary'},
                                 {'restriction_kind':'deleted'},{'restriction_evidence':[]}])
def test_conflicting_nested_denial_fails(extra):
    raw=payload();raw['claims'][0].update(extra)
    with pytest.raises(ValueError):normalize_shallow_json(raw,ingestion=False)


@pytest.mark.parametrize('block',[True,{}, {'restriction_kind':'deleted','restriction_evidence':[]},
    {'restriction_kind':'deleted','restriction_evidence':'invented text'},
    {'restriction_kind':'deleted','restriction_evidence':[{}],'allow':True}])
def test_incomplete_or_ambiguous_denial_fails(block):
    raw=payload();raw['claims'][0]['block']=block
    with pytest.raises(ValueError):normalize_shallow_json(raw,ingestion=False)


def test_missing_bind_on_release_still_fails():
    raw=payload();raw['claims']=[{'slot':'note','candidate_id':'t1','keep':True}]
    with pytest.raises(ValueError,match='explicit bind'):normalize_shallow_json(raw,ingestion=False)


def test_nested_denial_cannot_bypass_exact_source_validation():
    instance=MemoryInstance('id','synthetic','ep',[],'The note?','guest',None,None)
    evidence=[RetrievedEvidence('m1','Do not share the private note.',1,'test','test',source_message_ids=['t1'])]
    class LLM:
        def __init__(self,span):self.span=span
        def chat_json(self,**kwargs):
            raw=payload();raw['claims'][0]['block']['restriction_evidence'][0]['span']=self.span;return raw
    def run(span):
        return reason_v8_claims(instance=instance,evidence=evidence,state_projection={},graph_context={},
            llm_client=LLM(span),model_name='test',memory_mode='shallow',response_protocol='json',candidate_id_style='source')
    assert run('Do not share the private note.')['claims'][0]['delivery']=='block'
    with pytest.raises(ValueError):run('This sentence never appeared.')
