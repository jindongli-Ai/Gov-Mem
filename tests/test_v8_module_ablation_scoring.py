import pytest
from scripts.score_v8_module_ablations import component_metrics,judge_input


def test_judge_reuse_key_includes_actual_scored_fields():
    p={'checkpoint_id':'x','action':'answer','answer':'A','answer_structured':{'k':'v'}}
    assert judge_input(p)!=judge_input({**p,'answer':'B'})
    assert judge_input(p)!=judge_input({**p,'action':'refuse'})
    assert judge_input(p)!=judge_input({**p,'answer_structured':{'k':'z'}})
    assert judge_input(p)==judge_input({**p,'memory_audit':{'text':'not in official LLM judge prompt'}})


def test_worst_case_keeps_error_samples_and_nulls_raise():
    judged=[{'checkpoint_id':q,'query_type':q,'judge':{'parse_ok':True,'action_ok':True,k:value}}
            for q,k,value in [('utility','utility_ok',True),('privacy','privacy_leak',False),('safety','deletion_leak',False)]]
    m=component_metrics(judged,{'safety'})
    assert m['MGS']==1 and m['pessimistic_MGS']==0
    assert judged[2]['judge']['deletion_leak'] is False
    judged[2]['judge']['deletion_leak']=None
    with pytest.raises(ValueError):component_metrics(judged)
