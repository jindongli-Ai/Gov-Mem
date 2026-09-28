import pytest
from gov_mem.extraction.v8_line_protocol import parse_lines, bind_claim_events
from gov_mem.llm.client import LLMClient, LLMConfig


def test_lines_preserve_quotes_backslashes_equals_and_literal_none():
    value = 'Use "backup" at C:\\new\\team=a, not primary.'
    text = ('SLOT\ts1\tslot=route\nCLAIM\tc1\tslot=route\tcandidate_id=candidate_0'
            f'\tvalue={value}\tdelivery=exact\tbind=NONE\nEVENTS\nACTION\tanswer\nEND')
    raw = parse_lines(text, ingestion=True)
    assert raw['claims'][0]['value'] == value
    assert raw['claims'][0]['permission_event_ids'] == []
    assert raw['events'] == {'scenes': [], 'entities': [], 'facts': [], 'relations': [], 'governance_events': []}


@pytest.mark.parametrize('bad', [
    'EVENTS\nACTION\tanswer',
    'EVENTS\nACTION\tanswer\nEND\nCLAIM\tc1\tvalue=leak',
    'EVENTS\nACTION\tanswer\nACTION\trefuse\nEND',
    'EVENTS\nEVENT\tp\teffect=deny\nACTION\tno_memory\nEND',
    'EVENTS\nSOURCE\tmissing\tturn_id=t1\tspan=source\nACTION\tno_memory\nEND',
    'EVENTS\nCLAIM\tc\tslot=x\tcandidate_id=candidate_0\tvalue=x\tdelivery=exact\nACTION\tanswer\nEND',
    'EVENTS\nSLOT\ts\tslot=a\tslot=b\nACTION\tanswer\nEND',
])
def test_malformed_or_incomplete_records_fail_explicitly(bad):
    with pytest.raises(ValueError):
        parse_lines(bad, ingestion=True)


def test_bind_uses_exact_event_identity_and_rejects_unknown_or_incompatible_events():
    claim = {'permission_event_ids': ['p1']}
    raw = {'claims': [claim]}
    graph = {'policies': [{'event_id': 'p1', 'resource_id': 'r', 'action': 'receive', 'scene_id': 's', 'effect': 'deny'}]}
    bind_claim_events(raw, graph)
    assert (claim['resource_id'], claim['action'], claim['scene_id']) == ('r', 'receive', 's')
    claim['permission_event_ids'] = ['invented']
    with pytest.raises(ValueError, match='unknown event'): bind_claim_events(raw, graph)
    claim['permission_event_ids'] = ['p1', 'p2']
    graph['policies'].append({'event_id': 'p2', 'resource_id': 'other', 'effect': 'deny'})
    with pytest.raises(ValueError, match='incompatible'): bind_claim_events(raw, graph)


def test_text_api_omits_json_mode_preserves_literals_and_does_not_retry_truncation():
    client = LLMClient(LLMConfig(provider='openlux', max_retries=2))
    client.is_available = lambda: True
    calls = []
    response = {'choices': [{'message': {'content': 'C:\\new "literal"'}, 'finish_reason': 'stop'}]}
    def post(**kwargs): calls.append(kwargs); return response
    client._post_json = post
    assert client.chat_text(model='gemini-2.5-flash-lite', system_prompt='', user_prompt='') == 'C:\\new "literal"'
    assert 'response_format' not in calls[0]['payload']
    response['choices'][0]['finish_reason'] = 'length'
    with pytest.raises(ValueError, match='truncated'): client.chat_text(model='test', system_prompt='', user_prompt='')
    assert len(calls) == 2
