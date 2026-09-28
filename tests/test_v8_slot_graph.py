import pytest
from gov_mem.memory.v8_slot_graph import V8SlotGraphIndex
from gov_mem.backbones.v4_support import RAGChunk


def chunk(i, text):
    return RAGChunk(chunk_id=i, instance_id='e', text=text, source_message_ids=[i],
                    speaker_ids=['owner'], timestamp_range=(None,None), metadata={})


class Extractor:
    fail = False
    def chat_json(self, **kwargs):
        import json
        if self.fail:
            raise RuntimeError('network unavailable')
        messages = json.loads(kwargs['user_prompt'])['messages']
        return {'records': [{'source_chunk_id': m['source_chunk_id'],
                             'source_span': m['text'], 'target_entities': ['appointment'],
                             'slots': [{'slot_name': 'date', 'value': 'June', 'value_span': 'June'}],
                             'policies': [], 'lifecycle': []} for m in messages]}


def run(idx, llm, chunks):
    return idx.retrieve(conversation_id='e', chunks=chunks, question='appointment date',
                        llm_client=llm, model_name='fake', config={})


def test_graph_retrieves_original_sources_and_reuses_only_unchanged_prefix():
    idx, llm = V8SlotGraphIndex(), Extractor()
    rows, audit = run(idx,llm,[chunk('1','appointment date June')])
    assert rows and rows[0].content == 'appointment date June'
    assert audit['llm_calls'] == 1
    assert run(idx,llm,[chunk('1','appointment date June')])[1]['llm_calls'] == 0
    run(idx,llm,[chunk('1','appointment date June'),chunk('2','appointment June updated')])
    rows,audit=run(idx,llm,[chunk('1','appointment date June')])
    assert audit['cache_reset']
    assert all(r.source_message_ids == ['1'] for r in rows)


def test_failed_graph_extension_does_not_commit_partial_memory():
    idx,llm=V8SlotGraphIndex(),Extractor()
    run(idx,llm,[chunk('1','appointment June')])
    before=idx.cache['e'][0].to_dict()
    llm.fail=True
    rows, audit = run(idx,llm,[chunk('1','appointment June'),chunk('2','appointment June')])
    assert rows == [] and audit['fallback'] == 'dense_only_graph_failure'
    assert idx.cache['e'][0].to_dict()==before
    assert set(idx.cache['e'][1]) == {'1'}
