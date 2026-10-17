"""Fig. 2 wiring under the published shallow-memory Full configuration."""
import json
from copy import deepcopy
from pathlib import Path

from gov_mem.backbones.govmem_v8_late_governance import GovMemV8LateGovernanceBackbone
from gov_mem.data.schema import MemoryInstance
from gov_mem.memory.dense_index import DenseMemoryIndex
from gov_mem.utils.config import load_yaml_config


class Embeddings:
    def embed_texts(self, *, model, texts):
        return [[1.0, 0.0] for _ in texts]


class FrameworkLLM:
    def __init__(self, with_warning):
        self.with_warning = with_warning
        self.graph_inputs = []
        self.reasoning_inputs = []
        self.answer_inputs = []

    def chat_json(self, *, model, system_prompt, user_prompt):
        payload = json.loads(user_prompt)
        if 'episode-local governed slot graph' in system_prompt:
            self.graph_inputs.append(payload)
            return {'records': [dict(
                source_chunk_id=row['source_chunk_id'], source_message_id=row['source_message_id'],
                source_span=row['text'], target_entities=['appointment'], slots=[], lifecycle=[],
                policies=([{'effect': 'deny', 'subject': None, 'resource': None,
                            'span': 'Do not share the private note.'}] if self.with_warning and
                          'private note' in row['text'] else []))
                for row in payload['messages']]}
        if 'rag_candidates' in payload:
            self.reasoning_inputs.append(payload)
            return {'events': [], 'query_slots': [{'slot': 'appointment date'}],
                    'answer_action': 'answer', 'claims': [
                        {'slot': 'appointment date', 'candidate_id': 't002',
                         'value': 'June 20', 'quote': 'appointment is June 20',
                         'delivery': 'exact', 'bind': []}]}
        self.answer_inputs.append(payload)
        return {'answer': 'The appointment is June 20.', 'used_claim_ids': []}


def run_full(tmp_path, monkeypatch, with_warning):
    config = deepcopy(load_yaml_config(Path(__file__).resolve().parents[1] / 'configs/govmem.yaml'))
    config['embedding']['cache_dir'] = str(tmp_path / 'embeddings')
    assert config['v8_memory']['mode'] == 'shallow'
    assert config['memory_governed_slot_graph']['enabled'] is True
    llm = FrameworkLLM(with_warning)
    instance = MemoryInstance(
        instance_id='checkpoint', conversation_id='episode', domain='medical',
        messages=[
            {'turn_id': 't001', 'message_id': 't001', 'speaker_id': 'owner',
             'text': 'Do not share the private note.', 'timestamp': '2026-01-01T09:00'},
            {'turn_id': 't002', 'message_id': 't002', 'speaker_id': 'owner',
             'text': 'The appointment is June 20.', 'timestamp': '2026-01-01T09:01'},
            {'turn_id': 't003', 'message_id': 't003', 'speaker_id': 'owner',
             'text': 'When is the appointment?', 'timestamp': '2026-01-01T09:02'}],
        question='When is the appointment?', asking_user_id='owner', choices=None, answer=None,
        metadata={'requester': {'principal_id': 'owner'}, 'observable': {'as_of_turn_id': 't003'}})
    monkeypatch.setattr(DenseMemoryIndex, 'query', lambda *a, **kw: [('chunk_0002_t002_t002', 1.0)])
    backbone = GovMemV8LateGovernanceBackbone(
        llm_client=llm, embedding_client=Embeddings(), config=config,
        output_dir=tmp_path, dataset_name='synthetic')
    result = backbone.run_instance(instance)
    return llm, result


def test_full_graph_findings_reach_neural_reasoner_and_safe_boundary(tmp_path, monkeypatch):
    llm, result = run_full(tmp_path, monkeypatch, with_warning=True)
    assert llm.graph_inputs and llm.reasoning_inputs and llm.answer_inputs
    audit = llm.reasoning_inputs[0]['governed_slot_graph_audit']
    assert audit['mode'] == 'advisory_only'
    assert audit['findings'][0]['span'] == 'Do not share the private note.'
    graph_audit = result.answer_result.raw_response['governed_slot_graph']
    assert graph_audit['observer']['finding_count'] == 1
    assert graph_audit['selection']['graph_selected'] == 0
    assert result.answer_result.action == 'answer'
    assert result.answer_result.answer_text == 'The appointment is June 20.'
    assert 'private note' not in json.dumps(llm.answer_inputs)
    assert all(row['source_message_id'] != 't003'
               for batch in llm.graph_inputs for row in batch['messages'])


def test_graph_silence_preserves_normal_governance_reasoning(tmp_path, monkeypatch):
    llm, result = run_full(tmp_path, monkeypatch, with_warning=False)
    assert llm.graph_inputs and llm.reasoning_inputs and llm.answer_inputs
    assert 'governed_slot_graph_audit' not in llm.reasoning_inputs[0]
    assert result.answer_result.raw_response['governed_slot_graph']['observer']['finding_count'] == 0
    assert result.answer_result.action == 'answer'
