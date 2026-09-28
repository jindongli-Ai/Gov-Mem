from pathlib import Path
import ast
import json
import sys
import types

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from baselines.run import adapt_official, replace_once, disk2


def test_mem0_generation_profile_and_openlux_mapping():
    source = (ROOT/'third_party/GateMem-official/bench/agents/mem0.py').read_text()
    adapted = adapt_official('bench/agents/mem0.py', source)
    ast.parse(adapted)
    assert 'float(self.llm_router.config.temperature)' in adapted
    assert 'int(self.llm_router.config.max_output_tokens)' in adapted
    assert '"openlux", "openai-compatible-openlux"' in adapted
    assert 'embed_openai_base += "/v1"' in adapted
    assert 'self._up_memory.add(' in adapted
    assert 'self._up_memory.search(' in adapted


def test_error_checkpoint_is_preserved_and_next_query_runs(tmp_path, monkeypatch):
    sys.path.insert(0, str(ROOT/'third_party/GateMem-official'))
    source = (ROOT/'third_party/GateMem-official/bench/eval/runner.py').read_text()
    adapted = adapt_official('bench/eval/runner.py', source)
    module = types.ModuleType('test_official_adapted_runner')
    exec(compile(adapted, '<adapted runner>', 'exec'), module.__dict__)
    class Agent:
        def reset(self, ep):
            self.turns = []
        def ingest(self, turn):
            self.turns.append(turn.turn_id)
        def query(self, ckpt):
            if ckpt.checkpoint_id == 'q1':
                raise RuntimeError('synthetic transport failure')
            assert self.turns == ['t1', 't2']
            return {'action':'answer','answer':'ok'}
    episode = {'episode_id':'e1','turns':[{'turn_id':t,'text':'source','speaker':{}} for t in ('t1','t2')]}
    checkpoints = [{'checkpoint_id':q,'episode_id':'e1','as_of_turn_id':t,'query_text':'question'} for q,t in [('q1','t1'),('q2','t2')]]
    p = tmp_path/'predictions.jsonl'
    result = module.run_episode(agent=Agent(),episode=episode,checkpoints=checkpoints,prediction_path=str(p))
    assert [r['output']['action'] for r in result] == ['error','answer']
    assert len(p.read_text().splitlines()) == 2
    assert json.loads((tmp_path/'execution_errors.jsonl').read_text())['checkpoint_id'] == 'q1'


def test_patch_refuses_ambiguous_source_and_wrong_disk():
    import pytest
    with pytest.raises(ValueError):
        replace_once('x x','x','y')
    with pytest.raises(ValueError):
        disk2('/tmp/not_an_artifact')


def test_non_adapter_official_code_unchanged():
    for name in ('bench/agents/a_mem.py','bench/agents/remem.py','bench/eval/judge.py','bench/eval/metrics.py'):
        source = (ROOT/'third_party/GateMem-official'/name).read_text()
        assert adapt_official(name, source) == source
