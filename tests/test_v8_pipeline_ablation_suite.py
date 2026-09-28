import ast
from pathlib import Path
from scripts.run_v8_pipeline_ablation_suite import transform, constant

ROOT=Path(__file__).resolve().parents[1]
SHALLOW='src/gov_mem/governance_runtime/v8_shallow_memory.py'
REASONER='src/gov_mem/governance_runtime/v8_claim_reasoner.py'


def sources(): return {p:(ROOT/p).read_text() for p in (SHALLOW,REASONER)}


def test_old_semantics_only_changes_prompt_not_protocol_or_validator():
    src=sources(); old=src[SHALLOW].replace('Check deletion separately from access:', 'Earlier semantics:')
    out=transform('old_semantics',src,old)
    assert out[REASONER]==src[REASONER]
    before=ast.parse(src[SHALLOW]); after=ast.parse(out[SHALLOW])
    def without_prompt(tree):
        tree.body=[n for n in tree.body if not (isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='SHALLOW_REASONER_PROMPT' for t in n.targets))]
        return ast.dump(tree)
    assert without_prompt(before)==without_prompt(after)


def test_no_projection_keeps_every_delete_update_in_order():
    out=transform('no_advisory_projection',sources(),'')
    namespace={};exec(out[SHALLOW],namespace)
    events=[{'event_id':'e1','effect':'delete'},{'event_id':'e2','effect':'update'}]
    assert namespace['shallow_lifecycle_view']({'lifecycle_events':events})==(events,0)


def test_flat_wire_preserves_exact_nested_values_and_empty_containers():
    out=transform('flat_ledger',sources(),'')
    tree=ast.parse(out[REASONER])
    fn=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='flatten')
    namespace={'json':__import__('json')}
    exec(compile(ast.Module(body=[fn],type_ignores=[]),'flatten','exec'),namespace)
    graph={'permission_events':[{'event_id':'p1','source_spans':[{'span':'secret\nquote','turn_id':'t1'}]}],
           'empty':[], 'null':None, 'path/key':{'x~y':False}}
    actual=namespace['flatten'](graph)
    assert '/permission_events/0/source_spans/0/span = "secret\\nquote"' in actual
    assert '/empty = []' in actual and '/null = null' in actual
    assert '/path~1key/x~0y = false' in actual
    assert '_include_visible_source_candidates(candidates, evidence, sources)' in out[REASONER]


def test_no_source_expansion_retains_visible_graph_and_grounding():
    src=sources();out=transform('no_source_expansion',src,'')
    assert out[SHALLOW]==src[SHALLOW]
    assert 'candidates, evidence = _include_visible_source_candidates' not in out[REASONER]
    assert 'sources = collect_prompt_sources(candidates, graph_context, ingestion)' in out[REASONER]
    assert 'bind_claim_events(raw, graph_context, source_texts=sources)' in out[REASONER]
