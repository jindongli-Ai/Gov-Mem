"""Prepare isolated frozen V8 ablations; never modify canonical production code.

All arms replay complete episodes from an empty event store. Shared embeddings
are content-addressed; chat outputs/event stores are never shared across arms.
"""
from __future__ import annotations
import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import yaml
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
from scripts.run_v8_paired_episode_suite import prepare, run_job, dump
from scripts.run_gatemem_suite import _discover_api_keys

ARMS=('old_semantics','flat_ledger','no_advisory_projection','no_source_expansion')


def constant(source, name):
    tree=ast.parse(source)
    for node in tree.body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in node.targets):
            return ast.literal_eval(node.value),node
    raise ValueError(name)


def replace_constant(source,name,value):
    _,node=constant(source,name)
    lines=source.splitlines(keepends=True)
    return ''.join(lines[:node.lineno-1])+name+' = '+repr(value)+'\n'+''.join(lines[node.end_lineno:])


def transform(arm, files, old_shallow):
    files=dict(files)
    shallow='src/gov_mem/governance_runtime/v8_shallow_memory.py'
    reasoner='src/gov_mem/governance_runtime/v8_claim_reasoner.py'
    if arm=='old_semantics':
        prompt,_=constant(old_shallow,'SHALLOW_REASONER_PROMPT')
        # Protocol tokens are adapted to the *unchanged* new JSON contract.
        # Reintroducing TAB-era parsing failures would not isolate semantics.
        prompt=prompt.replace('requested SLOTs','requested slots').replace('prefer KEEP with its candidate ID:', 'prefer keep=true with its candidate ID:')
        prompt=prompt.replace('Never KEEP','Never keep').replace('use CLAIM','use claims').replace('block CLAIMs','blocked claims')
        prompt=prompt.replace('A blocked CLAIM','A blocked claim').replace('RESTRICTION citation(s)','restriction_evidence citation(s)')
        files[shallow]=replace_constant(files[shallow],'SHALLOW_REASONER_PROMPT',prompt)
    elif arm=='no_advisory_projection':
        tree=ast.parse(files[shallow]); node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='shallow_lifecycle_view')
        lines=files[shallow].splitlines(keepends=True)
        files[shallow]=''.join(lines[:node.lineno-1])+('def shallow_lifecycle_view(state: dict) -> tuple[list[dict], int]:\n'
            '    """Ablation: retain all lifecycle events, including superseded updates."""\n'
            '    return list(state.get("lifecycle_events") or []), 0\n')+''.join(lines[node.end_lineno:])
    elif arm=='flat_ledger':
        # Only the wire representation changes. Binding/validation still use
        # the original typed graph. JSON-pointer leaf rows preserve EVERY value,
        # ordering, exact quote, ID, time and empty container, with an inverse.
        insertion='''    # Experiment-only lossless ledger representation.
    def flatten(value, path=""):
        if isinstance(value, dict) and value:
            return [line for key, item in value.items()
                    for line in flatten(item, path+"/"+key.replace("~", "~0").replace("/", "~1"))]
        if isinstance(value, list) and value:
            return [line for i,item in enumerate(value) for line in flatten(item,path+"/"+str(i))]
        return [path+" = "+json.dumps(value,ensure_ascii=False)]
    wire_payload = dict(wire_payload)
    wire_payload["graph_context"] = "\\n".join(flatten(graph_context))
    system += ("\\nLEDGER ENCODING: graph_context is lossless text, one JSON-pointer path = JSON value per line. "
               "Paths retain original fields, event indices, IDs, exact source quotations, activation and scopes. "
               "Use the same policy and binding rules on these facts.")
'''
        needle='    user_prompt = json.dumps(wire_payload, ensure_ascii=False, separators=(",", ":"))'
        if files[reasoner].count(needle)!=1: raise ValueError('Unexpected reasoner source')
        files[reasoner]=files[reasoner].replace(needle,insertion+needle)
    elif arm=='no_source_expansion':
        needle='    if candidate_id_style == "source":\n        candidates, evidence = _include_visible_source_candidates(candidates, evidence, sources)'
        if files[reasoner].count(needle)!=1: raise ValueError('Unexpected source expansion')
        files[reasoner]=files[reasoner].replace(needle,'    # Ablation: only RAG/adjacent candidates can supply answer claims.')
        old=('"A turn quoted only in graph source_spans or ingestion is also selectable by that exact turn ID, "\n'
             '                       "using only its supplied text. If only disjoint fragments are supplied, do not KEEP the whole turn. "')
        new=('"Only rag_candidates can supply answer claim values. Graph/ingestion quotations remain "\n'
             '                       "available for permissions and restriction_evidence, but are not additional answer candidates. "')
        if old not in files[reasoner]: raise ValueError('Unexpected source instruction')
        files[reasoner]=files[reasoner].replace(old,new)
    else: raise ValueError(arm)
    for path,source in files.items(): compile(source,path,'exec')
    return files


def setup(output, source, old_source, arm):
    marker=output/'ablation_identity.json'
    source_identity=json.loads((source/'execution_identity.json').read_text())
    wanted={'arm':arm,'source_runtime_sha256':source_identity['runtime_sha256'],
            'old_semantics_runtime_sha256':json.loads((old_source/'execution_identity.json').read_text())['runtime_sha256'],
            'harness_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'complete_episode_rerun':True,'upstream_state':'fresh per episode; no event/chat cache reuse'}
    if marker.exists():
        if json.loads(marker.read_text())!=wanted: raise ValueError('Ablation identity changed')
        return json.loads((output/'execution_identity.json').read_text())
    if (output/'govmem').exists(): raise ValueError('Unattested existing inference')
    identity=prepare(source/'manifest.json',output)
    snapshot=output/'runtime_snapshot'
    frozen=source/'runtime_snapshot'
    files={str(f.relative_to(frozen)):f.read_text() for f in frozen.rglob('*.py')}
    old_shallow=(old_source/'runtime_snapshot/src/gov_mem/governance_runtime/v8_shallow_memory.py').read_text()
    transformed=transform(arm,files,old_shallow)
    digest=hashlib.sha256(); changes={}
    for name in sorted(files):
        data=transformed[name].encode(); target=snapshot/name
        target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
        digest.update(name.encode()+b'\0'+data)
        if files[name]!=transformed[name]: changes[name]={'before':hashlib.sha256(files[name].encode()).hexdigest(),'after':hashlib.sha256(data).hexdigest()}
    # Shared embeddings are keyed by text/model. Never duplicate inode-heavy caches.
    config_path=output/'configs/govmem.yaml'
    config=yaml.safe_load((source/'configs/govmem.yaml').read_text())
    config['embedding']['cache_dir']=str(source/'embedding_cache')
    config_path.write_text(yaml.safe_dump(config,sort_keys=False))
    identity.update(runtime_sha256=digest.hexdigest(),govmem_config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest())
    dump(output/'execution_identity.json',identity)
    dump(output/'ablation_diff.json',changes);dump(marker,wanted)
    return identity


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--old_source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--arm',choices=ARMS,required=True)
    p.add_argument('--execute',action='store_true');p.add_argument('--workers',type=int,default=3)
    args=p.parse_args();output=args.output.resolve()
    identity=setup(output,args.source.resolve(),args.old_source.resolve(),args.arm)
    print(json.dumps({'arm':args.arm,'episodes':len(identity['jobs']),'checkpoints':sum(j['checkpoint_count'] for j in identity['jobs']),
                      'runtime_sha256':identity['runtime_sha256']}),flush=True)
    if not args.execute:return
    keys=_discover_api_keys(provider='openlux')
    if not keys:raise ValueError('No configured API credentials')
    width=min(max(args.workers,1),len(keys),4)
    for start in range(0,len(identity['jobs']),width):
        with ThreadPoolExecutor(max_workers=width) as pool:
            futures=[pool.submit(run_job,'govmem',j,output,identity,keys[i],True) for i,j in enumerate(identity['jobs'][start:start+width])]
            for f in futures:f.result()
    dump(output/'complete.json',{'episodes':len(identity['jobs']),'checkpoints':sum(j['checkpoint_count'] for j in identity['jobs'])})

if __name__=='__main__':main()
