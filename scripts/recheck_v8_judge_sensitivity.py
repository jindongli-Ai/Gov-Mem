"""Post-hoc judge stability diagnostic, never replacement benchmark scores.

Select every checkpoint whose applicable scored outcome changed in any late-veto
arm, then blindly rejudge all distinct full/arm answers twice. Selection is
outcome-dependent and is explicitly unsuitable for estimating overall accuracy.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT),str(ROOT/'third_party/GateMem-official')]
from scripts.score_v8_module_ablations import judge_input
from scripts.score_v8_paired_episode_suite import read,rows,dump
from scripts.run_gatemem_suite import _discover_api_keys
from bench.scripts.score_predictions import _normalize_prediction_row
from bench.eval.judge import run_llm_judge
from bench.llm.router import LLMRouter
from bench.llm.types import LLMConfig


def outcome(row):
    j=row['judge'];q=row['query_type']
    if q=='utility':return j['utility_ok'] is True and j['action_ok'] is True
    return j['privacy_leak' if q=='privacy' else 'deletion_leak']


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--source_scored',type=Path,required=True)
    p.add_argument('--ablation',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();source=args.source.resolve();scored=args.source_scored.resolve();ablation=args.ablation.resolve();out=args.output.resolve()
    manifest=read(source/'manifest.json');arms=read(ablation/'identity.json')['arms'];tasks=[]
    for domain in manifest['domains']:
        systems={'full':scored/'govmem'/domain,**{a:ablation/a/domain for a in arms}}
        judges={a:{r['checkpoint_id']:r for r in rows(path/'official_eval/judge_scores.jsonl')} for a,path in systems.items()}
        predictions={a:{r['checkpoint_id']:r for r in rows(path/'predictions.jsonl')} for a,path in systems.items()}
        selected={cid for a in arms for cid,r in judges[a].items() if outcome(r)!=outcome(judges['full'][cid])}
        distinct={}
        for a in systems:
            for cid in sorted(selected):
                pred=predictions[a][cid];key=judge_input(pred)
                if key not in distinct:distinct[key]={'domain':domain,'checkpoint_id':cid,'prediction':pred,'original':judges[a][cid],'arms':[]}
                distinct[key]['arms'].append(a)
        tasks.extend(distinct.values())
    identity={'selection':'post-hoc: every changed applicable scored outcome; NOT benchmark rescore',
              'repetitions':2,'unique_answers':len(tasks),'checkpoints':len({t['checkpoint_id'] for t in tasks}),
              'inputs_sha256':hashlib.sha256(json.dumps(tasks,sort_keys=True).encode()).hexdigest()}
    if (out/'identity.json').exists() and read(out/'identity.json')!=identity:raise ValueError('Diagnostic input changed')
    dump(out/'identity.json',identity)
    keys=_discover_api_keys(provider='openlux');os.environ['OPENLUX_API_KEY']=keys[0]
    cfg=LLMConfig(provider='openlux',model='gpt-4o',temperature=0,max_output_tokens=4096,api_base='https://api.openlux.ai/v1',api_key_env='OPENLUX_API_KEY')
    def evaluate(task):
        domain=task['domain'];stem=hashlib.sha256(json.dumps(judge_input(task['prediction'])).encode()).hexdigest()
        results=[]
        for rep in range(2):
            path=out/'judgments'/f'{stem}.{rep}.jsonl'
            judged,_=run_llm_judge(episodes=rows(source/'selected_dataset'/domain/'episodes.jsonl'),
                checkpoints=rows(source/'selected_dataset'/domain/'checkpoints.jsonl'),
                predictions=[_normalize_prediction_row(task['prediction'])],judge_router=LLMRouter(cfg),
                prompt_path=str(ROOT/'third_party/GateMem-official/bench/prompts/judge_prompt.txt'),
                out_path=str(path),resume=True,gate_by_action=False,concurrency=1)
            if len(judged)!=1 or not judged[0]['judge']['parse_ok']:raise ValueError('Incomplete repeat judgment')
            results.append(judged[0])
        return {**task,'repeat_judgments':results,'outcomes':[outcome(task['original']),*[outcome(r) for r in results]]}
    with ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(evaluate,tasks))
    dump(out/'sensitivity.json',{'identity':identity,'rows':results,
        'disagreements':sum(len(set(r['outcomes']))>1 for r in results),
        'note':'All original official labels remain unchanged. This post-hoc diagnostic does not estimate aggregate MGS.'})
    print(json.dumps({**identity,'disagreements':sum(len(set(r['outcomes']))>1 for r in results)}))

if __name__=='__main__':main()
