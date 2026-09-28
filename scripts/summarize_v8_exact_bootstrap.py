"""Exact stratified episode bootstrap for the small fixed 4 x 3 episode suite.

Enumerate all 27 ordered resamples per domain, hence 27**4=531441 joint draws.
This removes Monte Carlo jitter near zero; it does not remove small-sample or
provider/judge uncertainty. Reads saved judgments only; never calls an API.
"""
from __future__ import annotations
import argparse
import itertools
import math
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
from scripts.score_v8_module_ablations import component_metrics
from scripts.score_v8_paired_episode_suite import read,rows,dump


def exact_interval(full,control,manifest):
    joint=np.array([0.0]);domain_count=len(manifest['domains'])
    for domain,spec in manifest['domains'].items():
        episodes=list(spec['selected_episodes'])
        if joint.size * len(episodes)**len(episodes)>2_000_000:
            raise ValueError('Exact enumeration exceeds configured small-suite budget')
        values=[]
        f={ep:[r for r in full[domain] if r['episode_id']==ep] for ep in episodes}
        c={ep:[r for r in control[domain] if r['episode_id']==ep] for ep in episodes}
        for sample in itertools.product(episodes,repeat=len(episodes)):
            values.append(component_metrics([r for ep in sample for r in f[ep]])['MGS']-
                          component_metrics([r for ep in sample for r in c[ep]])['MGS'])
        joint=np.add.outer(joint,np.asarray(values)).ravel()
    joint.sort();joint/=domain_count
    return {'interval':[float(joint[math.ceil(p*joint.size)-1]) for p in (.025,.975)],
            'ordered_resamples':int(joint.size)}


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--source_scored',type=Path,required=True);args=p.parse_args()
    manifest=read(args.source/'manifest.json');result=read(args.root/'module_metrics.json')
    full={d:rows(args.source_scored/'govmem'/d/'official_eval/judge_scores.jsonl') for d in manifest['domains']}
    counts={}
    for arm in result['full_minus_arm_MGS']:
        control={d:rows(args.root/arm/d/'official_eval/judge_scores.jsonl') for d in manifest['domains']}
        exact=exact_interval(full,control,manifest)
        result['episode_bootstrap_95pct'][arm]=exact['interval'];counts[arm]=exact['ordered_resamples']
    result['bootstrap_note']='Exact percentile bootstrap of whole episodes, stratified by domain; 3 episodes/domain. No provider-repeat or multiple-comparison correction. Original scores unchanged.'
    result['ordered_bootstrap_resamples']=counts
    dump(args.root/'module_metrics_exact.json',result)
    print(result['episode_bootstrap_95pct'])

if __name__=='__main__':main()
