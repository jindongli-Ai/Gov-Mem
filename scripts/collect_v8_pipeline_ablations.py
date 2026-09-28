"""Merge completed frozen episode suites for exact-input official score reuse."""
from __future__ import annotations
import argparse
import hashlib
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
from scripts.score_v8_paired_episode_suite import merge,read,dump


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--suite',action='append',required=True)
    args=p.parse_args();source=args.source.resolve();output=args.output.resolve()
    manifest=read(source/'manifest.json'); suites={}
    for value in args.suite:
        arm,path=value.split('=',1);root=Path(path).resolve()
        if read(root/'ablation_identity.json')['arm']!=arm:raise ValueError('Arm identity mismatch')
        if (root/'manifest.json').read_bytes()!=(source/'manifest.json').read_bytes():raise ValueError('Manifest mismatch')
        for domain in manifest['domains']:
            for name in ('episodes.jsonl','checkpoints.jsonl'):
                if (root/'selected_dataset'/domain/name).read_bytes()!=(source/'selected_dataset'/domain/name).read_bytes():
                    raise ValueError('Selected data differs')
        suites[arm]=root
    identity={'manifest_sha256':hashlib.sha256((source/'manifest.json').read_bytes()).hexdigest(),
              'arms':list(suites),'suites':{a:{'root':str(r),'runtime_sha256':read(r/'execution_identity.json')['runtime_sha256']} for a,r in suites.items()}}
    if (output/'identity.json').exists() and read(output/'identity.json')!=identity:raise ValueError('Merged experiment identity changed')
    dump(output/'identity.json',identity)
    for arm,root in suites.items():
        temp=output/arm
        merge(root,'govmem',manifest,temp)
        for domain in manifest['domains']:
            src=temp/'govmem'/domain/'predictions.jsonl';target=temp/domain/'predictions.jsonl'
            target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists() and target.read_bytes()!=src.read_bytes():raise ValueError('Merged predictions changed')
            target.write_bytes(src.read_bytes())
    print('Merged complete episode suites:',','.join(suites))

if __name__=='__main__':main()
