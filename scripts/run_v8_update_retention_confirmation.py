"""Predeclared paired confirmation on V8-unseen episodes (NOT pristine holdout).

Historical V7 evaluated all GateMem episodes. Exclude every audited V8 episode,
select only by episode ID/hash, run both frozen arms end to end, keep errors.
All writable paths, including subprocess temporary/cache paths, stay on disk 2.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sys
import yaml
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
from scripts.run_v8_paired_episode_suite import prepare,run_job,dump
from scripts.run_v8_pipeline_ablation_suite import transform
from scripts.score_v8_paired_episode_suite import merge,judge,rows,read
from scripts.score_v8_module_ablations import component_metrics,judge_input
from scripts.summarize_v8_exact_bootstrap import exact_interval
from scripts.run_gatemem_suite import _discover_api_keys
DOMAINS=('medical','office','education','household')
SEED=20260920


def disk2(path):
    resolved=Path(path).resolve()
    if not resolved.is_relative_to('/mnt/data_disk_2'):
        raise ValueError(f'Writable path outside data_disk_2: {resolved}')
    return resolved


def choose(episode_ids, excluded, domain, seed=SEED, count=3):
    eligible=set(episode_ids)-set(excluded)
    if len(eligible)<count:raise ValueError('Not enough V8-unseen episodes')
    return sorted(sorted(eligible,key=lambda e:hashlib.sha256(f'v8-update-retention:{seed}:{domain}:{e}'.encode()).hexdigest())[:count])


def manifest_from_ids(audit):
    result={'schema_version':'govmem-paired-complete-episodes-1','selection_unit':'complete_episodes',
        'seed':SEED,'episodes_per_domain':3,'selection_uses':['episode_id'],
        'scope':'V8-unseen confirmation only; all episodes previously evaluated by historical V7; NOT pristine holdout',
        'excluded_prior_v8_development_episodes':sorted(audit['episodes']),'domains':{},'entries':[]}
    for domain in DOMAINS:
        data=ROOT/'dataset/GateMem/gatemem/data'/domain
        episodes=[r['episode_id'] for r in rows(data/'episodes.jsonl')]
        if len(episodes)!=len(set(episodes)):raise ValueError('Duplicate episode ID')
        selected=choose(episodes,audit['episodes'],domain)
        # Selection above never sees question/labels/scores. IDs only below.
        checkpoints=rows(data/'checkpoints.jsonl');counts={e:0 for e in selected}
        for r in checkpoints:
            if r['episode_id'] in counts:
                counts[r['episode_id']]+=1
                result['entries'].append({k:r[k] for k in ('checkpoint_id','episode_id')}|{'domain':domain})
        if not all(counts.values()):raise ValueError('Empty episode')
        result['domains'][domain]={'eligible_episode_count':len(set(episodes)-set(audit['episodes'])),
            'selected_episodes':counts,'checkpoint_count':sum(counts.values()),
            'episodes_sha256':hashlib.sha256((data/'episodes.jsonl').read_bytes()).hexdigest(),
            'checkpoints_sha256':hashlib.sha256((data/'checkpoints.jsonl').read_bytes()).hexdigest()}
    return result


def prepare_suite(output,audit_path):
    output=disk2(output);audit=read(audit_path)
    source=ROOT/'outputs/v8_shallow_source_projection_random3_20260919'
    source_identity=read(source/'execution_identity.json')
    spec={'purpose':'Isolate advisory-update folding, not model training or schema changes',
        'scope':'V8-unseen confirmation, historically V7-exposed',
        'seed':SEED,'arms':['full','retain_updates'],
        'source_runtime_sha256':source_identity['runtime_sha256'],
        'harness_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'exposure_audit_sha256':hashlib.sha256(audit_path.read_bytes()).hexdigest(),
        'decision_rule':'Report both official and failure-pessimistic MGS, U/A/F, episode interval and cost. No mid-run tuning or episode replacement. If retention fails to confirm, do not assert benefit or train on these episodes.',
        'train_exclusion':'All selected episodes reserved from any future training or teacher distillation.',
        'model':'gemini-2.5-flash-lite','temperature':0,'judge':'gpt-4o',
        'judge_prompt_sha256':hashlib.sha256((ROOT/'third_party/GateMem-official/bench/prompts/judge_prompt.txt').read_bytes()).hexdigest(),
        'official_code':{str(f.relative_to(ROOT/'third_party/GateMem-official/bench')):hashlib.sha256(f.read_bytes()).hexdigest()
            for f in (ROOT/'third_party/GateMem-official/bench').rglob('*.py')}}
    if (output/'prepared.json').exists():
        if read(output/'protocol.json')!=spec:raise ValueError('Frozen protocol changed')
        return read(output/'manifest.json')
    manifest=manifest_from_ids(audit)
    output.mkdir(parents=True,exist_ok=True);dump(output/'protocol.json',spec);dump(output/'manifest.json',manifest)
    cache=disk2(output/'embedding_cache');cache.mkdir(exist_ok=True)
    temporary=disk2(output/'tmp');temporary.mkdir(exist_ok=True)
    frozen=source/'runtime_snapshot'
    files={str(f.relative_to(frozen)):f.read_text() for f in frozen.rglob('*.py')}
    for arm in spec['arms']:
        dest=output/arm;identity=prepare(output/'manifest.json',dest)
        transformed=files if arm=='full' else transform('no_advisory_projection',files,'')
        digest=hashlib.sha256()
        for name in sorted(transformed):
            data=(frozen/name).read_bytes() if transformed[name]==files[name] else transformed[name].encode()
            target=dest/'runtime_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
            digest.update(name.encode()+b'\0'+data)
        config=yaml.safe_load((source/'configs/govmem.yaml').read_text())
        config['embedding']['cache_dir']=str(cache)
        config['pipeline']['record_execution_errors']=True
        configpath=dest/'configs/govmem.yaml';configpath.write_text(yaml.safe_dump(config,sort_keys=False))
        identity.update(runtime_sha256=digest.hexdigest(),govmem_config_sha256=hashlib.sha256(configpath.read_bytes()).hexdigest())
        dump(dest/'execution_identity.json',identity)
        if arm=='full' and digest.hexdigest()!=source_identity['runtime_sha256']:raise ValueError('Full differs from measured source')
    shutil.copyfile(__file__,output/Path(__file__).name)
    shutil.copyfile(audit_path,output/'prior_exposure_audit.json')
    # Run order is independent of outcomes, interleaves both arms.
    jobs=[]
    ids=read(output/'full/execution_identity.json')['jobs']
    rng=random.Random(SEED)
    for job in ids:
        arms=list(spec['arms']);rng.shuffle(arms)
        for arm in arms:jobs.append({'arm':arm,'episode_id':job['episode_id']})
    dump(output/'run_order.json',jobs)
    dump(output/'prepared.json',{'arms':spec['arms'],'checkpoints':len(manifest['entries'])})
    return manifest


def verify_frozen(output):
    for arm in ('full','retain_updates'):
        root=output/arm;identity=read(root/'execution_identity.json');snapshot=root/'runtime_snapshot';digest=hashlib.sha256()
        for path in sorted(snapshot.rglob('*.py'),key=lambda p:str(p.relative_to(snapshot))):
            name=str(path.relative_to(snapshot));digest.update(name.encode()+b'\0'+path.read_bytes())
        if digest.hexdigest()!=identity['runtime_sha256']:raise ValueError('Snapshot changed')
        configpath=root/'configs/govmem.yaml'
        if hashlib.sha256(configpath.read_bytes()).hexdigest()!=identity['govmem_config_sha256']:raise ValueError('Config changed')
        disk2(yaml.safe_load(configpath.read_text())['embedding']['cache_dir'])
        for j in identity['jobs']:
            disk2(j['data_dir']);disk2(j['manifest'])


def execute(output,workers):
    keys=_discover_api_keys(provider='openlux')
    if not keys:raise ValueError('No API credentials')
    os.environ.update(TMPDIR=str(output/'tmp'),TMP=str(output/'tmp'),TEMP=str(output/'tmp'),PYTHONDONTWRITEBYTECODE='1')
    verify_frozen(output)
    order=read(output/'run_order.json');width=min(max(1,workers),4,len(keys))
    def run(item,key):
        root=output/item['arm'];identity=read(root/'execution_identity.json')
        job=next(j for j in identity['jobs'] if j['episode_id']==item['episode_id'])
        run_job('govmem',job,root,identity,key,True)
    for start in range(0,len(order),width):
        with ThreadPoolExecutor(max_workers=width) as pool:
            futures=[pool.submit(run,item,keys[i]) for i,item in enumerate(order[start:start+width])]
            for f in futures:f.result()
    dump(output/'inference_complete.json',{'episodes_per_arm':12,'checkpoints_per_arm':len(read(output/'manifest.json')['entries'])})


def score(output):
    manifest=read(output/'manifest.json');out=output/'scored';keys=_discover_api_keys(provider='openlux')
    if not keys:raise ValueError('No judge credentials')
    os.environ.update(TMPDIR=str(output/'tmp'),TMP=str(output/'tmp'),TEMP=str(output/'tmp'),PYTHONDONTWRITEBYTECODE='1')
    verify_frozen(output);predictions={};judgments={};metrics={};judge_cache={}
    for arm in ('full','retain_updates'):
        merge(output/arm,'govmem',manifest,out/arm)
        for domain in DOMAINS:
            source=out/arm/'govmem'/domain/'predictions.jsonl';target=out/arm/domain/'predictions.jsonl';target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists() and target.read_bytes()!=source.read_bytes():raise ValueError('Predictions changed')
            target.write_bytes(source.read_bytes())
            predictions[arm,domain]=rows(target)
            dest=target.parent;fingerprint=hashlib.sha256(target.read_bytes()).hexdigest()
            if (dest/'judge_reuse.json').exists():
                if read(dest/'judge_reuse.json')['predictions_sha256']!=fingerprint:raise ValueError('Judged input changed')
            else:
                reused=[judge_cache[domain,judge_input(r)] for r in predictions[arm,domain] if (domain,judge_input(r)) in judge_cache]
                path=dest/'official_eval/judge_scores.jsonl';path.parent.mkdir(parents=True,exist_ok=True)
                if path.exists():raise ValueError('Unattested judgment cache')
                path.write_text(''.join(json.dumps(r)+'\n' for r in reused))
                dump(dest/'judge_reuse.json',{'predictions_sha256':fingerprint,'reused':len(reused)})
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(judge,arm,d,output/arm/'selected_dataset',out,keys[i%len(keys)]) for i,d in enumerate(DOMAINS)]
            for f in futures:f.result()
        metrics[arm]={};judgments[arm]={}
        for domain in DOMAINS:
            directory=out/arm/domain;rs=rows(directory/'official_eval/judge_scores.jsonl');judgments[arm][domain]=rs
            errors={r['checkpoint_id'] for r in predictions[arm,domain] if r.get('execution_status')=='error' or r['action']=='error'}
            m=component_metrics(rs,errors);summary=read(directory/'official_eval/summary.json')
            for key,official in [('U','utility_accuracy'),('A','privacy_leakage_rate'),('F','deletion_leakage_rate')]:
                if abs(m[key]-summary[official])>1e-10:raise ValueError('Metric mismatch')
            m['execution_errors']=len(errors);metrics[arm][domain]=m
            by_id={r['checkpoint_id']:r for r in rs}
            for pred in predictions[arm,domain]:judge_cache[domain,judge_input(pred)]=by_id[pred['checkpoint_id']]
        print('SCORED',arm,sum(m['MGS'] for m in metrics[arm].values())/4,flush=True)
    interval=exact_interval(judgments['retain_updates'],judgments['full'],manifest)
    result={'scope':manifest['scope'],'checkpoints_per_arm':len(manifest['entries']),'per_domain':metrics,
        'mean_MGS':{a:sum(m['MGS'] for m in ds.values())/4 for a,ds in metrics.items()},
        'mean_pessimistic_MGS':{a:sum(m['pessimistic_MGS'] for m in ds.values())/4 for a,ds in metrics.items()},
        'retain_minus_full_95pct':interval,'training_performed':False}
    result['retain_minus_full_MGS']=result['mean_MGS']['retain_updates']-result['mean_MGS']['full']
    costs={}
    for arm in ('full','retain_updates'):
        costs[arm]={}
        for path in (output/arm/'govmem').glob('*/*/http_telemetry.jsonl'):
            for r in rows(path):
                c=costs[arm].setdefault(r['requested_model'],{'requests':0,'tokens':0,'http_failures':0})
                c['requests']+=1;c['tokens']+=(r.get('usage')or{}).get('total_tokens',0);c['http_failures']+=r.get('status_code',200)>=400
    result['inference_costs']=costs
    dump(output/'confirmation_metrics.json',result);print(json.dumps(result,ensure_ascii=False,indent=2))


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--audit',type=Path,default=ROOT/'experiments/manifests/v8_prior_exposure_audit_20260919.json')
    p.add_argument('--stage',choices=['prepare','infer','score','all'],default='prepare');p.add_argument('--workers',type=int,default=4)
    args=p.parse_args();output=disk2(args.output)
    manifest=prepare_suite(output,args.audit.resolve())
    print(json.dumps({'output':str(output),'episodes':12,'checkpoints_per_arm':len(manifest['entries']),'scope':manifest['scope']}),flush=True)
    if args.stage in {'infer','all'}:execute(output,args.workers)
    if args.stage in {'score','all'}:score(output)

if __name__=='__main__':main()
