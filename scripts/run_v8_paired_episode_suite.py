"""Execute a fixed paired complete-episode manifest, with immutable snapshots.

Sampling is performed before this runner; it never selects by checkpoint type,
answer, evidence, failure, or score. Official RAG-Naive is used unchanged.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT))
from gov_mem.utils.config import load_yaml_config
from scripts.run_gatemem_suite import _discover_api_keys


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def prepare(manifest_path, output, disable_graph=False):
    import yaml
    manifest = json.loads(manifest_path.read_text())
    identity_path = output / 'execution_identity.json'
    if identity_path.exists():
        identity = json.loads(identity_path.read_text())
        if identity['manifest_sha256'] != hashlib.sha256(manifest_path.read_bytes()).hexdigest():
            raise ValueError('Manifest changed')
        return identity
    output.mkdir(parents=True, exist_ok=True)
    snapshot = output / 'runtime_snapshot'
    files = subprocess.check_output(['git','ls-files','--cached','--others','--exclude-standard','-z','--','src/gov_mem','run_govmem.py'],cwd=ROOT).decode().split('\0')
    digest = hashlib.sha256()
    for relative in sorted(set(files)):
        if not relative or not relative.endswith('.py'):
            continue
        source = ROOT / relative
        data = source.read_bytes()
        digest.update(relative.encode() + b'\0' + data)
        target = snapshot / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    jobs = []
    for domain, spec in manifest['domains'].items():
        source = ROOT / 'dataset/GateMem/gatemem/data' / domain
        episode_lines = [line for line in (source/'episodes.jsonl').read_text().splitlines() if line.strip()]
        checkpoint_lines = [line for line in (source/'checkpoints.jsonl').read_text().splitlines() if line.strip()]
        assert hashlib.sha256((source/'episodes.jsonl').read_bytes()).hexdigest() == spec['episodes_sha256']
        assert hashlib.sha256((source/'checkpoints.jsonl').read_bytes()).hexdigest() == spec['checkpoints_sha256']
        selected = set(spec['selected_episodes'])
        selected_eps = [line for line in episode_lines if json.loads(line)['episode_id'] in selected]
        selected_ckpts = [line for line in checkpoint_lines if json.loads(line)['episode_id'] in selected]
        dest = output/'selected_dataset'/domain
        dest.mkdir(parents=True,exist_ok=True)
        (dest/'episodes.jsonl').write_text('\n'.join(selected_eps)+'\n')
        (dest/'checkpoints.jsonl').write_text('\n'.join(selected_ckpts)+'\n')
        for episode in sorted(selected):
            eps = [line for line in selected_eps if json.loads(line)['episode_id']==episode]
            ckpts = [line for line in selected_ckpts if json.loads(line)['episode_id']==episode]
            ids = [json.loads(line)['checkpoint_id'] for line in ckpts]
            expected = [r['checkpoint_id'] for r in manifest['entries'] if r['domain']==domain and r['episode_id']==episode]
            assert ids == expected and len(eps)==1 and len(ids)==spec['selected_episodes'][episode]
            epdata = output/'episode_datasets'/episode/domain
            epdata.mkdir(parents=True,exist_ok=True)
            (epdata/'episodes.jsonl').write_text('\n'.join(eps)+'\n')
            (epdata/'checkpoints.jsonl').write_text('\n'.join(ckpts)+'\n')
            epmanifest = output/'episode_manifests'/f'{episode}.json'
            dump(epmanifest, {'selection_unit':'complete_episode','episode_id':episode,'checkpoint_count':len(ids),'checkpoint_ids':ids})
            jobs.append({'domain':domain,'episode_id':episode,'checkpoint_count':len(ids),'manifest':str(epmanifest),'data_dir':str(epdata)})
    v8 = load_yaml_config(ROOT/'configs/govmem_v8_late_governance_gemini25flashlite.yaml')
    if disable_graph:
        v8.setdefault('memory_governed_slot_graph', {})['enabled'] = False
    v8['embedding']['cache_dir'] = str(output/'embedding_cache')
    v8.setdefault('pipeline', {})['record_execution_errors'] = True
    v8['evaluation']['official_judge']['concurrency'] = 3
    v8path = output/'configs/govmem.yaml'; v8path.parent.mkdir(parents=True,exist_ok=True)
    v8path.write_text(yaml.safe_dump(v8,sort_keys=False))
    baseline = load_yaml_config(ROOT/'configs/gatemem_official_rag_naive_openlux_gemini25flashlite.yaml')
    baseline.update(temperature=0.0,max_retries=2,use_llm_judge=False,episode_concurrency=1,judge_concurrency=3)
    baseline.pop('out_dir',None)
    baselinepath=output/'configs/rag_naive_official.yaml'
    baselinepath.write_text(yaml.safe_dump(baseline,sort_keys=False))
    shutil.copy2(manifest_path,output/'manifest.json')
    identity={'manifest_sha256':hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
              'runtime_sha256':digest.hexdigest(),'git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
              'started_unix':time.time(),'jobs':jobs,'memory_model':'gemini-2.5-flash-lite','temperature':0,
              'embedding_model':'text-embedding-3-small','top_k':20,'judge_model':'gpt-4o','gate_by_action':False,
              'baseline':'vendored official rag_naive, unchanged; paired temperature 0.0',
              'govmem_config_sha256':hashlib.sha256(v8path.read_bytes()).hexdigest(),
              'baseline_config_sha256':hashlib.sha256(baselinepath.read_bytes()).hexdigest()}
    dump(identity_path,identity)
    return identity


def run_job(system, job, output, identity, key, resume):
    destination=output/system/job['domain']/job['episode_id']
    marker=destination/'complete.json'
    if marker.exists():
        print('COMPLETE cached',system,job['episode_id'],flush=True)
        return
    destination.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy()
    env.update(OPENLUX_API_KEY=key,OPENLUX_API_KEYS=key,PYTHONUNBUFFERED='1',
               GOVMEM_RUNTIME_FINGERPRINT=identity['runtime_sha256'],
               GOVMEM_DATASET_ROOT=str(output/'selected_dataset'),
               GOVMEM_OFFICIAL_BENCHMARK_ROOT=str(ROOT/'third_party/GateMem-official'),
               OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    entry=ROOT/'scripts/run_metered_entrypoint.py'
    cmd=[sys.executable,str(entry),'--telemetry',str(destination/'http_telemetry.jsonl'),'--reuse_connections','--prefer_ip','15.204.105.134']
    if system=='govmem':
        cmd += ['--capture_dir',str(destination/'raw_chat_responses'),'--',str(output/'runtime_snapshot/run_govmem.py'),
                '--dataset_name','checkpoint_benchmark','--data_path',job['data_dir'],
                '--output_dir',str(destination),'--config',str(output/'configs/govmem.yaml'),
                '--experiment_mode','govmem_v8_late_governance','--stage','all',
                '--checkpoint_manifest',job['manifest'],'--skip_official_eval']
        pred=destination/'predictions/checkpoint_benchmark/predictions.jsonl'
        if (destination/'run_metadata.json').exists():
            if not resume: raise ValueError('Existing incomplete V8 run requires --resume')
            cmd += ['--resume']
    else:
        cmd += ['--',str(ROOT/'third_party/GateMem-official/bench/scripts/run_eval.py'),
                '--config',str(output/'configs/rag_naive_official.yaml'),'--data_dir',job['data_dir'],
                '--out_dir',str(destination),'--run_name','inference','--no_progress','--log_file','run.log']
        pred=destination/'inference/predictions.jsonl'
        if pred.exists():
            if not resume: raise ValueError('Existing baseline requires --resume')
            cmd += ['--resume']
    print('START',system,job['episode_id'],job['checkpoint_count'],'checkpoints',flush=True)
    started=time.time()
    with (destination/'process.log').open('a') as log:
        result=subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f'{system}/{job["episode_id"]} failed ({result.returncode}); inspect {destination}/process.log')
    rows=[json.loads(line) for line in pred.read_text().splitlines() if line.strip()]
    expected=json.loads(Path(job['manifest']).read_text())['checkpoint_ids']
    if len(rows)!=len(expected) or {r['checkpoint_id'] for r in rows}!=set(expected):
        raise RuntimeError('Incomplete episode predictions; refusing success')
    errors=sum(row.get('execution_status')=='error' for row in rows)
    dump(marker,{'prediction_path':str(pred),'checkpoint_count':len(rows),'execution_error_count':errors,
                 'successful_checkpoint_count':len(rows)-errors,'wall_time_s':time.time()-started,
                 'runtime_sha256':identity['runtime_sha256']})
    print('DONE',system,job['episode_id'],len(rows),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--system',choices=['prepare','govmem','rag_naive'],default='prepare')
    parser.add_argument('--episode')
    parser.add_argument('--workers',type=int,default=1)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--key_offset',type=int,default=0)
    parser.add_argument('--disable-graph',action='store_true',help='freeze a graph-disabled Gov-Mem configuration')
    args=parser.parse_args(); output=args.output.resolve()
    identity=prepare(args.manifest.resolve(),output,args.disable_graph)
    if args.system=='prepare':
        print(json.dumps({'output':str(output),'runtime_sha256':identity['runtime_sha256'],'episodes':len(identity['jobs']),'checkpoints':sum(j['checkpoint_count'] for j in identity['jobs'])}));return
    keys=_discover_api_keys(provider='openlux')[max(0,args.key_offset):]
    if not keys:raise RuntimeError('No OpenLux credentials configured')
    jobs=[j for j in identity['jobs'] if not args.episode or j['episode_id']==args.episode]
    if not jobs:raise ValueError('Episode not in fixed manifest')
    # Bounded rounds stop scheduling new episodes after any failed round.
    width=min(max(1,args.workers),len(keys),len(jobs))
    if len(set(keys[:width])) != width:
        raise ValueError('Each concurrent episode requires a distinct API key')
    print(f'CONCURRENCY episodes={len(jobs)} workers={width} distinct_keys={width}',flush=True)
    for start in range(0,len(jobs),width):
        with ThreadPoolExecutor(max_workers=width) as pool:
            futures=[pool.submit(run_job,args.system,j,output,identity,keys[i],args.resume) for i,j in enumerate(jobs[start:start+width])]
            failures=[]
            for f in as_completed(futures):
                try:f.result()
                except Exception as exc:failures.append(str(exc))
            if failures:raise RuntimeError('; '.join(failures))


if __name__=='__main__':main()
