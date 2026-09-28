"""Reusable full GateMem baseline matrix with method/model separation.

Official agents and prompts are frozen per run. Narrow, recorded patches adapt
Mem0's internal provider/parameters and preserve terminal checkpoint errors.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import difflib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import queue
import shutil
import subprocess
import sys
import threading
import time
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from scripts.run_gatemem_suite import _discover_api_keys
from scripts.run_v8_full_evaluation import full_manifest

METHODS = ('a_mem', 'mem0', 'remem_i', 'remem_s')
DOMAINS = ('medical', 'office', 'education', 'household')
METHOD_DIRS = {'a_mem': 'A-Mem', 'mem0': 'Mem0', 'remem_i': 'ReMem-I', 'remem_s': 'ReMem-S'}
DEFAULT_MODEL = ROOT / 'baselines/models/openlux_gemini25flashlite.yaml'
DEPS = Path('/mnt/data_disk_2/fuyali/govmem_caches/baseline_python_deps')


def read(p):
    return json.loads(Path(p).read_text())


def rows(p):
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]


def dump(p, obj):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def disk2(p):
    p = Path(p).resolve()
    if not p.is_relative_to('/mnt/data_disk_2'):
        raise ValueError(f'Writable location outside data_disk_2: {p}')
    return p


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError('Upstream changed: patch no longer matches exactly once')
    return text.replace(old, new)


def adapt_official(relative, text):
    if relative == 'bench/agents/mem0.py':
        text = replace_once(text, 'if p in {"openai", "gemini", "deepseek"}:',
                            'if p in {"openlux", "openai-compatible-openlux"}:\n            return "openai"\n        if p in {"openai", "gemini", "deepseek"}:')
        text = replace_once(text, '"temperature": 0.1,\n            "max_tokens": 2000,',
                            '"temperature": float(self.llm_router.config.temperature),\n            "max_tokens": int(self.llm_router.config.max_output_tokens),')
        text = replace_once(text, 'embed_openai_base = self._mem0_upstream_embed_api_base',
                            'embed_openai_base = self._mem0_upstream_embed_api_base\n        if raw_llm_provider in {"openlux", "openai-compatible-openlux"} and embed_openai_base:\n            embed_openai_base = embed_openai_base.rstrip("/")\n            if not embed_openai_base.endswith("/v1"):\n                embed_openai_base += "/v1"')
    if relative == 'bench/eval/runner.py':
        start = text.index('        t_ingest0 = time.perf_counter()')
        end_marker = '        query_s = time.perf_counter() - t_query0\n'
        end = text.index(end_marker, start) + len(end_marker)
        original = text[start:end]
        wrapped = ('        ingest_s = query_s = 0.0\n        failure_start = time.perf_counter()\n        try:\n'
                   + ''.join('    ' + line if line.strip() else line for line in original.splitlines(keepends=True))
                   + '        except Exception as exc:\n'
                   + '            logger.exception("Recorded checkpoint execution error: %s", ckpt.checkpoint_id)\n'
                   + '            query_s = time.perf_counter() - failure_start\n'
                   + '            output = {"action": "error", "answer": "", "answer_structured": {}, "used_record_ids": [],\n'
                   + '                      "execution_status": "error", "error_type": type(exc).__name__}\n'
                   + '            if prediction_path:\n'
                   + '                _append_jsonl(os.path.join(os.path.dirname(prediction_path), "execution_errors.jsonl"),\n'
                   + '                              {"checkpoint_id": ckpt.checkpoint_id, "error_type": type(exc).__name__, "error": str(exc)})\n')
        text = text[:start] + wrapped + text[end:]
    return text


def prepare(out, methods, model_path):
    if (out / 'protocol.json').exists():
        p = read(out / 'protocol.json')
        if p['methods'] != list(methods) or p['model_profile_sha256'] != digest(model_path):
            raise ValueError('Existing matrix differs; use its original method/model settings')
        return p
    if (out / 'runtime').exists():
        raise ValueError('Incomplete prepare; inspect before resuming')
    out.mkdir(parents=True, exist_ok=True)
    manifest = full_manifest()
    dump(out / 'manifest.json', manifest)
    upstream = ROOT / 'third_party/GateMem-official'
    source_hashes, runtime_hashes, patches = {}, {}, []
    for prefix in ('bench', 'third_party/mem0_upstream', 'configs'):
        for p in sorted((upstream / prefix).rglob('*')):
            if not p.is_file() or p.suffix not in {'.py', '.txt', '.yaml', '.json', '.md'} or '__pycache__' in p.parts:
                continue
            relative = str(p.relative_to(upstream))
            old = p.read_text()
            new = adapt_official(relative, old)
            dest = out / 'runtime' / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(new)
            source_hashes[relative] = digest(p)
            runtime_hashes[relative] = digest(dest)
            if new != old:
                patches.extend(difflib.unified_diff(old.splitlines(True), new.splitlines(True), fromfile=relative, tofile='frozen/' + relative))
    (out / 'compatibility.patch').write_text(''.join(patches))
    model = yaml.safe_load(model_path.read_text())
    if model['llm_provider'] != 'openlux' or model['api_base'] != 'https://api.openlux.ai/v1':
        raise ValueError('This matrix requires unified OpenLux')
    shutil.copyfile(model_path, out / 'model.yaml')
    configs = {}
    for method in methods:
        settings = yaml.safe_load((ROOT / 'baselines' / METHOD_DIRS[method] / 'method.yaml').read_text())
        overlap = settings.keys() & model.keys()
        if overlap:
            raise ValueError(f'Method/model configuration collision: {overlap}')
        config = model | settings
        path = out / 'configs' / (method + '.yaml')
        path.parent.mkdir(exist_ok=True)
        path.write_text(yaml.safe_dump(config, sort_keys=False))
        configs[method] = digest(path)
    jobs = []
    for domain in DOMAINS:
        source = ROOT / 'dataset/GateMem/gatemem/data' / domain
        selected = out / 'selected_dataset' / domain
        selected.mkdir(parents=True)
        for name in ('episodes.jsonl', 'checkpoints.jsonl'):
            shutil.copyfile(source / name, selected / name)
        eps, ckpts = rows(source / 'episodes.jsonl'), rows(source / 'checkpoints.jsonl')
        for ep in eps:
            eid = ep['episode_id']
            subset = [r for r in ckpts if r['episode_id'] == eid]
            dest = out / 'episode_datasets' / eid / domain
            dest.mkdir(parents=True)
            for filename, rs in (('episodes.jsonl', [ep]), ('checkpoints.jsonl', subset)):
                (dest / filename).write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rs))
            jobs.append({'domain': domain, 'episode_id': eid, 'data_dir': str(dest),
                         'checkpoint_ids': [r['checkpoint_id'] for r in subset]})
    harness = {}
    for p in sorted((ROOT / 'baselines').rglob('*')):
        if p.is_file() and p.suffix in {'.py', '.yaml', '.md'}:
            rel = p.relative_to(ROOT)
            target = out / 'harness_snapshot' / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(p, target)
            harness[str(rel)] = digest(target)
    protocol = {'methods': list(methods), 'jobs': jobs, 'queries_per_method': 2218, 'episodes_per_method': 91,
                'model_profile_sha256': digest(model_path), 'model': model['llm_model'], 'provider': 'OpenLux',
                'temperature': model['temperature'], 'judge': {'provider': 'OpenLux', 'model': 'gpt-4o', 'temperature': 0, 'gate_by_action': False},
                'source_hashes': source_hashes, 'runtime_hashes': runtime_hashes, 'config_hashes': configs,
                'manifest_sha256': digest(out / 'manifest.json'), 'harness_hashes': harness,
                'scope': 'Full historically exposed GateMem, not holdout; same provider/model settings as Gov-Mem',
                'compatibility': 'Mem0 OpenLux provider/embedding URL and generation parameter alignment; terminal checkpoint errors retained; all other agent algorithms/prompts unchanged',
                'fallbacks': 'Official method-internal heuristic/soft-error fallbacks preserved and audited separately',
                'cache': 'Only exact URL+payload embedding responses shared; no chat/prediction/state sharing',
                'created_unix': time.time()}
    dump(out / 'protocol.json', protocol)
    return protocol


def verify(out, protocol):
    for name, expected in protocol['runtime_hashes'].items():
        if digest(out / 'runtime' / name) != expected:
            raise ValueError('Frozen runtime changed: ' + name)
    for method, expected in protocol['config_hashes'].items():
        if digest(out / 'configs' / (method + '.yaml')) != expected:
            raise ValueError('Frozen config changed')
    if digest(out / 'manifest.json') != protocol['manifest_sha256'] or read(out / 'manifest.json') != full_manifest():
        raise ValueError('Dataset identity changed')
    for domain, spec in read(out / 'manifest.json')['domains'].items():
        for filename, k in (('episodes.jsonl', 'episodes_sha256'), ('checkpoints.jsonl', 'checkpoints_sha256')):
            if digest(out / 'selected_dataset' / domain / filename) != spec[k]:
                raise ValueError('Copied dataset changed')
    for job in protocol['jobs']:
        if [r['checkpoint_id'] for r in rows(Path(job['data_dir'])/'checkpoints.jsonl')] != job['checkpoint_ids']:
            raise ValueError('Episode coverage changed')
    for name, expected in protocol['harness_hashes'].items():
        if digest(ROOT / name) != expected:
            raise ValueError('Execution harness changed: ' + name)


def environment(out, key, dest):
    env = os.environ.copy()
    for k in ('OPENROUTER_API_KEY', 'OPENAI_BASE_URL', 'GEMINI_API_KEY', 'GOOGLE_API_KEY'):
        env.pop(k, None)
    env.update(OPENLUX_API_KEY=key, OPENLUX_API_KEYS=key, PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1',
               PYTHONPATH=str(DEPS), TMPDIR=str(out/'tmp'), TMP=str(out/'tmp'), TEMP=str(out/'tmp'),
               MEM0_DIR=str(dest/'mem0_state'), MEM0_TELEMETRY='false',
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    return env


def run_job(out, method, job, key):
    dest = disk2(out / 'runs' / method / job['domain'] / job['episode_id'])
    if (dest/'complete.json').exists():
        return
    dest.mkdir(parents=True, exist_ok=True)
    pred = dest / 'inference/predictions.jsonl'
    if pred.exists():
        raise ValueError(f'Incomplete official agent state needs inspection before replay: {dest}')
    env = environment(out, key, dest)
    cmd = [sys.executable, '-B', str(out/'harness_snapshot/baselines/transport.py'), '--output', str(dest),
           '--embedding-cache', str(out/'embedding_cache'), '--', str(out/'runtime/bench/scripts/run_eval.py'),
           '--config', str(out/'configs'/f'{method}.yaml'), '--data_dir', job['data_dir'],
           '--out_dir', str(dest), '--run_name', 'inference', '--no_progress', '--log_file', 'run.log']
    print('START', method, job['episode_id'], len(job['checkpoint_ids']), flush=True)
    start = time.time()
    with (dest/'process.log').open('a') as log:
        p = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    if p.returncode:
        raise RuntimeError(f'{method}/{job["episode_id"]} failed; inspect {dest}/process.log')
    predictions = rows(pred)
    ids = [r['checkpoint_id'] for r in predictions]
    if len(ids) != len(job['checkpoint_ids']) or set(ids) != set(job['checkpoint_ids']):
        raise ValueError('Incomplete episode predictions')
    errors = sum(r['output'].get('action') == 'error' for r in predictions)
    logtext = (dest/'process.log').read_text()
    fallback_lines = [line for line in logtext.splitlines() if any(s in line for s in ('falling back to heuristic', 'metadata LLM call failed', 'mem0 upstream add failed', 'mem0 upstream search failed'))]
    dump(dest/'complete.json', {'queries': len(ids), 'execution_errors': errors, 'prediction_sha256': digest(pred),
                              'fallback_or_memory_warning_count': len(fallback_lines), 'wall_time_s': time.time()-start})
    dump(dest/'memory_warning_audit.json', fallback_lines)
    print('DONE', method, job['episode_id'], len(ids), 'errors', errors, 'memory_warnings', len(fallback_lines), flush=True)


def infer(out, protocol, workers, pilot, keys):
    jobs = protocol['jobs']
    # First complete episode from each method is included in production results.
    if pilot:
        jobs = [next(j for j in jobs if j['domain'] == 'medical')]
    else:
        groups = [[j for j in jobs if j['domain'] == d] for d in DOMAINS]
        jobs = [g[i] for i in range(max(map(len, groups))) for g in groups if i < len(g)]
    pending = queue.Queue()
    for job in jobs:
        for method in protocol['methods']:
            if not (out/'runs'/method/job['domain']/job['episode_id']/'complete.json').exists():
                pending.put((method, job))
    stop = threading.Event()
    def work(i):
        while not stop.is_set():
            try:
                method, job = pending.get_nowait()
            except queue.Empty:
                return
            try:
                run_job(out, method, job, keys[i])
            except Exception:
                stop.set()
                raise
            finally:
                pending.task_done()
    # The launcher is explicitly configured by the experiment. Keep the key
    # count as the only cap; episode history remains sequential inside each
    # child process. Older runs were frozen with a 16-worker cap, but new
    # matrices may use the authorized 30-key setting.
    width = min(max(1, workers), len(keys))
    with ThreadPoolExecutor(max_workers=width) as pool:
        futures = [pool.submit(work, i) for i in range(width)]
        for f in futures:
            f.result()
    if stop.is_set() or not pending.empty():
        raise ValueError('Incomplete inference queue')


def score_one(out, method, domain, key, protocol):
    jobs = [j for j in protocol['jobs'] if j['domain'] == domain]
    merged = {}
    for job in jobs:
        source = out/'runs'/method/domain/job['episode_id']
        marker = read(source/'complete.json')
        path = source/'inference/predictions.jsonl'
        if digest(path) != marker['prediction_sha256']:
            raise ValueError('Prediction changed')
        for row in rows(path):
            if row['checkpoint_id'] in merged:
                raise ValueError('Duplicate checkpoint')
            merged[row['checkpoint_id']] = row
    order = [r['checkpoint_id'] for r in read(out/'manifest.json')['entries'] if r['domain']==domain]
    if set(order) != set(merged):
        raise ValueError('Incomplete full domain')
    dest = out/'scored'/method/domain
    dest.mkdir(parents=True, exist_ok=True)
    serialized = ''.join(json.dumps(merged[c], ensure_ascii=False)+'\n' for c in order)
    p = dest/'predictions.jsonl'
    if p.exists() and p.read_text() != serialized:
        raise ValueError('Scored predictions changed')
    p.write_text(serialized)
    cmd = [sys.executable, '-B', str(out/'harness_snapshot/baselines/transport.py'), '--output', str(dest),
           '--embedding-cache', str(out/'embedding_cache'), '--', str(out/'runtime/bench/scripts/score_predictions.py'),
           '--data_dir', str(out/'selected_dataset'/domain), '--predictions', str(p), '--out_dir', str(dest/'official_eval'),
           '--use_llm_judge', '--judge_provider', 'openlux', '--judge_model', 'gpt-4o', '--judge_temperature', '0',
           '--judge_max_output_tokens', '4096', '--judge_api_base', 'https://api.openlux.ai/v1',
           '--judge_api_key_env', 'OPENLUX_API_KEY', '--judge_concurrency', '3', '--resume_judge']
    print('JUDGE_START', method, domain, flush=True)
    with (dest/'judge.log').open('a') as log:
        p = subprocess.run(cmd, cwd=ROOT, env=environment(out, key, dest), stdout=log, stderr=subprocess.STDOUT)
    if p.returncode:
        raise RuntimeError(f'Judge failed: {method}/{domain}')
    judged = rows(dest/'official_eval/judge_scores.jsonl')
    if len(judged)!=len(order) or {r['checkpoint_id'] for r in judged}!=set(order):
        raise ValueError('Judge coverage mismatch')
    print('JUDGE_DONE', method, domain, len(judged), flush=True)


def summarize(out, protocol):
    result = {'scope': protocol['scope'], 'provider': protocol['provider'], 'model': protocol['model'],
              'temperature': protocol['temperature'], 'metric': 'Official GateMem only: U*(1-A)*(1-F), macro over domains', 'methods': {}}
    for method in protocol['methods']:
        ds = {}
        for domain in DOMAINS:
            dest = out/'scored'/method/domain
            summary = read(dest/'official_eval/summary.json')
            assert summary['gated_by_action'] is False
            judged = rows(dest/'official_eval/judge_scores.jsonl')
            predictions = rows(dest/'predictions.jsonl')
            missing = []
            for row in judged:
                key = {'utility':'utility_ok','privacy':'privacy_leak','safety':'deletion_leak'}[row['query_type']]
                if not row['judge'].get('parse_ok') or row['judge'].get(key) is None:
                    missing.append({'checkpoint_id':row['checkpoint_id'],'field':key})
            u,a,f = [summary[k] for k in ('utility_accuracy','privacy_leakage_rate','deletion_leakage_rate')]
            ds[domain] = {'queries':len(predictions),'U':u,'A':a,'F':f,'MGS':u*(1-a)*(1-f),
                          'execution_errors':sum(r['output'].get('action')=='error' for r in predictions),
                          'missing_applicable_labels':missing}
        result['methods'][method]={'per_domain':ds, 'macro':{k:sum(d[k] for d in ds.values())/4 for k in ('U','A','F','MGS')},
                                   'execution_errors':sum(d['execution_errors'] for d in ds.values())}
    costs = {}
    for p in out.glob('**/http_telemetry.jsonl'):
        for r in rows(p):
            c=costs.setdefault(r.get('requested_model','unknown'),{'requests':0,'reported_tokens':0,'http_failures':0,'missing_usage':0})
            c['requests']+=1;c['reported_tokens']+=(r.get('usage')or{}).get('total_tokens',0)
            c['http_failures']+=int(bool(r.get('error_type')) or r.get('status_code',200)>=400)
            c['missing_usage']+=int(not r.get('usage'))
    result['costs']=costs
    dump(out/'official_metrics.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


def main(default_method=None):
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--methods',nargs='+',choices=METHODS,default=[default_method] if default_method else list(METHODS))
    p.add_argument('--model-profile',type=Path,default=DEFAULT_MODEL)
    p.add_argument('--stage',choices=['prepare','pilot','infer','score','all','summarize'],default='prepare')
    p.add_argument('--workers',type=int,default=8)
    args=p.parse_args();out=disk2(args.output);out.mkdir(parents=True,exist_ok=True)
    disk2(out/'tmp').mkdir(exist_ok=True)
    with (out/'orchestrator.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            protocol=prepare(out,args.methods,args.model_profile.resolve());verify(out,protocol)
            dump(out/'run_status.json',{'status':'running','stage':args.stage,'pid':os.getpid(),'updated_unix':time.time()})
            print('MATRIX',protocol['methods'],protocol['model'],'queries_each',2218,'stage',args.stage,flush=True)
            if args.stage not in {'prepare','summarize'}:
                keys=_discover_api_keys(provider='openlux')
                if not keys:raise ValueError('Missing OpenLux credentials')
                if args.stage in {'pilot','infer','all'}:infer(out,protocol,args.workers,args.stage=='pilot',keys)
                if args.stage in {'score','all'}:
                    with ThreadPoolExecutor(max_workers=4) as pool:
                        futures=[pool.submit(score_one,out,m,d,keys[(i+16)%len(keys)],protocol)
                                 for i,(m,d) in enumerate((m,d) for m in protocol['methods'] for d in DOMAINS)]
                        for f in futures:f.result()
            if args.stage in {'score','all','summarize'}:summarize(out,protocol)
            dump(out/'run_status.json',{'status':'complete','stage':args.stage,'updated_unix':time.time()})
        except Exception as exc:
            dump(out/'run_status.json',{'status':'failed','stage':args.stage,'error_type':type(exc).__name__,'error':str(exc),'updated_unix':time.time()})
            raise


if __name__=='__main__':
    main()
