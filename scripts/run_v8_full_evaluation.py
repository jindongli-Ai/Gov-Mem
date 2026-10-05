"""Frozen current V8 on all 91 GateMem episodes / 2218 checkpoints.

Reuses the complete-episode runner and unchanged official scoring. No training,
model search, old prediction reuse, or baseline inference is performed.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
from scripts.run_v8_paired_episode_suite import prepare, run_job, dump
from scripts.score_v8_paired_episode_suite import merge, judge, rows, read
from scripts.score_v8_module_ablations import component_metrics
from scripts.run_gatemem_suite import _discover_api_keys

DOMAINS = ('medical', 'office', 'education', 'household')
HARNESS = ('scripts/run_v8_full_evaluation.py', 'scripts/run_v8_paired_episode_suite.py',
           'scripts/score_v8_paired_episode_suite.py', 'scripts/score_v8_module_ablations.py',
           'scripts/run_metered_entrypoint.py', 'scripts/run_gatemem_suite.py')


def disk2(path):
    path = Path(path).resolve()
    if not path.is_relative_to('/mnt/data_disk_2'):
        raise ValueError(f'Writable path outside data_disk_2: {path}')
    return path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def full_manifest():
    result = {'schema_version': 'govmem-full-complete-episodes-1',
              'selection_unit': 'complete_episodes', 'seed': None,
              'scope': 'Full GateMem; 91 complete episodes / 2218 checkpoints; historically exposed, not holdout',
              'domains': {}, 'entries': []}
    seen_episodes, seen_ids = set(), set()
    for domain in DOMAINS:
        source = ROOT / 'dataset/GateMem/gatemem/data' / domain
        episodes = rows(source / 'episodes.jsonl')
        counts = {r['episode_id']: 0 for r in episodes}
        if len(counts) != len(episodes) or seen_episodes & counts.keys():
            raise ValueError('Duplicate episode identity')
        seen_episodes.update(counts)
        for r in rows(source / 'checkpoints.jsonl'):
            if r['checkpoint_id'] in seen_ids or r['episode_id'] not in counts:
                raise ValueError('Invalid checkpoint identity')
            seen_ids.add(r['checkpoint_id'])
            counts[r['episode_id']] += 1
            result['entries'].append({k: r[k] for k in ('checkpoint_id', 'episode_id')} | {'domain': domain})
        if not all(counts.values()):
            raise ValueError('Empty episode')
        result['domains'][domain] = {'selected_episodes': counts, 'checkpoint_count': sum(counts.values()),
            'episodes_sha256': sha(source / 'episodes.jsonl'),
            'checkpoints_sha256': sha(source / 'checkpoints.jsonl')}
    if len(seen_episodes) != 91 or len(seen_ids) != 2218:
        raise ValueError('Full dataset differs from requested 91 episodes / 2218 queries')
    return result


def prepare_full(output):
    import yaml
    if (output / 'full_prepared.json').exists():
        return
    if (output / 'execution_identity.json').exists():
        raise ValueError('Incomplete preparation; inspect before resuming')
    manifest = full_manifest()
    dump(output / 'full_manifest.json', manifest)
    identity = prepare(output / 'full_manifest.json', output)
    # Copy only existing content-addressed embeddings; never old predictions or state.
    cache = disk2(output / 'embedding_cache')
    cache.mkdir(exist_ok=True)
    copied = 0
    for name in ('v8_shallow_source_projection_random3_20260919', 'v8_update_retention_confirmation_20260919'):
        source = disk2(ROOT / 'outputs' / name / 'embedding_cache')
        for path in source.rglob('*.json'):
            disk2(path)
            target = disk2(cache / path.relative_to(source))
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
                copied += 1
    cfg = yaml.safe_load((output / 'configs/govmem.yaml').read_text())
    assert cfg['v8_memory']['mode'] == 'shallow'
    assert cfg['llm']['base_model'] == 'gemini-2.5-flash-lite'
    assert cfg['llm']['temperature'] == 0
    disk2(cfg['embedding']['cache_dir'])
    protocol = {'scope': manifest['scope'], 'system': 'current default V8 only',
                'runtime_sha256': identity['runtime_sha256'], 'config_sha256': identity['govmem_config_sha256'],
                'model': 'gemini-2.5-flash-lite', 'temperature': 0, 'judge': 'gpt-4o',
                'gate_by_action': False, 'training': False, 'mid_run_tuning': False,
                'execution_errors': 'retain every attempted query; report official and pessimistic MGS',
                'baseline': 'not run; historical small-sample baseline is not comparable',
                'copied_embedding_files': copied, 'harness': {p: sha(ROOT / p) for p in HARNESS}}
    bench = ROOT / 'third_party/GateMem-official/bench'
    protocol['official_files'] = {str(p.relative_to(bench)): sha(p) for p in bench.rglob('*')
                                  if p.is_file() and p.suffix in {'.py', '.txt'}}
    for name in HARNESS:
        dest = output / 'harness_snapshot' / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, dest)
    dump(output / 'protocol.json', protocol)
    dump(output / 'full_prepared.json', {'episodes': 91, 'checkpoints': 2218, 'prepared_unix': time.time()})


def verify(output):
    identity = read(output / 'execution_identity.json')
    protocol = read(output / 'protocol.json')
    digest = hashlib.sha256()
    snapshot = output / 'runtime_snapshot'
    for path in sorted(snapshot.rglob('*.py'), key=lambda p: str(p.relative_to(snapshot))):
        digest.update(str(path.relative_to(snapshot)).encode() + b'\0' + path.read_bytes())
    if digest.hexdigest() != identity['runtime_sha256'] or digest.hexdigest() != protocol['runtime_sha256']:
        raise ValueError('Frozen runtime changed')
    if sha(output / 'configs/govmem.yaml') != protocol['config_sha256']:
        raise ValueError('Frozen config changed')
    if sha(output / 'manifest.json') != identity['manifest_sha256'] or read(output / 'manifest.json') != full_manifest():
        raise ValueError('Dataset or manifest changed')
    for name, expected in protocol['harness'].items():
        if sha(ROOT / name) != expected:
            raise ValueError(f'Harness changed: {name}')
    bench = ROOT / 'third_party/GateMem-official/bench'
    for name, expected in protocol['official_files'].items():
        if sha(bench / name) != expected:
            raise ValueError(f'Official scorer changed: {name}')
    for job in identity['jobs']:
        directory = disk2(job['data_dir'])
        episode = job['episode_id']
        for filename in ('episodes.jsonl', 'checkpoints.jsonl'):
            source = ROOT / 'dataset/GateMem/gatemem/data' / job['domain'] / filename
            expected = [r for r in rows(source) if r['episode_id'] == episode]
            if rows(directory / filename) != expected:
                raise ValueError('Episode dataset copy changed')
        ids = [r['checkpoint_id'] for r in rows(directory / 'checkpoints.jsonl')]
        if read(disk2(job['manifest']))['checkpoint_ids'] != ids:
            raise ValueError('Episode manifest changed')
    return identity


def infer(output, identity, workers, keys):
    # Round-robin domains, preserving chronological processing within each episode.
    groups = [[j for j in identity['jobs'] if j['domain'] == d] for d in DOMAINS]
    jobs = [g[i] for i in range(max(map(len, groups))) for g in groups if i < len(g)]
    jobs = [j for j in jobs if not (output / 'govmem' / j['domain'] / j['episode_id'] / 'complete.json').exists()]
    # Keep one isolated provider key per episode worker.  The project-wide
    # protocol permits up to 30 distinct keys; callers still choose a smaller
    # width when fewer independent jobs remain.
    width = min(max(1, workers), 30, len(keys))
    for start in range(0, len(jobs), width):
        with ThreadPoolExecutor(max_workers=width) as pool:
            futures = [pool.submit(run_job, 'govmem', job, output, identity, keys[i], True)
                       for i, job in enumerate(jobs[start:start + width])]
            for f in futures:
                f.result()
    dump(output / 'inference_complete.json', {'episodes': 91, 'checkpoints': 2218, 'finished_unix': time.time()})


def score(output, keys):
    manifest = read(output / 'manifest.json')
    scored = output / 'scored'
    merge(output, 'govmem', manifest, scored)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(judge, 'govmem', d, output / 'selected_dataset', scored, keys[(i + 8) % len(keys)])
                   for i, d in enumerate(DOMAINS)]
        for f in futures:
            f.result()
    metrics = {}
    for d in DOMAINS:
        dest = scored / 'govmem' / d
        predictions = rows(dest / 'predictions.jsonl')
        judgments = rows(dest / 'official_eval/judge_scores.jsonl')
        errors = {r['checkpoint_id'] for r in predictions if r.get('execution_status') == 'error' or r['action'] == 'error'}
        m = component_metrics(judgments, errors)
        summary = read(dest / 'official_eval/summary.json')
        if summary.get('gated_by_action') is not False:
            raise ValueError('Wrong action-gating protocol')
        for key, official in [('U', 'utility_accuracy'), ('A', 'privacy_leakage_rate'), ('F', 'deletion_leakage_rate')]:
            if abs(m[key] - summary[official]) > 1e-10:
                raise ValueError('Official component mismatch')
        m.update(checkpoints=len(predictions), execution_errors=len(errors), action_accuracy=summary.get('action_accuracy'))
        metrics[d] = m
    costs = {}
    paths = list((output / 'govmem').glob('*/*/http_telemetry.jsonl')) + list(scored.glob('govmem/*/judge_http_telemetry.jsonl'))
    for path in paths:
        for r in rows(path):
            c = costs.setdefault(r.get('requested_model', 'unknown'), {'requests': 0, 'tokens': 0, 'http_failures': 0, 'missing_usage': 0})
            c['requests'] += 1
            c['tokens'] += (r.get('usage') or {}).get('total_tokens', 0)
            c['http_failures'] += int(bool(r.get('error_type')) or r.get('status_code', 200) >= 400)
            c['missing_usage'] += int(not r.get('usage'))
    result = {'scope': manifest['scope'], 'per_domain': metrics,
              'mean_MGS': sum(m['MGS'] for m in metrics.values()) / 4,
              'mean_pessimistic_MGS': sum(m['pessimistic_MGS'] for m in metrics.values()) / 4,
              'execution_errors': sum(m['execution_errors'] for m in metrics.values()),
              'costs': costs, 'runtime_sha256': read(output / 'execution_identity.json')['runtime_sha256']}
    dump(output / 'full_metrics.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--stage', choices=['prepare', 'infer', 'score', 'all'], default='prepare')
    p.add_argument('--workers', type=int, default=8)
    args = p.parse_args()
    output = disk2(args.output)
    output.mkdir(parents=True, exist_ok=True)
    temporary = disk2(output / 'tmp')
    temporary.mkdir(exist_ok=True)
    os.environ.update(TMPDIR=str(temporary), TMP=str(temporary), TEMP=str(temporary),
                      PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1')
    with (output / 'orchestrator.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            prepare_full(output)
            identity = verify(output)
            dump(output / 'run_status.json', {'stage': args.stage, 'status': 'running', 'pid': os.getpid(), 'updated_unix': time.time()})
            print(json.dumps({'stage': args.stage, 'episodes': 91, 'checkpoints': 2218,
                              'runtime_sha256': identity['runtime_sha256']}), flush=True)
            if args.stage != 'prepare':
                keys = _discover_api_keys(provider='openlux')
                if not keys:
                    raise ValueError('No API credentials')
                if args.stage in {'infer', 'all'}:
                    infer(output, identity, args.workers, keys)
                if args.stage in {'score', 'all'}:
                    score(output, keys)
            dump(output / 'run_status.json', {'stage': args.stage, 'status': 'complete', 'updated_unix': time.time()})
        except Exception as exc:
            dump(output / 'run_status.json', {'stage': args.stage, 'status': 'failed', 'error_type': type(exc).__name__,
                                            'error': str(exc), 'updated_unix': time.time()})
            raise


if __name__ == '__main__':
    main()
