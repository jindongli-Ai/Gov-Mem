"""Finish the authorized A-Mem matrix while preserving its live pilot episode.

The source orchestrator owns only the pilot; this process owns the other 90
episodes in a separate frozen output. No live memory state is copied/resumed.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from baselines import run as matrix


def import_pilot(source, out, job, source_protocol, protocol):
    for field in ('runtime_hashes', 'manifest_sha256', 'model_profile_sha256'):
        if source_protocol[field] != protocol[field]:
            raise ValueError('Pilot identity differs: ' + field)
    if source_protocol['config_hashes']['a_mem'] != protocol['config_hashes']['a_mem']:
        raise ValueError('Pilot A-Mem config differs')
    relative = Path('runs/a_mem') / job['domain'] / job['episode_id']
    src, dest = source / relative, out / relative
    if (dest / 'complete.json').exists():
        return
    deadline = time.monotonic() + 12 * 3600
    while not (src / 'complete.json').exists():
        if time.monotonic() > deadline:
            raise TimeoutError('Pilot still incomplete after 12 hours; inspect without replay')
        status = matrix.read(source / 'run_status.json')
        if status['status'] != 'running':
            raise RuntimeError('Source pilot stopped before completion')
        os.kill(status['pid'], 0)
        time.sleep(10)
    marker = matrix.read(src / 'complete.json')
    predictions = src / 'inference/predictions.jsonl'
    ids = [r['checkpoint_id'] for r in matrix.rows(predictions)]
    if len(ids) != len(job['checkpoint_ids']) or set(ids) != set(job['checkpoint_ids']):
        raise ValueError('Pilot coverage mismatch')
    if matrix.digest(predictions) != marker['prediction_sha256']:
        raise ValueError('Pilot prediction hash mismatch')
    temporary = dest.with_name(dest.name + '.importing')
    if dest.exists() or temporary.exists():
        raise ValueError('Partial pilot import exists; inspect before resuming')
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, temporary)
    if matrix.digest(temporary / 'inference/predictions.jsonl') != marker['prediction_sha256']:
        raise ValueError('Copied pilot prediction hash mismatch')
    temporary.rename(dest)
    matrix.dump(out / 'pilot_import.json', {
        'source': str(src), 'source_protocol_sha256': matrix.digest(source / 'protocol.json'),
        'prediction_sha256': marker['prediction_sha256'], 'queries': len(ids),
        'imported_unix': time.time(), 'rerun': False,
    })
    print('PILOT_IMPORTED', job['episode_id'], len(ids), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    source, out = matrix.disk2(args.source), matrix.disk2(args.output)
    out.mkdir(parents=True, exist_ok=True)
    matrix.disk2(out / 'tmp').mkdir(exist_ok=True)
    with (out / 'orchestrator.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        def status(stage, **extra):
            matrix.dump(out / 'run_status.json', dict(stage=stage, pid=os.getpid(),
                        updated_unix=time.time(), **extra))
        try:
            sp = matrix.read(source / 'protocol.json')
            matrix.verify(source, sp)
            protocol = matrix.prepare(out, ['a_mem'], matrix.DEFAULT_MODEL)
            matrix.verify(out, protocol)
            pilot = next(j for j in protocol['jobs'] if j['domain'] == 'medical')
            amendment = out / 'scheduling_provenance.json'
            if not amendment.exists():
                matrix.dump(amendment, {
                    'source': str(source), 'pilot_episode': pilot['episode_id'],
                    'workers': args.workers, 'created_unix': time.time(),
                    'policy': 'Import complete pilot unchanged; run other 90 episodes once; official scoring after 2218 predictions',
                    'transport': 'Original frozen transport; no DNS, retry, timeout, prompt or algorithm change',
                    'embedding_cache': 'Independent output cache; imported pilot retains its original shared-cache provenance',
                })
                shutil.copyfile(__file__, out / 'continuation_snapshot.py')
            elif matrix.digest(__file__) != matrix.digest(out / 'continuation_snapshot.py'):
                raise ValueError('Continuation script changed')
            keys = matrix._discover_api_keys(provider='openlux')
            if not keys:
                raise ValueError('Missing OpenLux credentials')
            status('inference', status='running')
            print('A_MEM_FULL_START', '91 episodes / 2218 queries', 'workers', args.workers, flush=True)
            remaining = dict(protocol, jobs=[j for j in protocol['jobs'] if j['episode_id'] != pilot['episode_id']])
            with ThreadPoolExecutor(max_workers=2) as pool:
                adoption = pool.submit(import_pilot, source, out, pilot, sp, protocol)
                inference = pool.submit(matrix.infer, out, remaining, args.workers, False, keys)
                inference.result()
                adoption.result()
            status('official_scoring', status='running')
            with ThreadPoolExecutor(max_workers=4) as pool:
                jobs = [pool.submit(matrix.score_one, out, 'a_mem', domain, keys[i % len(keys)], protocol)
                        for i, domain in enumerate(matrix.DOMAINS)]
                for job in jobs:
                    job.result()
            matrix.summarize(out, protocol)
            status('complete', status='complete')
        except Exception as exc:
            status('failed', status='failed', error_type=type(exc).__name__, error=str(exc))
            raise


if __name__ == '__main__':
    main()
