"""Keep eight episode workers occupied while preserving the frozen V8 run.

Resume only unattempted checkpoints after terminal errors have been recorded.
Stop a worker on infrastructure failure without prediction progress. Never
delete/retry error predictions. This changes scheduling, not model inference.
"""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
import argparse
import fcntl
import json
import os
from pathlib import Path
import queue
import shutil
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
from scripts.run_v8_full_evaluation import disk2, verify, score, sha
from scripts.run_v8_paired_episode_suite import run_job, dump
from scripts.run_gatemem_suite import _discover_api_keys


def prediction_ids(destination):
    path = destination / 'predictions/checkpoint_benchmark/predictions.jsonl'
    if not path.exists():
        return set()
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    ids = {row['checkpoint_id'] for row in rows}
    if len(ids) != len(rows):
        raise ValueError('Duplicate prediction identity')
    return ids


def resume_episode(job, output, identity, key, recovery_lock, stop):
    destination = output / 'govmem' / job['domain'] / job['episode_id']
    while not stop.is_set():
        before = prediction_ids(destination)
        try:
            run_job('govmem', job, output, identity, key, True)
            return
        except RuntimeError as exc:
            after = prediction_ids(destination)
            if not before < after:
                raise RuntimeError('No new terminal predictions; stop rather than repeat paid attempts') from exc
            log = (destination / 'process.log').read_text()
            if 'Aborted incomplete run after 3 consecutive execution errors' not in log:
                raise
            # The runner's strict resume skips both successes and errors. State
            # and costs are retained, and each continuation must make progress.
            record = {'at_unix': time.time(), 'episode_id': job['episode_id'],
                      'before': len(before), 'after': len(after),
                      'reason': 'Continue remaining queries after recorded terminal errors; no prediction retried'}
            with recovery_lock:
                with (output / 'recovery_events.jsonl').open('a') as handle:
                    handle.write(json.dumps(record) + '\n')
            print('CONTINUE_UNATTEMPTED', job['episode_id'], len(after), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = disk2(args.output)
    os.environ.update(PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1',
                      TMPDIR=str(output / 'tmp'), TMP=str(output / 'tmp'), TEMP=str(output / 'tmp'))
    with (output / 'orchestrator.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            identity = verify(output)
            keys = _discover_api_keys(provider='openlux')
            if not keys:
                raise ValueError('No API credentials')
            protocol = {'driver_sha256': sha(Path(__file__)), 'workers': min(8, len(keys)),
                        'scope': 'Scheduling only; unchanged frozen model/config/scorer; retain terminal errors',
                        'resume_policy': 'Only remaining queries, only after progress and the consecutive-error stop; no-progress stops scheduling'}
            protocol_path = output / 'scheduling_amendment.json'
            if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
                raise ValueError('Continuation driver changed')
            dump(protocol_path, protocol)
            shutil.copyfile(__file__, output / 'harness_snapshot/scripts/continue_v8_full_evaluation.py')
            dump(output / 'run_status.json', {'stage': 'all', 'status': 'running', 'pid': os.getpid(),
                                             'driver': 'continuous_episode_queue', 'updated_unix': time.time()})
            jobs = queue.Queue()
            groups = [[j for j in identity['jobs'] if j['domain'] == d] for d in ('medical','office','education','household')]
            for i in range(max(map(len, groups))):
                for group in groups:
                    if i < len(group):
                        j = group[i]
                        if not (output / 'govmem' / j['domain'] / j['episode_id'] / 'complete.json').exists():
                            jobs.put(j)
            stop = threading.Event()
            recovery_lock = threading.Lock()
            def worker(index):
                while not stop.is_set():
                    try:
                        job = jobs.get_nowait()
                    except queue.Empty:
                        return
                    try:
                        resume_episode(job, output, identity, keys[index], recovery_lock, stop)
                    except Exception:
                        stop.set()
                        raise
                    finally:
                        jobs.task_done()
            with ThreadPoolExecutor(max_workers=protocol['workers']) as pool:
                futures = [pool.submit(worker, i) for i in range(protocol['workers'])]
                for f in futures:
                    f.result()
            if stop.is_set() or not jobs.empty():
                raise RuntimeError('Incomplete queue')
            verify(output)
            dump(output / 'inference_complete.json', {'episodes': 91, 'checkpoints': 2218, 'finished_unix': time.time()})
            score(output, keys)
            dump(output / 'run_status.json', {'stage': 'all', 'status': 'complete', 'updated_unix': time.time()})
        except Exception as exc:
            dump(output / 'run_status.json', {'stage': 'all', 'status': 'failed', 'error_type': type(exc).__name__,
                                             'error': str(exc), 'updated_unix': time.time()})
            raise


if __name__ == '__main__':
    main()
