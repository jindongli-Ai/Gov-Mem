"""Run one frozen baseline with 20 distinct keys, importing a completed pilot."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import os
from pathlib import Path
import queue
import shutil
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from baselines import run as m


def import_pilot(source, out, method, protocol):
    previous = m.read(source / 'protocol.json')
    for field in ('runtime_hashes', 'manifest_sha256', 'model_profile_sha256'):
        if previous[field] != protocol[field]:
            raise ValueError('Pilot identity differs: ' + field)
    if previous['config_hashes'][method] != protocol['config_hashes'][method]:
        raise ValueError('Pilot method config differs')
    job = next(j for j in protocol['jobs'] if j['domain'] == 'medical')
    relative = Path('runs') / method / job['domain'] / job['episode_id']
    src, dest = source / relative, out / relative
    marker = m.read(src / 'complete.json')
    pred = src / 'inference/predictions.jsonl'
    ids = [r['checkpoint_id'] for r in m.rows(pred)]
    if len(ids) != len(job['checkpoint_ids']) or set(ids) != set(job['checkpoint_ids']):
        raise ValueError('Pilot checkpoint coverage differs')
    if m.digest(pred) != marker['prediction_sha256']:
        raise ValueError('Pilot prediction hash differs')
    if dest.exists():
        if m.read(dest / 'complete.json') != marker or m.digest(dest / 'inference/predictions.jsonl') != marker['prediction_sha256']:
            raise ValueError('Existing pilot import differs')
        return
    temporary = dest.with_name(dest.name + '.importing')
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, temporary)
    if m.digest(temporary / 'inference/predictions.jsonl') != marker['prediction_sha256']:
        raise ValueError('Copied pilot prediction hash differs')
    temporary.rename(dest)
    m.dump(out / 'pilot_import.json', {'source': str(src), 'queries': len(ids), 'rerun': False,
           'source_protocol_sha256': m.digest(source / 'protocol.json'),
           'prediction_sha256': marker['prediction_sha256'], 'imported_unix': time.time()})


def infer(out, protocol, method, keys, workers):
    pending = queue.Queue()
    groups = [[j for j in protocol['jobs'] if j['domain'] == d] for d in m.DOMAINS]
    for i in range(max(map(len, groups))):
        for group in groups:
            if i < len(group):
                job = group[i]
                if not (out / 'runs' / method / job['domain'] / job['episode_id'] / 'complete.json').exists():
                    pending.put(job)
    stop = threading.Event()
    def worker(index):
        while not stop.is_set():
            try:
                job = pending.get_nowait()
            except queue.Empty:
                return
            try:
                m.run_job(out, method, job, keys[index])
            except Exception:
                stop.set()
                raise
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(worker, i) for i in range(workers)]
        for future in futures:
            future.result()
    if stop.is_set() or not pending.empty():
        raise RuntimeError('Inference incomplete')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', choices=m.METHODS, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=20)
    args = parser.parse_args()
    source, out = m.disk2(args.source), m.disk2(args.output)
    out.mkdir(parents=True, exist_ok=True)
    m.disk2(out / 'tmp').mkdir(exist_ok=True)
    with (out / 'orchestrator.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        def status(stage, **extra):
            m.dump(out / 'run_status.json', dict(stage=stage, pid=os.getpid(), workers=args.workers,
                                               updated_unix=time.time(), **extra))
        try:
            keys = m._discover_api_keys(provider='openlux')
            if not 1 <= args.workers <= len(keys) or len(set(keys[:args.workers])) != args.workers:
                raise ValueError('Insufficient distinct keys for requested workers')
            # The historical four-method pilot protocol predates directory
            # capitalization (a_mem -> A-Mem, remem_i -> ReMem-I). Its
            # frozen runtime/config/data hashes remain authoritative, while
            # its harness path list is only provenance and cannot be checked
            # against the renamed live source tree.
            source_protocol = m.read(source / 'protocol.json')
            for name, expected in source_protocol['runtime_hashes'].items():
                if m.digest(source / 'runtime' / name) != expected:
                    raise ValueError('Pilot runtime changed: ' + name)
            if m.digest(source / 'manifest.json') != source_protocol['manifest_sha256']:
                raise ValueError('Pilot manifest changed')
            if m.digest(source / 'configs' / (args.method + '.yaml')) != source_protocol['config_hashes'][args.method]:
                raise ValueError('Pilot method config changed')
            protocol = m.prepare(out, [args.method], m.DEFAULT_MODEL)
            m.verify(out, protocol)
            snapshot = out / 'full_runner_snapshot.py'
            if snapshot.exists() and m.digest(snapshot) != m.digest(__file__):
                raise ValueError('Full runner changed')
            if not snapshot.exists():
                shutil.copyfile(__file__, snapshot)
                m.dump(out / 'scheduling_provenance.json', {
                    'method': args.method, 'workers': args.workers, 'source': str(source),
                    'created_unix': time.time(), 'policy': 'Import completed first Medical pilot unchanged; run remaining episodes once with distinct keys; automatic official scoring',
                    'transport': 'Unchanged frozen transport; no DNS, timeout, retry, algorithm or prompt changes',
                    'cache': 'Independent exact embedding cache; imported pilot retains source cache provenance'})
            import_pilot(source, out, args.method, protocol)
            status('inference', status='running')
            print('FULL_START', args.method, '91 episodes / 2218 queries', 'distinct_keys', args.workers, flush=True)
            infer(out, protocol, args.method, keys, args.workers)
            status('official_scoring', status='running')
            with ThreadPoolExecutor(max_workers=4) as pool:
                futures = [pool.submit(m.score_one, out, args.method, domain, keys[i % len(keys)], protocol)
                           for i, domain in enumerate(m.DOMAINS)]
                for future in futures:
                    future.result()
            m.summarize(out, protocol)
            status('complete', status='complete')
        except Exception as exc:
            status('failed', status='failed', error_type=type(exc).__name__, error=str(exc))
            raise


if __name__ == '__main__':
    main()
