"""Take over only the scheduler; preserve all live episode processes and state."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import os
from pathlib import Path
import queue
import shutil
import signal
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from baselines import run as m


def proc(pid):
    try:
        data = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return {'state': data[0], 'parent': int(data[1]), 'start': data[19]}
    except FileNotFoundError:
        return None


def finish_existing(out, job):
    dest = out / 'runs/a_mem' / job['domain'] / job['episode_id']
    pred = dest / 'inference/predictions.jsonl'
    records = m.rows(pred)
    ids = [r['checkpoint_id'] for r in records]
    if len(ids) != len(job['checkpoint_ids']) or set(ids) != set(job['checkpoint_ids']):
        raise ValueError('Adopted episode incomplete: ' + job['episode_id'])
    # Official run_eval writes this only after inference and its local scoring.
    summary = m.read(dest / 'inference/summary.json')
    if summary['n_checkpoints'] != len(ids):
        raise ValueError('Adopted episode summary coverage mismatch')
    warnings = [l for l in (dest / 'process.log').read_text().splitlines()
                if any(s in l for s in ('falling back to heuristic', 'metadata LLM call failed'))]
    m.dump(dest / 'memory_warning_audit.json', warnings)
    m.dump(dest / 'complete.json', {
        'queries': len(ids), 'execution_errors': sum(r['output'].get('action') == 'error' for r in records),
        'prediction_sha256': m.digest(pred), 'fallback_or_memory_warning_count': len(warnings),
        'wall_time_s': None, 'adopted_without_replay': True,
        'completion_evidence': 'Process exited; official terminal summary and full prediction identities verified; parent exit code unavailable',
    })
    print('ADOPTED_DONE', job['episode_id'], len(ids), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--old-pid', type=int, required=True)
    args = p.parse_args()
    out = m.disk2(args.output)
    protocol = m.read(out / 'protocol.json')
    m.verify(out, protocol)
    keys = m._discover_api_keys(provider='openlux')
    if len(set(keys)) < 20 or len(set(keys[:20])) != 20:
        raise ValueError('20 distinct keys required')
    amendment = out / 'scheduling_20keys.json'
    if amendment.exists():
        raise ValueError('Takeover already attempted; inspect before restarting')
    state = m.read(out / 'run_status.json')
    if state.get('pid') != args.old_pid or state.get('stage') != 'inference':
        raise ValueError('Unexpected original scheduler state')
    old_cmd = Path(f'/proc/{args.old_pid}/cmdline').read_bytes()
    if b'run_a_mem_full_from_pilot.py' not in old_cmd or str(out).encode() not in old_cmd:
        raise ValueError('Original PID identity mismatch')
    stopped = False
    killed = False
    try:
        os.kill(args.old_pid, signal.SIGSTOP)
        stopped = True
        for _ in range(100):
            if proc(args.old_pid)['state'] in ('T', 't'):
                break
            time.sleep(.01)
        else:
            raise RuntimeError('Scheduler did not stop')
        active = {}
        for entry in Path('/proc').iterdir():
            if not entry.name.isdigit():
                continue
            info = proc(entry.name)
            if not info or info['parent'] != args.old_pid or info['state'] == 'Z':
                continue
            cmd = (entry / 'cmdline').read_bytes().split(b'\0')
            if not any(b'transport.py' in s for s in cmd):
                raise ValueError('Unexpected child process')
            dest = Path(os.fsdecode(cmd[cmd.index(b'--output') + 1])).resolve()
            env = dict(s.split(b'=', 1) for s in (entry / 'environ').read_bytes().split(b'\0') if b'=' in s)
            index = keys.index(os.fsdecode(env[b'OPENLUX_API_KEY']))
            if index >= 20 or index in active:
                raise ValueError('Active key assignment not unique within target 20')
            job = next(j for j in protocol['jobs'] if dest == out / 'runs/a_mem' / j['domain'] / j['episode_id'])
            active[index] = {'pid': int(entry.name), 'start': info['start'], 'job': job}
        active_ids = {x['job']['episode_id'] for x in active.values()}
        finished_unmarked = []
        pending = queue.Queue()
        for job in protocol['jobs']:
            dest = out / 'runs/a_mem' / job['domain'] / job['episode_id']
            if (dest / 'complete.json').exists() or job['episode_id'] in active_ids:
                continue
            if (dest / 'process.log').exists():
                # A child can exit between SIGSTOP and the process inspection.
                m.read(dest / 'inference/summary.json')
                finished_unmarked.append(job)
            else:
                pending.put(job)
        m.dump(amendment, {'created_unix': time.time(), 'old_pid': args.old_pid, 'new_pid': os.getpid(),
                         'workers_before': 8, 'workers_after': 20, 'active': active,
                         'pending_episodes': pending.qsize(), 'finished_unmarked': finished_unmarked,
                         'policy': 'Stop and replace scheduler only; existing episode children continue without signals or replay; frozen algorithms/config/transport unchanged'})
        shutil.copyfile(__file__, out / 'continuation_20keys_snapshot.py')
        os.kill(args.old_pid, signal.SIGKILL)
        killed = True
    finally:
        if stopped and not killed:
            os.kill(args.old_pid, signal.SIGCONT)
    with (out / 'orchestrator.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        def status(stage, **extra):
            m.dump(out / 'run_status.json', dict(stage=stage, pid=os.getpid(), workers=20,
                                              updated_unix=time.time(), **extra))
        try:
            status('inference', status='running')
            for job in finished_unmarked:
                finish_existing(out, job)
            stop = threading.Event()
            def worker(index):
                try:
                    if index in active:
                        old = active[index]
                        while True:
                            info = proc(old['pid'])
                            if not info or info['state'] == 'Z' or info['start'] != old['start']:
                                break
                            time.sleep(5)
                        finish_existing(out, old['job'])
                    while not stop.is_set():
                        try:
                            job = pending.get_nowait()
                        except queue.Empty:
                            return
                        m.run_job(out, 'a_mem', job, keys[index])
                except Exception:
                    stop.set()
                    raise
            print('20_KEYS_START', 'adopted', len(active), 'pending', pending.qsize(), flush=True)
            with ThreadPoolExecutor(max_workers=20) as pool:
                futures = [pool.submit(worker, i) for i in range(20)]
                for f in futures:
                    f.result()
            if not pending.empty():
                raise ValueError('Pending inference remains')
            status('official_scoring', status='running')
            with ThreadPoolExecutor(max_workers=4) as pool:
                futures = [pool.submit(m.score_one, out, 'a_mem', d, keys[i], protocol) for i, d in enumerate(m.DOMAINS)]
                for f in futures:
                    f.result()
            m.summarize(out, protocol)
            status('complete', status='complete')
        except Exception as exc:
            status('failed', status='failed', error_type=type(exc).__name__, error=str(exc))
            raise


if __name__ == '__main__':
    main()
