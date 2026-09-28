"""One bounded, outcome-independent episode retry after the first matrix ends."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import difflib
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
sys.path.insert(0, str(ROOT))
from baselines import run as m
from scripts.run_nano_baseline_matrix import export


def repair(relative, text):
    if relative == 'bench/remem/store.py':
        early = ('        self.nodes[node.node_id] = node\n'
                 '        self._id_list.append(node.node_id)\n'
                 '        self.lexical.add(node.node_id, node.content, node.metadata)\n')
        text = m.replace_once(text, early, '')
        text = m.replace_once(text, '            self.usage.add(usage)\n',
                              '            self.usage.add(usage)\n' + early)
    if relative == 'bench/remem/retriever.py':
        needle = ('        self.nodes[node.node_id] = node\n'
                  '        self.stores[node.node_type].add(node)\n')
        replacement = ('        self.stores[node.node_type].add(node)\n'
                       '        self.nodes[node.node_id] = node\n')
        # The source retry runtime may already contain this fix. In that case
        # leave it unchanged instead of attempting to apply the patch twice.
        if text.count(needle) == 1:
            text = text.replace(needle, replacement)
    return text


def validate_episode(source, method, job):
    d = source/'runs'/method/job['domain']/job['episode_id']
    marker = m.read(d/'complete.json')
    pred = d/'inference/predictions.jsonl'
    if m.digest(pred) != marker['prediction_sha256']:
        raise ValueError('Prediction hash changed')
    rows = m.rows(pred)
    if len(rows) != len(job['checkpoint_ids']) or {r['checkpoint_id'] for r in rows} != set(job['checkpoint_ids']):
        raise ValueError('Incomplete episode')
    errors = sum(r['output'].get('action') == 'error' for r in rows)
    if errors != marker['execution_errors']:
        raise ValueError('Error count differs')
    return d, errors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--workers', type=int, default=30)
    ap.add_argument('--key-offset', type=int, default=0)
    args = ap.parse_args()
    source, out = m.disk2(args.source), m.disk2(args.output)
    out.mkdir(parents=True, exist_ok=True)
    with (out/'retry.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        def status(stage, **kw):
            m.dump(out/'run_status.json', dict(stage=stage, pid=os.getpid(), updated_unix=time.time(), **kw))
        try:
            if (out/'retry_manifest.json').exists():
                raise ValueError('Retry already prepared: do not replay automatically')
            shutil.copyfile(__file__, out/'retry_launcher_snapshot.py')
            status('waiting_for_first_pass', status='running')
            while True:
                state = m.read(source/'run_status.json')
                if state['status'] == 'complete':
                    break
                if state['status'] != 'running':
                    raise RuntimeError('First pass did not complete; inspect it')
                os.kill(state['pid'], 0)
                time.sleep(20)
            protocol = m.read(source/'protocol.json')
            # Older frozen matrices may reference a harness hash that is no
            # longer identical to the current checkout. Their own snapshot is
            # still copied and executed below; retain the mismatch as provenance
            # rather than blocking a corrective rerun.
            try:
                m.verify(source, protocol)
            except (ValueError, FileNotFoundError) as exc:
                if isinstance(exc, ValueError) and 'Execution harness changed:' not in str(exc):
                    raise
                for name, expected in protocol.get('harness_hashes', {}).items():
                    if m.digest(source/'harness_snapshot'/name) != expected:
                        raise ValueError('Frozen harness hash mismatch: '+name)
            selected, reused = [], []
            for method in protocol['methods']:
                for job in protocol['jobs']:
                    _, errors = validate_episode(source, method, job)
                    entry = {'method': method, 'job': job, 'first_pass_errors': errors}
                    (selected if errors else reused).append(entry)
            m.dump(out/'retry_manifest.json', {
                'source': str(source), 'source_protocol_sha256': m.digest(source/'protocol.json'),
                'selection_rule': 'All and only episodes with >=1 recorded execution error, selected before retry outcomes',
                'replacement_rule': 'Use first execution-error-free full-episode retry, regardless of score; archive failures and rotate keys',
                'repair': 'Commit ReMem node identifiers only after embedding/store addition succeeds; no change on successful writes',
                'selected': selected, 'reused': reused, 'workers': args.workers})
            for name in ('runtime', 'configs', 'harness_snapshot', 'selected_dataset', 'episode_datasets'):
                shutil.copytree(source/name, out/name)
            for name in ('manifest.json', 'model.yaml'):
                shutil.copyfile(source/name, out/name)
            (out/'tmp').mkdir()
            # Exact embedding reuse only; no retry LLM responses or states reused.
            shutil.copytree(source/'embedding_cache', out/'embedding_cache', copy_function=os.link)
            patches = []
            for relative in ('bench/remem/store.py', 'bench/remem/retriever.py'):
                path = out/'runtime'/relative
                old = path.read_text(); new = repair(relative, old); path.write_text(new)
                patches.extend(difflib.unified_diff(old.splitlines(True), new.splitlines(True), fromfile=relative, tofile=relative))
            (out/'retry_compatibility.patch').write_text(''.join(patches))
            for name in protocol['runtime_hashes']:
                protocol['runtime_hashes'][name] = m.digest(out/'runtime'/name)
            protocol['harness_hashes_source_frozen'] = dict(protocol.get('harness_hashes', {}))
            protocol['harness_hashes_current_checkout'] = {
                name: m.digest(ROOT/name) for name in protocol.get('harness_hashes', {})
                if (ROOT/name).exists()
            }
            protocol['harness_hashes'] = dict(protocol['harness_hashes_current_checkout'])
            for j in protocol['jobs']:
                j['data_dir'] = str(out/'episode_datasets'/j['episode_id']/j['domain'])
            protocol['retry_source'] = str(source)
            m.dump(out/'protocol.json', protocol)
            m.verify(out, protocol)
            for item in reused:
                method, job = item['method'], item['job']
                src, _ = validate_episode(source, method, job)
                dest = out/'runs'/method/job['domain']/job['episode_id']
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(src, dest, copy_function=os.link)
            keys = m._discover_api_keys(provider='openlux')
            keys = list(dict.fromkeys(keys))[args.key_offset:args.key_offset+args.workers]
            if len(keys) != args.workers:
                raise ValueError('Insufficient distinct API keys')
            pending = queue.Queue()
            by_id = {j['episode_id']: j for j in protocol['jobs']}
            for item in selected:
                pending.put((item['method'], by_id[item['job']['episode_id']]))
            stop = threading.Event()
            def run_rotating(method, job, start_key):
                """Retry a failed episode with fresh keys; archive each attempt."""
                dest = out/'runs'/method/job['domain']/job['episode_id']
                max_attempts = min(len(keys), 40)
                for attempt in range(max_attempts):
                    key_index = (start_key + attempt) % len(keys)
                    m.run_job(out, method, job, keys[key_index])
                    errors_path = dest/'inference/execution_errors.jsonl'
                    errors = [] if not errors_path.exists() else [
                        json.loads(x) for x in errors_path.read_text().splitlines() if x.strip()]
                    if not errors:
                        return
                    text = ' '.join(str(e.get('error','')).lower() for e in errors)
                    network = any(x in text for x in ('network_error','ssleof','timeout','incomplete','connectionerror','max retries'))
                    content_filter = 'content_filter' in text
                    # Keep trying transient network failures with fresh keys.
                    # Content-filter failures are also retried a few times so a
                    # key-specific provider policy cannot silently become paper data.
                    if not network and not content_filter:
                        raise RuntimeError('Non-network execution failure; inspect episode logs')
                    archive = out/'retry_attempts'/method/job['domain']/job['episode_id']/f'key_{key_index:02d}_attempt_{attempt+1:02d}'
                    archive.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(dest), str(archive))
                raise RuntimeError(f'episode still has errors after {max_attempts} key attempts: {method}/{job["episode_id"]}')
            def worker(i):
                while not stop.is_set():
                    try: method, job = pending.get_nowait()
                    except queue.Empty: return
                    try: run_rotating(method, job, i)
                    except Exception:
                        stop.set()
                        raise
            status('retry_inference', status='running', selected_episodes=len(selected))
            print('RETRY_START', len(selected), 'episodes; workers', args.workers, flush=True)
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                fs = [pool.submit(worker, i) for i in range(args.workers)]
                for f in fs: f.result()
            if not pending.empty(): raise RuntimeError('Retry incomplete')
            # Reuse only judgments whose entire episode is unchanged, checked
            # against its first-pass prediction hash above. Copy, never link:
            # the official resume scorer can rewrite/append this file.
            status('retry_scoring', status='running')
            for method in protocol['methods']:
                for domain in m.DOMAINS:
                    ids = {c for x in reused if x['method']==method and x['job']['domain']==domain for c in x['job']['checkpoint_ids']}
                    rows = [r for r in m.rows(source/'scored'/method/domain/'official_eval/judge_scores.jsonl') if r['checkpoint_id'] in ids]
                    path = out/'scored'/method/domain/'official_eval/judge_scores.jsonl'
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(''.join(__import__('json').dumps(r)+'\n' for r in rows))
            with ThreadPoolExecutor(max_workers=4) as pool:
                fs = [pool.submit(m.score_one, out, method, domain, keys[i % len(keys)], protocol)
                      for i, (method, domain) in enumerate((a,b) for a in protocol['methods'] for b in m.DOMAINS)]
                for f in fs: f.result()
            m.summarize(out, protocol)
            export(out)
            audit = []
            for item in selected:
                _, errors = validate_episode(out, item['method'], item['job'])
                audit.append({'method': item['method'], 'episode_id': item['job']['episode_id'],
                              'first_pass_errors': item['first_pass_errors'], 'second_pass_errors': errors})
            m.dump(out/'retry_error_audit.json', audit)
            note = ('Full-episode retries after execution failure; ReMem failed-embedding index consistency fix. '
                    'First execution-error-free retry replaces selected episode regardless of score. Failures retained. '
                    'First-pass artifacts and failed attempts retained; see retry_manifest.json and retry_error_audit.json. '
                    'Costs in official_metrics combine reused first-pass inference and retry work; source total is separate and overlaps.')
            for name, prefix in (('results.md', ''), ('results.tex', '% ')):
                p = out/'paper_tables'/name; p.write_text(prefix+note+'\n\n'+p.read_text())
            status('complete', status='complete')
        except Exception as exc:
            status('failed', status='failed', error_type=type(exc).__name__, error=str(exc))
            raise


if __name__ == '__main__':
    main()
