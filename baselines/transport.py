"""OpenLux request metering and exact embedding cache; never cache LLM answers.

Executes an unchanged CLI entry point from the frozen benchmark copy. All
traffic/usage is measured; cache hits are separately logged without fake usage.
"""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
import threading
import time
from urllib.parse import urlsplit
import requests


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--embedding-cache', type=Path, required=True)
    p.add_argument('command', nargs=argparse.REMAINDER)
    args = p.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        p.error('Missing Python entry point')
    for path in (args.output, args.embedding_cache):
        if not path.resolve().is_relative_to('/mnt/data_disk_2'):
            raise ValueError('Writable paths must be on data_disk_2')
        path.mkdir(parents=True, exist_ok=True)
    raw_dir = args.output / 'raw_chat_responses'
    raw_dir.mkdir(exist_ok=True)
    original = requests.Session.request
    lock = threading.Lock()
    local = threading.local()
    sequence = 0
    def append(name, value):
        with lock:
            with (args.output / name).open('a') as f:
                f.write(json.dumps(value, ensure_ascii=False) + '\n')
    def measured(self, method, url, **kw):
        nonlocal sequence
        parts = urlsplit(url)
        payload = kw.get('json') if kw.get('json') is not None else kw.get('data')
        if isinstance(payload, (str, bytes)):
            payload = json.loads(payload)
        if not isinstance(payload, dict) or not parts.path.endswith(('/embeddings', '/chat/completions', '/responses')):
            return original(self, method, url, **kw)
        if parts.hostname != 'api.openlux.ai':
            raise ValueError('Model request outside unified OpenLux endpoint')
        if parts.path.endswith('/chat/completions') and payload.get('model') == 'gpt-5.4-nano':
            from responses_bridge import to_responses, to_chat
            converted = to_responses(payload)
            forwarded = dict(kw)
            forwarded.pop('data', None)
            forwarded['json'] = converted
            # Re-enter metering for the actual Responses request: archive the
            # original wire response and usage before adapting its envelope.
            response = measured(self, method, url.rsplit('/chat/completions', 1)[0] + '/responses', **forwarded)
            if response.status_code == 200:
                chat = requests.Response()
                chat.status_code = 200
                chat._content = json.dumps(to_chat(response.json())).encode()
                chat.headers['Content-Type'] = 'application/json'
                chat.url = response.url
                return chat
            return response
        digest = hashlib.sha256((url + '\n' + json.dumps(payload, sort_keys=True, ensure_ascii=False)).encode()).hexdigest()
        cache = args.embedding_cache / (digest + '.json') if parts.path.endswith('/embeddings') else None
        if cache and cache.exists():
            content = cache.read_bytes()
            body = json.loads(content)
            if body.get('data'):
                response = requests.Response()
                response.status_code = 200
                # Provider usage belongs to the originating request, not this hit.
                body.pop('usage', None)
                response._content = json.dumps(body).encode()
                response.url = url
                response.headers['Content-Type'] = 'application/json'
                append('embedding_cache_hits.jsonl', {'at_unix': time.time(), 'sha256': digest, 'model': payload.get('model')})
                return response
        with lock:
            sequence += 1
            rid = f'{time.time_ns()}_{os.getpid()}_{sequence}'
        row = {'request_id': rid, 'started_unix': time.time(), 'endpoint': parts.path,
               'host': parts.hostname, 'requested_model': payload.get('model'),
               'temperature': payload.get('temperature'), 'max_tokens': payload.get('max_tokens', payload.get('max_output_tokens')),
               'request_sha256': digest}
        start = time.monotonic()
        try:
            response = original(self, method, url, **kw)
            row['status_code'] = response.status_code
            try:
                body = response.json()
            except ValueError:
                body = {}
            if isinstance(body, dict):
                row['response_model'] = body.get('model')
                row['usage'] = body.get('usage')
                if parts.path.endswith(('/chat/completions', '/responses')):
                    with gzip.open(raw_dir / (rid + '.json.gz'), 'wt', encoding='utf-8') as f:
                        json.dump({'request_payload': payload, 'response': body}, f, ensure_ascii=False)
                elif cache and response.status_code == 200 and body.get('data'):
                    tmp = cache.with_suffix(f'.{rid}.tmp')
                    tmp.write_text(json.dumps(body))
                    tmp.replace(cache)
            return response
        except Exception as exc:
            row['error_type'] = type(exc).__name__
            raise
        finally:
            row['elapsed_s'] = time.monotonic() - start
            append('http_telemetry.jsonl', row)
    requests.Session.request = measured
    def post(url, data=None, json=None, **kw):
        if not hasattr(local, 'session'):
            local.session = requests.Session()
        return local.session.post(url, data=data, json=json, **kw)
    requests.post = post
    sys.argv = command
    sys.path.insert(0, str(Path(command[0]).resolve().parent))
    runpy.run_path(command[0], run_name='__main__')


if __name__ == '__main__':
    main()
