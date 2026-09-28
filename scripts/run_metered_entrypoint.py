"""Read-only HTTP metering around a Python experiment entry point.

Metering rows contain no headers, keys or prompts. Optional separate chat
capture includes input messages and responses for protocol debugging, outside
the model input and scorer; it never records authentication headers.
"""
import argparse
import hashlib
import json
import runpy
import socket
import sys
import threading
import time
from pathlib import Path
from urllib.parse import urlsplit
import requests


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--telemetry', type=Path, required=True)
    parser.add_argument('--capture_dir', type=Path)
    parser.add_argument('--reuse_connections', action='store_true')
    parser.add_argument('--prefer_ip')
    parser.add_argument('--connect_timeout', type=float, default=8.0)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('Python entry point required after --')
    args.telemetry.parent.mkdir(parents=True, exist_ok=True)
    if args.capture_dir:
        args.capture_dir.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    original = requests.Session.request
    if args.prefer_ip:
        resolve = socket.getaddrinfo
        def preferred(host, *a, **kw):
            addresses = resolve(host, *a, **kw)
            if host == 'api.openlux.ai':
                addresses.sort(key=lambda entry: entry[4][0] != args.prefer_ip)
            return addresses
        socket.getaddrinfo = preferred
    sequence = 0

    def measured(self, method, url, **kwargs):
        nonlocal sequence
        raw = kwargs.get('json') if kwargs.get('json') is not None else kwargs.get('data')
        try:
            payload = json.loads(raw) if isinstance(raw, (str, bytes)) else (raw or {})
        except (ValueError, TypeError):
            payload = {}
        if not isinstance(payload, dict):
            payload = {}
        endpoint = urlsplit(url).path
        if not any(endpoint.endswith(p) for p in ('/chat/completions', '/embeddings', '/responses')):
            return original(self, method, url, **kwargs)
        with lock:
            sequence += 1
            request_id = f'{time.time_ns()}_{sequence}'
        row = {'request_id': request_id, 'started_unix': time.time(), 'endpoint': endpoint,
               'requested_model': payload.get('model'),
               'request_sha256': hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()}
        timeout = kwargs.get('timeout')
        if isinstance(timeout, (int, float)):
            kwargs['timeout'] = (min(args.connect_timeout, float(timeout)), float(timeout))
        started = time.monotonic()
        try:
            response = original(self, method, url, **kwargs)
            row['status_code'] = response.status_code
            try:
                body = response.json()
            except ValueError:
                body = {}
            if isinstance(body, dict):
                row['response_model'] = body.get('model')
                row['usage'] = body.get('usage')
                if args.capture_dir and endpoint.endswith('/chat/completions'):
                    (args.capture_dir / f'{request_id}.json').write_text(
                        json.dumps({"response": body, "request_payload": payload}, ensure_ascii=False), encoding='utf-8')
            return response
        except Exception as exc:
            row['error_type'] = type(exc).__name__
            raise
        finally:
            row['elapsed_s'] = time.monotonic() - started
            with lock:
                with args.telemetry.open('a', encoding='utf-8') as handle:
                    handle.write(json.dumps(row, ensure_ascii=False) + '\n')
    requests.Session.request = measured
    if args.reuse_connections:
        # The official runner uses module-level requests.post for every turn.
        # Reuse only transport connections, leaving payloads/models unchanged.
        local = threading.local()
        def persistent_post(url, data=None, json=None, **kwargs):
            if not hasattr(local, 'session'):
                local.session = requests.Session()
            return local.session.post(url, data=data, json=json, **kwargs)
        requests.post = persistent_post
    sys.argv = command
    sys.path.insert(0, str(Path(command[0]).resolve().parent))
    runpy.run_path(command[0], run_name='__main__')


if __name__ == '__main__':
    main()
