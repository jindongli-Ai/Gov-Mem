"""Retry failed full episodes with fresh OpenLux keys, preserving attempts."""
from pathlib import Path
import json, shutil, subprocess, sys, time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.retry_nano_failed_episodes import validate_episode
from baselines import run as m

METHODS = ('a_mem', 'mem0', 'remem_i', 'remem_s')

def error_rows(dest):
    p = dest/'inference/execution_errors.jsonl'
    if not p.exists(): return []
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]

def retryable(row):
    s = str(row.get('error','')).lower()
    return any(x in s for x in ('network_error','ssleof','timeout','incomplete','connectionerror','max retries'))

def main(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists(): raise RuntimeError(f'existing output: {output}')
    # Reuse the existing bounded-retry launcher to prepare a frozen matrix,
    # then patch its selected jobs below with key rotation.
    import scripts.retry_nano_failed_episodes as base
    # Run preparation through a subprocess only after the output is created by
    # this script's caller; this file is intentionally a small orchestration
    # wrapper around the frozen harness.
    raise RuntimeError('Use the patched launcher entrypoint instead')

if __name__ == '__main__':
    print('This helper is documentation-only; key-rotation is applied in the launcher.')
