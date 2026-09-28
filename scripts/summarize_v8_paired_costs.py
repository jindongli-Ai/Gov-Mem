"""Aggregate recorded HTTP usage, including failed integration attempts.

No monetary price is assumed. Repeated payloads are reported separately from
transport errors: a repeated payload may be a legitimate request or a resume.
"""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path


def summarize(files):
    groups = {}
    for path in files:
        for line in path.read_text().splitlines():
            row = json.loads(line)
            endpoint = row['endpoint'].rsplit('/', 1)[-1]
            model = row.get('requested_model') or row.get('response_model') or 'unrecorded'
            key = f'{endpoint}:{model}'
            group = groups.setdefault(key, {'requests': 0, 'http_successes': 0, 'http_errors': 0,
                'responses_with_usage': 0, 'prompt_tokens': 0, 'completion_tokens': 0,
                'total_tokens': 0, 'sum_request_latency_s': 0, 'response_models': Counter(),
                'status_codes': Counter(), 'error_types': Counter(), 'payloads': Counter()})
            group['requests'] += 1
            ok = 200 <= row.get('status_code', 0) < 300 and not row.get('error_type')
            group['http_successes' if ok else 'http_errors'] += 1
            group['sum_request_latency_s'] += row.get('elapsed_s', 0)
            group['status_codes'][str(row.get('status_code'))] += 1
            if row.get('error_type'): group['error_types'][row['error_type']] += 1
            if row.get('response_model'): group['response_models'][row['response_model']] += 1
            group['payloads'][row.get('request_sha256')] += 1
            usage = row.get('usage')
            if isinstance(usage, dict):
                group['responses_with_usage'] += 1
                for field in ('prompt_tokens', 'completion_tokens', 'total_tokens'):
                    group[field] += usage.get(field, 0) or 0
    for group in groups.values():
        group['repeated_payload_requests'] = sum(n-1 for n in group.pop('payloads').values())
        group['sum_request_latency_s'] = round(group['sum_request_latency_s'], 3)
    return groups


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--attempt_roots', type=Path, nargs='+', required=True)
    parser.add_argument('--scoring_root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = {'note': 'Provider-reported tokens only; missing usage and failed calls may still incur provider charges. '
                      'HTTP success does not imply valid JSON or a successful checkpoint. No dollar estimate.',
              'inference_attempts': {}}
    for root in args.attempt_roots:
        by_system = {}
        for system in ('govmem', 'rag_naive'):
            base = root / system
            files = list(base.rglob('http_telemetry.jsonl'))
            if not files: continue
            by_system[system] = {'complete_episodes': len(list(base.rglob('complete.json'))),
                                'http': summarize(files)}
        result['inference_attempts'][str(root)] = by_system
    if args.scoring_root:
        result['official_judge'] = summarize(list(args.scoring_root.rglob('judge_http_telemetry.jsonl')))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(args.output)


if __name__ == '__main__': main()
