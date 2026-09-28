"""Paired final-critic intervention on frozen full episodes; no Stage-2 API calls.

Reconstruct validated pre-critic claims from exact captured responses, require
post-critic equivalence, and reuse only identical Stage-3 computations. This
estimates a conditional direct effect, NOT removal of the whole symbolic system.
"""
from __future__ import annotations
import argparse
import copy
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT)]
from gov_mem.data.schema import MemoryInstance, RetrievedEvidence
from gov_mem.governance_runtime.v8_claim_reasoner import reason_v8_claims
from gov_mem.governance_runtime.v8_symbolic_critic import criticize_v8_claims
from gov_mem.governance_runtime.v8_safe_evidence import build_v8_safe_evidence
from gov_mem.backbones.govmem_v8_late_governance import _answer_from_safe_evidence
from gov_mem.llm.json_parser import parse_json_response
from gov_mem.llm.client import LLMClient, LLMConfig
from scripts.run_gatemem_suite import _discover_api_keys


def read(path): return json.loads(path.read_text())
def rows(path): return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]
def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
def sha(value): return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
def prompt_key(prompt): return sha([prompt['system_prompt'], prompt['user_prompt']])


def make_instance(request, cid, domain):
    return MemoryInstance(instance_id=cid, conversation_id=None, domain=domain,
        messages=[], question=request['question'], asking_user_id=request['requester']['principal_id'],
        choices=None, answer=None, metadata={'requester': request['requester']})


def reconstruct(saved, captures, domain):
    final = saved['v8_claim_ledger']
    prompt = final['prompt_audit']
    request = json.loads(prompt['user_prompt'])
    content = captures[prompt_key(prompt)]
    class Captured:
        def chat_json(self, **kwargs):
            if json.loads(kwargs['user_prompt']) != request:
                raise ValueError('Reconstructed Stage-2 input differs')
            if kwargs['system_prompt'] != prompt['system_prompt']:
                raise ValueError('Reconstructed Stage-2 system prompt differs')
            return parse_json_response(content)
    instance = make_instance(request, saved['instance_id'], domain)
    evidence = [RetrievedEvidence(memory_id=c['source_memory_id'], content=c['text'],
        source_message_ids=c['source_message_ids'], time=c.get('timestamp'),
        score=0, retrieval_source='frozen', reason='captured input') for c in request['rag_candidates']]
    failures = final.get('audit', {}).get('contract_failures', [])
    before = reason_v8_claims(instance=instance, evidence=evidence, state_projection={},
        graph_context=request['graph_context'], llm_client=Captured(), model_name='offline',
        ingestion=request.get('ingestion'), access_policy=request.get('access_policy'),
        memory_mode='shallow', response_protocol='json',
        candidate_id_style=request.get('candidate_id_style', 'indexed'),
        repair_feedback=failures[-1] if failures else None)
    after = criticize_v8_claims(before, state_projection=saved['v8_state_projection'],
        requester_principal_id=instance.asking_user_id, graph_context=saved['v8_graph_side_context'])
    # Event extraction is not rerun/committed: saved state is held constant.
    # All validated decisions and grounding must match, including omitted claims.
    for key in ('claims', 'query_slots', 'answer_action', 'rejected_claims', 'symbolic_critic'):
        if after.get(key) != final.get(key):
            raise ValueError(f'Post-critic reconstruction differs: {instance.instance_id} / {key}')
    before.pop('events', None)
    return before, instance


def branch_ledger(before, final, arm):
    if arm == 'no_late_veto':
        return copy.deepcopy(before)
    reasons_off = {
        'no_bound_deletion': {'bound_lifecycle_deletion_veto'},
        'no_bound_permission': {'bound_explicit_permission_veto'},
        'no_deleted_classification': {'source_grounded_deletion_tombstone'},
    }[arm]
    result = copy.deepcopy(before)
    for item, reviewed in zip(result['claims'], final['claims'], strict=True):
        remaining = [r for r in reviewed.get('symbolic_veto_reasons', []) if r not in reasons_off]
        if remaining:
            item['delivery'] = 'block'
            item['symbolic_veto_reasons'] = remaining
    return result


def answer_call(instance, ledger, client, prompt):
    evidence, audit = build_v8_safe_evidence(ledger)
    return _answer_from_safe_evidence(instance=instance, safe_evidence=evidence,
        safe_audit=audit, llm_client=client, model_name='gemini-2.5-flash-lite', stage2_prompt_audit=prompt)


class Probe:
    """Capture actual Stage-3 request, including deterministic no-call branches."""
    def __init__(self): self.request = None
    def chat_json(self, **kwargs):
        self.request = kwargs
        return {'answer': 'probe', 'used_claim_ids': []}


def computation(instance, ledger, prompt):
    probe = Probe()
    result = answer_call(instance, ledger, probe, prompt)
    # Action and provenance postprocessing can differ despite an identical LLM
    # request, so include them in the computation key.
    return {'request': probe.request, 'action': result.action,
            'used_memory_ids': result.used_memory_ids,
            'deterministic_answer': result.answer_text if probe.request is None else None}


def prediction(cid, result):
    audit = result.raw_response['prompt_audit']
    ap = audit.get('answer_prompt')
    sp = audit['stage2_rerank_prompt']
    return {'checkpoint_id': cid, 'action': result.action, 'answer': result.answer_text,
        'answer_structured': {}, 'used_record_ids': result.used_memory_ids,
        'memory_audit': {'schema_version': 1, 'audit_status': audit['audit_status'],
            'prompt_context': {'source': 'answer_prompt.context_text' if ap else 'no_runtime_answer_prompt',
                               'text': ap['context_text'] if ap else ''},
            'stage2_rerank_contexts': [{'stage': sp.get('stage', 'stage2_rerank'), 'text': sp['context_text']}]}}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--execute', action='store_true')
    args = p.parse_args()
    root, out = args.root.resolve(), args.output.resolve()
    identity = read(root/'execution_identity.json')
    # Do not silently replay another implementation against old records.
    for old in (root/'runtime_snapshot/src/gov_mem').rglob('*.py'):
        live = ROOT/'src/gov_mem'/old.relative_to(root/'runtime_snapshot/src/gov_mem')
        if old.read_bytes() != live.read_bytes():
            raise ValueError(f'Frozen production source differs: {live}')
    experiment = {'source_root': str(root), 'source_runtime_sha256': identity['runtime_sha256'],
        'manifest_sha256': hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest(),
        'harness_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'estimand': 'conditional direct final-critic effect; Stage-2 and accumulated state held fixed',
        'arms': ['no_late_veto', 'no_bound_deletion', 'no_bound_permission', 'no_deleted_classification']}
    existing = out/'identity.json'
    if existing.exists() and read(existing) != experiment: raise ValueError('Experiment identity changed')
    dump(existing, experiment)
    captures = {}
    for path in sorted((root/'govmem').glob('*/*/raw_chat_responses/*.json')):
        capture = read(path); messages = capture['request_payload']['messages']
        if 'Stage-2 governance reasoner' not in messages[0]['content']: continue
        key = prompt_key({'system_prompt': messages[0]['content'], 'user_prompt': messages[-1]['content']})
        content = capture['response']['choices'][0]['message']['content']
        if key in captures and captures[key] != content: raise ValueError('Ambiguous captured response')
        captures[key] = content
    manifest = read(root/'manifest.json')
    prepared, coverage = [], []
    for domain, spec in manifest['domains'].items():
        for episode in spec['selected_episodes']:
            directory = root/'govmem'/domain/episode
            records = rows(directory/'predictions/checkpoint_benchmark/predictions.jsonl')
            expected = [e['checkpoint_id'] for e in manifest['entries'] if e['domain']==domain and e['episode_id']==episode]
            if len(records)!=len(expected) or {r['checkpoint_id'] for r in records}!=set(expected):
                raise ValueError('Incomplete episode or duplicate predictions')
            for record in records:
                cid = record['checkpoint_id']
                entry = {'domain': domain, 'episode': episode, 'checkpoint_id': cid,
                         'original': record, 'branches': {}}
                if record.get('execution_status') == 'error':
                    coverage.append({'checkpoint_id': cid, 'status': 'original_error_retained'})
                else:
                    saved = read(directory/'debug_cases/checkpoint_benchmark'/f'{cid}.json')
                    before, instance = reconstruct(saved, captures, domain)
                    final, prompt = saved['v8_claim_ledger'], saved['v8_claim_ledger']['prompt_audit']
                    original = computation(instance, final, prompt)
                    original_audit = read(directory/'prompt_audit/checkpoint_benchmark'/f'{cid}.json')
                    if original['request'] is not None:
                        ap = original_audit['answer_prompt']
                        if any(original['request'][k] != ap[k] for k in ('system_prompt', 'user_prompt')):
                            raise ValueError('Original Stage-3 prompt not reproduced')
                    elif original['deterministic_answer'] != record['answer'] or original['action'] != record['action']:
                        raise ValueError('Original deterministic answer not reproduced')
                    changes = {}
                    for arm in experiment['arms']:
                        ledger = branch_ledger(before, final, arm)
                        comp = computation(instance, ledger, prompt)
                        changed = comp != original
                        changes[arm] = changed
                        if changed:
                            entry['branches'][arm] = {'ledger': ledger, 'computation': comp}
                    entry.update(instance=asdict(instance), prompt=prompt)
                    coverage.append({'checkpoint_id': cid, 'status': 'exact_reconstruction', 'changed': changes,
                        'release_to_block': sum(a['delivery']!='block' and b['delivery']=='block'
                            for a,b in zip(before['claims'], final['claims'], strict=True))})
                prepared.append(entry)
    dump(out/'coverage.json', coverage)
    summary = {'checkpoints': len(prepared), 'reconstructed': sum(c['status']=='exact_reconstruction' for c in coverage),
        'errors_retained': sum(c['status']=='original_error_retained' for c in coverage),
        'release_to_block_claims': sum(c.get('release_to_block', 0) for c in coverage),
        'changed_checkpoints': {a: sum(c.get('changed', {}).get(a, False) for c in coverage) for a in experiment['arms']}}
    summary['unique_changed_computations'] = len({sha([e['checkpoint_id'], b['computation']]) for e in prepared for b in e['branches'].values()})
    summary['unique_answer_requests'] = len({sha([e['checkpoint_id'], b['computation']]) for e in prepared for b in e['branches'].values() if b['computation']['request'] is not None})
    dump(out/'preparation.json', summary); print(json.dumps(summary), flush=True)
    if not args.execute: return
    keys = _discover_api_keys(provider='openlux')
    if not keys: raise ValueError('No configured API credential')
    os.environ['OPENLUX_API_KEY'] = keys[0]
    client = LLMClient(LLMConfig(provider='openlux', api_base='https://api.openlux.ai/v1',
        api_key_env='OPENLUX_API_KEY', temperature=0, max_output_tokens=4096,
        max_retries=2, json_max_attempts=1, allow_fallback=False))
    outputs = {a: {d: [] for d in manifest['domains']} for a in experiment['arms']}
    for entry in prepared:
        cid = entry['checkpoint_id']
        for arm in experiment['arms']:
            branch = entry['branches'].get(arm)
            if branch is None:
                row = entry['original']
            else:
                # Share exact same computation across intervention arms.
                cache = out/'answer_cache'/f'{sha([cid, branch["computation"]])}.json'
                if cache.exists(): row = read(cache)['prediction']
                else:
                    instance = MemoryInstance(**entry['instance'])
                    try:
                        result = answer_call(instance, branch['ledger'], client, entry['prompt'])
                        row = prediction(cid, result)
                    except Exception as exc:
                        row = {'checkpoint_id': cid, 'action': 'error', 'answer': '',
                               'answer_structured': {}, 'used_record_ids': [],
                               'execution_status': 'error', 'error': f'{type(exc).__name__}: {exc}'}
                    dump(cache, {'prediction': row, 'computation': branch['computation']})
                    print('ANSWER', cid, arm, row['action'], flush=True)
            outputs[arm][entry['domain']].append(row)
    for arm, domains in outputs.items():
        for domain, records in domains.items():
            path = out/arm/domain/'predictions.jsonl'; path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in records))
    dump(out/'telemetry_latest_process.json', client.telemetry_snapshot())
    dump(out/'complete.json', summary)


if __name__ == '__main__': main()
