"""Offline inventory for prospective LoRA event extraction; never trains.

Only already-used V8 development captures are inspected. Confirmation episodes
are excluded before reading any prompt. Parse-valid is NOT semantic gold.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
from gov_mem.llm.json_parser import parse_json_response


def fingerprint(value):return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--reserved_manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();reserved={r['episode_id'] for r in json.loads(args.reserved_manifest.read_text())['entries']}
    if not args.output.resolve().is_relative_to('/mnt/data_disk_2'):raise ValueError('Output must stay on disk 2')
    selected={r['episode_id'] for r in json.loads((args.root/'manifest.json').read_text())['entries']}
    if selected & reserved:raise ValueError('Training candidate episode intersects reserved confirmation set')
    counts=Counter();effects=Counter();prefixes=defaultdict(set);outputs=defaultdict(set);per_episode=Counter()
    for path in sorted((args.root/'govmem').glob('*/*/raw_chat_responses/*.json')):
        episode=path.relative_to(args.root/'govmem').parts[1]
        if episode not in selected or episode in reserved:raise ValueError('Unexpected capture episode')
        capture=json.loads(path.read_text());messages=capture['request_payload']['messages']
        if 'Stage-2 governance reasoner' not in messages[0]['content']:continue
        request=json.loads(messages[-1]['content']);ingestion=request.get('ingestion')
        if not ingestion:counts['query_only_requests']+=1;continue
        counts['joint_ingestion_requests']+=1
        key=(episode,fingerprint(ingestion.get('new_turns')))
        prefixes[key].add(fingerprint(request));per_episode[episode]+=1
        try:raw=parse_json_response(capture['response']['choices'][0]['message']['content'])
        except (ValueError,TypeError,KeyError):counts['unparseable_joint_responses']+=1;continue
        events=raw.get('events')
        if not isinstance(events,list) or any(not isinstance(e,dict) for e in events):counts['malformed_event_envelopes']+=1;continue
        counts['event_envelope_responses']+=1;counts['empty_event_responses']+=not events
        counts['event_proposals']+=len(events)
        outputs[key].add(fingerprint(events))
        effects.update(str(e.get('effect','missing')) for e in events)
    result={'scope':'Prospective event-extraction LoRA, inventory only; no GPU/API calls; proposals are not gold labels',
        'source':str(args.root.resolve()),'source_runtime_sha256':json.loads((args.root/'execution_identity.json').read_text())['runtime_sha256'],
        'reserved_manifest_sha256':hashlib.sha256(args.reserved_manifest.read_bytes()).hexdigest(),
        'selected_development_episodes':sorted(selected),'reserved_episodes':sorted(reserved),
        'episode_overlap':sorted(selected & reserved),'counts':dict(counts),'proposal_effect_counts':dict(effects),
        'unique_episode_new_turn_windows':len(prefixes),
        'windows_with_multiple_requests':sum(len(v)>1 for v in prefixes.values()),
        'windows_with_disagreeing_event_outputs':sum(len(v)>1 for v in outputs.values()),
        'critical_limits':['12 correlated development episodes are not a validated training corpus.',
            'Joint extraction saw current query and may be query-dependent; cannot assume query-independent teacher gold.',
            'Protocol/source validity does not prove permission scope, deletion semantics or event completeness.',
            'Never split checkpoints from one episode across train and validation.',
            'Reserved confirmation episodes must never be distilled or trained on.',
            'Training requires independent documents/synthetic scenario families with reviewed labels and held-out families.'],
        'recommended_training_target':'Optional local LoRA for typed permission/lifecycle extraction with exact source spans; retain language review and deterministic late veto.',
        'mlp_decision':'Do not replace natural-language permission/scope judgment with an MLP now; no calibrated features or trusted labels.'}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['counts','unique_episode_new_turn_windows','windows_with_disagreeing_event_outputs','episode_overlap']},indent=2))

if __name__=='__main__':main()
