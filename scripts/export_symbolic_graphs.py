"""Export symbolic audit payloads from Gov-Mem v4 prediction shards."""
import json
from pathlib import Path
import argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--predictions-root', type=Path, required=True)
    ap.add_argument('--output-root', type=Path, required=True)
    args = ap.parse_args()
    count = 0
    seen = set()
    for shard in args.predictions_root.rglob('episodes/*/predictions/checkpoint_benchmark/predictions.jsonl'):
        episode = shard.parents[2].name
        out = args.output_root / episode / 'symbolic_graphs'
        for line in shard.read_text(encoding='utf-8').splitlines():
            row = json.loads(line)
            if row.get('checkpoint_id') in seen:
                continue
            seen.add(row.get('checkpoint_id'))
            payload = row.get('output', {}).get('memory_audit', {})
            graph = {'checkpoint_id': row.get('checkpoint_id'), 'memory_audit': payload}
            target = out / f"{row.get('checkpoint_id','unknown')}.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(graph, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
            count += 1
    print(f'exported={count}')
if __name__ == '__main__':
    main()
