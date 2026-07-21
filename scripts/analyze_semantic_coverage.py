from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path


def main(root: str) -> None:
    base = Path(root)
    for path in sorted(base.glob("*/amem_memory/checkpoint_benchmark/*/atomic_memories.jsonl")):
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        sources = Counter(str((row.get("access_tags") or {}).get("extraction_source") or "unknown") for row in rows)
        tagged = [row for row in rows if (row.get("access_tags") or {}).get("semantic_tags")]
        acts = Counter(str(((row.get("access_tags") or {}).get("semantic_tags") or {}).get("discourse_act") or "unknown") for row in tagged)
        event_ids = sum(bool(((row.get("access_tags") or {}).get("semantic_tags") or {}).get("event_identity")) for row in rows)
        deltas = sum(bool(((row.get("access_tags") or {}).get("semantic_tags") or {}).get("state_delta")) for row in rows)
        print(json.dumps({
            "domain": path.parts[-5],
            "checkpoint": path.parent.name,
            "n": len(rows),
            "sources": dict(sources),
            "semantic_tag_rate": len(tagged) / len(rows) if rows else 0.0,
            "discourse_acts": dict(acts),
            "event_identity_rate": event_ids / len(rows) if rows else 0.0,
            "state_delta_rate": deltas / len(rows) if rows else 0.0,
        }, sort_keys=True))


if __name__ == "__main__":
    main(sys.argv[1])
