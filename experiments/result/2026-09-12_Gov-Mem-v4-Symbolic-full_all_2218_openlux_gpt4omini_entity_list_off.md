# Gov-Mem-v4-Symbolic full benchmark: GPT-4o-mini, Entity Relation List off

Date: 2026-09-12
Run directory: `experiments/runs/gpt4omini_entity_list_off_full_20260912`

## Protocol

- GateMem checkpoint benchmark, all 2,218 checkpoints across 91 episodes.
- Experiment mode: `govmem_v4_symbolic` (current semantic-compiler path).
- Memory-system/base model: OpenLux `gpt-4o-mini`.
- Entity Relation List / Memory Governed Slot Graph: **disabled during memory storage** (`enabled=false`).
- Embeddings: OpenLux `text-embedding-3-small`; shared cache reused from the same embedding model.
- Official GateMem judge: OpenLux `gpt-4o`, `gate_by_action=false`.
- Checkpoint visibility: clean observable-prefix protocol; gold feedback disabled.
- Official judge parse failures: 0.

## GateMem metrics

`MGS = U * (1 - A) * (1 - F)`. Domain MGS is computed from each domain's
U/A/F; the headline overall MGS is the arithmetic mean of the four domain MGS
values. Overall U/A/F below are checkpoint-count-weighted pooled rates.

| Domain | Checkpoints | U | A | F | MGS |
|---|---:|---:|---:|---:|---:|
| Medical | 579 | 44.76% | 25.52% | 18.64% | 27.12% |
| Office | 547 | 64.29% | 24.56% | 13.51% | 41.94% |
| Education | 540 | 31.67% | 9.44% | 16.11% | 24.06% |
| Household | 552 | 33.70% | 17.93% | 9.24% | 25.10% |
| **Overall (pooled U/A/F; mean domain MGS)** | **2,218** | **42.86%** | **19.39%** | **14.29%** | **29.55%** |

Exact four-domain arithmetic mean MGS: `29.554714438226787%`.

The Entity Relation List remains an explicit optional Stage-2 authorization
ablation in the code, but it is not part of this result and did not participate
in memory ingestion or retrieval.
