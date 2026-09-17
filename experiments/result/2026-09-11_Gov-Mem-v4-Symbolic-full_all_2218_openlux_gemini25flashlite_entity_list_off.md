# Gov-Mem-v4-Symbolic full benchmark: Gemini 2.5 Flash Lite, Entity Relation List off

Date: 2026-09-11
Run directory: `experiments/runs/gemini25flashlite_entity_list_off_full_20260911`

## Protocol

- GateMem checkpoint benchmark, all 2,218 checkpoints.
- Experiment mode: `govmem_v4_symbolic` (current semantic-compiler path).
- Memory-system/base model: OpenLux `gemini-2.5-flash-lite`.
- Entity Relation List / Memory Governed Slot Graph: **disabled during memory storage** (`enabled=false`).
- Embeddings: OpenLux `text-embedding-3-small`.
- Official GateMem judge: OpenLux `gpt-4o`, `gate_by_action=false`.
- Checkpoint visibility: clean observable-prefix protocol; gold feedback disabled.
- Official judge parse failures: 0.

## GateMem metrics

`MGS = U * (1 - A) * (1 - F)`. The reported overall value is the arithmetic
mean of the four domain MGS values.

| Domain | Checkpoints | U | A | F | MGS |
|---|---:|---:|---:|---:|---:|
| Medical | 579 | 42.86% | 24.48% | 14.12% | 27.79% |
| Office | 547 | 52.60% | 14.04% | 11.26% | 40.12% |
| Education | 540 | 16.11% | 20.56% | 12.78% | 11.16% |
| Household | 552 | 30.98% | 19.57% | 5.98% | 23.43% |
| **Overall (pooled U/A/F; mean domain MGS)** | **2,218** | **35.30%** | **19.81%** | **11.01%** | **25.63%** |

Exact four-domain arithmetic mean: `25.62741773355028%`.

For reference, applying the MGS formula to the separately pooled U/A/F values
gives `25.1931%`; the paper-facing headline here follows the repository
convention of averaging the four domain MGS values.

The Entity Relation List remains available in the code as an explicit,
optional Stage-2 authorization ablation, but it is not part of this result and
does not participate in memory ingestion or retrieval for this run.
