# Gov-Mem-v4-Symbolic-dev3 State Ledger: Paired Holdout

Date: 2026-08-15

This is a controlled engineering diagnostic, not a paper performance result.
It compares the current `Gov-Mem-v4-Symbolic-dev3-state-ledger` implementation
with the frozen `rag_naive_v3_typed_rerank` baseline on the same eight complete
GateMem episodes and 203 checkpoints.

## Fixed Protocol

- Dataset unit: complete episodes; no checkpoint-only resampling
- Manifest: `experiments/gatemem_suites/govmem_v4_symbolic_dev3_state_ledger_holdout8_full_episodes_seed20260815.json`
- Memory-system provider/model: OpenLux, `gpt-4o-mini`
- Stage 1 embedding: OpenLux, `text-embedding-3-small`
- Official judge: OpenLux, `gpt-4o`, `gate_by_action=false`
- Temperature: `0.2` for the memory system and `0.0` for the judge
- Scheduling: four episode workers, one leased local key per worker, no worker-internal parallelism
- Runtime storage: local scratch; completed results published to `outputs/`
- Symbolic increment: auxiliary principal/role consistency, provenance graph,
  lifecycle assertions, and retrieved-evidence-only state ledger
- Added Symbolic LLM calls: `0`
- Candidate reordering/filtering: unchanged

The same Education data warning appeared in both runs for checkpoint
`education_episode_custom_en_015_aster_waiver_aster_fellows_dual_track_ckpt_16`:
the evaluator could not find safety `leak_targets` before the checkpoint's
`as_of_turn_id`. All 36 Education cases were still officially scored in both
runs, with zero judge parse failures.

## Official GateMem Results

`U` is utility accuracy, `A` is answer-level privacy leakage, `F` is deletion
leakage, and `MGS = U * (1-A) * (1-F)`.

| Domain | N | Baseline U | Baseline A | Baseline F | Baseline MGS | Symbolic U | Symbolic A | Symbolic F | Symbolic MGS | Delta MGS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Medical | 55 | 55.00% | 55.56% | 5.88% | 23.01% | 45.00% | 55.56% | 11.76% | 17.65% | -5.36 pp |
| Office | 64 | 66.67% | 10.00% | 11.54% | 53.08% | 72.22% | 10.00% | 3.85% | 62.50% | +9.42 pp |
| Education | 36 | 8.33% | 8.33% | 8.33% | 7.00% | 25.00% | 8.33% | 8.33% | 21.01% | +14.00 pp |
| Household | 48 | 25.00% | 31.25% | 0.00% | 17.19% | 31.25% | 37.50% | 0.00% | 19.53% | +2.34 pp |
| **Four-domain mean** | **203** | - | - | - | **25.07%** | - | - | - | **30.17%** | **+5.10 pp** |

Across all query types, the checkpoint-weighted aggregate is:

| Run | U | A | F | Aggregate MGS |
|---|---:|---:|---:|---:|
| Frozen typed-rerank baseline | 42.42% | 27.27% | 7.04% | 28.68% |
| Gov-Mem-v4-Symbolic-dev3 | 45.45% | 28.79% | 5.63% | 30.55% |
| Delta | +3.03 pp | +1.52 pp | -1.41 pp | +1.86 pp |

## Interpretation

On this matched eight-episode holdout, the state-ledger version improves the
four-domain mean MGS and aggregate MGS. The improvement is not uniform:
Medical is negative, while Office, Education, and Household are positive. The
state ledger also lowers deletion leakage overall but increases privacy leakage
slightly. These results support a promising but not yet conclusive engineering
signal.

## State-Ledger Coverage Audit

The paired outputs show that the ledger is not uniformly populated. The
following counts are over all checkpoints in each domain; `present` means that
the ledger block reached the answer prompt, while `resolved>0` means that the
ledger contained at least one resolved state slot.

| Domain | N | Ledger present | Empty ledger | Resolved > 0 |
|---|---:|---:|---:|---:|
| Medical | 55 | 35 | 35 | 0 |
| Office | 64 | 23 | 6 | 16 |
| Education | 36 | 15 | 3 | 9 |
| Household | 48 | 23 | 11 | 10 |

Medical therefore provides a concrete implementation warning: the ledger
block is sometimes inserted, but it resolves no requested slots in this
holdout. The next development step should audit query-slot extraction and
state-ledger coverage, especially for Medical, before adding another Symbolic
module or enabling stronger enforcement.

The comparison is stronger than comparing different episode samples, but it is
not a deterministic causal ablation: the memory-system temperature is `0.2`,
and the two runs make separate stochastic OpenLux calls. This result must not
be merged into the frozen 2,218-checkpoint `rag_naive_v3_typed_rerank` table or
used as the final paper claim. A repeat paired run or a frozen-seed protocol is
needed before promoting this increment.

## Outputs

- Symbolic: `outputs/2026-08-15_Gov-Mem-v4-Symbolic-dev3-state-ledger-holdout8_gpt4omini/`
- Baseline: `outputs/2026-08-15-rag_naive_v3_typed_rerank-holdout8_gpt4omini/`
