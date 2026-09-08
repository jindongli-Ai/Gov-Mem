# Gov-Mem-v4-Symbolic-dev4 Policy Consistency: Full Medical Evaluation

Date: 2026-08-16

This is the complete Medical evaluation of the opt-in policy consistency
certificate layer. It supersedes the earlier 2-episode and 11-episode
diagnostic decisions for the Medical-domain conclusion. It remains separate
from the frozen paper table until the full cross-domain experiment is rerun.

## Protocol

- Dataset unit: all 21 Medical episodes, 579 checkpoints
- Memory-system provider/model: OpenLux, `gpt-4o-mini`
- Embedding: OpenLux, `text-embedding-3-small`
- Official judge: OpenLux, `gpt-4o`, `gate_by_action=false`
- Scheduling: 5 episode workers, one request in flight per worker
- Discovered OpenLux key pool: 10; maximum simultaneously leased workers: 5
- Added Symbolic LLM calls: 0
- Policy consistency: enabled, `enforcement=false`
- Official judge: 579/579 scored, parse failures 0/579
- Context audit coverage: 579/579

`U` is effective utility accuracy, `A` is answer-level privacy leakage, `F` is
answer-level deletion/staleness leakage, and `MGS = U * (1-A) * (1-F)`.

## Final Medical Comparison

| System | Checkpoints | U | A | F | MGS |
|---|---:|---:|---:|---:|---:|
| Frozen `rag_naive_v3_typed_rerank` | 579 | 64.29% | 43.75% | 9.04% | 32.89% |
| Gov-Mem-v4-Symbolic-dev4 policy consistency | 579 | 55.71% | 33.33% | 5.65% | **35.04%** |
| **Delta** | | **-8.58 pp** | **-10.42 pp** | **-3.39 pp** | **+2.15 pp** |

The final Medical MGS therefore **increases by 2.15 percentage points**
(approximately 6.55% relative to the baseline MGS). Utility accuracy decreases,
but the reduction in privacy and deletion leakage more than compensates under
the official GateMem MGS objective.

The earlier 2-episode (-0.56 pp) and 11-episode (-0.97 pp overlap diagnostic)
signals were not representative of the complete Medical distribution. The
full 579-checkpoint result is the conclusion to use for the Medical domain.

This is still not a causal ablation because the baseline and dev4 runs were
separate stochastic executions. The result is a complete-domain performance
signal, not yet a paper-table number. A full four-domain rerun is required
before promoting dev4 as the final paper framework.

## Artifacts

- First 2 episodes:
  `outputs/2026-08-16_Gov-Mem-v4-Symbolic-dev4-policy-consistency-medical-2_gpt4omini_embedding3small_retry/`
- Next 11 episodes:
  `outputs/2026-08-16_Gov-Mem-v4-Symbolic-dev4-policy-consistency-medical-11_gpt4omini_embedding3small/`
- Final 8 episodes:
  `outputs/2026-08-16_Gov-Mem-v4-Symbolic-dev4-policy-consistency-medical-remaining8_gpt4omini_embedding3small/`
- Frozen baseline record:
  `experiments/result/2026-08-05_Gov-Mem_v3_paper_compatible_2218_openlux_gpt4omini_strict.md`
