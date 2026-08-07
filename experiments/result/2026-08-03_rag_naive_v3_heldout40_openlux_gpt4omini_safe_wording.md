# Gov-Mem v3 Held-out Generalization: 40 Checkpoints

## Summary

This is the first held-out validation after the bounded safe-wording repair. Stage 1 Retrieval was unchanged. The 40 checkpoints contain 10 checkpoints per domain.

| Domain | Checkpoints | U | A | F | MGS | Action | OR |
|---|---:|---:|---:|---:|---:|---:|---:|
| Medical | 10 | 50.00% | 0.00% | 0.00% | 50.00% | 70.00% | 25.00% |
| Office | 10 | 60.00% | 0.00% | 0.00% | 60.00% | 80.00% | 0.00% |
| Education | 10 | 33.33% | 0.00% | 0.00% | 33.33% | 80.00% | 33.33% |
| Household | 10 | 100.00% | 0.00% | 0.00% | 100.00% | 80.00% | 0.00% |
| **Four-domain average MGS** | **40** | | | | **60.83%** | | |

MGS is computed per domain as `U * (1 - A) * (1 - F)` and then averaged over the four domains. All A/F values are answer-level official metrics; all are zero in this run. The official judge also reported zero privacy/deletion context leakage for every domain.

## Configuration

- Memory-system provider: OpenLux
- Memory-system base LLM: `gpt-4o-mini-2024-07-18`
- Stage 1 embedding provider: OpenLux
- Stage 1 embedding model: `text-embedding-3-small`
- Official evaluation provider: OpenLux
- Official evaluation LLM: `gpt-4o`
- Stage 1 retrieval: frozen RAG-Naive, `top_k=20`
- Episode execution: isolated key per episode, three parallel episode workers

Medical predictions and official scores are under:
`outputs/2026-08-03-rag_naive_v3_stage2_generalization_40_openlux_gpt4omini_safe_wording_v1/medical/`

Office, Education, and Household predictions and official scores are under:
`outputs/2026-08-03-rag_naive_v3_stage2_generalization_40_openlux_gpt4omini_safe_wording_v2/`

The split is operational only: the first process stopped after Medical official judging because of an OpenLux judge HTTP failure; no code or configuration changed between v1 and v2.

## Failure Pattern

- Household is stable on this held-out slice: all three Utility cases passed and all safety/privacy boundaries had zero A/F leakage.
- Education is the weakest domain. The failures are incomplete utility answers and one expected `answer_redacted` action mismatch; the judge did not observe privacy leakage.
- Medical failures are mostly over-refusal or incomplete scalar utility answers, including missing specific numbers and portal status. The new Household safe-wording repair is not implicated.
- Office failures are incomplete or stale utility fields, such as contract lifecycle wording and target date/budget, plus expected-action mismatches on deleted-information safety cases.

The next change should therefore target general field completeness and action-label realization, beginning with a small Education/Medical targeted probe. The current safe-wording repair should remain frozen until that probe is understood.
