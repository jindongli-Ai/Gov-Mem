# Gov-Mem

This workspace uses GateMem as the default benchmark dataset for Gov-Mem.

## Current Research Snapshot (2026-09-09)

This section records the current development path and historical benchmark
records. Treat the commit containing this snapshot as a recovery point: later
framework changes should be committed separately, so a regression can be
diagnosed or the previous version can be restored from Git history.

### Active development version (2026-09-09)

The current recovery snapshot is **Gov-Mem-v4-Symbolic-dev7**. It is based on the frozen
`rag_naive_v3_typed_rerank` framework (`Gov-Mem-v3.0`) and preserves the
complete checkpoint-visible GateMem turn as structured retrieval provenance:
`turn_id`, timestamp, principal, role, turn kind, original text, checkpoint,
and the source turn object. Stage 2 receives this data as valid JSON and does
not need to re-extract these fields from prose.

The paper-facing main path is deliberately fixed:

```text
run_govmem.py
  -> pipeline
  -> experiment.mode = govmem_v4_symbolic
  -> src/gov_mem/backbones/rag_naive.py
  -> RAGNaiveBackbone
```

`govmem_symbolic` and `rag_policy_amem` are historical/compatibility paths,
not the canonical implementation of this snapshot.

The current implementation line is **Gov-Mem-v4-Symbolic-dev7**,
selected by `experiment.mode: govmem_v4_symbolic`. It retains v3 retrieval and
adds lightweight typed Symbolic annotations: principal-role consistency, an
Evidence-Principal-Entity relation graph, explicit lifecycle-event assertions,
and a source-bound current-state ledger. Dev7 adds an authorization-aware
evidence boundary and a deterministic claim-level provenance verifier. The
ledger records requested slots, candidate values, source memory/turn
provenance, and conflicts from retrieved evidence only. It does not recover
hidden transcript fields or make permission decisions; dev7 may filter
evidence only when the explicit authorization boundary denies it.

The current v7 development configuration also enables the bounded
**Lexicon-Free Query-Conditioned Semantic Compiler**. After Stage-1 Top-20
retrieval, the configured base LLM dynamically induces open-vocabulary query
slots and proposes source-grounded candidate atoms. Deterministic code then
checks source IDs, turn IDs, source-span membership, slot compatibility, and
closed-evidence membership; it measures required-slot coverage and may run one
targeted repair pass over the same Top-20 evidence. The compiler's atoms enter
the existing governed symbolic graph. They are neither final facts nor
authorization decisions: grounding, identity consistency, temporal/lifecycle
reasoning, authorization, state-ledger transitions, and claim provenance stay
in the symbolic control layer.

The active semantic-compiler experiment configuration is
[`configs/govmem_v4_symbolic_openlux_gpt5mini_embedding3small_semantic_compiler.yaml`](configs/govmem_v4_symbolic_openlux_gpt5mini_embedding3small_semantic_compiler.yaml).
It fixes Stage-1 at `top_k: 20`, enables at most one repair round, and sets
`use_fixed_lexicon: false` and `use_dataset_classifier: false`. The configured
OpenLux model identifiers use provider aliases (for example `gpt-5-mini`), not
dated model identifiers.

### Lexical hardening status

Commit `96d581d` (tag `v7-pre-lexicon-hardening-20260909`) is a recoverable
snapshot made **before completion of the remaining lexical-rule audit and
cleanup**. The Semantic Compiler itself has no fixed domain/benchmark ontology
fallback, but older lexical helpers remain, including in paths reachable by
the current v7 implementation, and require removal or disconnection before
public release. This snapshot must therefore not be presented as a final
lexicon-free release or as evidence that all benchmark-specific wording has
been eliminated.

At runtime, semantic induction, extraction, repair, governance, and answering
must use only the adapter's observable prefix and the closed Stage-1 Top-20
evidence set. They must not receive or read gold answers/evidence,
`expected_action`, `judge_spec`, `leak_targets`, dataset `query_type`, oracle
evidence, rationale, scorer fields, or any future episode suffix. Episode-local
principals, roles, and relations may be retained only when they were observed
in that runtime-visible prefix; they are not a fixed predefined vocabulary.

Dev7's claim-level provenance explanation module is now connected to the actual
`govmem_v4_symbolic` RAG-Naive path. The answering model may return a
source-bound `claim_contract` in the same JSON response; the adapter maps
GateMem source-message IDs to retrieved chunk IDs only within the current
checkpoint, then records a source-grounded explanation in the official
prediction's `memory_audit` and in
`answer_grounding.provenance_explanation`. The explanation includes the final
action, selected evidence references, Symbolic consistency/lifecycle/temporal
reasons, and claim-level support when a contract is available. A missing
contract is recorded as `claim-level explanation incomplete`; no synthetic
source is created and no additional LLM call is made. This is a
non-intervention explanation channel: it never changes the original answer and
is not scored by GateMem. The stateful executor continues to use the full
fail-closed field-state projection contract. When a field cites several
retrieved chunks, the explanation retains one source span for each cited chunk;
the typed state ledger is used only to complete missing source bindings from
the current evidence.
The first real integration smoke found that the base model's optional claim
contracts are not yet reliable enough for hard enforcement; this diagnostic is
recorded in
[`2026-08-19 claim-provenance smoke`](experiments/result/2026-08-19_Gov-Mem-v4-Symbolic-dev7_claim_provenance_smoke.md)
and is not a paper performance result.

The follow-up source-span validation reran the same complete Medical episode
with one OpenLux memory-system key and one episode worker. All 28 predictions
and all 28 official judge records completed with zero judge parse failures, and
the non-intervention audit was present in 28/28 predictions. The diagnostic
metrics were U 30.00%, A 44.44%, F 0.00%, and MGS 16.67%; these values are not
paper performance results. The explanation module added no LLM calls. The full artifact
is `experiments/smoke/2026-08-19_dev7_claim_provenance_source_completion_medical_episode002`.

### Historical Gov-Mem-v4-Symbolic-dev7 full-benchmark result (2026-08-20)

The complete GateMem benchmark was evaluated with OpenLux `gpt-4o-mini` as the
memory-system base LLM and OpenLux `text-embedding-3-small`. The official
GateMem judge was OpenLux `gpt-4o` with `gate_by_action=false`. Gold feedback,
experience, skill updates, and the long-context ledger were disabled. All
2,218 predictions and all 2,218 official judge records completed with zero
judge parse failures.

| Domain | Checkpoints | U | A | F | MGS | Action accuracy | OR |
|---|---:|---:|---:|---:|---:|---:|---:|
| Medical | 579 | 74.29% | 46.35% | 10.17% | **35.80%** | 70.47% | 10.48% |
| Office | 547 | 39.61% | 4.09% | 1.80% | **37.30%** | 75.69% | 39.61% |
| Education | 540 | 40.00% | 18.89% | 7.22% | **30.10%** | 71.85% | 17.78% |
| Household | 552 | 43.48% | 27.17% | 2.72% | **30.80%** | 70.11% | 19.02% |
| **Four-domain average / pooled U-A-F** | **2,218** | **49.72%** | **24.47%** | **5.53%** | **33.50%** | | |

The overall MGS is the arithmetic mean of the four domain MGS values; the
overall U/A/F values are checkpoint-count-weighted pooled values. The detailed
protocol and artifact paths are in the dated
[dev7 full-benchmark result](experiments/result/2026-08-20_Gov-Mem-v4-Symbolic-dev7_full_all_2218_openlux_gpt4omini.md).
This is a complete historical dev7 measurement, not a causal ablation against
the frozen v3 typed-rerank table. It predates the current semantic-compiler
development and the unresolved legacy lexical-rule hardening described above;
it must not be reported as a result of the eventual public lexicon-free
release.

The explanation channel was present in 2,218/2,218 official predictions. It
records selected evidence and Symbolic reasoning facts while keeping
`answer_unchanged=true` and `scored_by_gatemem=false`; it does not contribute
to U, A, F, MGS, action accuracy, or OR.

The intended paper-facing name is **Gov-Mem-v4-Symbolic**. Dev7 is the current
recovery snapshot; subsequent hardening must create a new version and a new
benchmark record rather than overwrite this historical result. The complete
naming and promotion record is in
[`VERSION_LOG.md`](VERSION_LOG.md).

The dev3 state-ledger increment was validated on one complete episode per
domain: Medical 27, Office 32, Education 18, and Household 24 checkpoints
(101 total). The run used OpenLux, `gpt-4o-mini`,
`text-embedding-3-small`, a 30-key pool with four concurrently leased keys,
and four episode workers.
All 101 prompt audits are complete; 46 state-relevant prompts contain the
retrieved-evidence-only ledger, while certificate and target-binding internal
fields occur in zero prompts. Candidate ordering and filtering were unchanged
and the added LLM-call count was zero. This is an engineering validation, not a
2,218-checkpoint paper result and must not be mixed into the frozen full-
benchmark U/A/F/MGS table.

The official GateMem judge completed all 101 cases for this state-ledger run:

| Domain | Checkpoints | U | A | F | MGS | Action accuracy |
|---|---:|---:|---:|---:|---:|---:|
| Medical | 27 | 60.00% | 55.56% | 0.00% | 26.67% | 59.26% |
| Office | 32 | 55.56% | 0.00% | 0.00% | 55.56% | 78.13% |
| Education | 18 | 83.33% | 16.67% | 0.00% | 69.44% | 88.89% |
| Household | 24 | 75.00% | 12.50% | 0.00% | 65.63% | 75.00% |
| **All scored cases** | **101** | **66.67%** | **21.21%** | **0.00%** | **52.53%** | **74.26%** |

The all-cases row is weighted by the number of GateMem query types (33 utility,
33 privacy, and 35 safety cases). The four-domain arithmetic mean of domain MGS
is 54.32%. Relative to the preceding lifecycle smoke on the same 101
checkpoints, this run is a positive engineering signal (`U` +3.03 points,
`A` -3.03 points, weighted `MGS` +4.32 points), not a causal ablation because
the memory-model temperature is 0.2 and the comparison is a single stochastic
run. The full diagnostic record is in
[`experiments/result/2026-08-15_Gov-Mem-v4-Symbolic-dev3-state-ledger-smoke4_openlux_gpt4omini.md`](experiments/result/2026-08-15_Gov-Mem-v4-Symbolic-dev3-state-ledger-smoke4_openlux_gpt4omini.md).

The evidence-validity audit adds deterministic `EvidenceValidityCertificate`
records for explicit lifecycle states. The target-binding v2 increment also
links a lifecycle claim to an earlier retrieved target only when the match is
unique and temporally valid; ambiguous and unbound cases are retained as
explicit audit states. These records remain internal diagnostic metadata and
are deliberately excluded from Stage 2 and answer prompts, so the
non-intervention recording mode does not change the LLM-visible evidence
contract. Across the validated
101-checkpoint run, all answer and Stage 2 prompt audits contained zero
certificate and target-binding fields while internal records were retained.
The official judge completed all 101 cases:

| Domain | Checkpoints | U | A | F | MGS |
|---|---:|---:|---:|---:|---:|
| Medical | 27 | 70.00% | 66.67% | 0.00% | 23.33% |
| Office | 32 | 55.56% | 0.00% | 0.00% | 55.56% |
| Education | 18 | 0.00% | 16.67% | 0.00% | 0.00% |
| Household | 24 | 62.50% | 12.50% | 0.00% | 54.69% |
| **All scored cases** | **101** | **51.52%** | **24.24%** | **0.00%** | **39.03%** |

This is an engineering diagnostic rather than a paper performance result:
it uses one episode per domain, Education has only six utility cases, and the
memory-model temperature is 0.2. It must not be mixed into the frozen
2,218-checkpoint typed-rerank table. Enforcement is not enabled. The complete
score is recorded in
`outputs/2026-08-14_Gov-Mem-v4-Symbolic-dev2-target-binding-shadow-v2_gpt4omini_embedding3small_smoke4/official_score.json`.

### Canonical v7 symbolic experiment method

The canonical method is **Gov-Mem-v4-Symbolic**, selected by
`experiment.mode: govmem_v4_symbolic` and executed through `RAGNaiveBackbone`.
Its runtime flow is:

```text
observable episode prefix
  -> turn-preserving structured records
  -> Stage-1 dense retrieval (closed Top-20)
  -> query-conditioned semantic induction (LLM)
  -> source-grounded atomic extraction (LLM)
  -> deterministic grounding and slot-coverage verification
  -> optional one-round missing-slot repair over the same Top-20 (LLM)
  -> governed symbolic slot graph and authorization/lifecycle/state reasoning
  -> Stage-2 bounded reranking
  -> answer realization
  -> claim-level provenance verification
```

Neural stages may propose semantic structure; the Symbolic Rule Layer controls
grounding, identity consistency, temporal state, lifecycle, authorization,
conflict handling, and provenance. The semantic compiler is strictly
source-grounded and closed-evidence: it cannot expand retrieval or access a
future suffix. `govmem_symbolic`, `rag_policy_amem`, and
`rag_naive_v3_typed_rerank` are retained only as historical or compatibility
tracks and must not be substituted for this main path in a v7 experiment.

### Latest frozen typed-rerank full-benchmark performance

The following table is the frozen `rag_naive_v3_typed_rerank` track, not the
current semantic-compiler path. The historical dev7 full-benchmark table above
also predates semantic-compiler development and lexical hardening.

These results cover all 2,218 GateMem checkpoints: Medical 579, Office 547,
Education 540, and Household 552. All seven runs use the same checkpoint
manifest, Stage 1 retrieval, embedding model (`text-embedding-3-small`), Stage
2 configuration, and official evaluator (`gpt-4o`, temperature 0.0). Only the
base LLM changes. The official GateMem scorer uses
`gate_by_action=false`.

`U` is utility accuracy, `A` is answer-level access-control/privacy leakage,
and `F` is answer-level active-forgetting/deletion leakage. The headline score
is `MGS = U * (1 - A) * (1 - F)`, averaged across the four domain MGS values.
Action accuracy and over-refusal are supplementary metrics.

| Frozen typed-rerank base LLM | Medical MGS | Office MGS | Education MGS | Household MGS | Four-domain avg. MGS |
|---|---:|---:|---:|---:|---:|
| GPT-4o-mini | 32.89% | 43.35% | 25.66% | 34.39% | **34.07%** |
| GPT-5-mini | 42.35% | 65.26% | 27.73% | 43.09% | **44.61%** |
| GPT-5.4 | **56.88%** | 62.61% | 23.14% | 38.21% | **45.21%** |
| GPT-5.4-mini | 47.56% | 54.43% | 24.53% | 38.07% | **41.15%** |
| Gemini-2.5-Flash-Lite | 49.17% | **63.82%** | **30.43%** | 37.31% | **45.18%** |
| DeepSeek-V4-Flash | **58.89%** | 61.56% | 26.88% | **47.47%** | **48.70%** |
| Llama-3.3-70B-Instruct | 27.78% | 34.92% | 14.71% | 20.70% | 24.53% |

DeepSeek-V4-Flash is currently the best of these seven tested base LLMs by
average MGS, with an absolute improvement of 14.63 percentage points over
GPT-4o-mini. It also gives the best Medical and Household domain MGS in this
comparison.
The full set is important: smaller 40-, 200-, or 800-checkpoint diagnostics
are not interchangeable with these results.

GPT-5.4-mini, DeepSeek-V4-Flash, and Llama-3.3-70B-Instruct are included in
the main table as full-benchmark model comparisons. Their complete
domain-level metrics and protocols are documented in the linked result
artifacts below.

### Safety and interpretation boundaries

The official MGS is an answer-level benchmark metric. It is not equivalent to
zero exposure of restricted evidence in intermediate prompts. The independent
context audit for the strict runs reports non-zero privacy/deletion context
exposure in particular for Medical and Household. Therefore this repository
does not claim that the current v3 system has zero end-to-end leakage.

The latest frozen typed-rerank results are not a complete four-model, full-set
paired comparison against locally regenerated plain RAG-Naive predictions.
GateMem's released GPT-5.4 RAG-Naive reference Utility values are Medical
64.8%, Office 74.0%, Education 32.8%, and Household 51.1%. These are Utility
references only and should not be presented as MGS comparisons. A future
paper table must report the exact baseline protocol, checkpoint manifest,
model, evaluator, and whether the comparison is on U, A, F, or MGS.

### Main result artifacts

- [GPT-5.4 strict full result](experiments/result/2026-08-05-22-53-20_Gov-Mem_v3_full_all_2218_openlux_gpt54_strict.md)
- [GPT-5-mini strict full result](experiments/result/2026-08-05-23-57-05_Gov-Mem_v3_full_all_2218_openlux_gpt5mini_strict.md)
- [Gemini-2.5-Flash-Lite strict full result](experiments/result/2026-08-06-01-41-13_Gov-Mem_v3_full_all_2218_openlux_gemini25flashlite_strict.md)
- [GPT-4o-mini strict full result](experiments/result/2026-08-05_Gov-Mem_v3_paper_compatible_2218_openlux_gpt4omini_strict.md)
- [GPT-5.4-mini strict full result](experiments/result/2026-08-05_Gov-Mem_v3_full_all_2218_openlux_gpt54mini_strict.md)
- [DeepSeek-V4-Flash strict full result (2026-08-11)](experiments/result/2026-08-11_Gov-Mem_v3_full_all_2218_openlux_deepseekv4flash_strict.md)
- [Llama-3.3-70B-Instruct strict full result (2026-08-11)](experiments/result/2026-08-11_Gov-Mem_v3_full_all_2218_openlux_llama33_70b_instruct_strict.md)
- [Full U/A/F/MGS Markdown summary table (2026-08-11)](experiments/result/2026-08-11_Gov-Mem_v3_full_all_2218_performance_summary.md)
- [Current implementation and contribution reconstruction](report.md)

When reporting a new result, record the commit, config, model/provider,
manifest, evaluator, and whether long-context or gold-derived feedback was
enabled. Commit improvements as new history points instead of overwriting the
current snapshot.

## 迁移到新服务器

当前项目目录为：

```text
/data_nvme/user/jli/codes/2027_ICLR_Gov-Mem
```

迁移的目标是保留代码、GateMem 数据、实验配置、结果记录和 Git 历史，
这样换服务器后仍然可以继续实验，也可以通过 Git 回到当前 dev7 快照。

### 1. 迁移前检查

先在旧服务器确认没有实验进程仍在运行：

```bash
cd /data_nvme/user/jli/codes/2027_ICLR_Gov-Mem
ps -ef | rg 'run_gatemem_suite|run_parallel_official_judge_shards|run_govmem.py' || true
git status --short
git log --oneline --decorate -8
```

当前可恢复版本是 `Gov-Mem-v4-Symbolic-dev7`，对应冻结提交为：

```text
96d581d backup: Gov-Mem v7 before lexicon-free hardening
```

迁移前不要删除旧服务器上的任何文件。旧服务器应作为回退副本保留，直到
新服务器完成基本验证。

### 2. 推荐复制内容

以下内容属于项目的可复现实验框架，应复制到新服务器：

- `.git/`：Git 历史和回滚点，必须保留；
- `src/`、`scripts/`、`tests/`、`run_govmem.py`：实现、运行器和测试；
- `configs/`、`experiments/gatemem_suites/`：实验配置和 checkpoint manifest；
- `experiments/result/`、`README.md`、`report.md`、`VERSION_LOG.md`：结果、协议和版本记录；
- `dataset/GateMem/`：GateMem 原始数据，只读，不要修改；
- `third_party/GateMem-official/`：官方 GateMem 评测工具；
- `third_party/python_deps/gatemem_eval/`：当前服务器已有的评测依赖包；
- `cache/`：可选，已有 embedding 缓存可以减少重复请求，但不是代码运行的必要条件。

### 3. 使用 rsync 复制整个工作区

在旧服务器执行。把下面的目标路径替换成新服务器挂载后的目录：

```bash
rsync -aH --info=progress2 \
  /data_nvme/user/jli/codes/2027_ICLR_Gov-Mem/ \
  /new_server_path/2027_ICLR_Gov-Mem/
```

如果通过本地电脑中转，应确保隐藏目录 `.git/` 也被复制；只下载网页显示的
源文件而不复制 `.git/`，会丢失所有历史版本和当前恢复点。

### 4. 大型 outputs 的处理

`outputs/` 被 `.gitignore` 忽略，不属于 GitHub 备份的一部分。它包含大量旧
实验和中间文件，可能占用数 GB；可以按需要迁移。

至少建议迁移当前完整 dev7 结果目录：

```text
outputs/govmem_v4_symbolic_dev7_full_all_2218_20260820/
```

如果磁盘空间有限，可以只迁移代码和文档，之后在新服务器重新生成实验输出。
当前完整结果的可提交摘要已经保存在 Git 跟踪文件中，完整协议见：

```text
experiments/result/2026-08-20_Gov-Mem-v4-Symbolic-dev7_full_all_2218_openlux_gpt4omini.md
```

不要把旧的临时 smoke 输出、失败重试输出和所有历史中间文件混入新的论文结果。

### 5. API key 的处理

`API-Key_OpenLux.md` 只用于本地运行，不能提交到 GitHub，也不能放进公开的
README。迁移时通过安全方式单独复制，并在新服务器设置严格权限：

```bash
chmod 600 /new_server_path/2027_ICLR_Gov-Mem/API-Key_OpenLux.md
```

运行实验前，在新服务器的 shell 中设置程序实际使用的密钥环境变量，或按照
本地运行器的说明配置 key pool。不要把 key 直接写入 YAML、Python 文件、
shell 历史或实验日志。

### 6. 新服务器验证

进入新目录后，先只做本地检查，不要立即启动全量实验：

```bash
cd /new_server_path/2027_ICLR_Gov-Mem
git status --short
git log --oneline --decorate -1
python3 scripts/inspect_gatemem.py
python3 -m pytest -q tests/test_symbolic_evidence.py tests/test_official_evaluation_contract.py
```

应确认以下事实：

- `git log` 能看到 `96d581d`（或其后续文档/整改提交）；
- GateMem 四个 domain 可以被读取；
- Symbolic 相关测试通过；
- `API-Key_OpenLux.md` 没有出现在 `git status` 的待提交文件中；
- 新服务器没有自动恢复旧实验进程。

迁移完成后，第一次网络实验只运行一个小 episode，确认 OpenLux、embedding、
官方 GateMem judge 和输出目录均正常，再逐步增加并行度。不要直接恢复旧服务器
上的高并行配置。

## Dataset

The raw GateMem dataset is stored under `dataset/GateMem`.

Raw dataset files are treated as read-only:

- Do not edit files under `dataset/GateMem/`
- Put any derived artifacts in new directories outside the raw dataset tree

## Local Utilities

- `src/gov_mem/data/gatemem.py`: GateMem loader utilities
- `scripts/inspect_gatemem.py`: quick dataset sanity check
- `scripts/score_gov_mem_with_gatemem.py`: score Gov-Mem predictions with the official GateMem evaluator
- `scripts/validate_gov_mem_predictions.py`: validate prediction-file format before scoring

## Official Evaluation

The official GateMem evaluation toolkit is vendored at:

- `third_party/GateMem-official`

Gov-Mem should use the official GateMem scorer for reported benchmark results.
This keeps evaluation aligned with the GateMem paper and its released metric
definitions.

## Historical Frozen Framework Diagnostic

The current frozen Gov-Mem v3 framework uses RAG-Naive Retrieval in Stage 1 and
typed, constrained reasoning reranking over retrieved evidence in Stage 2. The
table below preserves an earlier strict combined 800-checkpoint diagnostic for
traceability. That run used the long-context field ledger and therefore is not
directly comparable to the GateMem paper main table or to the formal protocol
below.
checkpoints from the earlier 200-case run plus 150 new checkpoints per domain.
The framework code and all experiment settings were unchanged across the two
runs.

| Domain | Checkpoints | U | A | F | MGS | Action | OR |
|---|---:|---:|---:|---:|---:|---:|---:|
| Medical | 200 | 64.38% | 34.43% | 4.55% | 40.30% | 76.50% | 13.70% |
| Office | 200 | 59.65% | 8.96% | 2.63% | 52.88% | 79.00% | 22.81% |
| Education | 200 | 43.75% | 19.44% | 6.25% | 33.04% | 74.50% | 10.94% |
| Household | 200 | 49.12% | 23.68% | 2.99% | 36.37% | 65.00% | 21.05% |
| **Four-domain average MGS** | **800** | | | | **40.65%** | | |

MGS is computed per domain as `U * (1 - A) * (1 - F)`, followed by the
arithmetic mean of the four domain MGS values. `A` is answer-level privacy
leakage, `F` is answer-level deletion leakage, `Action` is action accuracy, and
`OR` is over-refusal rate among utility cases.

| Evaluation | Four-domain average MGS |
|---|---:|
| Earlier 200-case slice | 56.04% |
| New 600-case slice | 35.13% |
| Strict combined 800-case result | **40.65%** |

Configuration for this historical diagnostic:

- Memory-system provider/model: OpenLux, `gpt-4o-mini-2024-07-18`
- Stage 1 embedding provider/model: OpenLux, `text-embedding-3-small`
- Official GateMem judge provider/model: OpenLux, `gpt-4o`
- Stage 1 retrieval: frozen RAG-Naive, `top_k=20`
- Stage 2: typed reasoning rerank plus long-context field ledger, source-bound safe wording
- This result must be labeled as a strict/ablation diagnostic, not as a GateMem paper-compatible main result

Formal GateMem-compatible Gov-Mem evaluations use the official scorer with
`gate_by_action=false`, memory-system temperature `0.2`, judge temperature
`0.0`, `text-embedding-3-small`, turn-level RAG-Naive retrieval with
`top_k=20`, and the retrieved-evidence-only Stage 2 configuration
`configs/rag_naive_v3_openlux_gpt4omini_embedding3small_pure.yaml`. Context-audit
rates are reported separately because they are not part of GateMem's paper MGS.

The detailed privacy/deletion diagnostics and Overleaf-ready LaTeX tables are
available in [`experiments/result/2026-08-03_rag_naive_v3_stage2_generalization_800_overleaf_tables.tex`](experiments/result/2026-08-03_rag_naive_v3_stage2_generalization_800_overleaf_tables.tex).

## Quick Start

```bash
python3 scripts/inspect_gatemem.py
```

Validate a predictions file:

```bash
python3 scripts/validate_gov_mem_predictions.py \
  --predictions outputs/gov_mem_medical/predictions.jsonl
```

Score predictions with the official GateMem scorer:

```bash
python3 scripts/score_gov_mem_with_gatemem.py \
  --domain medical \
  --predictions outputs/gov_mem_medical/predictions.jsonl \
  --out_dir outputs/gov_mem_medical_scored
```

## Environment Note

The official GateMem scorer is used as-is from `third_party/GateMem-official`.
Its Python dependencies are not bundled into this repo automatically.
For this workspace, a local scorer dependency bundle has already been installed at:

- `third_party/python_deps/gatemem_eval`

The Gov-Mem wrapper scripts automatically add that directory to `PYTHONPATH`
when invoking the official GateMem scorer.

At minimum, the official scorer may require packages such as:

- `numpy`
- `tqdm`
- `PyYAML`
- `requests`
- `scikit-learn`

If you enable the official LLM judge or retrieval-heavy baselines, the full
`third_party/GateMem-official/requirements.txt` stack may also be needed.

## Gov-Mem Pipeline

The v7 semantic-compiler path uses the adapter's observable episode prefix and
retrieved evidence only. Stage 1 is fixed at Top-20 and compiler repair cannot
retrieve more evidence or access the full transcript. In particular,
`long_context_field_ledger.enabled` must remain `false`: that optional
component reads the complete visible checkpoint transcript and is reserved for
explicitly labeled long-context ablations.

Main entry:

```bash
python3 run_govmem.py \
  --dataset_name gatemem \
  --data_path dataset/GateMem/gatemem/data/medical \
  --output_dir outputs/govmem_medical_debug \
  --config configs/govmem_v4_symbolic_openlux_gpt5mini_embedding3small_semantic_compiler.yaml \
  --experiment_mode govmem_v4_symbolic \
  --max_instances 10 \
  --stage all
```

## Base LLM Configuration

Gov-Mem now treats the experiment-time backbone model as a configurable `base LLM`
rather than hard-coding a specific model into the framework.

The main config interface is:

```yaml
llm:
  base_model: gpt-5-mini
  role_models: {}
```

- `base_model`: the default LLM used by the framework for query planning, action decision, and other base-model-controlled stages
- `role_models`: optional per-role overrides such as `memory_ingestion`, `query_planning`, `action_decision`, or `answering`

Example: use one base LLM for the whole framework

```yaml
llm:
  base_model: DeepSeek-V4-Flash
  role_models: {}
```

Example: keep one base LLM but override answering only

```yaml
llm:
  base_model: Qwen3.5-plus
  role_models:
    answering: gpt-5
```

The preferred v7 development configuration is:

- `configs/govmem_v4_symbolic_openlux_gpt5mini_embedding3small_semantic_compiler.yaml`

Other configs under `configs/` may target historical baselines or compatibility
paths and must not be silently substituted for the v7 experiment. In
particular, use OpenLux aliases such as `gpt-5-mini` and `gpt-5.4-mini`, rather
than date-suffixed identifiers.

Historical example configs include:

- `configs/govmem_gpt5_nano.yaml`
- `configs/govmem_gpt5.yaml`
- `configs/govmem_deepseek_v4_flash.yaml`
- `configs/govmem_qwen35_plus.yaml`

Example runs:

```bash
python3 run_govmem.py \
  --dataset_name gatemem \
  --data_path dataset/GateMem/gatemem/data/medical \
  --output_dir outputs/govmem_gpt5_nano_medical \
  --config configs/govmem_gpt5_nano.yaml \
  --experiment_mode govmem_v4_symbolic \
  --max_instances 30 \
  --stage all
```

```bash
python3 run_govmem.py \
  --dataset_name gatemem \
  --data_path dataset/GateMem/gatemem/data/medical \
  --output_dir outputs/govmem_deepseek_v4_flash_medical \
  --config configs/govmem_deepseek_v4_flash.yaml \
  --experiment_mode govmem_v4_symbolic \
  --max_instances 30 \
  --stage all
```

You can also override the base LLM at runtime before an experiment without editing
the YAML file:

```bash
python3 run_govmem.py \
  --dataset_name gatemem \
  --data_path dataset/GateMem/gatemem/data/medical \
  --output_dir outputs/govmem_runtime_override \
  --config configs/govmem_default.yaml \
  --experiment_mode govmem_v4_symbolic \
  --base_model DeepSeek-V4-Flash \
  --llm_provider yunwu \
  --llm_api_base https://yunwu.ai/v1 \
  --llm_api_key_env YUNWU_API_KEY \
  --max_instances 10 \
  --stage all
```

Optional role-specific overrides are also supported:

```bash
python3 run_govmem.py \
  --dataset_name gatemem \
  --data_path dataset/GateMem/gatemem/data/medical \
  --output_dir outputs/govmem_role_override \
  --config configs/govmem_default.yaml \
  --experiment_mode govmem_v4_symbolic \
  --base_model Qwen3.5-plus \
  --role_model action_decision=gpt-5 \
  --role_model answering=gpt-5 \
  --max_instances 10 \
  --stage all
```

Each run writes the fully resolved LLM settings to:

- `outputs/.../run_metadata.json`
- `outputs/.../debug_cases/<dataset>/<instance>.json`

Supported stages:

```bash
python3 run_govmem.py --stage all
python3 run_govmem.py --stage ingest
python3 run_govmem.py --stage retrieve
python3 run_govmem.py --stage answer
python3 run_govmem.py --stage evaluate
```

Main Gov-Mem outputs:

- `outputs/.../memory/<dataset>/<instance>/memory_items.jsonl`
- `outputs/.../query_plan/<dataset>/<instance>.json`
- `outputs/.../retrieval/<dataset>/<instance>.json`
- `outputs/.../reasoning/<dataset>/<instance>.json`
- `outputs/.../predictions/<dataset>/<instance>.json`
- `outputs/.../predictions/<dataset>/predictions.jsonl`
- `outputs/.../experience/<dataset>/experience_bank.jsonl`
- `outputs/.../eval/<dataset>/metrics.json`
- `outputs/.../eval/<dataset>/case_results.jsonl`

If the dataset is GateMem, the runner also emits official scorer outputs under:

- `outputs/.../official_eval/gatemem/<domain>/summary.json`

## API Environment

Gov-Mem now distinguishes explicitly between real LLM mode and heuristic fallback mode.

Default provider is `yunwu`, using:

- `YUNWU_API_KEY`
- `YUNWU_BASE_URL`

Default Yunwu base URL:

```bash
export YUNWU_BASE_URL="https://yunwu.ai/v1"
```

If a valid API key is detected, logs will show:

```text
[Gov-Mem] Real LLM mode enabled: provider=..., model=...
```

If no valid API key is detected and fallback is allowed, logs will show:

```text
[Gov-Mem Warning] No valid LLM API key detected. Falling back to heuristic mode. Accuracy may be invalid.
```
