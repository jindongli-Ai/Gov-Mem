# Gov-Mem 会话交接（2026-09-25）

## 2026-09-25 当前状态（最高优先级）

当前 canonical pipeline 是 V8 Late Governance：Naive RAG dense Top-20 为主
召回，Governed Slot Graph 作为权限审计旁观者，LLM 负责开放语义权限/生命
周期判断，symbolic/neuro-symbolic layer 负责来源、绑定、生命周期和明确
违规的确定性校验。Graph 不直接授权、拒答或替换 dense RAG；信息不足时只
记录 audit gap。

有效的 graph 配对确认消融已经完成：同一 12 集、306 checkpoints/arm，官方
GPT-4o judge，Graph-on **17.22%**，真正 Graph-off **15.57%**，差值
**+1.64 个百分点 avg.MGS**。两组均覆盖完整，零网络/API/非 200 错误；这
是历史已暴露数据上的 confirmation/development evidence，不是 pristine
holdout，也不是最终 2218-query 结果。

此前 `outputs/v8_reserved_confirmation_graph_off_20250925` 的配置实际仍为
`enabled: true`，其 on/off 对照无效，仅保留作审计，禁止引用。有效目录、报
告和评分路径见 `RESUME_GOVMEM.md`。

## 2026-09-24 历史进展（低于 2026-09-25 状态）

用户明确论文第一篇聚焦**记忆权限治理**：Naive RAG 是强基座；Governed Slot Graph 和 symbolic/neuro-symbolic reasoning 是辅助机制，图无法稳定解析角色、关系、权限，缺失解析不得视为授权或拒绝。主比较只看四领域 avg.MGS；U/A/F 用来解释原因。避免针对已见 checkpoint 写领域特判。

同一固定随机清单每领域 1 个完整 episode，共 102 checkpoint，历史已暴露，仅作开发诊断。官方 GPT-4o judge，`gate_by_action=false`。最新已测 4/20 图候选预算冻结快照 `outputs/v8_slot_graph_rag_majority_20260924`，runtime SHA256 `493c352a3428c2f42e225d22689191a56e9636a9badeb73787f6b836d2507c64`：avg.MGS **34.07%**，同题官方 Naive RAG **18.20%**，差 +15.87 个百分点；Gov-Mem 有 **6/102 执行错误**，RAG 为 0。Gov-Mem 四领域 MGS：Medical 25.93%、Office 26.67%、Education 55.56%、Household 28.13%。Household U 仅 37.50%（RAG 62.50%）。失败最坏情况 avg.MGS 31.25%。仅 4 集，不能声称总体或论文性能提升。

完整报告 `experiments/result/2026-09-24_Gov-Mem_slot_graph_four_domain_development_diagnostic.md`；机器结果 `outputs/v8_slot_graph_rag_majority_vs_rag_diagnostic_20260924/diagnostic_metrics.json`；同题历史 RAG 预测与 judge 子集按 checkpoint ID、官方源码 hash 原样复用，来源链在该输出 `protocol.json`。旧 fix2 同题 avg.MGS 25.07%，两次生成不能作因果消融。

代码现状：canonical V8 配置 `max_graph_candidates: 4`、dense Top-K 20，图失败回退 dense-only 并审计；相关 99 项测试通过。**在上述冻结运行之后**，工作区撤回了过宽的 lifecycle `resource_surface` 回退，恢复严格 source grounding；回退后 89 项相关测试通过。当前工作区不是已评分快照，未来新性能须重新运行，不能把 34.07% 直接归给当前工作区。源代码/报告/旧输出均未提交，工作区还有大量历史未跟踪文件，勿 reset/clean。

当前主要工作：解决 6 个通用合同错误与 Household 效用损失；固定协议后扩大完整 episode 样本。按 30 个不同 key 并发启动足够多独立 episode，4 集诊断则用 4 个不同 key。不要无理由重跑三种已完成 baseline 或旧消融。保留所有原始错误、区分网络故障、不要做成功样本筛选。

## 新会话从这里开始

用户退出是为了重启笔记本，当前任务没有取消。先读本文件和 `AGENTS.md`，然后继续完善 Gov-Mem。不要重新开始 baseline 实验，不要把“测试通过”说成“真实性能提升”。

工作目录必须显式设为：
`/mnt/data_disk_2/fuyali/codes/2027_TOIS_Gov-Mem`

所有代码、输出、缓存、临时文件必须在 `/mnt/data_disk_2`。不得使用旧项目 `/mnt/data_disk/home/fuyali/codes/2027_TOIS_Gov-Mem` 写入，不得用 `/tmp` 存实验文件。新实验先核实目录/软链接的真实目标。

工作区有大量历史修改和未跟踪文件；不要 reset、clean、覆盖或假定都来自本轮。最新 Slot Graph 改动保存在工作区，未在本轮提交 git。

## 用户目标与要求

1. 三种 base LLM（Gemini 2.5 Flash-Lite、gpt-4o-mini、gpt-5.4-nano）的 baseline 已测完并补跑网络失败。现在集中精力检查、提升 Gov-Mem。
2. 核心创新必须是 **Governed Slot Graph + Symbolic Rule Layer**，必须实际参与执行，不能只生成 debug artifact。要准确区分布尔规则、语义绑定、一般一阶逻辑推理；不能因代码里有 and/or/not 就声称实现了通用 FOL prover。
3. 继续审计词表：数量、每表词数、实际主路径是否使用、来源和测试集专有信息泄露风险。上一轮只是初步审计，不是完整调用链/全部常量的泄露认证。
4. LoRA 暂后置，先理清 pipeline。训练如有必要须先读 `docs/GOVMEM_V8_LOCAL_TRAINING_DECISION.md`；不得使用 reserved confirmation episodes 做训练/蒸馏；91 个历史 episode 全部已暴露，不能称 pristine holdout。
5. 默认新实验 30 个**不同** API key 并行，核验真实进程/不同 key 数，不能只看 worker 参数；独立任务少时减少并发，episode 内必须按历史顺序。不要记录凭据。
6. 网络/SSL/API 超时等导致未完成，必须明确及时报告，不能弱化为“普通执行错误”并宣布论文结果完成。用户明确授权网络失败换 key 重跑；失败尝试保留，完整 episode 从空 memory 重建，不按分数挑选。服务 content_filter 要与网络故障分开报告，不能保证换 key 一定解决。
7. 结果每个 baseline 一张表，4 个 domain（Medical/Office/Education/Household）、U/A/F/MGS，以及 avg.MGS。avg.MGS = 四 domain 原始 MGS 算术平均，不加权。单位 %；U/MGS 高好，A/F 低好。
8. 原实验快照和原始错误记录冻结保留。新代码结果必须独立运行，不与旧成绩混为一谈。

## 2026-09-24 历史进展：Slot Graph 初次接入

最后用户问“此刻进展如何”，已明确回复：代码接入和 76 项测试完成；**没有启动这次改动后的付费实验，无新 U/A/F/MGS**。下一步应四 domain 完整 episode 真实 API 验证，再决定全量 2218 query。

相关代码：
- `src/gov_mem/memory/v8_slot_graph.py`：本轮新增，V8SlotGraphIndex 适配器，复用现有 graph builder。
- `src/gov_mem/memory/governed_slot_graph.py`：现有 MemoryGovernedSlotGraph，实现 slots、policy/lifecycle 候选、nodes/edges、entity relation lists。
- `src/gov_mem/backbones/govmem_v8_late_governance.py`：接入图检索；与 dense 交错合并、按 memory ID 去重、截到原 Stage-1 Top-K；图审计写入 `answer_result.raw_response.governed_slot_graph`；图成功逻辑调用数计入 v8_cost。
- `configs/govmem_v8_late_governance_gemini25flashlite.yaml`：本轮启用 `memory_governed_slot_graph.enabled: true`，batch_size=8、top_k=20、dense_index=false。
- `tests/test_v8_slot_graph.py`：增量缓存、历史回退、失败不提交测试。
- `tests/test_v8_late_governance_backbone.py`：新增启用图的 pipeline 集成测试，模拟 dense 没命中而图证据进入 Stage 2 和回答。
- `docs/GOVMEM_V8_SLOT_GRAPH_INTEGRATION.md`：实现细节和边界。

实际流程：
可见历史 → Dense 检索 + episode-local Slot Graph 原文证据检索 → 合并候选 → 邻近上下文扩展 → Stage 2 LLM 抽取/权限范围绑定 → V8 event state projection → deterministic symbolic critic → safe evidence → Answer Agent。

重要边界：
- 这次实现的图首先是**证据检索图**；它的 policy 候选不直接授予权限，现有 V8 event store/绑定/critic 才执行授权检查。不能宣称已经有独立统一的图上 FOL rule engine。
- V8 仍保留 shallow policy/event ledger；不是把 memory mode 改成 full world graph。
- 图仅从可见历史抽取，按 observable.as_of_turn_id 排除当前问题 turn。
- 构图用 deep copy，batch failure 显式抛错且不提交图缓存/processed watermark；历史变短或相同 chunk ID 内容变化会重建。
- 缓存当前是进程内的；重启会从可见历史重建，未实现持久化图 resume。
- Stage-1 Top-K=20 不等于 Stage-2 全部可见文本上限，现有 adjacent/ingestion 上下文仍存在，需要做公平性和成本说明。
- graph 模块仍有 observed-token lexical matching 和通用短语 helper，不能声称整个系统完全无词法规则。
- 新集成测试图检索被 mock；它证明调用链，不能证明真实 LLM 构图质量或性能。必须再跑真实 API 验证。
- 需进一步检查图 builder 对不合格 records 的处理、初始/部分失败成本统计和重启行为；目前成功调用计入 logical calls，失败/重试以 provider telemetry 为准。

已运行：
`PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src pytest -q tests/test_v8_slot_graph.py tests/test_v8_late_governance_backbone.py tests/test_v8_late_governance.py tests/test_governed_slot_graph.py --basetemp=/mnt/data_disk_2/fuyali/govmem_caches/test_v8_graph3 -o cache_dir=/mnt/data_disk_2/fuyali/govmem_caches/pytest`
结果：76 passed。不是全仓测试。

## 旧 Gov-Mem 全量结果：不能作为新版本最终性能

目录：`outputs/v8_full2218_20260919`
- `full_metrics_with_missing_label.json`：91 episodes / 2218 checkpoints，275 execution errors。
- 113 transport errors（112 ReadTimeout + 1 ConnectionError）；151 ValueError、11 JSONDecodeError。
- 1 个 Medical privacy label 缺失；评分状态 failed / Incomplete applicable judgments。
- 旧 available-label avg.MGS 约 30.02%，不是标签完整的最终成绩，更不是本轮图版本成绩。
- 高频内部问题：prefill extraction validation、joint extraction、claim grounding、bind、deleted-release contract 等。
- 用户先要求审清架构/词表再完善图，因此**尚未开始旧 Gov-Mem 错误的大规模补跑**。

读 `experiments/result/2026-09-19_Gov-Mem-v8_full2218.md`、`experiments/result/2026-09-19_Gov-Mem-v8_module_ablation_results.md`。已完成付费消融不要无理由重跑。

## 词表与架构初审结论（不能当完整审计）

- `general_lexicon.py` 中 GENERAL_OBJECT_LEXICON、GENERAL_TOPIC_LEXICON、GENERAL_OBJECT_PREFIXES、GOVMEM_GOVERNANCE_ONTOLOGY 为空。
- `query_semantics.py` 多个旧 alias/cue 表为空。
- `v8_scene_schema.py` 通用结构：9 entity types、10 relation types、4 governance effects、5 lifecycle effects、3 temporal states、4 duty attributes；shallow 模式不直接使用完整 world-graph schema。
- config 的 access_policy.by_domain 是 GateMem 公共任务规则，需核对 baseline 同条件输入，不能当作自动学到的 ontology。
- `govmem_default.yaml` 仍是旧 V4 配置、slot graph disabled，**不能把它等同于当前 V8 canonical config**；命名/默认入口还需收口。
- `RESUME_GOVMEM.md` 下方大量历史“勿重跑/正在运行”已过时，优先按本文件和实际输出判断。

## 三种模型最终 baseline：不要重复跑

本轮交接前已再次核对相应 run_status：全部 complete。

### Gemini 2.5 Flash-Lite

统一最终目录：`outputs/gemini_baseline_network_repair_20260923`
- `official_metrics.json`
- `paper_tables/results.md`、`results.tex`、`table_audit.json`
- A-Mem、ReMem-I、ReMem-S 子目录分别为独立修复运行；Mem0 复用 `outputs/mem0_gemini25flashlite_full2218_20260920`。
- 修复 45 个 method-episode：A-Mem 13、ReMem-I 21、ReMem-S 11。
- 最终四 baseline：0 terminal execution errors，所有适用 judge 标签完整；hash 和覆盖已核验。
- avg.MGS：A-Mem **14.75%**、Mem0 **8.83%**、ReMem-I **16.41%**、ReMem-S **15.28%**。
- 报告：`experiments/result/2026-09-23_gemini_baselines_network_repair_final.md`。

### gpt-4o-mini

`outputs/baselines_gpt4omini_keyrot_retry_20260923`
- `official_metrics.json`、`paper_tables/results.md` / `.tex` / `table_audit.json`。
- 12 个受影响 episode 补跑成功，四方法最终 0 terminal errors、0 missing labels。
- avg.MGS：A-Mem **10.53%**、Mem0 **6.79%**、ReMem-I **5.85%**、ReMem-S **6.23%**。
- 最初 export 因 nano model assertion 停止，推理/评分已完成；随后只修复导表、核验并标 complete。`table_export_failure_original.json` 保留错误证据。
- 报告：`experiments/result/2026-09-23_gpt4omini_corrected_tables.md`。
- Naive RAG 单独目录：`outputs/rag_naive_gpt4omini_full2218_20260921`。

### gpt-5.4-nano

`outputs/baselines_gpt54nano_keyrot_retry_20260923`
- `official_metrics.json`、`paper_tables/results.md` / `.tex`。
- 最终四方法 0 terminal errors、适用标签完整。
- avg.MGS：Naive RAG **17.07%**、A-Mem **16.14%**、Mem0 **7.51%**、ReMem-I **13.54%**、ReMem-S **12.86%**。
- Naive RAG：`outputs/rag_naive_gpt54nano_responses_full2218_20260921/official_score.json`。
- 报告：`experiments/result/2026-09-23_gpt54nano_baselines_key_rotation_final.md`。

零 terminal errors 不代表内部调用全无失败：ReMem 官方回退警告仍在 source audits，应如实披露。
旧统一比较表可能仍有修复前数值；以上 final paths 为准。部分旧自动生成说明仍写“one bounded retry/no third attempt”，与后来用户明确授权的多次网络重跑不符，论文需按完整源目录链描述，勿照抄过时说明。

## 运行工具与后续注意

- `scripts/retry_nano_failed_episodes.py` 名字虽然有 nano，后续用于三个模型；有 --workers / --key-offset，冻结 source、补跑错误 episode、保留 retry_attempts。
- `scripts/run_nano_baseline_matrix.py` export 已兼容非 nano 方法集；非 nano 不自动加入 RAG。
- 当前轮换实现最多使用分配 key 池数量次 attempt；会在完整 episode 结束后判错误，不是单请求即时换 key。仍非“无限重试”；若未来使用必须明确报告上限，不声称自动永远重试。
- 旧 launcher 对工作区/冻结 harness hash 的处理有兼容逻辑；新运行应进一步保证验证实际执行 snapshot，而不是改 hash 掩盖变化。
- API key 不在此文件中。通过现有安全配置读取，不能打印。用户曾指定 `/mnt/data_disk_2/fuyali/codes/2027_PERSA/API-Key_OpenLux.md`；实际唯一数以运行时去重验证为准。
- 本轮没有新 Gov-Mem 付费任务在等待完成；不要向用户声称后台正在跑新版本实验。其他任务/进程要实际检查再判断。

## 推荐下一步

1. 读取上述图接入文档和代码，确认本轮边界，完成 canonical entry/config 与图审计输出检查。
2. 四领域少量完整 episode 做真实 API 集成验证（遵守可见前缀、逐集顺序、新输出目录）。评估图抽取完整性、source grounding、实际候选贡献、symbolic veto、成本和全部错误；不能靠 76 个合成测试宣称完成质量验证。
3. 修复高频 Gov-Mem contract/JSON 内部问题，并区分 transport failures；维护同一版本、统一补跑协议。
4. 验证通过后做新版本全量 2218 query + 官方 judge，再与对应模型最终 baseline 比较。保留旧 shallow 结果作历史参考；任何性能提升须实测。
5. 用户原始词表/ontology 公平性审计仍需完成可复现清单，包含实际调用范围、通用/领域/专有词来源，不可用“没有发现”替代证明。
