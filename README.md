# Gov-Mem

> Historical sections below are retained for provenance. For the active
> pipeline and current decisions, use the Current Status section above and
> `handoff.md`; do not treat older V3/V4/V7 numbers as current V8 results.

## Current Status (2026-09-25)

The active research system is **Gov-Mem V8 Late Governance**. It uses Naive
RAG as the primary retrieval path and adds two auxiliary permission-aware
components: a Governed Slot Graph observer and symbolic/neuro-symbolic
source, binding, lifecycle, and claim checks. The graph does not replace dense
Top-20 retrieval, directly grant permission, or directly deny an answer. It
adds an audit finding to the governance prompt when grounded evidence exists;
missing graph evidence is recorded as an audit gap.

The valid paired Governed Slot Graph ablation on 12 complete confirmation
episodes (306 checkpoints per arm; historically exposed data, not a pristine
holdout) is:

| Arm | Four-domain average MGS |
|---|---:|
| Graph-on | 17.22% |
| Graph-off | 15.57% |
| Difference | +1.64 percentage points |

The earlier graph-off attempt was invalid because its frozen configuration
still had `memory_governed_slot_graph.enabled: true`; it is retained only as
an audit artifact and must not be cited. The valid report is
`experiments/result/2026-09-25_Gov-Mem_confirmation_graph_prompt_ablation.md`.

No final full-2218-query Gov-Mem result for the current graph-enabled source
has been declared. Do not mix historical V4/V7/V8 scores with the current
pipeline. The next formal experiment must freeze one runtime/config pair,
run complete episodes, and report the four-domain arithmetic mean MGS as the
primary metric.

For recovery and current decisions, read `handoff.md` first, then
`RESUME_GOVMEM.md`. `VERSION_LOG.md` is historical provenance; dated reports
under `experiments/result/` contain the authoritative machine-linked results.

For a code-grounded explanation of the active implementation, see
[`docs/GOVMEM_V8_ARCHITECTURE.md`](docs/GOVMEM_V8_ARCHITECTURE.md). It explicitly
documents that the Governed Slot Graph is built from the full observable prefix,
not from Dense RAG Top-20.

This workspace uses GateMem as the default benchmark dataset for Gov-Mem.

Baseline统一OpenLux全量实验入口见 [baselines/README.md](baselines/README.md)：
`a_mem/`、`mem0/`、`remem_i/`、`remem_s/`各自维护方法配置与入口，
`models/`独立维护base LLM设置，支持后续固定方法更换模型。

## V8 全量2218 query（2026-09-19，已结束）

91个完整episode全部覆盖。四领域平均官方可用标签 **MGS 30.02%**，
失败最坏情况 **21.09%**；执行错误 **275/2218（12.40%）**，
其中113条网络/连接失败、162条解析/校验失败。
唯一Medical隐私标签经限定补评仍为空；全分母MGS边界 **29.97%–30.04%**，
不是置信区间。原始判分保留，未人工补标签；没有全量baseline配对或新训练。
详见 [全量结果、费用与评分缺口](experiments/result/2026-09-19_Gov-Mem-v8_full2218.md)。

当前 V8 调用链、每步职责、状态写入时机与诊断边界见
[Pipeline 源码核对说明（2026-09-19）](docs/GOVMEM_V8_PIPELINE_AUDIT_20260919.md)。
LoRA 暂缓，先梳理现有 pipeline 和历史失败轨迹；该说明不代表新实验成绩。

## V8 确认性评测与本地训练判断（2026-09-19）

另选未进入此前 V8 调优的四领域各3集、**306 checkpoints/组**，关闭旧更新折叠的
收益未复现：Full **31.18%**，全部保留旧更新 **24.90%**；错误 **12 vs 18**。
因此**未切换默认折叠行为**。旧开发集46.98%不能作为通用收益。
历史 V7 已跑过全部91集，新的样本也不能称 pristine holdout。

已确认 A100 80GB 资源；暂不训练，先准备独立可靠的权限/生命周期抽取标注。
新增嵌套拒绝格式兼容，358份历史响应仅恢复1份、无新增失败；相关测试108项通过。
此修复不在上述冻结评测中，当前源码新增修复后尚无端到端MGS。
[完整确认性结果、费用与训练判断](experiments/result/2026-09-19_Gov-Mem-v8_confirmation_and_training_readiness.md)。

## V8 模块消融（2026-09-19）

已完成 **5 组端到端消融＋4 组末端规则消融**，每组覆盖同样的 12 个完整 episode、
303 checkpoints。回退新语义提示：MGS **32.24%**；移除持久化账本：**33.14%**；
同信息平铺：**38.62%**；关闭来源扩展：**35.90%**；关闭旧更新折叠：**46.98%**。
默认 V8 仍为下述 **40.06%**，本轮未修改生产实现，不能将消融最优分替换成默认成绩。

主要发现：语义审查和累积状态有正向证据；复杂图表示的增益尚不明确；旧更新折叠
出现反向信号，应优先复核。末端 critic 的直接净效应仅约 **+0.64 个百分点**，
不能用它解释此前全部涨幅。开发集区间较宽，另发现一条官方 judge 的不稳定标签，
原始评分完整保留。详见 [完整消融、区间、费用与机制检查](experiments/result/2026-09-19_Gov-Mem-v8_module_ablation_results.md)。

## 当前 V8 完整复测（2026-09-19）

固定四领域各 3 个完整 episode、303 checkpoints 已完成：**官方平均 MGS
40.06%，RAG-Naive 19.75%（+20.31 个百分点）**。四领域 MGS 均超过 baseline；
将技术失败按最坏情况计入后仍为 **32.91%**。这是反复使用的开发样本，不是独立
holdout 或整个 GateMem 的结论。

- 原文 bank、浅层权限/生命周期表、Stage 2 语言审查和 Stage 3 前的符号否决保留。
  实测有 **8 次符号层独立将 LLM 放行改为拦截**，其中 6 次涉及删除。
- 候选引用统一为源 turn ID；只使用已提供的检索/graph/ingestion 原文。
  有效事件先入库，随后验证 query claim，避免重复抽取。
- 执行错误 **47 → 17**，仍未清零；Office Utility 低于 baseline，Education
  删除泄露率仍有 50%。这些限制没有从评分中剔除。
- Memory/answering 全部使用 **Gemini 2.5 Flash-Lite**。推理调用 **619 → 522**，
  但 token **3,164,624 → 3,336,077**，仍约为 baseline 的 **5.21 倍**；费用问题
  尚未解决。新官方评分另用 303 次 GPT-4o。前置失败联调费用单独列出。
- 当前源码与冻结实测版本一致。相关测试 **79 passed**；仓库 tests 共 **463 passed、
  13 项既有失败**。未新增 dev 版本。

[完整结果、费用、失败与复现身份](experiments/result/2026-09-19_Gov-Mem-v8_source_projection_random3_per_domain_paired.md)。

## 浅层方案建立与上一轮评测记录（2026-09-19）

用户明确要求：symbolic/neuro-symbolic reasoning 必须实际参与，graph 尽量浅用。
当前默认配置已选择 shallow 模式，直接修改 V8，没有新增 dev 版本。
模型接口使用精简 JSON（候选引用＋扁平 events），history 仍保留原文。
首次真实 TAB 协议联调系统性失败后已停止，未将不完整结果当成 MGS。

**原文 bank → RAG Top-20 → LLM 权限审查 → 符号时间/绑定核验与明确否决 → Answering。**

- 普通 history 不再重建 scene/entity/fact/通用 relation 图，只抽取权限与
  生命周期 EVENT；保留请求者一跳身份/职责证据，不多跳推理或按角色自动授权。
- 对整段安全记录，LLM 用 KEEP 引用 candidate ID，程序原样保留信息；
  混合记录仍按具体受限区间处理，避免重述压缩造成 Utility 损失。
- 符号层始终执行。端到端测试故意让 LLM 放行绑定到 deny 的信息，确认它被
  符号层拦下；另有权限到期与跨 checkpoint 状态复用测试。
- 默认仍是 Gemini Flash-Lite，通常两次主要调用。固定 12 集、303 checkpoints
  已全部测完：**官方平均 MGS 22.19%，RAG-Naive 19.75%**。但 V8 有 47 个执行
  错误，失败按最坏情况计分仅 **8.28%**；只有 Medical 超过 baseline，尚不能称稳定胜出。
- 实测有 3 次符号层独立将 LLM 放行改为拦截。推理 619 次、3,164,624 tokens，
  仍约为 baseline 的 4.95 倍。评分另用 GPT-4o；memory 全部为 Gemini Flash-Lite。
- 评测后继续离线修复通用协议：相同 413 份响应从 257 份通过增至 330 份，
  修复 73 份、无新增失败。**当前修复未重测 MGS，不能沿用冻结版本成绩。**

[完整评测、费用与限制](experiments/result/2026-09-19_Gov-Mem-v8_shallow_random3_per_domain_paired.md)。

[浅层方案、实现边界与验证](docs/GOVMEM_V8_SHALLOW_MEMORY.md)。
以下来源校验回放和完整图版本记录保留为历史，不作为浅层方案的评测成绩。

## V8 来源校验回放结果（2026-09-19）

继续修复了“Stage 2 看得到检索/graph 引文，但入库校验不认这些来源”的错误。
现在 claim 和 graph 共用实际提示词来源集合；旧记录不重复入库，新增记录
仍须有精确 NEW_TURN 引文。跨来源的相同被拒值不再被去重丢失。

相同的 310 份可校验历史抽取响应，关键校验失败由 **46 份降至 11 份**：
35 份修复，0 份新增失败。相关测试 **91 passed**；全仓库 **416 passed，
13 项既有失败**。这是不调用 API 的固定响应回放，**不是新的 MGS 结果**。
详见 [来源校验修复与回放](experiments/result/2026-09-19_Gov-Mem-v8_source_validation_replay.md)。

## V8 前序修复记录（2026-09-19）

在现有 V8 中继续修复，未新增版本入口、未启动付费重测：

- 默认配置显式提供官方 baseline 使用的四领域公开 access policy，逐字对照
  vendored 官方源码。它是应用规则输入，不是从 episode/答案生成的词表；
  通用推理代码不按领域硬编码授权。规则来源与 SHA256 记录在 prompt audit。
- lifecycle EVENT（包含不带 grantee 的删除/更新/取消）完整进入可见前缀的
  Stage 2 graph context；原先只传 FACT tombstone，会漏掉另一种合法表示。
- 其他 scene 的 allow 不再抵消当前 scene 的明确绑定 deny。
- 已通过来源/schema 校验的独立 graph delta 先入库；claim 出错后的修复或重启
  复用已处理 history，不再因回答字段错误重复抽取。无效治理事件仍不入库。
- 可选无损 text-pool 压缩离线回放 275 份已保存 prompt 全部还原一致，但输入
  字符仅减少约 3.1%，默认关闭；尚未测量模型理解和 token 费用影响。
- 相关离线测试 84 项通过。**这不是 MGS 提升证明**；已公布的 4.37%/19.75%
  属于之前冻结实现，历史结果不覆盖。

详情见 [本次修复与验证](experiments/result/2026-09-19_Gov-Mem-v8_correctness_followup.md)。

## 现行 V8：语言权限审查与合并调用（2026-09-18）

按用户要求，直接维护 `src/gov_mem/backbones/govmem_v8_late_governance.py`，
不创建 dev2。`govmem_v8_dev1_late_governance` 现在是同一实现的兼容入口，
并非独立冻结算法。此前 Medical/Atlas/Beacon 结果保留为历史诊断。

现行流程：Dense Top-20 + 有限相邻上下文 → 一次 Stage-2 调用同时抽取新增
权限/关系事件并做语言权限审查 → 确定性状态投影、精确来源/绑定校验 →
经过审查的原文 claims（含安全上下文）→ Answering Agent。

- 默认每个正常 checkpoint 两次主要 LLM 调用，完全拦截时一次；新增事件
  抽取包含在 Stage 2 中。超过 64 个新 turns 或 24,000 字符的冷启动前缀
  使用显式计数的预填充调用，默认每 checkpoint 最多四次；不会静默丢掉历史。
- 使用公开的小型通用结构 ontology，所有领域使用相同 schema；没有 GateMem
  实体/答案词表或按 domain 分支的推理代码。2026-09-19 起通过配置显式注入
  官方公开的任务访问规则；此规则输入与通用 ontology 分开管理。
- 权限图保留带原文来源的事件和关系；按可见前缀合并 scene、参与者和职责边。
  状态区分主体、资源、动作、范围、条件、签发者和时间，未来记录不会进入早期投影。
- LLM 判断权限的语义适用性。去除词语重叠、缺图边、场景名单和同主体传播的
  硬拦截；符号层只执行明确绑定的限制和结构校验。
- 拦截必须引用可见原文；同一原文中的受限区间不会夹带在放行长句里。
  Stage 3 不接收被拦值、原始混合记录或自由文本拒绝理由。
- `json_max_attempts: 1`，HTTP 最多两次尝试。完整 episode 联调发现协议错误后，
  新增 `max_contract_repairs: 1`：Stage 2 仅在协议校验失败时最多再生成一次，单独计费。
  默认遇错即停；固定样本评测开启 `record_execution_errors`，将最终失败明确记为
  `action=error` 并继续整集，不冒充成功拒答，也不删除失败 checkpoint。
  费用审计含重试、延迟及
  provider 返回的 tokens；没有 usage 时不凭空估算费用。

默认配置仍是 `configs/govmem_v8_late_governance_gemini25flashlite.yaml`：
`gemini-2.5-flash-lite`、temperature 0、`text-embedding-3-small`、Top-20。
新实现须使用新输出目录，避免加载旧 schema 的 event cache；不覆盖历史实验输出。

本地验证与详细边界见 [V8 设计说明](docs/GOVMEM_V8_LATE_GOVERNANCE.md) 和
[实现验证记录](experiments/result/2026-09-18_Gov-Mem-v8_inplace_implementation_validation.md)。
固定随机样本配对评测已经完成（seed 20260918，四领域各 3 个完整 episode，
每种方法 303 checkpoints）：**冻结 V8 平均官方 MGS 4.37%，官方 RAG-Naive
19.75%**；V8 有 28 个执行错误，chat tokens 为基线 5.47 倍。该快照未达到目标。
详见 [完整结果与口径限制](experiments/result/2026-09-18_Gov-Mem-v8_random3_per_domain_paired.md)。

随后按用户要求，当前 V8 原地切换为 `response_protocol: lines`：Stage 2
每条信息一行，字段以 TAB 分隔；来源单独一行。history 提取和 Stage 2
不再要求模型生成 JSON；入库仍由 Python 序列化为结构化 JSON。Stage 3 保持
原有接口：JSON 包装中包含文本 answer 和引用信息，最终回答仍是文本。
Stage 2 原文响应保存在 `v8_text_responses/<dataset>/*.stage2.txt`。
每条 claim 必须显式填写 `bind=事件ID` 或 `bind=NONE`，程序根据选中的已知事件
生成符号绑定，不再依赖模型重复输出 resource/action/scene 字段。

**逐行文本修订只完成离线验证，尚未付费重测。上述 MGS 属于此前冻结 JSON
快照，不能当作当前文本协议的成绩。** 模型仍是 Gemini，未新增 dev2。

## 历史接管说明：原地修改前的诊断与计划（2026-09-18）

本节保留原地修改前的研究约束和诊断背景；涉及冻结 dev1、另建 dev2 的旧计划
已由本文开头的用户要求取代。下述性能数字均属于旧实现，不能当作现行 V8 的结果。
工作树包含尚未提交的 v8 开发文件，接管前先运行
`git status --short`，不得覆盖或清理用户已有修改。

### 不可变研究约束

以下约束是 Gov-Mem 后续所有版本和实验的硬性要求，不得为提高某个 episode
分数而放宽：

1. 默认数据集是 GateMem。`dataset/GateMem/` 中的原始数据只读，不得修改。
2. 实验选择单位必须是一个完整 episode 的全部 checkpoints。禁止随机抽取、
   人工挑选或只运行 episode 内部分 checkpoints 后声称获得可比较结果。
3. 运行时只能看到当前 checkpoint 的 observable prefix，禁止读取未来 episode
   suffix、gold answer/evidence、`expected_action`、`judge_spec`、`leak_targets`、
   数据集 `query_type`、oracle evidence、rationale 或 scorer 字段。
4. memory-system base LLM 固定使用 `gemini-2.5-flash-lite`（除非用户之后明确
   改变要求），temperature 为 `0.0`。当前 embedding 是
   `text-embedding-3-small`，Stage-1 dense retrieval 保留 RAG-Naive Top-20。
5. Gov-Mem 必须保留 symbolic reasoning；不得退化成纯 LLM/RAG。Symbolic 的
   职责是保存可审计的 entity/relation/permission/lifecycle/state、做确定性的
   source grounding、时间投影和末端 claim-level hard veto。Symbolic 不应在
   早期大面积删除候选内容，也不能用“未知即拒绝”的方式牺牲 Utility。
6. Language reasoning 是主要的语义抽取和细粒度授权判断能力；symbolic
   reasoning 辅助、校验并拦截明确违规。最终 Answering Agent 只能看到通过
   late governance 的 safe claims，不能看到被拦截值。
7. 可以使用一个小型、冻结、公开、benchmark-independent 的结构 ontology，
   例如 principal、role、scene、resource、allow、deny、revoke、update、delete、
   current、historical 以及通用 relation types。禁止预置 GateMem 实体名、答案
   值、case-specific 短语、从 GateMem 提前统计出的词表或 dataset classifier。
8. entity、relation、permission、lifecycle 抽取必须由 observable prefix 实例化，
   并保留精确 source span。抽取质量是图、缓存和状态投影可用的前提，不能只看
   最终分数而跳过抽取审计。
9. 必须保住 RAG-Naive 的 Utility：dense Top-20 是主召回通道，图和状态是辅助
   信号；不能假设“query 阶段只做图检索和状态投影”就足够。
10. 用户已明确要求直接修改现有 V8，不再创建 dev2。保留历史结果，
    在 Git 和 `VERSION_LOG.md` 中记录变更；旧 dev1 入口仅为兼容别名。
11. 本地 action accuracy、结构审计和少量 complete-episode 结果都只是诊断。
    U/A/F/MGS 只有在规定的 official GateMem scorer/protocol 下才能作为正式
    benchmark 指标；不得把诊断结果包装成官方或论文结果。
12. 目标调用量是接近 RAG：正常 checkpoint 原则上不超过两次主要 LLM 调用，
    repair 必须是有统计依据的异常路径。每次实验必须记录 provider requests、
    retries、latency 和 requests/checkpoint。

### 当前实现

当前新架构为 v8 Late Governance：

```text
observable prefix
  -> dense Top-20（保持 RAG 主召回）
  -> bounded adjacent-turn context
  -> incremental scene/entity/relation/permission/lifecycle extraction
  -> append-only episode event cache
  -> checkpoint-time state projection
  -> Stage-2 atomic claim ledger（language reasoning）
  -> late symbolic critic（仅明确证据触发 hard veto）
  -> value-minimal safe claims
  -> Answering Agent
```

版本入口：

- 冻结诊断版本：`src/gov_mem/backbones/govmem_v8_dev1_late_governance.py`
- 冻结 mode：`govmem_v8_dev1_late_governance`
- 配置：`configs/govmem_v8_dev1_late_governance_gemini25flashlite.yaml`
- 配对 baseline：`configs/rag_naive_v3_paired_dev1_gemini25flashlite.yaml`
- 设计说明：`docs/GOVMEM_V8_LATE_GOVERNANCE.md`
- 完整版本记录：`VERSION_LOG.md`

v8 已实现 scene schema、closed observable principal registry、精确 source-span
验证、增量 event cache、checkpoint 投影、permission/revoke/delete/cancel、删除
tombstone、claim ledger、末端 symbolic critic、safe-evidence boundary，以及完整
episode manifest 强校验。v8 按 visible-prefix 长度处理 checkpoints，即使原数据
按 query type 分组，也不会破坏 episode 时间顺序。

### 已完成的完整 episode 验证

已经创建并保留三个完整 episode manifests：

- `experiments/manifests/v8_education_atlas_complete_episode.json`
- `experiments/manifests/v8_education_beacon_complete_episode.json`
- `experiments/manifests/v8_medical_early_pregnancy_complete_episode.json`

Atlas（18 checkpoints）最终诊断为 17/18 action correct，Safety 6/6，未观察到
privacy/forgetting violation；最后一个 mixed-record 问题随后用通用规则修复。
Beacon（18 checkpoints）用于发现 summary redaction、虚假 explicit permission、
operational-duty provenance 和 sibling propagation 问题；为避免针对同一 episode
反复调参，没有持续刷 Beacon。

最新 Medical 完整 episode 包含 28 checkpoints（10 Utility、9 Privacy、9
Safety），Gov-Mem v8-dev1 与 RAG-Naive 配对运行均正常完成，均为零 HTTP retry。
本地诊断如下；它不是 official MGS：

| System | Action | Utility action | Privacy action | Safety action | Privacy violations | Forgetting violations |
|---|---:|---:|---:|---:|---:|---:|
| Gov-Mem v8-dev1 | 20/28 | 5/10 | 6/9 | 9/9 | 1/28 | 0/28 |
| Paired RAG-Naive | 17/28 | 8/10 | 3/9 | 6/9 | 3/28 | 0/28 |

v8 共调用 chat provider 65 次（2.32/checkpoint），配对 RAG 为 59 次
（2.11/checkpoint）。v8 不再是 RAG 十倍调用量，但仍需减少非必要的 query-time
调用。完整诊断和逐项根因见：

`experiments/result/2026-09-18_Gov-Mem-v8-dev1_medical_complete_episode_paired_diagnostic.md`

输出目录：

- `outputs/govmem_v8_dev1_medical_early_pregnancy_complete28_20260918`
- `outputs/rag_naive_v3_paired_medical_early_pregnancy_complete28_r2_20260918`

Medical 结构审计：225 visible turns，25 次 logical incremental extraction，最终
event store 有 2 scenes、46 entities、33 facts、26 relations、20 governance
events；跨 checkpoints 共 release 17 claims、block 31 claims。结构/source-grounding
审计通过不等于语义 F1；真正的 extraction F1 仍需 episode-disjoint 人工标注。

### 当前最重要的已知问题

1. **Scene 增量合并不一致。** Relation store 已抽到
   `nurse_casey_liu participates_in`，但 canonical scene participant roster 没有
   回填；`scheduler_jules` 也缺失。late critic 因此把护士和 scheduler 的合法
   operational-duty 请求误判为 scene mismatch，造成 Utility 过拒。
2. **Scene membership 太粗。** `billing_cho` 在大场景 roster 中，当前 critic
   就错误地把它当成访问 hCG 临床解释的 operational duty，造成 Medical
   checkpoint 16 的唯一隐私泄漏。同一 episode 的参与者身份不等于字段权限。
3. **Permission resource 对齐太弱。** 当前 hard veto 仍依赖 claim 与 resource
   surface/scope 的 token overlap，出现无关电话限制误伤 family-access setting。
4. **Sibling veto 边界过宽。** 一个 clinical field 被拦后，会按同一 subject
   传播到独立的安全字段，例如 suite location。
5. **Action label 有边界错误。** scoped permission 下已经完整回答所有 requested
   fields 时仍输出 `answer_redacted`；只有确实存在被拦 requested slot 时才应使用
   redacted。
6. **抽取评测尚不充分。** 当前只有 schema/source-grounding audit，没有独立的
   entity/relation/permission semantic precision/recall/F1 标注集。

### 下一步执行计划（按顺序）

1. **先保持 dev1 冻结。** 不得基于本次 Medical episode 原地修改
   `govmem_v8_dev1_late_governance.py`。
2. **建立 v8-dev2 独立文件/config。** 文件名必须可直接辨认版本，并在
   `VERSION_LOG.md` 登记；不要把实验性修复写回 dev1。
3. **修 scene/event projection。** 将 `participates_in` relation 确定性合并进
   canonical scene roster；处理同 ID scene 的 participant/status/source-span
   增量 union，并补充一致性测试。
4. **重做 operational duty 表示。** 抽取/投影 typed duty edge：
   `(principal, role, action, resource_category, subject/scene, source_span)`。
   同场景 membership 只能做候选信号，不能单独授权。billing、reception、nurse、
   scheduler、pharmacist 等必须按请求字段和可见职责语言判断。
5. **重做 permission matching。** 优先匹配 source-grounded resource ID、resource
   category、action 和 scope；自然语言模型处理模糊范围；token overlap 只能做
   advisory，不能单独 hard veto。
6. **收窄 sibling propagation。** 仅在同一 protected record/resource family
   内传播，不能只凭 subject 相同。
7. **修 action rendering。** `answer_redacted` 必须由“至少一个 requested slot
   被 block 且至少一个被 release”推导；完整 scoped answer 仍是 `answer`。
8. **先写 synthetic/unit regressions。** 覆盖 Medical 2/3/5/8/9/13/16/19 的
   失败机制，但测试和运行时代码不得硬编码 GateMem 答案值或专属触发短语。
9. **建立 extraction gold audit。** 从与调参 episode 隔离的完整 episode 选取
   observable prefixes，由人工标 scene/entity/relation/permission/lifecycle，
   单独报告 micro/macro precision、recall、F1、source-span exactness 和 invalid
   record rate。
10. **验证顺序防止过拟合。** unit/synthetic -> 一个新的完整 episode -> 第二个
    未见完整 episode -> 最后才回归 Medical/Atlas/Beacon。任何阶段都不能随机抽
    checkpoints。
11. **每轮配对 RAG。** 使用相同 episode、LLM、embedding、Top-20、temperature
    和 evaluator；比较 Utility、Privacy、Safety、泄漏、调用量和 latency。
12. **达到 promotion gate 后再跑全量 2,218。** 建议 gate：Safety 不低于 RAG，
    Privacy violations 明显更低，Utility action 不低于 RAG 超过可解释容差，
    chat requests/checkpoint 接近 2，且 extraction audit 无结构性缺陷。未达到前
    禁止用昂贵全量实验代替诊断。

### 接管后的第一组命令

```bash
cd /mnt/data_disk/home/fuyali/codes/2027_TOIS_Gov-Mem
git status --short
PYTHONPATH=src pytest -q \
  tests/test_v8_event_extractor.py \
  tests/test_v8_late_governance.py \
  tests/test_v8_late_governance_backbone.py \
  tests/test_v8_state_projector.py
PYTHONPATH=src python scripts/audit_v8_extraction.py \
  --output_dir outputs/govmem_v8_dev1_medical_early_pregnancy_complete28_20260918
```

Focused v8 tests 在本次交接前为 `24 passed`。全项目测试的已知基线是
`350 passed, 13 failed`；13 个失败属于 clean HEAD 已存在的 legacy
`field_state_projection` / `stateful_policy` 测试，不是 v8 新回归。接管者应重新
运行确认，并对比失败集合，不能为了变绿而删除或放宽 legacy 测试。

## Current Research Snapshot (2026-09-17)

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

### Known limitations of the current version

The current lexicon-free semantic-compiler version is an experimental snapshot,
not a final paper-ready design. Three unresolved problems must be addressed
before another full benchmark run.

#### 1. Pipeline token cost is too high

The Gov-Mem pipeline, excluding the required official judge, makes substantially
more model calls than RAG-Naive. Each checkpoint can invoke query induction,
atom extraction, extraction repair, a second question-analysis pass, mixed-query
reranking, and final answer generation. The same Top-20 evidence can therefore
be sent to the model several times.

Existing full-run telemetry records 10,766 main-pipeline chat calls for the
2,218-checkpoint GPT-4o-mini run, 15,537 for GPT-5-mini, and 18,445 for
DeepSeek-V4-Flash. JSON parse failures are especially expensive because
`LLMClient.chat_json` currently repeats the complete paid request up to three
times. The semantic repair path is also no longer exceptional: it was triggered
for 64.5% to 87.2% of checkpoints in the audited full runs. At this cost, routine
full-benchmark iteration is not practical.

#### 2. The measured gain over the reproduced baseline is too small

With the same live Gemini-2.5-Flash-Lite alias and official evaluation protocol,
the reproduced GateMem RAG-Naive baseline obtains 24.37% mean-domain MGS. The
current Gov-Mem run obtains 25.63%, an absolute improvement of only 1.26
percentage points. Gov-Mem reduces access-control and forgetting leakage, but
its pooled Utility falls from 59.88% to 35.30%. The resulting cost/performance
tradeoff does not support the current multi-call design as a final method.

The paired reports are:

- [`experiments/result/2026-09-17_GateMem_official_RAG-Naive_full_all_2218_openlux_gemini25flashlite.md`](experiments/result/2026-09-17_GateMem_official_RAG-Naive_full_all_2218_openlux_gemini25flashlite.md)
- [`experiments/result/2026-09-11_Gov-Mem-v4-Symbolic-full_all_2218_openlux_gemini25flashlite_entity_list_off.md`](experiments/result/2026-09-11_Gov-Mem-v4-Symbolic-full_all_2218_openlux_gemini25flashlite_entity_list_off.md)

#### 3. Lexicon removal went too far

The previous implementation contained broad deterministic vocabularies. Some
entries were legitimate general concepts, while others were close to GateMem
entities, event descriptions, or answer-bearing phrases and could reasonably
be criticized as benchmark-derived information leakage. Removing those
benchmark-specific triggers was necessary. The current version, however, also
removed the general-purpose vocabulary: `GENERAL_OBJECT_LEXICON`,
`GENERAL_TOPIC_LEXICON`, and `GOVMEM_GOVERNANCE_ONTOLOGY` are empty in the
paper-facing path.

As a result, repeated LLM calls dynamically reconstruct basic concepts that can
be declared independently of GateMem, such as principal, role, time, location,
credential, allow, deny, revoke, update, delete, current, and historical. This
raises cost and makes extraction less stable. The intended next design is a
small, frozen, published, benchmark-independent ontology of structural types,
governance operators, lifecycle operators, and relation types. Concrete names,
roles, resources, and values must still be instantiated only from the observable
conversation prefix; GateMem entity names, answer values, evaluator labels, and
case-specific trigger phrases must not be included.

The target pipeline should use one bounded semantic pass only when deterministic
structure is insufficient, followed by deterministic governance/state reasoning
and one final answer call. A normal checkpoint should require at most two main
model calls, with repair reserved for a measured exceptional path.

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

Commit `96d581d` (tag `v7-pre-lexicon-hardening-20260909`) remains a recoverable
pre-cleanup snapshot. Since then, the four residual semantic tables identified
by the audit were removed: attribute-type terms, location-specific terms,
policy scope terms, and generic-object terms. The paper-facing v4 path now
uses the query-conditioned LLM contract plus source-grounded atoms; it does
not use a GateMem-specific ontology, trigger table, or dataset-trained
classifier.

Historical compatibility modules still contain disabled helpers and closed
schema enums. They are not imported by `experiment.mode: govmem_v4_symbolic`.
The remaining regular expressions are structural validators (JSON, timestamps,
source spans, IDs, and provider-safe answer checks), not lists of GateMem
entities, values, or answer-bearing phrases.

At runtime, semantic induction, extraction, repair, governance, and answering
must use only the adapter's observable prefix and the closed Stage-1 Top-20
evidence set. They must not receive or read gold answers/evidence,
`expected_action`, `judge_spec`, `leak_targets`, dataset `query_type`, oracle
evidence, rationale, scorer fields, or any future episode suffix. Episode-local
principals, roles, and relations may be retained only when they were observed
in that runtime-visible prefix; they are not a fixed predefined vocabulary.

The post-cleanup validation ran one complete episode per domain (102
checkpoints), with OpenLux `gpt-4o-mini` for the memory system and OpenLux
`gpt-4o` for the official judge. Memory and judge pools each had 20 keys, judge
concurrency was 20, and all 102 judge records parsed successfully. Domain MGS
was Medical 20.00%, Office 39.49%, Education 13.89%, and Household 28.13%;
the four-domain arithmetic mean was **25.38%**. This is a development smoke,
not the 2,218-checkpoint paper table. It is 0.82 percentage points below the
previous cleanup smoke (26.20%), so the removed tables were compensating for a
semantic-recall gap; no table was restored.

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

The optional Memory Governed Slot Graph is intentionally only an
entity-relation index.  Each grounded entity owns an append-only relation list
(`entity_lists[entity_id]`); facts, permission changes, and lifecycle events
are appended with their source chunk/message/span.  `current_permission()` is
just a latest-event projection, so an `allow -> revoke` sequence remains
auditable in full.  The graph proposes source chunks for retrieval and never
decides authorization.  In practice the entities can be slightly abstract
matters such as a medical condition, a meeting's time/place, work-system
access, or an appointment; these labels are induced from the observed source,
not hard-coded into the implementation. Entity-history retrieval uses a small
structured lexical index and does not issue a second embedding query; repeated
identical assertions are collapsed while distinct changes remain in order. The
list is retrieved as an auxiliary Stage-2 authorization signal: it may help
decide whether the requested entity/relation is answerable, but it does not
add, replace, reorder, or summarize the source chunks used for factual answer
generation.

The graph-node dense index is disabled by default because it is not used for
factual answering or authorization replay; enabling it is only a compatibility
ablation. Thus normal query-time graph work is a deterministic entity/resource
token lookup over the compact lists.

When checkpoints from one episode are processed in chronological order, the
backbone keeps the episode-local graph in memory and sends only newly observed
history chunks through the extraction LLM. A non-monotonic or resumed
checkpoint prefix resets that cache, so a later turn can never become visible
to an earlier checkpoint. This is the same write-time pattern as A-Mem/Mem0:
the extraction cost is paid when new history arrives, while query-time list
retrieval remains a compact deterministic lookup.

Terminology is deliberately kept separate: this component is never abbreviated
as `MGS`. In GateMem, `MGS` means the paper's performance metric, the Memory
Governance Score `U * (1 - A) * (1 - F)`.

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
- [Current V8 implementation and contribution notes](handoff.md)

When reporting a new result, record the commit, config, model/provider,
manifest, evaluator, and whether long-context or gold-derived feedback was
enabled. Commit improvements as new history points instead of overwriting the
current snapshot.

## 迁移到新服务器

历史迁移说明中的旧项目目录为（仅供溯源）：

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
- `experiments/result/`、`README.md`、`handoff.md`、`VERSION_LOG.md`：结果、协议和版本记录；
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
  provider: openlux
  api_key_env: OPENLUX_API_KEY
  api_base: https://api.openlux.ai/v1
  base_model: gemini-2.5-flash-lite
  role_models: {}
```

- `base_model`: the default LLM used by the framework for memory ingestion, query planning, action decision, reasoning, and answering stages
- `role_models`: optional per-role overrides such as `memory_ingestion`, `query_planning`, `action_decision`, or `answering`

The repository default (`configs/govmem_default.yaml`) uses OpenLux
`gemini-2.5-flash-lite` for the memory system. Historical experiment configs
that name another model, including GPT-4o-mini baselines, remain unchanged for
reproducibility.

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

- `configs/govmem_v4_symbolic_openlux_gemini25flashlite_embedding3small_semantic_compiler.yaml`

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
