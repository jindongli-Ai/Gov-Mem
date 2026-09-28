# V8 模块消融：完整 episode 实验记录

本次是模块归因实验，canonical V8 未修改。正式模型继续保留 symbolic reasoning。
所有运行仍为四领域各 3 个完整 episode、303 checkpoints；这是已反复使用的开发集，
不是 holdout。推理模型固定 Gemini 2.5 Flash-Lite、temperature 0；官方 judge 为 GPT-4o、
temperature 0、gate_by_action=false。失败不删除、不当作正确拒绝。

## 第一阶段：固定 Stage 2 的末端规则直接效应（已完成）

对全部 286 个正常检查点，从实际 API 响应重建经过来源/协议校验、尚未 critic 的 ledger，
逐字段复现历史 critic 的 claims、slots、action、rejected_claims 与 veto audit，
并验证原 Stage 3 请求完全一致。17 个原始执行错误保留。
没有猜测原始 delivery，也没有撤销 LLM 已作出的 block。

| 对照 | 平均 MGS | Full − 对照（百分点） | episode bootstrap 95% 区间（百分点） |
|---|---:|---:|---:|
| Full | 40.0573% | — | — |
| 去掉整个末端 critic | 39.4136% | +0.6437 | [−0.1736, +1.6667] |
| 仅去掉绑定删除否决 | 39.0638% | +0.9935 | [0, +2.1484] |
| 仅去掉绑定权限否决 | 40.0573% | 0 | [0, 0] |
| 仅去掉 deleted 标签落实规则 | 40.4071% | −0.3498 | [−1.2821, 0] |

区间是按领域分层、以完整 episode 为抽样单位的精确 bootstrap：枚举全部
27⁴ = 531,441 种有序重采样组合，避免有限次 Monte Carlo 在零附近抖动。每领域只有
3 集，解释仅限探索性证据。它没有覆盖 provider/judge 的重复运行波动，也没有做多重
比较校正；单次点估计不是稳定因果结论。

重要区分：

- 真正 release→block 共 8 个 claim。关闭 critic 影响 73 个检查点，是因为规则标签
  也进入 blocked_slots，并影响确定性的 no_memory/refuse 文案。不能把 73 次输入变化
  说成 73 次纠正放行。
- 删除绑定规则关闭后，出现 3 条新增删除泄露、1 条新增隐私泄露，但也修复了 1 道
  Utility 题；说明有实际拦截收益，也有错误 scope binding 引发的过度拦截风险。
- 权限否决关闭后虽然改变两个检查点的末端计算，最终适用指标未变；这只能说明
  在该小样本中没有量到净收益，不能证明该规则普遍无用。
- deleted 标签的负贡献不能直接作为删除规则的理由。其中 Office ckpt_22 的 Full
  回答仅为 “The requested information has been deleted and is not available.”，
  官方 judge 却写成“confirmed a deleted token prefix”。这是可疑判分，原始标签和
  分数完整保留，未人工改分。另外一题还涉及改变标签后 Answering 的输出变化；
  temperature=0 不能排除生成波动。
- 这是条件于共同 Stage 2 和历史状态的**直接效应**，不是移除整个 symbolic 系统的
  总效应。权限/生命周期抽取、来源校验、状态投影对 LLM 的帮助仍然存在。

新费用：17 次 Gemini 请求、10,010 tokens；73 次 GPT-4o 请求、57,588 tokens；
HTTP 失败 0。另对全部 7 个适用结果发生变化的检查点、14 个不同回答做两次独立
judge 复核（事后敏感性诊断，不是重算 benchmark）：只有上述 Office ckpt_22 的 Full
判分出现不一致，原 True，两次复核均 False；其他 13 个回答的适用结果一致。
所有原始官方标签仍保留，不用复核标签替换原成绩。相同末端计算跨消融组共享答案，官方判分只复用相同 checkpoint、action、
answer、structured answer；固定数据与 official code，其他检查点仍完整计入。

## 第二阶段：从空状态重跑的端到端消融

这些对照都冻结源码、manifest 与配置，从每集空 event store 开始，完整重建后续状态。
共享的只有由文本和模型标识的 embedding cache；不共享 chat 结果或 event store。

| 实验 | 唯一目标干预 | 解释边界 |
|---|---|---|
| old_semantics | 回退旧版 Stage 2 语义提示，沿用新版稳定 JSON contract | 测新增语义指令整体，不把 TAB/JSON 协议差异混入 |
| flat_ledger | 同样 graph 信息改成逐行路径和值；保留 ID、引文、时间、scope、空值 | 测表示方式；后台 typed graph/critic 仍保留，不是“无图” |
| no_persistent_ledger | 不抽取/积累 permission/lifecycle ledger；保留原文 RAG、邻近记录、身份和语言审查 | 系统级移除账本；同时失去账本引文与绑定，不能单独归因于表示 |
| no_source_expansion | graph/ingestion 仍用于审查，但只允许 RAG/邻近候选提供回答 claim | 测额外证据供给；配套提示明确候选限制，协议失败也计入 |
| no_advisory_projection | 不折叠旧普通 update/supersede/cancel，所有删除信息仍保留 | 测旧状态整理；同时报告 tokens 与错误，而不只 MGS |

只看到平铺表示的效果，并不能判断账本是否有用；只有结合 no_persistent_ledger，
才能区分表示收益与累积状态/证据供给收益。以上单项差值也不能相加。

全部五组已经完成：每组 12 个完整 episode、303 checkpoints。原始评分未修改。

| 对照 | MGS | Full − 对照（百分点） | 精确 bootstrap 95% 区间（百分点） | 错误 | 失败最坏情况 MGS | 推理 calls / tokens |
|---|---:|---:|---:|---:|---:|---:|
| Full | 40.0573% | — | — | 17 | 32.9092% | 522 / 3,336,077 |
| 旧语义提示词 | 32.2399% | +7.8175 | [+0.2165, +16.8307] | 18 | 24.5857% | 545 / 2,845,883 |
| 同信息平铺账本 | 38.6188% | +1.4385 | [-7.5773, +8.6830] | 19 | 34.7046% | 510 / 3,451,036 |
| 移除持久化账本 | 33.1385% | +6.9188 | [+0.0325, +14.7812] | 11 | 28.3414% | 531 / 1,979,918 |
| 关闭额外来源选择 | 35.8956% | +4.1617 | [-5.4771, +14.7342] | 14 | 30.8817% | 524 / 2,970,690 |
| 关闭旧更新折叠 | 46.9814% | -6.9241 | [-15.2123, +0.6255] | 15 | 41.2877% | 521 / 3,568,120 |

### 现在可以支持的结论与 pipeline 决策

1. **新增语义提示优先保留。** 回退后 MGS 下降 7.82 点，错误仅 18 对 17；这不是主要靠解析成功率解释的差距。不能把这部分贡献归到末端符号规则。
2. **浅层持久化状态有正向证据，但需要降低成本。** 移除后下降 6.92 点，信息帮助主要出现在 Education 与 Household；同时 tokens 减少约 40.65%。其精确区间下端仅高于零约 0.03 点，不能忽略重复运行和多重比较的不确定性。
3. **复杂图表示没有得到支持。** 同信息平铺只下降 1.44 点，区间很宽且跨零。这不是无图实验，但说明应优先保护有效状态和引用，而不是增加图的复杂度。
4. **来源扩展有覆盖收益，也有副作用。** 关闭后整体下降 4.16 点；Education Utility 从 88.89% 降到 50%，但 Household MGS 反而提高约 10.09 点。它同时改变可用证据集合及授权判断，不能全算成 reasoning 的价值。
5. **旧更新折叠是优先停用/重做的候选，不能再默认视为有益。** 关闭后 MGS 达到 46.98%，四领域均上升，失败最坏情况 MGS 也从 32.91% 升到 41.29%；tokens 仅增加约 6.96%。不过置信区间仍跨零，不能把一次开发集胜出写成已稳定证明的因果结论。
6. **末端符号删除绑定保留为明确违规的兜底。** 已有新增泄露被阻断的配对实例；其效应规模约 1 点，不是最近 17.87 点增幅的主体。权限否决在本样本中无净 MGS 收益，尚不足以判无用；deleted 标签的负估计受 judge 波动污染，不据此删除规则。

上述单项差值不能相加：模块间存在交互，五组端到端干预还会改变后续抽取/缓存。不能据此写成某模块解释了历史涨幅的固定百分比。

no_persistent_ledger 仍保留来源校验、安全证据构造及 source_grounded_deletion_tombstone 规则；没有持久化事件及其绑定，不等于彻底删除所有 symbolic 操作。正式模型本轮未改动，46.98% 是冻结消融实现的实测结果，不能替换成当前默认 V8 的成绩。

### 折叠的独立机制检查

离线构造不含 benchmark 词表的反例：同一个 visit 资源，事件 A 更新地址，事件 B 更新到访时间；两者同 issuer/grantee/resource/action/scene。现实现会隐藏 A，只保留 B，即使 B 从未取代地址。**绑定相同不足以证明语义完整替代。** 原文仍在 bank 并不保证它出现在当前 RAG 候选中。

该反例说明折叠条件不充分，并不证明它解释了每一道 benchmark 的改善。反例保存在 `2026-09-19_v8_advisory_folding_counterexample.json`。后续应先检验取消这项信息丢弃的独立集表现，再决定默认配置；不能按各领域成绩硬编码分支。

### 统计与费用边界

先前中途更新使用 4,000 次 Monte Carlo bootstrap，靠近零的区间端点有抽样抖动；本报告统一改用所有 531,441 种组合的精确区间。所有分数与样本保持不变。区间仍未计入 provider/judge 的重复运行方差，且未做多重比较校正。

本轮新增总请求（含五组完整推理、四组末端分支和两次 judge 敏感性复核）：

| 模型 | 请求 | tokens | HTTP 失败 | 缺失 usage |
|---|---:|---:|---:|---:|
| gemini-2.5-flash-lite | 2,648 | 14,825,657 | 0 | 0 |
| gpt-4o | 932 | 762,879 | 0 | 0 |

推理全部为 Gemini；GPT-4o 仅用于官方判分与判分复核。没有新增 embedding API 请求。实际金额以 OpenLux 账单为准，没有按未知单价估算。已验证全部 1,515 个端到端预测、源码/config hash、17 个原始错误的末端保留，以及所有官方适用判分非空、parse_ok。

### 数据与复现

- 汇总机器可读结果：`experiments/result/2026-09-19_v8_module_ablation_audit.json`。
- 端到端评分：`outputs/v8_pipeline_ablations_scored_20260919/module_metrics_exact.json`。
- 末端评分：`outputs/v8_late_veto_ablation_20260919/module_metrics_exact.json`。
- 原始运行：`outputs/v8_ablation_<arm>_20260919`，每组含 frozen runtime、配置、manifest、API 记录、失败及逐题审计。
- `scripts/summarize_v8_exact_bootstrap.py` 只读取已保存评分，精确区间无需再调用 API。
- 所有实验所用脚本已保存到新盘相应输出的 `harness_snapshot/`；不改变 canonical production 源码。

## 实现与存储

- `scripts/run_v8_late_veto_ablation.py`：精确重建、四种末端干预、共享末端计算。
- `scripts/run_v8_pipeline_ablation_suite.py`：冻结 snapshot 内的提示词/表示/来源/投影对照。
- `scripts/run_v8_ledger_removal_ablation.py`：仅实验使用的无持久化 ledger 对照。
- `scripts/collect_v8_pipeline_ablations.py`：完整 episode 与数据身份核验、合并。
- `scripts/score_v8_module_ablations.py`：官方评分、精确 judge 输入缓存与 episode bootstrap。
- 当前实验相关测试 14 项通过；未声称全仓库测试重新通过。

账号 HOME 已在 `/mnt/data_disk_2/fuyali`，但本会话打开的项目实体仍在旧盘
`/mnt/data_disk/home/fuyali/codes/2027_TOIS_Gov-Mem`。旧盘 inode 逼近耗尽时暂停新集
调度，让正在执行的集结束后迁移本次输出到
`/mnt/data_disk_2/fuyali/govmem_ablation_outputs/`，原 outputs 路径保留软链接后恢复。
已经保存的 checkpoint 不重测，未出现磁盘类 execution_error，未删除历史数据。
