# Gov-Mem V8 Slot Graph 四领域开发诊断（2026-09-24）

## 范围与结论

这是一组固定的、历史已暴露的完整 episode：Medical 28、Office 32、Education 18、Household 24，共 102 个 checkpoint。它只用于定位 Gov-Mem 的权限治理、效用和执行可靠性问题，不能作为独立 holdout 或全量论文成绩。比较主指标为四领域原始 MGS 的算术平均。U/A/F 为原因分析；MGS = U × (1−A) × (1−F)。

当前 4/20 图候选预算快照的 **avg.MGS 34.07%**，同题冻结 Naive RAG 为 **18.20%**，差 **+15.87 个百分点**。四集仅各领域一集，模型和 judge 波动未估计，不能推断总体提升。Gov-Mem 有 6/102 个 query execution errors；RAG 对照为 0。错误在原始预测中保留，另列失败最坏情况 MGS，不以成功子集替代官方成绩。

| 系统 / 领域 | U ↑ | A ↓ | F ↓ | MGS ↑ | 执行错误 |
|---|---:|---:|---:|---:|---:|
| Gov-Mem Medical | 70.00% | 44.44% | 33.33% | 25.93% | 1/28 |
| Gov-Mem Office | 66.67% | 60.00% | 0.00% | 26.67% | 4/32 |
| Gov-Mem Education | 100.00% | 16.67% | 33.33% | 55.56% | 0/18 |
| Gov-Mem Household | 37.50% | 25.00% | 0.00% | 28.13% | 1/24 |
| **Gov-Mem avg.MGS** | — | — | — | **34.07%** | **6/102** |
| Naive RAG Medical | 70.00% | 33.33% | 44.44% | 25.93% | 0/28 |
| Naive RAG Office | 66.67% | 60.00% | 76.92% | 6.15% | 0/32 |
| Naive RAG Education | 66.67% | 50.00% | 83.33% | 5.56% | 0/18 |
| Naive RAG Household | 62.50% | 25.00% | 25.00% | 35.16% | 0/24 |
| **Naive RAG avg.MGS** | — | — | — | **18.20%** | **0/102** |

Gov-Mem 的四领域平均失败最坏情况 MGS 为 **31.25%**（每条执行错误按 utility 失败、privacy/deletion 泄露处理）；这是可靠性补充，不替换官方 MGS。Household 的 U 从 62.50% 降到 37.50%，当前机制在该领域有明显效用风险。

## 实际机制与错误

本快照固定 dense Top-K=20，最多 4 个图候选替换 dense 候选；dense 命中不足时图可填剩余位置。成功检索的 96 个 checkpoint 共保留 dense 候选 1,548 条、图候选 372 条。图抽取成功逻辑调用 134 次，Office 有 1 次图解析失败后回退 dense-only。图候选提供原文证据，不直接授权；权限与删除的否决仍取决于来源、语义绑定和确定性 symbolic critic。图未解析到角色、关系或权限表示未知，不构成允许或拒绝。

6 个 query errors 均为终止 ValueError：Office 2 个缺 bind、1 个 deleted claim grounding、1 个缺 value/delivery；Medical 1 个权限事件 resource grounding；Household 1 个不兼容的多事件绑定。它们不能被记作普通无害噪声，也不能从 MGS 分母中删除。本轮未观察到终止网络错误。推理 telemetry 中记录 Gemini 311 次、embedding 197 次；新 Gov-Mem 评分记录 GPT-4o 102 次。

### 推理调用与 token（仅已报告 usage）

| 模型 | HTTP 请求 | reported tokens | 终止 HTTP 失败 |
|---|---:|---:|---:|
| Gemini 2.5 Flash-Lite | 311 | 1,299,984 | 0 |
| text-embedding-3-small | 197 | 27,623 | 0 |
| GPT-4o judge（本次新 Gov-Mem 评分） | 102 | 82,780 | 0 |

推理调用包含图抽取和主流程，不能由本表单独识别图的增量成本；judge 请求在独立评分目录的 telemetry 中，不计入推理成本。正式论文需在相同缓存和同一输入下测无图、无符号层消融与 API tokens、延迟。

## 对照与版本边界

- Gov-Mem 推理：`outputs/v8_slot_graph_rag_majority_20260924`，runtime SHA256 `493c352a3428c2f42e225d22689191a56e9636a9badeb73787f6b836d2507c64`；配置与运行源码在该目录的快照内。
- 官方评分和 102 题 Naive RAG 子集：`outputs/v8_slot_graph_rag_majority_vs_rag_diagnostic_20260924`；`protocol.json` 保存 manifest、Gov-Mem runtime、baseline 原预测与判分 SHA256。`diagnostic_metrics.json` 为机器可读分数。
- Naive RAG 是更早同一固定随机清单中保存的官方实现预测和 GPT-4o 判分，按相同 checkpoint ID 原样选出；没有重新运行 baseline、没有按结果择题或重判其预测。两边 base model 均为 Gemini 2.5 Flash-Lite、temperature 0、Top-K 20；judge 为 GPT-4o、temperature 0、`gate_by_action=false`。
- 旧 fix2 图版本同题 avg.MGS 25.07%，不是当前 4/20 版本；两次 Gov-Mem 生成独立运行且有模型波动，不能把 9.00 点差异归因于图预算。旧结果在 `outputs/v8_slot_graph_fix2_vs_rag_diagnostic_20260924`，失败记录和评分原样保留。
- 报告是在 4/20 快照完成后写的；后续工作区安全校验改动未进入该冻结 runtime。不能把此成绩作为后续代码的实测结果。

## 后续决策

继续优先解决通用合同错误和 Household 效用损失，避免按这 102 题添加案例词表或领域特判。下一次扩大样本必须固定实现与协议，按完整 episode 运行，报告 U/A/F/MGS、错误、图解析覆盖和成本。全量 91 集/2218 题前先确认主流程的错误率可控；第一篇论文只讨论记忆权限治理相对于强 Naive RAG 的净 MGS，图与 symbolic/neuro-symbolic 是辅助机制，不声称通用一阶逻辑证明器。
