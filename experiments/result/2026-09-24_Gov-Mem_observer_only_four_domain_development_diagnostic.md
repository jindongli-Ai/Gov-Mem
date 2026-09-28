# Gov-Mem observer-only graph 四领域开发诊断（2026-09-24）

同一固定清单四领域各一个完整 episode，共 102 个历史已暴露 checkpoint。Graph 只旁观审计：不进入 Stage-2 证据、不改变 dense Top-20、不直接授权或拒绝。四个独立 API key 并行、集内顺序执行。推理冻结在 `outputs/v8_slot_graph_observer_current_20260924`，runtime SHA256 `a56658de22475f32fef7b54449b9d62cf438e07992fb121a71681f2f51604b1d`。官方评分在 `outputs/v8_slot_graph_observer_current_vs_rag_diagnostic_20260924`，同题 Naive RAG 原预测和标签按哈希复用，无 baseline 重跑。GPT-4o judge，`gate_by_action=false`。

| 系统 / 领域 | U ↑ | A ↓ | F ↓ | MGS ↑ | 执行错误 |
|---|---:|---:|---:|---:|---:|
| Gov-Mem Medical | 90.00% | 33.33% | 33.33% | 40.00% | 0/28 |
| Gov-Mem Office | 66.67% | 90.00% | 69.23% | 2.05% | 0/32 |
| Gov-Mem Education | 66.67% | 16.67% | 16.67% | 46.30% | 0/18 |
| Gov-Mem Household | 75.00% | 25.00% | 12.50% | 49.22% | 1/24 |
| **Gov-Mem avg.MGS** | — | — | — | **34.39%** | **1/102** |
| Naive RAG Medical | 70.00% | 33.33% | 44.44% | 25.93% | 0/28 |
| Naive RAG Office | 66.67% | 60.00% | 76.92% | 6.15% | 0/32 |
| Naive RAG Education | 66.67% | 50.00% | 83.33% | 5.56% | 0/18 |
| Naive RAG Household | 62.50% | 25.00% | 25.00% | 35.16% | 0/24 |
| **Naive RAG avg.MGS** | — | — | — | **18.20%** | **0/102** |

探索性差值为 +16.19 个百分点。101 个成功 checkpoint 的 debug audit 均记录 `graph_selected=0`；Graph 不参与回答证据。一个 Household 执行错误保留原样。Graph finding 是对累积历史逐 checkpoint 重复呈现，不能把总出现次数当作独立问题数。Graph 审计目前只记录明显的候选警示，不做自动 veto；因此本次 MGS 提升不能归因于 Graph 本身，主要是在 observer 边界下主 Gov-Mem pipeline 的一次实测。Office A=90% 是必须后续处理的权限泄露风险。四集已暴露且模型/judge 波动未估计，不能作为论文总体性能声明，也不能把与其他独立运行的差值解释为纯 graph 因果效应。

复测前相关回归测试 78 passed。下一步应做同版本 Graph 开/关对照，确认旁观图对主流程的输入、输出和成本的实际影响；再审核 Office 隐私错误，避免按已见题写特判。
