# Gov-Mem Graph prompt advisory 四领域开发诊断（2026-09-24）

同一固定清单，四领域各一个完整 episode，共 102 个历史已暴露 checkpoint。Graph audit 进入 Stage-2 prompt，LLM 判断问题、请求者、来源、时间和 scope；Graph 不加入 RAG 证据，也不执行 Python 语义 veto。原有 source grounding 和 symbolic critic 保留。四个不同 key 并行，集内按历史顺序。推理冻结在 `outputs/v8_slot_graph_prompt_advisory_20260924`，runtime SHA256 `6697a88360df89427aba8676dd1bf1287c36e84a5a9af2f852dda56a564c98db`。同题官方评分在 `outputs/v8_slot_graph_prompt_advisory_vs_rag_diagnostic_20260924`；Naive RAG 原预测和 judge 标签按 checkpoint ID 与 hash 复用，未重跑 baseline。GPT-4o judge，`gate_by_action=false`。

| 系统 / 领域 | U ↑ | A ↓ | F ↓ | MGS ↑ | 执行错误 |
|---|---:|---:|---:|---:|---:|
| Gov-Mem Medical | 90.00% | 33.33% | 11.11% | 53.33% | 0/28 |
| Gov-Mem Office | 66.67% | 60.00% | 7.69% | 24.62% | 2/32 |
| Gov-Mem Education | 100.00% | 50.00% | 33.33% | 33.33% | 0/18 |
| Gov-Mem Household | 37.50% | 37.50% | 0.00% | 23.44% | 1/24 |
| **Gov-Mem avg.MGS** | — | — | — | **33.68%** | **3/102** |
| **同题 Naive RAG avg.MGS** | — | — | — | **18.20%** | **0/102** |

探索性差值为 +15.48 个百分点。99 个成功 checkpoint 的 Stage-2 prompt 都包含 `governed_slot_graph_audit`，且 `graph_selected=0`；这证明 Graph 经 prompt 参与推理而没有改变 RAG 证据集合。Graph finding 是候选审计信号，不自动授权或否决，且同一历史 finding 会在多个 checkpoint 重复出现。三个终止执行错误按原样保留，无终止网络错误。

与先前 observer-only 独立运行（avg.MGS 34.39%）相比，本次 Office F 从 69.23% 降到 7.69%，但 Household U 从 75% 降到 37.5%。两次生成和 judge 都独立，不能把差异精确归因于 Graph advisory；Household 的效用损失需要优先检查，避免 graph 审计被 LLM 误作普遍拒答。样本已暴露且每领域只有一集，不能作总体或论文性能声明。下一步固定该版本做 Graph prompt 开关对照，并统计 token 成本、findings 命中与过度拒答。
