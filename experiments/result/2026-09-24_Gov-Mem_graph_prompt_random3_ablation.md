# Governed Slot Graph prompt advisory 消融（2026-09-24）

这是同一固定清单的严格配对消融：`experiments/manifests/v8_paired_random3_per_domain_seed20260918.json`，四领域各 3 个完整 episode，共 303 checkpoint。Graph-on 与 Graph-off 使用相同 runtime Python、相同 manifest、相同 12 集并发协议；唯一配置差异是 `memory_governed_slot_graph.enabled`。Graph-on 的 audit findings 写入 Stage-2 prompt，Graph-off 完全不调用 graph。Naive RAG 预测和官方 GPT-4o 标签按同一 manifest/hash 复用。评分 `gate_by_action=false`，无适用标签缺失。

## 主结果

| 版本 | Medical MGS | Office MGS | Education MGS | Household MGS | 四领域 avg.MGS | 执行错误 |
|---|---:|---:|---:|---:|---:|---:|
| Graph-on | 56.19% | 18.04% | 46.26% | 60.16% | **45.16%** | 22/303 |
| Graph-off | 50.57% | 19.74% | 48.29% | 53.47% | **43.02%** | 13/303 |
| Graph-on − off | +5.62 | −1.70 | −2.03 | +6.69 | **+2.14** | +9 |

Graph-on 的错误最坏情况 avg.MGS 为 38.04%，Graph-off 为 36.91%。按 episode 比较，Graph-on 胜 5/12、负 7/12；episode MGS 差值的平均为 +0.43 个百分点，显示正向平均差主要受少数 episode 影响，稳定性不足。Graph 不是回答证据来源；281 个 Graph-on 成功题的 Stage-2 prompt 收到 audit，`graph_selected=0`。Graph-off 没有 graph audit。

## 解释边界

这组结果支持“Graph prompt advisory 在该开发集上有小幅平均正向信号”，不支持“Graph 普遍提升”或“Graph 单独解释 Gov-Mem 相对 Naive RAG 的整体提升”。Graph-on 相对 Graph-off 的 +2.14 个百分点远小于 Gov-Mem 相对 Naive RAG 的差值，且 episode 间方向不一致。Graph-on 还多 9 个执行错误，说明审计 prompt 可能增加合同压力或模型不稳定性。所有 12 集和 303 题都已历史暴露，不能作为论文独立测试集。

运行目录：

- Graph-on：`outputs/v8_graph_prompt_random3_20260924`
- Graph-off：`outputs/v8_graph_prompt_random3_no_graph_20260924`
- 配对评分：`outputs/v8_graph_prompt_random3_scored_20260924/paired_metrics.json`、`outputs/v8_graph_prompt_random3_no_graph_scored_20260924/paired_metrics.json`

下一步应保留这个简单的 prompt advisory 设计，降低执行合同错误，再用独立场景族验证；不要添加针对开发题的 Python 语义过滤。
