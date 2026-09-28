# Gov-Mem Graph prompt advisory：12 集开发评测（2026-09-24）

预先固定的 `experiments/manifests/v8_paired_random3_per_domain_seed20260918.json`：四领域各 3 个完整 episode，共 303 checkpoint。该数据历史上已暴露，不是独立 holdout。当前 Graph 只将来源审计候选写入 Stage-2 prompt，由 LLM 判断权限、时间和 scope；不插入 RAG 证据，不执行新的 Python 语义 veto。现有来源校验、绑定和 symbolic critic 保留。

推理冻结目录 `outputs/v8_graph_prompt_random3_20260924`，runtime SHA256 `6697a88360df89427aba8676dd1bf1287c36e84a5a9af2f852dda56a564c98db`。12 个不同 OpenLux key 同时运行 12 集，集内按历史顺序。评分目录 `outputs/v8_graph_prompt_random3_scored_20260924`；Naive RAG 预测和官方 GPT-4o 标签按同一 manifest/hash 复用，没有重跑。Judge `gate_by_action=false`，适用标签无缺失。

| 系统 / 领域 | U ↑ | A ↓ | F ↓ | MGS ↑ | 执行错误 |
|---|---:|---:|---:|---:|---:|
| Gov-Mem Medical | 86.21% | 25.93% | 12.00% | 56.19% | 4/81 |
| Gov-Mem Office | 70.37% | 60.00% | 35.90% | 18.04% | 8/96 |
| Gov-Mem Education | 77.78% | 23.53% | 22.22% | 46.26% | 5/54 |
| Gov-Mem Household | 75.00% | 12.50% | 8.33% | 60.16% | 5/72 |
| **Gov-Mem avg.MGS** | — | — | — | **45.16%** | **22/303** |
| Naive RAG Medical | 79.31% | 40.74% | 52.00% | 22.56% | 0/81 |
| Naive RAG Office | 74.07% | 63.33% | 64.10% | 9.75% | 0/96 |
| Naive RAG Education | 27.78% | 50.00% | 66.67% | 4.63% | 0/54 |
| Naive RAG Household | 75.00% | 29.17% | 20.83% | 42.06% | 0/72 |
| **Naive RAG avg.MGS** | — | — | — | **19.75%** | **0/303** |

探索性差值 +25.41 个百分点。Gov-Mem 执行失败按最坏情况计入 U/A/F 时 avg.MGS 为 38.04%，仍高于同题 RAG，但不代替官方主指标。281 个成功题的 Stage-2 prompt 均包含 graph audit，`graph_selected=0`；图经提示词发挥作用，未改 RAG 证据集合。这仍不能把总体收益单独归因于图，需同版本关闭 Graph prompt 的消融。

22 条终止错误均为 ValueError：joint extraction validation 4、缺 value/delivery 4、缺 bind 5、block 缺来源/范围 4，其余 unknown bind 1、prefill validation 1、claim grounding 1、无 grounded claim 1、KEEP unknown candidate 1。无终止网络错误。推理 telemetry：Gemini 965 请求、5,276,261 reported tokens；embedding 572 请求、71,259 reported tokens；HTTP 失败 0。错误原样保留，没有挑选成功样本。旧 baseline `complete.json` 含迁移前绝对路径，评分脚本新增只读路径回退，在同 episode 当前目录读取原预测并核验所有 checkpoint；旧快照未修改。

结论限于已暴露开发集：整个 Gov-Mem framework 相对强 Naive RAG 有正向 MGS 信号，但 Graph 的独立净贡献尚未证明。优先做同版本 Graph prompt 开关对照，并降低合同错误；不要再加领域词表或 Python 语义过滤。
