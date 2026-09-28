# Gov-Mem 合同修复四领域开发复测（2026-09-24）

同一固定清单、四领域各一个完整 episode、共 102 checkpoint。全部是历史已暴露的开发题，不是独立测试集。本轮冻结推理位于 `outputs/v8_slot_graph_contract_repair_20260924`，runtime SHA256 为 `3cbd81ac6bf1f7ed1132c091d1701f75c96032eb0e9157d61f9e377afcd373da`。评分与同题已冻结 Naive RAG 预测及标签位于 `outputs/v8_slot_graph_contract_repair_vs_rag_diagnostic_20260924`。四集由四个不同 OpenLux key 并行，每集内部按历史顺序运行；没有重跑 baseline。官方 GPT-4o judge，`gate_by_action=false`。首次评分命令误将诊断汇总目录作为 baseline 源，在任何 judge 请求前失败；随后从 `outputs/v8_paired_random3_seed20260918_scored` 读取哈希核验一致的原始 baseline 预测和标签，完成评分。

| 系统 / 领域 | U ↑ | A ↓ | F ↓ | MGS ↑ | 执行错误 |
|---|---:|---:|---:|---:|---:|
| Gov-Mem Medical | 60.00% | 22.22% | 66.67% | 15.56% | 2/28 |
| Gov-Mem Office | 33.33% | 90.00% | 30.77% | 2.31% | 2/32 |
| Gov-Mem Education | 66.67% | 66.67% | 50.00% | 11.11% | 0/18 |
| Gov-Mem Household | 37.50% | 25.00% | 0.00% | 28.13% | 2/24 |
| **Gov-Mem avg.MGS** | — | — | — | **14.27%** | **6/102** |
| Naive RAG Medical | 70.00% | 33.33% | 44.44% | 25.93% | 0/28 |
| Naive RAG Office | 66.67% | 60.00% | 76.92% | 6.15% | 0/32 |
| Naive RAG Education | 66.67% | 50.00% | 83.33% | 5.56% | 0/18 |
| Naive RAG Household | 62.50% | 25.00% | 25.00% | 35.16% | 0/24 |
| **Naive RAG avg.MGS** | — | — | — | **18.20%** | **0/102** |

本次 Gov-Mem 比同题 Naive RAG **低 3.92 个百分点**，且比上一次 Gov-Mem 冻结运行的 34.07% 低 19.79 个百分点。两次 Gov-Mem 运行的 manifest 相同，配置仅输出缓存路径不同，runtime 差异是两处合同处理；仍有生成与 judge 波动，不能把差异视作精确因果效应。执行错误数未下降。六个终止错误为 `resource_not_grounded` 两题、`deleted_release_not_grounded` 一题、缺 `value/delivery` 一题、无 grounded claim 一题、claim 未精确落地一题。无终止网络错误。失败按原预测保留，没有筛除。

这次负结果说明新增的五类专门修复提示没有可见收益。实验完成后先删去提示，随后按用户反馈将本轮剩余的严格缺失 `bind` 改动也撤回。当前 `src/gov_mem` 与 `run_govmem.py` 的所有 Python 文件均与上次 34.07% 运行的冻结快照逐字节一致；本次负结果的运行快照仍原样保留。撤回后相关测试为 88 passed、1 failed：`test_shallow_valid_delta_is_durable_before_claim_contract_repair[missing_bind]` 要求修复轮，但旧冻结实现为 `keep=true` 补空 `bind`，测试与冻结行为不一致。不能把历史 34.07% 自动归给新的随机运行，也不能把本次下降全部因果归于代码修改。下一步应优先简化通用合同与诊断路径，保持来源和删除校验，避免针对这 102 题做领域特判。此样本不能作为论文性能结论。
