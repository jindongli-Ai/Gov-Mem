# 统一OpenLux全量比较（持续更新）

每方法91个完整episode / 2218 query；memory与回答base LLM为Gemini2.5FlashLite，temperature=0；官方GPT-4o judge，gate_by_action=false。

| 方法 | U ↑ | A ↓ | F ↓ | MGS ↑ | 状态 |
|---|---:|---:|---:|---:|---|
| A-Mem | 58.85% | 47.90% | 59.00% | 12.52% | 完整判分 |
| Mem0 | 33.43% | 46.48% | 47.23% | 8.83% | 完整判分 |
| ReMem-I | 48.25% | 42.29% | 53.70% | 12.50% | 缺1个适用判分标签，按官方可用标签汇总 |
| ReMem-S | 38.77% | 34.90% | 40.36% | 14.27% | 完整判分 |
| Gov-Mem V8 | 66.78% | 36.45% | 28.35% | 30.02% | 缺1个适用判分标签，按官方可用标签汇总 |

表中各指标均为四领域宏平均；MGS取领域MGS的均值，不将宏平均U/A/F再次相乘。A为隐私泄露率，F为删除泄露率。
只有一套官方MGS；缺失标签不人工补造，未完成实验不填写估计值。
四个baseline最终记忆条数上限为20，内部检索/迭代保留各自官方机制。Gov-Mem含治理账本与额外证据，不能声称所有系统输入信息量或计算预算相同。
本表是同服务商/模型/数据/评测设置下的方法比较，不是等token或等计算预算比较。所有数据历史已暴露，不称为holdout。
警告及执行错误保留在各实验原始审计，不另定义performance。

## 分领域结果

| 方法 | 领域 | U ↑ | A ↓ | F ↓ | MGS ↑ |
|---|---|---:|---:|---:|---:|
| A-Mem | medical | 79.52% | 53.12% | 36.72% | 23.59% |
| A-Mem | office | 71.43% | 62.57% | 63.06% | 9.87% |
| A-Mem | education | 33.89% | 41.11% | 66.11% | 6.76% |
| A-Mem | household | 50.54% | 34.78% | 70.11% | 9.85% |
| Mem0 | medical | 42.38% | 54.69% | 61.02% | 7.49% |
| Mem0 | office | 39.61% | 60.23% | 46.40% | 8.44% |
| Mem0 | education | 29.44% | 40.56% | 49.44% | 8.85% |
| Mem0 | household | 22.28% | 30.43% | 32.07% | 10.53% |
| ReMem-I | medical | 62.38% | 54.17% | 55.93% | 12.60% |
| ReMem-I | office | 67.53% | 46.78% | 47.75% | 18.78% |
| ReMem-I | education | 26.11% | 30.73% | 62.22% | 6.83% |
| ReMem-I | household | 36.96% | 37.50% | 48.91% | 11.80% |
| ReMem-S | medical | 69.05% | 44.79% | 38.42% | 23.48% |
| ReMem-S | office | 46.10% | 42.69% | 38.29% | 16.31% |
| ReMem-S | education | 11.11% | 23.33% | 47.22% | 4.50% |
| ReMem-S | household | 28.80% | 28.80% | 37.50% | 12.82% |
| Gov-Mem V8 | medical | 79.05% | 31.41% | 24.86% | 40.74% |
| Gov-Mem V8 | office | 75.32% | 55.56% | 33.33% | 22.32% |
| Gov-Mem V8 | education | 60.56% | 30.56% | 38.89% | 25.70% |
| Gov-Mem V8 | household | 52.17% | 28.26% | 16.30% | 31.33% |

## 数据来源

- A-Mem: `outputs/a_mem_gemini25flashlite_full2218_20260919/official_metrics.json`
- Mem0: `outputs/mem0_gemini25flashlite_full2218_20260920/official_metrics.json`
- ReMem-I: `outputs/remem_i_gemini25flashlite_full2218_20260920_retry/official_metrics.json`
- ReMem-S: `outputs/remem_s_gemini25flashlite_full2218_20260920/official_metrics.json`
- Gov-Mem V8: `outputs/v8_full2218_20260919/full_metrics_with_missing_label.json`

## gpt-4o-mini 四baseline全量结果

| 方法 | U ↑ | A ↓ | F ↓ | MGS ↑ |
|---|---:|---:|---:|---:|
| A-Mem | 26.18% | 40.71% | 21.60% | 10.53% |
| Mem0 | 19.71% | 44.62% | 15.76% | 6.66% |
| ReMem-I | 15.22% | 38.56% | 23.47% | 5.80% |
| ReMem-S | 16.33% | 39.42% | 23.02% | 6.42% |

完整分领域表见 [gpt-4o-mini报告](2026-09-21_OpenLux_gpt4omini_baselines_full2218.md)。各方法2218query，判分标签无缺失；Gov-Mem尚无本轮gpt-4o-mini结果，不跨模型直接作同条件比较。

## gpt-4o-mini 朴素RAG全量结果

| Domain | U ↑ | A ↓ | F ↓ | MGS ↑ |
|---|---:|---:|---:|---:|
| Medical | 51.43% | 57.81% | 21.47% | 17.04% |
| Office | 32.47% | 61.40% | 22.52% | 9.71% |
| Education | 11.11% | 26.67% | 28.89% | 5.79% |
| Household | 19.57% | 19.57% | 17.93% | 12.91% |
| **Macro average** | **28.64%** | **41.36%** | **22.70%** | **11.36%** |

来源：`outputs/rag_naive_gpt4omini_full2218_20260921/official_score.json`。各domain runner退出码均0。
本次使用原官方runner直接运行，未使用四baseline矩阵的冻结/HTTP审计包装；保留配置、日志、预测和judge记录，不声称具有同等原始HTTP审计覆盖。
以下为启动历史。


## gpt-5.4-nano 朴素RAG（OpenLux Responses）

| Domain | U ↑ | A ↓ | F ↓ | MGS ↑ |
|---|---:|---:|---:|---:|
| Medical | 69.52% | 64.06% | 37.29% | 15.67% |
| Office | 62.34% | 39.77% | 31.08% | 25.88% |
| Education | 15.56% | 17.22% | 41.67% | 7.51% |
| Household | 30.98% | 18.48% | 23.91% | 19.22% |
| **Macro average** | **44.60%** | **34.88%** | **33.49%** | **17.07%** |

全2218query，适用评分标签无缺失；此模型结果独立展示，不混入其他base LLM行。
