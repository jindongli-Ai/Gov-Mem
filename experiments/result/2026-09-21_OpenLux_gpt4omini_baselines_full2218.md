# OpenLux gpt-4o-mini：四个 GateMem baseline 全量实测

## 完成结果（2026-09-21 08:07:44 北京时间）

四方法各91集/2218预测与判分，合计8872条。逐集预测hash、checkpoint集合、judge覆盖及适用标签核验通过，无缺失标签。

| 方法 | U ↑ | A ↓ | F ↓ | MGS ↑ |
|---|---:|---:|---:|---:|
| A-Mem | 26.18% | 40.71% | 21.60% | 10.53% |
| Mem0 | 19.71% | 44.62% | 15.76% | 6.66% |
| ReMem-I | 15.22% | 38.56% | 23.47% | 5.80% |
| ReMem-S | 16.33% | 39.42% | 23.02% | 6.42% |

以上为四领域宏平均，MGS为领域MGS的均值。模型OpenLux gpt-4o-mini，temperature0；judge固定gpt-4o。
执行错误A-Mem5、Mem0 2、ReMem-I 3、ReMem-S27，全部保留。ReMem-S包含此前索引越界异常，属于当前实现运行结果，不宣称无实现故障。

## 分领域结果

| 方法 | 领域 | U ↑ | A ↓ | F ↓ | MGS ↑ |
|---|---|---:|---:|---:|---:|
| A-Mem | medical | 47.62% | 56.25% | 16.95% | 17.30% |
| A-Mem | office | 32.47% | 61.99% | 24.32% | 9.34% |
| A-Mem | education | 7.78% | 27.22% | 26.67% | 4.15% |
| A-Mem | household | 16.85% | 17.39% | 18.48% | 11.35% |
| Mem0 | medical | 29.05% | 65.62% | 24.29% | 7.56% |
| Mem0 | office | 38.31% | 66.08% | 13.96% | 11.18% |
| Mem0 | education | 3.89% | 27.22% | 14.44% | 2.42% |
| Mem0 | household | 7.61% | 19.57% | 10.33% | 5.49% |
| ReMem-I | medical | 33.81% | 61.46% | 22.60% | 10.09% |
| ReMem-I | office | 15.58% | 43.86% | 31.08% | 6.03% |
| ReMem-I | education | 3.89% | 25.56% | 23.89% | 2.20% |
| ReMem-I | household | 7.61% | 23.37% | 16.30% | 4.88% |
| ReMem-S | medical | 37.62% | 57.81% | 19.77% | 12.73% |
| ReMem-S | office | 16.23% | 50.29% | 31.53% | 5.53% |
| ReMem-S | education | 2.78% | 30.56% | 26.11% | 1.43% |
| ReMem-S | household | 8.70% | 19.02% | 14.67% | 6.01% |

机器可读结果：`outputs/baselines_gpt4omini_full2218_20260921_retry/official_metrics.json`。
本轮已结束，勿重复运行。以下保留启动历史。


状态：已启动四方法矩阵，使用 `gpt-4o-mini` 作为 memory/answering base LLM。
输出：`outputs/baselines_gpt4omini_full2218_20260921_retry`；进程PID416179。
启动后已核实实际存在30个transport子进程、30个不同API key；四个方法各从空memory开始，
不复用Gemini实验的pilot、LLM响应或memory状态。暂无performance结果。

范围与设置：A-Mem、Mem0、ReMem-I、ReMem-S各91个完整episode/2218 query；
OpenLux `gpt-4o-mini`、temperature=0、max output=4096；embedding为OpenLux
`text-embedding-3-small`；官方judge为OpenLux `gpt-4o`、temperature=0、gate_by_action=false。
所有方法最终回答上下文上限为20，内部检索/迭代保留官方方法配置。

本轮使用独立模型profile `baselines/models/openlux_gpt4omini.yaml`、独立输出和embedding cache，
不会覆盖 `gemini-2.5-flash-lite` 的四个已完成结果。统一调度器的旧16路硬上限已修正为按实际key数，
新矩阵的30路并发已实际核验。一次早期启动因调度器缩进bug在发出模型请求前失败，未产生实验数据；
正式retry为PID416179，冻结协议以retry输出为准。

完成后自动生成 `official_metrics.json`，并补入统一比较表。论文应将这轮称为统一OpenLux/gpt-4o-mini
条件下的自测结果；与Gemini结果分开报告，不把不同base LLM的分数混成同一行。
