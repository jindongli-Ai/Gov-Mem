# 朴素 RAG：OpenLux gpt-4o-mini 全量实测

## 全量完成结果

91集/2218预测与判分记录覆盖核验通过；官方数据副本与统一数据集逐文件hash相同。适用judge标签无缺失，query执行错误0。

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


状态：已启动，尚无最终 performance。

- 官方 GateMem `rag_naive` runner，四个 domain 共91个完整 episode / 2218 query。
- Memory/answering base LLM：OpenLux `gpt-4o-mini`，temperature=0，max output=4096。
- Embedding：OpenLux `text-embedding-3-small`，Top-20。
- Judge：OpenLux `gpt-4o`，temperature=0，gate_by_action=false。
- 四个 domain 分片使用30个不同 API key：Medical/Office各8路，Education/Household各7路；启动后已核验四个 runner 子进程均存活。
- 采用官方朴素RAG协议，不使用 Gov-Mem 的治理账本、Stage-2或符号模块；仅保留官方 `rag_naive` 的检索与回答流程。

输出目录：`outputs/rag_naive_gpt4omini_full2218_20260921`。
入口：`scripts/run_gatemem_official_rag_naive_full.py`。
完成后自动生成 `official_score.json`，并将四个 domain 的 U/A/F/MGS 补入统一比较表。
当前所有推理、输出和临时文件均位于 `/mnt/data_disk_2`。
