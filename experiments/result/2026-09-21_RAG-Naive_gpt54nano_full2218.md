## Responses接口全量完成结果

91集/2218条预测及判分记录覆盖核验通过，冻结runtime hash未变；适用标签无缺失，query执行错误0。四domain runner退出码均0。

| Domain | U ↑ | A ↓ | F ↓ | MGS ↑ |
|---|---:|---:|---:|---:|
| Medical | 69.52% | 64.06% | 37.29% | 15.67% |
| Office | 62.34% | 39.77% | 31.08% | 25.88% |
| Education | 15.56% | 17.22% | 41.67% | 7.51% |
| Household | 30.98% | 18.48% | 23.91% | 19.22% |
| **Macro average** | **44.60%** | **34.88%** | **33.49%** | **17.07%** |

正式输出：`outputs/rag_naive_gpt54nano_responses_full2218_20260921/official_score.json`。
模型为OpenLux gpt-5.4-nano，Responses接口，temperature0，输出上限4096，reasoning_effort默认；GPT-4o官方judge。
旧Chat失败尝试保留，未混入本次完整成绩。以下为启动历史。

> 更新：初次Chat尝试被OpenLux max_tokens协议错误中断，仅11条预测，无正式performance。即使发送max_completion_tokens仍收到max_tokens错误。旧日志及结果保留。
> 新运行outputs/rag_naive_gpt54nano_responses_full2218_20260921使用OpenLux Responses接口，真实router预检成功；temperature0、4096输出上限、模型/算法不变，judge仍Chat GPT-4o。
> 新冻结副本保存compatibility.patch，修正key override以保持30个不同key；从空状态重建，旧尝试费用单独保留。

# 朴素RAG：OpenLux gpt-5.4-nano 全量实测

状态：2026-09-21启动，PID3238627；暂无最终performance。
输出：`outputs/rag_naive_gpt54nano_full2218_20260921`。

官方rag_naive，91个完整episode/2218 query，Top-20、turn chunk；不使用Gov-Mem治理/重排模块。
OpenLux gpt-5.4-nano，temperature=0、max_tokens=4096；embedding固定text-embedding-3-small，judge固定OpenLux gpt-4o、temperature=0、gate_by_action=false。
reasoning_effort未显式设置，使用服务商默认值。接口预检HTTP200、JSON有效，原始请求/响应见api_preflight.json；不据模型字段独立认证服务商权重。

实测四个runner存活，各自key分片8/8/7/7，合计30个不同key及30个episode worker，已进入incremental ingest/query。
runtime、配置和启动器独立冻结，protocol.json保留hash；逐文件核验数据副本与统一数据集一致。
启动器仅重定向冻结路径；官方算法未改。汇总脚本修复了历史硬编码gpt-4o-mini模型标签，现从实际配置读取模型和温度。

完成后自动生成official_score.json。报告前需核验2218预测及判分覆盖、适用标签非空；仅使用一套官方U/A/F/MGS。
本轮使用官方runner，不具备矩阵transport包装的逐请求原始HTTP审计；保存官方预测、judge和usage日志，不声称完整账单费用。
所有输出/临时文件在data_disk_2；运行期间不要重启或覆盖，恢复需先检查进程与已有预测。
