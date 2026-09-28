# Mem0：OpenLux Gemini 2.5 Flash-Lite 全量实测

状态：2026-09-20 02:26:30（北京时间）全量推理和官方评分完成，原PID4075324已退出。
91集、2218预测、2218判分记录齐全；逐集预测hash/checkpoint身份及评分覆盖已核验，适用标签无缺失。
输出 `outputs/mem0_gemini25flashlite_full2218_20260920`。

| 领域 | Query | U ↑ | A ↓ | F ↓ | MGS ↑ |
|---|---:|---:|---:|---:|---:|
| Medical | 579 | 42.38% | 54.69% | 61.02% | 7.49% |
| Office | 547 | 39.61% | 60.23% | 46.40% | 8.44% |
| Education | 540 | 29.44% | 40.56% | 49.44% | 8.85% |
| Household | 552 | 22.28% | 30.43% | 32.07% | 10.53% |
| 四领域宏平均 | 2218 | 33.43% | 46.48% | 47.23% | 8.83% |

唯一正式宏平均MGS为8.8272354064%，按领域MGS宏平均。A/F是隐私/删除泄露率。
全量query执行错误0；memory写入失败警告12条，均来自保留的pilot，不能将0 query错误解释为所有记忆写入成功。
Gemini请求40802次、返回usage累计69840300 tokens、HTTP失败17次；
embedding请求32302次、504470 tokens、HTTP失败0；GPT-4o判分2218次、1966146 tokens、HTTP失败0。
失败请求无usage，实际费用以OpenLux账单为准。机器可读结果为输出下 `official_metrics.json`。
论文应注明GateMem官方适配Mem0 upstream、统一OpenLux、temperature=0以及上述失败记录。
以下保留固定协议和启动历史。

范围为GateMem全部91个完整episode、2218query（Medical579、Office547、Education540、Household552）。
实现为GateMem官方适配的Mem0 upstream，message_window=10、top_s=5、max_facts=20。
记忆与回答均为OpenLux Gemini2.5FlashLite、temperature=0、max output=4096；
embedding为OpenLux text-embedding-3-small；judge为OpenLux GPT-4o、temperature=0、gate_by_action=false。
模型设置与A-Mem/Gov-Mem统一；论文profile的temperature=0.2不作为本次设置。

复用原矩阵已完成的首个Medical pilot，源码/config/数据/预测hash和28个checkpoint身份已核验。
pilot有0条query执行错误、12条memory写入失败警告；14次HTTP失败，最近50次HTTP均成功。
成功的Gemini请求396次、embedding450次；Gemini generation参数均为temperature0、max_tokens4096。
上述失败和费用随pilot完整导入，不因网络失败重跑或挑选结果；其余90集从空memory执行。
Mem0内部add/search fail-soft属于官方实现，单独审计，不能把0条query错误视为记忆写入全部成功。

共用冻结方法实现与传输层保持原样，原third_party源码未修改。
入口 `scripts/run_baseline_full_from_completed_pilot.py`，支持独立方法、完成pilot导入及20路并发；
输出保存 `full_runner_snapshot.py`、`scheduling_provenance.json`、`pilot_import.json`和原始请求/响应。
新集使用独立精确embedding缓存；导入pilot保留旧矩阵缓存来源，不共享LLM回答或memory状态。
导入pilot费用不能与原矩阵重复相加。所有写入、缓存和临时文件位于data_disk_2。

全量推理完成后自动官方评分，最终 `official_metrics.json` 报告四领域及宏平均U/A/F/MGS。
只有这一套正式MGS，A/F为泄露率；执行错误、记忆失败警告、HTTP失败、缺失标签和用量另列。
在覆盖核验与评分完成前，不用中途结果代替全量实测。数据历史已暴露，不称为pristine holdout。

启动前离线验证：pilot原样导入及幂等跳过、20个不同worker同时调度通过；无额外模型调用。
恢复前检查 `run_status.json`、`orchestrator_pid.json` 和实际进程，避免重复启动；
不完整episode禁止直接重跑，需检查持久化结果及状态。
