# ReMem-S：OpenLux Gemini 2.5 Flash-Lite 全量实测

状态更新：2026-09-20 22:05:43（北京时间）全部推理与官方评分完成，PID2023154已退出。
91集/2218预测/2218判分记录覆盖、预测hash及适用标签均核验通过，无缺失适用标签。

| 领域 | U ↑ | A ↓ | F ↓ | MGS ↑ |
|---|---:|---:|---:|---:|
| medical | 69.05% | 44.79% | 38.42% | 23.48% |
| office | 46.10% | 42.69% | 38.29% | 16.31% |
| education | 11.11% | 23.33% | 47.22% | 4.50% |
| household | 28.80% | 28.80% | 37.50% | 12.82% |
| 宏平均 | 38.77% | 34.90% | 40.36% | 14.27% |

正式MGS为14.2734073763%。执行错误15条，memory警告775条，仅保留审计。
Gemini请求42101次/16633996 tokens；embedding134207次/3201689 tokens；judge2218次/1944059 tokens。
HTTP失败分别62/28/0次；失败usage未知，实际金额以账单为准。
下文为启动历史；统一比较表已更新。

状态：已启动全量推理，PID2023154；已从实际进程环境核实30个episode进程使用30个不同API key。
输出：`outputs/remem_s_gemini25flashlite_full2218_20260920`。暂无最终performance。

范围：GateMem全部91个完整episode、2218query，Medical579、Office547、Education540、Household552。
采用GateMem官方ReMem single variant，retrieval_top_k=10、linking_top_k=5、standard回答协议；
保持原算法和提示词。Memory/answering均使用OpenLux Gemini2.5FlashLite，temperature=0、max output=4096。
Embedding为OpenLux text-embedding-3-small；judge为OpenLux GPT-4o、temperature=0、gate_by_action=false。
与已有baseline保持统一设置，不称为论文temperature=0.2设置的逐项复刻。

已验证并导入原矩阵首个Medical完整pilot（28query），不重跑；其余90集从空memory运行。
pilot的1条query执行错误、24条memory警告与原始响应/费用记录均原样保留，仅作审计。
新集使用独立精确embedding缓存，pilot保留原共享缓存来源；不共享LLM生成或memory状态。
算法、配置、数据、脚本快照及调度记录冻结在输出；新盘TMPDIR与输出真实路径已核验。

入口：`scripts/run_baseline_full_from_completed_pilot.py --method remem_s --workers 30`。
沿用ReMem-I的目录大小写兼容：历史pilot harness路径不与当前重命名目录比对，
仍验证历史runtime/config/manifest、pilot checkpoint集合及预测hash，以及新输出完整冻结配置。

推理结束自动官方评分，生成`official_metrics.json`，按四领域与宏平均报告唯一一套U/A/F/MGS。
最终报告前核验2218预测/判分覆盖、适用judge标签是否齐全；不以parse_ok代替标签完整性。
任何评分缺口应明确记录，不补造标签、不删失败样本。警告保留审计，不额外定义performance。
启动或恢复前检查run_status、orchestrator_pid和实际进程，勿重复运行付费实验。
