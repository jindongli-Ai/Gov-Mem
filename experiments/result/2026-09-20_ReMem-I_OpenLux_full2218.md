# ReMem-I：OpenLux Gemini 2.5 Flash-Lite 全量实测

状态：2026-09-20 15:45:58（北京时间）全量推理与官方评分请求结束，PID176574已退出。
91集/2218条预测及2218条判分记录齐全；逐集预测hash、checkpoint身份与判分覆盖核验通过。
注意：Education有1个适用privacy_leak标签为null，当前是官方可用标签汇总，不能声称标签完整。

| 领域 | U ↑ | A ↓ | F ↓ | MGS ↑ |
|---|---:|---:|---:|---:|
| Medical | 62.38% | 54.17% | 55.93% | 12.60% |
| Office | 67.53% | 46.78% | 47.75% | 18.78% |
| Education（缺1个隐私标签） | 26.11% | 30.73% | 62.22% | 6.83% |
| Household | 36.96% | 37.50% | 48.91% | 11.80% |
| 四领域宏平均 | 48.25% | 42.29% | 53.70% | 12.50% |

官方可用标签宏平均MGS为12.5028512651%。未人工补标签、未替换判分。
缺失checkpoint：education_episode_custom_en_006_harbor_review_harbor_leaders_dual_track_ckpt_11。
该judge响应parse_ok=true，但privacy_leak=null，因此评分记录齐全不等于适用标签齐全。
query执行错误27条，均保留；memory warning/fallback共854条，仅作审计。
Gemini请求49766次、返回usage累计31578047 tokens、HTTP失败82；embedding请求130837次、3125096 tokens、失败15；
GPT-4o请求2219次、1966475 tokens、失败1。失败请求无usage，实际费用以账单为准。
机器可读结果为输出目录下official_metrics.json。以下保留启动历史和协议。
输出：`outputs/remem_i_gemini25flashlite_full2218_20260920_retry`。

范围为GateMem全部91个完整episode、2218query；ReMem-I采用官方iterative variant：最多5个工具步骤、
检索Top-10、链接Top-5、标准answer protocol。Memory与answering统一使用OpenLux
`gemini-2.5-flash-lite`、temperature=0、max output=4096；embedding为OpenLux
`text-embedding-3-small`；judge为OpenLux GPT-4o、temperature=0、gate_by_action=false。

已有首个Medical pilot的28条预测保留并在启动前校验；pilot中有2条query执行错误和22条官方fallback/warning，
不重跑、不删除。其余90集从空memory运行。错误和fallback只保留审计，最终主表使用全量真实官方评分。

启动时发现历史四方法协议保存了旧的小写目录路径，而当前源码目录规范化为`ReMem-I/`；
本次只跳过无法对应当前目录的历史harness路径校验，仍校验pilot runtime、manifest、method config及prediction hash，
不改ReMem-I算法、prompt、模型、传输或评分口径。该兼容说明和脚本快照保存在输出目录。

推理完成后自动评分并生成`official_metrics.json`，报告唯一一套官方U/A/F/MGS及错误、fallback、HTTP和用量。
完成前不使用中途数值作为论文结果。
