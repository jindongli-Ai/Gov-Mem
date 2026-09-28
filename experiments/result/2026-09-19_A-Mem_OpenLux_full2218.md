# A-Mem：OpenLux Gemini 2.5 Flash-Lite 全量实测

状态：2026-09-20 00:04:06（北京时间）全量推理和官方评分完成。
91个episode、2218条预测及2218条judge记录齐全，适用评分标签无缺失。
已逐集核验预测hash及checkpoint身份/覆盖。

| 领域 | Query | U ↑ | A ↓ | F ↓ | MGS ↑ |
|---|---:|---:|---:|---:|---:|
| Medical | 579 | 79.52% | 53.12% | 36.72% | 23.59% |
| Office | 547 | 71.43% | 62.57% | 63.06% | 9.87% |
| Education | 540 | 33.89% | 41.11% | 66.11% | 6.76% |
| Household | 552 | 50.54% | 34.78% | 70.11% | 9.85% |
| 四领域宏平均 | 2218 | 58.85% | 47.90% | 59.00% | 12.52% |

唯一正式宏平均MGS为12.5195882859%。A/F分别为隐私/删除泄露率。
执行错误13条（Medical2、Office1、Education4、Household6），保留在全量评分中；
记忆抽取失败/heuristic回退警告共52条。HTTP失败59次，不能和query错误或警告直接相加。
Gemini请求22173次、返回usage累计9496569 tokens；embedding请求21849次、1907821 tokens；
GPT-4o判分请求2218次、1950936 tokens、HTTP失败0。失败请求缺少usage，实际金额以账单为准。
机器可读结果：`outputs/a_mem_gemini25flashlite_full2218_20260919/official_metrics.json`。
论文应标为统一OpenLux设置下的GateMem官方适配A-Mem实测，并披露上述错误/回退及temperature=0。
以下保留运行过程与实验协议。

调度更新：用户要求从8个key增加到20个key。新调度进程3514363已接管，
实测同时存在20个A-Mem episode进程、20个不同key；原8个episode PID全部保留。
仅停止/替换无算法状态的父调度进程3081557，其子进程继续摄入/回答；未重跑任何episode。
记录在 `scheduling_20keys.json`、`scheduler20_pid.json`、`orchestrator_20keys.log`，
脚本为 `scripts/continue_a_mem_20keys.py`，冻结副本为 `continuation_20keys_snapshot.py`。
被接管的子进程退出后，核对官方终态summary和完整checkpoint集合才写完成标记；
其原父进程退出码不可获取，该完成证据区别在完成标记中明确记录。
既有模型、配置、传输、预测及评分口径未变。下文8并发/PID3081557为初始启动记录。

## 固定实验设置

- GateMem 全部 91 个完整 episode、2218 query：Medical 579、Office 547、Education 540、Household 552。
- 实现为 GateMem 官方 benchmark-adapted A-Mem，`a_mem_metadata_mode=llm`；保留官方链接、图扩展、rerank 与提示词。
- Memory 与 answering：OpenLux `gemini-2.5-flash-lite`，temperature=0，max output=4096。
- Embedding：OpenLux `text-embedding-3-small`，Top-20；judge：OpenLux `gpt-4o`，temperature=0，gate_by_action=false。
- 统一服务商和模型参数与 Gov-Mem 对比；GateMem 论文配置 temperature=0.2，因此不声称逐项复刻论文设置。
- 数据历史上已被评测，不称为未见 holdout。

## 调度与复现

独立输出：`outputs/a_mem_gemini25flashlite_full2218_20260919`。
全量进程启动 PID：3081557；实时状态以输出内 `run_status.json`、进程及日志为准。
新增 90 集以 8 个 episode 并发运行，集内按官方 runner 执行。
原矩阵仍拥有首个 Medical pilot，完成后验证源码、配置、数据与预测 hash，原样导入；
不重新运行该集，不复制运行中的 memory 状态。最终含该集共 91 集。
其余三个方法未扩大全量，只保留已启动的联调。

入口：`scripts/run_a_mem_full_from_pilot.py`，脚本快照保存在输出根目录；
算法与共用 harness 在 runtime/harness_snapshot 中冻结，调度说明见 scheduling_provenance.json。
新集使用独立 embedding cache；导入 pilot 保留原矩阵共享精确 embedding cache 的来源记录。
首个 pilot 的原始响应、用量与警告一起导入，因此本输出中的费用统计包含该集，不能再与原矩阵重复相加。

在扩大队列前，A-Mem 已有 4 条有效预测、0 条 query error，最近 50 次HTTP请求均成功。
同时 OpenLux 仍出现间歇连接问题；没有改变 DNS、重试、超时或算法来隐藏失败。
这不能保证全量期间连接稳定，所有执行错误和内部 metadata fallback 必须单列。

## 最终结果验收

全量推理完成后自动执行四领域官方评分，生成 `official_metrics.json`。
正式表只使用一套官方 U/A/F/MGS：MGS=U×(1−A)×(1−F)，
总 MGS 是四领域 MGS 的宏平均。A/F 是泄露率，越低越好。
额外报告 query 执行错误、metadata heuristic fallback、HTTP失败、缺失适用 judge 标签及实际返回 tokens。
如果存在缺失标签，须披露缺口；不得补造标签、删除失败样本或另选更高的一套 MGS。
该实现沿用官方内部 fallback，因此只看正常退出或完整 query 数不足以说明所有记忆抽取成功。

完整覆盖、判分缺口和错误审计完成前，不把中途结果或论文数字填入实测表。
