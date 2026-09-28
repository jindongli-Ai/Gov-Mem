# Baseline 全量实验接口

每个方法一个目录：`a_mem/`、`mem0/`、`remem_i/`、`remem_s/`。
每个目录有 `method.yaml`（算法参数）、`run.py`（单方法入口）、README。
模型/provider参数独立位于 `models/`；调度与官方评分共用 `run.py`，
HTTP费用与精确embedding缓存共用 `transport.py`。不复制四套调度代码。

本次四方法都使用 OpenLux 的 Gemini 2.5 Flash-Lite 处理记忆与回答，
temperature=0、max output=4096；OpenLux text-embedding-3-small、Top-20；
OpenLux GPT-4o官方judge，temperature=0、gate_by_action=false。
这是与Gov-Mem统一服务商/模型参数的对照，不是论文temperature=0.2的逐项复刻。
保留官方代理的prompt、证据访问和算法；不能声称这些输入与Gov-Mem完全相同。

## 运行

从项目根目录运行，所有output必须在data_disk_2：

```bash
PYTHONDONTWRITEBYTECODE=1 python -B baselines/run.py \
  --output outputs/baselines_gemini25flashlite_full2218_20260919 \
  --model-profile baselines/models/openlux_gemini25flashlite.yaml \
  --stage prepare
```

`--stage pilot --workers 4` 为每方法运行同一个完整Medical episode，结果计入全量。
联调核对后 `--stage all --workers 16` 接续剩余完整episodes，随后自动官方评分。
`--stage score` 只评分；`--stage summarize` 无API调用，只汇总已有完整结果。
每方法91集/2218query，四方法共364集/8872query。

单方法可用：`python -B baselines/mem0/run.py --output <新目录> --stage all`。
也可在统一入口使用 `--methods a_mem remem_i`。同一输出目录的方法集合和模型配置
冻结后不能改变；切换模型使用新profile与新output，不覆盖原实验。

## 冻结、适配与恢复

运行前复制 GateMem 官方 `bench/` 和vendored Mem0到输出 `runtime/`，保存逐文件hash、
数据hash、配置hash、`compatibility.patch`和harness快照。原third_party源码不修改。

只有以下接口适配：

- Mem0 upstream将openlux映射到其OpenAI-compatible客户端，模型仍为所选base LLM。
- Mem0 upstream embedding URL补齐`/v1`；其他官方embedding接口自行追加`/v1`。
- Mem0内部硬编码temperature=0.1/max_tokens=2000改为使用所选模型profile。
- 官方episode runner将终止异常保存为`output.action=error`并继续后续checkpoint，
  不伪装成refuse/no_memory，不删除样本。失败ingest不推进已处理turn位置；
  后续checkpoint会从尚未成功ingest的位置继续，这种影响链保留在日志。

官方内部的heuristic fallback和Mem0 fail-soft保留，不将其改成Gov-Mem规则；
在逐集warning audit中单独报告。主表只使用官方U/A/F/MGS，错误与用量另列。
judge缺失适用标签会标注，不人工补标签、不另造一套正式MGS。

已完成episode用哈希核对过的完成标记跳过。若进程异常退出留下不完整episode，
调度停止等待检查，不盲目调用官方resume：官方resume会重建记忆，可能重复付费或
改变后续状态。排他锁防止同一矩阵重复启动。

## 存储与审计

- `protocol.json` / `model.yaml` / `configs/<method>.yaml`：模型、算法和来源身份。
- `runs/<method>/<domain>/<episode>/`：预测、原始chat响应（gzip）、HTTP用量、warning。
- `scored/<method>/<domain>/official_eval/`：官方判分与summary。
- `official_metrics.json`：每方法四领域U/A/F/MGS及宏平均、错误、用量。
- `run_status.json` / `orchestrator.log`：运行状态。开始或恢复前先检查进程，勿重复运行。

仅缓存完全相同URL+payload的embedding响应；不缓存/共享LLM回答或事件状态。
缓存命中不冒充新增API费用。HTTP日志不记录密钥；原始chat保存在非公开outputs。
Mem0补充依赖pytz安装在 `/mnt/data_disk_2/fuyali/govmem_caches/baseline_python_deps`，
通过子进程PYTHONPATH使用，不修改旧盘Python安装；Python bytecode禁写。

## 后续更换 base LLM

复制模型profile，改 `llm_model` 及明确需要的生成参数，仍使用OpenLux。
例如建立 `models/openlux_<model>.yaml`，以新的输出目录运行相同方法集合。
judge和embedding默认固定，避免把base LLM差异与评分/检索模型变化混在一起。
新模型若需要特殊协议参数，应先完成完整episode联调并记录兼容改动，不能静默
删除不支持的参数或换模型。模型返回的别名会被HTTP audit记录，但无法独立认证
服务商背后的真实权重，因此不用分数高低推断论文模型“掺水”。
