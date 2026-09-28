# 四种官方baseline：统一OpenLux全量评测

状态：准备完成，四方法各一个完整Medical episode正在进行接口联调，结果计入全量。暂无performance结果。

用户授权A-Mem、Mem0、ReMem-I、ReMem-S各自运行91个完整episode/2218query。
所有memory/answering都使用OpenLux Gemini 2.5 Flash-Lite，temperature=0、输出上限4096，
与当前Gov-Mem模型设置对齐。Embedding为OpenLux text-embedding-3-small；judge为
OpenLux GPT-4o、temperature=0、gate_by_action=false。最终主表只报告官方U/A/F/MGS，
执行错误、memory内部fallback与费用另列。

## 实现与复用

`baselines/a_mem`、`baselines/mem0`、`baselines/remem_i`、`baselines/remem_s`
分别包含method.yaml/run.py/README；模型profile在`baselines/models/`。
共用`baselines/run.py`调度、官方评分，`baselines/transport.py`审计请求和缓存embedding。
后续替换base LLM只建立新profile和新output，不复制方法代码。

GateMem官方bench及vendored Mem0冻结在本次output/runtime，逐文件hash与patch留档。
保持所有方法的算法与prompt；Mem0 upstream接口需要OpenLux provider映射、embedding URL
补齐/v1、内部temperature/max_tokens对齐。官方runner额外保存异常query为action=error，
不把失败当拒绝、不删除样本。官方内部soft-error/heuristic fallback保持原样并审计。

不声称这是GateMem论文的逐项原样复现：公开paper profile使用temperature=0.2，
本次统一temperature=0是为与Gov-Mem作同服务商模型对照。方法输入/记忆访问方式仍不同。
分数差异不能单独证明论文使用了“掺水”模型；服务商返回model字段可记录但无法独立认证权重。

## 全量与计费边界

每方法Medical579/Office547/Education540/Household552，共2218。
四方法共8872条query，需分别摄入可见历史。原数据含20293个turn，
A-Mem每turn需元数据提取，ReMem每turn需gist/fact提取，成本和时长会明显高于简单RAG。
只共享完全相同请求的embedding缓存，不共享LLM生成、预测或memory状态。
四个联调episode包含在全量任务中，成功后直接跳过，不重复收费。

## 产物

- 输出根 `outputs/baselines_gemini25flashlite_full2218_20260919`。
- 协议/快照 `protocol.json`、`compatibility.patch`、`runtime/`、`harness_snapshot/`。
- 状态 `run_status.json`、`orchestrator.log`、`orchestrator_pid.json`，恢复前检查存活进程。
- 逐集 `runs/<method>/<domain>/<episode>/`：预测、gzip原始chat、HTTP费用、warning审计。
- 评分 `scored/<method>/<domain>/official_eval`；完整指标 `official_metrics.json`。

准备检查：4项编排回归通过；Mem0 offline reset、真实provider映射和generation参数检查通过。
本报告将在实测结束后补充完整U/A/F/MGS，不能用pilot或论文数字充当全量实测。

## 联调连接检查（2026-09-19 22:00 左右）

原pilot进程3014421仍存活，未重复启动、未更改冻结实现。阶段性HTTP记录：
A-Mem 122次中121次HTTP200，Mem0 15次中11次HTTP200，ReMem-I 4次均SSL失败，
ReMem-S 84次中82次HTTP200。这是检查时快照，不是最终错误率或performance。
Mem0已记录真实add失败，ReMem存在终止query错误，不能把进程存活当作记忆正常。

无密钥、不调用模型的独立TLS探测：api.openlux.ai解析到15.204.105.133、
15.204.105.134、15.204.105.135、15.204.110.74；其中.133和110.74握手成功，
.134和.135在8秒超时。单次探测不能证明持续故障或归因所有API失败，
但需要在扩大全量队列前复核网络与真实memory写入。未修改DNS、证书验证或模型参数。
