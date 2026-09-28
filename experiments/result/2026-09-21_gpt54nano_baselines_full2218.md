# gpt-5.4-nano：四baseline全量实验与Overleaf表格

状态：全量矩阵已启动，PID3944479。实测40个episode子进程、40个不同key，启动时各方法10路，空位动态补任务。
输出：`outputs/baselines_gpt54nano_full2218_20260921`。
四方法A-Mem、Mem0 upstream、ReMem-I、ReMem-S各91集/2218query，总364集/8872query。

OpenLux gpt-5.4-nano；temperature0，输出上限4096，reasoning_effort服务商默认，与已完成nano RAG一致。
Embedding仍OpenLux text-embedding-3-small；最终记忆上限20，内部检索/迭代按各方法官方实现；judge仍OpenLux GPT-4o、temperature0、gate_by_action=false。
四方法各从空memory运行；不复用其他模型的预测/chat/memory。

为规避已确认的OpenLux nano Chat参数错误，transport内responses_bridge.py将nano请求映射为Responses：
保持原始messages/temperature/top_p/JSON格式，max_tokens转换为max_output_tokens；将Responses文本/usage封装回原客户端所需结构。
适配同时覆盖Mem0 upstream shim。未完成的Responses不伪装为成功文本。原始wire请求/响应及计费usage在HTTP日志和gzip中保留。
judge与embedding保持原接口。单元断言和真实官方router/Mem0 shim调用检查通过。
已确认四方法均有真实nano Responses HTTP200，包括Mem0内部抽取调用；存在少量连接失败，保留原始审计。

用户提供的PERSA文件实际20个唯一key；Gov-Mem现有配置40个唯一key，本次使用该40-key池，不复制/输出密钥。
冻结runtime/config/harness、模型profile、调度amendment和启动器快照均在输出目录。原已测结果不覆盖。

入口：scripts/run_nano_baseline_matrix.py；推理后自动官方评分，随后核验预测/判分覆盖及标签缺失，并导出：
- paper_tables/results.md：含已完成nano朴素RAG和四baseline宏平均及各领域U/A/F/MGS。
- paper_tables/results.tex：Overleaf可用LaTeX tabular，需booktabs。
- paper_tables/table_audit.json：来源数值及缺失标签审计。

目前表格尚未生成；不以其他模型或论文数值填补。预检2次nano请求在preflight目录，费用单列于其日志，矩阵costs扫描也包含它们。
如评分标签缺失，导出表注必须标明，不人工补标签。Gov-Mem nano尚未评测，不混入跨模型对照。
恢复前检查run_status、orchestrator_pid及进程，勿重复启动；完成episode不重跑。
