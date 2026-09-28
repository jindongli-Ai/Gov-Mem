# A-Mem

来源：GateMem官方 `bench/agents/a_mem.py`，benchmark-adapted A-Mem。
使用LLM元数据抽取、语义链接、一跳图扩展与官方rerank；参数见method.yaml。
不使用仅heuristic的廉价替代。官方遇LLM/解析问题时的fallback保持原样并记录。

入口：`python -B baselines/a_mem/run.py --output <新盘新目录> --stage all`。
模型与评分参数独立指定 `--model-profile baselines/models/<profile>.yaml`。
公共协议见[上级说明](../README.md)。
