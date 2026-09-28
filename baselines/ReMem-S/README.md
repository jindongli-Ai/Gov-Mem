# ReMem-S

来源：GateMem官方 `bench/agents/remem.py` / `bench/remem/`。
variant=single；与ReMem-I使用相同记忆提取实现，查询采用单步语义检索，
不执行iterative工具选择。其余参数见method.yaml，官方fallback保留并记录。

入口：`python -B baselines/remem_s/run.py --output <新盘新目录> --stage all`。
模型与评分参数独立指定 `--model-profile baselines/models/<profile>.yaml`。
公共协议见[上级说明](../README.md)。
