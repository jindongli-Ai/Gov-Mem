# ReMem-I

来源：GateMem官方 `bench/agents/remem.py` / `bench/remem/`。
variant=iterative，最多5个工具步骤；记忆提取与工具选择使用所选base LLM。
检索宽度10、图链接5、回答Top-20上限，采用官方standard回答协议。
原实现提取/工具解析失败后的heuristic fallback保留并记录。

入口：`python -B baselines/remem_i/run.py --output <新盘新目录> --stage all`。
模型与评分参数独立指定 `--model-profile baselines/models/<profile>.yaml`。
公共协议见[上级说明](../README.md)。
