# Mem0（Mem-0）

来源：GateMem官方 `bench/agents/mem0.py` 与 vendored `mem0_upstream/`。
按论文配置使用upstream Memory()，不静默替换成builtin后端。
官方owner buckets、记忆提取/更新/删除与跨bucket查询保持原样。

OpenLux需要provider映射、内部generation参数对齐与embedding URL适配，
全部仅发生在冻结副本，差异保存为compatibility.patch。官方add/search fail-soft
警告单独计数，不能把“无异常退出”当成记忆成功。

入口：`python -B baselines/mem0/run.py --output <新盘新目录> --stage all`。
模型与评分参数独立指定 `--model-profile baselines/models/<profile>.yaml`。
公共协议见[上级说明](../README.md)。
