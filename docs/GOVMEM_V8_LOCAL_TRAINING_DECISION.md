# V8：本地训练是否值得做（2026-09-19）

用户已允许必要时使用 A100 80GB 训练 LoRA / MLP。当前可见三张 A100 80GB，
检查时其中两张基本空闲、一张占用约 36 GB。没有启动训练或占用其他任务显存。

## 当前决定

先完成“是否隐藏旧更新”的固定模型对照。此时训练/替换模型会同时改变两个因素，
无法知道收益来自哪个模块。生产 memory/answering 继续 Gemini 2.5 Flash-Lite。
有 GPU 不代表需要新增神经网络，也不改变必须保留 symbolic reasoning 的约束。

优先考虑的小模型任务是**抽取权限和生命周期事件**：输入可见原文、来源 ID、
身份/资源注册表，输出 allow/deny/revoke/delete/update 等带精确引文的事件。
它只学习把语言变成结构化事件，来源校验、时间状态计算和末端明确否决仍由程序负责。
不让小模型直接决定是否可披露某个自然语言范围。

MLP 暂不采用：现在没有稳定的数值特征或可信标签，授权依赖文本语义、条件、时间和
例外；压成几项打分很容易恢复以前“用词表和相似度判断权限”的问题。
MLP 以后可研究非安全关键的成本/延迟估计，但不直接作为自动放行开关。

## 已有数据为什么不能直接开训

离线审计仅使用原 V8 开发样本，不读取本轮确认性样本的答案/轨迹：

- 12 个相关开发 episode，289 次联合抽取请求，274 段不同的新历史窗口。
- 285 个响应能提供 events 列表，合计 504 个事件提议；102 个 events 为空，4 个响应解析失败。
- 12 个相同 episode/新历史窗口出现不同的 events 输出。它可能来自 repair、上下文变化或
  模型不稳定，不等于 12 个已证实错误，但不能当作无噪声 teacher gold。
- 504 个事件提议中 415 个是 update（82.3%）；delete 42、deny 22、allow 16、revoke 5、supersede 3、require_permission 1。直接训练会偏向多数类，少数权限变化缺乏覆盖。
- 联合抽取还看到了当前 query，不能假设抽取完全不依赖问题。
- 引文匹配/schema 通过，仅证明格式与来源，不证明事件作用域、时间、删除含义和漏抽正确。
- 不能按 checkpoint 随机切训练/验证：同一 episode 重复的历史会泄露到两边。

审计文件：`experiments/result/2026-09-19_v8_training_readiness.json`。
原始请求不是训练语料导出，这次没有把它们送入任何训练任务。

## 合理的 LoRA 试验步骤

1. 用与 GateMem 无关的、来源授权明确的文档/对话或独立生成的场景族，构造抽取训练任务。
   覆盖新增权限、撤销、限时授权、否定、条件、删除、部分字段更新，以及“相同资源不同字段”。
   标注作用域与精确 source span；包括无事件的负例，不能只采会触发规则的句子。
2. 在生成 teacher 标注之前按场景族/文档来源切 train/validation/test，近重复和改名版本
   必须在同一分组。训练过程不读取 GateMem 的 gold、未来轮次、judge 标签或测试回答。
3. 在固定的 7B–8B 本地 instruction 模型上比较零样本与 LoRA，记录许可、checkpoint
   revision、tokenizer、训练数据 hash、超参数与 seed；具体模型等独立数据完成后再选。
4. 先测 event precision/recall、来源跨度、resource/subject binding、条件/时间、删除误报漏报
   和格式失败，不能只测 JSON 合法率。再接入冻结 V8 做完整 episode 对照。
5. 保持最终语言审查和符号否决不变，报告 MGS、U/A/F、错误、额外本地延迟、显存与 API tokens。
   当前抽取和 Stage 2 合并调用，所以本地抽取不会自动消掉一次 API：它也可能只是增加一步。
   必须实测缩短 prompt 是否足够抵消新增本地计算和潜在错误。

## 数据隔离与确认性评测

历史 V7 已经跑过 GateMem 全部 91 个 episode，现有数据不能声称 pristine holdout。
本次排除 17 个已有 V8 使用记录的 episode，固定四领域各 3 个新 episode、306 checkpoints。
定义是 V8-unseen confirmation，仍有历史 V7 暴露，不是完全独立的论文测试集。

`outputs/v8_update_retention_confirmation_20260919/manifest.json` 的所有 episode
从任何未来训练和 teacher 蒸馏中排除。真正独立的训练与论文测试需另建来源/场景族。
若看过本次错误后继续优化，下一轮只能叫开发评测，不能继续声称该批数据未参与调优。

所有代码、数据、模型权重、HF/Torch 缓存、checkpoint 和任务临时文件都必须放在
`/mnt/data_disk_2`。本轮没有下载模型、创建训练权重或启动 GPU 任务。
