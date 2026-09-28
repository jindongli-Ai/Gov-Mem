# V8 旧更新保留确认性实验与训练可行性（已完成）

## 评测前提修正

历史 V7 的四个 full2218 输出均包含 GateMem 全部 91 个 episode。因此不能将
现有 GateMem 剩余 episode 称为从未使用过的 pristine holdout。
对迁移后本地历史 V8 prediction/error/manifest 审计，发现 17 个已使用 episode。
本次用固定 seed 20260920、episode ID 的 SHA256 排序，从其余 episode 中每领域
选 3 个完整 episode，共 **306 checkpoints / arm**。选择不读取题型、答案或分数。
这是 **V8 未调优样本上的确认性对照**，仍存在历史 V7 暴露。

相关文件：
- `experiments/manifests/v8_prior_exposure_audit_20260919.json`
- `outputs/v8_update_retention_confirmation_20260919/manifest.json`
- 同目录 `protocol.json`、`run_order.json`、两个 frozen runtime/config。

## 唯一实验变量

Full 使用上一轮默认 V8 的精确字节级冻结源码，runtime SHA256：
`1346fd7a2492030d310809430553f9ce6ba9156ee87684d27539a82e71e1941c`。
retain_updates 仅替换 `shallow_lifecycle_view`，保留所有可见 lifecycle events；
权限、删除事件、来源校验、LLM、提示词、检索及 Stage 3 不变。
两组从空 event store 重跑完整轨迹、共享内容寻址 embedding cache、交错执行。

推理模型 Gemini 2.5 Flash-Lite、temperature 0。官方 judge GPT-4o，gate_by_action=false。
执行错误全部保留，同时计算失败按最坏情况处理的 MGS。
预算范围为两组各 12 个 episode，不增加模型搜索或 LoRA sweep。
没有同时重跑 RAG-Naive，本实验仅用于测试折叠机制，不能将旧开发集 baseline 分数
直接挪到新样本上宣称优于 baseline。

所有工作目录、输出、缓存和 TMPDIR 都位于 data_disk_2；子进程禁写 Python bytecode，
避免迁移前 Python 安装路径产生旧盘写入。没有覆盖/删除历史实验。

## 可靠性修复（不进入上述冻结对照）

旧 no_advisory_projection 开发运行的 15 个终止错误：

| 错误 | 数量 |
|---|---:|
| 缺少显式 bind 数组 | 4 |
| 缺少 value/delivery | 3 |
| 已标记 deleted 却请求放行 | 3 |
| 未知 candidate | 1 |
| claim 引文不精确 | 1 |
| 同一 claim 绑定不兼容的资源/动作/场景 | 1 |
| resource 引文不匹配 | 1 |
| JSON 解析失败 | 1 |

没有把“缺少 bind 的放行”默认为无约束，也没有自动忽略 deleted 标签。
只添加一个确定性的格式兼容：`block:{restriction_kind, restriction_evidence}`
转换成相同语义的平铺 delivery=block；矛盾的 KEEP/release 或平铺字段冲突仍报错，
原文来源匹配校验仍执行，不推断未知权限。

在同样 358 个已保存请求上回放，成功 288→289，仅恢复 1 个、0 个回退。
这是包括 repair 的 contract 诊断，不是新 MGS，不表示最终错误数已经下降。
新增嵌套拒绝测试与原 shallow 测试共 34 项通过；随后相关 V8 来源/状态/末端回归合计 **108 项通过**。

## A100 与 LoRA / MLP 决策

已确认 3 张 A100-SXM4-80GB。尚未占用 GPU、下载权重或训练。
最值得考虑的可训练任务是带原文引用的权限/生命周期事件抽取；保留语言审查和
符号时间/绑定否决。不能让 MLP 凭少量相似度特征替代自然语言授权判断。

已有 12 个开发 episode 仅提供 274 段不同的新历史窗口；同窗口的事件输出存在
12 处差异，不能直接把 teacher 输出当成可信标注。联合抽取看到 query，也有
查询相关性风险。模型格式通过不代表语义正确。

本轮 12 个确认性 episode 已明确保留，不进入任何后续训练/蒸馏。
真正独立的训练与测试需要按场景族/文档来源隔离的新增语料；先看零样本抽取质量，
再做 LoRA 对照，不为使用 GPU 添加模块。
详细说明见 `docs/GOVMEM_V8_LOCAL_TRAINING_DECISION.md`。

## 实验结果

两组完整推理和官方评分已完成，各 12 个完整 episode、306 个检查点；所有适用判分非空且 parse_ok，无样本删除。

| 实现 | 平均 MGS | 失败最坏情况 MGS | 执行错误 | Gemini calls / tokens |
|---|---:|---:|---:|---:|
| Full（隐藏旧普通更新） | 31.1834% | 26.3797% | 12 | 548 / 3,681,465 |
| retain_updates（全部保留） | 24.9023% | 19.1751% | 18 | 565 / 3,840,274 |

保留 − Full = **−6.2811 个百分点**；按领域分层的完整 episode 精确 bootstrap
95% 区间 **[−13.3875, +0.3195] 个百分点**（531,441 种有序组合）。区间跨零，
不能宣称确定地有害，但没有复现旧开发集 +6.92 点的正向收益，不能据此切换默认。

| 领域 | Full MGS | 保留旧更新 MGS |
|---|---:|---:|
| Medical | 51.10% | 34.84% |
| Office | 12.56% | 16.02% |
| Education | 28.09% | 20.16% |
| Household | 32.99% | 28.59% |

Medical 的删除泄露率从12%升到28%，隐私泄露率从35.71%升到46.43%；
Education 和 Household 的隐私泄露也增加。Office Utility 改善，但错误增多。
这与“保留所有旧状态一定更好”的解释不一致。不能把旧开发集46.98%当成通用提升。

### 本轮决定

**不将全部保留旧更新切成默认。** 保留先前同资源更新不同字段的通用反例：原折叠
条件仍然不充分，但全量保留也可能增加无关/过期状态和抽取负担。下一步应研究
“明确更新同一个字段，并且确实完整替代”这样的可审计关系，而不是按领域成绩
硬编码折叠规则或继续给当前测试样本调 prompt。

这个关系是否需要 LoRA，要由独立标注数据和零样本基线决定。当前没有可靠标签，
不启动训练；先保留 symbolic 时间/来源/绑定检查、完善通用语义反例和数据规范。
以后若根据本批结果调整实现，这批数据即成为开发数据，不再称 V8-unseen。

### 新费用与复现

- Gemini：1,113 次请求 / 7,521,739 tokens。
- Embedding：796 次请求 / 116,850 tokens（新样本，两组并发可能出现同键冷缓存重复计算，均计入费用）。
- GPT-4o judge：475 次请求 / 387,178 tokens；其余完全相同判分输入复用。
- HTTP 失败0，usage缺失0。没有训练、下载权重或新增 GPU 占用。

完整指标：`experiments/result/2026-09-19_v8_update_retention_confirmation_metrics.json`。
覆盖/费用/错误审计：`experiments/result/2026-09-19_v8_update_retention_confirmation_audit.json`。

生产源码仅新增了前述嵌套拒绝格式兼容，未做新的端到端性能测试；冻结确认性组不含此修复。
历史31.18%/24.90%及40.06%均属于各自冻结实现，不能声称已测得当前含修复源码的MGS。

