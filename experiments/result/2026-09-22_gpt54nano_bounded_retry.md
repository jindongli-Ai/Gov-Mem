# gpt-5.4-nano 一次有界重试

用户授权：第一遍完成后，执行错误的case重跑一次；第二次仍失败就保留失败。
已部署等待进程3199911，输出outputs/baselines_gpt54nano_bounded_retry_20260922。
此刻仍等待首轮全矩阵推理和评分结束，尚未启动第二遍付费推理。

预先固定规则：四方法都以首轮output.action=error选完整episode，不按judge成绩筛选；
从空memory重建选中的整集，第二遍整集结果统一替换第一遍对应集，无论好坏，不逐题择优，不第三次重试。
精确embedding cache可复用，chat和memory状态不复用。没有执行错误的episode及其判分原样保留。
第二遍若进程异常退出导致覆盖不全则停止并报告，不能偷换回首轮或自动再跑。

ReMem-I/S根因：NodeStore在embedding成功前写入nodes/id_list/lexical，失败后ID与向量行数不一致；
外层ReMemIndex也提前写入nodes，导致后续重复写入被跳过。修复将两层节点登记移动到成功写入之后。
仅修改第二遍冻结runtime的bench/remem/store.py与retriever.py，保存差异和新hash；原实验不改。
测试tests/test_nano_retry_store.py的2项测试通过：复现原outage->IndexError，修复后状态一致，失败节点可重试且无重复；
正常写入的向量及检索结果与原版一致。提示词/模型/温度/检索宽度不变。

第一遍结束后保存retry_manifest.json，含全部选中及复用episode。40个不同key并行，评分仅新增选中episode，
未变预测按hash核验后复用原judge记录。最终table_audit.json审计缺失标签；results.md与results.tex注明一次有界重试及修复。
retry_error_audit.json报告每集两遍错误数。费用需区分首轮总成本与第二遍新增；汇总目录含复用记录，不可直接与首轮相加。
不删除错误，不因表格用于论文而声称没有故障。缺失适用标签须在最终表注标明。
