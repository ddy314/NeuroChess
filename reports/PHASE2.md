# 第二阶段交付记录

准备结果：沿用原第 10000 步 checkpoint 的模型权重；新增引擎监督导入、policy/value/Q 联合训练、12.5% 左右人类策略回放、history-free 模式、验证最优权重保存与扩展看板。正式配置 3000 步。未启动云端实例或正式训练。

数据来自 Lichess eval database，实际接受 851213 个完整记录（842756 train / 8457 val）；请求 100 万，但源数据流在扫描 1464524 行后结束。保留有效记录，未虚报达到目标。源版本、过滤条件、目标定义、前缀哈希见 `data/evals-v1/manifest.json`。全量数据审计检查标签范围/归一化、padding、重复及精确跨分割泄漏：通过；0 重复，0 精确跨分割泄漏。前缀非全库均匀抽样，哈希留出也不保证相邻局面独立。

校验：全套 61 tests passed；含黑方分数符号、特殊着法、Q 掩码、历史特征屏蔽，以及两个 worker 的回放断点续训与连续训练模型权重逐项完全一致。

在本机 RTX 4060 使用已有 CUDA 环境加载真实 155M 模型，运行四步并恢复续训一步；未付费租用测试 GPU。BF16，batch32×accum2；这是流水线短验证，不是 T4 吞吐基准或棋力提升证据。真实 Q 输出层 weight/bias 均有非零更新，见 `phase2-q-update.json`。末步各损失和梯度有限，新增显存/梯度/数据指标落盘。T4 上正式配置使用自动 FP16；本轮未作远程验证。

扩展看板在真实 GPU 短验证日志上运行，Playwright 浏览器零控制台错误；提供 train/val 分量、Top-1、Q MAE、学习率、梯度、显存、样本数、回放、剩余时间估算、保存/同步时间和验证明细。旧日志无对应字段时显示未记录。截图位于 `output/playwright/phase2-dashboard.png`。

启动包：`output/phase2/neurochess-phase2.tar.gz`，约 677 MiB，包含初始权重、预处理分片、代码与逐文件 SHA256；远端训练前校验。`bash scripts/launch_phase2.sh` 创建 T4，Drive 授权由账户持有人操作。详细说明见 `docs/phase2.md`。每 500 步验证保存，每 1000 步轻量对局；最佳验证 loss 不等于最佳棋力。
