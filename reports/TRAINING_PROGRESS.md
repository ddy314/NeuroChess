# 训练进展与评测记录

本页整理截至 2026-10-31 的训练记录。步数以各阶段记录为准：人类预训练完成后，引擎监督以权重初始化开始单独计数。

## 人类预训练与第二阶段

- 人类棋谱阶段完成 10,000 步。该检查点另在 Stockfish 18 固定强、弱两档各进行 32 局直接走法及 32 局 256 次 MCTS 测试；[直接模式汇总](evidence/training-progress/human-10000-policy-summary.json)、[搜索模式汇总](evidence/training-progress/human-10000-mcts256-summary.json)和[配对比较](evidence/training-progress/human-10000-paired-comparison.json)保留协议与结果，逐局 PGN 分别见[直接](evidence/training-progress/human-10000-policy-64-games.pgn)和[搜索](evidence/training-progress/human-10000-mcts256-64-games.pgn)。
- 引擎监督先运行至第 1,500 步，随后把引擎训练集从 842,756 扩到 5,000,000 局面，以完整检查点保留优化器状态接续。
- 扩容阶段完成 6,000 次更新，引擎监督累计 **7,500 步**。固定 8,457 局面验证集的 loss 从扩容开始时的 2.5629 降至 2.4389，策略 top-1 从 42.19% 升至 44.00%。完整汇总见 [第二阶段记录](evidence/training-progress/phase2-summary.json)。

## 第三阶段：curriculum-v3

从累计第 7,500 步的完整检查点保留优化器状态接续。Stage A 完成 4,000 次更新；Stage B 将核心引擎数据、困难局面和人类策略回放按 70/20/10 混合，分四波再完成 4,000 次更新。最终引擎监督累计 **15,500 步**。困难局面是对既有局面的重新标注，不能声称是新局面。[课程设计](../docs/CURRICULUM_V3.md)记录标注预算及具体协议。

固定验证集最终 loss 为 **2.4178**，top-1 为 **44.19%**；曲线只保留从原始云端日志提取的部分区间，中间 11,500–14,500 步记录未取回，见[提取记录及限制](evidence/training-progress/curriculum-v3-validation-extract.json)。不能据此证明模型已达到容量上限或声称独立泛化提升。

云端完成记录显示，最终评测采用 Stockfish 18、1,000 nodes、32 个换色开局对，每种模式 64 局；直接走法 **1 胜 9 和 54 负**，512 次 PUCT **29 胜 14 和 21 负**，无截断。该记录报告棋谱合法性、结果核验和持久化校验通过；[完成摘要](evidence/training-progress/curriculum-v3-completion.json)、[冻结开局](evidence/training-progress/curriculum-v3-eval-openings.fen)和[协议](evidence/training-progress/curriculum-v3-evaluation-freeze.json)保留在此。云端完整 PGN 与检查点仍在私有存储，不包含在本仓库中。

选中模型 `selected.pt` 的 SHA-256 为 `e4ed8978db91eeef8360c4dea87faf7a4f826325bce45aa7efc024692d585ef7`；本地副本已重新计算哈希一致。该权重不随仓库分发。

## 额外本地对局

对上述选中模型另做了 512 次 PUCT 对 Stockfish 18 不同节点预算的测试，每档 16 个开局换色，共 32 局：

| Stockfish nodes | 胜／和／负 | 得分率 |
|---:|---:|---:|
| 500 | 23／7／2 | 82.81% |
| 1,000 | 16／6／10 | 59.38% |
| 1,500 | 10／9／13 | 45.31% |

这是与特定引擎预算的比较，样本量小，不是绝对 Elo。[逐局 PGN](evidence/training-progress/curriculum-v3-local-96-games.pgn)和[结果及区间](evidence/training-progress/curriculum-v3-local-96-results.json)已收录。

## 后续 fresh-v4

截至 2026-10-31，该实验在准备新数据：选定 500,000 个训练、10,000 个验证和 2,000 个测试局面；独立检查记录未发现与既有局面键的交集或新数据跨集合棋局重叠，但无法确认旧预训练来源的棋局完全互不重叠。[选择审计](evidence/training-progress/fresh-v4-selection-audit.json)和[实验设计](../docs/FRESH_V4.md)说明边界。教师标注和最终质量门槛仍在进行；两组训练及最终对局不应视为已完成。
