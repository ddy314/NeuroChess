# 本地 CP 与候选排序蒸馏

`scripts/distill_rank_frontier.py` 从本地 PGN 按棋局固定训练、验证、评测分组，
用现有小模型生成搜索边界，再由本地 Stockfish 的同一次 MultiPV 搜索标注父局面和最多四个孩子。
原始 CP 单独保存；将杀另存，不能把将杀当作普通 CP 参与回归。
完整走子历史和搜索路径交给老师，原始 EP 字段保留。

数据位于 `train.npy`、`val.npy`，每行是父局面与孩子组成的一组。孩子的 CP、WDL
都从孩子的行棋方视角表示，因此训练候选排序时需要取负号。
不同训练/验证组的已标注局面键重叠会被删除，最终评测棋局的已走前缀键也从训练/验证删除。
`scripts/audit_rank_frontier.py` 检查这些条件，并抽样重建历史与合法孩子。
这只能保证新小模型的数据隔离，不能证明既有大模型的原始训练来源与评测棋局独立。

`scripts/train_rank_cheap.py` 使用本地 CUDA 从头训练同尺寸网络：

- 主头继续学习既有 WDL 标量，二进制推理格式及大模型校正尺度保持一致。
- 辅助 CP 头学习 `clip(raw_cp, -3000, 3000) / 600` 的 Huber 损失。
- 排序直接监督主头的未压缩 logits，使用同一父局面下候选 CP 差的软概率目标。
  小于 30 cp 的差距和将杀标签不参与排序。CP 原始标签不因训练裁剪而丢弃。

CP 辅助头只在训练时使用，导出的 CPU 搜索模型仍使用 WDL 主头。这项实验检验辅助监督和
排序能否改善共享表示及候选判断，没有把大模型 WDL 反算成不存在的精确 CP。
同数据、同随机序列的消融应包含 WDL、WDL+CP、WDL+CP+排序，且采用相同验证选模标准。
最终留出评测不参与选模。

本轮固定计划：[`experiment-plan.json`](../reports/evidence/rank-frontier-20261031/experiment-plan.json)。
评测包含相同推理预算 PUCT-31、PUCT-128 和原 PUCT-512。主要看候选损失至少 100 cp 的位置数，
同时报告平均损失、实际神经网络位置数、耗时及配对差异；有限节点老师标签不是数学真值或 Elo。

完整结果：[`CP_RANK_FRONTIER_20261031.md`](../reports/CP_RANK_FRONTIER_20261031.md)。
另有按节点计数的探索选项 `BestFirstPlayer(..., local_exploration=True)`；
对局脚本对应 `--local-exploration`。它在每个节点自己的第 16 次访问触发探索，
在未充分访问的孩子中用策略先验排序，避免全局探索轮次只分配给根的其他分支。
它仍不能保证强制变化被完全证明，默认关闭。
