# NeuroChess

从零实现的稠密国际象棋 Transformer。主模型 **155,000,581 参数**；目标是通过棋谱预训练、Stockfish 蒸馏和明确条件下的对局评测，逐步争取 2500+ 棋力。

**当前尚未通过绝对 Elo 棋力认证。** 已完成人类棋谱阶段 10000 步；引擎监督阶段在第 1500 步保存后，将训练集从 842756 扩到 5000000 个局面，并在 L4 上保留优化器状态接续。配置和验证方式见 [扩容训练说明](docs/phase2-expanded.md)。

## 实现

- 16 层、896 隐藏维、14 注意力头；64 棋格 token，行棋方视角，棋子／行／列嵌入和规则状态。
- 一次前向输出全部合法动作的策略、动作价值，以及局面胜／和／负分布。四种升变均独立编码。
- PyTorch SDPA、FP16/BF16、fused AdamW、梯度累积、可选 `torch.compile`。T4 强制选原生 FP16，避免 BF16 模拟路径。
- 本地或 HTTPS PGN / PGN.zst 流式转换为固定结构 mmap 分片；训练时不做棋规计算。内存锁页与异步 CPU→GPU 拷贝。
- Stockfish 多进程 MultiPV 蒸馏，未标注动作不计算 Q 损失；记录引擎版本、二进制哈希和标注预算。
- 原子保存模型／优化器／精度缩放器／数据游标；`--resume` 恢复同一训练，`--init` 以权重开始下一阶段，`--continue-from` 在新数据阶段保留优化器状态。
- 无搜索 policy、实验性 Q 策略、批量一步展开；配对换色对局输出 PGN、结果和按开局对抽样的置信区间。
- 实验性最佳优先 Minimax：小型 CPU 增量评估器支撑搜索，按独立预算查询大模型校正；小评估器支持本地 CUDA 蒸馏，见 [实现和复现实验](docs/BEST_FIRST_MINIMAX.md)。
- [战术漏算诊断](reports/SEARCH_DIAGNOSIS_20261031.md)：容量、搜索分布、WDL 饱和与校正机制的受控对照，以及改进版与原 PUCT 的结果。
- [CP 与相对排序蒸馏实验](reports/CP_RANK_FRONTIER_20261031.md)：按棋局分组的扩大边界数据、本地 GPU 消融，以及额外的节点探索诊断。
- [校正与 CPU 战术预算实验](reports/CORRECTION_SAFETY_20261031.md)：校正限制的负面结果、独立将杀证明与强制应手搜索，以及新留出组和 PUCT 对照。
- [CPU 加速与扩大直接对局](reports/CPU_SPEED_20261031.md)：内核约 3.9 倍 CPU 加速、新留出组与 192 局直接对弈的相对强度和区间。

这是受棋格 Transformer 启发的自主实现，**不是 Chessformer GAB 的逐项复现**。主模型保持稠密 Transformer；新增的小型增量评估器属于独立的搜索实验，目前尚未达到现有 PUCT 的选着质量。

训练监控 GUI 提供实时 loss、学习率、吞吐、数据准备进度和周期对局结果。Colab 启动器支持后台执行、持久数据缓存，并默认每 1000 步做限时对局监控。

## 本地安装与验证

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -e '.[dev]'
.venv/bin/python -m pytest -q
OMP_NUM_THREADS=2 .venv/bin/python scripts/local_integration.py
```

本机已有的 `.venv` 安装 CPU PyTorch；Colab 使用其自带 CUDA PyTorch。不要把 CPU 环境打包上传。

- **[推荐启动：近期精英数据 + 实时 GUI + 周期评测](docs/MONITOR.md)**
- [手工训练与 Colab 操作说明](docs/TRAINING.md)
- [设计、评测边界与训练路线](docs/DESIGN.md)
- [实测记录](reports/VALIDATION.md)

## 最小数据与训练链路

```bash
# SOURCE.pgn.zst 为你下载的标准国际象棋棋谱。
python -m neurochess.prepare --pgn SOURCE.pgn.zst --out data/human --min-elo 2000 --max-positions 1000000
python -m neurochess.audit --data data/human
python -m neurochess.train --data data/human --out runs/human --steps 10000 --batch-size 128 --accum 8

python -m neurochess.sample_fens --pgn SOURCE.pgn.zst --out data/teacher.fen --positions 100000 --min-elo 2000
python -m neurochess.distill --fens data/teacher.fen --stockfish /path/to/stockfish --out data/teacher --workers 4 --nodes 100000 --multipv 8
python -m neurochess.audit --data data/teacher
python -m neurochess.train --data data/teacher --out runs/teacher --init runs/human/last.pt --steps 10000 --batch-size 128 --accum 8 --lr 0.0001
```

上述数量是可修改的试验规模，不是达到目标棋力的充分条件。完整训练前先执行短 pilot，核对吞吐、损失和保存文件。

## 第二阶段与扩展监控

从第 10000 步转入引擎数据监督，参见 [第二阶段启动说明](docs/phase2.md)。准备好的启动入口为 `bash scripts/launch_phase2.sh`。看板新增验证分量、Top-1、Q MAE、梯度、显存、检查点及持久化状态；历史日志缺失项显示未记录。

此前的 `engine-v2-expanded` 运行 从首轮引擎监督第 1500 步接续，使用 500 万训练局面和原来的固定验证集，计划新增 6000 次更新，见 [扩容与接续说明](docs/phase2-expanded.md)。
