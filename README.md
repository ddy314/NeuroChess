# NeuroChess

NeuroChess 是从零实现的国际象棋 AI 系统：用人类棋谱预训练稠密 Transformer，再用 Stockfish 标注监督策略与价值，并结合搜索和配对对局评测棋力。主模型有 **155,000,581 个参数**。项目尚未完成绝对 Elo 标定，不能把固定节点数的 Stockfish 对局结果当作人类等级分。

## 训练进展

以下是截至 **2026-11-06** 的已核验进展。当前保留冻结的最终两秒 PUCT 基线；Lichess 主训练逻辑停在 4000 步，最近修复分支在第 50/100 步被搜索门控停止，没有自动续训或模型晋升。详细记录及可入库的小型证据见 [训练进展与评测](reports/TRAINING_PROGRESS.md)。

| 阶段 | 状态 | 结果与衔接 |
|---|---|---|
| 人类棋谱预训练 | 已完成 | 训练 10,000 步，作为后续引擎监督的初始化权重。 |
| 引擎监督及数据扩容 | 已完成 | 从第 1,500 步接续，训练集扩至 500 万局面，再完成 6,000 次更新；引擎监督累计至 7,500 步。 |
| 第三阶段 `curriculum-v3` | 已完成 | 保留优化器状态，从 7,500 步再训练 8,000 次更新，达到引擎监督累计第 15,500 步；完成最终验证与两种模式各 64 局对局评测。 |
| 原 `fresh-v4` A/B 计划 | 历史准备记录 | 原方案没有完整 A/B 闭环的完成证据；后续定向标注与联合训练另立分支，不能据此补记为原计划完成。见 [实验设计](docs/FRESH_V4.md)。 |
| 定向数据与三头联合训练 | 已完成独立分支 | 以 472,576 个新标注局面、15,761 个错题及旧数据回放完成 10,000 次联合更新；错题实际重复曝光较高，未建立可靠棋力提升或默认晋升。 |
| 动态错题管线 | 候选审计完成 | 52 万候选、22,953 个来源棋局通过审计；`teacher_labels_ready=false`，生产训练未启动。见 [动态数据说明](docs/DYNAMIC_CURRICULUM.md)。 |
| PAWN | 数据完成、候选未晋升 | 准备 2,000 万训练根；实际展示 6,582,592 个样本，其中 4,936,944 个 PAWN、1,645,648 个旧回放；旧任务和冻结搜索回归。 |
| Lichess 官方评估库 | 清洗完成 | depth ≥24、knodes ≥1000，1,000 万唯一局面，按 980 万／10 万／10 万划分 train/val/test。见 [清洗协议](docs/LICHESS_CLEAN_10M.md)。 |
| Lichess 联合训练及 A/B/C | 主训练暂停、对照完成 | 主训练 4000 步、3,072,000 个新局面；A/B/C 各 500 步，共用后续 384,000 个新局面序列。没有自动选组继续或完成一个 epoch。见 [训练协议](docs/LICHESS_EPOCH1.md)和[对照结果](docs/LICHESS_4000_ABLATION.md)。 |
| 搜索审计与修复 | 修复未通过 | D 组诊断未触发训练；512 局面审计发现局部退化；独立 256 节点修复在第 50 步监督门控通过、搜索门控失败后停止。见 [审计与修复](docs/LICHESS_SEARCH_REPAIR.md)。 |
| 原生推理与 10× 搜索优化 | 实现和评测已记录 | CUDA/TensorRT、INT8 Graph 和 C++ 驻留推理已实现；约 0.2 秒候选存在质量回退，速度与保留棋力两项要求尚未同时满足。见 [原生运行时](docs/NATIVE_INFERENCE.md)和[优化报告](reports/PUCT_10X_OPTIMIZATION_20261101.md)。 |

人类预训练的 10,000 步和引擎监督的累计步数属于不同训练阶段，不能直接相加。第三阶段选中的模型权重及训练数据保存在仓库外；其 SHA-256 和证据记录在[训练进展报告](reports/TRAINING_PROGRESS.md)。

后续联合训练、PAWN、Lichess 及本地修复也有各自的初始化、优化器和步数空间，不能把这些步数拼成同一条连续训练曲线。监督 loss、开发局面筛查、固定预算对局和绝对 Elo 分别报告；目前没有测定模型容量上限。

第三阶段对 Stockfish 18、每步 1,000 节点的固定条件测试中，直接策略走法为 **1 胜 9 和 54 负**，每步 512 次 PUCT 搜索为 **29 胜 14 和 21 负**；每种模式 64 局，共 32 个换色开局对。样本有限，搜索和直接走法的计算预算也不同，这些结果不能换算为绝对 Elo。

## 实现

- 16 层、896 维、14 注意力头的棋盘 Transformer；输出合法着法策略、动作价值和局面胜／和／负分布。
- 人类棋谱过滤、去重和 mmap 分片；Stockfish MultiPV 标注、策略／Q／价值联合监督与人类策略回放。
- AdamW、梯度累积、FP16/BF16 和可选 `torch.compile`；支持检查点续训及跨数据阶段保留优化器状态。
- 直接策略、一步展开和原生 C++ PUCT 搜索；对局评测保存棋谱、结果和配对统计。
- 只读训练看板、周期评测、Drive 持久化及云端任务启动器；包括实验性的最佳优先 Minimax 与搜索诊断。

这是受棋格 Transformer 启发的自主实现，**不是 Chessformer GAB 的逐项复现**。最佳优先 Minimax 和轻量评估器仍属实验研究，不应与已训练主模型混为一谈。

## 安装与验证

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -e '.[dev]'
.venv/bin/python scripts/build_native_mcts.py
.venv/bin/python scripts/build_native_pipeline.py
.venv/bin/python scripts/build_best_first.py
.venv/bin/python -m pytest -q
```

原生规则测试需要本机 C++17 编译器；上述构建不需要 TensorRT 或 CUDA SDK。GPU 训练需使用支持目标 CUDA 设备的 PyTorch。训练数据、外部 Stockfish 二进制和模型权重不随仓库分发。可选 CUDA/TensorRT 集成测试在缺少运行时或引擎计划时跳过；测试通过不等于新的构建获准替换冻结部署基线。

- [训练路线、进展与结果](reports/TRAINING_PROGRESS.md)
- [第三阶段课程设计](docs/CURRICULUM_V3.md) · [后续新数据实验](docs/FRESH_V4.md)
- [Lichess 清洗](docs/LICHESS_CLEAN_10M.md) · [联合训练](docs/LICHESS_EPOCH1.md) · [A/B/C](docs/LICHESS_4000_ABLATION.md) · [搜索审计与修复](docs/LICHESS_SEARCH_REPAIR.md)
- [动态错题管线](docs/DYNAMIC_CURRICULUM.md) · [冻结搜索下限](docs/SEARCH_BASELINE.md)
- [训练和监控操作](docs/TRAINING.md) · [看板与对局评测](docs/MONITOR.md)
- [模型设计](docs/DESIGN.md) · [PUCT 搜索](docs/NATIVE_MCTS.md) · [最佳优先 Minimax](docs/BEST_FIRST_MINIMAX.md)
- [搜索诊断](reports/SEARCH_DIAGNOSIS_20261031.md) · [CP 与排序蒸馏](reports/CP_RANK_FRONTIER_20261031.md) · [CPU 搜索对照](reports/CPU_SPEED_20261031.md)

## 基础训练示例

下面的示例展示数据转换和训练接口；它不会复现上表中完整的多阶段实验。

```bash
python -m neurochess.prepare --pgn SOURCE.pgn.zst --out data/human --min-elo 2000 --max-positions 1000000
python -m neurochess.audit --data data/human
python -m neurochess.train --data data/human --out runs/human --steps 10000 --batch-size 128 --accum 8
```

完整训练前应先运行小规模检查，核实数据、吞吐、验证和检查点保存。旧的 [第二阶段说明](docs/phase2.md) 与 [扩容说明](docs/phase2-expanded.md) 描述历史流程，不能当作当前云任务状态。
