# 第二阶段：引擎监督

当前运行已从本阶段第 1500 步接续到 500 万局面的扩容训练，见 [扩容与接续说明](phase2-expanded.md)。以下记录首次引擎监督阶段的配置与入口。

保留第 10000 步的 155M 权重，训练 3000 步；不重复启动人类棋谱预训练。
首次入口：`bash scripts/launch_phase2.sh`。脚本默认创建 L4（`NC_GPU=T4` 可选原 T4 配置）、上传本机已准备好的数据和初始权重，完成 Drive 授权后开始训练。终端会持续连接；不要用 Ctrl-C 作为暂停方式。

## 数据与目标

`data/evals-v1/manifest.json` 记录来源、HTTP 版本信息、读取前缀哈希、筛选数量和分布。来自 Lichess Stockfish Evaluation Database，筛选深度至少 20、至少 50 千节点、至少两个不同合法候选着法。同一局面只选一次完整的 MultiPV 搜索，不混合不同深度。局面去重，按局面哈希留出约 1% 验证；另排除已保存测试对局中的 8809 个局面。训练与验证均使用同一历史特征屏蔽。

这是 851213 个合格局面的流式前缀（请求 100 万，本次数据流提前结束；已保留完整解析和验证通过的记录），不是全库均匀抽样。它包含不同版本引擎的搜索结果，不能声称所有标签都是最新 Stockfish 18。分数由白方视角转换为行棋方视角，支持王车易位和升变。未标注动作没有 Q 监督；策略目标是在已标注候选上的分数 softmax，而非引擎访问次数分布。价值和 Q 来自固定 `sf16/ply32` 分数到 WDL 的近似映射，属于搜索分数代理，不能当作校准胜率。

每八个微批次约一个来自原有精英/近期人类棋谱，仅训练策略，避免忘记原有棋谱知识。原数据应在 Drive 的 `MyDrive/neurochess-data/elite-recent-2026q3`。L4 配置见 `configs/phase2-l4.json`：64×16，有效批量 1024，BF16、开启编译、2 个加载进程和 2 个 CPU 算子线程。学习率 5e-5，100 步预热，Q 权重 0.5 在 200 步内升到目标，value 权重 1。原 T4 配置保留在 `configs/phase2.json`，使用 128×8、FP16、不编译。调优依据见 `reports/L4_TUNING.md`。

## 保存、恢复与监控

实例先写 `/content/neurochess-runs/engine-v1`，每 30 秒尝试同步到 Drive 的 `MyDrive/neurochess/engine-v1`；最终保存成功后才结束启动器。同步可能落后于本地进度，突然丢失实例仍可能丢失尚未同步的步骤。每 500 步验证并保存 `last.pt`（可续训）和迄今验证损失最低的 `best.pt`（仅权重）。云端续训自动优先恢复持久化 `last.pt`，训练配置和数据指纹必须匹配。

GUI 端口 8765。在连接到该实例的 Colab notebook 中执行：

```python
from google.colab import output
output.serve_kernel_port_as_iframe(8765, height=900)
```

每 1000 步运行两组开局、交换颜色共四盘的纯策略小型监控，每轮最多 120 秒。这个结果只用于发现明显退化，不是绝对 Elo，也不代替训练后的 MCTS 评测。验证日志另外保存 policy/value/Q loss、Q MAE。当前 MCTS 使用策略与 value，训练 Q 不会自动让 MCTS 改为读取 Q；不要直接对未充分监督的所有动作取 Q 最大值。

若 Drive 挂载失败，训练不会启动。在 `colab url -s neurochess-engine-v1` 对应 notebook 内执行 `from google.colab import drive; drive.mount('/content/drive')` 并完成授权，再运行：

```bash
colab exec -s neurochess-engine-v1 --file scripts/phase2_colab.py --timeout 86400
```

已有实例可设置 `NC_REUSE_SESSION=1` 后运行启动脚本；新实例恢复会从 Drive 拷回 checkpoint。若要重做一次独立实验，先改运行目录，勿覆盖本轮结果。

3000 步约 307 万局面呈现，按本轮 L4 带回放性能测试约 528 positions/s 估算纯训练约 97 分钟，另加数据传输、编译、验证和保存；性能测试使用随机初始化的相同模型结构，回放输入是评估分片替身，不是正式训练完成时长。结束后比较同预算的纯策略和 MCTS 对局，再决定继续蒸馏，不能仅凭训练 loss 宣称达到 2500 Elo。
