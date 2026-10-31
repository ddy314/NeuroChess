# 手工训练与 Colab

推荐使用 [MONITOR.md](MONITOR.md) 的新版入口：默认近期精英数据、后台训练、GUI 和周期对局。以下保留自行准备单个数据集的低层用法，因此启动命令显式关闭默认 profile 检查与自动评测。

## 先看成本

默认 T4。此次账户观察到 1.07 计算单位/小时，实际随账户与运行环境变化，请先 `colab usage`。T4 155M 模型 FP16 合成完整优化步约 260 局面/秒；一千万局面仅纯计算就约 10.7 小时，尚未计入数据、验证、保存、断线和编译。低小时价格不必然是大规模训练的最低总价，但本轮不自动升级昂贵卡。

把模型训练到强棋力可能需要多轮、更多样本和更多预算。先做 100–1000 步 pilot，再按每百万局面的计算单位和实际棋力收益决定扩展规模。

## 本地准备

激活 `.venv` 后按 README 生成 `data/human` 和 `data/teacher`。生产数据建议采用较新的、质量过滤后的多个标准棋月份；本次 `data/smoke-*` 只有几千／几百样本，**只用于检查程序**。

训练数据必须同时有 train 和 val 分片及完整 `manifest.json`。目标目录必须不存在；失败产生的目录先另行保留检查，再用新名称重跑。当前转换器没有中途续转功能。

```bash
source .venv/bin/activate
python -m neurochess.audit --data data/human
python -m neurochess.audit --data data/teacher

tar -czf /tmp/neurochess-source.tar.gz neurochess configs scripts tests pyproject.toml
tar -czf /tmp/neurochess-data.tar.gz data/human data/teacher
colab new -s neurochess-train --gpu T4
colab upload -s neurochess-train /tmp/neurochess-source.tar.gz /content/neurochess-source.tar.gz
colab upload -s neurochess-train /tmp/neurochess-data.tar.gz /content/neurochess-data.tar.gz
```

大数据应直接在 VM 本地下载／转换或从持久存储复制，避免本地重复上传；上面打包方案用于中等规模首轮。

## 持久保存

用户在交互终端挂载自己的 Drive（可能需要浏览器授权）：

```bash
colab drivemount -s neurochess-train /content/drive
```

下面检查点目录为 `/content/drive/MyDrive/neurochess/human`。必须先确认 Drive 挂载成功，否则 `/content` 下的文件随实例释放而丢失。不要为了后台运行而假设 Colab 永不回收；可靠恢复依赖持久检查点。

## 启动预训练

先按计划的总步数运行前 100 步。pilot 的 learning-rate schedule 与之后继续训练一致。

```bash
colab exec -s neurochess-train -f scripts/colab_job.py --timeout 86400 --env NC_REQUIRE_PROFILE=0 --env NC_ARENA=0 \
  --env 'NC_JOB=["--data","data/human","--out","/content/drive/MyDrive/neurochess/human","--steps","10000","--stop-after","100","--batch-size","128","--accum","8","--workers","2","--save-every","100","--warmup","500"]'
```

检查损失和 Drive 里的 `last.pt`。之后恢复，去掉 `--stop-after`，保持 batch/accum/workers/seed/lr/warmup/steps 不变：

```bash
colab exec -s neurochess-train -f scripts/colab_job.py --timeout 86400 --env NC_REQUIRE_PROFILE=0 --env NC_ARENA=0 \
  --env 'NC_JOB=["--data","data/human","--out","/content/drive/MyDrive/neurochess/human","--steps","10000","--batch-size","128","--accum","8","--workers","2","--save-every","500","--warmup","500","--resume","/content/drive/MyDrive/neurochess/human/last.pt"]'
```

这条命令阻塞到完成；可以由用户放进本地 `tmux` 会话，避免终端关闭。Colab 执行超时／连接断开并不代表 VM 必然停止；用 `colab status` 核对，不要直接启动第二份训练。VM 被回收时重新分配、上传／准备数据、挂载 Drive，然后执行同一 resume 命令。

每 500 步（有效 batch≈1024）保存一次，可能丢失上次保存之后的训练。单 checkpoint 含优化器，体积约为参数 FP32 权重的三倍，Drive 保存会占用时间。当前保留一个原子替换的 `last.pt`，阶段结束可手动另存快照。只加载自己信任的 checkpoint。

`--compile` 可单独比较，不默认开启。先在同一卡上运行 `python -m neurochess.benchmark --batch-size 128 --compile`，总时间需计入首次编译；不要用不断变化的输入形状做比较。

## 蒸馏微调

CPU 端先生成并审核 teacher 分片；不要占用 GPU 等待 Stockfish 分析。`--multipv 8` 是省 CPU 的策略蒸馏起点，Q-only 对局前建议全合法动作标注。启动新阶段使用 `--init`，不是 `--resume`：

```bash
colab exec -s neurochess-train -f scripts/colab_job.py --timeout 86400 --env NC_REQUIRE_PROFILE=0 --env NC_ARENA=0 \
  --env 'NC_JOB=["--data","data/teacher","--out","/content/drive/MyDrive/neurochess/teacher","--init","/content/drive/MyDrive/neurochess/human/last.pt","--steps","10000","--batch-size","128","--accum","8","--workers","2","--save-every","500","--warmup","500","--lr","0.0001"]'
```

## 对局与结束

`openings.fen` 必须来自独立测试来源，每行一个合法非终局标准 FEN：

```bash
python -m neurochess.play --checkpoint runs/teacher/last.pt \
  --stockfish /path/to/stockfish --openings openings.fen --pairs 100 \
  --nodes 10000 --mode policy --device cuda --out runs/match-policy
# 用相同开局再运行 --mode oneply，输出到不同目录。
```

检查 PGN、截断局数、双方颜色、对手设置和置信区间；这一步提供相对比较，不能直接输出绝对 2500 Elo。

完成后核实持久文件，再释放实例：

```bash
colab stop -s neurochess-train
colab sessions
colab usage
```
