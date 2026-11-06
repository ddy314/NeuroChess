# 本地动态错题数据管线

这条 2026-11-03 候选准备分支的学生权重固定为 `output/joint-deep-analysis-20261103/final-weights.pt`，SHA-256 为
`99931bbb7f68a812e226364c9636a3725599a54dcc3affb0cc206a48318c4d3a`。
管线只筛选、标注和导出数据；不创建实例、不启动优化器、不自动晋升模型。

截至 2026-11-06，520,000 个候选和 22,953 个来源棋局的完整审计已通过，
但 `teacher_labels_ready=false`；生产规模训练没有启动。这个学生权重不是
替换冻结两秒 PUCT 的默认模型。见[候选证据](../reports/evidence/research-progress-20261106/dynamic/candidate-audit.json)
及[累计进展](../reports/TRAINING_PROGRESS.md)。下文保留准备和未来训练接口。

## 数据和标签

- `collect_dynamic_elite.py` 从校验过发布者 SHA 的广播 PGN 收集新候选。双方 Elo ≥2400；已知基础时长 ≥600秒，未知或无法解析的时限保留原字段。仅正常起始局面和合法标准棋，拒绝 BOT、未完成棋谱、解析错误和空着。
- 完整棋局轨迹决定来源组及 train/val/test。排除 8,229,922 个历史局面键和所有本地可得旧来源组。无法证明与缺失来源信息的历史棋局完全隔离，因此仅声明这些历史数据的精确局面排除。
- 新增根题、派生后继、Q-only 错题分别计数；不跨目录累加重复试验。未标注候选数不是合格错题数。
- 学生分支为：学生走子 → SF 最佳应对 → 学生续着 → SF 最佳应对。每个节点独立标注，另保留教师 PV 回放。记录完整起点/走子历史、棋钟、重复局面特征、来源组、根题、分支、深度、生成模型 SHA 和教师预算。
- WDL 数值继续使用 `sf16`、固定 ply=32；policy 温度100cp，Q=W−L。动作 Q 逐项判断稳定，缺失 Q 使用 qmask=False；缺失 policy 为零向量，缺失 value 的 weight=0。
- 候选集内100k/300k稳定不能独自证明全局 V。当前配置增加独立、无限制候选着的100k/300k PV1复核；V 的两个 WDL 目标均须与对应候选集搜索相符且全局 WDL 稳定。全局最佳着缺失时也屏蔽 policy。分歧保留供后续复核，不填入“和棋”等伪目标。
- 换学生权重后重新预测并判错。教师证据仅在完整历史、引擎 SHA、标签协议和必要动作覆盖一致时复用；全局 V 复核可在动作支持集变化时单独复用。新模型已掌握的题不再进入活动错题池。

## 回放与未来训练接口

普通新数据由独立、未查看学生错误的抽样选取并强制教师确认。`new` 池不是错题池的别名。
目标普通新候选160k；计划有效 batch320 = 160旧回放 +96普通新数据 +32后继 +32活动错题。
计划一条训练分支、总预算10k更新，按500更新重筛；这些是准备参数，当前没有接入云端训练循环。

同一训练 run 内，每个规范局面的全部动态池合计最多8次，单次更新不重复。
run 的种子和上限持久化；恢复同一次更新取得同一批 ID。缺题直接报错，不循环小题库。
10k更新至少需要120k合格普通根题和40k合格后继的独立容量；加上100k主口径错题目标。
实际容量不足或标签门禁不满足时不得直接增加步数。

`dynamic_training.losses` 对有效目标归一化；累积时传入整个320样本的 normalizers，微批 loss 相加，不再除 accumulation。
已验证16/32/64微批的梯度等价。旧训练入口拒绝直接读取这类缺失目标数据。
SQLite 抽样器是语义参考实现；大训练前应生成缓存/打包的采样计划并实测热路径，不能让每次更新全表扫描拖住 GPU。

## 命令

候选收集、审计及模型盲抽样的实物在 `output/dynamic-curriculum-20261103/`。
先检查 `new-elite/full-audit.json` 和文件 SHA，再用：

```sh
.venv/bin/python scripts/stage_dynamic_curriculum.py \
  --source output/dynamic-curriculum-20261103/new-elite \
  --out output/dynamic-curriculum-20261103/scale-inputs
```

标注按有界块推进；下面只处理首批2k根题。增大 `--roots` 可恢复同一输出，已完成工作不会重复。生产抽样每来源棋局上限12；试验可用更小配额。

```sh
/home/xia/code/ml_frontier/.venv/bin/python scripts/run_dynamic_curriculum.py \
  --inputs output/dynamic-curriculum-20261103/scale-inputs/train-inputs.jsonl \
  --out output/dynamic-curriculum-20261103/scale-labels \
  --roots 2000 --expand-roots 64 --max-roots-per-game 12 --workers 16 \
  --exclude-keys output/dynamic-curriculum-20261103/full-exclusions/known-prior-keys.npy \
  --frozen-groups output/dynamic-curriculum-20261103/full-exclusions/frozen-groups.json
```

`--refresh-store --model PATH --model-sha SHA` 可对已有根题和后继换权重重筛，不默认截断到128题。
异步队列和来源生产者都有界；无需先重建/插入全部50万历史才启动 GPU 和 SF。
`pipeline.lock` 防止重复进程；SQLite FULL/WAL、分批提交和原子状态文件支持恢复。

导出需使用 `export_dynamic_curriculum.py`，同时保存 NPY、逐行历史/监督掩码侧车和 SHA 清单。
只有实际主口径错题数、独立普通回放、各头验证/测试覆盖、同批深审、去重证明均通过才发布 `teacher-ready.json`。
当前是候选准备和试标阶段。禁止把原型的吞吐/标签通过率代替现行全局复核协议的生产结果。

来源：[Lichess broadcasts 与许可证](https://database.lichess.org/#broadcasts)、[发布者校验和](https://database.lichess.org/broadcast/sha256sums.txt)。
CPU引擎：[官方 SF18 发布](https://github.com/official-stockfish/Stockfish/releases/tag/sf_18)；原始二进制及冻结搜索配置未被替换。
