# 训练进展与评测记录

本页整理截至 **2026-11-06** 的已核验记录。步数以各阶段和独立分支为准，不能跨初始化、优化器或任务直接累加。证据归档只包含小型 JSON、协议和评测记录，模型、训练数据及运行日志仍在仓库外。

## 当前决策与搜索下限

最终优化的两秒 PUCT 仍为最低质量基线，配置、模型 SHA 和源码指纹固定在
[`search-baseline-lock.json`](../configs/search-baseline-lock.json)。没有将 PAWN、
Lichess 4000 步、A/B/C、权重插值或本地修复候选晋升为默认模型。

Lichess 主训练逻辑暂停于 4000 步，一个 epoch 未完成。最近独立修复试验于
2026-11-06 00:32（北京时间）在第 50/100 步触发搜索门控后结束，训练和教师
进程已退出。没有自动扩大训练。当前结果支持暂停现有训练配方，不能据此认定
155M 参数模型达到容量极限，也没有新的可靠 Elo 结论。

## 2026-11-01 至 11-06：新增工作

### 原生推理与搜索优化

增加 CUDA/Triton、BF16/INT8 TensorRT Graph、异步 C++ PUCT 流水线及原生
UCI 驻留推理；保留 PyTorch 训练、原搜索规则与冻结权重。约 0.2 秒候选在部分
协议下达到 P95 约 10×，但更深教师发现已知局面的 278cp 回退，八局直接对弈
也不足以证明质量不降。0.5/1/2 秒回投搜索的结果分别记录，不能借用 0.2 秒
版本的对弈成绩。**速度目标与保留棋力尚未同时达成。**

见[优化报告](PUCT_10X_OPTIMIZATION_20261101.md)、
[原生运行时报告](NATIVE_INFERENCE_20261101.md)及各自归档证据。

### 定向标注、联合训练与动态候选

后续定向流程交付 472,576 个新标注训练局面和 15,761 个已确认错题，另有旧
500 万教师数据回放。独立联合分支有效 batch 320，旧/新/错题为 40/40/20%，
训练 10,000 次更新，分支步号从 3000 到 13000，两个原编码分支及三头均训练。
这不是把历史 `curriculum-v3` 的 15500 步继续到 13000 步。

检查点/RNG 回放核对的实际曝光为：旧数据 1,280,000 次、1,128,978 个不同局面；
新数据 1,280,000 次、441,018 个不同局面；错题 640,000 次、15,761 个不同局面，
平均每题约 40.6 次。不能把展示次数全部解释为新数据。见
[曝光核对](evidence/research-progress-20261106/joint/exposure.json)和
[搜索评估边界](evidence/research-progress-20261106/joint/search-assessment.json)。
没有以监督指标改善推断可靠棋力提升，也没有晋升该分支。

动态管线随后完成 520,000 个候选、22,953 个来源棋局的完整审计，train/val/test
为 500k/10k/10k，精确历史键、重复键与新集合来源棋局交集为零。该审计仍明确
`teacher_labels_ready=false`，试标及后继生成不等于生产数据或训练已完成。
无限制 PV1 一致性复核、缺失目标掩码、来源组隔离、全 batch 归一化及重复曝光
上限见[动态管线](../docs/DYNAMIC_CURRICULUM.md)和
[候选审计](evidence/research-progress-20261106/dynamic/candidate-audit.json)。

### PAWN：准备与实际训练分开计数

完成 20,000,000 个唯一训练根的准备。实际分支完成 6,429 次更新，展示
6,582,592 个样本：其中 PAWN 4,936,944 个、旧回放 1,645,648 个。PAWN 验证
loss 从 2.84418 降至 2.63309（-7.42%），旧通用 loss 从 2.38547 升至
2.44171（+2.36%），超出原 2% 容忍范围。

冻结 48 根筛查中，平均 CP regret 从 6.04 到 21.85，新增两处百厘兵级错误。
对弈在用户要求后停止，仅完成九局、四个完整换色对，不能作 Elo 或因果结论。
该分支还存在此前双编码器路径与冻结单编码器基线的差异，不能将全部差异都
归因于 PAWN。没有晋升候选。见[训练摘要](evidence/research-progress-20261106/pawn/run-summary.json)、
[停止后的评测](evidence/research-progress-20261106/pawn/arena-analysis.json)和
[搜索路径对照](evidence/research-progress-20261106/pawn/search-work-analysis.json)。

### Lichess 清洗、4000 步与 A/B/C

严格清洗完成 10,000,000 个唯一局面：9.8M train、100k val、100k test；
depth ≥24、knodes ≥1000，全部目标行数值审计和选中 PV 合法回放通过，另重建
9,932 行检查。数据源没有原对局 ID，不能声明源对局独立。见
[完成证据](evidence/research-progress-20261106/lichess/cleaning-completion.json)及
[完整过滤协议](../docs/LICHESS_CLEAN_10M.md)。

主训练从冻结单编码器父模型开始，三头及主干联合训练，到第 4000 步展示
3,072,000 个新局面及 1,024,000 个旧回放。新验证 loss 从 2.61136 降至
2.59207（约 -0.74%），随后按授权保存并暂停。A/B/C 各完整完成 500 步，原
监督与 48 根搜索门控均通过；三个分支共用相同的后续 384,000 个新局面和
128,000 个旧回放，不能累计成三倍唯一新数据。

| 分支 | 设置 | 最终新 loss | 新 Q MAE | 新 Value MAE |
| --- | --- | ---: | ---: | ---: |
| 4000 步起点 | 原设置 | 2.592065 | 0.225610 | 0.236741 |
| A | 原 LR、完整 Value 均衡 | 2.592684 | 0.225586 | 0.237239 |
| B | LR 逐渐到 1.5× | 2.592679 | 0.225454 | 0.237211 |
| C | Value 均衡强度 0.5 | 2.589604 | 0.225584 | 0.236044 |

这些开发集结果没有自动选出续训组。八批原始梯度诊断是描述性几何分析，不是
Adam 更新的因果证明。另做的本地 D 组 16 批诊断未满足预先声明的触发规则，
所以 D **实际更新 0 步**。见[A/B/C](evidence/research-progress-20261106/lichess/ablation-results.json)、
[梯度诊断](evidence/research-progress-20261106/lichess/gradient-diagnosis.json)、
[D 组](evidence/research-progress-20261106/lichess/local-d-results.json)及
[对照协议](../docs/LICHESS_4000_ABLATION.md)。

### 512 局面审计、机制与修复早停

4000 步模型完成 384 随机＋128 困难局面的教师/两秒搜索审计，两个集合都未
通过原搜索门控；471/512 与父模型同招。纯 CP 子集的配对平均差区间跨零，
不能用混入将杀哨兵的平均数解释为真实 CP 或 Elo。

对五个事后选出的局面做固定工作量输出替换、共享主干/头和更深教师诊断。
147 的深标签将差距从百厘兵级修正到 77cp；420 的 mate/CP 只能作排序，
不能相减为 CP 或证明父分支不败。三种权重插值均留下 386/420 问题，没有
采用。详见[审计](evidence/research-progress-20261106/lichess/search-audit-512.json)、
[机制分析](evidence/research-progress-20261106/lichess/failure-mechanisms.json)和
[审计与修复说明](../docs/LICHESS_SEARCH_REPAIR.md)。

随后从冻结父模型启动独立最多 100 步三头修复。517 次节点补标审计中 309
合格，选定 64 组各 4 个、共 256 个节点。原五局面及保留后继只做回归，
原测试标签未用于调参。第 25/50 步监督门控通过，原 48 根搜索通过；额外
回归/保留对照在第 50 步因 147、386 失败，自动保存拒绝候选并停止。
386 的 5M-node、depth 24 确认仍差 459cp。没有继续到 100，也没有运行
条件式最终 512 审计。第 25 步 checkpoint 仅通过监督门控，不是获准替换。

实际 50 更新展示 clean 8,000、repair 1,600、old replay 3,200，共 12,800 次；
对应不同局面为 8,000、255、3,200。修复池等效曝光 6.25 遍，旧回放 25%。
结果及 1,258 个封存文件的 SHA 全部核对通过。见
[结果](evidence/research-progress-20261106/repair100/results.json)、
[交付核验](evidence/research-progress-20261106/repair100/verification.json)和
[数据/检查点核验](evidence/research-progress-20261106/repair100/data-checkpoint-verification.json)。

上述新归档的[来源清单](evidence/research-progress-20261106/provenance.json)同时
保存原文件及路径规范化副本的 SHA。数字和结果字段不改写；本地绝对路径作
占位符处理。完整训练池、checkpoint、优化器张量、日志和授权信息不入库。

## 早期训练记录（保留原协议）

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

2026-10-31 的历史快照选定 500,000 个训练、10,000 个验证和 2,000 个测试局面；独立检查记录未发现与既有局面键的交集或新数据跨集合棋局重叠，但无法确认旧预训练来源的棋局完全互不重叠。[选择审计](evidence/training-progress/fresh-v4-selection-audit.json)和[实验设计](../docs/FRESH_V4.md)说明边界。没有原计划两组训练及最终对局的完整闭环证据；后续定向/联合训练作为上文的独立分支记录，不将旧状态文件解释为当前活动任务。
