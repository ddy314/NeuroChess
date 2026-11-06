# Lichess 搜索审计与本地修复（2026-11-06）

当前结论：保留冻结父模型；Lichess 主训练停在 4000 步。独立修复分支只完成
50/100 步，监督门控通过、搜索门控失败后按原标准停止。没有候选晋升，
没有自动扩大训练。所有历史分支的步数独立计数。

## 固定参考与 512 局面审计

父模型是 `puct-2s-v1` 的单编码器、155,000,581 参数权重，SHA256
`9f197a12be729e70f8d7824ef6de1e743b3dce2033c7cc5bb06ce6a6d8832e5c`。
4000 步完整 checkpoint SHA 为
`7e62cea5558ae39dc2cc4925de7768cec18f2fd40ca1784aa6c1131420948655`。
搜索配置及源码指纹按 [`SEARCH_BASELINE.md`](SEARCH_BASELINE.md) 固定。

`scripts/lichess_search_audit.py` 冻结 384 随机＋128 模型输出选择的困难开发
局面，父/子交替顺序运行同 GPU、相同两秒 PUCT。原 1M-node 候选并集筛查
和额外 5M-node 确认分开保存，深确认不抹去原门控失败。纯 CP、mate 排序
哨兵、实际深度和 bounds 分别解释。

4000 步审计的两个集合都未通过原门控，471/512 与父模型同招。随机纯 CP
子集 310 个，平均 regret 差 +0.410cp，位置配对 bootstrap 95% 区间
[-2.094, 2.690]；困难纯 CP 子集 96 个为 -0.104cp，区间
[-24.604, 18.532]。这些不是对局 Elo 区间，困难集合也不是模型盲抽样。
见[分组摘要](../reports/evidence/research-progress-20261106/lichess/search-audit-512.json)。

## 五局面机制与权重对照

`lichess_failure_mechanisms.py`、`lichess_failure_budget_probe.py`、
`coherent_lichess_failure_teacher.py`、`lichess_manual_weight_probe.py`
及汇总脚本保存固定工作量 P/Q/V 输出替换、完整主干/MLP 替换、预算和
更深教师对照。原权重不修改；输出替换同时计算必要网络，不能拿其耗时
当作部署优化速度。这些是五个事后选择案例，不是模型整体因果归因。

- 296 的旧叶 Value 或旧 Policy 可恢复选择；不能只回滚 Value MLP 代替
  Value 的共享主干影响。
- 302 的全新 P/Q/V 组合在该固定预算下出错，任一旧输出可恢复。
- 386/420 在原配对预算下需要完整旧组合；不同预算的结果分别保留。
- 147 的 20M-node、完整 depth 25 标签给出 77cp 差距，不再满足绝对
  100cp 错误阈值。420 的 160M-node、完整 depth 25 找到新分支 mate -20，
  父分支仍是大负 CP；不能相减成 CP 或声称父分支避免必败。
- alpha 0.25/0.5/0.75 全权重插值通过监督门控，但仍留下 386/420 问题，
  全部拒绝。事后 max-root-Q 规则引入其他精确 CP 回退，也没有采用。

见[机制汇总](../reports/evidence/research-progress-20261106/lichess/failure-mechanisms.json)、
[权重插值](../reports/evidence/research-progress-20261106/lichess/manual-weight-probes.json)
和[386 深教师记录](../reports/evidence/research-progress-20261106/lichess/coherent-teachers/386.json)。

## 独立节点与深教师补标

`scripts/lichess_repair_pilot.py` 与 `neurochess/repair_pilot.py` 在本机准备
实际搜索节点，随后调用 `scripts/train_lichess_repair_pilot.py`。这是保留原
artifact 目录、不可变计划和哈希的实验入口，不是自动恢复或扩大训练的许可。

- 排除新 val/test、旧 val、general、冻结 48 根、五个回归及其两层合法后继、
  教师 PV 和诊断节点。新根/节点另排除旧训练键，以及主训练和 A/B/C 已用
  原新数据前缀 3,456,000 行。没有原对局 ID，只声明 canonical keys/根组隔离。
- 80 根搜索产生 1,264 个候选节点。规范化未知历史，排除观测到的重复及
  近五十步状态，保留原始 en-passant 特征。原生/Python 走法顺序不同，按
  动作 ID 集合校验，再以统一顺序重新预测和补标。
- 候选并集包含无限制教师最佳着、P/Q/fused 输出及父/4000 步模型实际锁定
  搜索选择。接受完整同深、无 bounds 的 MultiPV 迭代，每项实际 nodes ≥1M、
  depth ≥24，并作独立更深及真正无限制 PV1 一致性检查。
- 四个单线程 CPU 教师；标签阶段 8 小时、整个本地服务 10 小时上限。未达到
  64 根组各 4 个合格节点就不训练，不放宽门槛凑数。

517 次审计中 309 合格、208 拒绝，最终冻结 256 个唯一节点；逐条重建标签
及候选并集核验通过。见[数据证书](../reports/evidence/research-progress-20261106/repair100/data-ready.json)
及[重建核验](../reports/evidence/research-progress-20261106/repair100/data-checkpoint-verification.json)。

## 最多 100 步联合修复与实际早停

从冻结父模型重置 AdamW，三头及主干均训练；LR 3e-7、10 步 warmup，
有效 batch 256 =32 repair +160 未用 clean +64 old replay，microbatch 32。
Q/Value 继续用 W-L/WDL 语义；保留原损失和旧回放锚定，额外在旧回放及
教师确认的保留节点上约束 fused prior、Q seed 和 Value，对 ≥100cp 的
已知 CP 差增加小权重 Policy 排序项；mate/未知项不伪造 CP。

原监督门控在 25/50/75/100 检查，原两秒搜索标准在 50/100 检查。
fresh 4096 验证是额外读数；只有完整 100 步和主门控通过才运行最终
384 随机＋128 困难审计。五个回归局面不参与训练，原测试标签未用于调参。

实际 25/50 步监督门控均通过，P、Q、Value 与主干参数均有更新；第 50 步
原 48 根通过，额外五个回归＋五个保留对照均值恶化 56.5cp（原门槛 ≤10cp），
147/386 触发个别错误门槛（regret ≥100cp 且恶化 ≥75cp）。

| 回归 ID | 父模型 → 第 50 步 | 原 1M-node 差值 | 5M-node 确认 |
| --- | --- | ---: | --- |
| 147 | h7f6 → g7f6 | 118cp | 104cp，depth 21；先前深标签 77cp，严重度对预算敏感 |
| 296 | f3e3 → 同招 | 0 | 未触发确认 |
| 302 | d5b5 → 同招 | 0 | 未触发确认 |
| 386 | d3e5 → e2e1q | 416cp | 459cp，完整 depth 24 |
| 420 | a6a5 → 同招 | 0 | 未触发确认，不作输赢改善结论 |

本轮是 **50 步门控早停，非 100 步完成、非技术失败**；最终 512 审计未运行。
`rejected.pt` 保存 50 步；`last.pt` 是只有监督通过记录的 25 步；无 `best.pt`，
没有替换默认。第 50 步的实际曝光：

| 样本池 | 展示次数 | 不同局面 |
| --- | ---: | ---: |
| clean | 8,000 | 8,000 |
| repair | 1,600 | 255（冻结池 256） |
| old replay | 3,200 | 3,200 |

修复池等效曝光 6.25 遍，旧回放占 25%。Value 劣势分层 MAE 改善约 0.009，
均势恶化约 0.004–0.005，四个验证组整体 Value MAE 均略升；Policy/Q 变化很小，
仍在原监督容忍范围内。见[完整结果](../reports/evidence/research-progress-20261106/repair100/results.json)
和[解释及限制](../reports/evidence/research-progress-20261106/repair100/interpretation.json)。

报告及全部 1,258 个封存文件 SHA 核验通过，见[交付核验](../reports/evidence/research-progress-20261106/repair100/verification.json)。
本轮全程本地，服务、教师进程和训练 GPU 占用已结束，完成跟进已停用。
反复使用的发展集和一次限时筛查不能建立 Elo、普遍不退化或容量上限结论。
