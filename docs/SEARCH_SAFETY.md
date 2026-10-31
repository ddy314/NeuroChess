# 有限校正与独立 CPU 战术预算

所有选项默认关闭，可通过 `BestFirstPlayer` 或 `scripts/best_first_arena.py` 显式开启。
CPU 搜索使用本地小模型和规则库，不调用 Stockfish，也不请求大模型。
Stockfish 只用于离线、统一候选比较。

```python
player = BestFirstPlayer(
    cheap_weights, network,
    expansions=512, neural_budget=32, correction_batch=4,
    root_policy_weight=.1, align_root_objective=True, local_exploration=True,
    mate_nodes=16384, mate_depth=7,
    tactical_nodes=16384, tactical_depth=2,
)
```

这是本轮更有希望的 CPU 实验配置，仍未证明整体优于 PUCT。
后续 [CPU 加速及 192 局直接对局](../reports/CPU_SPEED_20261031.md) 显示当前实现仍明显落后于 PUCT，
该配置继续作为实验模式，不作为强度升级默认启用。

底层加速本身保留原搜索预算；高速减量配置和原预算控制的实战结果分开列于上述报告。
完整结果见 [`CORRECTION_SAFETY_20261031.md`](../reports/CORRECTION_SAFETY_20261031.md)。
三种校正幅度限制均未通过本轮效果对照，建议保持零值，保留实现仅用于复现消融。

`correction_clip` 限制每个备份/叶子融合操作中，神经网络锚点相对当前参考值的变化幅度。
应用前还乘该节点的锚定权重；原始网络输出保留在 `anchor`，不冒充真值。
这不能限制多层树备份的累计影响。

`root_value_clip` 另外限制根候选最终价值及根选择价值，相对该孩子首次小模型估值的偏移。
它使用不可变的初始参考，避免后续校正和树备份将整个参考一起拉走。
根策略先验仍参与最终分数；精确胜负的优先级独立于这两个幅度限制。
两个阈值都是 WDL 量纲的启发式约束，不是置信区间，也不能保证保持正确排序。
初始小模型估值错误时，该约束也可能阻挡有益校正。

`cpu_correction_clip` 则只在该节点有完成的独立 CPU 深度结果时限制融合幅度，
其余节点保持原校正。完成一个搜索深度不代表评估正确；该条件版本在追加留出组也没有改善。

`mate_nodes` 是单独的将杀证明递归节点上限，按根孩子和攻击方向分配额度，并迭代深度。
攻击方可选择将军、吃子和静着；防守方的全部合法回应必须成立，才能证明将杀。
最后一步只检查将军着法，因为将杀必须将军。重复、和棋和终局依照主搜索规则处理。
只会把成功的证明写入 `solved`；失败、深度不足和预算耗尽都保持未知。
这不保证在给定额度内发现所有短将杀。

`tactical_nodes` 是另外的 CPU 价值搜索节点上限。它使用全宽 alpha-beta、将军/吃子排序、
被将军或只有一个应手时的有限延伸，以及吃子/升变/首层将军的静态搜索。
被将军时不使用 stand pat 截断，检查合法应手；超过应手延伸或资源上限返回不完整。
只有完成的深度结果作为根孩子的启发式参考，与后续树备份各占一半，再应用神经网络融合。
价值结果不会生成精确胜负证明。两类 CPU 预算都不占用神经网络位置预算。

`trace()` 提供 `reference`、`applied_correction`、`anchor_clipped`、`cpu_value`、`cpu_depth`、
`cpu_ready`。结果统计提供实际 `mate_nodes`、`tactical_nodes`、`cpu_mate_proofs`、`cpu_references`
与当前含裁剪的锚点数。`selected_value` 是原始树值，`decision_value` 是根幅度约束后的值，
`decision_score` 还包含策略先验。

设计参考 Stockfish 18 的
[search.cpp](https://github.com/official-stockfish/Stockfish/blob/sf_18/src/search.cpp) 和
[movepick.cpp](https://github.com/official-stockfish/Stockfish/blob/sf_18/src/movepick.cpp)：
区分静态估计和搜索结果，被将军时检查应手，并优先搜索有价值的强制着法。
本原型独立实现这些原则，没有移植 Stockfish 的置换表、SEE、LMR 或其调参常数。
