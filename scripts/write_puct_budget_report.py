"""Write the completed two-second PUCT baseline report from saved measurements."""
import argparse,json
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True);a=p.parse_args();r=Path(a.root)
    d=json.loads((r/'analysis.json').read_text());initial=d['summary'];final=d['final_validation']['summary'];paired=d['paired'];arena=d['arena'];ar=arena['result']
    names={'puct-512':'原 PUCT-512','fixed-1792':'固定1792次','fixed-2048':'固定2048次','cached-2048':'固定2048次＋缓存','reference-2s':'原逻辑＋2秒控制','optimized-2s':'缓存＋2秒控制'}
    rows=['| 初始32根 | 中位耗时 | P95 | 最慢 | 超过2秒 | 中位模拟数 |','| --- | ---: | ---: | ---: | ---: | ---: |']
    for name,x in initial.items():rows.append(f"| {names[name]} | {x['median_seconds']:.3f}s | {x['p95_seconds']:.3f}s | {x['max_seconds']:.3f}s | {x['over_2s']}/{x['samples']} | {x['simulations_median']:.0f} |")
    names2={'puct-512':'原 PUCT-512','reference-2s':'原逻辑2秒','cached-2s':'仅缓存2秒','guard-2s':'仅批量保护2秒','optimized-2s':'最终组合2秒'}
    confirms=['| 新16根 | 模拟数中位数 | P95耗时 | 最大耗时 | 平均CP损失 | ≥100cp失误 |','| --- | ---: | ---: | ---: | ---: | ---: |']
    for name,x in final.items():confirms.append(f"| {names2[name]} | {x['simulations_median']:.0f} | {x['p95_seconds']:.3f}s | {x['max_seconds']:.3f}s | {x['mean_cp_regret']:.2f} | {x['cp_blunders_100']} |")
    tim=arena['timings'];lo,hi=paired['time_budget_simulation_gain95']
    text=f'''# 本地每步2秒的 PUCT 基线与第一轮优化

建议使用 [`configs/puct-2s-local.json`](../configs/puct-2s-local.json)：**约2秒时间控制、batch64、32768条有效输入缓存、子树复用、批量等待保护**。
在本机RTX4060Laptop8GB及同一冻结模型上，最终新16根的中位模拟数约 **{final['optimized-2s']['simulations_median']:.0f}次**，
P95耗时 **{final['optimized-2s']['p95_seconds']:.3f}秒**，最慢 **{final['optimized-2s']['max_seconds']:.3f}秒**。
不建议把4096次设为2秒默认，它在试验中约需4秒。缓存版固定2048次也是更保守的固定次数档位；原版不加缓存时，1792次更稳妥。

## 搜索次数可以拉到多少

这里的次数是每步追加的模拟/回传数，不是树深度。根神经评估不计为一次追加模拟。
模拟数包含终局回传和缓存命中，不等于独立神经观察数，更不等于独立随机样本量。
完整单局延迟包括CPU树操作、传输、神经推理、头融合和结果校验；模型加载及预热另计。

4个旧根的初筛覆盖512/1024/2048/3072/4096次及batch32/64/128/256，batch64最佳：
512次约0.51秒、1024次约1.00秒、2048次约1.99秒、3072次约2.97秒、4096次约3.96秒。
加大batch到128或256没有收益。初筛不是尾延迟保证，新的32根验证表明固定2048次实际经常超过2秒：

{chr(10).join(rows)}

在初始32根中，固定2048次开关缓存的选招 **{paired['fixed2048_move_parity']}/32一致**，
配对平均耗时降低 **{100*paired['fixed2048_time_reduction_mean']:.1f}%**。
同样约2秒，30个两种设置均耗尽时间预算的根，模拟数配对平均增加 **{100*paired['time_budget_simulation_gain_mean']:.1f}%**，
按根bootstrap95%约[{100*lo:.1f}%,{100*hi:.1f}%]。这说明吞吐改善，不直接证明棋力提高。
32根有1根含将杀标签，CP比较只用其余31根；同组老师候选并集、100k提议＋1M节点MultiPV，有限搜索并非真值。

![搜索次数、尾延迟与同时间吞吐](../{r}/latency-budget.png)

## 已实施的优化

**输入缓存。** 按棋子编码、模型真正使用的元数据、完整有序合法动作ID缓存策略/Q/价值输出，
并在一个批次内去重。history-free模型仅掩掉自身确实忽略的字段。规则状态和访问统计仍分离；
重复局面、50步规则、易位和EP等规则判断始终由原生棋树处理，不把网络输出当成胜负证明。
缓存属于单个玩家，换模型/推理配置需调用`clear_cache()`并丢弃旧树。这里没有减少模型精度或蒸馏新网络。

**时间控制。** 在完整推理批次结束后检查剩余时间，预留最近观察到的最大批次耗时乘1.15及至少30ms，
正常完成根评估后才可停；不遗留pending节点或virtual loss。16384只是模拟数安全上限，不是目标搜索量。
时间模式按已用时间比例触发根剪枝，保留原0.75比例，避免高模拟上限导致实际2秒内一直不剪枝。
时间模式目前仅支持一个棋局槽位。GPU批次和操作系统调度不可抢占，所以这是近似时间控制；
本次样本未超时，不代表任意负载下的硬实时保证。

**批量等待保护。** 查出一个真实错误：`select`先排除了等待推理的可行子节点，再判断有无安全候选；
于是所有安全叶子都在pending时，会错误地退回已知败着，快速反复回传，烧完模拟预算才提交部分推理批次。
修复先判断是否存在可行候选，再排除pending；如全部可行候选都在等待，则提交部分批次继续推理。

诊断局面`2k4r/3bbp1r/1qp1p3/p3Pn2/3P4/PpR2QB1/1P2N1P1/1B1R2K1 w - - 1 29`中，
原逻辑仅38个神经输入，却做了16384次模拟，其中16347次为终局回传，耗时仅约45ms。
原512次、原逻辑2秒和只加缓存都选`b1f5`；修复后选`g1f2`。同一有限老师候选比较的损失从184cp降到0。
这不能算“每秒几十万次有效搜索”；是调度缺陷。合成原生测试和CPU假网络回归均能捕获，真实模型诊断单独保存。
该32根因参与发现错误，之后只作为这个修复的开发组；另取了下面16个新根确认。

CUDA Graph也实测过。它通过固定形状重放减少调用开销，约束见[PyTorch CUDA文档](https://docs.pytorch.org/docs/stable/notes/cuda.html)。
本机该模型没有稳定速度收益，因此未在基线启用。其独立实验实现保留，GPU输出一致性测试通过。

## 新组确认与棋力边界

初始32根来自源棋局99587–99909；修复后的新16根来自99929–100048，实战开局来自100504–100513。
每组按源棋局分离，并排除旧base精确键；原冻结大模型完整源棋局重叠仍未知。
新16根包含原512、2秒对照、只加缓存、只加批量保护和最终组合，全部共享冻结权重及相同老师候选并集。

{chr(10).join(confirms)}

本组所有候选均有CP分数。相比原逻辑2秒，最终组合的平均损失变化为
{d['final_validation']['paired_cp_change']:+.2f}cp，配对bootstrap95%区间{d['final_validation']['paired_cp_change95']}。
只有缓存时本组没有改变选招；明确的修复案例及本组改善来自批量调度保护，不能把这些棋力收益全归给更多模拟。
16根很少，不能宣称已解决所有漏算或给出普遍棋力结论。

## 完整对局检查

在4个新近均衡开局上交换颜色，最终组合2秒对原PUCT-512，共8局：
**{ar['wins']}胜、{ar['draws']}和、{ar['losses']}负**，全部完成并通过独立PGN重放/结果/预算审计。
这是更高计算预算对原约0.5秒配置的检查，不是等时间算法对弈，也不是可靠Elo估计；4个开局对不足以精确量化差距。
此前缓存单项的1盘完整记录及一次中断尝试因发现批量缺陷单独归档，正式8局全部在同一冻结开局组以最终版本重跑，没有混合版本。

最终组合在这8局共{tim['optimized-2s']['choices']}步选择，P95耗时{tim['optimized-2s']['p95_seconds']:.3f}秒，
最长{tim['optimized-2s']['max_seconds']:.3f}秒，超过2秒{tim['optimized-2s']['over_2s']}步；
每步平均实际大模型输入{tim['optimized-2s']['mean_inputs']:.1f}，对照{tim['puct-512']['mean_inputs']:.1f}。
含提前证明与重复终局的对局中，不能只比较平均模拟数或神经输入数来估计算法质量。

## 使用与后续对照

```python
import json
from pathlib import Path
from neurochess.native_mcts import NativeMCTSPlayer
options = json.loads(Path("configs/puct-2s-local.json").read_text())
player = NativeMCTSPlayer(network, **options)  # 原有MultiheadPlayer网络
move = player.choose(board)
```

使用通用MCTS参数的CLI也可传`--search-config configs/puct-2s-local.json`，配置覆盖单项搜索参数。
原始固定次数、无缓存、无批量保护的对照可继续复现；新基准由显式配置启用。
后续优化与这份2秒配置做同时间对照，同时报告延迟分位、真实推理输入、终局/缓存回传及选招质量，不能仅追求计数变大。
新的模型检查点和其他GPU应重新测量固定次数档位，时间控制本身会自适应吞吐变化。

完整测试 **174 passed, 1 skipped**（CPU环境跳过CUDA专用项），该CUDA输出一致性测试另在本地GPU **1 passed**；
原生UBSan回传、证明、perft、时间剪枝及pending安全测试通过。详细日志位于`{r}`。
该目录还保存模型/源码/二进制指纹、冻结协议、逐根结果、诊断案例和逐局PGN。
机器汇总为`analysis.json`，可读配置为`configs/puct-2s-local.json`。
'''
    Path(a.out).write_text(text)

if __name__=='__main__':main()
