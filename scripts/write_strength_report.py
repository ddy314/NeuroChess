"""Render the completed causal search experiment from its saved evidence."""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True);a=p.parse_args();r=Path(a.root)
    analysis=json.loads((r/'analysis.json').read_text());games=json.loads((r/'duel-analysis.json').read_text())
    assert games['total_completed_games']==160
    assert all(x['audit'] and x['truncated']==0 for x in games['summary'].values())
    fixed_warm=analysis['reuse-fixed-warm']
    stability=json.loads((r/'label-stability.json').read_text())['summary']
    capacity=analysis['capacity'];small=capacity['small'];large=capacity['large'];t=analysis['trace']['summary']
    names={'legacy':'原版','reuse':'仅开子树复用（冷树无效）','material':'纯子力评估','neutral':'零评估','visit-mean':'CPU 访问加权回传','evidence':'仅证据 Minimax','coverage-8':'仅根候选覆盖','coverage-evidence':'根覆盖＋证据回传','larger-student':'较大学生','puct-31':'PUCT-31','puct-512':'PUCT-512'}
    static=['| 方法 | 平均 CP 损失 | ≥100 cp 失误 | 相对原版差值及配对 bootstrap 95% 区间 |','| --- | ---: | ---: | --- |']
    for n,v in analysis['heldout'].items():
        lo,hi=v['delta_cp_bootstrap95'];static.append(f"| {names[n]} | {v['mean_cp_regret']:.2f} | {v['blunders_100']} | {v['delta_cp_vs_legacy']:+.2f} [{lo:+.2f}, {hi:+.2f}] |")
    labels={'candidate-duel':'根覆盖＋证据回传','baseline-duel':'原版','reuse-duel':'仅开子树复用','puct-reuse-control':'原版','coverage-only-duel':'仅根候选覆盖'}
    match=['| 非 PUCT 配置 | 对手 | 局数 | 胜 / 和 / 负 | 得分率 | 条件相对 Elo（bootstrap 95%） |','| --- | --- | ---: | --- | ---: | --- |']
    inference=['| 对局配置 / 对手 | 非 PUCT 实际大模型输入 / 步 | 对手实际输入 / 步 | 非 PUCT 复用比例 | 对手复用比例 |','| --- | ---: | ---: | ---: | ---: |']
    def endpoint(x):return '无界' if x is None else f'{x:.0f}'
    tactical_table=['| 方法 | 保留老师将杀线 / 16 | 可避免将杀误选 / 15 |','| --- | ---: | ---: |']
    for name,v in analysis['tactical-regression']['summary'].items():tactical_table.append(f"| {names[name]} | {v['winning_mate_choices']} | {v['avoidable_mate_errors']} |")
    for key,v in games['summary'].items():
        folder,opponent=key.split('/');elo=v['relative_elo'];ci=v['relative_elo_bootstrap95'];match.append(f"| {labels[folder]} | {opponent} | {v['completed']} | {v['wins']} / {v['draws']} / {v['losses']} | {100*v['paired_score']:.2f}% | {endpoint(elo)} [{endpoint(ci[0])}, {endpoint(ci[1])}] |")
        contender=next(n for n in v['players'] if n!=opponent);p1,p2=v['players'][contender],v['players'][opponent]
        inference.append(f"| {labels[folder]} / {opponent} | {p1['positions_per_choice']:.2f} | {p2['positions_per_choice']:.2f} | {100*p1['reuse_fraction']:.1f}% | {100*p2['reuse_fraction']:.1f}% |")
    contrasts=[]
    for key,v in games['paired_contrasts'].items():
        mean=v.get('mean_score_change',v.get('mean_contender_score_change'));lo,hi=v['bootstrap95']
        contrasts.append(f"- `{key}`：同 {v['common_opening_pairs']} 个开局对，非 PUCT 得分差 {100*mean:+.2f} 个百分点，配对 bootstrap 95% [{100*lo:+.2f}, {100*hi:+.2f}]。")
    text=f'''# 非 PUCT 棋力差距：复用、回传与学生评估消融

这轮没有继续压缩搜索。完成旧失败局面的可复现诊断、容量控制、新 64 根留出评测和 **160 局换色直接对局**。
本轮结果应按下列静态、实战及共同开局对的统计分别判断；不把开发集的小幅改善当成棋力升级。
原版默认配置保留，实验控制均显式启用。原 PUCT 的默认搜索逻辑不变。
本轮未找到稳定超过PUCT-31的改动。已经定位校正分支覆盖不足及直接评估错排，未发现本次检查范围内的回传符号错误；
更大容量、单独复用、均值/证据回传都没有消除差距。仅根覆盖对PUCT-512有局部改善，对PUCT-31则未改善。

## 已定位的机制与未能支持的解释

旧原预算版对 PUCT-31 对局中，按顺序取前 16 盘、逐步用有限 Stockfish 搜索定位首次 ≥100 cp 失误；
没有达到门槛的棋局取已检查的最大损失处。保留完整历史重放，原版 **{t['baseline_move_parity']}/16** 选着与原棋谱一致。
这些是经过挑选的失败案例，不是随机样本，下面的计数不能外推为全部失误的成因占比。

- **校正预算集中有直接证据。** {t['best_without_direct_correction']}/16 个老师优选候选没有直接校正。
  每个诊断根的最大分支平均占用 {100*t['mean_largest_branch_query_share']:.1f}% 的根以下神经观察；多个错误分支得到 25–28 个观察，
  更好的候选仅访问一次、没有直接校正。`root_coverage=8` 让这 6 个漏校正候选全部获得校正，机制缺口已关闭，但不保证正确选着。
- **直接神经校正也会排错。** {t['harmful_direct_flips']}/16 例中，原始学生对所比较的两招排序正确，直接神经值却相反。这是同树原始值的排序比较，不是唯一成因的因果分解。
  例如旧 case5，`e5d7` / `e5g4` 的老师分数为 +141 / -95 cp；根视角学生值为 +0.858 / +0.100，
  直接神经值却为 +0.409 / +0.774，最后仍选坏招。把“已校正”当成“更可靠”会丢掉正确的便宜信号。
- **未发现回传符号错误。** 原生/单测核对 negamax 的逐层符号、持久锚点公式、精确终局优先级和神经均值的符号/去重复计数。
  在这 16 个案例、两招都有直接校正且可比较的子集中，记录到 {t['harmful_backup_flips']} 次“直接值原本排对、随后回传排反”。
  这不证明回传永远无害；原回传仍可能放大有噪声的便宜极值并影响后续查询位置。单换均值或只保留神经证据没有稳定改善。
- **复用的效果必须动态评测。** 冷树下开关复用必然相同。沿旧棋谱预热重放时，仅开 BF 复用平均损失从 187.69 到 186.50 cp，
  PUCT-31 复用开 / 关约 104.56 / 107.12 cp。这一初版BF复用重放早于下述修复，不能当作最终复用证据。
  修复后，同16条旧轨迹重新对照的原版 / 复用平均损失为 {fixed_warm['legacy']['mean_cp_regret']:.2f} / {fixed_warm['reuse']['mean_cp_regret']:.2f} cp；
  两次候选集合不同，CP绝对值不跨轮直接相减。选择过的失败案例不能给出棋力归因，新的完整对局另列。
- **容量不是已证实的单一瓶颈。** 相同按棋局划分数据、CP＋排序监督、训练种子、20,000 步 GPU 训练和相同验证选择规则，
  参数从 {small['parameters']:,} 增至 {large['parameters']:,}（包含训练用 CP 辅助头），验证排序准确率从
  {100*small['best']['validation']['rank_accuracy']:.2f}% 到 {100*large['best']['validation']['rank_accuracy']:.2f}%，
  WDL MSE 从 {small['best']['validation']['wdl_mse']:.4f} 到 {large['best']['validation']['wdl_mse']:.4f}，没有实质下降。
  较大模型更贴合训练集，持续训练后验证误差上升；本轮新搜索留出表现也没有更好。这是一组种子的容量控制，不能据此断言其他规模永远无效。

**监督标签也有可测量的波动。** 随机冻结96个验证组，用500k节点重评相同兄弟候选，原标签来自20k节点MultiPV。
在{stability['cp_states']}个双方有CP的候选中，评分平均绝对变化{stability['mean_abs_cp_change']:.2f} cp，中位数{stability['median_abs_cp_change']:.0f} cp；
{stability['cp_groups']}个双方全有CP的组中，{stability['best_child_changed']}组最佳候选改变。
原标签差至少30cp的{stability['old_gap30_pairs']}对中，{stability['pair_flips']}对换序；两次差值都至少30cp的
{stability['robust_gap30_both_pairs']}对中，{stability['robust_pair_flips']}对换序。更大预算且限制同一候选集合仍不是绝对真值，
这只是标签稳定性检查。没有用重评标签重训或调参，因而不能断言标签波动解释了全部剩余差距。
学生验证排序从初始子力基线约49.8%提高到69.8%，说明它学到了信号；但较高的两两排序准确率并不保证最佳着法损失或实战棋力更好。

![同数据、同训练流程的容量对照](../{r}/capacity-validation.png)

## 消融实现及固定预算

非 PUCT 全部保留 512 次 CPU 扩展上限、每步大模型请求位置上限 32、batch4、双 CPU 预算各16,384，
深度和节点上限不变。小/大模型冻结；容量控制只更换学生，原版学生不被覆盖。PUCT-31 保留原31次附加模拟，
PUCT-512 保留原512次模拟，是更高推理预算对照。

`tree_reuse=True` 只按初始 FEN＋完整历史前缀收养实际走入的子树，重映射父子下标、重置深度并重新开放非证明截断节点。
不会合并仅 FEN 相同的不同重复历史。清理模型缓存也清理保留树；新一步仍有完整新扩展和查询预算。

`visit-mean` 用 CPU 访问数加权当前子节点值，不是 MCTS 样本回传。`neural-mean` 对每个首次直接查询的树节点沿路径交替符号计数，
支持的父 Q 加两份伪观察；计数不是独立样本量。`evidence-minimax` 让神经值/Q/证明形成被回传的边界，缺乏时回退便宜值，
最终优先这些候选。所有模式仍仅由真实终局及完整合法子节点证明胜负；近似值绝不成为证明。
`root_coverage=8` 在根策略前八候选各有直接校正后再恢复原差距调度，不增加推理上限。

新增复用实现还修复了两处具体错误：CPU根部参考刷新时，应对已展开的保留子节点重新回传，不能当成叶子覆盖；
树中“已证明胜负”也不等于棋盘“已经终局”，保留证明的未终局根仍须合法走子。
两项回归均在对应旧二进制失败、修复后二进制通过。后一错误曾中断复用组，4盘先前完整记录和中断尝试单独归档，
不混入正式结果；正式16盘全部以最终修复版从头重跑。它们是新增复用路径的缺陷，不是此前无复用原版棋力差距的原因。
各组自己的二进制/源码指纹保留；`source-final`是最初冻结版，`source-reuse-final`是最终复用修复版。

纯子力和零评估控制同时替换普通便宜评估和 CPU 价值搜索，保留精确将杀逻辑，避免“关学生但仍偷偷用于 CPU 校正”的混淆。
原学生辅助 CP 头导出也做过旧开发比较，没有稳定改善，未采用。

## 新留出结果

64 根来自源棋局 98,920–99,584；冻结的32个开局来自98,265–98,903，本轮实战使用前16个开局，
换色及多个方法共享开局，不能把160局当作160个独立开局。静态根与开局彼此棋局分离，
与新学生训练/验证及旧开发根的源棋局和精确键无交叉，旧500万数据精确键排除。
原大模型完整源棋局重叠仍未知，不能声称完全棋局独立于大模型。
统一100k老师提议＋1M节点候选并集 MultiPV；各方法轮换顺序，共用冻结大模型，冷树评测。
64根全部有 CP 标签；这里没有新短将杀留出，不能将CP误差等同于将杀解出率。

{chr(10).join(static)}

原版与PUCT-31的静态差值区间跨零。仅根覆盖比原版约好0.53 cp，区间跨零；
根覆盖＋证据回传反而差约13.14 cp。较大学生和关闭学生都未改善本批结果。
这批冷树分数不能推导完整对局棋力，尤其不测跨步复用。

![新留出配对损失差](../{r}/heldout-ablations.png)

## 已知战术组回归

旧32个FEN根（16个进攻将杀、16个防守）只是开发回归，未提供整盘历史，不作新留出证据。
统一候选并集的有限老师评测下，1个防守根所有比较候选均显示被将杀，不能计为“可避免漏防”，
其余15个才作为可避免将杀误选的分母。

{chr(10).join(tactical_table)}

非PUCT在14个进攻根由独立CPU证明后直接选招，无需大模型输入，三种设置相同。
组合方案虽在这些旧防守根从3错减为2错，却在新CP根和完整实战退步；仅根覆盖在这里从3错变4错。
因此，本轮不能宣称新改动已真正、稳定地减少战术漏算。

## 实际对弈

采用新的近均衡开局（源棋局每局一个、ply16–24、SF50k绝对值≤100 cp），交换双方颜色。
主要组合候选在查看新结果前由48旧根开发规则冻结；原版、仅复用有独立控制。
在组合方案静态退步后，补登记仅根覆盖的完整对局以分离两项变化，标为探索性跟进，未重调参数。
另关闭PUCT复用做反向控制。所有对局完整结束，没有把截断算和棋，没有在线Stockfish干预。

{chr(10).join(match)}

Elo只是对这些模型、预算、开局分布的条件等效值，不是标准绝对等级分。
按开局对而非单局 bootstrap，并假设这些开局对可视为独立样本；8对开局区间很宽，存在0/1得分和退化重采样时应查看JSON中的Hoeffding保守界。
共同开局上两组均0分时，差值bootstrap会退化为[0,0]；这不是两种搜索等强的证据。
比较不同非PUCT配置时，只比较共同开局对；主要候选全16对得分不能直接与原版8对当成同分布配对差：

{chr(10).join(contrasts)}

{chr(10).join(inference)}

相同预算指32个请求位置的硬上限，缓存、保留根及提前证明可减少实际输入，不是强迫消耗完全相同的输入数。
同时报告真实输入数；PUCT-512预算更高。
本轮原版对PUCT-31每步平均28.81次内部校正请求、25.43个实际输入，约3.37次缓存命中仍扣内部请求配额；
对手实际输入28.36。缓存配额利用是另一个实现差异，本轮未把两者强行补足到同样实际输入数，不能把它的贡献与其他因素精确拆分。时间记录可审计，但这轮没有做继续压缩或等时间棋力的主张。
旧开发的head/capacity/历史重放任务曾共用GPU，其耗时不作速度比较；最终留出和对局GPU任务串行执行。
训练标签/UBSan等CPU审计可与对局重叠，因此本轮耗时记录同样不用于宣称算法速度差。

## 采用状态、验证和可复现资料

本轮保留原版默认行为，新增复用、评估和回传开关作为可复现实验控制；不把未稳定改善的证据回传或较大学生设为默认升级。
单纯“多训、更大、把便宜值全丢掉”没有解决本批问题。已有证据指向评估失真与校正预算分配的耦合，
接下来若继续优化，应在新的开发失败轨迹上检验校正可靠性与分支覆盖，保留未来独立实战组验证，不能继续对本组调参再称留出提升。

完整测试 **158 passed**；独立AVX2 UBSan学生/搜索测试 **56 passed**。原版48个旧根选着完全一致，16盘失败重放完全复现；最终复用修复后二进制对64个新冷根的原版选着也64/64一致。
新增测试覆盖历史身份、复用后深度截断恢复、清缓存失效、独立新预算、神经均值视角和不重复计数、根覆盖、CP头导出及PUCT关闭复用的累计统计。
160局均通过独立PGN合法重放、初始FEN、颜色、终局、结果和累计推理上限检查。
旧32战术根回归见`tactical-regression/`；属于已知开发样本，单独计数而非新留出证据。

机器证据目录：`{r}`。主要文件为`protocol.json`、各补充protocol、`analysis.json`、`duel-analysis.json`、
`holdout-audit.json`、`final-suite.json`、`source-final-sha256.json`，各评测/对局目录含模型/原生二进制指纹、逐局JSON与PGN及审计。
源码/对局指纹在实验前保存，旧warm replay没有自己的内嵌指纹，使用同轮development冻结二进制及默认48/48、16/16复现核对；
其C++与Python桥接源码已通过逆转后续补丁重建，SHA256与实验开始时记录完全一致，保存在`source-first-ablation-reconstructed/`，
明确标注为哈希核对后的重建副本。最终复用重放另存完整指纹，初版资料限制保留说明。
'''
    Path(a.out).write_text(text)

if __name__=='__main__':main()
