"""Package reviewed, measured checkpoint evidence for the local report app."""
import json,time
from pathlib import Path

OUT=Path(__file__).resolve().parents[1]/'output/joint-deep-analysis-20261103'
def read(name):return json.loads((OUT/name).read_text())
def lines(name):return [json.loads(l) for l in (OUT/name).read_text().splitlines()]
def num(x):return f'{x:.4f}'
def interval(m):return f"{m['delta_final_minus_parent']:+.4f} [{m['ci95'][0]:+.4f}, {m['ci95'][1]:+.4f}]"


def main():
    raw=read('raw-analysis.json');search=read('search-assessment.json');exposure=read('checkpoint-and-exposure.json');sensitivity=read('sensitivity-analysis.json')
    drift=read('teacher-drift-and-overlap.json');mechanisms=read('mechanisms-cpu.json');features=read('representation-and-gradient.json')
    new=lines('artifacts/training/validation.jsonl');old=lines('artifacts/training/general-validation.jsonl')
    queries={}
    def query(key,rows,files,definition,component_ids):
        queries[key]={'rows':rows,'source':{'label':'实际训练记录与本地冻结权重复核','files':[str(OUT/f) for f in files],
            'filters':['父模型 cumulative step 3000；最终模型 cumulative step 13000；本轮 10000 次更新'],
            'metricDefinitions':[{'label':key,'definition':definition,'componentIds':component_ids}],
            'caveats':['新数据置信区间按来源棋局成组配对 bootstrap 2000 次；旧数据缺少历史棋局 ID，只能按局面重采样。','数值是 Stockfish 教师代理目标的误差，不是实战胜率或 Elo。']}}
    metrics=[]
    for name,label in [('new_test','独立新测试 3k / 125 棋局'),('new_val','新验证 3k / 132 棋局'),('old_val','旧验证 4,096 局面')]:
        for field,title in [('policy_top1','Policy Top-1 ↑'),('q_mae','Q MAE ↓'),('value_mae','Value MAE ↓')]:
            m=raw['datasets'][name]['metrics'][field]
            metrics.append(dict(dataset=label,metric=title,parent=m['parent'],final=m['final'],delta=m['delta_final_minus_parent'],ci=m['ci95'],previous=num(m['parent']),latest=num(m['final']),change=interval(m)))
    query('metrics',metrics,['raw-analysis.json'],'Top-1 对原教师最优标签；Q MAE 仅在原 qmask 上按动作数加权；Value MAE=|P(win)-P(loss)-教师值|。差值=最终-父模型；区间为配对 95% CI。',['supervised','summary'])
    curves=[]
    for n,o in zip(new,old):curves.append(dict(step=n['step'],new_loss=n['val_loss'],old_loss=o['val_loss'],new_reduction=new[0]['val_loss']-n['val_loss'],old_reduction=old[0]['val_loss']-o['val_loss'],gainShare=(new[0]['val_loss']-n['val_loss'])/(new[0]['val_loss']-new[-1]['val_loss']),**{k:o['value_strata'][k]['mae'] for k in ['winning','neutral','losing']}))
    query('curves',curves,['artifacts/training/validation.jsonl','artifacts/training/general-validation.jsonl'],'固定验证集每 1000 步测量。总 loss = policy CE + 0.5 Q MSE + value CE。图显示相对第 0 步的 loss 减少量，正数为改善。',['loss-curve','plateau','old-strata-curve'])
    strata=[]
    for name,label in [('old_val','旧验证'),('new_test','独立新测试')]:
        for k,title in [('winning','优势 >0.25'),('neutral','均势 ±0.25'),('losing','劣势 <-0.25')]:
            m=raw['datasets'][name]['value_strata'][k]
            strata.append(dict(dataset=label,stratum=title,n=m['n'],previous=num(m['mae']['parent']),latest=num(m['mae']['final']),change=interval(m),parent_bias=m['bias']['parent'],final_bias=m['bias']['final']))
    query('strata',strata,['raw-analysis.json'],'以教师 W-L 值分层；正 bias 是模型更乐观，负 bias 是更悲观。分层为预先指定探索性诊断，未做多重检验校正。',['value-strata','value-diagnosis'])
    rootrows=[]
    for mode,label in [('direct','无搜索，多头融合'),('locked2s','冻结 PUCT，两秒')]:
        for subset,title in [('all','全部 96 局面'),('teacher_stable','教师稳定 51 局面')]:
            x=search['primary'][mode+'_'+subset];m=x['cp_regret_capped1000']
            rootrows.append(dict(mode=label,subset=title,parent=m['parent'],final=m['final'],previous=f"{m['parent']:.2f}",latest=f"{m['final']:.2f}",change=interval(m),blunders=f"{int(round(x['policy_error']['parent']*x['n']))} → {int(round(x['policy_error']['final']*x['n']))}",n=x['n']))
    query('search',rootrows,['search-assessment.json','search-protocol.json','search-decisions.jsonl','deep-teacher.jsonl'],'每个局面取自不同来源棋局，教师优势/均势/劣势各 32；选取先于预测。regret=教师候选最优 CP-模型所选动作 CP，截断 1000；丢失强制胜势/陷入强制败势按 1000 计。100cp 以上记大错；不同赢棋杀棋长度不算失误。',['search-results','search-comparison','search-analysis'])
    swaps=[dict(combination=k,regret=v['mean_cp_regret'],bad=v['policy_error_fraction'],n=v['n']) for k,v in search['head_swaps'].items()]
    query('swaps',swaps,['search-assessment.json','decision-teacher-details.json'],'三位依次 P/Q/V：0 旧、1 新。24 个盲选棋局，完整 2×2×2 交换输出（包含各分支编码器），每格两套网络都前向，固定 512 次模拟以控制计算差异。仅机制诊断，不替代冻结两秒搜索。',['head-swaps','head-analysis'])
    future=[]
    for depth in [1,2,4]:
        for stable in [False,True]:
            m=sensitivity['future_sensitivity'][f'depth={depth},stable={stable}']
            future.append(dict(depth=depth,subset='稳定教师' if stable else '全部',n=m['n'],games=m['clusters'],parent=m['parent'],final=m['final'],previous=num(m['parent']),latest=num(m['final']),change=interval(m)))
    query('future',future,['future-value-details.json','sensitivity-analysis.json','teacher-drift-and-overlap.json'],'从 24 个盲选根局面沿原教师 PV、父模型直接走子、最终模型直接走子扩展到 1/2/4 ply；每个后继独立重标注。原始 143 个保留状态，排除与旧训练集重合的 2 个；与新训练集重合 0。按根棋局成组，不将后继当独立棋局。不是实际 MCTS 叶节点样本。',['future-value','future-analysis'])
    pools=[]
    for key,label,size in [('old','旧回放池',5000000),('new','新局面池',472576),('mistake','固定错题额外槽',15761)]:
        e=exposure['exact_exposure'][key];pools.append(dict(pool=label,pool_size=size,draws=e['draws'],unique=e['unique'],never_seen=e['never_seen'],repeats=e['mean_repeats'],display_repeats=f"{e['mean_repeats']:.2f}×",max_repeats=e['max_repeats']))
    query('exposure',pools,['checkpoint-and-exposure.json'],'完整重放所有采样调用，最终 NumPy RNG 状态与最终 checkpoint 精确一致。错题也是新池子集，因此额外 40.61 次不包含其在新池槽中的抽样。未访问仅指本轮，不能推出模型历史从未见过。',['exposure-table','exposure-analysis'])
    qrows=[]
    for labelled in [True,False]:
        for stable in [False,True]:
            m=sensitivity['q_error_paired'][f'labelled={labelled},stable={stable},support_union=True']
            qrows.append(dict(candidate='原已标注' if labelled else '原未标注',subset='教师稳定' if stable else '全部',n=m['n_candidates'],games=m['clusters'],previous=num(m['parent']),latest=num(m['final']),change=interval(m)))
    query('qcoverage',qrows,['data-audit.json','raw-analysis.json','sensitivity-analysis.json','q-deep-teacher-errors.json'],'在两模型 inference support 并集内保持固定候选分母，重新用更强教师监督。原未标注指该局面的原 qmask=False，不表示动作全局未训练过。相邻动作按来源棋局成组。',['q-coverage','q-analysis'])
    mechrows=[]
    for branch,label in [('policy_model','P/Q 编码器'),('value_model','Value 编码器')]:
        for layer in ['0','7','15']:
            f=features['features'][branch][layer];mechrows.append(dict(branch=label,layer=int(layer),cka=f['linear_cka'],cosine=f['coordinate_cosine']))
    query('representation',mechrows,['representation-and-gradient.json','weight-movement-summary.json'],'固定 256 个新测试局面，取 block 0/7/15 的均值池化特征，计算中心化线性 CKA 与逐坐标 cosine；不是棋力或容量上限证据。',['representation-table','mechanism-analysis'])
    teaching=[dict(item='根局面复核',count=96),dict(item='最终达到 30cp 全候选稳定条件',count=search['stable_roots']),dict(item='追加 3M nodes 复核',count=search['teacher_refined_roots']),dict(item='原教师值与深教师值相差 ≥0.25',count=drift['value_vs_deep_reference']['all']['target_shift_ge025'])]
    query('teacher',teaching,['deep-teacher.jsonl','teacher-drift-and-overlap.json'],'先 unrestricted 1M nodes MultiPV4，再对所有模型选步及策略支持候选的共同集合独立做 100k / 1M nodes；不稳定追加 3M。节点预算按整个 MultiPV 搜索，不是每动作。严格稳定要求全部候选 CP 变化≤30、mate 方向一致；不稳定不等于全部标签错误。',['teacher-table','teacher-analysis'])
    run=[dict(item='父模型累计步数',value='3,000'),dict(item='最终模型累计步数',value='13,000'),dict(item='本轮有效 batch / 更新数',value='320 / 10,000'),dict(item='训练配比（旧 / 新 / 额外错题）',value='128 / 128 / 64'),dict(item='续训前后 microbatch × 累积',value='16×20 → 64×5（第 7,000 步精确续训）'),dict(item='学习率',value='前 7k：1e-6；后 3k：cosine 到 3e-7'),dict(item='记录的训练累计耗时',value='17,132.97 秒（4.76 小时；含记录内验证/保存）'),dict(item='后 3k 实际累计吞吐',value='210.08 positions/s（含记录内验证/保存）'),dict(item='参数规模 / 编码器',value='310,001,162 / P+Q 与 V 分开'),dict(item='云端实例',value='已释放；本报告未启动新云端训练')]
    query('run',run,['artifacts/training/run.json','artifacts/training/metrics.jsonl','artifacts/training/resume-verification.json','checkpoint-and-exposure.json'],'step 13000=父模型 3000+本轮 10000；父模型不是随机初始化。优化器在本轮初始重置，第 7000 步优化时精确保留 model/Adam/RNG。',['run-table','methods'])
    cases=read('illustrative-cases.json')
    query('cases',cases,['illustrative-cases.json','deep-teacher.jsonl','decision-teacher-details.json'],'事后选择稳定教师根中搜索 regret 改善最多和退步最多的两个案例；只作机制示例，不代表总体发生率。CP regret 是共同候选集下的教师选步遗憾值。完整 FEN 和历史保留在来源行。',['case-analysis','case-table'])
    cloud=read('cloud-evaluation-summary.json')
    query('cloud', [{'mode':k,**v} for k,v in cloud.items()],['cloud-evaluation-summary.json','artifacts/training/fixed-position-evaluation.json'],'云端结束时 16 个不同棋局根、每根 3 次重复；先按棋局取重复均值，再按 16 棋局 bootstrap。教师100k/300k，L4 两秒搜索。与本地96根协议/硬件不同，不混合汇总。',['search-analysis'])
    sections=[]
    def section(id,title,body,q):sections.append(dict(id=id,title=title,body=f'## {title}\n\n{body}',queryIds=q))
    section('summary','这轮学到了东西，但不足以支持按原样继续拉长训练',
      '**监督泛化成立。** 独立新测试集 3k 局面中，Policy Top-1 从 38.53% 到 40.60%，Q MAE 从 0.2668 到 0.2349，Value MAE 从 0.2312 到 0.2166；按 125 个来源棋局计算的配对区间均支持改善。\n\n'
      '**搜索收益仍不确定。** 96 个盲选棋局局面上，两秒 PUCT 的平均截断 regret 为 37.80→32.34cp，差值 95% CI 为 [-20.28,+10.15]cp；严格稳定教师子集上为 17.63→16.12cp。当前证据既不足以证明搜索提升，也不支持“这一轮搜索明确退步”。\n\n'
      '**边际收益明显下降。** 首 1k 更新完成全程新验证 loss 降幅的 85.4%，末 3k 只贡献 1.7%；固定错题已被额外重复 40.6 次。建议结束这批静态数据的长训，保留联合蒸馏框架，把下一轮重点改成动态刷新错题、补充学生后继/搜索状态，以及分层 value 与搜索门禁。',['metrics','search','curves','exposure'])
    section('supervised','进步能迁移到独立棋局，不能只归为记住了训练错题',
      '表中差值都是最终减父模型；MAE 越小越好，Top-1 越大越好。新测试集与训练按来源棋局隔离，3k 局面仅来自 125 局棋，因此区间按棋局重采样。旧验证 4,096 局面来自原冻结集，历史棋局 ID 不可恢复，其区间可能偏窄。\n\n'
      '在父模型 value 初始误差≥0.25的可比子集内，训练错题 MAE 0.5277→0.3367，独立新测试错题 0.5296→0.4715。训练错题修得远多于未见棋局，但独立集确有改善，不能据此宣布只有记忆。独立错题仍有 79.5% 的最终误差≥0.25，说明问题也没有被解决。筛题过程及教师预算不同，这不是严格的过拟合因果实验。',['metrics'])
    section('plateau','越到后面，每一千步获得的新信息越少',
      '新验证总 loss 2.9316→2.7193。前 1k 降低 0.1812，末 3k 只降低 0.00362；前 3k 已完成 93.8% 的全程降幅。末 3k 的 cosine 退火与吞吐调整同时发生，没有未退火对照，不能把尾段收益全部归功或归咎于学习率。\n\n'
      '按总 loss 的收益衡量，最后 30% 更新只带来 1.7% 的全程改善。这是当前“数据×目标×优化器×步数”的局部结论，不能外推为 3.10 亿参数模型的能力上限，也不能证明增加全新且有效的数据无用。交叉熵本身有非零软标签熵底，但新测试 policy CE 仍高于其目标熵 1.216，不能把平台全解释为已达到数学最优。',['curves'])
    section('exposure-analysis','本轮最严重的重复集中在 1.58 万道固定错题',
      '有效 batch 始终为 320，总采样 3.2M 次；不是 batch=1024，也不是整套旧数据跑了很多轮。精确回放采样后，5M 旧池仅访问 1.129M 个不同局面，新池访问 441k / 473k；固定错题则全部访问，额外槽平均 40.61 次、最多 68 次，另有新池槽中的抽样。\n\n'
      '静态高频错题可以快速修正已有错误，但不随能力变化而更新，会把后续算力越来越多地花在相同监督上。旧池本轮尚未访问 3.871M 不代表这些局面在历史预训练中没见过。提高数据量应优先提高对当前学生有用的、标签可靠的独立局面覆盖，而不只是增大文件行数。',['exposure'])
    section('value-diagnosis','平均 value 改善掩盖了旧优势与劣势局面的持续回缩',
      '旧验证优势局面 MAE 0.2151→0.2270，劣势 0.2669→0.2835；均势 0.1303→0.1137 的收益掩盖了两端退化。优势 bias 更负（更悲观），劣势 bias 更正（更乐观）。这种变化在多个 1k 检查点持续出现，并非只观察第 500 步就下结论。\n\n'
      '新测试集的均势 value 改善也最清晰；优势、劣势差值区间都跨零。模型预测绝对值整体缩小，旧集预测对教师值的线性斜率由 0.765 降至 0.744。这支持“估值向中间收缩”的描述，但无法仅凭这些数据认定是某一损失权重或正则项的因果作用。门禁必须分别检查优势/均势/劣势和偏差，不能只看平均 value loss。',['strata','curves'])
    section('search-analysis','无搜索选步改善更明确，搜索收益没有跨过统计门槛',
      '保留冻结的两秒 PUCT 源码、库和所有配置，逐项 SHA 校验一致；两模型同硬件、独立树和缓存，运行顺序交替。每个根局面只把首次运行计入主要比较。24 根各追加两次，三个重复的最终选步 regret 一致，不能把 3 次重复充成 3 倍独立样本。\n\n'
      '多头无搜索 regret 68.19→46.58cp，全部根的差值 CI [-37.91,-7.66]；严格稳定教师子集 57.94→46.04cp，CI [-33.42,+1.49]。只用 policy 的无搜索 regret 同样下降：62.30→50.77cp。两秒搜索本就帮助父模型较大，新模型的无搜索收益未等比例转化为搜索收益；不能用这 96 个均衡诊断根推算真实对局胜率或 Elo。',['search'])
    sections[-1]['body']+='\n\n云端训练结束时的 48 对记录，实际是 16 个不同棋局根各重复 3 次，并非 48 盘对局。按棋局平均后，云端无搜索 regret 14.31→5.88cp，两秒搜索 13.42→9.81cp，搜索差值 CI [-13.00,+4.88]，同样不足以确认提升。其教师为 100k/300k、硬件为 L4，未与本地更强教师的 96 根混合统计。'
    sections[-1]['queryIds'].append('cloud')
    section('head-analysis','交换八种头组合，没有证实某一个新头是搜索退步的根源',
      '三位依次为 policy、Q、value；0=父模型，1=最终模型。全旧 000 的 fixed512 regret 为 47.83cp，全新 111 为 25.71cp。只换新 policy（100）为 23.08cp；在新 P/Q 上换新 value，110→111 为 23.08→25.71cp，但仅 +2.63cp，不能凭此宣布 value 是元凶。\n\n'
      '对四种其他头背景取平均，新 P、Q、V 的边际效果分别为 -13.28、-3.76、-4.11cp，全部 24 根的区间均跨零；教师稳定 19 根里 policy 效果为 -7.18cp，区间 [-15.03,-0.32]，Q/V 仍不确定。存在组合交互及样本不确定性，不能挑一个最好的交换格直接晋升，也不能把固定 512 模拟的结果当作两秒基线资格证明。',['swaps'])
    section('case-analysis','具体案例：当前局面估得更准，搜索仍可能走得更差',
      '稳定教师案例 `63975a72` 中，教师当前值为 +0.997，父模型估计 +0.9743，新模型 +0.9852，根估值更准确；但旧两秒搜索走 `a3d6`，新搜索走 `f7h5`，教师选步 regret 从 0 变成 118cp。这不能证明 value 导致退步：该根八种头组合在 512 次模拟时全部走 `f7h5`，说明当前交换预算对这个案例没有区分能力。\n\n'
      '另一稳定教师案例 `063c117e`，旧搜索走 `f5d4`、新搜索走教师认可的 `c6c5`，regret 152→0cp，同时根估值也改善。两类案例并存，故不能用某一盘或某一个标量推导所有搜索行为。这里展示的是事后选出的极端例子；完整 FEN、原始历史、来源棋局和教师 PV 在来源数据与同级 `illustrative-cases.json` 中保留。',['cases'])
    section('q-analysis','Q 的监督覆盖确有缺口，但未标注动作也出现了泛化改善',
      '新测试平均约 29.9 个合法动作、3.95 个标注动作，仅覆盖 13.2%。推理时 policy 支持的动作里，原标签覆盖由 57.7% 到 58.6%；原始 Q MAE 看不到其余约 41% 动作在该局面的误差。\n\n'
      '为避免只凭覆盖率猜测，已对两模型支持集合并后重新深标注。原未标注且教师稳定的支持候选，Q MAE 为 0.4279→0.3476，差值 CI [-0.1214,-0.0434]。所以“未标注 Q 全坏了、是退步根因”不符合此次证据；但误差仍大于原已标注支持候选的 0.3463→0.2721，补充这些学生实际会考虑的动作监督仍有价值。',['qcoverage'])
    section('future-analysis','后续几步的泛化尚不稳健，不能从根局面指标代替搜索叶估值',
      '从 24 个盲选棋局扩展教师与两个学生分支，得到 143 个保留后继，独立重新用 Stockfish 标注；剔除 2 个旧训练精确重合后为 141。1/2/4 ply 表示半回合，不是完整回合；这些是受控 rollout 状态，不是实际 MCTS 叶节点总体。\n\n'
      '全部样本中，1 ply 与 4 ply 平均 value 误差下降，2 ply 略增，三个差值区间都跨零。稳定教师子集仅 1 ply 有较明确改善，2/4 ply 仍不确定。一步残差 |Q(s,a)+V(s′)| 从 0.2521→0.2206，教师自身约 0.0347；模型仍明显不一致，但改善区间跨零。有限预算教师与固定 ply WDL 转换也会产生非零残差，不能直接强制模型残差为零。\n\n'
      '因此，现有结果支持把学生后继和搜索前沿纳入下一轮数据，而不支持“新 value 在所有多步局面都变坏了”的说法。真正的搜索叶数据尚需从后续采集流程直接记录，不能把本探针偷换成已完成该验证。',['future'])
    section('teacher-analysis','教师预算与目标表示还有重要盲区',
      '96 根中，67 根需要追加 3M nodes；最后 51 根通过严格的“全候选 CP 变化≤30”条件。这里候选还包含学生支持动作，要求比只看教师最优动作更严，因此不能说另外 45 根都是错标。原标签与深教师 value 平均绝对差 0.0506，4 根差≥0.25；原始监督并非绝对真值，难例复核需要自适应预算。\n\n'
      '当前 value 是 SF16 固定 ply=32 的 WDL 映射代理，不是实际比赛胜率；新测试约 29.7% 的已标注 Q 目标绝对值>0.99，旧验证为 37.7%。这种饱和会压缩已经赢棋的好坏差异。三个 WDL 输出能表示连续 W-L 值，问题不在于“三分类只能输出三种值”；单纯把同一饱和 WDL 目标改成 128 桶，也不会自动恢复丢掉的 CP 排序信息。',['teacher','qcoverage'])
    section('mechanism-analysis','编码器确实更新了，不能用“梯度消失”或三头共享冲突解释一切',
      '模型实际是两个约 155M 参数编码器：policy/Q 共享一套，value 单独一套。P/Q 主干相对 L2 位移 1.07%，value 主干 1.06%；最后层 CKA 分别 0.948、0.972，较早层更稳定。说明并非只改末端、主干完全不动，但 CKA 高也不是容量耗尽证据。\n\n'
      '固定 8 个微批次的 P/Q 梯度探针中，最终模型三层平均 cosine 约 0.047/0.019/0.020，负相关比例 3/8；存在批次冲突，但不足以证明它导致平台。value 没有与 policy 共享主干，不能套用“三头互相争同一主干”的解释。1,000 条记录的裁剪前梯度范数全在 1.26～4.72，均触发阈值 1 的裁剪，这也不是梯度消失。没有对照，不应据此贸然关闭裁剪或大幅提高学习率。',['representation','run'])
    section('next','下一轮应保留联合蒸馏，改成动态数据和搜索相关门禁',
      '**第一优先：换数据采集闭环，不是把这一批再训 10k。** 继续让 Stockfish 做教师，但候选必须包含教师最优、学生 policy 支持动作、学生搜索最终选步，以及实际叶节点和短后继。对模型分歧、教师预算不稳定的局面追加预算；只按教师确认的遗憾值和稳定性提权，不把学生低置信度直接当错题。保留独立来源棋局隔离和冻结测试。\n\n'
      '**数据结构应按状态—动作—后继成组。** 记录初始 FEN、完整历史、来源棋局/分组键、合法动作与动作索引、CP/mate 及视角、WDL 映射版本、教师版本/节点/深度、候选 mask、各预算结果、后继键和终局原因。根与后继同时独立标注，并记录教师自己的跨状态残差。旧版扁平行仍可保留作回放，不能用单个根 value 复制给后继。\n\n'
      '**减少静态错题永久提权。** 可先试有效 batch320 = 160旧回放 +96新独立局面 +32学生前沿 +32动态错题。这个 50/30/10/10 是待验证的起始方案，不是本轮实验求出的最优比例。已修好题降权，未稳定题先复核，持续补充新题；回放同时覆盖优势/均势/劣势，门禁检查每层而非只检查均值。\n\n'
      '**先只改变采样与监督完整性，仍联合训练 P/Q/V。** 起步保持既有头和损失定义，避免同时换模型、标签、损失后无法归因。若新数据仍出现 value 两端回缩，再在同一分支检查点上尝试小权重稳定教师一致性或 CP/排序辅助目标；HL-Gauss/分布式 Q 可作为之后的受控试验，不直接整体替换。\n\n'
      '**用证据分配步数。** 单条训练分支先给 1k～2k 更新，在多个检查点看相同的验证门禁与成组区间，不因单个 50/500 步波动判死刑。用另留的开发门禁扩大到至少 512 个不同棋局的盲选根，固定两秒搜索和稳定教师敏感性一起看；只有连续检查点显示新增收益、且旧 value 两端和大错率没有实质退化，再追加预算。最终冻结测试只用于最后确认，不能反复针对它调参。真正晋升默认搜索模型还需要配对开局/换色对局；本报告不提供 Elo。\n\n'
      '当前最终权重应保存为有监督进步的候选；无需因本轮数据把它扔掉，也不应仅凭 loss 取代冻结搜索基线。本次只完成分析和方案，没有启动下一轮训练或修改默认模型。',['metrics','search','exposure','future','strata'])
    section('literature','文献支持改变监督分布，而不是无条件加长训练',
      '[Ruoss 等，NeurIPS 2024](https://arxiv.org/abs/2402.04494) 的 ChessBench 使用约 530M 局面、15.3B 动作值标签，并对合法动作充分监督；其附录也提醒要控制动作标签总量，不能把结果简单归为“Q 一定胜过 V”。这说明大规模优质蒸馏仍值得做，不能用本项目一轮静态训练的平台宣告蒸馏无效；也不能照搬论文数据规模当作本模型的必需门槛。\n\n'
      '[Ross 等，DAgger，2011](https://proceedings.mlr.press/v15/ross11a.html) 说明序列决策中学生自身诱导的状态分布值得迭代交给专家标注；与本项目学生后继采集的动机一致。[Trudeau 与 Bowling，Go-Exploit，2023](https://arxiv.org/abs/2302.12359) 关注搜索状态分布与 value 学习，但实验是 Connect Four / 9×9 Go，不能声称已证明本棋模型会提升。\n\n'
      '[Farebrother 等，ICML 2024](https://proceedings.mlr.press/v235/farebrother24a.html) 提供分类式/HL-Gauss 回归目标的经验依据。它是后续 Q 表示试验的参考，并不能替代当前最先要修的分布覆盖与教师稳定性。文献机制与本项目证据的对应关系是推断，收益仍需本地门禁和实战验证。',[])
    section('methods','复现范围、权重身份与结论边界',
      '完成：训练记录与 SHA 复核；六个冻结数据子集的双模型原始输出；新验证/测试按棋局 bootstrap；精确 sampler RNG 回放；96 根两秒搜索及 24 根重复；24×8 个定模拟数输出交换；239 个状态的独立教师复核；后继值与一步一致性；表示、参数与梯度探针。没有进行长时间对局，也没有从这些局面推算 Elo。\n\n'
      '父模型 compact SHA256：`efb1209f5a0cd60f07fb7d0b79e1a6d70e479b2e1f453b008330ef5e157f5c44`。最终完整含优化器 SHA256：`7a9d1ef8a82036a9534863178720301d7bf9d881f1bb253376da36c0825325f5`。最终 compact SHA256：`99931bbb7f68a812e226364c9636a3725599a54dcc3affb0cc206a48318c4d3a`。\n\n'
      '新数据精确键与来源棋局划分通过既有审计；旧数据缺少原始棋局 ID，无法追溯所有历史棋局重合。后继仅 24 个来源棋局，置信区间较宽；多项分层/机制结果为探索性诊断，未做多重比较校正。96 根按 value 均衡选取，因此平均 regret 不是自然实战分布的估计。教师只是有限预算近似，报告同时给出稳定子集，避免把浅搜索标签当真理。\n\n'
      '原始 NPY / JSONL、协议、SHA 清单保存在本报告同级目录，脚本在项目 `scripts/`。按顺序运行 `analyze_joint_checkpoint` 的 prepare/raw、`analyze_joint_search`、`analyze_joint_teacher`、`summarize_joint_search` 的 search/future、`analyze_joint_mechanisms`、`analyze_joint_teacher_drift`、`analyze_joint_sensitivity` 可重做诊断；原始 journal 可复用。搜索脚本对冻结源码/动态库 SHA 强制断言，未修改搜索基线。',['run'])
    sections[-1]['body']+='\n\n保留的后继集合在本次生成时按局面键去重，历史完整保留；后续已把生成器改为按完整历史标识去重。报告仍严格使用冻结保存的 143 个状态，不用修改后的生成器悄悄扩充样本。下载后的不可变训练日志与模型 SHA 对上云端持久化清单；`status.json` 在清单写入后被 guard 更新为 complete，其旧清单哈希已过时，不把这个状态文件差异误报为权重损坏。'
    bound_ids={'metrics':['supervised-table'],'curves':['loss-curve','old-strata-curve'],'search':['search-results','search-comparison'],'strata':['value-strata'],'future':['future-value','future-value-chart'],'exposure':['exposure-table'],'swaps':['head-swaps'],'qcoverage':['q-coverage'],'teacher':['teacher-table'],'representation':['representation-table'],'run':['run-table'],'cases':['case-table']}
    for key,ids in bound_ids.items():queries[key]['source']['metricDefinitions'][0]['componentIds'].extend(ids)
    path=OUT/'report-app/src/data.json';snapshot=json.loads(path.read_text());snapshot.update(title='监督学习有进步，搜索收益仍不确定；下一轮应改变采样与门禁',status='reviewed',buildStatus='creating',generatedAt=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),report={'asOf':'2026-11-03'},queries=queries,analysisSections=sections)
    snapshot['literaturePreviews']={
      'https://arxiv.org/abs/2402.04494':{'title':'Grandmaster-Level Chess Without Search','summary':'大规模教师动作值监督的棋模型研究；本报告用来比较监督覆盖，不能直接外推本模型棋力。','approvedForReport':True},
      'https://proceedings.mlr.press/v15/ross11a.html':{'title':'A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning','summary':'讨论序列决策中学生诱导分布与迭代专家标注。','approvedForReport':True},
      'https://arxiv.org/abs/2302.12359':{'title':'Go-Exploit: Targeted Search Control in AlphaZero for Sample-Efficient Learning','summary':'搜索状态采样研究，实验对象是 Connect Four 和小棋盘 Go，非国际象棋。','approvedForReport':True},
      'https://proceedings.mlr.press/v235/farebrother24a.html':{'title':'Stop Regressing: Training Value Functions via Classification for Scalable Deep RL','summary':'比较分类式 value 目标与标量回归的经验研究，可作为之后 Q 表示试验的依据。','approvedForReport':True}}
    path.write_text(json.dumps(snapshot,ensure_ascii=False,indent=2));(OUT/'reviewed-report-data.json').write_text(json.dumps(snapshot,ensure_ascii=False,indent=2))
    print('Reviewed queries',len(queries),'sections',len(sections))


if __name__=='__main__':main()
