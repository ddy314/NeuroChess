"""Supplement: P/Q/V interventions at observed2s simulation counts, no training."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.lichess_failure_mechanisms import *
DEST=OUT/'matched-budget'

def main():
    deadline=time.time()+7200
    while time.time()<deadline:
        if (OUT/'failure.json').exists():raise RuntimeError('Primary mechanism run failed')
        if (OUT/'status.json').exists() and read(OUT/'status.json')['phase'] in ['deep-teacher','complete']:break
        time.sleep(5)
    else:raise TimeoutError('Waiting for primary GPU work')
    DEST.mkdir(exist_ok=True);torch.set_num_threads(2);torch.set_float32_matmul_precision('high');lock,_=verify_inputs()
    nets=[load_net(p) for p in [PARENT,CANDIDATE]]
    jobs=[j for j in read(OUT/'jobs.json') if j['index'] in FAILURES]
    budgets={j['index']:min(x['stats']['new_simulations'] for x in j['search'].values()) for j in jobs}
    write(DEST/'plan.json',dict(budgets=budgets,source='Min original audited2s root simulation count, fixed before these interventions',
        scope='Same simulations and two full networks per cell, retains all search settings except stopping rule. Not latency or strength evaluation.',training_resumed=False))
    allrows=[]
    for j in jobs:
        b=chess.Board(j['fen']);t,m,ids,_=pack(b)
        for n in nets:
            for size in [1,64]:n.raw_outputs(np.repeat(t,size,0),np.repeat(m,size,0),np.repeat(ids,size,0))
        modes=[(code,'normal') for code in [''.join(map(str,c)) for c in itertools.product([0,1],repeat=3)]]
        modes += [('111',r) for r in ['old_leaf_FPU','old_confidence','old_value_Q_seed','old_V_root_input_only']]+[('112','normal'),('113','normal')]
        for code,route in modes:
            path=DEST/f'{j["index"]:03d}-{code}-{route}.json'
            if path.exists():allrows.append(read(path));continue
            net=Mixed(nets,tuple(map(int,code)),route,root=b,capture=code in ['000','111'] and route=='normal')
            player=NativeMCTSPlayer(net,**dict(lock['config'],time_limit=None,simulations=budgets[j['index']]))
            trace=[];player.lib=ObserveLibrary(player.lib,trace);tick=time.time()
            try:move=player.choose(b);root=player.last_result;stats=player.search_stats
            finally:player.close();net.close()
            r=dict(index=j['index'],code=code,route=route,budget=budgets[j['index']],move=move.uci(),root=root,stats=stats,trace=trace,seconds=time.time()-tick)
            write(path,r);allrows.append(r);write(DEST/'status.json',dict(phase='searching',done=len(allrows),total=70,training_resumed=False))
    verify_inputs()
    write(DEST/'search-results.json',allrows);write(DEST/'status.json',dict(phase='search-complete',done=70,training_resumed=False))

if __name__=='__main__':
    try:main()
    except BaseException:
        write(OUT/'matched-budget-failure.json',dict(traceback=traceback.format_exc(),training_resumed=False));raise
