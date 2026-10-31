"""Paired uncertainty and severe-loss counts for frozen game-root comparisons."""
import argparse
import json
import math
from pathlib import Path

import numpy as np


def paired(a,b):
    """Negative delta favors a. Bootstrap positions (one root per heldout game)."""
    a,b=np.asarray(a),np.asarray(b);rng=np.random.default_rng(431903)
    indices=rng.integers(0,len(a),(10000,len(a)))
    delta=a-b;rate=(a>=100).astype(float)-(b>=100).astype(float)
    fixed=int(((a<100)&(b>=100)).sum());new=int(((a>=100)&(b<100)).sum());n=fixed+new
    p=min(1.,2*sum(math.comb(n,k) for k in range(min(fixed,new)+1))/2**n) if n else 1.
    return dict(positions=len(a),mean_cp_delta=float(delta.mean()),mean_cp_delta_bootstrap95=np.quantile(delta[indices].mean(1),[.025,.975]).tolist(),
        blunder_rate_delta=float(rate.mean()),blunder_rate_delta_bootstrap95=np.quantile(rate[indices].mean(1),[.025,.975]).tolist(),
        fixed_blunders=fixed,new_blunders=new,discordant_exact_two_sided_p=p)


def main():
    p=argparse.ArgumentParser();p.add_argument('--result',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    result=json.loads(Path(a.result).read_text());names=[v['name'] for v in result['suite']]
    rows=[{r['method']:r for r in x['methods']} for x in result['positions']]
    cp_rows=[r for r in rows if all(r[n]['cp_regret'] is not None for n in names)]
    comparisons={}
    pairs=[('cp','wdl'),('cp-rank','wdl'),('cp-rank','cp'),('cp-rank','old-aligned'),('cp-rank','puct-31'),('cp-rank','puct-512')]
    pairs += [(n+'-local',n) for n in ['old-aligned','wdl','cp-rank']]
    pairs += [('cp-rank-local','wdl-local'),('cp-rank-local','puct-31'),('cp-rank-local','puct-512')]
    for x,y in pairs:
        if x not in names or y not in names:continue
        comparisons[f'{x} minus {y}']=paired([r[x]['cp_regret'] for r in cp_rows],[r[y]['cp_regret'] for r in cp_rows])
    mates={n:dict(winning_mate_missed=0,avoidable_mate_loss=0) for n in names};mate_roots=0
    for x in result['positions']:
        labels=x['teacher_candidates'];has_win=any(s['mate'] is not None and s['mate']>0 for s in labels.values())
        has_nonloss=any(s['mate'] is None or s['mate']>0 for s in labels.values())
        if any(s['mate'] is not None for s in labels.values()):mate_roots+=1
        for r in x['methods']:
            m=r['teacher']['mate']
            mates[r['method']]['winning_mate_missed']+=int(has_win and (m is None or m<=0))
            mates[r['method']]['avoidable_mate_loss']+=int(has_nonloss and m is not None and m<0)
    summary=dict(source=a.result,summary=result['summary'],cp_roots=len(cp_rows),mate_roots=mate_roots,
        paired=comparisons,mate_events=mates,
        note='Frozen source-game roots; finite-node candidate-union labels, not exact chess truth. Original primary comparison is CP+rank versus matched WDL; local-exploration follow-up uses a separate later-game cohort. Exploratory comparisons are not multiplicity corrected.')
    Path(a.out).write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
