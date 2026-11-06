"""Fail closed: a fast candidate does not replace the user-locked PUCT floor."""
import argparse,json,math
from pathlib import Path
import numpy as np


def evaluate(report,candidate,arena=None):
    baseline=[];chosen=[];introduced=[];delta=[];mate_failures=[]
    for position in report['positions']:
        b=next(r for r in position['methods'] if r['method']=='baseline');c=next(r for r in position['methods'] if r['method']==candidate)
        baseline.append(b['seconds']);chosen.append(c['seconds'])
        if b['cp_regret'] is not None and c['cp_regret'] is not None:
            delta.append(c['cp_regret']-b['cp_regret'])
            if c['cp_regret']>=100 and b['cp_regret']<100:introduced.append(position['index'])
        bm,cm=b['teacher']['mate'],c['teacher']['mate']
        if (cm is not None and cm<0 and (bm is None or bm>=0)) or (bm is not None and bm>0 and (cm is None or cm<=0)):mate_failures.append(position['index'])
    ref95=float(np.quantile(baseline,.95));cand95=float(np.quantile(chosen,.95));speed=ref95/cand95
    rng=np.random.default_rng(103110);values=np.asarray(delta);ci=np.quantile(values[rng.integers(len(values),size=(20000,len(values)))].mean(1),[.025,.975]).tolist() if len(values) else [None,None]
    speed_ok=cand95<=.2 and speed>=10;position_ok=bool(delta) and not introduced and not mate_failures and ci[1]<=0
    reasons=[]
    if not speed_ok:reasons.append('P95 does not meet both <=0.2s and >=10x measured baseline speedup')
    if not position_ok:reasons.append('Frozen position quality floor failed or evidence incomplete')
    arena_ok=False
    if arena is not None:
        scores={}
        for row in arena['games']:
            if row['score'] is not None:scores.setdefault(row['opening'],[]).append(row['score'])
        pairs=[sum(v)/2 for v in scores.values() if len(v)==2]
        if pairs:
            lower=max(0.,float(np.mean(pairs))-math.sqrt(math.log(40)/(2*len(pairs))))
            arena_ok=arena.get('baseline_id')=='puct-2s-v1' and lower>=.5
    if not arena_ok:reasons.append('No sufficient paired direct-play evidence against frozen2s baseline')
    return dict(candidate=candidate,samples=len(chosen),baseline_p95=ref95,candidate_p95=cand95,speedup=speed,
        speed_ok=speed_ok,mean_cp_change=float(np.mean(delta)) if delta else None,cp_change_bootstrap95=ci,introduced_blunders=introduced,introduced_mate_failures=mate_failures,
        position_ok=position_ok,arena_ok=arena_ok,promote=speed_ok and position_ok and arena_ok,reasons=reasons,
        note='This is a conservative empirical promotion gate, not a mathematical guarantee over every chess position. Do not relax the locked floor to pass a candidate.')


def main():
    p=argparse.ArgumentParser();p.add_argument('--result',required=True);p.add_argument('--candidate',required=True);p.add_argument('--arena');p.add_argument('--out',required=True);a=p.parse_args()
    result=evaluate(json.loads(Path(a.result).read_text()),a.candidate,json.loads(Path(a.arena).read_text()) if a.arena else None)
    Path(a.out).write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
    raise SystemExit(0 if result['promote'] else 2)

if __name__=='__main__':main()
