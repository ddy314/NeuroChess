"""Aggregate the fixed fast duel and its original-budget control separately."""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True);a=p.parse_args();root=Path(a.root)
    report={};games={}
    for label,folder in [('fast','duel'),('original-budget','stable-duel'),('original-budget','stable31-duel')]:
        d=json.loads((root/folder/'result.json').read_text());games.setdefault(label,{}).update(d['games'])
        result={}
        for opponent,rows in d['games'].items():
            counts={}
            for name in [d['plan']['args']['contender'],opponent]:
                choices=sum(r['choices'][name] for r in rows);seconds=sum(r['seconds'][name] for r in rows)
                positions=sum(r['large_model_positions'][name] for r in rows)
                counts[name]=dict(choices=choices,large_model_positions=positions,mean_large_positions=positions/max(1,choices),
                    seconds=seconds,seconds_per_choice=seconds/max(1,choices))
            result[opponent]=dict(**d['results'][opponent],work=counts)
        report.setdefault(label,{}).update(result)
    def pairs(rows):
        grouped={}
        for r in rows:
            if r['score'] is not None:grouped.setdefault(r['opening'],[]).append(r['score'])
        return {k:float(np.mean(v)) for k,v in grouped.items() if len(v)==2}
    fast=pairs(games['fast']['puct-512']);stable=pairs(games['original-budget']['puct-512']);common=sorted(set(fast)&set(stable))
    delta=np.array([stable[k]-fast[k] for k in common]);rng=np.random.default_rng(713031)
    if len(delta):
        means=delta[rng.integers(len(delta),size=(20000,len(delta)))].mean(1)
        report['matched_opening_control']=dict(pairs=len(delta),original_budget_minus_fast_score=float(delta.mean()),
            bootstrap95=np.quantile(means,[.025,.975]).tolist(),note='Same initial opening pairs versus PUCT512; exploratory budget-control comparison, not extra independent games.')
    report['note']='All relative Elo estimates are conditional on these budgets, frozen model, opening selection and draw rules; no absolute Elo calibration. Game timing is descriptive; dedicated heldout benchmark supplies controlled latency.'
    Path(a.out).write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':main()
