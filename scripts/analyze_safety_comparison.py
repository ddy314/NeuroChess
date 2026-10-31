"""Paired tactical-loss analysis, including CPU work and mate-only outcomes."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.analyze_rank_comparison import paired


def main():
    p=argparse.ArgumentParser();p.add_argument('--result',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    d=json.loads(Path(a.result).read_text());names=[v['name'] for v in d['suite']]
    cp=[{r['method']:r for r in x['methods']} for x in d['positions'] if all(r['cp_regret'] is not None for r in x['methods'])]
    comparisons={}
    for x,y in [('guard','baseline'),('rank-guard','baseline'),('rank-guard','guard'),('proof','baseline'),('cpu','proof'),('combined','baseline'),('combined','cpu'),
                ('combined','guard'),('combined-16','combined'),('combined','puct-31'),('combined','puct-512'),('combined-16','puct-15'),
                ('cpu','baseline'),('cpu','puct-31'),('cpu','puct-512'),('cpu-gated','cpu'),('cpu-global-guard','cpu'),
                ('cpu-gated','baseline'),('cpu16','baseline'),('cpu16','puct-15'),('cpu-gated16','puct-15'),('cpu-gated16','cpu-gated')]:
        if x in names and y in names and cp:comparisons[f'{x} minus {y}']=paired([r[x]['cp_regret'] for r in cp],[r[y]['cp_regret'] for r in cp])
    mate={n:dict(winning_mate_missed=0,avoidable_mate_loss=0) for n in names};summary=d['summary']
    for x in d['positions']:
        labels=x['teacher_candidates'];has_win=any(s['mate'] is not None and s['mate']>0 for s in labels.values())
        has_nonloss=any(s['mate'] is None or s['mate']>0 for s in labels.values())
        for r in x['methods']:
            m=r['teacher']['mate'];mate[r['method']]['winning_mate_missed']+=int(has_win and (m is None or m<=0))
            mate[r['method']]['avoidable_mate_loss']+=int(has_nonloss and m is not None and m<0)
    for n in names:
        rows=[next(r for r in x['methods'] if r['method']==n) for x in d['positions']]
        summary[n]['p95_seconds']=float(np.quantile([r['seconds'] for r in rows],.95))
        summary[n]['mean_inference_batches']=float(np.mean([r['stats'].get('inference_batches',0) for r in rows]))
        summary[n]['selected_root_value_clipped']=sum(abs(r.get('root',{}).get('selected_value',0)-r.get('root',{}).get('decision_value',r.get('root',{}).get('selected_value',0)))>1e-5 for r in rows)
        for key in ['mate_nodes','tactical_nodes','cpu_mate_proofs','cpu_references','clipped_anchor_nodes']:
            summary[n]['mean_'+key]=float(np.mean([r.get('root',{}).get('stats',{}).get(key,0) for r in rows]))
        summary[n]['mate_events']=mate[n]
    result=dict(source=a.result,positions=len(d['positions']),cp_positions=len(cp),mate_positions=len(d['positions'])-len(cp),
        summary=summary,paired=comparisons,
        note='Finite-budget common candidate-union labels, not optimality or Elo. Mate events mean the teacher reports a mate versus an alternative without that report, not that a draw was available. One root per new source game; bootstrap paired roots. Primary comparison and stage are recorded in the corresponding frozen plan.')
    Path(a.out).write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
