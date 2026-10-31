"""Compare the fixed fast candidate with original and matched-batch PUCT."""
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
    summary=d['summary'];comparisons={}
    for other in ['conservative','kernel512','puct-31','puct-31-b16','puct-512']:
        if other in names and cp:comparisons['optimized minus '+other]=paired([r['optimized']['cp_regret'] for r in cp],[r[other]['cp_regret'] for r in cp])
    for name in names:
        rows=[next(r for r in x['methods'] if r['method']==name) for x in d['positions']]
        summary[name]['mean_native_seconds']=float(np.mean([r['stats'].get('native_seconds',0) for r in rows]))
        summary[name]['mean_inference_seconds']=float(np.mean([r['stats'].get('inference_seconds',0) for r in rows]))
        summary[name]['mean_inference_batches']=float(np.mean([r['stats'].get('inference_batches',0) for r in rows]))
        summary[name]['p95_seconds']=float(np.quantile([r['seconds'] for r in rows],.95))
        events=dict(winning_mate_missed=0,avoidable_mate_loss=0)
        for x,r in zip(d['positions'],rows):
            labels=x['teacher_candidates'];m=r['teacher']['mate']
            won=any(s['mate'] is not None and s['mate']>0 for s in labels.values())
            nonloss=any(s['mate'] is None or s['mate']>0 for s in labels.values())
            events['winning_mate_missed']+=int(won and (m is None or m<=0))
            events['avoidable_mate_loss']+=int(nonloss and m is not None and m<0)
        summary[name]['mate_events']=events
    result=dict(positions=len(d['positions']),cp_positions=len(cp),summary=summary,paired=comparisons,
        note='Fresh frozen roots, finite-node unified candidate union, not Elo. Fast and conservative points chosen only on old development roots. Matched-batch PUCT is an additional efficiency control, not the original PUCT-31 configuration.')
    Path(a.out).write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
