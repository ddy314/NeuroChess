"""Summarize controlled search ablations without treating teacher scores as proofs."""
import argparse
import gzip
import json
from pathlib import Path
import numpy as np


def paired_ci(values):
    values=np.asarray(values,float);rng=np.random.default_rng(813133)
    return np.quantile(values[rng.integers(len(values),size=(20000,len(values)))].mean(1),[.025,.975]).tolist()


def compare(report,group='positions'):
    positions=report[group];names=[r['method'] for r in positions[0]['methods']];summary={}
    for name in names:
        rows=[next(r for r in p['methods'] if r['method']==name) for p in positions]
        cp=[r['cp_regret'] for r in rows if r['cp_regret'] is not None]
        summary[name]=dict(positions=len(rows),cp_positions=len(cp),mean_cp_regret=float(np.mean(cp)) if cp else None,
            blunders_100=sum(x>=100 for x in cp))
        delta=[]
        for p,r in zip(positions,rows):
            base=next(x for x in p['methods'] if x['method']=='legacy')
            if r['cp_regret'] is not None and base['cp_regret'] is not None:delta.append(r['cp_regret']-base['cp_regret'])
        if delta:summary[name].update(delta_cp_vs_legacy=float(np.mean(delta)),delta_cp_bootstrap95=paired_ci(delta))
    return summary


def trace_diagnosis(root,jobs,replay):
    rows=[]
    for job,case in zip(jobs,replay['cases']):
        row=next(x for x in case['methods'] if x['method']=='legacy')
        nodes=json.load(gzip.open(root/f"game-{job['game']}-legacy.json.gz",'rt'))['nodes']
        children={n['move']:n for n in nodes if n['parent']==0}
        labels=case['teacher_candidates'];all_cp=all(x['cp'] is not None for x in labels.values())
        best=max(labels,key=lambda m:labels[m]['cp']) if all_cp else job['best']
        selected=children[row['move']];good=children[best];ordered=sorted(children.values(),key=lambda n:n['prior'],reverse=True)
        both=good['corrected'] and selected['corrected']
        rows.append(dict(game=job['game'],fen=job['fen'],selected=row['move'],teacher_best=best,
            baseline_reproduces_old_move=row['move']==job['move'],cp_regret=row['cp_regret'],
            teacher_prior_rank=1+next(i for i,n in enumerate(ordered) if n['move']==best),
            best_direct_corrected=good['corrected'],best_neural_observations=good['neural_count'],
            selected_neural_observations=selected['neural_count'],
            largest_branch_query_share=max(n['neural_count'] for n in children.values())/max(1,nodes[0]['neural_count']-1),
            both_direct_corrected=both,cheap_prefers_teacher=good['cheap']<selected['cheap'],
            direct_prefers_teacher=good['anchor']<selected['anchor'] if both else None,
            backup_prefers_teacher=good['value']<selected['value'],
            harmful_direct_flip=both and good['cheap']<selected['cheap'] and good['anchor']>selected['anchor'],
            harmful_backup_flip=both and good['anchor']<selected['anchor'] and good['value']>selected['value'],
            selected_node={k:selected[k] for k in ['cheap','anchor','value','prior','visits','cpu_value','cpu_ready','corrected','neural_count']},
            alternative={k:good[k] for k in ['cheap','anchor','value','prior','visits','cpu_value','cpu_ready','corrected','neural_count']}))
    return dict(cases=rows,summary=dict(cases=len(rows),baseline_move_parity=sum(r['baseline_reproduces_old_move'] for r in rows),
        best_without_direct_correction=sum(not r['best_direct_corrected'] for r in rows),
        harmful_direct_flips=sum(r['harmful_direct_flip'] for r in rows),harmful_backup_flips=sum(r['harmful_backup_flip'] for r in rows),
        mean_largest_branch_query_share=float(np.mean([r['largest_branch_query_share'] for r in rows]))),
        note='Selected historical failures, overlapping diagnostics rather than mutually exclusive causal shares. Finite teacher candidate union. Same traced query topology used for cheap/direct/backed ordering comparisons.')


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();root=Path(a.root)
    result=dict(development=compare(json.loads((root/'development/result.json').read_text())),
        warm=compare(json.loads((root/'warm-replay/result.json').read_text()),'cases'))
    result['trace']=trace_diagnosis(root/'warm-replay',json.loads((root/'old-game-errors.json').read_text())['critical'],json.loads((root/'warm-replay/result.json').read_text()))
    for folder in ['followup-development','heldout']:
        path=root/folder/'result.json'
        if path.exists():result[folder]=compare(json.loads(path.read_text()))
    if (root/'reuse-fixed-warm/result.json').exists():
        result['reuse-fixed-warm']=compare(json.loads((root/'reuse-fixed-warm/result.json').read_text()),'cases')
    if (root/'tactical-regression/result.json').exists():
        tactical=json.loads((root/'tactical-regression/result.json').read_text())
        plan=json.loads(Path('output/correction-safety-20261031/tactical-roots/plan.json').read_text())
        kinds={r['fen']:r['kind'] for r in plan['roots']};summary={}
        for variant in tactical['suite']:
            name=variant['name'];winning=[];defense=[]
            for position in tactical['positions']:
                row=next(x for x in position['methods'] if x['method']==name)
                if kinds[position['fen']]=='mate':winning.append(row['teacher']['mate'] is not None and row['teacher']['mate']>0)
                else:
                    labels=position['teacher_candidates'].values()
                    avoidable=any(x['mate'] is None or x['mate']>=0 for x in labels)
                    defense.append((avoidable,row['teacher']['mate'] is not None and row['teacher']['mate']<0))
            summary[name]=dict(winning_roots=len(winning),winning_mate_choices=sum(winning),defense_roots=len(defense),avoidable_by_teacher=sum(a for a,b in defense),avoidable_mate_errors=sum(a and b for a,b in defense))
        result['tactical-regression']=dict(summary=summary,note='Known development FEN roots, not new holdout; full trajectory history not supplied in this static regression. Finite teacher mate labels, not complete game proofs.')
    capacity={}
    for name,path in [('small',Path('output/rank-frontier-20261031/cp-rank')),('large',root/'capacity-512')]:
        rows=[json.loads(s) for s in (path/'metrics.jsonl').read_text().splitlines()];best=min(rows,key=lambda r:r['selection_score'])
        run=json.loads((path/'run.json').read_text());capacity[name]=dict(parameters=run['parameters'],config=run['config'],best=best,last=rows[-1])
    result['capacity']=capacity
    (root/'analysis.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k not in ['trace','capacity']},indent=2))

if __name__=='__main__':main()
