"""Sensitivity checks on frozen outputs; no further inference or model fitting."""
import json,sys
from pathlib import Path
import numpy as np
from scripts.summarize_joint_search import board_for,regret
from scripts.analyze_joint_checkpoint import paired_interval,write


def main(out):
    teachers=[json.loads(l) for l in (out/'deep-teacher.jsonl').read_text().splitlines()]
    refs={m['job']['key']:m for m in teachers if 'depth' not in m['job']}
    a=np.load(out/'new_test.npy');index={bytes(k).hex():i for i,k in enumerate(a['key'])}
    raw=[np.load(out/f'new_test-{l}-raw.npy') for l in ['parent','final']]
    report={'policy_only_direct':{},'q_error_paired':{},'head_effects_stable_only':{}}
    for stable in [False,True]:
        keys=[k for k,m in refs.items() if not stable or m['stable']];rows=[]
        for which in [0,1]:
            rows.append([regret(refs[k],list(board_for(refs[k]['job']).legal_moves)[int(raw[which][index[k],:256].argmax())].uci(),float(raw[which][index[k],512]-raw[which][index[k],514]))['cp_regret_capped1000'] for k in keys])
        report['policy_only_direct']['stable_only' if stable else 'all']=dict(n=len(keys),parent=float(np.mean(rows[0])),final=float(np.mean(rows[1])),**paired_interval(np.array(rows[1])-rows[0],keys))
    q=json.loads((out/'q-deep-teacher-errors.json').read_text())
    for labelled in [True,False]:
        for stable in [False,True]:
            for supported in [False,True]:
                p=[r for r in q if r['model']=='parent' and r['originally_labelled']==labelled and (not stable or refs[r['key']]['stable'])]
                lookup={(r['key'],r['move']):r for r in q if r['model']=='final'}
                # Fixed union of either model's inference support avoids changing denominator.
                if supported:p=[r for r in p if r['supported'] or lookup[r['key'],r['move']]['supported']]
                f=[lookup[r['key'],r['move']] for r in p]
                report['q_error_paired'][f'labelled={labelled},stable={stable},support_union={supported}']=dict(n_candidates=len(p),parent=float(np.mean([r['error'] for r in p])),final=float(np.mean([r['error'] for r in f])),**paired_interval([b['error']-a['error'] for a,b in zip(p,f)],[r['group'] for r in p]))
    results=json.loads((out/'decision-teacher-details.json').read_text())
    for i,head in enumerate(['policy','Q','value']):
        per_root={}
        for row in [r for r in results if r['mode']=='head-swap-fixed512' and r['sources'][i]=='0' and r['stable']]:
            other=row['sources'][:i]+'1'+row['sources'][i+1:]
            counter=next(r for r in results if r['mode']==row['mode'] and r['key']==row['key'] and r['sources']==other)
            per_root.setdefault(row['key'],[]).append(counter['cp_regret_capped1000']-row['cp_regret_capped1000'])
        report['head_effects_stable_only'][head]=paired_interval([np.mean(v) for v in per_root.values()],list(per_root))
    drift=json.loads((out/'teacher-drift-and-overlap.json').read_text());bad={k for d in drift['continuation_exact_train_overlap'].values() for k in d['keys']}
    children=json.loads((out/'future-value-details.json').read_text());report['future_sensitivity']={}
    for depth in [1,2,4]:
        for stable in [False,True]:
            rows=[r for r in children if r['job']['depth']==depth and r['job']['key'] not in bad and (not stable or r['stable'])]
            report['future_sensitivity'][f'depth={depth},stable={stable}']=dict(n=len(rows),parent=float(np.mean([r['parent_error'] for r in rows])),final=float(np.mean([r['final_error'] for r in rows])),**paired_interval([r['final_error']-r['parent_error'] for r in rows],[r['job']['group'] for r in rows]))
    future=json.loads((out/'future-value-assessment.json').read_text());report['one_ply_consistency']={}
    for stable in [False,True]:
        rows=[r for r in future['bellman_edges'] if not stable or r['stable']]
        report['one_ply_consistency']['stable_only' if stable else 'all']=dict(n=len(rows),teacher=float(np.mean([r['teacher_residual'] for r in rows])),parent=float(np.mean([r['parent_residual'] for r in rows])),final=float(np.mean([r['final_residual'] for r in rows])),**paired_interval([r['final_residual']-r['parent_residual'] for r in rows],[r['group'] for r in rows]))
    write(out/'sensitivity-analysis.json',report)


if __name__=='__main__':main(Path(sys.argv[1]))
