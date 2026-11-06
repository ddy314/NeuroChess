"""Combine fixed-budget evidence and label bounded, selected visited-node probes."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.lichess_failure_mechanisms import *
from math import factorial

def fresh_reference(index,extra_moves):
    source=OUT/'coherent-teachers'/f'{index:03d}.json'
    ref=read(source if source.exists() else OUT/'teachers'/f'{index:03d}.json');rows=ref['final'];known={r['move'] for r in rows}
    if set(extra_moves)<=known:return ref
    job=next(j for j in read(OUT/'jobs.json') if j['index']==index);b=chess.Board(job['fen'])
    engine=FastTeacher(ENGINE,threads=1,hash_mb=64,timeout=300)
    try:
        candidates=list(dict.fromkeys([r['move'] for r in rows]+extra_moves));nodes=ref['stages'][-1]['requested_nodes']
        teacher=scored_infos(engine,b,nodes,candidates)
        ref=dict(ref,additional_reference=teacher,final=teacher,depth24_exact=min(r['depth'] for r in teacher)>=24 and not any(r['lowerbound'] or r['upperbound'] for r in teacher))
        return ref
    finally:engine.quit()

def node_teacher(row):
    path=OUT/'node-probes'/(row['input_sha256']+'.json')
    if path.exists():return read(path)
    b=chess.Board(row['fen']);engine=FastTeacher(ENGINE,threads=1,hash_mb=64,timeout=180)
    try:ref=scored_infos(engine,b,1000000)[0]
    finally:engine.quit()
    target=teacher_wl(ref);result=dict(row,teacher=ref,target_wl=target,
        parent_error=abs(row['parent_value']-target),candidate_error=abs(row['candidate_value']-target),
        scope='Retrospectively selected maximum-drift visited input; FEN-only1M proxy, no original native path history; not population or training target')
    write(path,result);return result

def main():
    if not (OUT/'completion.json').exists() or not (OUT/'matched-budget/search-results.json').exists():raise RuntimeError('Required searches incomplete')
    report=read(OUT/'results.json');primary=read(OUT/'completion.json')
    if sha(OUT/'results.json')!=primary['report_sha256']:raise ValueError('Primary report differs')
    for name,h in primary['files_sha256'].items():
        if sha(OUT/name)!=h:raise ValueError('Primary artifact differs: '+name)
    matched=read(OUT/'matched-budget/search-results.json');results={};probes=[]
    for index in FAILURES:
        cases=[r for r in matched if r['index']==index];reference=fresh_reference(index,[r['move'] for r in cases]);write(OUT/'review'/f'teacher-{index:03d}.json',reference)
        scores={r['move']:r['sort_score'] for r in reference['final']};best=max(scores.values());bycode={}
        for r in cases:
            r['regret']=best-scores[r['move']]
            if r['route']=='normal':bycode[r['code']]=r
        budget=cases[0]['budget'];cells={c:bycode[c]['regret'] for c in [''.join(map(str,x)) for x in itertools.product([0,1],repeat=3)]}
        # Exact Shapley allocation of observed all-new minus all-old decision regret.
        allocation={}
        for head in range(3):
            effect=0.
            for code,a in cells.items():
                if code[head]!='0':continue
                k=code.count('1');weight=factorial(k)*factorial(2-k)/factorial(3)
                other=code[:head]+'1'+code[head+1:];effect+=weight*(cells[other]-a)
            allocation[['Policy','Q','Value'][head]]=effect
        delta=cells['111']-cells['000']
        result=dict(index=index,budget=budget,teacher_depth24_exact=reference['depth24_exact'],
            observed_old_move=bycode['000']['move'],observed_new_move=bycode['111']['move'],old_regret=cells['000'],new_regret=cells['111'],
            delta=delta,all_new_reproduces_extra_regression=cells['111']>=100 and delta>=75,
            factorial_cells={c:dict(move=bycode[c]['move'],regret=cells[c]) for c in cells},
            rollback_residual_regret={k:bycode[c]['regret'] for k,c in [('Policy','011'),('Q','101'),('Value','110'),('Value_MLP_only','112'),('Value_trunk_only','113')]},
            routing={r['route']:dict(move=r['move'],regret=r['regret']) for r in cases if r['route']!='normal'},
            shapley_allocation=allocation,shapley_scope='Exact decomposition on five postselected cases; conditional work budget, not general importance or causal training diagnosis')
        assert abs(sum(allocation.values())-delta)<1e-6
        results[str(index)]=result
        # Two one-network controls and all fixed-budget traces are retained, sample8 distinct maximum-drift inputs.
        samples={}
        for r in report['rows']:
            if r['index']==index and r['budget']==4096 and r['code'] in ['000','111']:
                for row in r.get('node_samples',[]):samples[row['input_sha256']]=dict(row,index=index)
        probes.extend(sorted(samples.values(),key=lambda x:-x['drift'])[:8])
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        labeled=list(executor.map(node_teacher,probes))
    verify_inputs()
    result=dict(cases=results,visited_node_probes=labeled,
        node_probe_summary=dict(n=len(labeled),parent_mae=float(np.mean([r['parent_error'] for r in labeled])),candidate_mae=float(np.mean([r['candidate_error'] for r in labeled]))),
        training_resumed=False,optimizer_steps=0,weights_unchanged=True,primary_report_sha256=primary['report_sha256'])
    write(OUT/'review/results.json',result);write(OUT/'review/completion.json',dict(completed=True,report_sha256=sha(OUT/'review/results.json'),files_sha256={str(p.relative_to(OUT/'review')):sha(p) for p in (OUT/'review').rglob('*') if p.is_file() and p.name!='completion.json'}))
    print(json.dumps(results,indent=2),flush=True)

if __name__=='__main__':main()
