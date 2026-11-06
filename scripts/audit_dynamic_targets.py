"""Compare masked stable targets with independent1M/3M teacher references."""
import argparse,concurrent.futures as futures,hashlib,json,multiprocessing,sqlite3,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from neurochess.teacher_stability import qualify,qualified_decision
from neurochess.targets import score_targets
from scripts import dynamic_teacher_worker as worker


def check(task):
    import chess.engine
    r=task['row'];b=worker.board_for(r['job'])
    # SF's finite-node search is sensitive to searchmoves order. Preserve the
    # recorded coherent order so budget drift is not conflated with reordering.
    moves=r.get('attempts',[{}])[-1].get('candidates') or [p['line'].split()[0] for p in r['teacher']];stages=[]
    discovery=None
    if task.get('discover'):
        infos=worker.ENGINE.analyse(b,chess.engine.Limit(nodes=3000000),min(4,b.legal_moves.count()))
        discovery=[i['pv'][0].uci() for i in infos]
        moves=list(dict.fromkeys(moves+discovery))
    for budget in [1000000,3000000]:
        infos=worker.ENGINE.analyse(b,chess.engine.Limit(nodes=budget),len(moves),root_moves=[b.parse_uci(m) for m in moves]);pvs=[]
        for info in infos:
            s=info['score'].white();p=dict(line=' '.join(m.uci() for m in info['pv']),depth=info['depth'],nodes=info['nodes']);p['mate' if s.is_mate() else 'cp']=s.mate() if s.is_mate() else s.score()
            if info.get('lowerbound') or info.get('upperbound'):p['bound']='lower' if info.get('lowerbound') else 'upper'
            pvs.append(p)
        stages.append(dict(budget=budget,independent=True,candidates=moves,teacher=pvs))
    q=qualify(stages[0]['teacher'],stages[1]['teacher'],b.turn);d=qualified_decision(stages[0]['teacher'],stages[1]['teacher'],r['prediction'],b.turn,q)
    old=task['supervision'];best_old=max(r['teacher'],key=lambda p:score_targets(p,b.turn)[0]);best_deep=max(stages[-1]['teacher'],key=lambda p:score_targets(p,b.turn)[0]);target_shift=abs(score_targets(best_old,b.turn)[1]-score_targets(best_deep,b.turn)[1])
    original_q={p['line'].split()[0]:score_targets(p,b.turn)[1] for p in r['teacher']};deep_q={p['line'].split()[0]:score_targets(p,b.turn)[1] for p in stages[-1]['teacher']}
    return dict(case_id=worker.state_id(r['job']),key=r['job']['key'],group=r['job']['group'],role=r['job'].get('role','root'),newly_discovered_moves=[] if discovery is None else [m for m in discovery if m not in original_q],old_fully_stable=old['policy'],old_qualification=old,deep_qualification=q,deep_decision=d,retained_value_error=d['value_mistake'],value_target_shift=target_shift,q_target_shifts=[abs(original_q[m]-deep_q[m]) for m in old['q_moves']],stages=stages)


def database_tasks(path,model_sha,limit,role):
    db=sqlite3.connect(f'file:{Path(path).resolve()}?mode=ro',uri=True)
    selected=[];groups=set()
    query='''SELECT s.job,p.prediction,t.metadata,d.decision FROM states s
        JOIN decisions d ON d.case_id=s.case_id JOIN teachers t ON t.label_id=d.label_id
        JOIN predictions p ON p.case_id=s.case_id AND p.model_sha=d.model_sha
        WHERE d.model_sha=? AND s.role=? AND s.quarantine=0 AND s.split='train' AND d.active=1
        AND t.stable=1 ORDER BY s.case_id'''
    for job,pred,meta,decision in db.execute(query,(model_sha,role)):
        j=json.loads(job);m=json.loads(meta);d=json.loads(decision)
        if not {'policy','value'}.intersection(d.get('repair_heads',[])):continue
        if role=='root' and j['group'] in groups:continue
        groups.add(j['group']);m.update(job=j,prediction=json.loads(pred))
        selected.append(dict(row=m,supervision=m['supervision'],decision=d))
        if len(selected)>=limit:break
    db.close();return selected


def main(args):
    folder=Path(args.root);out=[];summary={};selected=[]
    if args.db:
        selected=database_tasks(args.db,args.model_sha,args.limit,args.role)
        for task in selected:task['discover']=args.discover
        destination=Path(args.output);destination.parent.mkdir(parents=True,exist_ok=True);started=time.monotonic()
        cohort_sha=hashlib.sha256(('\n'.join(sorted(x['row']['job']['key'] for x in selected))+'\n').encode()).hexdigest()
        with futures.ProcessPoolExecutor(args.workers,mp_context=multiprocessing.get_context('spawn'),initializer=worker.initialize,initargs=(args.binary,)) as pool:
            for result in pool.map(check,selected):
                out.append(result)
                destination.write_text(json.dumps(dict(protocol='Unrestricted 3M move discovery then coherent independent 1M/3M union' if args.discover else 'Independent 1M/3M, preserved coherent candidate order; source-game-unique roots; partial targets remain masked',model_sha256=args.model_sha,selected=len(selected),audit_cohort_sha256=cohort_sha,complete=len(out)==len(selected),seconds=time.monotonic()-started,rows=out),indent=2))
        print(json.dumps(dict(output=str(destination),audited=len(out),seconds=time.monotonic()-started)),flush=True)
        return
    for label in ['uniform','priority']:
        rows=json.loads((folder/f'{label}-benchmark/fast-16-labels.json').read_text());accepted=[]
        for r in rows:
            if r['stable'] is None:continue
            turn=len(r['job']['history'])%2==0;q=qualify(r['teacher_previous'],r['teacher'],turn);d=qualified_decision(r['teacher_previous'],r['teacher'],r['prediction'],turn,q)
            if d['usable_error']:accepted.append(dict(row=r,supervision=q,decision=d))
        elapsed=json.loads((folder/f'{label}-benchmark/benchmark.json').read_text())['results'][0]['seconds']
        summary[label]=dict(roots=len(rows),accepted_mistakes=len(accepted),fully_cp_stable_mistakes=sum(x['row']['stable'] is True for x in accepted),masked_stable_mistakes=sum(x['row']['stable'] is False for x in accepted),seconds=elapsed,usable_mistakes_per_second=len(accepted)/elapsed)
        if label=='priority':
            rng=np.random.default_rng(20261103)
            for fully,quota in [(True,16),(False,32)]:
                candidates=[x for x in accepted if (x['row']['stable'] is True)==fully]
                selected.extend(candidates[i] for i in rng.choice(len(candidates),min(quota,len(candidates)),replace=False))
    folder.mkdir(exist_ok=True);(folder/'target-aware-summary.json').write_text(json.dumps(summary,indent=2));started=time.monotonic()
    with futures.ProcessPoolExecutor(args.workers,mp_context=multiprocessing.get_context('spawn'),initializer=worker.initialize,initargs=(args.binary,)) as pool:
        for r in pool.map(check,selected):
            out.append(r);(folder/'target-deep-audit.json').write_text(json.dumps(dict(protocol='Preselected16 fully CP-stable and32 masked-stable accepted priority roots; same candidates independently1M/3M nodes; no teacher target copied to successors',seconds=time.monotonic()-started,rows=out),indent=2))
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='output/dynamic-curriculum-20261103');p.add_argument('--binary',default='output/dynamic-curriculum-20261103/engines/stockfish/stockfish-ubuntu-x86-64-avxvnni');p.add_argument('--workers',type=int,default=16)
    p.add_argument('--db');p.add_argument('--model-sha');p.add_argument('--limit',type=int,default=256);p.add_argument('--role',choices=['root','descendant'],default='root');p.add_argument('--discover',action='store_true');p.add_argument('--output',default='output/dynamic-curriculum-20261103/deep-audit-256.json');main(p.parse_args())
