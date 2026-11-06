"""Paired teacher regret, head interventions and future-state value diagnostics."""
import argparse,importlib.util,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
if importlib.util.find_spec('chess') is None:sys.path.extend(str(p) for p in (ROOT/'.venv/lib').glob('python*/site-packages'))
import chess,numpy as np
from scripts.analyze_joint_checkpoint import write,paired_interval,load
from scripts.targeted_mistakes import classify
from neurochess.targets import score_targets
from neurochess.encoding import encode,legal_ids
from neurochess.multihead import MultiheadPlayer,fuse_heads


def board_for(j):
    b=chess.Board(j['initial_fen'])
    for u in j['history']:b.push_uci(u)
    return b


def regret(meta,move,value):
    b=board_for(meta['job']);d=classify(meta['teacher'],move,value,b.turn)
    scores={p['line'].split()[0]:score_targets(p,b.turn) for p in meta['teacher']}
    d['cp_regret_capped1000']=min(1000,d['regret_cp']) if d['regret_cp'] is not None else (1000 if d['policy_error'] else 0)
    d['wdl_regret']=max(v[1] for v in scores.values())-scores[move][1]
    return d


def summarize(out):
    teachers={r['job'].get('case_id',r['job']['key']):r for r in [json.loads(l) for l in (out/'deep-teacher.jsonl').read_text().splitlines()]}
    jobs=json.loads((out/'search-jobs.json').read_text());decisions=[json.loads(l) for l in (out/'search-decisions.jsonl').read_text().splitlines()]
    a=np.load(out/'new_test.npy');index={bytes(k).hex():i for i,k in enumerate(a['key'])};raw=[np.load(out/f'new_test-{l}-raw.npy') for l in ['parent','final']]
    results=[];q_errors=[]
    for j in jobs:
        m=teachers[j['key']];i=index[j['key']];b=board_for(j);legal=list(b.legal_moves)
        for which,label in enumerate(['parent','final']):
            r=raw[which][i:i+1];f,_,v,support=fuse_heads(r[:,:256],r[:,256:512],r[:,512:],a[i:i+1]['moves']>=0)
            move=legal[int(f.argmax())].uci();d=regret(m,move,float(v[0]))
            results.append(dict(key=j['key'],group=j['group'],model=label,mode='direct',sources='000' if which==0 else '111',repeat=0,stable=m['stable'],**d))
            for pv in m['teacher']:
                u=pv['line'].split()[0];k=legal.index(b.parse_uci(u));target=score_targets(pv,b.turn)[1]
                q_errors.append(dict(key=j['key'],group=j['group'],model=label,move=u,originally_labelled=bool(a[i]['qmask'][k]),supported=bool(support[0,k]),target=target,prediction=float(r[0,256+k]),error=abs(float(r[0,256+k])-target)))
        for d in [r for r in decisions if r['key']==j['key']]:
            value=float(raw[int(d['sources'][2])][i,512]-raw[int(d['sources'][2])][i,514]);r=regret(m,d['move'],value)
            results.append(dict(key=j['key'],group=j['group'],model='parent' if d['sources']=='000' else 'final' if d['sources']=='111' else d['sources'],mode=d['mode'],sources=d['sources'],repeat=d['repeat'],stable=m['stable'],seconds=d['seconds'],**r))
    report=dict(n_roots=len(jobs),stable_roots=sum(teachers[j['key']]['stable'] for j in jobs),teacher_refined_roots=sum(teachers[j['key']]['refined'] for j in jobs),primary={},head_swaps={},q_generalization={})
    def compare(rows,field):
        lookup={(r['key'],r['model']):r for r in rows};keys=sorted({r['key'] for r in rows})
        x=np.array([lookup[k,'parent'][field] for k in keys],float);y=np.array([lookup[k,'final'][field] for k in keys],float)
        return dict(parent=float(x.mean()),final=float(y.mean()),**paired_interval(y-x,keys))
    for mode in ['direct','locked2s']:
        for selection in ['all','teacher_stable']:
            rows=[r for r in results if r['mode']==mode and r['repeat']==0 and (selection=='all' or r['stable'])]
            report['primary'][mode+'_'+selection]=dict(n=len(rows)//2,**{field:compare(rows,field) for field in ['cp_regret_capped1000','wdl_regret','policy_error']})
    for sources in sorted({r['sources'] for r in results if r['mode']=='head-swap-fixed512'}):
        rows=[r for r in results if r['mode']=='head-swap-fixed512' and r['sources']==sources]
        report['head_swaps'][sources]=dict(n=len(rows),mean_cp_regret=float(np.mean([r['cp_regret_capped1000'] for r in rows])),policy_error_fraction=float(np.mean([r['policy_error'] for r in rows])))
    report['head_effects']={}
    for i,head in enumerate(['policy','Q','value']):
        per_root={}
        for row in [r for r in results if r['mode']=='head-swap-fixed512' and r['sources'][i]=='0']:
            other=row['sources'][:i]+'1'+row['sources'][i+1:]
            counterpart=next(r for r in results if r['mode']==row['mode'] and r['key']==row['key'] and r['sources']==other)
            per_root.setdefault(row['key'],[]).append(counterpart['cp_regret_capped1000']-row['cp_regret_capped1000'])
        report['head_effects'][head]=paired_interval([np.mean(v) for v in per_root.values()],list(per_root))
    for labelled in [True,False]:
        per_model={}
        for model in ['parent','final']:
            rows=[r for r in q_errors if r['model']==model and r['originally_labelled']==labelled]
            per_model[model]=dict(candidate_count=len(rows),mean_absolute_error=float(np.mean([r['error'] for r in rows])))
        report['q_generalization']['originally_labelled' if labelled else 'originally_unlabelled']=per_model
    # Repeated moves estimate timing/search variation; repeats are not new games.
    repeat_rows=[r for r in results if r['mode']=='locked2s'];report['search_variability']={}
    for model in ['parent','final']:
        roots={r['key'] for r in repeat_rows if r['model']==model and r['repeat']>0};spreads=[]
        for k in roots:
            values=[r['cp_regret_capped1000'] for r in repeat_rows if r['model']==model and r['key']==k];spreads.append(max(values)-min(values))
        report['search_variability'][model]=dict(repeated_roots=len(roots),mean_max_minus_min_cp=float(np.mean(spreads)),max_spread_cp=float(max(spreads)))
    write(out/'search-assessment.json',report);write(out/'decision-teacher-details.json',results);write(out/'q-deep-teacher-errors.json',q_errors)


def future(out,old,new):
    import torch
    torch.set_num_threads(2);torch.set_float32_matmul_precision('high')
    teachers=[json.loads(l) for l in (out/'deep-teacher.jsonl').read_text().splitlines()];children=[m for m in teachers if 'depth' in m['job']];raws=[]
    boards=[board_for(m['job']) for m in children];tms=[encode(b) for b in boards];ids=np.full((len(boards),256),-1,np.int16)
    for i,b in enumerate(boards):_,legal=legal_ids(b);ids[i,:len(legal)]=legal
    for path in [old,new]:
        model,_=load(path);net=MultiheadPlayer(model,'cuda');parts=[]
        for start in range(0,len(boards),16):parts.append(net.raw_outputs(np.stack([t[0] for t in tms[start:start+16]]),np.stack([t[1] for t in tms[start:start+16]]),ids[start:start+16]))
        raws.append(np.concatenate(parts));del net,model;torch.cuda.empty_cache()
    details=[]
    roots={m['job']['key']:m for m in teachers if 'depth' not in m['job']};rootarr=np.load(out/'new_test.npy');index={bytes(k).hex():i for i,k in enumerate(rootarr['key'])};rootraw=[np.load(out/f'new_test-{l}-raw.npy') for l in ['parent','final']]
    edges=[]
    for i,m in enumerate(children):
        b=boards[i];target=max(score_targets(p,b.turn)[1] for p in m['teacher']);values=[float(r[i,512]-r[i,514]) for r in raws]
        details.append(dict(job=m['job'],stable=m['stable'],target=target,parent=values[0],final=values[1],parent_error=abs(values[0]-target),final_error=abs(values[1]-target)))
        j=m['job']
        if j['depth']==1:
            root=roots[j['root_key']];rb=board_for(root['job']);legal=list(rb.legal_moves);move=b.peek();k=legal.index(move);pred=[float(r[index[j['root_key']],256+k]) for r in rootraw]
            pv=next((p for p in root['teacher'] if p['line'].split()[0]==move.uci()),None)
            if pv:edges.append(dict(group=j['group'],branch=j['branch'],stable=root['stable'] and m['stable'],teacher_residual=abs(score_targets(pv,rb.turn)[1]+target),parent_residual=abs(pred[0]+values[0]),final_residual=abs(pred[1]+values[1])))
    report=dict(n_states=len(details),unique_games=len({r['job']['group'] for r in details}),by_depth={},bellman_edges=edges)
    for depth in [1,2,4]:
        a=[r for r in details if r['job']['depth']==depth];delta=[r['final_error']-r['parent_error'] for r in a]
        report['by_depth'][str(depth)]=dict(n=len(a),parent_mae=float(np.mean([r['parent_error'] for r in a])),final_mae=float(np.mean([r['final_error'] for r in a])),**paired_interval(delta,[r['job']['group'] for r in a]))
    write(out/'future-value-assessment.json',report);write(out/'future-value-details.json',details)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['search','future']);p.add_argument('--out',required=True);p.add_argument('--old');p.add_argument('--new');a=p.parse_args()
    if a.phase=='search':summarize(Path(a.out))
    else:future(Path(a.out),Path(a.old),Path(a.new))
