"""Verify the complete audit and repeat bounded severe roots; never resumes training."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
sys.path.append('/home/xia/code/llmlearn/.venv/lib/python3.13/site-packages')
import concurrent.futures,hashlib,json,time,os,traceback
import numpy as np
import chess
from scripts.lichess_search_audit import (OUT,RULE,PARENT,CANDIDATE,verify_inputs,load_net,
    write,read,sha,rank_accuracy,teacher_wl,interval,FastTeacher,ENGINE,scored_infos)
from neurochess.encoding import encode,legal_ids
from neurochess.native_mcts import NativeMCTSPlayer

def main():
    limit=time.time()+6*3600
    # No output files are created before the primary completion manifest freezes.
    while time.time()<limit:
        if (OUT/'failure.json').exists():raise RuntimeError('Primary audit failed; no automatic rerun')
        if (OUT/'completion.json').exists():
            status=read(OUT/'status.json');pid=status['pid']
            if status['phase']=='complete' and not Path(f'/proc/{pid}').exists():break
        time.sleep(10)
    else:raise TimeoutError('Primary audit did not complete within6 hours')
    dest=OUT/'review';dest.mkdir(exist_ok=True)
    if (dest/'completion.json').exists():return
    completion=read(OUT/'completion.json')
    if sha(OUT/'results.json')!=completion['report_sha256']:raise ValueError('Primary report hash mismatch')
    for name,h in completion['files_sha256'].items():
        if sha(OUT/name)!=h:raise ValueError('Primary evidence hash mismatch: '+name)
    lock,_=verify_inputs();jobs=[read(OUT/'search'/f'{i:03d}.json') for i in range(512)]
    teachers=[read(OUT/'teachers'/f'{i:03d}.json') for i in range(512)]
    severe=sorted([t for t in teachers if t['final']['introduced_blunder']],key=lambda t:(-t['final']['delta'],t['index']))[:8]
    selected=[t['index'] for t in severe]
    write(dest/'plan.json',dict(critical_roots=selected,selection='Most severe up to8 final introduced-blunder roots, descending delta then index',
        repeats_per_root=2,config=lock['config'],primary_failures_preserved=True,training_resumed=False))
    torch.set_num_threads(2);torch.set_float32_matmul_precision('high')
    nets={n:load_net(p) for n,p in [('parent',PARENT),('candidate',CANDIDATE)]} if selected else {}
    if nets:
        b=chess.Board();t,m=encode(b);_,ids=legal_ids(b);moves=np.full((64,256),-1,np.int16);moves[:,:len(ids)]=ids
        for net in nets.values():
            for n in [1,64]:net.raw_outputs(np.repeat(t[None],n,0),np.repeat(m[None],n,0),moves[:n])
    repeat_rows=[]
    for index in selected:
        j=jobs[index];ref=teachers[index]['confirmation_teacher'] or teachers[index]['initial_teacher']
        scores={x['move']:x['sort_score'] for x in ref};best=max(scores.values())
        for repeat in [1,2]:
            path=dest/f'repeat-{index:03d}-{repeat}.json'
            if path.exists():repeat_rows.append(read(path));continue
            result=dict(index=index,key=j['key'],repeat=repeat,models={})
            for name in (['parent','candidate'] if (index+repeat)%2==0 else ['candidate','parent']):
                player=NativeMCTSPlayer(nets[name],**lock['config']);start=time.perf_counter()
                try:move=player.choose(chess.Board(j['fen']));stats=player.last_result
                finally:player.close()
                # Unexpected new choices require independent teacher scoring, not imputation.
                result['models'][name]=dict(move=move.uci(),seconds=time.perf_counter()-start,stats=stats,
                    regret=best-scores[move.uci()] if move.uci() in scores else None)
            if any(v['regret'] is None for v in result['models'].values()):
                engine=FastTeacher(ENGINE,hash_mb=64,threads=1,timeout=240)
                try:
                    union=list(dict.fromkeys(list(scores)+[v['move'] for v in result['models'].values()]))
                    extra=scored_infos(engine,chess.Board(j['fen']),5000000,union)
                    result['additional_5M_teacher']=extra
                    scores2={x['move']:x['sort_score'] for x in extra};best2=max(scores2.values())
                    for v in result['models'].values():v['regret']=best2-scores2[v['move']]
                finally:engine.quit()
            result['teacher_covers_both']=all(v['regret'] is not None for v in result['models'].values())
            result['introduced_blunder']=None
            if result['teacher_covers_both']:
                p,c=[result['models'][n]['regret'] for n in ['parent','candidate']]
                result['delta']=c-p;result['introduced_blunder']=c>=100 and c-p>=75
            write(path,result);repeat_rows.append(result)
            write(dest/'status.json',dict(phase='critical-repeats',completed=len(repeat_rows),total=2*len(selected),training_resumed=False))
    qualified={}
    for cohort in ['random','hard']:
        pairs=[(j,t) for j,t in zip(jobs,teachers) if j['cohort']==cohort]
        q=dict(models={},n=len(pairs),cp_only_search_delta=None,
            unresolved_bounds=[t['index'] for _,t in pairs if not t['final']['all_exact']],
            critical_repeats=[r for r in repeat_rows if jobs[r['index']]['cohort']==cohort])
        exact_cp=[t['final']['delta'] for _,t in pairs if t['final']['all_cp'] and t['final']['all_exact']]
        if exact_cp:q['cp_only_search_delta']=interval(exact_cp)
        for name in ['parent','candidate']:
            qe=[];ve=[];ranking={'policy':[],'Q':[]};strata={}
            for j,t in pairs:
                reference=[x for x in (t['confirmation_teacher'] or t['initial_teacher']) if not x['lowerbound'] and not x['upperbound']]
                discovery=(t['confirmation_discovery'] or t['discovery'])[0];d=j['direct'][name]
                qe.extend(abs(d['predicted_Q'][x['move']]-teacher_wl(x)) for x in reference)
                for k in ranking:
                    v=rank_accuracy(d['predicted_policy' if k=='policy' else 'predicted_Q'],reference)
                    if v is not None:ranking[k].append(v)
                if not discovery['lowerbound'] and not discovery['upperbound']:
                    error=abs(d['value']-teacher_wl(discovery));ve.append(error);strata.setdefault(j['selection_stratum'],[]).append(error)
            q['models'][name]=dict(Q_MAE=float(np.mean(qe)),Q_labels=len(qe),Value_MAE=float(np.mean(ve)),Value_roots=len(ve),
                Value_strata={k:dict(n=len(v),mae=float(np.mean(v))) for k,v in strata.items()},
                pairwise_rank_accuracy={k:dict(n=len(v),mean=float(np.mean(v)) if v else None) for k,v in ranking.items()})
        qualified[cohort]=q
    verify_inputs()
    report=dict(primary_results=read(OUT/'results.json'),qualified_metrics=qualified,
        weights_unchanged=True,source_and_primary_manifest_verified=True,optimizer_steps=0,main_training_resumed=False,
        warning='Original1M gate failures are never erased by deeper teachers or repeats. Bound-valued candidates are excluded from qualified Q/rank metrics. WDL is sf16 ply32 conversion of fresh SF18 scores, not observed game outcomes. Random and selected-hard cohorts remain separate. No Elo claim.')
    write(dest/'results.json',report)
    manifest={str(p.relative_to(dest)):sha(p) for p in dest.rglob('*') if p.is_file() and p.name not in ['status.json','completion.json'] and p.suffix not in ['.tmp','.log']}
    write(dest/'completion.json',dict(completed=True,report_sha256=sha(dest/'results.json'),files_sha256=manifest,
        primary_report_sha256=completion['report_sha256'],completed_at=time.time(),training_resumed=False))
    write(dest/'status.json',dict(phase='complete',training_resumed=False))

if __name__=='__main__':
    try:main()
    except BaseException:
        write(OUT/'review-failure.json',dict(traceback=traceback.format_exc(),training_resumed=False,at=time.time()));raise
