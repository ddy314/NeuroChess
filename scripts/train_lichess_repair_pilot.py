"""Exactly at most100 updates from frozen parent, immutable original guards."""
import torch
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
sys.path.append('/home/xia/code/llmlearn/.venv/lib/python3.13/site-packages')
import concurrent.futures,gc,hashlib,json,time,math
from dataclasses import asdict
import numpy as np
import chess
from scripts.lichess_repair_pilot import OUT,BASE,DATA,SPEC,SEED,status,key_array
from scripts.lichess_search_audit import (PARENT,CANDIDATE,ENGINE,load_net,verify_inputs,sha,read,write,scored_infos,assessment,needs_confirmation)
from scripts.train_pawn_cloud import ReplayRows
from scripts.train_lichess_epoch import choose_rows
from neurochess.guarded_epoch import joint_objective,balance_value_weights,guard_metrics
from neurochess.repair_pilot import functional_anchor,ranking_pairs,pair_loss,complete_analysis
from neurochess.epoch_ablation import validate,batch,screen_search_rows
from neurochess.multihead import MultiheadPlayer
from neurochess.native_mcts import NativeMCTSPlayer
from neurochess.encoding import encode,legal_ids
from neurochess.teacher_uci import FastTeacher
from neurochess.train import atomic_save

def warm(net,fen):
    b=chess.Board(fen);t,m=encode(b);_,ids=legal_ids(b);moves=np.full((64,256),-1,np.int16);moves[:,:len(ids)]=ids
    for n in [1,64]:net.raw_outputs(np.repeat(t[None],n,0),np.repeat(m[None],n,0),moves[:n])

def choose_search(net,fen,lock):
    player=NativeMCTSPlayer(net,**lock['config']);start=time.time()
    try:move=player.choose(chess.Board(fen));stats=player.last_result
    finally:player.close()
    return dict(move=move.uci(),stats=stats,seconds=time.time()-start)

def teacher_compare(fen,parent,candidate):
    engine=FastTeacher(ENGINE,threads=1,hash_mb=64,timeout=360);b=chess.Board(fen)
    try:
        discovery=complete_analysis(engine,b,1000000)
        moves=list(dict.fromkeys([discovery[0]['move'],parent,candidate]))
        first=complete_analysis(engine,b,1000000,moves);scores={r['move']:r['sort_score'] for r in first};best=max(scores.values())
        initial=dict(candidate_regret=best-scores[candidate],baseline_regret=best-scores[parent])
        initial['delta']=initial['candidate_regret']-initial['baseline_regret']
        refine=initial['candidate_regret']>=100 or initial['baseline_regret']>=100 or (parent!=candidate and abs(initial['delta'])>=25)
        deep=None;deep_discovery=None
        if refine:
            deep_discovery=complete_analysis(engine,b,5000000)
            moves=list(dict.fromkeys(moves+[deep_discovery[0]['move']]))
            deep=complete_analysis(engine,b,5000000,moves)
        final=deep or first;scores={r['move']:r['sort_score'] for r in final};best=max(scores.values())
        return dict(initial=initial,final=dict(candidate_regret=best-scores[candidate],baseline_regret=best-scores[parent],delta=scores[parent]-scores[candidate]),
            discovery=discovery,initial_teacher=first,confirmation_discovery=deep_discovery,confirmation_teacher=deep,
            cp_only=all(r['cp'] is not None for r in final),depth24_all=min(r['depth'] for r in final)>=24,training_labels_ready=False)
    finally:engine.quit()

def gate_rows(rows):
    return screen_search_rows([dict(index=r['index'],**{k:r['teacher']['initial'][k] for k in ['candidate_regret','baseline_regret']}) for r in rows])

def screen(model,roots,lock,step):
    net=MultiheadPlayer(model,'cuda');warm(net,roots[0]['fen']);rows=[]
    for i,r in enumerate(roots):
        decision=choose_search(net,r['fen'],lock);teacher=teacher_compare(r['fen'],r['baseline_move'],decision['move'])
        rows.append(dict(index=r['index'],fen=r['fen'],kind=r['kind'],baseline_move=r['baseline_move'],candidate=decision,teacher=teacher))
        if i%8==0:status('repair-search-gate',repair_updates=step,done=i+1,total=len(roots))
    original=[r for r in rows if r['kind']=='frozen48'];extra=[r for r in rows if r['kind']!='frozen48']
    a,b=gate_rows(original),gate_rows(extra)
    result=dict(step=step,passed=a['passed'] and b['passed'],original48=a,regressions_and_retained_controls=b,rows=rows,
        scope='Unchanged fixed1M screen thresholds, locked2s same local GPU;5M confirmation never erases original gate result; mate sort scores are not ordinary CP')
    write(OUT/f'search-gate-{step:03d}.json',result);return result

def full_audit(model,teacher,lock):
    folder=OUT/'final512';folder.mkdir(exist_ok=True)
    audit=ROOT/'output/lichess-search-audit-512-20261105';jobs=read(audit/'cohort.json');manifest=read(audit/'cohort-manifest.json')
    if sha(audit/'cohort.json')!=manifest['cohort_sha256']:raise ValueError('512 cohort changed')
    nets=dict(parent=MultiheadPlayer(teacher,'cuda'),candidate=MultiheadPlayer(model,'cuda'))
    for net in nets.values():warm(net,jobs[0]['fen'])
    def finish(job,decisions):
        path=folder/f"{job['index']:03d}.json"
        if path.exists():return read(path)
        comparison=teacher_compare(job['fen'],decisions['parent']['move'],decisions['candidate']['move'])
        r=dict(index=job['index'],key=job['key'],cohort=job['cohort'],fen=job['fen'],decisions=decisions,teacher=comparison)
        write(path,r);return r
    rows=[];futures=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        for i,job in enumerate(jobs):
            path=folder/f"{i:03d}.json"
            if path.exists():rows.append(read(path));continue
            decisions={}
            for name in (['parent','candidate'] if i%2==0 else ['candidate','parent']):decisions[name]=choose_search(nets[name],job['fen'],lock)
            futures.append(pool.submit(finish,job,decisions))
            for f in futures:
                if f.done():f.result()
            status('repair-final512',repair_updates=100,search_done=i+1,total=512,teacher_done=len(rows)+sum(f.done() for f in futures))
        for f in concurrent.futures.as_completed(futures):rows.append(f.result())
    groups={c:gate_rows([r for r in rows if r['cohort']==c]) for c in ['random','hard']}
    result=dict(completed=True,passed=all(r['passed'] for r in groups.values()),groups=groups,rows=sorted(rows,key=lambda r:r['index']),
        scope='384 random and128 retrospectively model-selected hard development positions; no game clustering, final test, or Elo claim')
    write(folder/'results.json',result);return result

def train(plan):
    ready=read(OUT/'data-ready.json')
    if not ready['passed'] or ready['unique_repair_nodes']!=256:raise ValueError('Strict data gate incomplete')
    for n,h in ready['files_sha256'].items():
        if sha(OUT/n)!=h:raise ValueError('Training data checksum differs: '+n)
    source=read(OUT/'source-manifest.json')
    for n,h in source['files_sha256'].items():
        if sha(ROOT/n)!=h:raise ValueError('Pipeline source differs after verification: '+n)
    if (OUT/'completion.json').exists():return
    if (OUT/'training-started.json').exists():raise RuntimeError('Interrupted training exists; refusing automatic replay or extra updates')
    lock,certificate=verify_inputs();torch.manual_seed(SEED);rng=np.random.default_rng(SEED)
    repair=np.load(OUT/'repair.npy');cp=np.load(OUT/'repair-cp.npy');trusted=np.load(OUT/'repair-trusted.npy');clean=np.load(OUT/'clean.npy')
    reserved=np.unique(np.concatenate([np.load(OUT/'replay-exclusions.npy'),key_array(repair),key_array(clean),key_array(np.load(OUT/'roots.npy'))]))
    replay=ReplayRows(ROOT/'data/evals-v2-5m',reserved,SEED+3)
    prior=read(BASE/'plan-r2.json');validation=dict(new=choose_rows(sorted(DATA.glob('val-*.npy')),8192,prior['seed']),
        old_core=choose_rows(sorted((ROOT/'data/evals-v2-5m').glob('val-*.npy')),8192,prior['seed']+1),old_general=np.load('/home/xia/Downloads/general-validation.npy'))
    for split in ['new','old_core']:
        expected=np.load(BASE/'auto4000/results/A'/f'positions-{split}.npz')['keys']
        if not np.array_equal(validation[split]['key'],expected):raise ValueError('Original monitoring population differs')
    # Fresh development positions, frozen before training, distinct from earlier diagnostic populations.
    blocked=[key_array(validation['new']),np.load(ROOT/'output/lichess-next-step-review-20261105/weights-new_disjoint-parent_trunk__parent_heads.npz')['keys'].copy().view('V16').ravel(),key_array(np.load(ROOT/'output/lichess-search-audit-512-20261105/cohort.npy')),
        key_array(np.load(ROOT/'output/lichess-five-failure-mechanisms-20261105/manual-weight-probes/fresh-validation.npy'))]
    val=np.concatenate([np.load(p,mmap_mode='r') for p in sorted(DATA.glob('val-*.npy'))]);val=val[~np.isin(key_array(val),np.unique(np.concatenate(blocked)))]
    fresh=val[np.random.default_rng(SEED+11).choice(len(val),4096,replace=False)];np.save(OUT/'fresh-validation.npy',fresh);del val
    model=load_net(PARENT).model;model.requires_grad_(True);teacher=load_net(PARENT).model;teacher.requires_grad_(False)
    optimizer=torch.optim.AdamW(model.parameters(),lr=SPEC['lr'],weight_decay=.01,fused=True)
    base={k:validate(model,r,microbatch=32) for k,r in validation.items()};external=certificate['gate']['baseline']
    write(OUT/'initial-validation.json',dict(local=base,original_frozen=external,fresh=validate(model,fresh,microbatch=32)))
    if not guard_metrics(external,base)['passed']:raise ValueError('Parent initial metrics do not meet unchanged original gate')
    original=torch.load(CANDIDATE,map_location='cpu',weights_only=False,mmap=True)['frozen'];roots=[]
    for i,r in enumerate(original):roots.append(dict(index=i,fen=r['fen'],kind='frozen48'))
    extra=read(ROOT/'output/lichess-five-failure-mechanisms-20261105/jobs.json')
    for j in extra:roots.append(dict(index=1000+j['index'],fen=j['fen'],kind=j['diagnostic_group']))
    parentnet=MultiheadPlayer(teacher,'cuda');warm(parentnet,roots[0]['fen'])
    for i,r in enumerate(roots):
        r['baseline_move']=choose_search(parentnet,r['fen'],lock)['move']
        if i%8==0:status('repair-frozen-parent-controls',done=i+1,total=len(roots),repair_updates=0)
    write(OUT/'local-frozen-controls.json',roots)
    write(OUT/'training-started.json',dict(parent_sha256=plan['parent_sha256'],optimizer_reset=True,at=time.time(),max_updates=100,source_manifest_sha256=sha(OUT/'source-manifest.json')))
    before={k:p.detach().cpu().clone() for k,p in model.named_parameters() if k in ['piece.weight','action.1.weight','value.2.weight']}
    used_repair=[];used_old=[];completed=0;last_good=0;state='running';gates=[]
    def payload(step):
        return dict(model=model.state_dict(),model_config=asdict(model.config),step=step,repair_updates=step,source_main_updates=4000,
            optimizer=optimizer.state_dict(),rng=rng.bit_generator.state,replay_rng=replay.rng.bit_generator.state,
            torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),parent_sha256=plan['parent_sha256'],
            clean_presentations=step*160,repair_presentations=step*32,replay_presentations=step*64,repair_unique=len(set(used_repair)),old_unique=len(set(used_old)),
            plan_sha256=sha(OUT/'plan.json'),data_ready_sha256=sha(OUT/'data-ready.json'),lineage='Independent100-step joint repair from frozen parent; original4000 run stays paused',
            default_promoted=False,last_good_step=last_good,requires_explicit_training_plan=True)
    try:
        for step in range(1,101):
            ix=rng.integers(len(repair),size=32);r=np.concatenate([repair[ix],clean[(step-1)*160:step*160],replay.sample(64)])
            old=np.r_[np.zeros(192,bool),np.ones(64,bool)];anchor=np.r_[trusted[ix],np.zeros(160,bool),np.ones(64,bool)]
            cps=np.full((256,256),np.nan,np.float32);cps[:32]=cp[ix];order=rng.permutation(256);r=r[order];old=old[order];anchor=anchor[order];cps=cps[order]
            used_repair.extend(bytes(repair[i]['key']) for i in ix);used_old.extend(bytes(k) for k in r['key'][old]);r=balance_value_weights(r)
            pairs=ranking_pairs(cps);npairs=len(pairs[0]);nt=len(r);nq=int(r['qmask'].sum());nv=float(r['value_weight'].sum());na=int(anchor.sum())
            optimizer.zero_grad(set_to_none=True);lr=SPEC['lr']*min(1,step/10)
            for g in optimizer.param_groups:g['lr']=lr
            sums=np.zeros(6);model.train()
            for start in range(0,256,32):
                rr=r[start:start+32];b=batch(rr);om=torch.as_tensor(old[start:start+32],device='cuda');am=torch.as_tensor(anchor[start:start+32],device='cuda')
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    with torch.no_grad():ref=teacher(b['tokens'],b['meta'],b['moves'])
                    out=model(b['tokens'],b['meta'],b['moves'])
                    objective,parts=joint_objective(out,b,nt,nq,nv,om,tuple(x[om] for x in ref),64)
                    functional=functional_anchor(out,ref,b['moves'],am,na);ranking=pair_loss(out[0],pairs,start,start+len(rr),npairs)
                    loss=objective+functional+ranking
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite repair objective')
                loss.backward();sums+=np.r_[parts.detach().cpu().numpy(),float(functional.detach()),float(ranking.detach())]
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            if not torch.isfinite(norm):raise FloatingPointError('Nonfinite repair gradient')
            optimizer.step();completed=step
            with (OUT/'metrics.jsonl').open('a') as f:f.write(json.dumps(dict(step=step,lr=lr,loss_parts=sums.tolist(),gradient_norm=float(norm),repair_presentations=step*32,clean_presentations=step*160,replay_presentations=step*64,rank_pairs=npairs,trusted_search_anchors=int(trusted[ix].sum()),at=time.time()))+'\n')
            status('repair-training',repair_updates=step,total=100,repair_unique=len(set(used_repair)))
            if step==5:
                params=dict(model.named_parameters());movement={k:float((params[k].detach().cpu()-a).norm()) for k,a in before.items()}
                action=params['action.1.weight'].detach().cpu()-before['action.1.weight']
                movement.update(policy_row=float(action[0].norm()),Q_row=float(action[1].norm()))
                write(OUT/'parameter-movement-step5.json',movement)
                if not all(v>0 for v in movement.values()):raise RuntimeError('Shared trunk or a head did not update')
            if step in SPEC['validation_steps']:
                model.eval();current={k:validate(model,a,microbatch=32) for k,a in validation.items()};fresh_metrics=validate(model,fresh,microbatch=32)
                a,b=guard_metrics(external,current),guard_metrics(base,current);ok=a['passed'] and b['passed']
                gate=dict(step=step,passed=ok,original=a,local_parent=b,current=current,fresh=fresh_metrics)
                if ok and step in [50,100]:gate['search']=screen(model,roots,lock,step);ok=ok and gate['search']['passed'];gate['passed']=ok
                write(OUT/f'gate-{step:03d}.json',gate);gates.append(gate)
                if not ok:
                    atomic_save(payload(step),OUT/'rejected.pt');state='guard-stopped';break
                last_good=step;atomic_save(payload(step),OUT/'last.pt')
                if step in [50,100]:atomic_save(dict(model=model.state_dict(),model_config=asdict(model.config),repair_updates=step,default_promoted=False,parent_sha256=plan['parent_sha256']),OUT/'best.pt')
        else:state='completed100'
    except BaseException:
        atomic_save(payload(completed),OUT/'interrupted.pt');raise
    model.eval();final512=None
    write(OUT/'training-final.json',dict(status=state,completed_updates=completed,at=time.time(),original_main_resumed=False))
    if state=='completed100':final512=full_audit(model,teacher,lock)
    verify_inputs()
    report=dict(completed=True,status=state,completed_updates=completed,gates=gates,final512=final512,
        presentation_counts=dict(clean=completed*160,repair=completed*32,old_replay=completed*64,total=completed*256),
        unique_counts=dict(clean=completed*160,repair=len(set(used_repair)),old_replay=len(set(used_old))),
        repair_pool_size=256,repair_pool_equivalent_passes=completed*32/256,
        effective_batch=256,microbatch=32,all_three_heads_and_trunk_trained=True,
        original_main_resumed=False,automatic_expansion=False,automatic_promotion=False,test_labels_used=False,
        parent_sha256=plan['parent_sha256'],data_ready_sha256=sha(OUT/'data-ready.json'),original_thresholds_unchanged=True,
        limitation='Bounded pilot, selected deep-node training, fixed development monitors; no independent Elo or universal nonregression guarantee')
    write(OUT/'results.json',report)
    files={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name not in ['status.json','completion.json'] and p.suffix not in ['.tmp','.lock','.log']}
    write(OUT/'completion.json',dict(completed=True,report_sha256=sha(OUT/'results.json'),files_sha256=files,finished_at=time.time(),original_main_resumed=False,automatic_expansion=False))
    status('complete',repair_updates=completed,trial_status=state,full512_passed=None if final512 is None else final512['passed'])
