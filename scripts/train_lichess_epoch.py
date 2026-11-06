"""One exact Lichess epoch; continual old/new gates, frozen-parent replay anchors.

A failed gate stops the run, leaves the failed candidate separately, and selects
only a previously fully screened best. No default model is ever overwritten.
"""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import fcntl,hashlib,json,math,os,shutil,sys,tarfile,time,traceback
from pathlib import Path
import chess,chess.engine
import numpy as np
import torch
from neurochess.data import DTYPE,FIELDS
from neurochess.encoding import position_key
from neurochess.guarded_epoch import ExactEpoch,replay_count,learning_rate,guard_metrics,joint_objective,GUARD,balance_value_weights
from neurochess.pawn_data import sha,write_json,load_published,contains
from neurochess.train import load_model,losses,atomic_save
from neurochess.persistence import RunMirror
from scripts.train_pawn_cloud import ReplayRows


def choose_rows(files,size,seed):
    """A fixed hash/seed sample spanning all shards; no prefix-only validation."""
    arrays=[np.load(p,mmap_mode='r',allow_pickle=False) for p in files]
    sizes=np.array([len(a) for a in arrays]);ends=np.cumsum(sizes)
    picks=np.sort(np.random.default_rng(seed).choice(int(ends[-1]),min(size,int(ends[-1])),replace=False))
    rows=[]
    for i,a in enumerate(arrays):
        lo=0 if i==0 else ends[i-1];local=picks[(picks>=lo)&(picks<ends[i])]-lo
        if len(local):rows.append(a[local])
    return np.concatenate(rows)


def train(plan,resume=False):
    work=Path(plan['work']);save=Path(plan['save']);source=Path(plan['data'])
    if not os.path.ismount('/content/drive'):raise OSError('Drive mount required')
    work.mkdir(exist_ok=True,parents=True);save.mkdir(exist_ok=True,parents=True)
    lock=(work/'training.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (save/'last.pt').exists() and not resume:raise ValueError('Existing training run; resume explicitly')
    mirror=RunMirror(work,save,'/content/drive',exclude_top_level=['data','old-core','cache','parent.pt','old-core.tar.gz'])
    started=time.time();step=new_seen=old_seen=0;phase='preparing';last_sync=0;success=False;stopped=False
    plan_sha=hashlib.sha256(json.dumps(plan,sort_keys=True).encode()).hexdigest()
    def status(**extra):
        nonlocal last_sync
        write_json(work/'status.json',dict(phase=phase,step=step,new_positions=new_seen,target_new_positions=plan['train_positions'],
            epoch_fraction=new_seen/plan['train_positions'],replay_positions=old_seen,updated_at=time.time(),seconds=time.time()-started,
            default_promoted=False,**extra))
        if time.time()-last_sync>=30 or phase!='training':mirror.sync();last_sync=time.time()
    def append(name,row):
        with (work/name).open('a') as f:f.write(json.dumps(row,default=lambda x:x.item() if isinstance(x,np.generic) else (_ for _ in ()).throw(TypeError(type(x).__name__)))+'\n')
    def batch(rows):return {k:torch.as_tensor(np.array(rows[k],copy=True),device='cuda') for k in FIELDS}
    try:
        torch.set_num_threads(2);torch.manual_seed(plan['seed']);torch.set_float32_matmul_precision('high')
        if not torch.cuda.is_available() or 'L4' not in torch.cuda.get_device_name():raise RuntimeError('Expected the single authorized L4')
        write_json(work/'plan.json',plan);write_json(work/'guard-spec.json',GUARD);status()
        ready_path=source/'ready.json'
        if sha(ready_path)!=plan['ready_sha256'] or sha(source/'completion.json')!=plan['audit_sha256']:
            raise ValueError('Clean dataset manifest/certificate changed')
        ready=json.loads(ready_path.read_text());audit=json.loads((source/'completion.json').read_text())
        if not ready['complete'] or not audit['passed'] or ready['counts']!=dict(train=9800000,val=100000,test=100000):
            raise ValueError('Dataset not fully audited')
        data=work/'data';data.mkdir(exist_ok=True)
        # Materialize only numeric training shards, never source-evidence labels
        # as alternative targets. Every compressed and decompressed SHA checked.
        done=0
        def materialize(item):
            a=load_published(item['data'],source,data);del a
            return item['split'],data/item['data']['name'].removesuffix('.zst')
        files={s:[] for s in ['train','val','test']}
        with ThreadPoolExecutor(max_workers=4) as pool:
            for split,path in pool.map(materialize,ready['shards']):
                files[split].append(path);done+=1
                if done%50==0:status(prepared_shards=done,total_shards=len(ready['shards']))
        files={k:sorted(v) for k,v in files.items()}
        if sum(len(np.load(p,mmap_mode='r')) for p in files['train'])!=plan['train_positions']:raise ValueError('Training count mismatch')
        reserved=[]
        for split in ['val','test']:
            reserved.extend(np.load(p,mmap_mode='r')['key'].copy().view('V16').reshape(-1) for p in files[split])
        local_parent=work/'parent.pt';shutil.copyfile(plan['parent'],local_parent)
        if sha(local_parent)!=plan['parent_sha256']:raise ValueError('Immutable parent hash differs')
        archive=work/'old-core.tar.gz';shutil.copyfile(plan['replay_archive'],archive)
        if sha(archive)!=plan['replay_sha256']:raise ValueError('Replay archive changed')
        old=work/'old-core';old.mkdir(exist_ok=True)
        with tarfile.open(archive) as package:package.extractall(old,filter='data')
        general=work/'general-validation.npy';shutil.copyfile(plan['general_validation'],general)
        if sha(general)!=plan['general_validation_sha256']:raise ValueError('General validation changed')
        general_rows=np.load(general,allow_pickle=False);reserved.append(general_rows['key'].copy().view('V16').reshape(-1))
        old_val=sorted(old.glob('val-*.npy'))
        reserved.extend(np.load(p,mmap_mode='r')['key'].copy().view('V16').reshape(-1) for p in old_val)
        frozen=json.loads(Path(plan['frozen_roots']).read_text())['roots']
        reserved.append(np.array([position_key(chess.Board(x['fen'])) for x in frozen],dtype='V16'))
        excluded=np.unique(np.concatenate(reserved))
        replay=ReplayRows(old,excluded,plan['seed']+3)
        # New corpus is certified fresh against all old keys, independently
        # verify our exact new heldouts/frozen roots are never optimization data.
        for p in files['train']:
            keys=np.load(p,mmap_mode='r')['key'].copy().view('V16').reshape(-1)
            if contains(excluded,keys).any():raise ValueError('New training/monitor key leakage')
        write_json(work/'data-verification.json',dict(passed=True,new_train=plan['train_positions'],old_replay=replay.total,
            heldout_excluded_keys=len(excluded),exact_overlap=0,ready_sha256=sha(ready_path),
            epoch='One traversal of each of 9,800,000 new training rows; replay additional and separately counted',test_usage='keys for isolation; labels only evaluated once at final selection'))
        validation=dict(new=choose_rows(files['val'],8192,plan['seed']),old_core=choose_rows(old_val,8192,plan['seed']+1),old_general=general_rows)
        model,state=load_model(local_parent,'cuda')
        if state.get('architecture') or not model.config.history_free:raise ValueError('Require immutable single-encoder history-free parent')
        origin=state.get('origin_step',0)+state['step'];del state
        model.requires_grad_(True)
        teacher,_=load_model(local_parent,'cuda');teacher.eval();teacher.requires_grad_(False)
        optimizer=torch.optim.AdamW(model.parameters(),lr=plan['lr'],weight_decay=.01,fused=True)
        lineage=dict(parent_sha256=plan['parent_sha256'],parent_origin_step=origin,architecture='single-encoder',heads=['policy','Q','value'],
                     optimizer_reset_at_new_stage=True,guard_scope='Fixed validation and finite frozen search probes; not proof of universal strength preservation',default_promoted=False)
        write_json(work/'lineage.json',lineage)
        write_json(work/'runtime.json',dict(gpu=torch.cuda.get_device_name(),parameters=sum(p.numel() for p in model.parameters()),precision='BF16',microbatch=plan['microbatch']))
        def validate(name,rows,network=None):
            network=model if network is None else network
            network.eval();tot=np.zeros(7);truth=[];pred=[]
            with torch.inference_mode():
                for i in range(0,len(rows),plan['microbatch']):
                    r=rows[i:i+plan['microbatch']];b=batch(r)
                    with torch.autocast('cuda',dtype=torch.bfloat16):out=network(b['tokens'],b['meta'],b['moves'])
                    loss,parts=losses(out,b,.5,1.);n=len(r);qm=b['qmask'];v=out[2].softmax(-1)
                    tot+=np.array([float(loss)*n,float(parts['policy'])*n,float(parts['q'])*n,float(parts['value'])*n,
                        int((out[0].argmax(-1)==b['policy'].argmax(-1)).sum()),float(((out[1]-b['q'].float()).abs()*qm).sum()),int(qm.sum())])
                    pred.extend((v[:,0]-v[:,2]).cpu().tolist());truth.extend((b['value'][:,0]-b['value'][:,2]).cpu().tolist())
            t=np.array(truth);err=abs(np.array(pred)-t)
            result=dict(step=step,split=name,positions=len(rows),loss=tot[0]/len(rows),policy_loss=tot[1]/len(rows),q_loss=tot[2]/len(rows),value_loss=tot[3]/len(rows),top1=tot[4]/len(rows),q_mae=tot[5]/max(1,tot[6]),value_mae=float(err.mean()),
                value_strata={k:dict(n=int(m.sum()),mae=float(err[m].mean()) if m.any() else None) for k,m in dict(winning=t>.25,neutral=abs(t)<=.25,losing=t<-.25).items()})
            append('validation.jsonl',result);return result
        def frozen_search():
            from neurochess.multihead import MultiheadPlayer
            from neurochess.native_mcts import NativeMCTSPlayer
            options=json.loads(Path('configs/search-baseline-lock.json').read_text())['config']
            choices=[];model.eval();net=MultiheadPlayer(model,'cuda')
            with chess.engine.SimpleEngine.popen_uci(plan['stockfish']) as sf:
                sf.configure({'Threads':1,'Hash':64,'Skill Level':20,'UCI_LimitStrength':False})
                for i,root in enumerate(frozen):
                    board=chess.Board(root['fen']);player=NativeMCTSPlayer(net,**options);tick=time.time()
                    try:move=player.choose(board);stats=player.last_result
                    finally:player.close()
                    baseline_move=root['baseline_move'] if 'baseline_move' in root else None
                    # Baseline choices are measured on this same GPU at step 0.
                    if baseline_move is None:baseline_move=move.uci()
                    proposal=sf.analyse(board,chess.engine.Limit(nodes=100000),game=object())['pv'][0]
                    moves=list(dict.fromkeys([proposal,board.parse_uci(baseline_move),move]))
                    info=sf.analyse(board,chess.engine.Limit(nodes=1000000),root_moves=moves,multipv=len(moves),game=object())
                    scores={x['pv'][0].uci():x['score'].pov(board.turn).score(mate_score=100000) for x in info};best=max(scores.values())
                    choices.append(dict(index=i,fen=root['fen'],move=move.uci(),baseline_move=baseline_move,
                        candidate_regret=best-scores[move.uci()],baseline_regret=best-scores[baseline_move],scores=scores,search=stats,seconds=time.time()-tick))
                    if step==0:root['baseline_move']=move.uci()
            bad=[r['index'] for r in choices if r['candidate_regret']>=100 and r['candidate_regret']-r['baseline_regret']>=75]
            mean_delta=float(np.mean([r['candidate_regret']-r['baseline_regret'] for r in choices]))
            result=dict(step=step,passed=not bad and mean_delta<=10.,introduced_blunders=bad,mean_regret_delta_cp=mean_delta,rows=choices,
                        scope='48 development roots; same locked 2s search and GPU; fresh finite 1M-node candidate-union teachers; no Elo or universal guarantee')
            write_json(work/f'search-gate-{step:06d}.json',result);mirror.sync();return result
        baseline=None;best_loss=math.inf;last_good_step=0
        def payload():
            return dict(model=model.state_dict(),model_config=asdict(model.config),step=origin+step,origin_step=origin,
                epoch_updates=step,optimizer=optimizer.state_dict(),new_positions=new_seen,replay_positions=old_seen,
                replay_rng=replay.rng.bit_generator.state,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
                baseline=baseline,best_loss=best_loss,last_good_step=last_good_step,frozen=frozen,plan_sha256=plan_sha,lineage=lineage,default_promoted=False)
        def weights(path):atomic_save(dict(model=model.state_dict(),model_config=asdict(model.config),step=origin+step,
             epoch_updates=step,new_positions=new_seen,lineage=lineage,default_promoted=False),path)
        def commit_checkpoint():
            atomic_save(payload(),work/'last.pt');mirror.sync()
            hashes={}
            for name in ['last.pt','best.pt']:
                expected=sha(work/name)
                if sha(save/name)!=expected:raise OSError('Checkpoint readback failed: '+name)
                hashes[name]=expected
            write_json(work/'checkpoint-verification.json',dict(step=step,new_positions=new_seen,sha256=hashes,verified_at=time.time()))
            mirror.sync()
        if resume:
            proof=json.loads((work/'checkpoint-verification.json').read_text())
            for name,want in proof['sha256'].items():
                if sha(work/name)!=want:raise ValueError('Resume checkpoint checksum differs: '+name)
            s=torch.load(work/'last.pt',map_location='cpu',weights_only=False)
            if s['plan_sha256']!=plan_sha:raise ValueError('Resume configuration changed')
            model.load_state_dict(s['model']);optimizer.load_state_dict(s['optimizer']);step=s['epoch_updates'];new_seen=s['new_positions'];old_seen=s['replay_positions']
            replay.rng.bit_generator.state=s['replay_rng'];torch.set_rng_state(s['torch_rng']);torch.cuda.set_rng_state_all(s['cuda_rng'])
            baseline=s['baseline'];best_loss=s['best_loss'];last_good_step=s['last_good_step'];frozen=s['frozen'];del s
            write_json(work/'resume-verification.json',dict(step=step,new_cursor=new_seen,optimizer_retained=True,random_states_restored=True))
        else:
            phase='baseline-validation';status();baseline={k:validate(k,v) for k,v in validation.items()}
            phase='baseline-search';status();frozen_search();best_loss=baseline['new']['loss']
            weights(work/'best.pt');weights(work/'selected.pt');commit_checkpoint()
        stream=ExactEpoch(files['train'],plan['seed'],new_seen)
        total_steps=math.ceil(plan['train_positions']/plan['new_per_update'])
        before={k:p.detach().cpu().clone() for k,p in model.named_parameters() if k in ['piece.weight','action.1.weight','value.2.weight']}
        train_start=time.time();start_new=new_seen
        while new_seen<plan['train_positions']:
            phase='training';status();nnew=min(plan['new_per_update'],plan['train_positions']-new_seen);nold=replay_count(nnew)
            rows=np.concatenate([stream.next(nnew),replay.sample(nold)]);is_old=np.r_[np.zeros(nnew,bool),np.ones(nold,bool)]
            order=np.random.default_rng(np.random.SeedSequence([plan['seed'],step])).permutation(len(rows));rows=rows[order];is_old=is_old[order]
            rows=balance_value_weights(rows)
            nt=len(rows);nq=int(rows['qmask'].sum());nv=float(rows['value_weight'].sum());tot=np.zeros(4)
            model.train();optimizer.zero_grad(set_to_none=True);lr=learning_rate(step+1,total_steps,plan['lr'],plan['warmup'])
            for group in optimizer.param_groups:group['lr']=lr
            for i in range(0,nt,plan['microbatch']):
                r=rows[i:i+plan['microbatch']];b=batch(r);mask=torch.as_tensor(is_old[i:i+len(r)],device='cuda')
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    with torch.no_grad():reference=teacher(b['tokens'][mask],b['meta'][mask],b['moves'][mask]) if mask.any() else None
                    out=model(b['tokens'],b['meta'],b['moves']);objective,parts=joint_objective(out,b,nt,nq,nv,mask,reference,nold)
                if not torch.isfinite(objective):raise FloatingPointError('Nonfinite training objective')
                objective.backward();tot+=parts.cpu().numpy()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            if not torch.isfinite(norm):raise FloatingPointError('Nonfinite gradient norm')
            optimizer.step();step+=1;new_seen+=nnew;old_seen+=nold
            if step%10==0:
                append('metrics.jsonl',dict(step=step,new_positions=new_seen,replay_positions=old_seen,loss=tot.tolist(),lr=lr,gradient_norm=float(norm),timestamp=time.time(),
                    new_positions_per_second=(new_seen-start_new)/max(1,time.time()-train_start),gpu_peak_gib=torch.cuda.max_memory_allocated()/2**30));status()
            final=new_seen==plan['train_positions']
            if step==plan['pilot_steps'] or step%plan['validate_every']==0 or final:
                phase='validating';status();current={k:validate(k,v) for k,v in validation.items()};gate=guard_metrics(baseline,current)
                full=step==plan['pilot_steps'] or step%plan['search_every']==0 or final
                if gate['passed'] and full:
                    phase='search-regression';status();search=frozen_search();gate['search_passed']=search['passed'];gate['passed'] &= search['passed']
                if step==plan['pilot_steps']:
                    params=dict(model.named_parameters());delta={k:(params[k].detach().cpu()-v).abs() for k,v in before.items()}
                    movement=dict(shared=float(delta['piece.weight'].max()),policy=float(delta['action.1.weight'][0].max()),Q=float(delta['action.1.weight'][1].max()),value=float(delta['value.2.weight'].max()))
                    gate['parameter_changes']=movement;gate['passed'] &= all(v>0 for v in movement.values())
                gate.update(step=step,new_positions=new_seen,current=current,baseline=baseline,full_search_screen=full)
                if not gate['passed']:
                    weights(work/'rejected.pt');mirror.sync()
                append('gates.jsonl',gate);write_json(work/'latest-gate.json',gate)
                if not gate['passed']:
                    stopped=True;phase='guard-stopped';status(failed_checks=gate.get('failed',[]));break
                last_good_step=step
                if full and current['new']['loss']<best_loss:
                    best_loss=current['new']['loss'];weights(work/'best.pt')
                    write_json(work/'best-selection.json',dict(step=step,new_positions=new_seen,new_loss=best_loss,all_metric_gates_passed=True,frozen_search_passed=True))
                commit_checkpoint();phase='training';status()
        phase='final-evaluation';status()
        best,best_state=load_model(work/'best.pt','cpu');selected_update=best_state.get('epoch_updates',0);model.load_state_dict(best.state_dict());del best,best_state
        selected={k:validate(k,v) for k,v in validation.items()}
        if not guard_metrics(baseline,selected)['passed']:raise RuntimeError('Selected checkpoint no longer passes saved validation gates')
        # Heldout labels are touched only after checkpoint selection is finished.
        full_val=np.concatenate([np.load(p) for p in files['val']]);full_test=np.concatenate([np.load(p) for p in files['test']])
        heldout=dict(validation=validate('final_full_validation',full_val),test=validate('final_test',full_test),
            parent_validation=validate('parent_full_validation',full_val,teacher),parent_test=validate('parent_final_test',full_test,teacher))
        shutil.copyfile(work/'best.pt',work/'selected.pt');mirror.sync()
        verified={}
        for name in ['last.pt','best.pt','selected.pt','plan.json','lineage.json']:
            if name=='lineage.json' and not (work/name).exists():write_json(work/name,lineage);mirror.sync()
            if sha(work/name)!=sha(save/name):raise OSError('Persistent readback mismatch: '+name)
            verified[name]=sha(save/name)
        report=dict(completed_epoch=not stopped and new_seen==plan['train_positions'],guard_stopped=stopped,new_positions=new_seen,
            target_new_positions=plan['train_positions'],replay_positions=old_seen,optimizer_updates=step,last_good_step=last_good_step,selected_epoch_update=selected_update,
            selected=selected,full_heldout=heldout,verified_persistent_files=verified,default_promoted=False,playing_strength_improvement_established=False,
            selection='Best new validation loss among checkpoints passing both metric and frozen-search screens; parent retained as fallback',finished_at=time.time())
        write_json(work/'completion.json',report);phase='guard-stopped' if stopped else 'complete';status();mirror.sync();success=True
    except BaseException:
        failure=traceback.format_exc()
        if step>0 and 'payload' in locals():
            try:
                emergency=payload();emergency['unvalidated_candidate']=True
                atomic_save(emergency,work/'interrupted.pt');mirror.sync()
            except BaseException:failure+='\nEmergency persistence failure:\n'+traceback.format_exc()
        phase='failed';write_json(work/'failure.json',dict(step=step,new_positions=new_seen,traceback=failure));status();raise
    finally:
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
        if (work/'console.log').exists():shutil.copyfile(work/'console.log',save/'console.log')
        if success or (save/'failure.json').exists():
            from neurochess.autostop import request_unassign,find_runtime_address
            write_json(save/'shutdown-request.json',dict(phase=phase,verified_completion=success,requested_at=time.time()))
            request_unassign(find_runtime_address())


if __name__=='__main__':train(json.loads(Path(sys.argv[1]).read_text()),resume='--resume' in sys.argv[2:])
