"""One exact policy/Q-value pair:750 head repair, bounded gate, conditional4000 continuation."""
import os,sys,json,hashlib,tarfile,threading,time,traceback,fcntl
from pathlib import Path
ROOT=Path('/content/neurochess');WORK=Path('/content/search-repair-run');MOUNT=Path('/content/drive')
SAVE=MOUNT/'MyDrive/neurochess/search-repair-v1/training'
OLD_SHA='e4ed8978db91eeef8360c4dea87faf7a4f826325bce45aa7efc024692d585ef7'
NEW_SHA='343d19699d86f3e0b8e2ff1a9cf3237a403fb25c4e0af98ad7d63a72d6c78247'


def gate_decision(reference,candidate):
    checks=dict(diagnostic_search=candidate['diagnostics']['correct']>=reference['diagnostics']['correct'],
        untouched_search=candidate['holdout']['correct']>=reference['holdout']['correct']-1,
        general_ce=candidate['general']['ce']<=reference['general']['ce']*1.03,
        general_mae=candidate['general']['mae']<=reference['general']['mae']+.02,
        balanced_mae=candidate['balanced']['mae']<=reference['balanced']['mae']+.005,
        per_value_stratum=all(candidate['balanced']['by_stratum'][k]['mae']<=reference['balanced']['by_stratum'][k]['mae']+.03 for k in ['winning','neutral','losing']),
        opponent_optimism=candidate['opponent_mean_value']<=max(0.,reference['opponent_mean_value']+.05))
    return dict(passed=all(checks.values()),checks=checks,scope='Bounded fixed-position first-move/value gate, not an Elo or universal quality guarantee.')


def main():
    if not os.path.ismount(MOUNT):raise RuntimeError('Mount Drive with CLI')
    lock=open('/tmp/search-repair-run.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    os.chdir(ROOT);sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
    import numpy as np,torch,chess
    from torch.nn import functional as F
    from neurochess.policy_value_pair import from_original_states,checkpoint,policy_digest
    from neurochess.multihead import MultiheadPlayer
    from neurochess.native_mcts import NativeMCTSPlayer
    from neurochess.encoding import encode,legal_ids
    from neurochess.persistence import RunMirror
    from neurochess.train import atomic_save
    from fresh_hard_data import atomic_json
    from search_repair_prepare_cloud import copy,sha
    WORK.mkdir(exist_ok=True);SAVE.mkdir(parents=True,exist_ok=True)
    mirror=RunMirror(WORK,SAVE,MOUNT,exclude_top_level=['data','core','data.tar.gz','core.tar.gz']);sync_lock=threading.Lock();stop=threading.Event();phase=['setup']
    torch.set_num_threads(2);torch.set_float32_matmul_precision('high');torch.manual_seed(20261102)
    def sync():
        with sync_lock:mirror.sync()
    def backup():
        while not stop.wait(60):
            try:atomic_json(WORK/'pipeline-status.json',dict(phase=phase[0],updated_at=time.time()));sync()
            except Exception as e:print('Persistence retry',repr(e),flush=True)
    thread=threading.Thread(target=backup,daemon=True);thread.start()
    try:
        assert 'L4' in torch.cuda.get_device_name()
        prep=MOUNT/'MyDrive/neurochess/search-repair-v1/preparation';ready=json.loads((prep/'ready.json').read_text())
        if not ready['complete'] or not ready['audit']['complete']:raise ValueError('Data not audited')
        copy(prep/'data.tar.gz',WORK/'data.tar.gz');assert sha(WORK/'data.tar.gz')==ready['archive_sha256']
        with tarfile.open(WORK/'data.tar.gz') as tar:tar.extractall(WORK,filter='data')
        copy(prep/'ready.json',WORK/'data-delivery.json')
        paths=[MOUNT/'MyDrive/neurochess/fresh-v4-training/fresh/best.pt',MOUNT/'MyDrive/neurochess/curriculum-v3/selected.pt']
        checkpoints=[]
        for name,path,expected in zip(['policy-parent.pt','value-parent.pt'],paths,[NEW_SHA,OLD_SHA]):
            copy(path,WORK/name);assert sha(WORK/name)==expected;checkpoints.append(torch.load(WORK/name,map_location='cpu',weights_only=False,mmap=True))
        model=from_original_states(*checkpoints,device='cuda');del checkpoints;model.eval();policy_sha=policy_digest(model)
        lineage=dict(policy_q_parent_sha256=NEW_SHA,value_parent_sha256=OLD_SHA,exact_original_trunks=True,training_branch_count=1)
        plan=json.loads((ROOT/'configs/search-repair-cloud-v1.json').read_text());atomic_json(WORK/'experiment-plan.json',plan)
        initial=WORK/'initial-pair.pt';atomic_save(checkpoint(model,0,lineage),initial);sync()
        assert sha(initial)==sha(SAVE/'initial-pair.pt')
        data=WORK/'data'
        def rows(folder,split):return np.concatenate([np.load(p) for p in sorted(folder.glob(split+'-*.npy'))])
        train_rows=rows(data/'train','train');val_rows=rows(data/'val','val');general_rows=rows(data/'general-val','val')
        amp=lambda:torch.autocast('cuda',dtype=torch.bfloat16)
        def value_metrics(records):
            predictions=[];ces=[];maes=[];model.eval()
            with torch.inference_mode():
                for start in range(0,len(records),128):
                    r=records[start:start+128];t=torch.as_tensor(r['tokens'].copy(),device='cuda');m=torch.as_tensor(r['meta'].copy(),device='cuda');target=torch.as_tensor(r['value'].copy(),device='cuda')
                    with amp():logits=model.value_only(t,m)
                    ce=-(target*logits.float().log_softmax(-1)).sum(-1);prob=logits.float().softmax(-1);pred=prob[:,0]-prob[:,2];truth=target[:,0]-target[:,2]
                    predictions.extend(pred.cpu().tolist());ces.extend(ce.cpu().tolist());maes.extend((pred-truth).abs().cpu().tolist())
            truth=records['value'][:,0]-records['value'][:,2];by={}
            for name,mask in [('winning',truth>.25),('neutral',abs(truth)<=.25),('losing',truth<-.25)]:
                if mask.any():by[name]=dict(n=int(mask.sum()),mae=float(np.asarray(maes)[mask].mean()))
            return dict(n=len(records),ce=float(np.mean(ces)),mae=float(np.mean(maes)),mean_value=float(np.mean(predictions)),by_stratum=by)
        rootsets=json.loads((data/'gate-roots.json').read_text());search_config=json.loads((ROOT/'configs/search-baseline-lock.json').read_text())['config']
        def gate(label):
            phase[0]='gate-'+label;model.eval();report=dict(balanced=value_metrics(val_rows),general=value_metrics(general_rows));network=MultiheadPlayer(model,'cuda');opponent=[]
            for group in ['diagnostics','holdout']:
                result=[]
                for j in rootsets[group]:
                    b=chess.Board(j['initial_fen'])
                    for u in j['history']:b.push_uci(u)
                    # A new native player ensures no cache/tree is carried between roots/checkpoints.
                    player=NativeMCTSPlayer(network,**search_config)
                    try:move=player.choose(b);stats=player.search_stats;elapsed=player.last_result['elapsed_seconds']
                    finally:player.close()
                    b.push(move);ok=move.uci()==j['solution'][0] or b.is_checkmate()
                    result.append(dict(key=j['key'],correct=ok,move=move.uci(),expected=j['solution'][0],elapsed=elapsed,stats=stats))
                    if group=='diagnostics':
                        b=chess.Board(j['initial_fen'])
                        for u in j['history']+[j['solution'][0]]:b.push_uci(u)
                        t,m=encode(b)
                        with torch.inference_mode(),amp():v=model.value_only(torch.as_tensor(t[None],device='cuda'),torch.as_tensor(m[None],device='cuda')).float().softmax(-1)
                        opponent.append(float((v[0,0]-v[0,2]).cpu()))
                    atomic_json(WORK/f'gate-{label}-progress.json',dict(group=group,completed=len(result),total=len(rootsets[group])))
                report[group]=dict(correct=sum(x['correct'] for x in result),total=len(result),rows=result)
            report['opponent_mean_value']=float(np.mean(opponent));report['policy_q_sha256']=policy_digest(model);assert report['policy_q_sha256']==policy_sha
            atomic_json(WORK/f'gate-{label}.json',report);sync();return report
        # Warm up outside the timed gate; same configuration/source as the frozen2s baseline.
        r=train_rows[:64]
        with torch.inference_mode(),amp():model(torch.as_tensor(r['tokens'].copy(),device='cuda'),torch.as_tensor(r['meta'].copy(),device='cuda'),torch.as_tensor(r['moves'].copy(),device='cuda'))
        reference=gate('initial')
        # Cache OLD value-encoder features; train only its small value head for750 updates.
        phase[0]='caching-value-features';features=[];targets=[];model.set_value_training('head')
        with torch.inference_mode():
            for start in range(0,len(train_rows),128):
                r=train_rows[start:start+128]
                with amp():x=model.value_model.encode_squares(torch.as_tensor(r['tokens'].copy(),device='cuda'),torch.as_tensor(r['meta'].copy(),device='cuda')).mean(1)
                features.append(x.float());targets.append(torch.as_tensor(r['value'].copy(),device='cuda'))
        features=torch.cat(features);targets=torch.cat(targets)
        with torch.inference_mode(),amp():anchor=model.value_model.value(features).float().softmax(-1).detach().clone()
        anchor=anchor.clone() # ordinary tensor; autograd may save the KL target
        short=WORK/'short';short.mkdir(exist_ok=True);atomic_json(short/'run.json',dict(steps=750,batch_size=1024,trainable='old-value-head-only',policy_q_frozen=True))
        copy(data/'train/manifest.json',short/'dataset.json')
        with (WORK/'dashboard.log').open('a') as log:
            import subprocess
            subprocess.Popen([sys.executable,'-m','neurochess.dashboard','--root',str(WORK),'--port','8765'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        opt=torch.optim.AdamW(model.value_model.value.parameters(),lr=1e-5,weight_decay=.01);phase[0]='value-head-repair';started=time.time()
        for step in range(1,751):
            index=torch.randint(len(features),(1024,),device='cuda');opt.zero_grad(set_to_none=True)
            with amp():logits=model.value_model.value(features[index]).float();ce=-(targets[index]*logits.log_softmax(-1)).sum(-1).mean();kl=F.kl_div(logits.log_softmax(-1),anchor[index],reduction='batchmean');loss=ce+.1*kl
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.value_model.value.parameters(),1.)
            if not torch.isfinite(norm):raise FloatingPointError('Nonfinite short-repair gradient')
            opt.step()
            if step%25==0:
                row=dict(step=step,loss=[float(loss.detach()),0.,0.,float(ce.detach())],value_ce=float(ce.detach()),anchor_kl=float(kl.detach()),timestamp=time.time(),batch_positions=1024,positions_per_second=step*1024/max(1e-6,time.time()-started))
                print(json.dumps(row),flush=True)
                with (short/'metrics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
                atomic_json(short/'status.json',dict(phase='training',step=step,total_steps=750,updated_at=time.time()))
        assert policy_digest(model)==policy_sha;atomic_save(checkpoint(model,750,lineage),short/'last.pt');atomic_json(short/'status.json',dict(phase='complete',step=750,total_steps=750,updated_at=time.time()));sync()
        del features,targets,anchor,opt;torch.cuda.empty_cache();candidate=gate('after-750');decision=gate_decision(reference,candidate)
        atomic_json(WORK/'short-gate-decision.json',decision);sync()
        if not decision['passed']:
            phase[0]='gate-failed-stopped';atomic_json(WORK/'pipeline-status.json',dict(phase=phase[0],large_training_started=False,decision=decision));return
        copy(short/'last.pt',WORK/'passed-selected.pt');sync()
        # Same branch continues ONLY after the fixed gate. Policy and Q remain frozen.
        phase[0]='large-value-continuation';large=WORK/'continuation';large.mkdir(exist_ok=True);model.set_value_training('backbone');model.train()
        # Broad general replay, rather than repeating the30k repair subset alone.
        core=WORK/'core'
        if not (core/'manifest.json').exists():
            copy(MOUNT/'MyDrive/neurochess-data/evals-v2-5m.tar.gz',WORK/'core.tar.gz')
            assert sha(WORK/'core.tar.gz')=='a0b5dbef770fcf2b7b4bc31a709f4a2b8379421965aa591f69291a6abbc7c2ef'
            core.mkdir(exist_ok=True)
            with tarfile.open(WORK/'core.tar.gz') as tar:tar.extractall(core,filter='data')
        # Exact heldout-state exclusions apply to broad replay as well.
        blocked={bytes(k).hex() for rows_ in [val_rows,general_rows] for k in rows_['key']}
        from neurochess.encoding import position_key
        for family in rootsets.values():
            for j in family:
                b=chess.Board(j['initial_fen'])
                for u in j['history']:b.push_uci(u)
                blocked.add(position_key(b).hex());b.push_uci(j['solution'][0]);blocked.add(position_key(b).hex())
        def core_stream():
            epoch=0
            while True:
                rng_=np.random.default_rng(20261102+epoch);files=sorted(core.glob('train-*.npy'));rng_.shuffle(files)
                for p in files:
                    records=np.load(p,mmap_mode='r');indices=rng_.permutation(len(records))
                    indices=[int(i) for i in indices if bytes(records['key'][i]).hex() not in blocked]
                    for start in range(0,len(indices)-63,64):
                        batch=records[indices[start:start+64]]
                        yield {k:torch.from_numpy(batch[k].copy()) for k in ['tokens','meta','value']}
                epoch+=1
        replay=core_stream()
        opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=2e-6,weight_decay=.05);rng=np.random.default_rng(20261102)
        atomic_json(large/'run.json',dict(steps=4000,batch_size=64,accum=16,trainable='value-backbone-and-head',policy_q_frozen=True,general_replay_fraction=.8,repair_fraction=.2))
        copy(data/'train/manifest.json',large/'dataset.json');large_started=time.time()
        for step in range(1,4001):
            model.train();opt.zero_grad(set_to_none=True)
            for group in opt.param_groups:group['lr']=2e-6*min(1,step/100)*(.1+.9*.5*(1+__import__('math').cos(__import__('math').pi*step/4000)))
            total=0.
            for micro in range(16):
                if ((step-1)*16+micro)%10<8:
                    batch=next(replay);t=batch['tokens'].to('cuda');m=batch['meta'].to('cuda');target=batch['value'].to('cuda')
                else:
                    r=train_rows[rng.integers(len(train_rows),size=64)];t=torch.as_tensor(r['tokens'].copy(),device='cuda');m=torch.as_tensor(r['meta'].copy(),device='cuda');target=torch.as_tensor(r['value'].copy(),device='cuda')
                with amp():logits=model.value_only(t,m).float();loss=-(target*logits.log_softmax(-1)).sum(-1).mean()
                (loss/16).backward();total+=float(loss.detach())/16
            norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
            if not torch.isfinite(norm):raise FloatingPointError('Nonfinite gradient')
            opt.step()
            if step%10==0:
                row=dict(step=step,cumulative_step=750+step,loss=[total,0.,0.,total],gradient_norm=float(norm),timestamp=time.time(),batch_positions=1024,positions_per_second=step*1024/max(1e-6,time.time()-large_started))
                with (large/'metrics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
                atomic_json(large/'status.json',dict(phase='training',step=step,total_steps=4000,updated_at=time.time()));print(json.dumps(row),flush=True)
            if step%250==0:
                assert policy_digest(model)==policy_sha
                atomic_save(checkpoint(model,750+step,lineage,continuation_step=step,optimizer=opt.state_dict()),large/'last.pt');sync()
            if step%1000==0:
                result=gate('continuation-'+str(step));decision=gate_decision(candidate,result);atomic_json(WORK/f'continuation-{step}-gate-decision.json',decision);sync()
                if not decision['passed']:
                    phase[0]='continuation-gate-failed';atomic_json(WORK/'pipeline-status.json',dict(phase=phase[0],continuation_step=step,default_promoted=False));return
                # Only passing checkpoints are eligible for future use.
                copy(large/'last.pt',WORK/'passed-selected.pt');sync();phase[0]='large-value-continuation'
        phase[0]='complete';atomic_json(large/'status.json',dict(phase='complete',step=4000,total_steps=4000,updated_at=time.time()));atomic_json(WORK/'pipeline-status.json',dict(phase='complete',default_promoted=False))
    except BaseException:
        phase[0]='failed';atomic_json(WORK/'failure.json',dict(traceback=traceback.format_exc(),time=time.time()));raise
    finally:
        stop.set();thread.join();sync();proof={}
        for p in WORK.rglob('*.pt'):
            dest=SAVE/p.relative_to(WORK);assert sha(p)==sha(dest);proof[str(p.relative_to(WORK))]=sha(p)
        atomic_json(WORK/'persistent-checkpoint-verification.json',dict(phase=phase[0],sha256=proof));sync()
        from neurochess.autostop import request_unassign,find_runtime_address
        request_unassign(find_runtime_address())

if __name__=='__main__':main()
