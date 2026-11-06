"""Bounded PAWN ingestion pilot followed by conditional time-budgeted training."""
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tarfile
import time
import traceback

import numpy as np
import torch

from neurochess.data import FIELDS, DTYPE
from neurochess.pawn_data import sha, write_json, load_published, contains
from neurochess.persistence import RunMirror
from neurochess.policy_value_pair import checkpoint
from neurochess.train import load_model, losses, atomic_save
from scripts.joint_training_runtime import grouped_losses


class ShardStream:
    def __init__(self, source, cache, seed=20261103, timeout=1800):
        self.source=Path(source);self.cache=Path(cache);self.rng=np.random.default_rng(seed)
        self.timeout=timeout;self.index=0;self.rows=None;self.order=None;self.cursor=0
        self.wait_seconds=0.;self.load_seconds=0.;self.consumed=0

    def next(self,n):
        parts=[];remaining=n
        while remaining:
            if self.rows is None or self.cursor==len(self.rows):
                start=time.time()
                while True:
                    ready_path=self.source/'ready.json'
                    ready=json.loads(ready_path.read_text()) if ready_path.exists() else None
                    if ready and len(ready['shards'])>self.index:
                        if not ready['heldout_isolation_complete']:raise ValueError('Isolation not ready')
                        metadata=ready['shards'][self.index]['data'];break
                    failure=self.source/'failure.json'
                    if failure.exists():raise RuntimeError('Producer failed: '+failure.read_text())
                    if ready and ready['complete']:raise EOFError('Published training corpus exhausted')
                    if time.time()-start>self.timeout:raise TimeoutError('No new verified PAWN shard within30min')
                    time.sleep(5)
                self.wait_seconds+=time.time()-start;start=time.time()
                self.rows=load_published(metadata,self.source,self.cache)
                self.order=self.rng.permutation(len(self.rows));self.cursor=0;self.index+=1
                self.load_seconds+=time.time()-start
            count=min(remaining,len(self.rows)-self.cursor)
            parts.append(self.rows[self.order[self.cursor:self.cursor+count]])
            self.cursor+=count;remaining-=count
        self.consumed+=n
        return np.concatenate(parts)


class ReplayRows:
    """Uniform old-teacher replay after exact exclusion of every new heldout."""
    def __init__(self,root,excluded,seed=20261104):
        self.files=[];self.indices=[];self.ends=[];total=0
        self.rng=np.random.default_rng(seed)
        for path in sorted(Path(root).glob('train-*.npy')):
            rows=np.load(path,mmap_mode='r',allow_pickle=False)
            if rows.dtype!=DTYPE:raise ValueError('Old replay schema mismatch')
            keys=rows['key'].copy().view('V16').reshape(-1)
            indices=np.flatnonzero(~contains(excluded,keys))
            if len(indices):
                self.files.append(rows);self.indices.append(indices);total+=len(indices);self.ends.append(total)
        if not total:raise ValueError('No isolated old-teacher replay positions')
        self.total=total;self.ends=np.array(self.ends)

    def sample(self,n):
        picks=self.rng.integers(self.total,size=n);which=np.searchsorted(self.ends,picks,side='right')
        result=np.empty(n,dtype=DTYPE)
        for file in np.unique(which):
            mask=which==file;start=0 if file==0 else self.ends[file-1]
            result[mask]=self.files[file][self.indices[file][picks[mask]-start]]
        return result


def train(plan_path):
    plan=json.loads(Path(plan_path).read_text());work=Path(plan['work']);save=Path(plan['save'])
    if not os.path.ismount('/content/drive'):raise OSError('Drive mount required')
    work.mkdir(parents=True,exist_ok=True);save.mkdir(parents=True,exist_ok=True)
    if (save/'last.pt').exists():raise ValueError('Existing run: do not overwrite a trained checkpoint')
    mirror=RunMirror(work,save,'/content/drive',exclude_top_level=['cache','parent.pt','general-validation.npy',
        'old-core','old-core.tar.gz','val-all-keys.npy','test-all-keys.npy'])
    started=time.time();deadline=plan['allocated_at']+plan['gpu_budget_seconds']-600
    phase='preparing';step=seen=pawn_seen=replay_seen=0;target=plan['pilot_steps']*plan['effective_batch'];last_mirror=0
    def status(**extra):
        nonlocal last_mirror
        write_json(work/'status.json',dict(phase=phase,step=step,positions=seen,target_positions=target,
            pawn_positions=pawn_seen,replay_positions=replay_seen,
            seconds=time.time()-started,updated_at=time.time(),default_promoted=False,**extra))
        if time.time()-last_mirror>30 or phase!='training':mirror.sync();last_mirror=time.time()
    def append(name,row):
        with (work/name).open('a') as f:f.write(json.dumps(row)+'\n')
    try:
        if not torch.cuda.is_available() or 'L4' not in torch.cuda.get_device_name():raise RuntimeError('Expected one L4')
        torch.set_num_threads(2);torch.manual_seed(20261103)
        write_json(work/'plan.json',plan);status()
        parent=Path(plan['parent']);local_parent=work/'parent.pt';shutil.copyfile(parent,local_parent)
        if sha(local_parent)!=plan['parent_sha256']:raise ValueError('Parent checkpoint changed')
        model,state=load_model(local_parent,'cuda')
        if state.get('architecture')!='policy_value_pair_v1' or not model.config.history_free:
            raise ValueError('Expected current history-free double-encoder model')
        origin=state['step'];del state
        model.set_value_training('joint')
        optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=plan['lr'],weight_decay=.01,fused=True)
        source=Path(plan['data']);cache=work/'cache';cache.mkdir(exist_ok=True)
        wait_start=time.time()
        while not (source/'val-sample.json').exists() or not (source/'ready.json').exists():
            if (source/'failure.json').exists():raise RuntimeError('Producer failed before first training shard')
            if time.time()-wait_start>1800:raise TimeoutError('PAWN preparation not ready after30min')
            time.sleep(5)
        isolation=json.loads((source/'heldout-isolation.json').read_text())
        if not isolation['complete'] or isolation['prior_sha256']!=plan['prior_sha256']:
            raise ValueError('Heldout/prior isolation certificate mismatch')
        val_meta=json.loads((source/'val-sample.json').read_text())
        pawn_val=load_published(val_meta,source,cache)
        local_general=work/'general-validation.npy';shutil.copyfile(plan['general_validation'],local_general)
        general=np.load(local_general,allow_pickle=False)
        if sha(local_general)!=plan['general_validation_sha256']:raise ValueError('General validation changed')
        validation={'pawn':pawn_val,'general':general}
        # Existing deep-search labels anchor the old task while the PAWN signal
        # changes the position distribution and teacher search budget.
        phase='preparing-isolated-replay';status()
        archive=work/'old-core.tar.gz';shutil.copyfile(plan['replay_archive'],archive)
        if sha(archive)!=plan['replay_archive_sha256']:raise ValueError('Old core archive changed')
        old=work/'old-core';old.mkdir(exist_ok=True)
        with tarfile.open(archive) as package:package.extractall(old,filter='data')
        reserved=[]
        for split in ['val','test']:
            local=work/f'{split}-all-keys.npy';shutil.copyfile(source/local.name,local)
            expected=isolation['validation_key_sha256' if split=='val' else 'test_key_sha256']
            if sha(local)!=expected:raise ValueError('New heldout index changed')
            reserved.append(np.load(local,allow_pickle=False))
        reserved.append(general['key'].copy().view('V16').reshape(-1))
        replay=ReplayRows(old,np.unique(np.concatenate(reserved)))
        write_json(work/'replay-isolation.json',dict(eligible_old_positions=replay.total,
            excluded='All official PAWN val/test position keys plus fixed old validation keys',
            replay_fraction=plan['replay_fraction'],archive_sha256=plan['replay_archive_sha256']))
        def validate(name,rows):
            model.eval();totals=np.zeros(7);value_error=0
            with torch.inference_mode():
                for start in range(0,len(rows),64):
                    r=rows[start:start+64];b={k:torch.as_tensor(np.array(r[k],copy=True),device='cuda') for k in FIELDS}
                    with torch.autocast('cuda',dtype=torch.bfloat16):output=model(b['tokens'],b['meta'],b['moves'])
                    loss,parts=losses(output,b,.5,1.)
                    n=len(r);qmask=b['qmask'];pred=output[2].softmax(-1);v=pred[:,0]-pred[:,2];truth=b['value'][:,0]-b['value'][:,2]
                    totals+=np.array([loss.item()*n,parts['policy'].item()*n,parts['q'].item()*n,parts['value'].item()*n,
                        (output[0].argmax(-1)==b['policy'].argmax(-1)).sum().item(),
                        ((output[1]-b['q'].float()).abs()*qmask).sum().item(),qmask.sum().item()])
                    value_error+=(v-truth).abs().sum().item()
            result=dict(step=step,split=name,positions=len(rows),loss=totals[0]/len(rows),policy_loss=totals[1]/len(rows),
                q_loss=totals[2]/len(rows),value_loss=totals[3]/len(rows),top1=totals[4]/len(rows),
                q_mae=totals[5]/max(1,totals[6]),value_mae=value_error/len(rows))
            if not all(math.isfinite(float(v)) for v in result.values() if isinstance(v,(int,float))):raise FloatingPointError('Nonfinite validation')
            append('validation.jsonl',result);return result
        phase='baseline-validation';status();baseline={name:validate(name,r) for name,r in validation.items()}
        params=dict(model.named_parameters());names=['policy_model.piece.weight','policy_model.action.1.weight',
            'value_model.piece.weight','value_model.value.2.weight']
        before={n:params[n].detach().cpu().clone() for n in names}
        stream=ShardStream(source,cache);mix_rng=np.random.default_rng(20261103)
        write_json(work/'lineage.json',dict(parent_checkpoint_sha256=plan['parent_sha256'],origin_step=origin,
            optimizer_reset=True,architecture='policy_value_pair_v1',training_mode='joint',default_promoted=False,
            effective_batch=plan['effective_batch'],microbatch=64,accumulation=plan['effective_batch']//64,
            label_semantics='Search policy/value/Q; raw NNUE sidecar is not used as a substitute',
            replay_fraction=plan['replay_fraction']))
        write_json(work/'runtime.json',dict(gpu=torch.cuda.get_device_name(),parameters=sum(p.numel() for p in model.parameters()),
            precision='bfloat16',gpu_budget_seconds=plan['gpu_budget_seconds']))
        phase='training';status();training_started=time.time();compute_seconds=0.;pilot_passed=False
        def save_checkpoint():
            atomic_save(checkpoint(model,origin+step,dict(plan=plan),optimizer=optimizer.state_dict(),
                pawn_step=step,positions_seen=seen,pawn_positions=pawn_seen,replay_positions=replay_seen,
                default_promoted=False,pilot_passed=pilot_passed),work/'last.pt')
            mirror.sync()
        while seen<target:
            if time.time()>=deadline:phase='time-budget-reached';break
            n=min(plan['effective_batch'],target-seen)
            if n%64:raise ValueError('Training target must contain complete64-position microbatches')
            old_n=int(n*plan['replay_fraction']);new_n=n-old_n
            rows=np.concatenate([stream.next(new_n),replay.sample(old_n)])
            rows=rows[mix_rng.permutation(n)];model.train();optimizer.zero_grad(set_to_none=True)
            if step<plan['pilot_steps']:lr=plan['lr']
            else:
                fraction=(seen-plan['pilot_steps']*plan['effective_batch'])/max(1,target-plan['pilot_steps']*plan['effective_batch'])
                lr=plan['lr']*(.3+.7*.5*(1+math.cos(math.pi*min(1,fraction))))
            for group in optimizer.param_groups:group['lr']=lr
            torch.cuda.synchronize();tick=time.time();total=np.zeros(4);accum=n//64
            for i in range(0,n,64):
                r=rows[i:i+64];b={k:torch.as_tensor(np.array(r[k],copy=True),device='cuda') for k in FIELDS}
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    outputs=model(b['tokens'],b['meta'],b['moves']);loss,parts=grouped_losses(outputs,b)
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
                (loss/accum).backward();total+=np.r_[float(loss.detach()),parts.detach().float().cpu().numpy()]/accum
            norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
            if not torch.isfinite(norm):raise FloatingPointError('Nonfinite gradient')
            optimizer.step();torch.cuda.synchronize();compute_seconds+=time.time()-tick;seen+=n;step+=1
            pawn_seen+=new_n;replay_seen+=old_n
            if step%10==0:
                row=dict(step=step,positions=seen,loss=total.tolist(),lr=lr,gradient_norm=float(norm),
                    elapsed_seconds=time.time()-training_started,compute_positions_per_second=seen/max(1,compute_seconds),
                    wall_positions_per_second=seen/max(1,time.time()-training_started),input_wait_seconds=stream.wait_seconds,
                    input_load_seconds=stream.load_seconds,effective_batch=plan['effective_batch'],microbatch=64,
                    pawn_positions=pawn_seen,replay_positions=replay_seen,
                    accumulation=accum,gpu_peak_gib=torch.cuda.max_memory_allocated()/2**30,timestamp=time.time())
                append('metrics.jsonl',row);status()
            if step==plan['pilot_steps']:
                phase='pilot-validation';status();after={name:validate(name,r) for name,r in validation.items()}
                movement={n:float((params[n].detach().cpu()-before[n]).abs().max()) for n in names}
                gates=dict(finite_and_parameter_updates=all(v>0 for v in movement.values()),
                    pawn_loss=after['pawn']['loss']<=baseline['pawn']['loss']*1.01,
                    general_loss=after['general']['loss']<=baseline['general']['loss']*1.02,
                    general_top1=after['general']['top1']>=baseline['general']['top1']-.015,
                    general_value_mae=after['general']['value_mae']<=baseline['general']['value_mae']+.02,
                    general_q_mae=after['general']['q_mae']<=baseline['general']['q_mae']+.02)
                pilot_passed=all(gates.values())
                throughput=seen/max(1,time.time()-training_started)
                capacity=int(max(0,deadline-time.time()-300)*throughput*.85)//64*64+seen
                target=min(plan['preferred_positions'],capacity,plan['corpus_positions']) if pilot_passed else seen
                target=max(seen,target//64*64)
                # Preserve the completed updates before formatting/reporting any
                # gate result; reporting failures must not discard the model.
                save_checkpoint()
                write_json(work/'pilot-gate.json',dict(passed=pilot_passed,criteria=gates,baseline=baseline,after=after,
                    parameter_max_changes=movement,positions_per_second=throughput,next_target_positions=target,
                    scope='Ingestion/numerical and fixed-validation non-regression screen; not evidence of improved playing strength'))
                mirror.sync()
                if not pilot_passed:phase='pilot-gate-failed';break
                phase='training';status()
            elif step%500==0:
                phase='periodic-validation';status()
                for name,r in validation.items():validate(name,r)
                save_checkpoint();phase='training';status()
        phase='final-validation';status();final={name:validate(name,r) for name,r in validation.items()}
        save_checkpoint()
        expected=sha(work/'last.pt');actual=sha(save/'last.pt')
        if actual!=expected:raise OSError('Final checkpoint readback failed')
        write_json(work/'completion.json',dict(pawn_updates=step,positions=seen,target_positions=target,pilot_passed=pilot_passed,
            pawn_positions=pawn_seen,replay_positions=replay_seen,
            final_validation=final,checkpoint_sha256=actual,seconds=time.time()-started,
            default_promoted=False,playing_strength_improvement_established=False))
        phase='complete';status();mirror.sync()
    except BaseException:
        failure=traceback.format_exc()
        if step>0 and 'save_checkpoint' in locals():
            try:save_checkpoint()
            except BaseException:failure+='\nEmergency checkpoint failure:\n'+traceback.format_exc()
        phase='failed';write_json(work/'failure.json',dict(step=step,positions=seen,traceback=failure));status();raise


if __name__=='__main__':train(sys.argv[1])
