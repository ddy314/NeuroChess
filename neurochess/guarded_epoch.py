"""Exact single-pass sampling and continuously enforced three-head guardrails."""
from pathlib import Path
import math
import numpy as np
import torch
import torch.nn.functional as F
from .data import DTYPE

GUARD = dict(old_loss_relative=.01, old_policy_relative=.01, old_value_relative=.02,
             old_top1_drop=.005, old_q_mae_increase=.005, old_value_mae_increase=.005,
             old_value_stratum_increase=.01, new_loss_relative=.01,
             new_q_mae_increase=.01, new_value_mae_increase=.01)


class ExactEpoch:
    """Each new row once, including the tail; deterministic committed cursor."""
    def __init__(self, files, seed, consumed=0):
        self.files=list(map(Path,files));self.seed=seed
        rng=np.random.default_rng(seed);self.order=rng.permutation(len(files))
        self.sizes=[len(np.load(p,mmap_mode='r',allow_pickle=False)) for p in self.files]
        self.total=sum(self.sizes)
        if not 0<=consumed<=self.total:raise ValueError('Invalid epoch cursor')
        self.consumed=consumed;self.index=0;self.cursor=consumed;self.rows=None
        while self.index<len(self.order) and self.cursor>=self.sizes[self.order[self.index]]:
            self.cursor-=self.sizes[self.order[self.index]];self.index+=1
    def next(self,n):
        if n<1 or self.consumed+n>self.total:raise EOFError('One epoch exhausted; no cycling or padding')
        parts=[]
        while n:
            i=int(self.order[self.index])
            if self.rows is None:
                self.rows=np.load(self.files[i],mmap_mode='r',allow_pickle=False)
                if self.rows.dtype!=DTYPE:raise ValueError('Dataset schema drift')
                self.row_order=np.random.default_rng(np.random.SeedSequence([self.seed,i])).permutation(len(self.rows))
            count=min(n,len(self.rows)-self.cursor)
            parts.append(self.rows[self.row_order[self.cursor:self.cursor+count]])
            self.cursor+=count;self.consumed+=count;n-=count
            if self.cursor==len(self.rows):self.rows=None;self.index+=1;self.cursor=0
        return np.concatenate(parts)


def replay_count(new_count, numerator=1, denominator=3):
    # 3 fresh +1 replay: fresh corpus remains exactly one pass.
    return (new_count*numerator+denominator//2)//denominator


def learning_rate(step,total,peak=1e-6,warmup=100):
    if step<=warmup:return peak*step/warmup
    t=min(1.,(step-warmup)/max(1,total-warmup))
    return peak*(.1+.9*.5*(1+math.cos(math.pi*t)))


def guard_metrics(baseline,current):
    checks={}
    for name,base in baseline.items():
        now=current[name]
        if not all(math.isfinite(float(now[k])) for k in ['loss','policy_loss','value_loss','top1','q_mae','value_mae']):
            checks[name+'/finite']=False;continue
        checks[name+'/loss']=now['loss']<=base['loss']*(1+GUARD['new_loss_relative' if name=='new' else 'old_loss_relative'])
        checks[name+'/Q']=now['q_mae']<=base['q_mae']+GUARD['new_q_mae_increase' if name=='new' else 'old_q_mae_increase']
        checks[name+'/value']=now['value_mae']<=base['value_mae']+GUARD['new_value_mae_increase' if name=='new' else 'old_value_mae_increase']
        if name=='new':
            checks[name+'/top1']=now['top1']>=base['top1']-.005
            for kind,group in base.get('value_strata',{}).items():
                if group['n']>=32:checks[name+'/'+kind]=now['value_strata'][kind]['mae']<=group['mae']+.015
        if name!='new':
            checks[name+'/policy']=now['policy_loss']<=base['policy_loss']*(1+GUARD['old_policy_relative'])
            checks[name+'/value_CE']=now['value_loss']<=base['value_loss']*(1+GUARD['old_value_relative'])
            checks[name+'/top1']=now['top1']>=base['top1']-GUARD['old_top1_drop']
            for kind,group in base.get('value_strata',{}).items():
                if group['n']>=32:checks[name+'/'+kind]=now['value_strata'][kind]['mae']<=group['mae']+GUARD['old_value_stratum_increase']
    checks={k:bool(v) for k,v in checks.items()}
    return dict(passed=all(checks.values()),checks=checks,failed=[k for k,v in checks.items() if not v])


def balance_value_weights(rows):
    truth=rows['value'][:,0]-rows['value'][:,2]
    groups=np.where(truth>.25,2,np.where(truth<-.25,0,1))
    counts=np.bincount(groups,minlength=3);present=int((counts>0).sum())
    weights=len(rows)/(present*counts[groups])
    rows['value_weight']*=weights.astype(np.float32)
    return rows


def joint_objective(outputs,batch,n_total,q_total,v_total,old_mask,teacher=None,old_total=0):
    """Normalize across the effective batch, independent of microbatch/tail size."""
    policy,q,value=outputs;target=batch['policy'].float();target=target/target.sum(-1,keepdim=True).clamp_min(1e-8)
    pl=-(target*F.log_softmax(policy.float(),-1)).sum()/n_total
    ql=((q.float()-batch['q'].float()).square()*batch['qmask']).sum()/max(1,q_total)
    vl=(-(batch['value'].float()*F.log_softmax(value.float(),-1)).sum(-1)*batch['value_weight']).sum()/max(1,v_total)
    anchor=pl*0
    if teacher is not None and old_mask.any() and old_total:
        tp,tq,tv=teacher;mask=old_mask
        # KL/MSE only on replay inputs. Teacher is the immutable initialization.
        pk=F.kl_div(F.log_softmax(policy[mask].float(),-1),tp.float().softmax(-1),reduction='sum')/old_total
        vk=F.kl_div(F.log_softmax(value[mask].float(),-1),tv.float().softmax(-1),reduction='sum')/old_total
        legal=batch['moves'][mask]>=0
        qk=(((q[mask]-tq.float()).square()*legal).sum(-1)/legal.sum(-1).clamp_min(1)).sum()/old_total
        anchor=.5*pk+.25*qk+4.*vk
    return pl+.5*ql+vl+anchor,torch.stack([pl.detach(),ql.detach(),vl.detach(),anchor.detach()])
