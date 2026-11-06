"""Masked-label normalization and the next single-branch320-position mixture."""
import torch
import torch.nn.functional as F

MIXTURE={'old':160,'new':96,'frontier':32,'mistake':32}
LOSS_MODE='available-target-normalized-v2'


def normalizers(batch):
    return dict(policy=(batch['policy'].float().sum(-1)>0).sum().clamp_min(1),q=batch['qmask'].sum().clamp_min(1),value=batch['value_weight'].float().sum().clamp_min(1))


def losses(outputs,batch,q_weight=.5,value_weight=1.,normalization=None):
    """For accumulation, pass full320-row normalizers and SUM microbatch losses.

    Do not divide again by accumulation. This keeps the effective objective
    independent of microbatch16/32/64 and missing-target placement.
    """
    p,q,v=outputs;target=batch['policy'].float();valid=target.sum(-1)>0
    norm=normalizers(batch) if normalization is None else normalization
    target=target/target.sum(-1,keepdim=True).clamp_min(1e-8)
    pl=(-(target*F.log_softmax(p.float(),-1)).sum(-1)*valid).sum()/norm['policy']
    mask=batch['qmask'];ql=torch.where(mask,q.float()-batch['q'].float(),0.).square().sum()/norm['q']
    w=batch['value_weight'].float();vt=torch.where(w[:,None]>0,batch['value'].float(),0.)
    vl=(-(vt*F.log_softmax(v.float(),-1)).sum(-1)*w).sum()/norm['value']
    return pl+q_weight*ql+value_weight*vl,dict(policy=pl.detach(),q=ql.detach(),value=vl.detach(),policy_positions=valid.sum().detach(),q_actions=mask.sum().detach(),value_positions=(w>0).sum().detach())


def dynamic_ids(store,model_sha,run_id,update_id,seed=0,max_repeats=8):
    """Reserve dynamic slots durably; old replay remains an independent pool.

    Fail on shortage rather than cycling a small mistake set indefinitely.
    Calls for an existing update return exactly the previously reserved IDs.
    """
    return {pool:store.sample(model_sha,MIXTURE[pool],run_id,update_id,pool,seed,max_repeats) for pool in ['mistake','frontier','new']}
