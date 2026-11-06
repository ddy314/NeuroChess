"""Split the exact objective and conservatively gate a Value-KL ablation."""
import math
import statistics
import torch
from torch.nn import functional as F

NAMES = ('new_policy', 'new_Q', 'new_value', 'old_policy', 'old_Q',
         'old_value', 'retention_policy', 'retention_Q', 'retention_value')
RULE = dict(batches=16, minimum_conflicting_batches=12,
            minimum_median_conflict_share=.6, minimum_median_norm_ratio=.25,
            minimum_adam_improvement_batches=12, minimum_median_adam_gain=.05)


def split_terms(outputs, batch, nt, nq, nv, old, teacher, nold, value_kl=4.):
    p, q, v = outputs
    target = batch['policy'].float()
    target = target / target.sum(-1, keepdim=True).clamp_min(1e-8)
    pl = -(target * p.float().log_softmax(-1)).sum(-1) / nt
    ql = .5 * ((q.float()-batch['q'].float()).square()*batch['qmask']).sum(-1) / max(1, nq)
    vl = -(batch['value'].float()*v.float().log_softmax(-1)).sum(-1)*batch['value_weight']/max(1, nv)
    terms = {}
    for name, mask in [('new', ~old), ('old', old)]:
        for head, loss in [('policy', pl), ('Q', ql), ('value', vl)]:
            terms[name+'_'+head] = loss[mask].sum()
    if teacher is not None and old.any():
        tp, tq, tv = teacher
        legal = batch['moves'][old] >= 0
        terms['retention_policy'] = .5*F.kl_div(p[old].float().log_softmax(-1), tp.float().softmax(-1), reduction='sum')/nold
        terms['retention_Q'] = .25*((q[old].float()-tq.float()).square()*legal).sum(-1).div(legal.sum(-1).clamp_min(1)).sum()/nold
        terms['retention_value'] = value_kl*F.kl_div(v[old].float().log_softmax(-1), tv.float().softmax(-1), reduction='sum')/nold
    else:
        for name in NAMES[6:]: terms[name] = p.sum()*0
    return terms


def gram(vectors):
    result = torch.zeros(len(NAMES), len(NAMES), dtype=torch.float64)
    # CPU accumulation avoids holding nine full gradients in GPU memory.
    for i, a in enumerate(vectors):
        for j in range(i, len(vectors)):
            value = sum(float(torch.dot(x.flatten(), y.flatten())) for x,y in zip(a,vectors[j]))
            result[i,j] = result[j,i] = value
    return result


def geometry(matrix):
    new = torch.tensor([1.,1.,1.,0.,0.,0.,0.,0.,0.], dtype=torch.float64)
    kv = torch.zeros(9, dtype=torch.float64); kv[8] = 1
    dot = float(new @ matrix @ kv)
    nnew = math.sqrt(max(0., float(new @ matrix @ new)))
    nkv = math.sqrt(max(0., float(matrix[8,8])))
    opposition = [max(0., -float((new @ matrix)[i])) for i in range(6,9)]
    return dict(value_kl_cosine_with_new=dot/max(nnew*nkv,1e-30),
                value_kl_conflict_share=opposition[2]/max(sum(opposition),1e-30),
                value_kl_norm_over_new=nkv/max(nnew,1e-30))


def adam_counterfactual(parameters, vectors, optimizer_state, lr):
    """Exact next AdamW displacement formula, without mutating checkpoint state."""
    groups = optimizer_state['param_groups']
    if len(groups)!=1: raise ValueError('Expected original single AdamW parameter group')
    group = groups[0]
    if group.get('amsgrad',False) or group.get('maximize',False): raise ValueError('Unsupported Adam variant')
    ids = group['params']
    if len(ids)!=len(parameters): raise ValueError('Optimizer parameter ordering differs')
    beta1,beta2=group['betas']; eps=group['eps']; wd=group['weight_decay']
    result={}
    for coefficient in (4.,2.):
        weights=[1.]*8+[coefficient/4.]
        norm2=0.
        for index in range(len(parameters)):
            g=sum(weights[j]*vectors[j][index] for j in range(9))
            norm2+=float(g.square().sum())
        scale=min(1.,1./(math.sqrt(norm2)+1e-6))
        dn=do=step_norm=0.
        for index,parameter in enumerate(parameters):
            state=optimizer_state['state'][ids[index]]
            step=int(state['step'])+1
            g=sum(weights[j]*vectors[j][index] for j in range(9))*scale
            m=beta1*state['exp_avg']+(1-beta1)*g
            v=beta2*state['exp_avg_sq']+(1-beta2)*g.square()
            delta=-lr*((m/(1-beta1**step))/((v/(1-beta2**step)).sqrt()+eps)+wd*parameter.detach().cpu())
            dn+=float((delta*sum(vectors[j][index] for j in range(3))).sum())
            do+=float((delta*sum(vectors[j][index] for j in range(3,6))).sum())
            step_norm+=float(delta.square().sum())
        result[str(int(coefficient))]=dict(new_first_order_change=dn,old_first_order_change=do,update_norm=math.sqrt(step_norm),clip_scale=scale)
    a,b=result['4']['new_first_order_change'],result['2']['new_first_order_change']
    result['relative_new_gain']=(a-b)/max(abs(a),1e-12)
    return result


def decision(records):
    if len(records)!=16: return dict(run_D=False,reason='Require all 16 batches',rule=RULE)
    scalars=[r[k] for r in records for k in ('value_kl_cosine_with_new','value_kl_conflict_share','value_kl_norm_over_new')]
    scalars += [r['adam']['relative_new_gain'] for r in records]
    if not all(math.isfinite(x) for x in scalars): return dict(run_D=False,reason='Nonfinite diagnostic',rule=RULE)
    checks=dict(conflicting_batches=sum(r['value_kl_cosine_with_new']<0 for r in records)>=12,
                dominant_conflict=statistics.median(r['value_kl_conflict_share'] for r in records)>=.6,
                material_norm=statistics.median(r['value_kl_norm_over_new'] for r in records)>=.25,
                adam_improves_consistently=sum(r['adam']['relative_new_gain']>0 for r in records)>=12,
                adam_improvement_material=statistics.median(r['adam']['relative_new_gain'] for r in records)>=.05)
    return dict(run_D=all(checks.values()),checks=checks,rule=RULE,
                scope='Predeclared conservative operational trigger; not proof of causality or future validation improvement')
