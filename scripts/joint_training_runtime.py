"""Batch regrouping and continuation schedule for the same 320-position update."""
import math
import numpy as np
import torch
from neurochess.data import FIELDS
from neurochess.train import losses

POOL_ORDER = ['old', 'new', 'mistake', 'old', 'new'] * 4


def canonical_update(sample):
    # Preserve the old RNG call order and exactly 128old/128new/64mistake rows.
    return np.concatenate([sample(pool,16) for pool in POOL_ORDER])


def grouped_losses(outputs, batch, base_microbatch=16):
    # Q and weighted value losses were normalized per original16-row block.
    # Simply taking one 64-row mean would silently change that objective.
    n=len(batch['tokens'])
    if n % base_microbatch:
        raise ValueError('Regroup only complete original microbatches')
    components=[];objectives=[]
    for start in range(0,n,base_microbatch):
        b={k:v[start:start+base_microbatch] for k,v in batch.items()}
        loss,parts=losses(tuple(v[start:start+base_microbatch] for v in outputs),b,.5,1.)
        objectives.append(loss)
        components.append(torch.stack([parts[k] for k in ['policy','q','value']]))
    return torch.stack(objectives).mean(),torch.stack(components).mean(0)


def backward_update(model, rows, microbatch):
    if len(rows)!=320 or microbatch not in [16,32,64]:
        raise ValueError('Require effective batch320 and a tested microbatch')
    totals=torch.zeros(4,device='cuda');q_labels=torch.zeros((),device='cuda',dtype=torch.int64)
    accum=320//microbatch
    for start in range(0,320,microbatch):
        r=rows[start:start+microbatch]
        b={k:torch.as_tensor(np.array(r[k],copy=True),device='cuda') for k in FIELDS}
        with torch.autocast('cuda',dtype=torch.bfloat16):
            output=model(b['tokens'],b['meta'],b['moves'])
            loss,parts=grouped_losses(output,b)
        (loss/accum).backward()
        totals+=torch.cat([loss.detach().reshape(1),parts])/accum
        q_labels+=b['qmask'].sum()
    return totals,q_labels


def remaining_lr(step, schedule, default):
    if not schedule:return default
    start,end=schedule['start_step'],schedule['end_step']
    fraction=min(1.,max(0.,(step-start)/max(1,end-start)))
    return schedule['min_lr']+.5*(schedule['initial_lr']-schedule['min_lr'])*(1+math.cos(math.pi*fraction))


def verify_restored_state(model, optimizer, checkpoint):
    counts=dict(model_tensors=0,optimizer_tensors=0)
    for name,tensor in model.state_dict().items():
        if not torch.equal(tensor.detach().cpu(),checkpoint['model'][name]):
            raise ValueError(f'Restored model tensor differs: {name}')
        counts['model_tensors']+=1
    actual=optimizer.state_dict();expected=checkpoint['optimizer']
    if actual['param_groups']!=expected['param_groups']:
        raise ValueError('Restored optimizer parameter groups differ')
    if set(actual['state'])!=set(expected['state']):
        raise ValueError('Restored optimizer state IDs differ')
    for key,state in actual['state'].items():
        for field,value in state.items():
            wanted=expected['state'][key][field]
            if isinstance(value,torch.Tensor):
                if not torch.equal(value.detach().cpu(),wanted):
                    raise ValueError(f'Restored optimizer tensor differs: {key}/{field}')
                counts['optimizer_tensors']+=1
            elif value!=wanted:raise ValueError(f'Restored optimizer state differs: {key}/{field}')
    return counts
