"""Inference for the engine-supervised policy, candidate-Q and WDL heads."""
from contextlib import nullcontext
import numpy as np
import torch

from .encoding import encode, legal_ids, MAX_MOVES


def fuse_heads(policy, q, wdl, legal_mask, top_k=5, q_weight=2.):
    """Policy-supported Q only; WDL confidence controls its reranking weight.

    The Q-label mask is unavailable at inference. Support here is an explicit
    approximation (top five policy moves, within three logits of the leader),
    never a claim that those moves are known to have had training labels.
    """
    policy=np.asarray(policy,dtype=np.float32);q=np.asarray(q,dtype=np.float32)
    wdl=np.asarray(wdl,dtype=np.float32);legal_mask=np.asarray(legal_mask,dtype=bool)
    if not np.isfinite(policy[legal_mask]).all() or not np.isfinite(q[legal_mask]).all() or not np.isfinite(wdl).all():
        raise FloatingPointError('Nonfinite model outputs')
    if np.any(np.abs(q[legal_mask])>1.0001):raise ValueError('Q must be in [-1,1]')
    logits=np.where(legal_mask,policy,-1e9)
    probabilities=np.exp(logits-logits.max(axis=1,keepdims=True))*legal_mask
    probabilities/=probabilities.sum(axis=1,keepdims=True)
    rank=np.argsort(-logits,axis=1,kind='stable')
    support=np.zeros_like(legal_mask)
    np.put_along_axis(support,rank[:,:top_k],True,axis=1)
    support &= legal_mask & (logits>=logits.max(axis=1,keepdims=True)-3.)
    weights=probabilities*support;weights/=weights.sum(axis=1,keepdims=True)
    centered=q-(q*weights).sum(axis=1,keepdims=True)
    entropy=-(wdl*np.log(np.maximum(wdl,1e-8))).sum(axis=1)/np.log(3.)
    confidence=np.clip(1-entropy,0,1)
    fused=logits+q_weight*(.25+.75*confidence[:,None])*centered*support
    value=wdl[:,0]-wdl[:,2]
    # Conservative two-observation Q prior; search evidence subsequently
    # dominates it. Value is from the same parent side-to-move perspective.
    prior_q=np.where(support,.5*q+.5*value[:,None],np.nan)
    return fused.astype(np.float32),prior_q.astype(np.float32),value.astype(np.float32),support


class MultiheadPlayer:
    mode='multihead-direct'
    def __init__(self,model,device='cuda'):
        self.model=model.eval();self.device=device
        self.search_stats=dict(forward_positions=0,q_supported_actions=0,q_changed_decisions=0)

    @torch.inference_mode()
    def raw_outputs(self,tokens,meta,moves):
        cuda=self.device.startswith('cuda')
        context=torch.autocast('cuda',dtype=torch.bfloat16) if cuda else nullcontext()
        with context:
            policy,q,wdl=self.model(torch.as_tensor(tokens,device=self.device),
                torch.as_tensor(meta,device=self.device),torch.as_tensor(moves,device=self.device))
        wdl=wdl.softmax(-1)
        return torch.cat([policy,q,wdl],dim=1).cpu().numpy()

    @torch.inference_mode()
    def evaluate_search_heads(self,tokens,meta,moves):
        outputs=self.raw_outputs(tokens,meta,moves)
        fused,prior_q,value,support=fuse_heads(outputs[:,:256],outputs[:,256:512],outputs[:,512:],moves>=0)
        self.search_stats['forward_positions']+=len(tokens)
        self.search_stats['q_supported_actions']+=int(support.sum())
        self.search_stats['q_changed_decisions']+=int((fused.argmax(1)!=outputs[:,:256].argmax(1)).sum())
        return fused,prior_q,value

    def choose_many(self,boards):
        encoded=[encode(b) for b in boards];moves=np.full((len(boards),MAX_MOVES),-1,dtype=np.int16)
        legal=[]
        for i,b in enumerate(boards):
            ms,ids=legal_ids(b);legal.append(ms);moves[i,:len(ids)]=ids
        fused,_,_=self.evaluate_search_heads(np.stack([x[0] for x in encoded]),np.stack([x[1] for x in encoded]),moves)
        return [ms[int(row[:len(ms)].argmax())] for ms,row in zip(legal,fused)]

    def choose(self,board):return self.choose_many([board])[0]
