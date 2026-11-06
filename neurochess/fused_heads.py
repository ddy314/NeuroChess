"""Experimental GPU head fusion with stable policy ties and explicit parity tests."""
import numpy as np
import torch
import triton
import triton.language as tl
from .multihead import MultiheadPlayer
from .cuda_inference import CudaGraphMultiheadPlayer


@triton.jit
def _fuse(P,Q,W,IDS,OUT):
    row=tl.program_id(0);cols=tl.arange(0,256);legal=tl.load(IDS+row*256+cols)>=0
    p=tl.where(legal,tl.load(P+row*256+cols),-1.e9);q=tl.load(Q+row*256+cols)
    mx=tl.max(p,0);e=tl.exp(p-mx);probs=tl.where(legal,e,0.)/tl.sum(tl.where(legal,e,0.),0)
    chosen=tl.full((256,),False,tl.int1)
    for _ in range(5):
        remaining=tl.where(legal & ~chosen,p,float('-inf'));best=tl.max(remaining,0)
        index=tl.min(tl.where(legal & ~chosen & (remaining==best),cols,256),0)
        chosen=chosen | (cols==index)
    support=chosen & legal & (p>=mx-3.)
    weights=tl.where(support,probs,0.);weights=weights/tl.sum(weights,0)
    mean=tl.sum(weights*q,0)
    lanes=tl.arange(0,4);w=tl.load(W+row*3+lanes,lanes<3,other=float('-inf'))
    ew=tl.exp(w-tl.max(w,0));w=ew/tl.sum(ew,0)
    value=tl.sum(tl.where(lanes==0,w,0.),0)-tl.sum(tl.where(lanes==2,w,0.),0)
    entropy=-tl.sum(w*tl.log(tl.maximum(w,1.e-8)),0)/1.0986122886681098
    confidence=tl.minimum(1.,tl.maximum(0.,1.-entropy))
    fused=p+2.*(.25+.75*confidence)*(q-mean)*support
    prior=tl.where(support,.5*q+.5*value,float('nan'))
    tl.store(OUT+row*515+cols,fused);tl.store(OUT+row*515+256+cols,prior)
    old=tl.min(tl.where(p==mx,cols,256),0);new=tl.min(tl.where(fused==tl.max(fused,0),cols,256),0)
    tl.store(OUT+row*515+512,value);tl.store(OUT+row*515+513,tl.sum(support.to(tl.int32),0));tl.store(OUT+row*515+514,(old!=new).to(tl.float32))


def fuse_cuda(policy,q,wdl_logits,ids):
    output=torch.empty((len(policy),515),dtype=torch.float32,device=policy.device)
    _fuse[(len(policy),)](policy,q,wdl_logits,ids,output,num_warps=4,enable_fp_fusion=False)
    return output


class FusedGraphMultiheadPlayer(CudaGraphMultiheadPlayer):
    def device_forward(self,tokens,meta,moves):
        with torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=False):p,q,w=self.model(tokens,meta,moves)
        return fuse_cuda(p,q,w,moves)

    @torch.inference_mode()
    def evaluate_search_heads(self,tokens,meta,moves):
        if len(tokens) in self.graphs:out=self.raw_outputs(tokens,meta,moves)
        else:
            args=[torch.as_tensor(x,device=self.device) for x in [tokens,meta,moves]]
            out=self.device_forward(*args).cpu().numpy()
        self.search_stats['forward_positions']+=len(tokens)
        self.search_stats['q_supported_actions']+=int(out[:,513].sum())
        self.search_stats['q_changed_decisions']+=int(out[:,514].sum())
        return out[:,:256],out[:,256:512],out[:,512]
