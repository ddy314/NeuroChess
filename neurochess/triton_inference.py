"""Inference-only fused residual add + FP32 LayerNorm + explicit output cast.

Independent implementation of the formula; no LC0 source is incorporated.
The residual remains FP32. BF16 rounding occurs before the following GEMM,
matching the original autocast boundary. Numerical parity must still be tested.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _add_norm(X,D,W,B,R,Y,N:tl.constexpr,EPS:tl.constexpr,ADD:tl.constexpr,BLOCK:tl.constexpr):
    row=tl.program_id(0);cols=tl.arange(0,BLOCK);mask=cols<N
    x=tl.load(X+row*N+cols,mask,other=0).to(tl.float32)
    if ADD:x=x+tl.load(D+row*N+cols,mask,other=0).to(tl.float32)
    mean=tl.sum(x,0)/N
    centered=tl.where(mask,x-mean,0.)
    variance=tl.sum(centered*centered,0)/N
    w=tl.load(W+cols,mask,other=0).to(tl.float32);b=tl.load(B+cols,mask,other=0).to(tl.float32)
    y=(x-mean)*tl.rsqrt(variance+EPS)*w+b
    if ADD:tl.store(R+row*N+cols,x,mask)
    tl.store(Y+row*N+cols,y,mask)


def add_layer_norm(x,delta,weight,bias,eps,output_dtype):
    if not x.is_cuda or x.dtype!=torch.float32 or not x.is_contiguous():raise ValueError('Contiguous FP32 CUDA residual required')
    if delta is not None and (delta.shape!=x.shape or not delta.is_contiguous()):raise ValueError('Residual delta mismatch')
    width=x.shape[-1];residual=torch.empty_like(x) if delta is not None else x
    output=torch.empty_like(x,dtype=output_dtype)
    _add_norm[(x.numel()//width,)](x,x if delta is None else delta,weight,bias,residual,output,width,eps,delta is not None,triton.next_power_of_2(width),num_warps=4,enable_fp_fusion=False)
    return residual,output
