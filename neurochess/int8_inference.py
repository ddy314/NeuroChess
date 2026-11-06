"""Experimental W8A8 Tensor Core linear layers, with Triton pack/epilogue.

Quantization is approximate and never enabled for the frozen baseline. Dynamic
per-row activation scales and per-output weight scales are explicit. A sample
parity/quality gate is mandatory before any use in a promoted search profile.
"""
import torch
from torch import nn
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _pack(X,Q,S,K:tl.constexpr,BLOCK:tl.constexpr):
    row=tl.program_id(0);cols=tl.arange(0,BLOCK);valid=cols<K
    x=tl.load(X+row*K+cols,valid,other=0).to(tl.bfloat16).to(tl.float32)
    scale=tl.maximum(tl.max(tl.abs(x),0),1.e-8)/127.
    q=libdevice.nearbyint(x/scale)
    tl.store(Q+row*K+cols,tl.minimum(127.,tl.maximum(-127.,q)).to(tl.int8),valid)
    tl.store(S+row,scale)


@triton.autotune(configs=[
    triton.Config({'BM':32,'BN':64,'BK':64},num_warps=4,num_stages=3),
    triton.Config({'BM':32,'BN':128,'BK':64},num_warps=4,num_stages=3),
    triton.Config({'BM':64,'BN':64,'BK':128},num_warps=4,num_stages=3),
    triton.Config({'BM':64,'BN':128,'BK':128},num_warps=4,num_stages=3),
    triton.Config({'BM':128,'BN':64,'BK':128},num_warps=8,num_stages=3),
],key=['M','N','K'])
@triton.jit
def _gemm(A,W,AS,WS,BIAS,Y,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,HAS_BIAS:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    rows=tl.program_id(0)*BM+tl.arange(0,BM)
    cols=tl.program_id(1)*BN+tl.arange(0,BN)
    ks=tl.arange(0,BK);acc=tl.full((BM,BN),0,tl.int32)
    for block in range(tl.cdiv(K,BK)):
        kk=block*BK+ks
        a=tl.load(A+rows[:,None]*K+kk[None,:],(rows[:,None]<M)&(kk[None,:]<K),other=0)
        w=tl.load(W+kk[:,None]*N+cols[None,:],(kk[:,None]<K)&(cols[None,:]<N),other=0)
        acc=tl.dot(a,w,acc)
    s=tl.load(AS+rows,rows<M,other=0);ws=tl.load(WS+cols,cols<N,other=0)
    out=acc.to(tl.float32)*s[:,None]*ws[None,:]
    if HAS_BIAS:out=out+tl.load(BIAS+cols,cols<N,other=0).to(tl.float32)[None,:]
    tl.store(Y+rows[:,None]*N+cols[None,:],out,(rows[:,None]<M)&(cols[None,:]<N))


class Int8Linear(nn.Module):
    def __init__(self,linear):
        super().__init__();w=linear.weight.detach().to(torch.bfloat16).float()
        scale=w.abs().amax(1).clamp_min(1.e-8)/127.
        quantized=(w/scale[:,None]).round().clamp(-127,127).to(torch.int8).T.contiguous()
        self.register_buffer('weight',quantized);self.register_buffer('scale',scale)
        self.register_buffer('bias',linear.bias.detach().to(torch.bfloat16) if linear.bias is not None else None)
        self.in_features=linear.in_features;self.out_features=linear.out_features

    def forward(self,x):
        shape=x.shape[:-1];k=self.in_features;n=self.out_features;x=x.contiguous();m=x.numel()//k
        q=torch.empty((m,k),dtype=torch.int8,device=x.device);s=torch.empty(m,dtype=torch.float32,device=x.device)
        _pack[(m,)](x,q,s,k,triton.next_power_of_2(k),num_warps=4)
        out=torch.empty((m,n),dtype=torch.bfloat16,device=x.device)
        _gemm[lambda meta:(triton.cdiv(m,meta['BM']),triton.cdiv(n,meta['BN']))](q,self.weight,s,self.scale,self.scale if self.bias is None else self.bias,out,m,n,k,self.bias is not None)
        return out.reshape(*shape,n)


def quantize_linears(model,scope='ffn'):
    from .fast_inference import resident_linears
    model=resident_linears(model)
    for block in model.blocks:
        block.ff[0]=Int8Linear(block.ff[0]);block.ff[2]=Int8Linear(block.ff[2])
        if scope=='all':block.qkv=Int8Linear(block.qkv);block.out=Int8Linear(block.out)
    return model.eval()


@triton.jit
def _norm_pack(X,D,W,B,R,Q,S,K:tl.constexpr,EPS:tl.constexpr,ADD:tl.constexpr,BLOCK:tl.constexpr):
    row=tl.program_id(0);cols=tl.arange(0,BLOCK);valid=cols<K
    x=tl.load(X+row*K+cols,valid,other=0).to(tl.float32)
    if ADD:x=x+tl.load(D+row*K+cols,valid,other=0).to(tl.float32)
    mean=tl.sum(x,0)/K;z=tl.where(valid,x-mean,0.);variance=tl.sum(z*z,0)/K
    w=tl.load(W+cols,valid,other=0);b=tl.load(B+cols,valid,other=0)
    n=((x-mean)*tl.rsqrt(variance+EPS)*w+b).to(tl.bfloat16).to(tl.float32)
    scale=tl.maximum(tl.max(tl.where(valid,tl.abs(n),0.),0),1.e-8)/127.
    q=libdevice.nearbyint(n/scale)
    tl.store(Q+row*K+cols,tl.minimum(127.,tl.maximum(-127.,q)).to(tl.int8),valid);tl.store(S+row,scale)
    if ADD:tl.store(R+row*K+cols,x,valid)


@triton.jit
def _gelu_pack(X,Q,S,K:tl.constexpr,BLOCK:tl.constexpr):
    row=tl.program_id(0);cols=tl.arange(0,BLOCK);valid=cols<K
    x=tl.load(X+row*K+cols,valid,other=0).to(tl.float32)
    y=(.5*x*(1.+libdevice.tanh(.7978845608028654*(x+.044715*x*x*x)))).to(tl.bfloat16).to(tl.float32)
    scale=tl.maximum(tl.max(tl.where(valid,tl.abs(y),0.),0),1.e-8)/127.
    q=libdevice.nearbyint(y/scale)
    tl.store(Q+row*K+cols,tl.minimum(127.,tl.maximum(-127.,q)).to(tl.int8),valid);tl.store(S+row,scale)


def norm_pack(x,delta,norm):
    k=x.shape[-1];m=x.numel()//k;r=torch.empty_like(x) if delta is not None else x
    q=torch.empty((m,k),dtype=torch.int8,device=x.device);s=torch.empty(m,dtype=torch.float32,device=x.device)
    _norm_pack[(m,)](x,x if delta is None else delta,norm.weight,norm.bias,r,q,s,k,norm.eps,delta is not None,triton.next_power_of_2(k),num_warps=4,enable_fp_fusion=False)
    return r,q,s


def gelu_pack(x):
    k=x.shape[-1];m=x.numel()//k;q=torch.empty((m,k),dtype=torch.int8,device=x.device);s=torch.empty(m,dtype=torch.float32,device=x.device)
    _gelu_pack[(m,)](x,q,s,k,triton.next_power_of_2(k),num_warps=4,enable_fp_fusion=False)
    return q,s


def _linear_packed(layer,q,s,shape):
    m,k=q.shape;n=layer.out_features;out=torch.empty((m,n),dtype=torch.bfloat16,device=q.device)
    _gemm[lambda meta:(triton.cdiv(m,meta['BM']),triton.cdiv(n,meta['BN']))](q,layer.weight,s,layer.scale,layer.scale if layer.bias is None else layer.bias,out,m,n,k,layer.bias is not None)
    return out.reshape(*shape,n)


from .fast_inference import TritonChessModel
from torch.nn import functional as F


class Int8FusedChessModel(TritonChessModel):
    def __init__(self,model):
        nn.Module.__init__(self);self.base=quantize_linears(model,'all');self.config=model.config

    def encode_squares(self,tokens,meta):
        from .triton_inference import add_layer_norm
        m=self.base
        if self.config.history_free:meta=meta*m.history_mask
        x=m.piece(tokens.long())+m.rank(m.squares//8)+m.file(m.squares%8)
        x=x+m.meta(meta.float()).unsqueeze(1);delta=None
        for block in m.blocks:
            x,qi,si=norm_pack(x,delta,block.norm1);b,s,d=x.shape
            q,k,v=_linear_packed(block.qkv,qi,si,(b,s)).view(b,s,3,block.heads,d//block.heads).permute(2,0,3,1,4).unbind(0)
            attention=block.out(F.scaled_dot_product_attention(q,k,v).transpose(1,2).reshape(b,s,d))
            x,qi,si=norm_pack(x,attention,block.norm2)
            hidden=_linear_packed(block.ff[0],qi,si,(b,s));qi,si=gelu_pack(hidden)
            delta=_linear_packed(block.ff[2],qi,si,(b,s))
        _,x=add_layer_norm(x,delta,m.norm.weight,m.norm.bias,m.norm.eps,torch.float32)
        return x
