"""Experimental frozen-model backends. The locked search baseline is unchanged."""
import torch
from torch import nn
from torch.nn import functional as F


def resident_linears(model):
    """Hoist the SAME BF16 linear-weight conversions used by CUDA autocast.

    Embeddings, residuals and LayerNorm parameters remain FP32. Use a separate
    loaded model: this mutates the candidate's linear storage, not the baseline.
    """
    if next(model.parameters()).device.type!='cuda':raise ValueError('CUDA required')
    for layer in model.modules():
        if isinstance(layer,nn.Linear):layer.to(dtype=torch.bfloat16)
    return model.eval()


class TritonChessModel(nn.Module):
    def __init__(self,model):
        super().__init__();self.base=resident_linears(model);self.config=model.config

    def encode_squares(self,tokens,meta):
        from .triton_inference import add_layer_norm
        m=self.base
        if self.config.history_free:meta=meta*m.history_mask
        x=m.piece(tokens.long())+m.rank(m.squares//8)+m.file(m.squares%8)
        x=x+m.meta(meta.float()).unsqueeze(1)
        delta=None
        for block in m.blocks:
            x,n=add_layer_norm(x,delta,block.norm1.weight,block.norm1.bias,block.norm1.eps,torch.bfloat16)
            b,s,d=x.shape
            q,k,v=block.qkv(n).view(b,s,3,block.heads,d//block.heads).permute(2,0,3,1,4).unbind(0)
            attention=block.out(F.scaled_dot_product_attention(q,k,v).transpose(1,2).reshape(b,s,d))
            x,n=add_layer_norm(x,attention,block.norm2.weight,block.norm2.bias,block.norm2.eps,torch.bfloat16)
            delta=block.ff(n)
        _,x=add_layer_norm(x,delta,m.norm.weight,m.norm.bias,m.norm.eps,torch.float32)
        return x

    def forward(self,tokens,meta,moves):
        m=self.base;x=self.encode_squares(tokens,meta);ids=moves.long().clamp_min(0)
        src,dst,promo=ids//320,(ids//5)%64,ids%5;dim=self.config.head_width
        a=m.source(x).gather(1,src.unsqueeze(-1).expand(-1,-1,dim))
        b=m.destination(x).gather(1,dst.unsqueeze(-1).expand(-1,-1,dim))
        logits,q=m.action(a+b+a*b/dim**.5+m.promotion(promo)).unbind(-1)
        return logits.float().masked_fill(moves<0,-1e9),q.float().tanh(),m.value(x.mean(1)).float()
