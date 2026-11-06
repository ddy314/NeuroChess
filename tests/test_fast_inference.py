import pytest
import torch

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA inference kernels')


def test_fused_residual_norm_preserves_residual_and_matches_fp32_formula():
    from neurochess.triton_inference import add_layer_norm
    torch.manual_seed(83)
    for width in [32,128,896]:
        x=torch.randn(17,width,device='cuda');delta=torch.randn_like(x,dtype=torch.bfloat16)
        w=torch.randn(width,device='cuda');b=torch.randn(width,device='cuda')
        for change in [None,delta]:
            r,y=add_layer_norm(x,change,w,b,1e-5,torch.float32)
            expected=x if change is None else x+change
            torch.testing.assert_close(r,expected,rtol=0,atol=0)
            torch.testing.assert_close(y,torch.nn.functional.layer_norm(expected,(width,),w,b,1e-5),atol=2e-5,rtol=2e-5)


def test_integer_tensor_core_gemm_matches_explicit_quantized_reference():
    from neurochess.int8_inference import Int8Linear,_pack
    import triton
    torch.manual_seed(84)
    for rows,k,n in [(17,96,48),(64,128,256)]:
        linear=torch.nn.Linear(k,n,device='cuda').eval();quant=Int8Linear(linear)
        x=torch.randn(rows,k,device='cuda');actual=quant(x)
        bf=x.to(torch.bfloat16).float();scale=bf.abs().amax(1).clamp_min(1e-8)/127
        q=(bf/scale[:,None]).round().clamp(-127,127).to(torch.int8)
        prod=q.cpu().to(torch.int32)@quant.weight.cpu().to(torch.int32)
        expected=(prod.cuda().float()*scale[:,None]*quant.scale[None,:]+quant.bias.float()).to(torch.bfloat16)
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)


def test_gpu_head_fusion_preserves_legal_support_and_stable_ties():
    import numpy as np
    from neurochess.fused_heads import fuse_cuda
    from neurochess.multihead import fuse_heads
    torch.manual_seed(85);n=7
    p=torch.randn(n,256,device='cuda');q=torch.randn_like(p).tanh();w=torch.randn(n,3,device='cuda')
    ids=torch.full((n,256),-1,dtype=torch.int16,device='cuda')
    for i in range(n):ids[i,:i+1]=0
    p[:,:7]=1.;p=p.masked_fill(ids<0,-1e9)
    y=fuse_cuda(p,q,w,ids).cpu().numpy()
    a,b,c,s=fuse_heads(p.cpu().numpy(),q.cpu().numpy(),w.softmax(-1).cpu().numpy(),ids.cpu().numpy()>=0)
    np.testing.assert_allclose(y[:,:256],a,atol=2e-6,rtol=2e-6)
    np.testing.assert_allclose(y[:,256:512],b,atol=2e-6,rtol=2e-6,equal_nan=True)
    np.testing.assert_allclose(y[:,512],c,atol=2e-6,rtol=2e-6)
    np.testing.assert_array_equal(np.isfinite(y[:,256:512]),s)
