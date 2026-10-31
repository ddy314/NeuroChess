import numpy as np
import torch

from neurochess.cheap import CheapConfig,CheapModel,load_cheap,save_cheap
from neurochess.frontier import FRONTIER_DTYPE,ranking_loss,ranking_metrics


def test_sibling_ranking_negates_child_side_and_ignores_small_gaps():
    cp=torch.tensor([[0.,-200.,100.,120.,500.]])
    mask=torch.ones((1,5),dtype=torch.bool)
    good=torch.tensor([[0.,-2.,1.,1.2,5.]],requires_grad=True)
    bad=-good
    assert ranking_loss(good,cp,mask)<ranking_loss(bad,cp,mask)
    metrics=ranking_metrics(good.detach().numpy(),cp.numpy(),mask.numpy())
    assert metrics['pairs']==5 and metrics['pairs_correct']==5
    assert metrics['cp_regret_sum']==0
    ranking_loss(bad,cp,mask).backward();assert torch.isfinite(good.grad).all()


def test_masked_mates_and_empty_pairs_have_finite_zero_loss():
    value=torch.zeros((2,5),requires_grad=True)
    cp=torch.zeros_like(value);valid=torch.zeros_like(value,dtype=torch.bool)
    loss=ranking_loss(value,cp,valid);assert loss==0;loss.backward()
    metrics=ranking_metrics(value.detach().numpy(),cp.numpy(),valid.numpy())
    assert metrics['pairs']==0 and metrics['groups']==0
    assert metrics['cp_regret_sum']==0


def test_cp_auxiliary_checkpoint_keeps_wdl_head_and_binary_format(tmp_path):
    torch.manual_seed(87);model=CheapModel(CheapConfig(24,12,True))
    tokens=torch.zeros((2,64),dtype=torch.uint8);meta=torch.zeros((2,12))
    wdl,cp=model.heads(tokens,meta)
    torch.testing.assert_close(wdl,model.logits(tokens,meta))
    (cp.sum()+wdl.sum()).backward()
    assert model.cp_output.weight.grad is not None
    checkpoint=tmp_path/'m.pt';save_cheap(model,checkpoint)
    loaded,_=load_cheap(checkpoint);torch.testing.assert_close(loaded(tokens,meta),model(tokens,meta))
    binary=tmp_path/'m.bin';model.export(binary);assert binary.read_bytes()[:8]==b'NCHEAP01'


def test_raw_cp_storage_preserves_distances_beyond_wdl_saturation():
    rows=np.zeros(1,dtype=FRONTIER_DTYPE)
    rows['cp'][0,1:3]=[-245,-1000];rows['wdl'][0,1:3]=-1
    assert rows['cp'][0,1]!=rows['cp'][0,2]
