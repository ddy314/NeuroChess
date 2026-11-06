import numpy as np
from scripts.joint_distill_train import sample_rows
from scripts.joint_training_runtime import canonical_update,grouped_losses,remaining_lr,POOL_ORDER,verify_restored_state
from neurochess.train import losses
import torch
import copy
import pytest


def test_sampling_is_uniform_over_positions_not_over_unequal_shards():
    # Every position has equal probability even when the final shard is tiny.
    arrays=[np.arange(100),np.arange(100,101)]
    rng=np.random.default_rng(42)
    rows=sample_rows(arrays,np.array([100,101]),rng,101000)
    assert rows.min()==0 and rows.max()==100
    assert .008 < np.mean(rows==100) < .012
    assert len(rows)==101000


def test_regrouping_preserves_original_objective_and_gradients():
    torch.manual_seed(8);n=320;m=5
    original=[torch.randn(n,m,requires_grad=True),torch.randn(n,m,requires_grad=True),torch.randn(n,3,requires_grad=True)]
    b=dict(tokens=torch.zeros(n,64),policy=torch.rand(n,m),q=torch.rand(n,m),qmask=torch.rand(n,m)>.6,
           value=torch.rand(n,3).softmax(-1),value_weight=torch.rand(n))
    reference=torch.stack([losses(tuple(x[s:s+16] for x in original),{k:v[s:s+16] for k,v in b.items()},.5,1.)[0] for s in range(0,n,16)]).mean()
    reference.backward();grads=[x.grad.clone() for x in original]
    for mb in [32,64]:
        outputs=[x.detach().clone().requires_grad_(True) for x in original]
        objective=torch.stack([grouped_losses(tuple(x[s:s+mb] for x in outputs),{k:v[s:s+mb] for k,v in b.items()})[0] for s in range(0,n,mb)]).mean()
        objective.backward();torch.testing.assert_close(objective,reference)
        for tensor,expected in zip(outputs,grads):torch.testing.assert_close(tensor.grad,expected)


def test_canonical_update_retains_mixture_sampling_calls_and_order():
    calls=[]
    def sample(pool,n):
        calls.append((pool,n));return np.full(n,dict(old=0,new=1,mistake=2)[pool])
    rows=canonical_update(sample)
    assert calls==[(pool,16) for pool in POOL_ORDER]
    assert np.bincount(rows).tolist()==[128,128,64]


def test_remaining_cosine_is_continuous_and_does_not_restart():
    schedule=dict(start_step=7000,end_step=10000,initial_lr=1e-6,min_lr=3e-7)
    rates=[remaining_lr(step,schedule,1e-6) for step in range(7000,10001)]
    assert rates[0]==1e-6 and rates[-1]==3e-7
    assert all(a>=b for a,b in zip(rates,rates[1:]))
    assert remaining_lr(6100,None,1e-6)==1e-6


def test_resume_keeps_weights_adam_moments_and_next_update():
    torch.manual_seed(99);a=torch.nn.Linear(3,2);opt=torch.optim.AdamW(a.parameters(),lr=1e-6)
    x=torch.randn(7,3)
    for _ in range(3):
        opt.zero_grad();a(x).square().mean().backward();opt.step()
    state=copy.deepcopy(dict(model=a.state_dict(),optimizer=opt.state_dict()))
    b=torch.nn.Linear(3,2);b.load_state_dict(state['model']);other=torch.optim.AdamW(b.parameters(),lr=.1);other.load_state_dict(state['optimizer'])
    proof=verify_restored_state(b,other,state)
    assert proof==dict(model_tensors=2,optimizer_tensors=6)
    for model,optimizer in [(a,opt),(b,other)]:
        optimizer.zero_grad();model(x).square().mean().backward();optimizer.step()
    for lhs,rhs in zip(a.parameters(),b.parameters()):torch.testing.assert_close(lhs,rhs,rtol=0,atol=0)
    with pytest.raises(ValueError):verify_restored_state(b,other,state)
