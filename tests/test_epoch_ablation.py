import copy
import numpy as np
import pytest
import torch
from neurochess.data import DTYPE
from neurochess.epoch_ablation import (objective_terms,value_weights,variant_lr,mixed_rows,
    gradient_geometry,screen_search_rows,paired_comparison,paired_q_comparison)
from neurochess.guarded_epoch import ExactEpoch,joint_objective,learning_rate


def test_diagnostic_terms_match_actual_objective_and_gradients():
    torch.manual_seed(31);n,m=6,5
    out=tuple(torch.randn(n,k,requires_grad=True) for k in [m,m,3])
    old=torch.tensor([False,True,False,True,True,False]);nold=int(old.sum())
    b=dict(policy=torch.rand(n,m),q=torch.randn(n,m),qmask=torch.rand(n,m)>.3,
           value=torch.softmax(torch.randn(n,3),-1),value_weight=torch.rand(n),moves=torch.arange(m).repeat(n,1))
    reference=tuple(torch.randn(nold,k) for k in [m,m,3])
    args=(out,b,n,int(b['qmask'].sum()),float(b['value_weight'].sum()),old,reference,nold)
    actual,_=joint_objective(*args);parts=objective_terms(*args);diagnostic=sum(parts.values())
    torch.testing.assert_close(actual,diagnostic)
    a=torch.autograd.grad(actual,out,retain_graph=True);c=torch.autograd.grad(diagnostic,out)
    for x,y in zip(a,c):torch.testing.assert_close(x,y)


def test_partial_balance_preserves_labels_confidence_and_zero_weights():
    a=np.zeros(12,dtype=DTYPE);a['value_weight']=np.arange(12,dtype=np.float32)/12
    a['value'][:8]=[0,1,0];a['value'][8:10]=[1,0,0];a['value'][10:]=[0,0,1]
    zero=value_weights(a.copy(),0);full=value_weights(a.copy(),1);half=value_weights(a.copy(),.5)
    np.testing.assert_array_equal(zero,a)
    np.testing.assert_allclose(half['value_weight'],.5*(a['value_weight']+full['value_weight']))
    np.testing.assert_array_equal(half['value'],a['value'])
    assert half['value_weight'][0]==0
    with pytest.raises(ValueError):value_weights(a,-.1)


def test_lr_retains_absolute_cosine_schedule_and_only_B_ramps():
    for offset in [1,25,50,100,500]:
        step=4000+offset;base=learning_rate(step,12761,3e-7,100)
        assert variant_lr(step,12761,3e-7,100,'A',offset)==base
        assert variant_lr(step,12761,3e-7,100,'C',offset)==base
        assert variant_lr(step,12761,3e-7,100,'B',offset)==pytest.approx(base*(1+.5*min(offset/50,1)))
    with pytest.raises(ValueError):variant_lr(4000,12761,3e-7,100,'B',0)


def test_all_branches_replay_identical_future_rows_and_respect_cursor(tmp_path):
    paths=[]
    for shard in range(3):
        a=np.zeros(12,dtype=DTYPE);a['value_weight']=1;a['value'][:8]=[0,1,0];a['value'][8:]=[1,0,0]
        a['key'][:,0]=np.arange(shard*12,(shard+1)*12);p=tmp_path/f'train-{shard}.npy';np.save(p,a);paths.append(p)
    class Replay:
        def __init__(self):self.rng=np.random.default_rng(13)
        def sample(self,n):
            r=np.zeros(n,dtype=DTYPE);r['value_weight']=1;r['value']=[0,0,1];r['key'][:,0]=self.rng.integers(100,200,n);return r
    records=[];arrays=[]
    for alpha in [1,1,.5]:
        stream=ExactEpoch(paths,9,6);replay=Replay();record=[];result=[]
        for step in range(3):
            rows,old,nnew,nold,digest=mixed_rows(stream,replay,dict(new_per_update=6,seed=9),4000+step,alpha)
            record.append(digest);result.append(rows.copy());assert (nnew,nold)==(6,2)
        records.append(record);arrays.append(result)
    assert records[0]==records[1]==records[2]
    for a,b in zip(arrays[0],arrays[1]):np.testing.assert_array_equal(a,b)
    assert any(not np.array_equal(a['value_weight'],b['value_weight']) for a,b in zip(arrays[0],arrays[2]))


def test_gradient_geometry_records_cancellation_without_nan():
    d=gradient_geometry(dict(policy=[torch.tensor([1.,0.])],Q=[torch.tensor([-1.,0.])],value=[torch.zeros(2)],retention=[torch.zeros(2)]))
    assert d['cosines']['policy|Q']==pytest.approx(-1)
    assert d['cancellation_ratio']==0
    assert all(np.isfinite(x) for x in d['cosines'].values())


def test_search_gate_exact_boundaries_preserve_parent_reference():
    def rows(candidate,baseline=0):return [dict(index=0,candidate_regret=candidate,baseline_regret=baseline)]
    assert screen_search_rows(rows(500,500))['passed']
    assert not screen_search_rows(rows(100,25))['passed']
    assert screen_search_rows(rows(10))['passed']
    assert not screen_search_rows(rows(11))['passed']


def test_optimizer_branches_start_from_identical_moments_and_do_not_mutate_anchor():
    model=torch.nn.Linear(2,1);opt=torch.optim.AdamW(model.parameters(),lr=.001)
    model(torch.ones(1,2)).sum().backward();opt.step()
    state=copy.deepcopy(opt.state_dict());weight=copy.deepcopy(model.state_dict())
    original={k:{q:v.clone() if torch.is_tensor(v) else v for q,v in x.items()} for k,x in state['state'].items()}
    for lr in [.001,.0015,.001]:
        branch=torch.nn.Linear(2,1);branch.load_state_dict(weight)
        optimizer=torch.optim.AdamW(branch.parameters(),lr=.001);optimizer.load_state_dict(copy.deepcopy(state))
        for group in optimizer.param_groups:group['lr']=lr
        branch(torch.ones(1,2)).sum().backward();optimizer.step()
    for k,x in original.items():
        for q,v in x.items():torch.testing.assert_close(state['state'][k][q],v)


def test_paired_development_interval_is_reproducible_and_labeled():
    d=paired_comparison(np.zeros(20),np.ones(20),repeats=30)
    assert d['mean_delta']==1 and d['bootstrap_95']==[1,1]
    assert 'not independent' in d['scope']


def test_paired_Q_preserves_label_weighted_denominator():
    d=paired_q_comparison([0,0],[1,0],[1,9],repeats=30)
    assert d['mean_delta']==pytest.approx(.1) and d['q_labels']==10
