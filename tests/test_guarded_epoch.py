import copy
import numpy as np
import pytest
import torch
from neurochess.data import DTYPE,FIELDS
from neurochess.guarded_epoch import ExactEpoch,replay_count,guard_metrics,joint_objective,learning_rate
from neurochess.model import ChessModel,ModelConfig


def test_exact_epoch_no_repeats_and_resume_mid_shard(tmp_path):
    files=[];start=0
    for i,n in enumerate([5,9,3]):
        a=np.zeros(n,dtype=DTYPE);a['tokens'][:,0]=np.arange(start,start+n);start+=n
        p=tmp_path/f'train-{i}.npy';np.save(p,a);files.append(p)
    s=ExactEpoch(files,17);first=s.next(7)['tokens'][:,0]
    resumed=ExactEpoch(files,17,7);rest=resumed.next(10)['tokens'][:,0]
    full=ExactEpoch(files,17).next(17)['tokens'][:,0]
    assert np.array_equal(np.r_[first,rest],full) and sorted(full.tolist())==list(range(17))
    with pytest.raises(EOFError):resumed.next(1)
    assert replay_count(768)==256 and replay_count(128)==43


def metrics():
    a=dict(loss=2.,policy_loss=1.5,value_loss=.4,top1=.4,q_mae=.2,value_mae=.2,
           value_strata=dict(winning=dict(n=100,mae=.2),neutral=dict(n=100,mae=.2),losing=dict(n=100,mae=.2)))
    return {k:copy.deepcopy(a) for k in ['new','old_core','old_general']}


def test_continuous_guard_rejects_previous_pawn_failure_even_if_new_improves():
    b=metrics();now=copy.deepcopy(b);now['new']['loss']=1.8
    assert guard_metrics(b,now)['passed']
    now['old_general']['loss']*=1.023574
    now['old_general']['value_mae']*=1.0909
    gate=guard_metrics(b,now)
    assert not gate['passed'] and 'old_general/value' in gate['failed'] and 'old_general/loss' in gate['failed']


def test_value_stratum_and_individual_heads_are_guarded():
    b=metrics();now=copy.deepcopy(b);now['old_core']['value_strata']['losing']['mae']+=.02
    assert not guard_metrics(b,now)['passed']
    now=copy.deepcopy(b);now['old_core']['policy_loss']*=1.02
    assert 'old_core/policy' in guard_metrics(b,now)['failed']
    now=copy.deepcopy(b);now['new']['loss']=float('nan')
    assert not guard_metrics(b,now)['passed']


def test_objective_microbatch_invariance_including_replay_anchor_and_tail():
    torch.manual_seed(7);n=7;m=5
    p=torch.randn(n,m,requires_grad=True);q=torch.randn(n,m,requires_grad=True);v=torch.randn(n,3,requires_grad=True)
    target=torch.softmax(torch.randn(n,m),-1);qm=torch.rand(n,m)>.5
    b=dict(policy=target,q=torch.randn(n,m),qmask=qm,value=torch.softmax(torch.randn(n,3),-1),value_weight=torch.ones(n),moves=torch.arange(m).repeat(n,1))
    old=torch.tensor([0,1,0,0,1,0,1],dtype=torch.bool);teacher=tuple(torch.randn(int(old.sum()),k) for k in [m,m,3])
    total,_=joint_objective((p,q,v),b,n,int(qm.sum()),n,old,teacher,int(old.sum()))
    grads=torch.autograd.grad(total,(p,q,v),retain_graph=True)
    micro=0.;offset=0
    for lo,hi in [(0,3),(3,6),(6,7)]:
        count=int(old[lo:hi].sum());t=tuple(x[offset:offset+count] for x in teacher);offset+=count
        loss,_=joint_objective((p[lo:hi],q[lo:hi],v[lo:hi]),{k:x[lo:hi] for k,x in b.items()},n,int(qm.sum()),n,old[lo:hi],t,int(old.sum()));micro+=loss
    torch.testing.assert_close(total,micro)
    for a,c in zip(grads,torch.autograd.grad(micro,(p,q,v))):torch.testing.assert_close(a,c)


def test_all_three_heads_and_shared_encoder_update_teacher_frozen():
    torch.manual_seed(17);model=ChessModel(ModelConfig(32,1,4,16,True));teacher=copy.deepcopy(model).requires_grad_(False)
    b=dict(tokens=torch.randint(0,13,(4,64)),meta=torch.zeros(4,12),moves=torch.arange(5).repeat(4,1),
           policy=torch.softmax(torch.randn(4,5),-1),q=torch.ones(4,5)*.7,qmask=torch.ones(4,5,dtype=torch.bool),value=torch.tensor([[.8,.1,.1]]*4),value_weight=torch.ones(4))
    with torch.no_grad():ref=teacher(b['tokens'][2:],b['meta'][2:],b['moves'][2:])
    loss,_=joint_objective(model(b['tokens'],b['meta'],b['moves']),b,4,20,4,torch.tensor([0,0,1,1],dtype=torch.bool),ref,2);loss.backward()
    for grad in [model.piece.weight.grad,model.action[1].weight.grad[0],model.action[1].weight.grad[1],model.value[2].weight.grad]:assert grad.abs().sum()>0
    assert all(p.grad is None for p in teacher.parameters())
    assert learning_rate(1,1000)<learning_rate(100,1000) and learning_rate(1000,1000)==pytest.approx(1e-7)


def test_numpy_validation_scalars_can_be_reported_without_losing_gate():
    import json
    from neurochess.pawn_data import write_json
    b=metrics();now=copy.deepcopy(b)
    for group in now.values():
        for k in ['loss','policy_loss','value_loss','top1','q_mae','value_mae']:
            group[k]=np.float64(group[k])
    now['old_general']['value_strata']['winning']['mae']=np.float64(.22)
    gate=guard_metrics(b,now)
    assert not gate['passed'] and 'old_general/winning' in gate['failed']
    assert all(type(x) is bool for x in gate['checks'].values())
    assert json.loads(json.dumps(gate))['passed'] is False


def test_value_weights_balance_strata_without_altering_targets():
    from neurochess.guarded_epoch import balance_value_weights
    a=np.zeros(12,dtype=DTYPE);a['value_weight']=1
    a['value'][:8]=[0,1,0];a['value'][8:10]=[1,0,0];a['value'][10:]=[0,0,1]
    target=a['value'].copy();balance_value_weights(a)
    assert np.array_equal(a['value'],target)
    assert [a['value_weight'][i:j].sum() for i,j in [(0,8),(8,10),(10,12)]]==[4,4,4]
