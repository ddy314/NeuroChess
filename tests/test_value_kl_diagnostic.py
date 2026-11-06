import copy
import torch
from neurochess.guarded_epoch import joint_objective
from neurochess.value_kl_diagnostic import NAMES, split_terms, adam_counterfactual, decision


def test_split_preserves_objective_and_gradients():
    torch.manual_seed(4)
    out=[torch.randn(7, k, requires_grad=True) for k in (5,5,3)]
    b=dict(policy=torch.rand(7,5),q=torch.randn(7,5),qmask=torch.rand(7,5)>.4,
           value=torch.rand(7,3).softmax(-1),value_weight=torch.rand(7),moves=torch.zeros(7,5,dtype=torch.long))
    old=torch.tensor([True,False,False,True,False,True,False])
    teacher=[torch.randn(int(old.sum()),k) for k in (5,5,3)]
    args=(out,b,7,int(b['qmask'].sum()),float(b['value_weight'].sum()),old,teacher,3)
    reference,_=joint_objective(*args)
    terms=split_terms(*args)
    assert tuple(terms)==NAMES
    torch.testing.assert_close(sum(terms.values()),reference)
    a=torch.autograd.grad(reference,out,retain_graph=True)
    z=torch.autograd.grad(sum(terms.values()),out,retain_graph=True)
    for x,y in zip(a,z): torch.testing.assert_close(x,y)
    reduced=sum(split_terms(*args,value_kl=2).values())
    torch.testing.assert_close(reduced,reference-.5*terms['retention_value'])


def test_no_replay_microbatch():
    out=[torch.randn(2,k,requires_grad=True) for k in (3,3,3)]
    b=dict(policy=torch.rand(2,3),q=torch.rand(2,3),qmask=torch.ones(2,3),value=torch.rand(2,3),value_weight=torch.ones(2),moves=torch.zeros(2,3))
    t=split_terms(out,b,1024,600,900,torch.zeros(2,dtype=torch.bool),None,256)
    assert all(float(t[n].detach())==0 for n in NAMES[3:])
    sum(t.values()).backward()
    assert all(torch.isfinite(x.grad).all() for x in out)


def test_adam_counterfactual_matches_real_update_without_mutation():
    torch.manual_seed(8)
    p=torch.nn.Parameter(torch.randn(17,dtype=torch.float64))
    opt=torch.optim.AdamW([p],lr=.001,weight_decay=.01)
    for _ in range(3):
        p.grad=torch.randn_like(p);opt.step()
    original=p.detach().clone(); state=copy.deepcopy(opt.state_dict())
    vectors=[[torch.randn_like(p)] for _ in NAMES]
    got=adam_counterfactual([p],vectors,state,.002)
    for coef in (4,2):
        candidate=torch.nn.Parameter(original.clone())
        optimizer=torch.optim.AdamW([candidate],lr=.001)
        optimizer.load_state_dict(copy.deepcopy(state));optimizer.param_groups[0]['lr']=.002
        candidate.grad=sum(v[0]*(coef/4 if i==8 else 1) for i,v in enumerate(vectors))
        torch.nn.utils.clip_grad_norm_([candidate],1.)
        optimizer.step();delta=candidate.detach()-original
        expected=float((delta*sum(v[0] for v in vectors[:3])).sum())
        assert abs(got[str(coef)]['new_first_order_change']-expected)<1e-10
    torch.testing.assert_close(p,original)
    torch.testing.assert_close(state['state'][0]['exp_avg'],opt.state_dict()['state'][0]['exp_avg'])


def test_fail_closed_trigger():
    r=dict(value_kl_cosine_with_new=-.4,value_kl_conflict_share=.8,value_kl_norm_over_new=.5,adam={'relative_new_gain':.1})
    assert not decision([r]*15)['run_D']
    assert decision([r]*16)['run_D']
    no=dict(r,value_kl_cosine_with_new=.3)
    assert not decision([no]*5+[r]*11)['run_D']
    no=dict(r,adam={'relative_new_gain':float('nan')})
    assert not decision([no]+[r]*15)['run_D']
