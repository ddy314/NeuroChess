import copy,json
import chess
import numpy as np
import pytest
import torch
from neurochess.dynamic_curriculum import Store,state_id,state_job,DYNAMIC_TARGET_SPEC,encode_jobs_grouped,board_for
from neurochess.encoding import position_key,encode
from neurochess.targets import TARGET_SPEC
from neurochess.teacher_stability import qualify,qualify_guarded,qualified_decision
from neurochess.data import DTYPE
from neurochess.dynamic_training import losses
from neurochess.dynamic_training import normalizers

SHA='a'*64


def test_grouped_encoding_preserves_repetition_clocks_and_diverging_histories():
    jobs=[job(['g1f3','g8f6','f3g1','f6g8']*2),job(['e2e4','a7a6','e4e5','d7d5']),job(['e2e4']),job(['d2d4','d7d5']),job([],group='other'),job(['g1f3','g8f6','f3g1','f6g8'])]
    tokens,meta=encode_jobs_grouped(jobs)
    for i,j in enumerate(jobs):
        reference=encode(board_for(j));np.testing.assert_array_equal(tokens[i],reference[0]);np.testing.assert_array_equal(meta[i],reference[1])
    assert meta[0,7] and meta[-1,6] and meta[1,11]
    wrong=dict(jobs[1],key='0'*32)
    with pytest.raises(ValueError,match='identity'):encode_jobs_grouped([wrong])
    null=job(['0000'])
    with pytest.raises(ValueError,match='Null'):encode_jobs_grouped([null])


def job(moves=(),group='game',split='train',**extra):
    b=chess.Board()
    for m in moves:b.push_uci(m)
    return dict(initial_fen=chess.STARTING_FEN,history=list(moves),key=position_key(b).hex(),group=group,split=split,role='root',**extra)


def evidence(j,sha=SHA):
    prev=[dict(line='e2e4',cp=600),dict(line='d2d4',cp=0)];current=[dict(line='e2e4',cp=900),dict(line='d2d4',cp=0)]
    pred=dict(model_sha256=sha,move='d2d4',top_moves=['d2d4'],value=0.,support_q={})
    s=qualify(prev,current,True);spec=dict(target_spec=DYNAMIC_TARGET_SPEC,minimum_confirm_nodes=300000,stability_cp=30,wdl_stability=.02)
    meta=dict(job=j,teacher=current,stable=True,prediction=pred,supervision=s,attempts=[dict(budget=100000,independent=True,teacher=prev),dict(budget=300000,independent=True,teacher=current)])
    return spec,meta,pred


def labelled(store,j=None):
    j=j or job();case=store.add(j);store.register_model(SHA,13000,'frozen.pt');spec,meta,pred=evidence(j)
    store.predict(case,pred);label=store.save_label(case,spec,meta);store.refresh(case,SHA,label)
    return case,spec,label


def test_target_aware_stability_preserves_value_without_inventing_policy():
    spec,meta,pred=evidence(job());s=meta['supervision']
    assert not s['policy'] and s['value'] and s['q_moves']==['d2d4','e2e4']
    assert qualified_decision(meta['attempts'][0]['teacher'],meta['teacher'],pred,True,s)['repair_heads']==['Q','value']
    # Stable CP is not enough when the actual WDL target remains unsettled.
    a=[dict(line='e2e4',cp=90)];b=[dict(line='e2e4',cp=115)]
    q=qualify(a,b,True);assert q['policy'] and not q['value'] and not q['q_moves']


def test_opposite_pov_and_bound_scores_are_not_accepted_as_stable():
    a=[dict(line='e7e5',cp=-600),dict(line='d7d5',cp=0)];b=[dict(line='e7e5',cp=-900),dict(line='d7d5',cp=0)]
    assert qualify(a,b,False)['value']
    b[0]['bound']='lower';assert not qualify(a,b,False)['value']
    with pytest.raises(ValueError):qualify(a,b[:1],False)


def test_unrestricted_guard_masks_subset_value_and_missing_best_policy(tmp_path):
    store=Store(tmp_path/'store.sqlite');case,spec,_=labelled(store)
    spec=dict(spec,unrestricted_value_guard=True,value_guard_nodes=[100000,300000])
    _,meta,pred=evidence(job())
    audit=[dict(budget=n,independent=True,unrestricted=True,candidates=None,teacher=[dict(line='g1f3',cp=0)]) for n in spec['value_guard_nodes']]
    meta['value_audit']=audit;meta['supervision']=qualify_guarded(meta['attempts'][0]['teacher'],meta['teacher'],True,spec,audit)
    assert not meta['supervision']['value'] and not meta['supervision']['policy'] and len(meta['supervision']['q_moves'])==2
    label=store.save_label(case,spec,meta);row=np.frombuffer(store.db.execute('SELECT payload FROM teachers WHERE label_id=?',(label,)).fetchone()[0],DTYPE)[0]
    assert row['value_weight']==0 and row['policy'].sum()==0 and row['qmask'].sum()==2
    assert store.cached_value_audit(case,spec)==audit
    meta['supervision']['value']=True
    with pytest.raises(ValueError,match='Forged'):store.save_label(case,spec,meta)
    store.close()


def test_full_history_identity_and_cross_split_canonical_quarantine(tmp_path):
    store=Store(tmp_path/'store.sqlite');a,_,_=labelled(store)
    repeated=job(['g1f3','g8f6','f3g1','f6g8'],group='holdout',split='val')
    b=store.add(repeated)
    assert a!=b and job()['key']==repeated['key']
    assert store.db.execute('SELECT SUM(quarantine) FROM states').fetchone()[0]==2
    assert store.counts(SHA)['verified_unique_mistake_roots']==0
    with pytest.raises(ValueError,match='Source game crosses'):store.add(job(['e2e4'],split='val'))
    store.close()


def test_new_model_retires_solved_mistakes_and_never_reuses_stale_predictions(tmp_path):
    store=Store(tmp_path/'store.sqlite');case,spec,label=labelled(store)
    assert store.counts(SHA)['verified_unique_mistake_roots']==1
    other='b'*64;store.register_model(other,15000,'new.pt')
    with pytest.raises(ValueError):store.refresh(case,other,label)
    pred=dict(model_sha256=other,move='e2e4',top_moves=['e2e4'],value=1.,support_q={'e2e4':1.})
    store.predict(case,pred);store.refresh(case,other,label)
    assert store.counts(other)['verified_unique_mistake_roots']==0
    assert store.cached_label(case,pred,spec)[0]==label
    assert store.cached_label(case,dict(pred,top_moves=['g1f3']),spec) is None
    changed=copy.deepcopy(spec);changed['minimum_confirm_nodes']=1000000
    assert store.cached_label(case,pred,changed) is None
    store.close()


def test_masked_payload_and_independent_teacher_gate(tmp_path):
    store=Store(tmp_path/'store.sqlite');case,spec,label=labelled(store)
    row=np.frombuffer(store.db.execute('SELECT payload FROM teachers WHERE label_id=?',(label,)).fetchone()[0],DTYPE)[0]
    assert row['policy'].sum()==0 and row['value_weight']==1 and row['qmask'].sum()==2
    _,meta,_=evidence(job());meta['attempts'][-1]['independent']=False
    with pytest.raises(ValueError,match='independent'):store.save_label(case,spec,meta)
    _,meta,_=evidence(job());meta['supervision']['policy']=True
    with pytest.raises(ValueError,match='Forged'):store.save_label(case,spec,meta)
    store.close()


def test_coarse_evidence_can_upgrade_without_mutating_prior_attempt(tmp_path):
    store=Store(tmp_path/'store.sqlite');j=job();case=store.add(j);spec,meta,_=evidence(j)
    coarse=dict(meta,stable=None,supervision=None,attempts=meta['attempts'][:1],teacher=meta['attempts'][0]['teacher'])
    old=store.save_label(case,spec,coarse);new=store.save_label(case,spec,meta)
    assert old!=new and store.db.execute('SELECT COUNT(*) FROM teachers').fetchone()[0]==2
    assert store.db.execute('SELECT stable FROM teachers WHERE label_id=?',(old,)).fetchone()[0]==0
    store.close()


def test_replay_is_durable_capped_and_has_no_silent_shortage_fallback(tmp_path):
    path=tmp_path/'store.sqlite';store=Store(path);case,_,_=labelled(store)
    a=store.sample(SHA,1,'run',1,max_repeats=2);assert a==[case]
    assert store.sample(SHA,1,'run',1,max_repeats=2)==a
    assert store.db.execute('SELECT SUM(total) FROM replay_usage').fetchone()[0]==1
    store.close();store=Store(path)
    assert store.sample(SHA,1,'run',1,max_repeats=2)==a
    store.sample(SHA,1,'run',2,max_repeats=2)
    with pytest.raises(RuntimeError,match='replenish'):store.sample(SHA,1,'run',3,max_repeats=2)
    with pytest.raises(ValueError,match='Resume draw'):store.sample('b'*64,1,'run',1,max_repeats=2)
    with pytest.raises(ValueError,match='seed/exposure'):store.sample(SHA,1,'run',1,max_repeats=3)
    assert store.db.execute('SELECT SUM(total) FROM replay_usage').fetchone()[0]==2
    store.exclude([job()['key']],'new holdout collision')
    with pytest.raises(ValueError,match='quarantined'):store.sample(SHA,1,'run',1,max_repeats=2)
    store.close()


def test_descendants_keep_legal_edges_and_do_not_inflate_root_count(tmp_path):
    store=Store(tmp_path/'store.sqlite');root=job();parent=store.add(root);b=chess.Board();b.push_uci('e2e4')
    child=state_job(root,b,'student',1);case=store.add(child);store.link(parent,'e2e4',case,'student');store.link(parent,'e2e4',case,'teacher')
    assert len(store.job(case)['history'])==1 and store.job(case)['root_id']==parent
    assert store.db.execute('SELECT COUNT(*) FROM edges').fetchone()[0]==2
    store.link(parent,'e2e4',case,'student',SHA,parent,1)
    assert store.db.execute('SELECT root_id,depth,model_sha FROM rollout_visits').fetchone()==(parent,1,SHA)
    with pytest.raises(ValueError,match='provenance'):store.link(parent,'e2e4',case,'student',SHA,parent,2)
    with pytest.raises(ValueError):store.link(parent,'d2d4',case,'student')
    terminal=job(['f2f3','e7e5','g2g4','d8h4'],role_override='unused')
    end=store.add(terminal);assert store.job(end)['terminal_value']==-1
    assert store.job(end)['terminal_reason']=='CHECKMATE'
    store.register_model(SHA,13000,'frozen.pt');label=store.exact_terminal(end,SHA)
    row=np.frombuffer(store.db.execute('SELECT payload FROM teachers WHERE label_id=?',(label,)).fetchone()[0],DTYPE)[0]
    assert row['policy'].sum()==0 and row['qmask'].sum()==0 and row['value'][2]==1
    claim=store.add(job(['g1f3','g8f6','f3g1','f6g8']*2,group='repeat-game'))
    assert store.job(claim)['claimable_draw'] and not store.job(claim).get('terminal')
    with pytest.raises(ValueError,match='Optional'):store.exact_terminal(claim,SHA)
    store.close()


def test_missing_policy_targets_do_not_dilute_policy_gradient():
    logits=torch.tensor([[1.,-1.]],requires_grad=True)
    def batch(policy):
        return dict(policy=torch.tensor(policy),q=torch.zeros(len(policy),2),qmask=torch.zeros(len(policy),2,dtype=torch.bool),value=torch.ones(len(policy),3)/3,value_weight=torch.zeros(len(policy)))
    loss,_=losses((logits,torch.zeros(1,2),torch.zeros(1,3)),batch([[0.,1.]]));gradient=torch.autograd.grad(loss,logits)[0]
    expanded=torch.tensor([[1.,-1.],[2.,-3.]],requires_grad=True)
    loss,_=losses((expanded,torch.zeros(2,2),torch.zeros(2,3)),batch([[0.,1.],[0.,0.]]));g=torch.autograd.grad(loss,expanded)[0]
    torch.testing.assert_close(g[:1],gradient);assert not g[1].any()


def test_q_only_error_can_be_mined_with_correct_policy_and_value():
    _,meta,_=evidence(job());pred=dict(move='e2e4',value=1.,support_q={'d2d4':-1.})
    decision=qualified_decision(meta['attempts'][0]['teacher'],meta['teacher'],pred,True,meta['supervision'])
    assert decision['repair_heads']==['Q'] and decision['usable_error']


def test_general_replay_is_an_independent_cohort_not_an_alias_of_mistakes(tmp_path):
    store=Store(tmp_path/'store.sqlite');case,_,_=labelled(store)
    with pytest.raises(RuntimeError,match='replenish'):store.sample(SHA,1,'run',1,pool='new')
    store.add(dict(job(),replay_anchor=True))
    assert store.counts(SHA)['qualified_general_replay_roots']==1
    assert store.sample(SHA,1,'run',1,pool='new')==[case]
    with pytest.raises(RuntimeError,match='replenish'):store.sample(SHA,1,'run',1,pool='mistake')
    store.close()


def test_masked_effective_batch_objective_is_invariant_to_regrouping():
    torch.manual_seed(4);n=320
    batch=dict(policy=torch.rand(n,3),q=torch.rand(n,3),qmask=torch.rand(n,3)>.6,value=torch.rand(n,3),value_weight=(torch.rand(n)>.3).float())
    batch['policy'][torch.rand(n)>.5]=0;batch['value']/=batch['value'].sum(-1,keepdim=True)
    outputs=[torch.randn(n,3,requires_grad=True) for _ in range(3)];reference=losses(outputs,batch)[0]
    gradients=torch.autograd.grad(reference,outputs);norm=normalizers(batch)
    for micro in [16,32,64]:
        result=sum(losses([x[i:i+micro] for x in outputs],{k:v[i:i+micro] for k,v in batch.items()},normalization=norm)[0] for i in range(0,n,micro))
        actual=torch.autograd.grad(result,outputs)
        for a,b in zip(actual,gradients):torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-7)
