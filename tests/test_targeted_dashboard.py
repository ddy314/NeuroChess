import json
from scripts.targeted_dashboard import snapshot


def test_screening_and_training_are_reported_separately(tmp_path):
    (tmp_path/'plan.json').write_text(json.dumps(dict(screen_states=10000,steps=2000)))
    (tmp_path/'status.json').write_text(json.dumps(dict(phase='screening',step=0,roots_completed=1)))
    row=dict(job=dict(split='test',group='game1',depth=2),repair=True,stable=True,decision=dict(policy_error=True,value_mistake=False))
    train=dict(row,job=dict(split='train',group='game2',depth=4))
    (tmp_path/'screened.jsonl').write_text(json.dumps(row)+'\n'+json.dumps(train)+'\n'+'{"job":')
    d=snapshot(tmp_path)
    assert d['screened']==2 and d['confirmed']==2 and d['confirmed_by_split']['train']==1
    assert d['by_depth']=={'2':1,'4':1} and not d['training_started']
    assert d['metrics']==[] and d['validations']==[]
    (tmp_path/'status.json').write_text(json.dumps(dict(phase='training',step=50)))
    (tmp_path/'metrics.jsonl').write_text(json.dumps(dict(step=50,loss=.8))+'\n')
    assert snapshot(tmp_path)['training_started']
    assert snapshot(tmp_path)['metrics'][0]['loss']==.8


def test_full_dashboard_maps_only_recorded_value_metrics(tmp_path):
    from scripts.targeted_full_dashboard import full_snapshot
    p=dict(screen_states=10000,steps=2000,lr=1e-6,microbatch=16,effective_batch=80,trainable='value')
    (tmp_path/'plan.json').write_text(json.dumps(p))
    (tmp_path/'status.json').write_text(json.dumps(dict(phase='training',step=500)))
    (tmp_path/'metrics.jsonl').write_text(json.dumps(dict(step=500,loss=.4,timestamp=100))+'\n')
    (tmp_path/'validation-500.json').write_text(json.dumps(dict(n=4096,ce=.38,mae=.18)))
    d=full_snapshot(tmp_path)
    assert d['metrics'][0]['loss']==[.4,None,None,.4]
    assert d['validation'][0]['value_mae']==.18
    assert 'val_top1' not in d['validation'][0] and 'val_q_mae' not in d['validation'][0]
    assert d['status']['step']==500
