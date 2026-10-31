import json,os,subprocess,sys
import numpy as np
import torch
import chess
from neurochess.data import ShardWriter,record,Batches,MixtureBatches


def make_dataset(root,value):
    writer=ShardWriter(root,size=16)
    for i in range(64):
        row,_=record(chess.Board());row['policy'][0]=1;row['q'][0]=value;row['qmask'][0]=True
        row['value']=[.2,.6,.2];row['value_weight']=1;writer.add('train',row)
        if i<8:writer.add('val',row)
    writer.close({'fixture':True})


def test_mixture_ratio_and_human_masks(tmp_path):
    for name,value in [('core',.1),('hard',.9),('human',.2)]:make_dataset(tmp_path/name,value)
    batches=list(MixtureBatches(Batches(tmp_path/'core','train',4),tmp_path/'hard',tmp_path/'human'))
    group=batches[:10]
    assert all(abs(float(x['q'][0,0])-.1)<.001 for x in group[:7])
    assert all(abs(float(x['q'][0,0])-.9)<.001 for x in group[7:9])
    assert not group[9]['qmask'].any() and not group[9]['value_weight'].any()


def test_two_worker_hard_mixture_resume_is_exact(tmp_path):
    for name,value in [('core',.1),('hard',.9),('human',.2)]:make_dataset(tmp_path/name,value)
    cmd=[sys.executable,'-m','neurochess.train','--data',str(tmp_path/'core'),'--hard-data',str(tmp_path/'hard'),
        '--replay-data',str(tmp_path/'human'),'--config','configs/tiny.json','--steps','4','--batch-size','4','--accum','4',
        '--workers','2','--cpu-threads','2','--save-every','2','--eval-batches','2','--history-free']
    env={**os.environ,'OMP_NUM_THREADS':'2'}
    subprocess.run(cmd+['--out',str(tmp_path/'full')],check=True,capture_output=True,env=env)
    subprocess.run(cmd+['--out',str(tmp_path/'resumed'),'--stop-after','2'],check=True,capture_output=True,env=env)
    subprocess.run(cmd+['--out',str(tmp_path/'resumed'),'--resume',str(tmp_path/'resumed/last.pt')],check=True,capture_output=True,env=env)
    a=torch.load(tmp_path/'full/last.pt',weights_only=False);b=torch.load(tmp_path/'resumed/last.pt',weights_only=False)
    for key in a['model']:torch.testing.assert_close(a['model'][key],b['model'][key],rtol=0,atol=0)
    assert (tmp_path/'resumed/hard-dataset.json').exists()


def test_new_mixture_and_book_replay_are_deterministic_policy_only(tmp_path):
    for name,value in [('core',.1),('hard',.9),('human',.2),('book',.7)]:make_dataset(tmp_path/name,value)
    def batches():return list(MixtureBatches(Batches(tmp_path/'core','train',4),tmp_path/'hard',tmp_path/'human',5,4,tmp_path/'book',2))
    rows=batches();again=batches()
    assert len(rows)==31
    for group,expected in [(rows[:10],.2),(rows[10:20],.7)]:
        assert all(abs(float(r['q'][0,0])-.1)<.001 for r in group[:5])
        assert all(abs(float(r['q'][0,0])-.9)<.001 for r in group[5:9])
        assert abs(float(group[9]['q'][0,0])-expected)<.001
        assert not group[9]['qmask'].any() and not group[9]['value_weight'].any()
    for a,b in zip(rows,again):
        for k in a:torch.testing.assert_close(a[k],b[k],rtol=0,atol=0)
