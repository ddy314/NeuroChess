import numpy as np
import chess
import torch
from neurochess.import_evals import convert, score_targets
from neurochess.encoding import move_id
from neurochess.model import ChessModel, ModelConfig
from neurochess.data import Batches, ReplayBatches


def entry(fen, moves):
    return dict(fen=' '.join(fen.split()[:4]),evals=[dict(depth=24,knodes=100,pvs=[dict(cp=cp,line=move) for move,cp in moves])])


def test_black_scores_and_q_mask():
    b=chess.Board();b.push_uci('e2e4')
    converted,reason=convert(entry(b.fen(), [('e7e5',-100),('c7c5',100)]))
    assert reason is None
    _,row,_,_=converted
    ids=row['moves'].tolist()
    best=ids.index(move_id(chess.Move.from_uci('e7e5'),b.turn))
    worse=ids.index(move_id(chess.Move.from_uci('c7c5'),b.turn))
    assert row['policy'][best]>row['policy'][worse]
    assert row['q'][best]>0 and row['q'][worse]<0
    assert row['qmask'].sum()==2 and not row['qmask'][row['moves']<0].any()
    assert row['value'][0]>row['value'][2]


def test_castling_and_underpromotion():
    for fen,moves in [('r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1',[('e1h1',100),('e1a1',0)]),
                      ('4k3/P7/8/8/8/8/8/4K3 w - - 0 1',[('a7a8n',0),('a7a8q',100)])]:
        converted,_=convert(entry(fen,moves))
        assert converted[1]['qmask'].sum()==2


def test_one_coherent_search_and_mates():
    e=entry(chess.STARTING_FEN,[('e2e4',100),('d2d4',0)])
    e['evals'].append(dict(depth=50,knodes=1000,pvs=[dict(cp=500,line='g1f3')]))
    converted,_=convert(e)
    assert converted[2]==24  # Deeper single-PV must not supply a fake multi-action target.
    assert score_targets({'mate':-3},False)[1]==1
    assert score_targets({'mate':3},False)[1]==-1
    assert score_targets({'mate':2},True)[0]>score_targets({'mate':10},True)[0]


def test_history_free_is_checkpointed_and_invariant():
    torch.set_num_threads(2)
    model=ChessModel(ModelConfig(64,2,4,32,True)).eval()
    tokens=torch.zeros(2,64,dtype=torch.long);meta=torch.zeros(2,12);meta[1,[5,6,7,10]]=1
    moves=torch.zeros(2,4,dtype=torch.long)
    with torch.no_grad():
        outputs=model(tokens,meta,moves)
    for tensor in outputs:torch.testing.assert_close(tensor[0],tensor[1])


def test_q_supervision_changes_q_output():
    from neurochess.train import losses
    from neurochess.data import record
    row,_=record(chess.Board());row['policy'][0]=1;row['qmask'][0]=True;row['q'][0]=.7
    batch={k:torch.tensor(np.array(row[k])[None]) for k in ['policy','q','qmask','value','value_weight']}
    q=torch.zeros(1,256,requires_grad=True)
    loss,_=losses((torch.zeros(1,256),q,torch.zeros(1,3)),batch,q_weight=.5)
    loss.backward();assert q.grad[0,0]<0 and q.grad[0,1:].abs().sum()==0


def test_replay_resume_matches_uninterrupted(tmp_path):
    import json, os, subprocess, sys
    from neurochess.data import ShardWriter
    primary=tmp_path/'primary';replay=tmp_path/'replay'
    for root in [primary,replay]:
        writer=ShardWriter(root,size=16)
        for i in range(64):
            _,row,_,_=convert(entry(chess.STARTING_FEN,[('e2e4',i*10),('d2d4',-20)]))[0]
            writer.add('train',row)
            if i<8:writer.add('val',row)
        writer.close({'fixture':True})
    ds=ReplayBatches(Batches(primary,'train',4),replay,every=2)
    batches=list(ds)
    assert batches[0]['qmask'].any() and not batches[1]['qmask'].any()
    assert not batches[1]['value_weight'].any()
    command=[sys.executable,'-m','neurochess.train','--data',str(primary),'--replay-data',str(replay),
        '--replay-every','2','--config','configs/tiny.json','--steps','4','--batch-size','4',
        '--accum','2','--workers','2','--save-every','2','--eval-batches','1',
        '--q-warmup','2','--history-free']
    env={**os.environ,'OMP_NUM_THREADS':'2'}
    full=tmp_path/'full';resumed=tmp_path/'resumed'
    subprocess.run(command+['--out',str(full)],check=True,env=env,capture_output=True)
    subprocess.run(command+['--out',str(resumed),'--stop-after','2'],check=True,env=env,capture_output=True)
    subprocess.run(command+['--out',str(resumed),'--resume',str(resumed/'last.pt')],check=True,env=env,capture_output=True)
    a=torch.load(full/'last.pt',weights_only=False);b=torch.load(resumed/'last.pt',weights_only=False)
    for key in a['model']:torch.testing.assert_close(a['model'][key],b['model'][key],rtol=0,atol=0)
    assert (resumed/'best.pt').exists() and b['model_config']['history_free']
