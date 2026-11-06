import numpy as np
import chess,chess.engine
from scripts.analyze_joint_checkpoint import paired_interval,row_metrics
from neurochess.data import DTYPE
from neurochess.encoding import position_key
from scripts import joint_diagnostic_teacher as teacher


def test_group_bootstrap_does_not_count_repeated_game_rows_as_new_games():
    delta=np.array([1.,2.,-1.,0.]);groups=np.array(['a','a','b','b'])
    a=paired_interval(delta,groups);b=paired_interval(np.repeat(delta,10),np.repeat(groups,10))
    assert a==b and a['clusters']==2


def test_masked_q_metric_cannot_claim_quality_on_unlabelled_actions():
    a=np.zeros(2,dtype=DTYPE);a['moves']=-1;a['moves'][:,:4]=[1,2,3,4];a['qmask'][:,:2]=True;a['policy'][:,0]=1;a['value'][:]=[.2,.5,.3]
    raw=np.zeros((2,515),np.float32);raw[:,:256]=-1e9;raw[:,:4]=[1.,.5,0.,-.5];raw[:,512:]=[.2,.5,.3]
    first=row_metrics(raw,a)[0];raw[:,258:260]=1
    second=row_metrics(raw,a)[0]
    np.testing.assert_array_equal(first['q_mae'],second['q_mae'])
    np.testing.assert_allclose(second['q_support_label_coverage'],.5)


def test_deep_teacher_includes_student_choices_and_refines_independently(monkeypatch):
    class Engine:
        def __init__(self):self.calls=[]
        def analyse(self,b,limit,multipv,game,root_moves=None):
            self.calls.append((limit.nodes,root_moves));moves=root_moves or [chess.Move.from_uci('e2e4'),chess.Move.from_uci('d2d4')]
            return [dict(pv=[m],score=chess.engine.PovScore(chess.engine.Cp(400 if limit.nodes>=1000000 else 100),b.turn),depth=20,nodes=limit.nodes) for m in moves]
    engine=Engine();monkeypatch.setattr(teacher,'ENGINE',engine);b=chess.Board()
    r=teacher.adjudicate(dict(job=dict(initial_fen=b.fen(),history=[],key=position_key(b).hex()),moves=['g1f3']))
    assert [n for n,_ in engine.calls]==[1000000,100000,1000000,3000000]
    assert r['stable'] and r['refined']
    assert all(chess.Move.from_uci('g1f3') in moves for _,moves in engine.calls[1:])
