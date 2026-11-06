import chess
import pytest
from scripts.targeted_mistakes import classify,stable_labels


def test_equivalent_moves_and_side_to_move():
    p=[dict(line='e2e4',cp=10),dict(line='d2d4',cp=15)]
    assert not classify(p,'e2e4',0,chess.WHITE)['policy_error']
    assert classify([dict(line='a7a6',cp=-150),dict(line='b7b6',cp=10)],'b7b6',0,chess.BLACK)['policy_error']


def test_winning_mate_distance_is_not_blunder():
    p=[dict(line='e2e4',mate=2),dict(line='d2d4',mate=5)]
    assert not classify(p,'d2d4',1,True)['policy_error']
    assert classify([dict(line='e2e4',mate=2),dict(line='d2d4',cp=0)],'d2d4',0,True)['policy_error']


def test_candidate_union_and_teacher_stability():
    with pytest.raises(ValueError):classify([dict(line='e2e4',cp=0)],'d2d4',0,True)
    assert not stable_labels([dict(line='e2e4',cp=50)],[dict(line='e2e4',cp=100)],True)
    assert not stable_labels([dict(line='e2e4',mate=2)],[dict(line='e2e4',cp=500)],True)
    assert stable_labels([dict(line='e2e4',mate=2)],[dict(line='e2e4',mate=4)],True)


def test_cloud_teacher_checks_union_and_independently_refines(monkeypatch):
    import chess.engine
    from scripts import targeted_distill_cloud as cloud
    from neurochess.encoding import position_key
    class Engine:
        def __init__(self):self.budgets=[]
        def analyse(self,board,limit,multipv,game,root_moves=None):
            self.budgets.append(limit.nodes)
            moves=root_moves or [chess.Move.from_uci('e2e4'),chess.Move.from_uci('d2d4')]
            return [dict(pv=[m],score=chess.engine.PovScore(chess.engine.Cp(400 if m.uci()=='e2e4' else 0),board.turn),depth=15,nodes=limit.nodes) for m in moves]
    engine=Engine();monkeypatch.setattr(cloud,'ENGINE',engine)
    board=chess.Board();job=dict(initial_fen=board.fen(),history=[],key=position_key(board).hex())
    result,row=cloud.label((job,dict(move='d2d4',top_moves=['d2d4'],value=0)))
    assert result['repair'] and result['stable'] and result['decision']['policy_error']
    assert engine.budgets==[10000,30000,100000,300000]
    assert row['value_weight']==1 and row['qmask'].sum()==2
