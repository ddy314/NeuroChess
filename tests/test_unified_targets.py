import chess
import chess.engine
import numpy as np
import pytest
from neurochess.import_evals import convert
from neurochess import distill


@pytest.mark.parametrize('black',[False,True])
def test_live_teacher_matches_cache_targets_independent_of_game_ply(monkeypatch,black):
    board=chess.Board()
    if black:board.push_uci('e2e4')
    board.fullmove_number=90;board.halfmove_clock=3
    moves=list(board.legal_moves)[:3];scores=[-130,45,220]
    pvs=[dict(cp=cp,line=m.uci()) for cp,m in zip(scores,moves)]
    entry=dict(fen=' '.join(board.fen().split()[:4]),evals=[dict(depth=30,knodes=200,pvs=pvs)])
    cached=convert(entry)[0][1]
    class Engine:
        def analyse(self,*args,**kwargs):
            return [dict(pv=[move],score=chess.engine.PovScore(chess.engine.Cp(cp),chess.WHITE)) for cp,move in zip(scores,moves)]
    monkeypatch.setattr(distill,'_ENGINE',Engine())
    live=distill.annotate((board.fen(),200000,3,100))[1]
    for field in ['policy','q','qmask','value','value_weight']:
        np.testing.assert_array_equal(cached[field],live[field])
