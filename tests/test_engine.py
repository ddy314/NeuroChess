"""End-to-end UCI test; runs only when the engine and a TensorRT plan are available.

    NEUROCHESS_ENGINE=engine/build/neurochess NEUROCHESS_PLAN=models/neurochess-13750-int8.plan pytest
"""
import os
from pathlib import Path

import chess
import chess.engine
import pytest

ENGINE = Path(os.environ.get('NEUROCHESS_ENGINE', 'engine/build/neurochess'))
PLAN = Path(os.environ.get('NEUROCHESS_PLAN', 'models/neurochess-13750-int8.plan'))

pytestmark = pytest.mark.skipif(not (ENGINE.is_file() and PLAN.is_file()),
                                reason='native engine or TensorRT plan not available')


@pytest.fixture(scope='module')
def engine():
    process = chess.engine.SimpleEngine.popen_uci(
        [str(ENGINE), '--engine', str(PLAN), '--uci', '--history-free', '--batch', '16',
         '--q-prior-beta', '2', '--cache', '32768', '--time', '-1', '--simulations', '8192'], timeout=120)
    yield process
    process.quit()


def test_plays_legal_moves(engine):
    board = chess.Board()
    for _ in range(6):
        move = engine.play(board, chess.engine.Limit(nodes=256)).move
        assert move in board.legal_moves
        board.push(move)


def test_finds_mate_in_one(engine):
    board = chess.Board('6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1')
    assert engine.play(board, chess.engine.Limit(nodes=1024)).move == chess.Move.from_uci('d1d8')
