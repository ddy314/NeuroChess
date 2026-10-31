import ctypes as C
import random

import chess
import numpy as np
import pytest

from neurochess.encoding import encode, legal_ids
from neurochess.mcts import terminal_value
from neurochess.native_mcts import NativeMCTSPlayer, history, load_library


class Network:
    device = 'cpu'
    def __init__(self, favorite=None):
        self.favorite = favorite
        self.batches = []
    def evaluate_encoded(self, tokens, meta, ids):
        self.batches.append(len(tokens))
        logits = np.zeros(ids.shape, np.float32)
        if self.favorite is not None:
            logits[ids == self.favorite] = 10
        return logits, np.zeros(len(tokens), np.float32)


def probe(board):
    lib = load_library()
    tokens, meta, ids = np.empty(64, np.uint8), np.empty(12, np.float32), np.empty(256, np.int16)
    n, v = C.c_int(), C.c_int()
    rc = lib.nc_probe(*history(board), tokens.ctypes.data, meta.ctypes.data, ids.ctypes.data, C.byref(n), C.byref(v))
    assert rc == 0, lib.nc_error()
    return tokens, meta, ids[:n.value], None if v.value == 2 else float(v.value)


def compare(board):
    tokens, meta, ids, value = probe(board)
    expected_tokens, expected_meta = encode(board)
    np.testing.assert_array_equal(tokens, expected_tokens)
    np.testing.assert_array_equal(meta, expected_meta)
    assert sorted(ids) == sorted(legal_ids(board)[1])
    assert value == terminal_value(board), board.fen()


def test_native_differential_random_histories():
    rng = random.Random(114)
    for _ in range(15):
        board = chess.Board()
        for _ in range(110):
            compare(board)
            if board.is_game_over(claim_draw=True):
                break
            board.push(rng.choice(list(board.legal_moves)))


@pytest.mark.parametrize('fen', [
    'r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1',
    '7k/P7/8/8/8/8/8/7K w - - 0 1',
    '7k/8/8/3pP3/8/8/8/7K w - d6 0 1',
    '7k/8/8/r4pPK/8/8/8/8 w - f6 0 1',
    '7k/8/8/8/8/8/R7/7K w - - 99 50',
    '7k/8/8/8/8/8/R7/7K w - - 100 50',
    '7k/8/8/8/8/B7/8/4B2K w - - 0 1',
    '7k/5Q2/5K2/8/8/8/8/8 b - - 0 1',
    '7k/6Q1/5K2/8/8/8/8/8 b - - 0 1',
])
def test_native_special_rules(fen):
    compare(chess.Board(fen))


def test_native_claim_on_next_move_and_raw_ep():
    board = chess.Board()
    for move in ['g1f3','g8f6','f3g1','f6g8','g1f3','g8f6','f3g1']:
        board.push_uci(move); compare(board)
    assert board.can_claim_threefold_repetition()
    with pytest.raises(RuntimeError, match='Terminal root'):
        NativeMCTSPlayer(Network()).choose(board)
    board = chess.Board(); board.push_uci('e2e4'); compare(board)


@pytest.mark.parametrize('batch', [1, 8, 16])
def test_native_budget_pruning_and_virtual_loss(batch):
    network = Network()
    player = NativeMCTSPlayer(network, simulations=64, leaf_batch_size=batch, initial_width=2,
                               widening=.25, root_keep=2, prune_after=.5)
    board = chess.Board(); before = board.fen()
    assert player.choose(board) in board.legal_moves
    assert player.search_stats['completed_simulations'] == 64
    assert player.search_stats['pruned_root_edges'] > 0
    assert player.search_stats['max_depth'] > 1
    assert player.search_stats['leaf_evaluations'] == 65
    assert board.fen() == before
    if batch > 1:
        assert max(network.batches) > 1
        assert len(network.batches) < 64


@pytest.mark.parametrize('mirror', [False, True])
def test_native_low_prior_mate_and_fools_mate(mirror):
    board = chess.Board('7k/8/5KQ1/8/8/8/8/8 w - - 0 1')
    if mirror:
        board = board.mirror()
    player = NativeMCTSPlayer(Network(), simulations=32)
    board.push(player.choose(board)); assert board.is_checkmate()
    board = chess.Board(); board.push_uci('f2f3'); board.push_uci('e7e5')
    from neurochess.encoding import move_id
    player = NativeMCTSPlayer(Network(move_id(chess.Move.from_uci('g2g4'), board.turn)), simulations=48)
    assert player.choose(board) != chess.Move.from_uci('g2g4')
    assert player.search_stats['proven_loss_skips'] > 0


def test_native_reuse_history_identity_and_new_budget():
    board = chess.Board(); player = NativeMCTSPlayer(Network(), simulations=128, leaf_batch_size=8)
    move = player.choose(board); board.push(move)
    # Search the next ply to ensure it is in the retained tree.
    player.choose(board)
    assert player.search_stats['tree_reuses'] == 1
    assert player.decision_samples[-1]['retained_visits'] > 0
    assert player.decision_samples[-1]['new_simulations'] == 128
    assert player.search_stats['completed_simulations'] == 256
    # Equal board/FEN with missing history must not reuse history-dependent values.
    player.choose(chess.Board(board.fen()))
    assert player.search_stats['tree_reuses'] == 1
    assert player.decision_samples[-1]['retained_visits'] == 0


def test_native_batch_slots_and_two_ply_reuse():
    boards = [chess.Board(), chess.Board()]
    player = NativeMCTSPlayer(Network(), simulations=128, leaf_batch_size=16)
    moves = player.choose_many(boards)
    for b,m in zip(boards,moves): b.push(m)
    moves = player.choose_many(boards)
    assert player.search_stats['tree_reuses'] == 2
    for b,m in zip(boards,moves): b.push(m)
    player.choose_many(boards)
    assert player.search_stats['tree_reuses'] == 4
    assert player.search_stats['completed_simulations'] == 6*128


def test_disable_reuse_keeps_cumulative_stats_and_fresh_budget():
    player=NativeMCTSPlayer(Network(),simulations=31,leaf_batch_size=4,tree_reuse=False)
    b=chess.Board();player.choose(b);player.choose(b)
    assert player.search_stats['tree_reuses']==0
    assert player.search_stats['choices']==2
    assert player.search_stats['completed_simulations']==62
    assert player.decision_samples[-1]['retained_visits']==0


def test_time_budget_stops_at_batch_boundary_and_reuse_remains_valid():
    import time
    class Slow(Network):
        def evaluate_encoded(self,tokens,meta,ids):
            time.sleep(.005)
            return super().evaluate_encoded(tokens,meta,ids)
    p=NativeMCTSPlayer(Slow(),simulations=4096,leaf_batch_size=4,time_limit=.045,time_margin=.005)
    b=chess.Board();assert p.choose(b) in b.legal_moves
    assert p.last_result['time_stopped']
    assert p.last_result['new_simulations']<4096
    # A second call must not encounter pending leaves or stranded virtual loss.
    assert p.choose(b) in b.legal_moves
    assert p.search_stats['tree_reuses']==1 and p.search_stats['time_stops']==2
    with pytest.raises(ValueError,match='one game slot'):p.choose_many([b,b])


def test_tiny_time_budget_still_establishes_root_and_returns_legal_move():
    p=NativeMCTSPlayer(Network(),simulations=128,leaf_batch_size=8,time_limit=.000001,time_margin=0)
    b=chess.Board();assert p.choose(b) in b.legal_moves
    assert p.last_result['time_stopped']
    assert p.last_result['new_simulations']==0
    for value in [0,-1,float('nan'),float('inf')]:
        with pytest.raises(ValueError):NativeMCTSPlayer(Network(),time_limit=value)


def test_cli_search_config_loads_reproducible_native_options(tmp_path):
    import argparse,json
    from neurochess.mcts import add_mcts_arguments,wrap_mcts
    parser=argparse.ArgumentParser();add_mcts_arguments(parser)
    path=tmp_path/'search.json';path.write_text(json.dumps(dict(simulations=4096,leaf_batch_size=64,time_limit=2.,inference_cache_size=1024)))
    p=wrap_mcts(Network(),parser.parse_args(['--search-config',str(path)]))
    assert p.config['simulations_per_move']==4096
    assert p.config['time_limit_seconds']==2. and p.config['inference_cache_size']==1024
    with pytest.raises(ValueError,match='cpp'):wrap_mcts(Network(),parser.parse_args(['--mcts-backend','python','--search-seconds','2']))


def test_pending_loss_guard_prevents_terminal_backups_starving_inference():
    b=chess.Board('2k4r/3bbp1r/1qp1p3/p3Pn2/3P4/PpR2QB1/1P2N1P1/1B1R2K1 w - - 1 29')
    from neurochess.encoding import move_id
    class Pessimistic(Network):
        def evaluate_encoded(self,tokens,meta,ids):
            policy,value=super().evaluate_encoded(tokens,meta,ids);value[:]=-1
            return policy,value
    favorite=move_id(chess.Move.from_uci('g3f2'),b.turn)
    results=[]
    for guard in [False,True]:
        p=NativeMCTSPlayer(Pessimistic(favorite),simulations=512,leaf_batch_size=64,pending_loss_guard=guard)
        assert p.choose(b) in b.legal_moves;results.append(dict(p.search_stats));p.close()
    old,new=results
    assert old['terminal_leaves']>400
    assert new['leaf_evaluations']>old['leaf_evaluations']
    assert new['terminal_leaves']<old['terminal_leaves']
    assert new['completed_simulations']==512
