import chess
import numpy as np
import torch

from neurochess import ModelConfig, NeuroChessModel, QHeadConfig, load_model
from neurochess.encoding import MAX_MOVES, encode_inputs, move_id


def tiny_model():
    torch.manual_seed(0)
    return NeuroChessModel(ModelConfig(width=32, layers=2, heads=4, head_width=16, history_free=True),
                           QHeadConfig(width=8, bins=8)).eval()


def test_encoding_is_side_to_move_relative():
    white = chess.Board()
    black = chess.Board()
    black.push_uci('e2e4')
    black = chess.Board(black.fen().replace(' b ', ' w ', 1)).mirror()  # black to move, mirrored
    t_white, *_ = encode_inputs(white)
    t_black, *_ = encode_inputs(black)
    assert t_white[0] == chess.ROOK and t_white[63] == chess.ROOK + 6
    assert t_black[0] == chess.ROOK  # own rook on a1 in the side-to-move frame


def test_move_ids_are_unique_and_padded():
    board = chess.Board('r3k2r/pPppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1')
    tokens, meta, moves, legal = encode_inputs(board)
    ids = moves[:len(legal)]
    assert tokens.shape == (64,) and meta.shape == (12,) and moves.shape == (MAX_MOVES,)
    assert len(set(ids.tolist())) == len(legal)
    assert (moves[len(legal):] == -1).all()
    assert ids.tolist() == [move_id(m, board.turn) for m in legal]


def test_forward_shapes_and_masking():
    model = tiny_model()
    tokens, meta, moves, legal = encode_inputs(chess.Board())
    batch = [torch.as_tensor(np.stack([x, x])) for x in (tokens, meta, moves)]
    policy, q, wdl = model(*batch)
    assert policy.shape == q.shape == (2, MAX_MOVES) and wdl.shape == (2, 3)
    n = len(legal)
    assert (policy[:, n:] <= -1e8).all() and (q[:, n:] == 0).all()
    assert q[:, :n].abs().max() <= 1


def test_checkpoint_round_trip(tmp_path):
    model = tiny_model()
    path = tmp_path / 'tiny.pt'
    torch.save(model.checkpoint(), path)
    loaded = load_model(path)
    inputs = [torch.as_tensor(x[None]) for x in encode_inputs(chess.Board())[:3]]
    for a, b in zip(model(*inputs), loaded(*inputs)):
        assert torch.equal(a, b)
