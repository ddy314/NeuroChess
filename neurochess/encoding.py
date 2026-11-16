"""Board encoding shared by the network, the exporter and the native engine.

Positions are encoded from the side to move's point of view: for black, ranks
are flipped (files never are, which keeps castling semantics intact).
"""
import chess
import numpy as np

MAX_MOVES = 256
META_SIZE = 12
PROMOTIONS = [None, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]


def move_id(move: chess.Move, turn: chess.Color) -> int:
    """Map a move to ``(from * 64 + to) * 5 + promotion`` in the side-to-move frame."""
    flip = 0 if turn == chess.WHITE else 56
    return ((move.from_square ^ flip) * 64 + (move.to_square ^ flip)) * 5 + PROMOTIONS.index(move.promotion)


def encode(board: chess.Board):
    """Return ``(tokens[64] uint8, meta[12] float32)``."""
    tokens = np.zeros(64, dtype=np.uint8)
    flip = 0 if board.turn else 56
    for square, piece in board.piece_map().items():
        tokens[square ^ flip] = piece.piece_type + (0 if piece.color == board.turn else 6)
    meta = np.zeros(META_SIZE, dtype=np.float32)
    meta[:4] = [board.has_kingside_castling_rights(board.turn),
                board.has_queenside_castling_rights(board.turn),
                board.has_kingside_castling_rights(not board.turn),
                board.has_queenside_castling_rights(not board.turn)]
    meta[4] = (board.ep_square ^ flip) / 63 if board.ep_square is not None else -1
    meta[5] = min(board.halfmove_clock, 150) / 150
    meta[6:9] = [board.is_repetition(2), board.is_repetition(3), board.is_check()]
    meta[9] = board.turn == chess.WHITE
    meta[10] = min(board.fullmove_number, 300) / 300
    meta[11] = board.ep_square is not None
    return tokens, meta


def legal_ids(board: chess.Board):
    """Return the legal moves and their ids (int16)."""
    moves = list(board.legal_moves)
    if len(moves) > MAX_MOVES:
        raise ValueError('Legal move capacity exceeded')
    return moves, np.asarray([move_id(m, board.turn) for m in moves], dtype=np.int16)


def encode_inputs(board: chess.Board):
    """Return network inputs ``(tokens, meta, moves[256])`` plus the legal move list.

    ``moves`` is padded with -1 after the legal move ids.
    """
    tokens, meta = encode(board)
    legal, ids = legal_ids(board)
    moves = np.full(MAX_MOVES, -1, dtype=np.int32)
    moves[:len(ids)] = ids
    return tokens, meta, moves, legal
