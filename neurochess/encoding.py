import hashlib
import chess
import numpy as np

MAX_MOVES = 256
META_SIZE = 12
PROMOTIONS = [None, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]


def move_id(move, turn):
    flip = 0 if turn == chess.WHITE else 56
    return ((move.from_square ^ flip) * 64 + (move.to_square ^ flip)) * 5 + PROMOTIONS.index(move.promotion)


def encode(board):
    """Side-to-move frame: black flips ranks, never files (castling semantics)."""
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


def position_key(board):
    # Ignore clocks/history so identical boards cannot cross train/validation.
    return hashlib.blake2b(' '.join(board.fen().split()[:4]).encode(), digest_size=16).digest()


def split_for(board):
    return 'val' if int.from_bytes(position_key(board)[:8], 'little') % 100 == 0 else 'train'


def legal_ids(board):
    moves = list(board.legal_moves)
    if len(moves) > MAX_MOVES:
        raise ValueError('Legal move capacity exceeded')
    return moves, np.asarray([move_id(m, board.turn) for m in moves], dtype=np.int16)
