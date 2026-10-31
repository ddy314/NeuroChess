"""Sample real PGN positions for CPU annotation (bounded, reproducible)."""
import argparse
import io
from pathlib import Path
import random
import chess.pgn
import zstandard
from .encoding import position_key


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pgn', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--positions', type=int, default=100000)
    p.add_argument('--min-elo', type=int, default=2000)
    p.add_argument('--seed', type=int, default=42)
    a = p.parse_args()
    rng = random.Random(a.seed)
    seen = set()
    raw = open(a.pgn, 'rb')
    source = zstandard.ZstdDecompressor().stream_reader(raw) if a.pgn.endswith('.zst') else raw
    with io.TextIOWrapper(source) as stream, open(a.out, 'x') as output:
        while len(seen) < a.positions and (game := chess.pgn.read_game(stream)) is not None:
            try:
                if min(int(game.headers.get(k, 0)) for k in ['WhiteElo', 'BlackElo']) < a.min_elo:
                    continue
            except ValueError:
                continue
            if game.errors or game.headers.get('Variant', 'Standard') != 'Standard':
                continue
            board = game.board()
            candidates = []
            for move in game.mainline_moves():
                board.push(move)
                if board.ply() >= 12 and not board.is_game_over():
                    candidates.append(board.fen())
            rng.shuffle(candidates)
            for fen in candidates[:8]:
                key = position_key(chess.Board(fen))
                if key not in seen:
                    output.write(fen + '\n'); seen.add(key)
                if len(seen) == a.positions:
                    break
    print(f'Wrote {len(seen)} positions to {a.out}')


if __name__ == '__main__':
    main()
