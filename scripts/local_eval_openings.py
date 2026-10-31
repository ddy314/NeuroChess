"""Select a reproducible, ECO-stratified opening sample from a separate PGN archive."""
import argparse
from collections import defaultdict
import hashlib
import io
import json
from pathlib import Path
import random

import chess
import chess.pgn
import zstandard


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--pairs', type=int, default=128)
    p.add_argument('--seed', type=int, default=20261030)
    a = p.parse_args()
    rng = random.Random(a.seed)
    buckets = defaultdict(list)
    seen = set()
    monitor = {tuple(chess.Board(f).fen().split()[:4]) for f in
               Path('configs/monitor-openings.fen').read_text().splitlines() if f.strip()}
    scanned = 0
    with open(a.source, 'rb') as raw, zstandard.ZstdDecompressor().stream_reader(raw) as stream:
        text = io.TextIOWrapper(stream)
        while scanned < 20000:
            game = chess.pgn.read_game(text)
            if game is None:
                break
            scanned += 1
            h = game.headers
            try:
                rating = min(int(h.get('WhiteElo', 0)), int(h.get('BlackElo', 0)))
                base = int(h.get('TimeControl', '0').split('+')[0])
            except ValueError:
                continue
            if game.errors or rating < 1800 or base < 180 or h.get('Result') not in ('1-0', '0-1', '1/2-1/2'):
                continue
            eco = h.get('ECO', '?')
            if eco == '?' or len(buckets[eco]) >= 8:
                continue
            plies = rng.choice([8, 10, 12])
            moves = list(game.mainline_moves())
            if len(moves) < plies + 4:
                continue
            board = game.board()
            if board != chess.Board():
                continue
            for move in moves[:plies]:
                board.push(move)
            key = tuple(board.fen().split()[:4])
            if key in seen or key in monitor or board.is_game_over(claim_draw=True):
                continue
            seen.add(key)
            buckets[eco].append(dict(fen=board.fen(), eco=eco, plies=plies,
                                     site=h.get('Site'), opening=h.get('Opening'), min_player_elo=rating))
    for values in buckets.values():
        rng.shuffle(values)
    codes = sorted(buckets)
    rng.shuffle(codes)
    selected = []
    for depth in range(8):
        for eco in codes:
            if len(buckets[eco]) > depth:
                selected.append(buckets[eco][depth])
                if len(selected) == a.pairs:
                    break
        if len(selected) == a.pairs:
            break
    if len(selected) < a.pairs:
        raise ValueError(f'Only {len(selected)} unique openings available')
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'openings.fen').write_text(''.join(row['fen'] + '\n' for row in selected))
    report = dict(seed=a.seed, pairs=a.pairs, scanned_games=scanned, source=a.source,
                  source_sha256=hashlib.file_digest(open(a.source, 'rb'), 'sha256').hexdigest(),
                  distinct_eco=len({r['eco'] for r in selected}), openings=selected,
                  scope='Different archive and larger opening set than cloud monitor; opening positions may overlap training data.')
    (out / 'openings.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != 'openings'}))


if __name__ == '__main__':
    main()
