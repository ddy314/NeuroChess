"""Independently replay completed arena PGNs and reconcile inference counters."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import chess
import chess.pgn
import numpy as np


def audit(folder):
    folder = Path(folder)
    rows = []
    times = {}
    for filename in sorted(folder.glob('[0-9][0-9][0-9]-[01].json')):
        row = json.loads(filename.read_text())
        with filename.with_suffix('.pgn').open() as f:
            game = chess.pgn.read_game(f)
        assert game is not None and not game.errors, filename
        board = game.board()
        assert board.fen(en_passant='fen') == row['fen'], filename
        color = row['contender_color'] == 'white'
        names = list(row['decisions'])
        contender = next(n for n in names if n != 'baseline')
        counts = dict.fromkeys(names, 0)
        for move in game.mainline_moves():
            assert move in board.legal_moves, filename
            name = contender if board.turn == color else 'baseline'
            counts[name] += 1
            board.push(move)
        outcome = board.outcome(claim_draw=True)
        score = None if outcome is None else .5 if outcome.winner is None else float(outcome.winner == color)
        assert row['score'] == score, filename
        assert row['result'] == ('*' if outcome is None else outcome.result()), filename
        assert row['plies'] == sum(counts.values()), filename
        for name, decisions in row['decisions'].items():
            assert len(decisions) == counts[name], (filename, name)
            stats = row['stats'][name]
            assert stats['choices'] == len(decisions), (filename, name)
            assert sum(d['root']['new_simulations'] for d in decisions) == stats['completed_simulations'], (filename, name)
            assert sum(d['large_positions'] for d in decisions) == stats['inference_cache']['forward_positions'], (filename, name)
            assert all(d['seconds'] >= 0 and d['large_positions'] >= 0 for d in decisions)
            times.setdefault(name, []).extend(d['seconds'] for d in decisions)
        rows.append(dict(file=filename.name, plies=row['plies'], score=score))
    return dict(verified_games=len(rows), games=rows,
        latency={name: dict(moves=len(t), median=float(np.median(t)),
                    p95=float(np.quantile(t, .95)), maximum=max(t)) for name, t in times.items()},
        note='Replayed legal moves and claimable outcomes; reconciled choice, simulation and actual network-input counters. Incomplete arena plans remain incomplete.')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--arena', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    report = audit(a.arena)
    Path(a.out).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
