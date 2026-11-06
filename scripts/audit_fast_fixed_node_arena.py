"""Replay fixed-node arena PGNs and independently reconcile results and counters."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import chess
import chess.pgn
import numpy as np


def digest(path):
    with open(path, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def audit(folder):
    folder = Path(folder)
    plan = json.loads((folder / 'plan.json').read_text())
    times, actual_inputs, simulations = [], [], []
    rows = []
    identities = set()
    for filename in sorted((folder / 'games').glob('*.json')):
        row = json.loads(filename.read_text())
        with filename.with_suffix('.pgn').open() as f:
            game = chess.pgn.read_game(f)
        assert game is not None and not game.errors, filename
        board = game.board()
        assert board.fen(en_passant='fen') == row['fen'] == plan['openings'][row['opening']], filename
        assert game.headers['ModelSHA256'] == plan['checkpoint_sha256'], filename
        assert int(game.headers['SFNodes']) == row['nodes'], filename
        assert float(game.headers['SearchSeconds']) == plan['search']['options']['time_limit'], filename
        color = row['contender_color'] == 'white'
        assert 'NeuroChess' in game.headers['White' if color else 'Black'], filename
        count, opponent_count, plies = 0, 0, 0
        for move in game.mainline_moves():
            assert move in board.legal_moves, filename
            if board.turn == color:
                count += 1
            else:
                opponent_count += 1
            board.push(move)
            plies += 1
        outcome = board.outcome(claim_draw=True)
        score = None if outcome is None else .5 if outcome.winner is None else float(outcome.winner == color)
        assert score == row['score'], filename
        assert row['result'] == ('*' if outcome is None else outcome.result()) == game.headers['Result'], filename
        assert plies == row['plies'], filename
        assert len(row['decisions']) == count == row['stats']['choices'], filename
        assert len(row['opponent_decisions']) == opponent_count, filename
        assert sum(d['root']['new_simulations'] for d in row['decisions']) == row['stats']['completed_simulations'], filename
        assert sum(d['large_positions'] for d in row['decisions']) == row['stats']['inference_cache']['forward_positions'], filename
        identity = row['nodes'], row['opening'], row['contender_color']
        assert identity not in identities, filename
        identities.add(identity)
        times.extend(d['seconds'] for d in row['decisions'])
        actual_inputs.extend(d['large_positions'] for d in row['decisions'])
        simulations.extend(d['root']['new_simulations'] for d in row['decisions'])
        rows.append(row)
    summary = {}
    for nodes in plan['nodes']:
        selected = [r for r in rows if r['nodes'] == nodes]
        completed = [r for r in selected if r['score'] is not None]
        pair_count = sum(all(any(r['opening'] == opening and r['contender_color'] == color and r['score'] is not None
                                      for r in selected) for color in ['white', 'black'])
                         for opening in range(plan['pairs']))
        summary[str(nodes)] = dict(games=len(selected), completed=len(completed),
            wins=sum(r['score'] == 1 for r in completed), draws=sum(r['score'] == .5 for r in completed),
            losses=sum(r['score'] == 0 for r in completed), truncated=len(selected) - len(completed),
            completed_pairs=pair_count, score=sum(r['score'] for r in completed) / len(completed) if completed else None)
    expected = 2 * plan['pairs'] * len(plan['nodes'])
    return dict(verified_games=len(rows), scheduled=expected,
        complete=len(rows) == expected and all(r['score'] is not None for r in rows), summary=summary,
        latency=dict(model_moves=len(times), mean=float(np.mean(times)), median=float(np.median(times)),
            p95=float(np.quantile(times, .95)), maximum=max(times), over_budget=sum(t > plan['search']['options']['time_limit'] for t in times)) if times else {},
        inference=dict(total_network_inputs=sum(actual_inputs), median_network_inputs=float(np.median(actual_inputs)),
            median_simulations=float(np.median(simulations))) if actual_inputs else {},
        note='Legal PGN replay, claimable outcomes, colors, budgets, model fingerprint and actual inference/simulation counters reconciled. Counts are from model perspective.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arena', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--check-runtime', action='store_true')
    args = parser.parse_args()
    if args.check_runtime:
        plan = json.loads((Path(args.arena) / 'plan.json').read_text())
        assert digest(plan['checkpoint']) == plan['checkpoint_sha256']
        for filename, sha in plan['source_sha256'].items():
            assert digest(filename) == sha, filename
    report = audit(args.arena)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
