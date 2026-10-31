"""End-to-end fixture test; output is deliberately not a playable trained model."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import chess
from neurochess.data import record, ShardWriter


def run(*args):
    subprocess.run([sys.executable, *args], check=True, env={**os.environ, 'OMP_NUM_THREADS': '2'})


with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    writer = ShardWriter(root / 'data', size=64)
    for split in ['train', 'val']:
        b = chess.Board()
        for ply in range(100):
            row, moves = record(b)
            move = moves[ply % len(moves)]
            row['policy'][moves.index(move)] = 1
            row['value'][1] = 1; row['value_weight'] = 1
            writer.add(split, row)
            b.push(move)
            if b.is_game_over():
                b = chess.Board()
    writer.close({'kind': 'fixture-only'})
    command = ['-m', 'neurochess.train', '--data', str(root / 'data'), '--out', str(root / 'run'),
        '--config', 'configs/tiny.json', '--steps', '2', '--batch-size', '8', '--accum', '2',
        '--workers', '0', '--save-every', '1', '--warmup', '1', '--eval-batches', '2', '--device', 'cpu']
    run(*command, '--stop-after', '1')
    run(*command, '--resume', str(root / 'run/last.pt'))
    import torch
    uninterrupted = command.copy()
    uninterrupted[uninterrupted.index(str(root / 'run'))] = str(root / 'full')
    run(*uninterrupted)
    resumed = torch.load(root / 'run/last.pt', weights_only=False)
    full = torch.load(root / 'full/last.pt', weights_only=False)
    for name in full['model']:
        torch.testing.assert_close(resumed['model'][name], full['model'][name], rtol=0, atol=0)
    from neurochess.play import Player
    for mode in ['policy', 'q', 'oneply']:
        board = chess.Board()
        move = Player(root / 'run/last.pt', mode=mode).choose(board)
        assert move in board.legal_moves
    engine = Path('local/stockfish/stockfish/stockfish-ubuntu-x86-64-avx2')
    if engine.exists():
        import json
        (root / 'openings.fen').write_text(chess.STARTING_FEN + '\n')
        run('-m', 'neurochess.play', '--checkpoint', str(root / 'run/last.pt'),
            '--stockfish', str(engine), '--openings', str(root / 'openings.fen'),
            '--pairs', '1', '--nodes', '100', '--max-plies', '12', '--out', str(root / 'match'))
        report = json.loads((root / 'match/result.json').read_text())
        assert report['completed'] + report['truncated'] == 2
    print('INTEGRATION_OK')
