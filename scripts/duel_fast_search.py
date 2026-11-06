"""Resumable, color-paired candidate games against the immutable two-second PUCT."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if importlib.util.find_spec('chess') is None:
    sys.path.append(str(next((ROOT / '.venv/lib').glob('python*/site-packages'))))
import chess
import torch
from neurochess.encoding import encode, legal_ids
from neurochess.multihead import MultiheadPlayer
from neurochess.native_mcts import NativeMCTSPlayer
from neurochess.train import load_model
from scripts.search_duel import setup_game, summarize
import numpy as np


def digest(path):
    with open(path, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    p = argparse.ArgumentParser()
    for key in ['suite', 'candidate', 'openings', 'out']:
        p.add_argument('--' + key, required=True)
    p.add_argument('--pairs', type=int, default=8)
    p.add_argument('--max-plies', type=int, default=600)
    p.add_argument('--resume', action='store_true')
    a = p.parse_args()
    root = ROOT / 'output/puct-10x-20261031'
    manifest = json.loads((root / 'frozen-baseline/manifest.json').read_text())
    for filename, sha in manifest['source_sha256'].items():
        if digest(filename) != sha:
            raise ValueError('Frozen baseline changed: ' + filename)
    checkpoint = manifest['checkpoint']
    if digest(checkpoint) != manifest['checkpoint_sha256']:
        raise ValueError('Checkpoint changed')
    variants = {v['name']: v for v in json.loads(Path(a.suite).read_text())}
    candidate = variants[a.candidate]
    if candidate['backend'] not in ['tensorrt-int8', 'tensorrt-int8graph']:
        raise ValueError('This runner requires an explicit TensorRT candidate')
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=a.resume)
    fens = Path(a.openings).read_text().splitlines()[:a.pairs]
    if len(fens) != a.pairs or a.pairs < 1:
        raise ValueError('Insufficient paired openings')
    for fen in fens:
        b = chess.Board(fen)
        if not b.is_valid() or b.is_game_over(claim_draw=True):
            raise ValueError('Invalid opening')
    sources = [Path(__file__), Path('neurochess/tensorrt_inference.py'),
               Path('neurochess/tensorrt_graph.py'), Path('neurochess/fused_heads.py'),
               Path('neurochess/cuda_inference.py'), Path('neurochess/pipeline_mcts.py'),
               Path('neurochess/native/mcts_pipeline.cpp'), Path('neurochess/native/libmcts_pipeline.so')]
    plan = dict(candidate=candidate, baseline=variants['baseline'], baseline_id='puct-2s-v1',
                openings=fens, checkpoint_sha256=manifest['checkpoint_sha256'],
                engine_sha256=digest(root / 'tensorrt-int8/model.plan'),
                source_sha256={str(s): digest(s) for s in sources}, max_plies=a.max_plies,
                rules='Swapped colors; fresh players per game; exact board history and claimable draws; unfinished games excluded.')
    if a.resume and json.loads((out / 'plan.json').read_text()) != plan:
        raise ValueError('Resume requires identical plan and code')
    (out / 'plan.json').write_text(json.dumps(plan, indent=2))
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('high')
    setup = time.perf_counter()
    model, state = load_model(checkpoint, 'cuda')
    del state
    baseline_net = MultiheadPlayer(model)
    sys.path.insert(0, str(root / 'trt-deps'))
    from neurochess.tensorrt_inference import TensorRTModel
    quant = TensorRTModel(root / 'tensorrt-int8/model.plan', model.config)
    if candidate['backend'] == 'tensorrt-int8graph':
        from neurochess.tensorrt_graph import TensorRTGraphMultiheadPlayer
        candidate_net = TensorRTGraphMultiheadPlayer(quant, batch_sizes=range(2, candidate['options']['leaf_batch_size'] + 1))
    else:
        candidate_net = MultiheadPlayer(quant)
    b = chess.Board(fens[0])
    t, m = encode(b)
    for name, net, variant in [('baseline', baseline_net, variants['baseline']),
                               (a.candidate, candidate_net, candidate)]:
        for n in range(1, variant['options']['leaf_batch_size'] + 1):
            ids = np.full((n, 256), -1, np.int16)
            li = legal_ids(b)[1]
            ids[:, :len(li)] = li
            net.evaluate_search_heads(np.tile(t, (n, 1)), np.tile(m, (n, 1)), ids)
    setup = time.perf_counter() - setup
    nets = {'baseline': baseline_net, a.candidate: candidate_net}
    rows = []
    start = time.perf_counter()
    for opening, fen in enumerate(fens):
        for color in [chess.WHITE, chess.BLACK]:
            path = out / f'{opening:03d}-{int(color)}.json'
            if path.exists():
                rows.append(json.loads(path.read_text()))
                continue
            from neurochess.pipeline_mcts import PipelineMCTSPlayer
            cls = PipelineMCTSPlayer if candidate.get('engine') == 'pipeline' else NativeMCTSPlayer
            players = {'baseline': NativeMCTSPlayer(baseline_net, **variants['baseline']['options']),
                       a.candidate: cls(candidate_net, **candidate['options'])}
            board = chess.Board(fen)
            game = setup_game(board, a.candidate if color else 'baseline', 'baseline' if color else a.candidate)
            node = game
            decisions = {name: [] for name in players}
            try:
                for ply in range(a.max_plies):
                    if board.is_game_over(claim_draw=True):
                        break
                    name = a.candidate if board.turn == color else 'baseline'
                    net = nets[name]
                    before = net.search_stats['forward_positions']
                    tic = time.perf_counter()
                    move = players[name].choose(board)
                    elapsed = time.perf_counter() - tic
                    if move not in board.legal_moves:
                        raise RuntimeError('Illegal move')
                    decisions[name].append(dict(seconds=elapsed,
                        large_positions=net.search_stats['forward_positions'] - before,
                        root=players[name].last_result))
                    board.push(move)
                    node = node.add_variation(move)
                    if ply % 16 == 0:
                        (out / 'live.json').write_text(json.dumps(dict(opening=opening,
                            candidate_color=bool(color), ply=ply + 1, fen=board.fen(),
                            elapsed_seconds=time.perf_counter() - start)))
                outcome = board.outcome(claim_draw=True)
                score = None if outcome is None else .5 if outcome.winner is None else float(outcome.winner == color)
                game.headers['Result'] = '*' if outcome is None else outcome.result()
                game.headers['Termination'] = outcome.termination.name if outcome else 'ply-limit'
                row = dict(opening=opening, contender_color='white' if color else 'black',
                    fen=fen, score=score, result=game.headers['Result'], plies=len(board.move_stack),
                    termination=game.headers['Termination'], decisions=decisions,
                    stats={name: dict(player.search_stats) for name, player in players.items()})
                path.with_suffix('.pgn').write_text(str(game) + '\n\n')
                temporary = path.with_suffix('.tmp')
                temporary.write_text(json.dumps(row, indent=2))
                temporary.replace(path)
                rows.append(row)
            finally:
                for player in players.values():
                    player.close()
            report = dict(baseline_id='puct-2s-v1', plan=plan, games=rows,
                          summary=summarize(rows, 2 * a.pairs), setup_seconds=setup,
                          elapsed_seconds=time.perf_counter() - start)
            (out / 'progress.json').write_text(json.dumps(report, indent=2))
            print(json.dumps(report['summary']), flush=True)
    report = dict(baseline_id='puct-2s-v1', plan=plan, games=rows,
                  summary=summarize(rows, 2 * a.pairs), setup_seconds=setup,
                  elapsed_seconds=time.perf_counter() - start)
    (out / 'result.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
