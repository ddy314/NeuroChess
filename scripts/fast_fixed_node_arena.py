"""Serial 0.5-second optimized PUCT versus fixed-node full-strength Stockfish."""
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
sys.path.insert(0, str(ROOT / 'output/puct-10x-20261031/trt-deps'))
import chess
import chess.engine
import numpy as np
import torch
from neurochess.encoding import encode, legal_ids
from neurochess.multihead import MultiheadPlayer
from neurochess.pipeline_mcts import PipelineMCTSPlayer
from neurochess.tensorrt_inference import TensorRTModel
from neurochess.tensorrt_graph import TensorRTGraphMultiheadPlayer
from neurochess.train import load_model
from scripts.search_duel import setup_game, summarize


def digest(path):
    with open(path, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def summarize_nodes(rows, nodes, pairs):
    return {str(n): summarize([r for r in rows if r['nodes'] == n], 2 * pairs) for n in nodes}


def main():
    parser = argparse.ArgumentParser()
    for name in ['checkpoint', 'engine', 'search-suite', 'openings', 'stockfish', 'out']:
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--candidate', default='trt-graph-pipeline-0.5s')
    parser.add_argument('--nodes', type=int, nargs='+', default=[1500, 2000])
    parser.add_argument('--pairs', type=int, default=5)
    parser.add_argument('--max-plies', type=int, default=1000)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=args.resume)
    suite = {v['name']: v for v in json.loads(Path(args.search_suite).read_text())}
    variant = suite[args.candidate]
    if variant['backend'] != 'tensorrt-int8graph' or variant.get('engine') != 'pipeline':
        raise ValueError('Expected optimized INT8 graph pipeline candidate')
    build = json.loads(Path(args.engine).with_name('build.json').read_text())
    if digest(args.checkpoint) != build['checkpoint_sha256'] or digest(args.engine) != build['engine_sha256']:
        raise ValueError('Engine and selected checkpoint fingerprints differ')
    fens = Path(args.openings).read_text().splitlines()[:args.pairs]
    if args.pairs < 1 or len(fens) != args.pairs or len(set(fens)) != len(fens):
        raise ValueError('Invalid opening sample')
    for fen in fens:
        board = chess.Board(fen)
        if not board.is_valid() or board.is_game_over(claim_draw=True):
            raise ValueError('Invalid opening')
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('high')
    setup = time.perf_counter()
    model, state = load_model(args.checkpoint, 'cuda')
    config = model.config
    step, origin = state.get('step'), state.get('origin_step')
    del state
    quant = TensorRTModel(args.engine, config)
    net = TensorRTGraphMultiheadPlayer(quant, batch_sizes=range(2, variant['options']['leaf_batch_size'] + 1))
    eager = MultiheadPlayer(model)
    quant_eager = MultiheadPlayer(quant)
    validation = []
    # Every possible partial batch is warmed before the first timed move.
    for n in range(1, variant['options']['leaf_batch_size'] + 1):
        boards = [chess.Board(fens[i % len(fens)]) for i in range(n)]
        encoded = [encode(b) for b in boards]
        ids = np.full((n, 256), -1, np.int16)
        for i, board in enumerate(boards):
            li = legal_ids(board)[1]
            ids[i, :len(li)] = li
        inputs = (np.stack([e[0] for e in encoded]), np.stack([e[1] for e in encoded]), ids)
        actual = net.evaluate_search_heads(*inputs)
        reference = quant_eager.evaluate_search_heads(*inputs)
        for x, y in zip(actual, reference):
            np.testing.assert_allclose(x, y, atol=3e-6, rtol=3e-6, equal_nan=True)
        original = eager.evaluate_search_heads(*inputs)
        if not np.isfinite(actual[0][ids >= 0]).all() or not np.isfinite(actual[2]).all():
            raise FloatingPointError('Nonfinite optimized outputs')
        validation.append(dict(batch=n, graph_matches_same_engine=True,
            original_policy_top1_agreement=int((actual[0].argmax(1) == original[0].argmax(1)).sum()),
            max_value_difference=float(np.max(np.abs(actual[2] - original[2])))))
    del model, eager, quant_eager
    torch.cuda.empty_cache()
    setup = time.perf_counter() - setup
    with chess.engine.SimpleEngine.popen_uci(args.stockfish, timeout=120) as sf:
        options = {'Threads': 1, 'Hash': 64, 'Skill Level': 20, 'UCI_LimitStrength': False}
        sf.configure(options)
        files = ['neurochess/pipeline_mcts.py', 'neurochess/native/mcts_pipeline.cpp',
                 'neurochess/native/libmcts_pipeline.so', 'neurochess/tensorrt_inference.py',
                 'neurochess/tensorrt_graph.py', 'neurochess/fused_heads.py',
                 'neurochess/cuda_inference.py', 'neurochess/model.py',
                 'neurochess/multihead.py', 'scripts/fast_fixed_node_arena.py']
        plan = dict(checkpoint=str(Path(args.checkpoint).resolve()), checkpoint_sha256=build['checkpoint_sha256'],
            checkpoint_step=step, origin_step=origin, cumulative_step=step + (origin or 0),
            engine_sha256=build['engine_sha256'], search=variant, nodes=args.nodes, pairs=args.pairs,
            openings=fens, stockfish=sf.id, stockfish_sha256=digest(args.stockfish), stockfish_options=options,
            source_sha256={f: digest(f) for f in files}, max_plies=args.max_plies,
            protocol='Serial games; five fixed openings swapped colors at each node budget. Fresh player/cache and cleared Stockfish hash each game. Claimable terminal outcomes; truncated games never draws.',
            precision='INT8 Transformer blocks, BF16 heads, CUDA Graph; freshly calibrated from this selected checkpoint',
            rating_claim='Fixed-node match results only; no absolute/human Elo estimate.')
        if args.resume and json.loads((out / 'plan.json').read_text()) != plan:
            raise ValueError('Resume requires identical model, code, settings and openings')
        save(out / 'plan.json', plan)
        save(out / 'setup.json', dict(seconds=setup, validation=validation, build_seconds=build['total_seconds']))
        games = out / 'games'
        games.mkdir(exist_ok=True)
        rows = [json.loads(p.read_text()) for p in games.glob('*.json')]
        completed = {(r['nodes'], r['opening'], r['contender_color']) for r in rows}
        start = time.perf_counter()
        scheduled = 2 * args.pairs * len(args.nodes)
        for opening, fen in enumerate(fens):
            order = args.nodes[opening % len(args.nodes):] + args.nodes[:opening % len(args.nodes)]
            for nodes in order:
                for color in [chess.WHITE, chess.BLACK]:
                    label = 'white' if color else 'black'
                    if (nodes, opening, label) in completed:
                        continue
                    player = PipelineMCTSPlayer(net, **variant['options'])
                    sf.configure({'Clear Hash': None})
                    board = chess.Board(fen)
                    name = 'NeuroChess-selected-PUCT-0.5s'
                    opponent = sf.id['name'] + f'-nodes{nodes}'
                    game = setup_game(board, name if color else opponent, opponent if color else name)
                    game.headers.update(Event='NeuroChess optimized 0.5s fixed-node arena',
                        Round=str(opening + 1), SFNodes=str(nodes), SearchSeconds=str(variant['options']['time_limit']),
                        ModelSHA256=build['checkpoint_sha256'])
                    node, decisions, opponent_decisions = game, [], []
                    try:
                        for ply in range(args.max_plies):
                            if board.is_game_over(claim_draw=True):
                                break
                            if board.turn == color:
                                before = net.search_stats['forward_positions']
                                tic = time.perf_counter()
                                move = player.choose(board)
                                elapsed = time.perf_counter() - tic
                                decisions.append(dict(seconds=elapsed, large_positions=net.search_stats['forward_positions'] - before,
                                    root=player.last_result))
                            else:
                                response = sf.play(board, chess.engine.Limit(nodes=nodes), game=game,
                                                   info=chess.engine.INFO_BASIC)
                                move = response.move
                                opponent_decisions.append(dict(nodes=response.info.get('nodes'), depth=response.info.get('depth')))
                            if move not in board.legal_moves:
                                raise RuntimeError('Illegal move')
                            board.push(move)
                            node = node.add_variation(move)
                            if ply % 16 == 0:
                                save(out / 'live.json', dict(completed=len(rows), scheduled=scheduled, nodes=nodes,
                                    opening=opening, color=label, ply=ply + 1, fen=board.fen(), elapsed_seconds=time.perf_counter() - start))
                        outcome = board.outcome(claim_draw=True)
                        score = None if outcome is None else .5 if outcome.winner is None else float(outcome.winner == color)
                        game.headers['Result'] = '*' if outcome is None else outcome.result()
                        game.headers['Termination'] = outcome.termination.name if outcome else 'ply-limit'
                        row = dict(nodes=nodes, opening=opening, contender_color=label, fen=fen, score=score,
                            result=game.headers['Result'], termination=game.headers['Termination'], plies=len(board.move_stack),
                            decisions=decisions, opponent_decisions=opponent_decisions, stats=dict(player.search_stats))
                        stem = games / f'{nodes}-{opening:03d}-{label}'
                        stem.with_suffix('.pgn').write_text(str(game) + '\n\n')
                        save(stem.with_suffix('.json'), row)
                        rows.append(row)
                        completed.add((nodes, opening, label))
                    finally:
                        player.close()
                    report = dict(plan=plan, games=rows, summary=summarize_nodes(rows, args.nodes, args.pairs),
                        completed=len(rows), scheduled=scheduled, elapsed_seconds=time.perf_counter() - start)
                    save(out / 'progress.json', report)
                    print(json.dumps(dict(nodes=nodes, opening=opening, color=label, score=score,
                        summary=report['summary'])), flush=True)
        if len(rows) != scheduled or any(r['score'] is None for r in rows):
            raise RuntimeError('Incomplete games; inspect PGNs rather than counting truncations as draws')
        report = dict(plan=plan, games=rows, summary=summarize_nodes(rows, args.nodes, args.pairs),
            completed=len(rows), scheduled=scheduled, elapsed_seconds=time.perf_counter() - start)
        save(out / 'result.json', report)
        (out / 'games.pgn').write_text(''.join(p.read_text() for p in sorted(games.glob('*.pgn'))))


if __name__ == '__main__':
    main()
