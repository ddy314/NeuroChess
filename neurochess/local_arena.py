"""Batched, paired local comparison of two policy checkpoints."""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import time

import chess
import chess.engine
import chess.pgn
import numpy as np
import torch

from .play import Player
from .mcts import add_mcts_arguments, wrap_mcts


def intervals(values):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return None
    rng = np.random.default_rng(20261030)
    means = values[rng.integers(len(values), size=(20000, len(values)))].mean(1)
    # Pair scores are bounded in [0, 1]. This bound remains nonzero when all
    # observed scores coincide, unlike a percentile bootstrap at the boundary.
    radius = math.sqrt(math.log(40) / (2 * len(values)))
    return dict(mean=float(values.mean()), independent_openings=len(values),
                opening_bootstrap_95=np.quantile(means, [.025, .975]).tolist(),
                hoeffding_95=[max(0., float(values.mean()) - radius),
                              min(1., float(values.mean()) + radius)])


def summarize(rows):
    result = {}
    groups = defaultdict(list)
    for row in rows:
        groups[row['condition']].append(row)
    for condition, games in groups.items():
        complete = [g for g in games if g['score'] is not None]
        pairs = defaultdict(list)
        for g in complete:
            pairs[g['opening']].append(g['score'])
        paired = {k: sum(v) / 2 for k, v in pairs.items() if len(v) == 2}
        result[condition] = dict(games=len(games), completed=len(complete),
                                 truncated=len(games) - len(complete),
                                 wins=sum(g['score'] == 1 for g in complete),
                                 draws=sum(g['score'] == .5 for g in complete),
                                 losses=sum(g['score'] == 0 for g in complete),
                                 paired=intervals(list(paired.values())))
    for level in ['strong', 'weak']:
        tables = []
        for who in ['old', 'new']:
            table = defaultdict(list)
            for g in groups.get(f'{level}-{who}', []):
                if g['score'] is not None:
                    table[g['opening']].append(g['score'])
            tables.append({k: sum(v) / 2 for k, v in table.items() if len(v) == 2})
        common = sorted(tables[0].keys() & tables[1].keys())
        if common:
            delta = np.array([tables[1][i] - tables[0][i] for i in common])
            rng = np.random.default_rng(20261030)
            means = delta[rng.integers(len(delta), size=(20000, len(delta)))].mean(1)
            radius = 2 * math.sqrt(math.log(40) / (2 * len(delta)))
            result[f'{level}-new-minus-old'] = dict(
                matched_openings=len(common), mean=float(delta.mean()),
                paired_bootstrap_95=np.quantile(means, [.025, .975]).tolist(),
                hoeffding_95=[max(-1., float(delta.mean()) - radius), min(1., float(delta.mean()) + radius)])
    return result


def play_batch(jobs, players, engines, pool, nodes, max_plies, finish):
    active = []
    for slot, job in enumerate(jobs):
        board = chess.Board(job['fen'])
        game = chess.pgn.Game()
        game.setup(board)
        game.headers.update(Event='NeuroChess local paired evaluation',
                            White=job['white'], Black=job['black'],
                            Round=str(job['opening'] + 1), GameId=job['id'])
        game.headers['ModelMode'] = getattr(next(iter(players.values())), 'mode', 'policy')
        game.headers['SearchDepth'] = {'oneply': '1', 'mcts': 'variable'}.get(game.headers['ModelMode'], '0')
        active.append(dict(job=job, board=board, game=game, node=game, plies=0, slot=slot))

    def settle():
        nonlocal active
        remaining = []
        for g in active:
            outcome = g['board'].outcome(claim_draw=True)
            if outcome is None and g['plies'] < max_plies:
                remaining.append(g)
                continue
            job = g['job']
            score = None if outcome is None else (.5 if outcome.winner is None else float(outcome.winner == job['scored_color']))
            g['game'].headers['Result'] = '*' if outcome is None else outcome.result()
            finish(dict(id=job['id'], condition=job['condition'], opening=job['opening'],
                        scored_color='white' if job['scored_color'] else 'black', score=score,
                        plies=g['plies'], result=g['game'].headers['Result'],
                        termination='max_plies' if outcome is None else outcome.termination.name), g['game'])
        active = remaining

    def push(g, move):
        if move not in g['board'].legal_moves:
            raise RuntimeError('Illegal evaluation move')
        g['board'].push(move)
        g['node'] = g['node'].add_variation(move)
        g['plies'] += 1

    while active:
        settle()
        for name, player in players.items():
            pending = [g for g in active if (g['job']['white'] if g['board'].turn else g['job']['black']) == name]
            if not pending:
                continue
            moves = player.choose_many([g['board'] for g in pending])
            for g, move in zip(pending, moves):
                push(g, move)
            settle()
        pending = [g for g in active if (g['job']['white'] if g['board'].turn else g['job']['black']) == 'Stockfish']
        def engine_move(g):
            return engines[g['slot']].play(g['board'], chess.engine.Limit(nodes=nodes), game=g['game']).move
        for g, move in zip(pending, pool.map(engine_move, pending)):
            push(g, move)
        settle()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--old')
    p.add_argument('--new')
    p.add_argument('--model', help='Evaluate one checkpoint against Stockfish')
    p.add_argument('--openings', required=True)
    p.add_argument('--stockfish', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--mode', choices=['policy', 'oneply', 'mcts'], default='policy')
    add_mcts_arguments(p)
    p.add_argument('--batch-games', type=int, default=16)
    p.add_argument('--nodes', type=int, default=1000)
    p.add_argument('--max-plies', type=int, default=800)
    p.add_argument('--conditions', nargs='+', default=['strong', 'weak', 'h2h'], choices=['strong', 'weak', 'h2h'])
    a = p.parse_args()
    if a.model:
        if a.old or a.new or 'h2h' in a.conditions:
            p.error('--model requires Stockfish-only conditions and cannot be combined with --old/--new')
        a.new = a.model
        model_names = ['new']
    else:
        if not a.old or not a.new:
            p.error('provide --model or both --old and --new')
        model_names = ['old', 'new']
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('high')
    root = Path(a.out)
    root.mkdir(parents=True, exist_ok=True)
    fens = [f.strip() for f in Path(a.openings).read_text().splitlines() if f.strip()]
    if not fens or len(set(' '.join(chess.Board(f).fen().split()[:4]) for f in fens)) != len(fens):
        p.error('Openings must be nonempty and unique')
    labels = {}
    players = {}
    checkpoint_info = {}
    from .train import load_model
    for who in model_names:
        path = Path(getattr(a, who))
        model, state = load_model(path, a.device)
        label = f'NeuroChess-step-{state["step"]}'
        labels[who] = label
        network = Player.from_model(model, a.device, 'policy' if a.mode == 'mcts' else a.mode)
        players[label] = wrap_mcts(network, a) if a.mode == 'mcts' else network
        checkpoint_info[who] = dict(path=str(path.resolve()), step=state['step'],
                                    sha256=hashlib.file_digest(path.open('rb'), 'sha256').hexdigest())
        del state
    plan = dict(checkpoints=checkpoint_info, openings_sha256=hashlib.sha256('\n'.join(fens).encode()).hexdigest(),
                opening_pairs=len(fens), nodes=a.nodes, max_plies=a.max_plies, conditions=a.conditions,
                mode=a.mode, batch_games=a.batch_games, device=a.device, torch=torch.__version__,
                gpu=torch.cuda.get_device_name() if a.device.startswith('cuda') else None,
                draw_rule='Claimable threefold repetition and 50-move draw are adjudicated as draws.',
                caveat='Fixed local comparison, not an absolute Elo calibration. Weak opponent uses Skill Level 0.')
    if a.mode == 'oneply':
        plan['search'] = dict(depth_plies=1, candidates='all legal moves',
                              evaluator='negated child WDL expectation; exact terminal outcomes', leaf_batch_size=128)
    elif a.mode == 'mcts':
        plan['search'] = next(iter(players.values())).config
    manifest = root / 'plan.json'
    if manifest.exists() and json.loads(manifest.read_text()) != plan:
        raise ValueError('Existing evaluation has a different plan')
    manifest.write_text(json.dumps(plan, indent=2))
    records = root / 'games.jsonl'
    rows = [json.loads(s) for s in records.read_text().splitlines()] if records.exists() else []
    done = {r['id'] for r in rows}
    total = len(fens) * 2 * sum(len(model_names) if c != 'h2h' else 1 for c in a.conditions)
    start = time.time()
    def status(phase):
        data = dict(phase=phase, completed=len(rows), scheduled=total, updated_at=time.time(),
                    elapsed_seconds=time.time() - start, results=summarize(rows),
                    search_stats={name: player.search_stats for name, player in players.items()})
        if a.mode == 'mcts':
            data['decision_samples'] = {name: player.decision_samples for name, player in players.items()}
        tmp = root / 'status.tmp'
        tmp.write_text(json.dumps(data, indent=2))
        tmp.replace(root / 'status.json')
    with records.open('a') as log, (root / 'games.pgn').open('a') as pgn:
        def finish(row, game):
            rows.append(row)
            log.write(json.dumps(row) + '\n')
            log.flush()
            print(game, file=pgn, end='\n\n')
            pgn.flush()
            if len(rows) % 32 == 0:
                status('running')
                print(json.dumps(dict(games=len(rows), total=total, last_condition=row['condition'], seconds=time.time() - start)), flush=True)
        engines = []
        try:
            if any(c != 'h2h' for c in a.conditions):
                for _ in range(a.batch_games):
                    engine = chess.engine.SimpleEngine.popen_uci(a.stockfish, timeout=30)
                    engine.configure({'Threads': 1, 'Hash': 16})
                    engines.append(engine)
                plan['opponent'] = engines[0].id
                (root / 'engine.json').write_text(json.dumps(engines[0].id, indent=2))
            status('running')
            with ThreadPoolExecutor(max_workers=min(a.batch_games, 8)) as pool:
                # Alternate model order between opening blocks to share thermal/time effects.
                for offset in range(0, len(fens), 16):
                    for level in a.conditions:
                        models = (model_names if offset // 16 % 2 == 0 else list(reversed(model_names))) if level != 'h2h' else ['new']
                        for who in models:
                            condition = f'{level}-{who}' if level != 'h2h' else 'h2h-new'
                            if engines:
                                for engine in engines:
                                    engine.configure({'Skill Level': 0 if level == 'weak' else 20})
                            jobs = []
                            for i in range(offset, min(offset + 16, len(fens))):
                                for color in [chess.WHITE, chess.BLACK]:
                                    name = f'{condition}-{i:04d}-{int(color)}'
                                    if name in done:
                                        continue
                                    other = 'Stockfish' if level != 'h2h' else labels['old']
                                    jobs.append(dict(id=name, condition=condition, opening=i, fen=fens[i],
                                                     white=labels[who] if color else other,
                                                     black=other if color else labels[who], scored_color=color))
                            for k in range(0, len(jobs), a.batch_games):
                                play_batch(jobs[k:k+a.batch_games], players, engines, pool, a.nodes, a.max_plies, finish)
            status('complete')
            (root / 'result.json').write_text(json.dumps(summarize(rows), indent=2))
            print(json.dumps(summarize(rows)), flush=True)
        except BaseException:
            status('failed')
            raise
        finally:
            for engine in engines:
                engine.quit()


if __name__ == '__main__':
    main()
