import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import time
import chess
import chess.engine
import chess.pgn
import numpy as np
import torch
from .encoding import encode, legal_ids, MAX_MOVES


def history_has_repetition(board):
    """Check reversible history once per root, instead of once per child."""
    work = board.copy()
    seen = {work._transposition_key()}
    while work.move_stack:
        move = work.pop()
        if work.is_irreversible(move):
            break
        key = work._transposition_key()
        if key in seen:
            return True
        seen.add(key)
    return False


def oneply_outcome(child, root_has_repetition):
    # With no repeated root-history position, a child cannot already have a
    # third occurrence or claim one on its next move (which flips the turn).
    # Automatic draws, mate/stalemate and the 50-move claim still apply.
    result = child.outcome(claim_draw=root_has_repetition)
    if result is None and not root_has_repetition and child.can_claim_fifty_moves():
        result = chess.Outcome(chess.Termination.FIFTY_MOVES, None)
    return result


class Player:
    def __init__(self, checkpoint, device='cpu', mode='policy'):
        if mode not in ('policy', 'q', 'oneply'):
            raise ValueError('Use MCTSPlayer to wrap a policy/value network for MCTS')
        from .train import load_model
        self.model, _ = load_model(checkpoint, device)
        self.model.eval()
        self.device, self.mode = device, mode
        self.search_stats = dict(choices=0, leaf_evaluations=0, terminal_leaves=0)

    @classmethod
    def from_model(cls, model, device='cpu', mode='policy'):
        if mode not in ('policy', 'q', 'oneply'):
            raise ValueError('Use MCTSPlayer to wrap a policy/value network for MCTS')
        player = cls.__new__(cls)
        player.model, player.device, player.mode = model, device, mode
        player.search_stats = dict(choices=0, leaf_evaluations=0, terminal_leaves=0)
        model.eval()
        return player

    @torch.inference_mode()
    def evaluate(self, boards):
        encoded = [encode(b) for b in boards]
        moves = np.full((len(boards), MAX_MOVES), -1, dtype=np.int16)
        legal = []
        for i, board in enumerate(boards):
            ms, ids = legal_ids(board)
            legal.append(ms); moves[i, :len(ids)] = ids
        cuda = self.device.startswith('cuda')
        ctx = torch.autocast('cuda', dtype=torch.bfloat16 if (torch.cuda.get_device_capability()[0] >= 8 and torch.cuda.is_bf16_supported()) else torch.float16) if cuda else nullcontext()
        with ctx:
            policy, q, value = self.model(torch.as_tensor(np.stack([e[0] for e in encoded]), device=self.device),
                torch.as_tensor(np.stack([e[1] for e in encoded]), device=self.device),
                torch.as_tensor(moves, device=self.device))
        values = value.softmax(-1)
        return legal, policy.cpu(), q.cpu(), (values[:, 0] - values[:, 2]).cpu()

    @torch.inference_mode()
    def evaluate_encoded(self, tokens, meta, moves):
        """Native MCTS bridge: native movegen/encoding; one D2H transfer, no Q transfer."""
        cuda = self.device.startswith('cuda')
        ctx = torch.autocast('cuda', dtype=torch.bfloat16 if (torch.cuda.get_device_capability()[0] >= 8 and torch.cuda.is_bf16_supported()) else torch.float16) if cuda else nullcontext()
        with ctx:
            policy, _, value = self.model(torch.as_tensor(tokens, device=self.device),
                torch.as_tensor(meta, device=self.device), torch.as_tensor(moves, device=self.device))
        wdl = value.softmax(-1)
        packed = torch.cat((policy, (wdl[:, 0] - wdl[:, 2]).unsqueeze(1)), dim=1).cpu().numpy()
        return packed[:, :256], packed[:, 256]

    @torch.inference_mode()
    def evaluate_values(self, boards):
        encoded = [encode(b) for b in boards]
        cuda = self.device.startswith('cuda')
        ctx = torch.autocast('cuda', dtype=torch.bfloat16 if (torch.cuda.get_device_capability()[0] >= 8 and torch.cuda.is_bf16_supported()) else torch.float16) if cuda else nullcontext()
        with ctx:
            logits = self.model.value_only(
                torch.as_tensor(np.stack([e[0] for e in encoded]), device=self.device),
                torch.as_tensor(np.stack([e[1] for e in encoded]), device=self.device))
        values = logits.softmax(-1)
        return (values[:, 0] - values[:, 2]).cpu()

    def choose_many(self, boards, leaf_batch_size=128):
        if not boards or leaf_batch_size < 1:
            raise ValueError('Nonempty boards and a positive leaf batch size required')
        if any(b.is_game_over(claim_draw=True) for b in boards):
            raise ValueError('Game already ended')
        self.search_stats['choices'] += len(boards)
        if self.mode != 'oneply':
            legal, policy, q, _ = self.evaluate(boards)
            scores = policy if self.mode == 'policy' else q
            return [moves[int(row[:len(moves)].argmax())] for moves, row in zip(legal, scores)]
        legal = [list(board.legal_moves) for board in boards]
        scores = [np.zeros(len(moves)) for moves in legal]
        children, indices = [], []
        for j, (board, moves) in enumerate(zip(boards, legal)):
            repeated = history_has_repetition(board)
            for i, move in enumerate(moves):
                child = board.copy(); child.push(move)
                result = oneply_outcome(child, repeated)
                if result:
                    scores[j][i] = 0 if result.winner is None else 1 if result.winner == board.turn else -1
                    self.search_stats['terminal_leaves'] += 1
                else:
                    children.append(child); indices.append((j, i))
        if children:
            self.search_stats['leaf_evaluations'] += len(children)
            for offset in range(0, len(children), leaf_batch_size):
                values = self.evaluate_values(children[offset:offset+leaf_batch_size])
                for (j, i), value in zip(indices[offset:offset+leaf_batch_size], values):
                    # Child WDL is expressed from the opponent's point of view.
                    scores[j][i] = -float(value)
        return [moves[int(row.argmax())] for moves, row in zip(legal, scores)]

    def choose(self, board):
        return self.choose_many([board])[0]


def run_match(player, stockfish, fens, out, nodes=10000, max_plies=400, max_seconds=None, progress=None):
    """Reuse an existing model; no second GPU model or concurrent optimizer job."""
    import hashlib
    fens = list(fens)
    if not fens or min(nodes, max_plies) < 1 or (max_seconds is not None and max_seconds <= 0):
        raise ValueError('Positive limits and at least one opening required')
    for fen in fens:
        board = chess.Board(fen)
        if not board.is_valid() or board.is_game_over(claim_draw=True):
            raise ValueError('Opening must be valid and nonterminal')
    root = Path(out); root.mkdir(parents=True, exist_ok=False)
    results, pair_scores, truncated, attempted = [], [], 0, 0
    start = time.perf_counter()
    deadline = start + max_seconds if max_seconds is not None else float('inf')
    settings = dict(nodes=nodes, max_plies=max_plies, max_seconds=max_seconds, pairs=len(fens), mode=player.mode,
                    openings_sha256=hashlib.sha256('\n'.join(fens).encode()).hexdigest())
    if player.mode == 'mcts':
        settings['search'] = player.config
    with chess.engine.SimpleEngine.popen_uci(stockfish, timeout=10) as engine, (root / 'games.pgn').open('w') as output:
        engine.configure({'Threads': 1, 'Hash': 64})
        opponent = engine.id
        for fen in fens:
            pair = []
            for color in [chess.WHITE, chess.BLACK]:
                if time.perf_counter() >= deadline:
                    break
                attempted += 1
                board = chess.Board(fen)
                game = chess.pgn.Game(); game.setup(board)
                game.headers['White'] = 'NeuroChess' if color else engine.id['name']
                game.headers['Black'] = engine.id['name'] if color else 'NeuroChess'
                node = game
                for _ in range(max_plies):
                    remaining = deadline - time.perf_counter()
                    if board.is_game_over(claim_draw=True) or remaining <= 0:
                        break
                    move = player.choose(board) if board.turn == color else engine.play(board,
                        chess.engine.Limit(nodes=nodes, time=min(remaining, 10)), game=game).move
                    board.push(move); node = node.add_variation(move)
                outcome = board.outcome(claim_draw=True)
                if outcome is None:
                    truncated += 1
                    game.headers['Result'] = '*'
                    pair.append(None)
                else:
                    score = .5 if outcome.winner is None else float(outcome.winner == color)
                    results.append(score); pair.append(score)
                    game.headers['Result'] = outcome.result()
                print(game, file=output, end='\n\n'); output.flush()
                if progress:
                    progress(attempted, len(fens) * 2)
            if len(pair) == 2 and None not in pair:
                pair_scores.append(float(np.mean(pair)))
            if time.perf_counter() >= deadline:
                break
    report = dict(settings=settings, opponent=opponent, completed=len(results), truncated=truncated,
        attempted=attempted, scheduled=len(fens) * 2, unplayed=len(fens) * 2 - attempted,
        wins=results.count(1.), draws=results.count(.5), losses=results.count(0.), seconds=time.perf_counter() - start,
        rating_claim='No absolute Elo calibration. Stockfish node limits are not Elo ratings.')
    if player.mode == 'mcts':
        report['search_stats'] = player.search_stats
        report['decision_samples'] = player.decision_samples
    if pair_scores:
        values = np.asarray(pair_scores)
        rng = np.random.default_rng(42)
        means = values[rng.integers(len(values), size=(10000, len(values)))].mean(1)
        report.update(paired_score=float(values.mean()), paired_score_bootstrap_95=np.quantile(means, [.025, .975]).tolist(), independent_pairs=len(values))
    (root / 'result.json').write_text(json.dumps(report, indent=2))
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--stockfish', required=True)
    p.add_argument('--openings', required=True, help='Held-out FENs, one per line; both colors for each')
    p.add_argument('--pairs', type=int, default=20)
    p.add_argument('--nodes', type=int, default=10000)
    p.add_argument('--max-plies', type=int, default=400)
    p.add_argument('--max-seconds', type=float)
    p.add_argument('--mode', choices=['policy', 'q', 'oneply', 'mcts'], default='policy')
    from .mcts import add_mcts_arguments, wrap_mcts
    add_mcts_arguments(p)
    p.add_argument('--device', default='cpu')
    p.add_argument('--out', required=True)
    a = p.parse_args()
    fens = [s.strip() for s in Path(a.openings).read_text().splitlines() if s.strip()][:a.pairs]
    if a.pairs < 1 or len(fens) < a.pairs:
        p.error('not enough independent opening pairs')
    player = Player(a.checkpoint, a.device, 'policy' if a.mode == 'mcts' else a.mode)
    if a.mode == 'mcts':
        player = wrap_mcts(player, a)
    report = run_match(player, a.stockfish, fens, a.out,
                       a.nodes, a.max_plies, a.max_seconds)
    print(json.dumps(report))


if __name__ == '__main__':
    main()
