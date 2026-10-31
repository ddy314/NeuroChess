"""Policy-guided PUCT with history-preserving trees and conservative pruning."""
from dataclasses import dataclass, field
import math

import chess
import numpy as np


@dataclass
class Edge:
    move: chess.Move
    prior: float
    forcing: bool = False
    visits: int = 0
    total: float = 0.
    child: object = None
    pruned: bool = False

    @property
    def q(self):
        return self.total / self.visits if self.visits else 0.


@dataclass
class Node:
    board: chess.Board
    visits: int = 0
    value: float = 0.
    edges: list = field(default_factory=list)
    expanded: bool = False
    solved: object = None  # Exact outcome from this node's side-to-move view.
    pruned_once: bool = False


def terminal_value(board):
    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        return None
    return 0. if outcome.winner is None else (1. if outcome.winner == board.turn else -1.)


class MCTSPlayer:
    """One leaf per tree per round, batched across independent games.

    Edge Q is always from the parent player's perspective. Neural WDL values
    and solved node outcomes are always from the node's side-to-move view.
    No board-only transposition merging: history affects both draws and inputs.
    """
    mode = 'mcts'

    def __init__(self, network, simulations=256, c_puct=1.5, fpu_reduction=.2,
                 initial_width=6, widening=1.5, root_keep=12, prune_after=.75,
                 max_depth=128):
        if simulations < 1 or c_puct <= 0 or fpu_reduction < 0 or min(initial_width, root_keep, max_depth) < 1 or widening <= 0 or not 0 < prune_after <= 1:
            raise ValueError('Invalid MCTS configuration')
        self.network = network
        self.device = getattr(network, 'device', 'cpu')
        self.simulations, self.c_puct, self.fpu_reduction = simulations, c_puct, fpu_reduction
        self.initial_width, self.widening = initial_width, widening
        self.root_keep, self.prune_after, self.max_depth = root_keep, prune_after, max_depth
        self.search_stats = dict(choices=0, leaf_evaluations=0, terminal_leaves=0,
                                 completed_simulations=0, expanded_nodes=0, pruned_root_edges=0,
                                 proven_loss_skips=0, mate_proofs=0, max_depth=0, depth_limit_hits=0)
        self.decision_samples = []

    @property
    def config(self):
        return dict(algorithm='PUCT', simulations_per_move=self.simulations, c_puct=self.c_puct,
                    fpu_reduction=self.fpu_reduction, initial_width=self.initial_width,
                    fpu='max(-1, node_value - fpu_reduction)',
                    q_definition='mean backed-up value from parent side to move; not the network Q head',
                    progressive_widening=self.widening, root_keep=self.root_keep,
                    prune_after_fraction=self.prune_after, max_depth=self.max_depth,
                    root_noise=False, final_move='most visits; exact proven outcome overrides',
                    root_pruning='retain visit leaders, top-4 priors, checks/captures/promotions and proven wins',
                    tree_reuse=False, transposition_merging=False,
                    value_view='side to move; sign reversed on each backed-up edge')

    def _expand(self, nodes):
        legal, logits, _, values = self.network.evaluate([n.board for n in nodes])
        self.search_stats['leaf_evaluations'] += len(nodes)
        for node, moves, policy, value in zip(nodes, legal, logits, values):
            scores = np.asarray(policy[:len(moves)], dtype=np.float64)
            value = float(value)
            if not len(moves) or not np.isfinite(scores).all() or not math.isfinite(value):
                raise FloatingPointError('Invalid network evaluation in MCTS')
            priors = np.exp(scores - scores.max())
            priors = np.maximum(priors / priors.sum(), 1e-8)
            priors /= priors.sum()
            node.value = max(-1., min(1., value))
            for move, prior in zip(moves, priors):
                check = node.board.gives_check(move)
                edge = Edge(move, float(prior), check or node.board.is_capture(move) or bool(move.promotion))
                # Scan checks for mate even if a winning move has tiny prior.
                if check:
                    board = node.board.copy()
                    board.push(move)
                    if board.is_checkmate():
                        edge.child = Node(board, expanded=True, solved=-1., value=-1.)
                        node.solved = 1.
                        self.search_stats['mate_proofs'] += 1
                node.edges.append(edge)
            node.edges.sort(key=lambda e: e.prior, reverse=True)
            node.expanded = True
            self.search_stats['expanded_nodes'] += 1

    @staticmethod
    def _update_proof(node):
        known = [e.child.solved if e.child is not None else None for e in node.edges]
        if any(v == -1. for v in known):
            node.solved = 1.
        elif known and all(v is not None for v in known):
            # All legal moves, including pruned ones, are required for loss/draw proof.
            node.solved = max(-v for v in known)

    def _prune_root(self, root):
        if root.pruned_once or root.visits < math.ceil(self.simulations * self.prune_after):
            return
        root.pruned_once = True
        ranked = sorted(root.edges, key=lambda e: (e.visits, e.q, e.prior), reverse=True)
        keep = {id(e) for e in ranked[:self.root_keep] + root.edges[:4]}
        for edge in root.edges:
            proven_win = edge.child is not None and edge.child.solved == -1.
            if id(edge) not in keep and not edge.forcing and not proven_win:
                edge.pruned = True
                self.search_stats['pruned_root_edges'] += 1

    def _select(self, node):
        eligible = [e for e in node.edges if not e.pruned]
        width = min(len(eligible), self.initial_width + int(self.widening * math.sqrt(node.visits)))
        allowed = {id(e) for e in eligible[:width]}
        candidates = [e for e in eligible if id(e) in allowed or e.forcing or
                      (e.child is not None and e.child.solved == -1.)]
        nonlosing = [e for e in candidates if e.child is None or e.child.solved != 1.]
        if nonlosing:
            self.search_stats['proven_loss_skips'] += len(candidates) - len(nonlosing)
            candidates = nonlosing
        def score(edge):
            q = edge.q if edge.visits else max(-1., node.value - self.fpu_reduction)
            return q + self.c_puct * edge.prior * math.sqrt(max(1, node.visits)) / (1 + edge.visits)
        return max(candidates, key=lambda e: (score(e), e.prior))

    def _backup(self, leaf, path, value):
        leaf.visits += 1
        for parent, edge in reversed(path):
            value = -value
            edge.visits += 1
            edge.total += value
            parent.visits += 1
            self._update_proof(parent)
            if parent.solved is not None:
                value = parent.solved

    def choose_many(self, boards):
        if not boards or any(b.is_game_over(claim_draw=True) for b in boards):
            raise ValueError('MCTS requires nonterminal root positions')
        self.search_stats['choices'] += len(boards)
        roots = [Node(b.copy()) for b in boards]
        self._expand(roots)
        for _ in range(self.simulations):
            pending, paths = [], []
            for root in roots:
                if root.solved is not None:
                    continue
                self._prune_root(root)
                node, path = root, []
                while node.expanded and node.solved is None and len(path) < self.max_depth:
                    edge = self._select(node)
                    if edge.child is None:
                        board = node.board.copy()
                        board.push(edge.move)
                        value = terminal_value(board)
                        edge.child = Node(board, value=0. if value is None else value,
                                          expanded=value is not None, solved=value)
                    path.append((node, edge))
                    node = edge.child
                self.search_stats['max_depth'] = max(self.search_stats['max_depth'], len(path))
                if node.solved is not None:
                    self.search_stats['terminal_leaves'] += 1
                    self._backup(node, path, node.solved)
                elif not node.expanded:
                    pending.append(node)
                    paths.append(path)
                    continue
                else:
                    self.search_stats['depth_limit_hits'] += 1
                    self._backup(node, path, node.value)
                self.search_stats['completed_simulations'] += 1
            if pending:
                self._expand(pending)
                for leaf, path in zip(pending, paths):
                    self._backup(leaf, path, leaf.value if leaf.solved is None else leaf.solved)
                    self.search_stats['completed_simulations'] += 1
        moves = []
        for root in roots:
            if root.solved is not None:
                choices = [e for e in root.edges if e.child is not None and e.child.solved == -root.solved]
            else:
                choices = [e for e in root.edges if not e.pruned]
                safe = [e for e in choices if e.child is None or e.child.solved != 1.]
                choices = safe or choices
            edge = max(choices, key=lambda e: (e.visits, e.q, e.prior))
            moves.append(edge.move)
            if len(self.decision_samples) < 8:
                self.decision_samples.append(dict(fen=root.board.fen(), move=edge.move.uci(),
                    simulations=root.visits, proven_value=root.solved,
                    pruned=sum(e.pruned for e in root.edges),
                    leaders=[dict(move=e.move.uci(), visits=e.visits, q=e.q, prior=e.prior,
                                  pruned=e.pruned) for e in sorted(root.edges, key=lambda e:e.visits, reverse=True)[:5]]))
        return moves

    def choose(self, board):
        return self.choose_many([board])[0]


def add_mcts_arguments(parser):
    parser.add_argument('--mcts-backend', choices=['python', 'cpp'], default='cpp')
    parser.add_argument('--leaf-batch-size', type=int, default=16)
    parser.add_argument('--simulations', type=int, default=256)
    parser.add_argument('--c-puct', type=float, default=1.5)
    parser.add_argument('--root-keep', type=int, default=12)
    parser.add_argument('--search-seconds',type=float,help='Native single-game search time budget')
    parser.add_argument('--inference-cache-size',type=int,default=0)
    parser.add_argument('--search-config',help='Native player JSON options; overrides individual search flags')


def wrap_mcts(network, args):
    seconds=getattr(args,'search_seconds',None);cache=getattr(args,'inference_cache_size',0)
    config=getattr(args,'search_config',None)
    if args.mcts_backend == 'cpp':
        from .native_mcts import NativeMCTSPlayer
        options=dict(simulations=args.simulations,c_puct=args.c_puct,root_keep=args.root_keep,
                     leaf_batch_size=args.leaf_batch_size,time_limit=seconds,inference_cache_size=cache)
        if config:
            import json
            from pathlib import Path
            options.update(json.loads(Path(config).read_text()))
        return NativeMCTSPlayer(network,**options)
    if seconds is not None or cache or config:
        raise ValueError('Time limits, inference cache and search config require the cpp backend')
    return MCTSPlayer(network, simulations=args.simulations, c_puct=args.c_puct, root_keep=args.root_keep)
