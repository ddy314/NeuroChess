import chess
import pytest
import torch
from neurochess.mcts import Edge, Node, MCTSPlayer, terminal_value


class Network:
    device = 'cpu'
    def __init__(self, favorite=None):
        self.favorite = favorite
        self.histories = []

    def evaluate(self, boards):
        assert all(not b.is_game_over(claim_draw=True) for b in boards)
        self.histories.extend(len(b.move_stack) for b in boards)
        legal = [list(b.legal_moves) for b in boards]
        logits = torch.zeros((len(boards), 256))
        for i, moves in enumerate(legal):
            for j, move in enumerate(moves):
                if move.uci() == self.favorite:
                    logits[i, j] = 10
        return legal, logits, logits, torch.zeros(len(boards))


def test_backup_alternates_viewpoint():
    p = MCTSPlayer(Network(), simulations=8)
    root = Node(chess.Board())
    child_board = root.board.copy(); child_board.push_uci('e2e4')
    child = Node(child_board)
    leaf_board = child_board.copy(); leaf_board.push_uci('e7e5')
    leaf = Node(leaf_board)
    a, b = Edge(chess.Move.from_uci('e2e4'), 1, child=child), Edge(chess.Move.from_uci('e7e5'), 1, child=leaf)
    root.edges, child.edges = [a], [b]
    p._backup(leaf, [(root, a), (child, b)], .7)
    assert a.q == pytest.approx(.7) and b.q == pytest.approx(-.7)
    assert root.visits == child.visits == leaf.visits == 1


@pytest.mark.parametrize('mirror', [False, True])
def test_mate_in_one_is_found_with_tiny_policy_prior(mirror):
    board = chess.Board('7k/8/5KQ1/8/8/8/8/8 w - - 0 1')
    if mirror:
        board = board.mirror()
    favorite = next(m.uci() for m in board.legal_moves if not board.gives_check(m))
    player = MCTSPlayer(Network(favorite), simulations=8)
    move = player.choose(board)
    board.push(move)
    assert board.is_checkmate()
    assert player.search_stats['mate_proofs'] > 0


def test_search_avoids_fools_mate_even_with_bad_prior():
    board = chess.Board()
    board.push_uci('f2f3'); board.push_uci('e7e5')
    player = MCTSPlayer(Network('g2g4'), simulations=48)
    move = player.choose(board)
    assert move != chess.Move.from_uci('g2g4')
    assert player.search_stats['proven_loss_skips'] > 0
    bad = next(e for e in player.decision_samples[0]['leaders'] if e['move']=='g2g4')
    assert bad['q'] == -1


def test_budget_progressive_widening_pruning_and_input_immutability():
    board = chess.Board(); fen = board.fen()
    player = MCTSPlayer(Network('e2e4'), simulations=64, initial_width=2,
                        widening=.25, root_keep=2, prune_after=.5, max_depth=8)
    move = player.choose(board)
    assert move in board.legal_moves and board.fen() == fen and not board.move_stack
    assert player.search_stats['completed_simulations'] == 64
    assert player.decision_samples[0]['simulations'] == 64
    assert player.search_stats['pruned_root_edges'] > 0
    assert player.search_stats['leaf_evaluations'] <= 65


def test_pruning_never_discards_forcing_moves_or_policy_leaders():
    root = Node(chess.Board(), visits=48, expanded=True)
    moves = list(root.board.legal_moves)
    root.edges = [Edge(m, 1/(i+2), forcing=(i==len(moves)-1)) for i,m in enumerate(moves)]
    root.edges[-2].child = Node(chess.Board(), solved=-1.)
    player = MCTSPlayer(Network(), simulations=64, root_keep=2, prune_after=.5)
    player._prune_root(root)
    assert all(not e.pruned for e in root.edges[:4])
    assert not root.edges[-1].pruned and sum(e.pruned for e in root.edges)>0
    assert not root.edges[-2].pruned


def test_repetition_history_is_kept_and_terminal_roots_are_rejected():
    board = chess.Board()
    for m in ['g1f3','g8f6','f3g1','f6g8']:
        board.push_uci(m)
    network = Network()
    player = MCTSPlayer(network, simulations=16)
    player.choose(board)
    assert min(network.histories) == 4 and max(network.histories) > 4
    assert len(board.move_stack) == 4
    for m in ['g1f3','g8f6','f3g1','f6g8']:
        board.push_uci(m)
    with pytest.raises(ValueError):
        player.choose(board)


def test_pruned_unknown_move_prevents_false_loss_proof():
    root = Node(chess.Board(), expanded=True)
    root.edges = [Edge(chess.Move.from_uci('e2e4'), .9, child=Node(chess.Board(), solved=1.)),
                  Edge(chess.Move.from_uci('d2d4'), .1, pruned=True)]
    MCTSPlayer._update_proof(root)
    assert root.solved is None


@pytest.mark.parametrize('children,expected', [([0.,1.],0.), ([1.,1.],-1.), ([-1.,None],1.)])
def test_exact_proof_propagation(children, expected):
    root = Node(chess.Board(), expanded=True)
    root.edges = [Edge(m,.5,child=Node(chess.Board(),solved=v) if v is not None else None)
                  for m,v in zip(list(root.board.legal_moves),children)]
    MCTSPlayer._update_proof(root)
    assert root.solved == expected


def test_stalemate_and_checkmate_have_exact_values():
    assert terminal_value(chess.Board('7k/5Q2/5K2/8/8/8/8/8 b - - 0 1')) == 0
    assert terminal_value(chess.Board('7k/6Q1/5K2/8/8/8/8/8 b - - 0 1')) == -1
