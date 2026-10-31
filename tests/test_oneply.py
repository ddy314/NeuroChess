import chess
import torch
from neurochess.model import ChessModel, ModelConfig
from neurochess.play import Player, history_has_repetition, oneply_outcome


def fake_oneply(evaluator):
    player = Player.__new__(Player)
    player.mode, player.device = 'oneply', 'cpu'
    player.search_stats = dict(choices=0, leaf_evaluations=0, terminal_leaves=0)
    player.evaluate_values = evaluator
    return player


def test_child_value_is_negated_for_both_colors_and_batching():
    boards = [chess.Board(), chess.Board()]
    boards[1].push_uci('e2e4')
    targets = {'e2e4', 'e7e5'}
    calls = []
    def values(children):
        calls.append(len(children))
        return torch.tensor([-.7 if b.peek().uci() in targets else .8 for b in children])
    player = fake_oneply(values)
    moves = player.choose_many(boards, leaf_batch_size=3)
    assert [m.uci() for m in moves] == ['e2e4', 'e7e5']
    assert max(calls) <= 3 and player.search_stats['choices'] == 2
    assert sum(calls) == player.search_stats['leaf_evaluations']


def test_exact_mate_beats_network_values_and_terminal_leaves_are_not_evaluated():
    board = chess.Board('7k/8/5KQ1/8/8/8/8/8 w - - 0 1')
    def values(children):
        assert all(not b.is_game_over(claim_draw=True) for b in children)
        return torch.full((len(children),), -.99)
    player = fake_oneply(values)
    board.push(player.choose(board))
    assert board.is_checkmate()
    assert player.search_stats['terminal_leaves'] > 0


def test_oneply_keeps_repetition_history_and_does_not_mutate_root():
    board = chess.Board()
    for move in ['g1f3', 'g8f6', 'f3g1', 'f6g8']:
        board.push_uci(move)
    fen = board.fen()
    def values(children):
        assert all(len(b.move_stack) == 5 for b in children)
        return torch.zeros(len(children))
    fake_oneply(values).choose(board)
    assert board.fen() == fen and len(board.move_stack) == 4


def test_value_only_path_matches_full_forward():
    torch.manual_seed(17)
    model = ChessModel(ModelConfig(width=32, layers=2, heads=4, head_width=16)).eval()
    tokens = torch.randint(0, 13, (3, 64))
    meta = torch.randn(3, 12)
    moves = torch.randint(0, 20480, (3, 20))
    with torch.inference_mode():
        expected = model(tokens, meta, moves)[2]
        actual = model.value_only(tokens, meta)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_optimized_draw_checks_match_chess_rules_on_random_histories():
    import random
    rng = random.Random(20261030)
    board = chess.Board()
    histories = []
    for _ in range(120):
        if board.is_game_over(claim_draw=True):
            board = chess.Board()
        histories.append(board.copy())
        board.push(rng.choice(list(board.legal_moves)))
    repeated = chess.Board()
    for move in ['g1f3', 'g8f6', 'f3g1', 'f6g8']:
        repeated.push_uci(move)
    histories.extend([repeated, chess.Board('7k/8/5K2/8/8/8/8/R7 w - - 98 60')])
    for root in histories:
        flag = history_has_repetition(root)
        for move in root.legal_moves:
            child = root.copy()
            child.push(move)
            assert oneply_outcome(child, flag) == child.outcome(claim_draw=True)


def test_fifty_move_claim_still_applies_without_repetition():
    board = chess.Board('7k/8/5K2/8/8/8/8/R7 w - - 98 60')
    assert not history_has_repetition(board)
    child = board.copy()
    child.push(next(iter(child.legal_moves)))
    assert oneply_outcome(child, False) == child.outcome(claim_draw=True)
    assert oneply_outcome(child, False).termination == chess.Termination.FIFTY_MOVES
