from concurrent.futures import ThreadPoolExecutor
import chess
import torch
from neurochess.local_arena import intervals, play_batch, summarize


class FixedPlayer:
    def choose_many(self, boards):
        legal, logits, _, _ = self.evaluate(boards)
        return [moves[int(row[:len(moves)].argmax())] for moves, row in zip(legal, logits)]

    def evaluate(self, boards):
        legal = [list(b.legal_moves) for b in boards]
        logits = torch.zeros((len(boards), max(map(len, legal))))
        for i, moves in enumerate(legal):
            for j, move in enumerate(moves):
                b = boards[i].copy()
                b.push(move)
                if b.is_checkmate():
                    logits[i, j] = 1
        return legal, logits, None, None


def test_head_to_head_scores_and_pgn():
    rows = []
    job = dict(id='mate', condition='h2h-new', opening=0,
               fen='7k/8/5KQ1/8/8/8/8/8 w - - 0 1',
               white='old', black='new', scored_color=chess.BLACK)
    with ThreadPoolExecutor(1) as pool:
        play_batch([job], {'old': FixedPlayer(), 'new': FixedPlayer()}, [], pool,
                   1, 20, lambda row, game: rows.append((row, game)))
    row, game = rows[0]
    assert row['score'] == 0 and row['result'] == '1-0'
    assert row['termination'] == 'CHECKMATE' and len(list(game.mainline_moves())) == 1


def test_truncation_not_a_draw():
    rows = []
    job = dict(id='cap', condition='h2h-new', opening=0, fen=chess.STARTING_FEN,
               white='new', black='old', scored_color=chess.WHITE)
    with ThreadPoolExecutor(1) as pool:
        play_batch([job], {'old': FixedPlayer(), 'new': FixedPlayer()}, [], pool,
                   1, 1, lambda row, game: rows.append(row))
    assert rows[0]['score'] is None and rows[0]['result'] == '*'


def test_pairing_and_nonzero_boundary_uncertainty():
    rows = [dict(condition='strong-new', opening=0, score=1),
            dict(condition='strong-new', opening=0, score=0),
            dict(condition='strong-old', opening=0, score=0),
            dict(condition='strong-old', opening=0, score=0),
            dict(condition='strong-new', opening=1, score=.5),
            dict(condition='strong-new', opening=1, score=None)]
    result = summarize(rows)
    assert result['strong-new']['truncated'] == 1
    assert result['strong-new']['paired']['independent_openings'] == 1
    assert result['strong-new-minus-old']['mean'] == .5
    assert intervals([0] * 128)['hoeffding_95'][1] > 0
