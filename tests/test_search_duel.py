import io
import chess
import chess.pgn
from scripts.search_duel import rating,summarize,setup_game


def test_duel_scores_are_contender_relative_and_pairs_are_sampling_units():
    rows=[dict(opening=0,score=1.),dict(opening=0,score=.5),
          dict(opening=1,score=0.),dict(opening=1,score=.5)]
    result=summarize(rows,4)
    assert result['wins']==1 and result['draws']==2 and result['losses']==1
    assert result['independent_completed_pairs']==2
    assert result['paired_score']==.5 and result['relative_elo']==0
    assert result['pentanomial']==[0,1,0,1,0]


def test_unfinished_games_are_neither_draws_nor_completed_pairs():
    result=summarize([dict(opening=0,score=1.),dict(opening=0,score=None)],2)
    assert result['completed']==1 and result['truncated']==1 and result['draws']==0
    assert result['independent_completed_pairs']==0
    assert 'relative_elo' not in result


def test_rating_extremes_and_degenerate_bootstrap_do_not_claim_exact_certainty():
    assert rating(0) is None and rating(1) is None
    rows=[dict(opening=i//2,score=.5) for i in range(64)]
    result=summarize(rows,64)
    assert result['score_bootstrap95']==[.5,.5]
    assert result['score_hoeffding95'][0]<.5<result['score_hoeffding95'][1]


def test_pgn_preserves_raw_ep_observation_with_no_legal_ep_capture():
    board=chess.Board();board.push_uci('e2e4')
    assert not board.has_legal_en_passant()
    game=setup_game(board,'a','b')
    restored=chess.pgn.read_game(io.StringIO(str(game)))
    assert restored.board().fen(en_passant='fen')==board.fen(en_passant='fen')
    assert restored.headers['White']=='a' and restored.headers['Black']=='b'
