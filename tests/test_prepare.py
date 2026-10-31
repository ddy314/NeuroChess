import io
import json
import chess.pgn
from neurochess.prepare import rejection, base_seconds, prepare


def game():
    return chess.pgn.read_game(io.StringIO('[WhiteElo "2500"]\n[BlackElo "2450"]\n[TimeControl "1800+30"]\n[Result "1/2-1/2"]\n\n1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 d6 8. c3 O-O 1/2-1/2\n'))


def test_quality_filters():
    g = game(); source = {'min_elo':2400,'min_base_seconds':600}
    assert rejection(g, source) is None
    g.headers['TimeControl'] = '60+1'
    assert rejection(g, source) == 'short_or_missing_time_control'
    g.headers['TimeControl'] = '?'
    assert rejection(g, source) == 'short_or_missing_time_control'
    assert rejection(g, dict(source,allow_unknown_time_control=True)) is None
    g.headers['TimeControl'] = '60+1'
    assert rejection(g, dict(source,allow_unknown_time_control=True)) == 'short_or_missing_time_control'
    g.headers['TimeControl'] = '7200+30'; g.headers['BlackElo'] = '2200'
    assert rejection(g, source) == 'low_or_missing_rating'
    assert base_seconds('40/7200:3600+30') == 7200


def test_multiple_sources_are_deduplicated(tmp_path):
    pgn = tmp_path / 'elite.pgn'; pgn.write_text(str(game()) + '\n')
    source = dict(url=str(pgn), min_elo=2400,min_base_seconds=600,max_positions=100)
    prepare([dict(source,name='first'),dict(source,name='duplicate')],tmp_path/'data',deduplicate=True)
    meta = json.loads((tmp_path/'data/manifest.json').read_text())
    first,second=meta['provenance']['sources']
    assert first['positions'] > 0 and second['positions'] == 0
    assert second['duplicates_removed'] == first['positions']
    assert sum(meta['counts'].values()) == first['positions']
