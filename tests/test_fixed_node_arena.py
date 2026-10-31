import importlib.util,json,sys
from pathlib import Path
from types import SimpleNamespace
import chess,chess.pgn


def test_fixed_node_schedule_complete_pairs_and_idempotent_resume(tmp_path,monkeypatch):
    spec=importlib.util.spec_from_file_location('fixed_arena',Path('scripts/fixed_node_arena.py'))
    runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
    openings=tmp_path/'openings.fen'
    openings.write_text('\n'.join(f'7k/5Q2/6K1/8/8/8/8/8 w - - 0 {i+1}' for i in range(16))+'\n')
    weights=tmp_path/'model.pt';weights.write_bytes(b'fixture');sf=tmp_path/'sf';sf.write_bytes(b'fixture')
    def mate(board):
        for move in board.legal_moves:
            child=board.copy();child.push(move)
            if child.is_checkmate():return move
        raise AssertionError('Fixture must have mate in one')
    class Player:
        def __init__(self,*a,**kw):self.search_stats={};self.config={'simulations':512}
        def choose_many(self,boards):return [mate(b) for b in boards]
        def close(self):pass
    calls=[]
    class Engine:
        id={'name':'Fixture Stockfish'}
        def configure(self,options):calls.append(options)
        def play(self,board,limit,game):
            assert limit.nodes in [500,1000,1500]
            return SimpleNamespace(move=mate(board))
        def quit(self):pass
    monkeypatch.setattr(runner,'load_model',lambda *a:(SimpleNamespace(config=SimpleNamespace(history_free=True)),{'step':1000,'origin_step':14500}))
    monkeypatch.setattr(runner,'MultiheadPlayer',Player);monkeypatch.setattr(runner,'NativeMCTSPlayer',Player)
    monkeypatch.setattr(runner.native_module,'load_library',lambda:object())
    monkeypatch.setattr(runner.chess.engine.SimpleEngine,'popen_uci',lambda *a,**kw:Engine())
    monkeypatch.setattr(runner.torch.cuda,'get_device_name',lambda:'fixture')
    out=tmp_path/'arena';args=['arena','--checkpoint',str(weights),'--stockfish',str(sf),'--openings',str(openings),'--out',str(out)]
    monkeypatch.setattr(sys,'argv',args);runner.main()
    result=json.loads((out/'result.json').read_text());assert result['completed']==96
    for stats in result['results'].values():
        assert stats['completed']==32 and stats['independent_pairs']==16
        assert (stats['wins'],stats['draws'],stats['losses'])==(16,0,16)
        assert stats['relative_elo']==0 and stats['hoeffding_score95'][0]<.5<stats['hoeffding_score95'][1]
    assert {'Clear Hash':None} in calls
    with (out/'games.pgn').open() as f:
        games=[]
        while (game:=chess.pgn.read_game(f)) is not None:games.append(game)
    assert len(games)==96 and all(not g.errors and g.end().board().is_checkmate() for g in games)
    monkeypatch.setattr(sys,'argv',args+['--resume']);runner.main()
    assert len(list((out/'games').glob('*.json')))==96
