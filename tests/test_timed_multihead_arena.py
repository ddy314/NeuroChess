"""Timed evaluation preserves per-game native player slots and terminal PGNs."""
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
import chess


def test_timed_arena_uses_locked_options_and_terminal_games(tmp_path, monkeypatch):
    spec=importlib.util.spec_from_file_location('timed_arena_under_test',Path(__file__).resolve().parents[1]/'scripts/multihead_arena.py')
    arena=importlib.util.module_from_spec(spec);spec.loader.exec_module(arena)
    options=json.loads((arena.ROOT/'configs/puct-2s-local.json').read_text())
    config=tmp_path/'search.json';config.write_text(json.dumps(options))
    checkpoint=tmp_path/'weights.pt';checkpoint.write_bytes(b'fixture')
    openings=tmp_path/'openings.fen';openings.write_text('7k/5Q2/6K1/8/8/8/8/8 w - - 0 1\n')
    def move(board):
        for candidate in board.legal_moves:
            after=board.copy();after.push(candidate)
            if after.is_checkmate():return candidate
        raise AssertionError('Fixture expects mate in one')
    class Network:
        search_stats={}
        def __init__(self,*args):pass
        def choose_many(self,boards):return [move(b) for b in boards]
    players=[]
    class Search(Network):
        def __init__(self,network,**kwargs):
            assert kwargs==options
            self.config=kwargs;self.closed=False;self.calls=0;players.append(self)
        def choose(self,board):self.calls+=1;return move(board)
        def choose_many(self,boards):raise AssertionError('Timed player must use one game slot')
        def close(self):self.closed=True
    class Engine:
        id={'name':'Fixture Stockfish'}
        def configure(self,options):pass
        def play(self,board,*args,**kwargs):return SimpleNamespace(move=move(board))
        def quit(self):pass
    monkeypatch.setattr(arena,'load_model',lambda *a:(SimpleNamespace(config=SimpleNamespace(history_free=True)),{'step':0}))
    monkeypatch.setattr(arena,'MultiheadPlayer',Network);monkeypatch.setattr(arena,'NativeMCTSPlayer',Search)
    monkeypatch.setattr(arena.chess.engine.SimpleEngine,'popen_uci',lambda *a,**k:Engine())
    monkeypatch.setattr(arena.torch.cuda,'get_device_name',lambda:'fixture')
    out=tmp_path/'arena'
    monkeypatch.setattr(sys,'argv',['arena','--checkpoint',str(checkpoint),'--stockfish','fixture','--openings',str(openings),'--out',str(out),'--pairs','1','--batch-games','4','--search-config',str(config)])
    arena.main()
    assert len(players)==2 and all(p.closed for p in players)
    assert sum(p.calls for p in players)==1
    report=json.loads((out/'result.json').read_text())
    assert report['completed']==4 and all(r['played']==2 and r['truncated']==0 for r in report['results'].values())
    with (out/'games.pgn').open() as stream:
        games=list(iter(lambda:chess.pgn.read_game(stream),None))
    assert len(games)==4 and all(not g.errors and g.end().board().is_checkmate() for g in games)
    assert {g.headers['SearchSimulations'] for g in games if g.headers['ModelMode']=='search'}=={'16384'}
