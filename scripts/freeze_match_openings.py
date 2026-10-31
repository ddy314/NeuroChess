"""Freeze near-balanced new game openings before any tested engine plays."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import random
import sqlite3
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import chess
import chess.engine
import chess.pgn
import zstandard
from neurochess.encoding import position_key
from neurochess.prepare import FilteredBuilder


def main():
    p=argparse.ArgumentParser()
    for name in ['pgn','stockfish','key-db','out']:p.add_argument('--'+name,required=True)
    p.add_argument('--after-game',type=int,default=96659);p.add_argument('--pairs',type=int,default=32)
    p.add_argument('--seed',type=int,default=713031);a=p.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False);rng=random.Random(a.seed);roots=[];index=0;seen=set()
    db=sqlite3.connect(f'file:{Path(a.key_db).resolve()}?mode=ro',uri=True)
    with open(a.pgn,'rb') as raw,io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(raw)) as stream,chess.engine.SimpleEngine.popen_uci(a.stockfish) as sf:
        sf.configure({'Threads':1,'Hash':32})
        while len(roots)<a.pairs:
            game=chess.pgn.read_game(stream,Visitor=lambda:FilteredBuilder({'min_elo':1800}))
            if game is None:break
            index+=1
            if index<=a.after_game or game.errors:continue
            b=game.board();choices=[]
            for ply,move in enumerate(game.mainline_moves(),1):
                b.push(move)
                if 16<=ply<=24 and not b.is_game_over(claim_draw=True):choices.append((ply,b.copy()))
            rng.shuffle(choices)
            for ply,b in choices[:3]:
                key=position_key(b)
                if key in seen or db.execute('select 1 from positions where key=?',(key,)).fetchone():continue
                info=sf.analyse(b,chess.engine.Limit(nodes=50000),game=object());score=info['score'].pov(b.turn)
                if score.is_mate() or abs(score.score())>100:continue
                roots.append(dict(game=index,ply=ply,fen=b.fen(en_passant='fen'),key=key.hex(),teacher_cp=score.score()))
                seen.add(key);break
    if len(roots)!=a.pairs:raise ValueError(f'Insufficient source games:{len(roots)}')
    text='\n'.join(r['fen'] for r in roots)+'\n';(out/'openings.fen').write_text(text)
    (out/'plan.json').write_text(json.dumps(dict(args=vars(a),roots=roots,sha256=hashlib.sha256(text.encode()).hexdigest(),
        note='One new source game per opening; ply16..24,teacher50k abs cp<=100,old base exact keys excluded. Both colors. Selection never uses tested-player outputs.'),indent=2))
    print(json.dumps(dict(pairs=len(roots),first_game=roots[0]['game'],last_game=roots[-1]['game'])),flush=True)


if __name__=='__main__':main()
