"""Freeze later-game roots with exact training-key exclusions, before evaluation."""
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
import chess.pgn
import zstandard
from neurochess.encoding import position_key
from neurochess.prepare import FilteredBuilder


def main():
    p=argparse.ArgumentParser();p.add_argument('--pgn',required=True);p.add_argument('--key-db',required=True)
    p.add_argument('--after-game',type=int,required=True);p.add_argument('--exclude',nargs='+',required=True)
    p.add_argument('--out',required=True);p.add_argument('--roots',type=int,default=32);p.add_argument('--seed',type=int,default=317133)
    p.add_argument('--min-elo',type=int,default=2000)
    a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=False);rng=random.Random(a.seed)
    excluded={position_key(chess.Board(s)) for path in a.exclude for s in Path(path).read_text().splitlines() if s.strip()}
    db=sqlite3.connect(f'file:{Path(a.key_db).resolve()}?mode=ro',uri=True);rows=[];index=0
    raw=open(a.pgn,'rb');reader=zstandard.ZstdDecompressor().stream_reader(raw) if a.pgn.endswith('.zst') else raw
    with io.TextIOWrapper(reader) as stream:
        while len(rows)<a.roots:
            game=chess.pgn.read_game(stream,Visitor=lambda:FilteredBuilder({'min_elo':a.min_elo}))
            if game is None:break
            index+=1
            if index<=a.after_game or game.errors:continue
            b=game.board();possible=[];related=False
            for ply,move in enumerate(game.mainline_moves(),1):
                b.push(move);key=position_key(b)
                if key in excluded:related=True
                material=sum(v*(len(b.pieces(pt,True))-len(b.pieces(pt,False))) for pt,v in [(1,100),(2,320),(3,330),(4,500),(5,900)])
                if 16<=ply<=100 and abs(material)<=500 and not b.is_game_over(claim_draw=True):
                    possible.append(b.fen(en_passant='fen'))
            if related:continue
            rng.shuffle(possible)
            for fen in possible:
                key=position_key(chess.Board(fen))
                if key in excluded or db.execute('select 1 from positions where key=?',(key,)).fetchone():continue
                rows.append(dict(game=index,fen=fen,key=key.hex()));excluded.add(key);break
    if len(rows)!=a.roots:raise ValueError('Source exhausted before requested root count')
    text='\n'.join(r['fen'] for r in rows)+'\n';(out/'roots.fen').write_text(text)
    (out/'plan.json').write_text(json.dumps(dict(args=vars(a),roots=rows,sha256=hashlib.sha256(text.encode()).hexdigest(),
        note='Later, disjoint augmentation/evaluation game groups; exact base keys excluded. Original base source/game overlap remains unknown.'),indent=2))
    print(json.dumps(dict(roots=len(rows),first_game=rows[0]['game'],last_game=rows[-1]['game'])),flush=True)


if __name__=='__main__':main()
