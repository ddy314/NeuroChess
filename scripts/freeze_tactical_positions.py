"""Teacher-only selection of new short-mate and avoidable-mate game roots."""
import argparse
import hashlib
import io
import json
from pathlib import Path
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
    for arg in ['pgn','stockfish','key-db','out']:p.add_argument('--'+arg,required=True)
    p.add_argument('--after-game',type=int,default=94419);p.add_argument('--per-kind',type=int,default=16)
    a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    roots=[];count={'mate':0,'defense':0};index=0;db=sqlite3.connect(f'file:{Path(a.key_db).resolve()}?mode=ro',uri=True)
    with open(a.pgn,'rb') as raw,io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(raw)) as stream,chess.engine.SimpleEngine.popen_uci(a.stockfish) as sf:
        sf.configure({'Threads':1,'Hash':32})
        while min(count.values())<a.per_kind:
            g=chess.pgn.read_game(stream,Visitor=lambda:FilteredBuilder({'min_elo':1600}))
            if g is None:break
            index+=1
            if index<=a.after_game or g.errors or not g.end().board().is_checkmate():continue
            moves=list(g.mainline_moves())
            if len(moves)<20:continue
            for kind,back in [('defense',6),('mate',5)]:
                if count[kind]>=a.per_kind:continue
                b=g.board()
                for move in moves[:-back]:b.push(move)
                if b.is_game_over(claim_draw=True) or db.execute('select 1 from positions where key=?',(position_key(b),)).fetchone():continue
                info=sf.analyse(b,chess.engine.Limit(nodes=50000),game=object());score=info['score'].pov(b.turn)
                played=moves[-back];refs=[info['pv'][0].uci(),played.uci()]
                if kind=='mate':
                    if not score.is_mate() or not 1<=score.mate()<=3:continue
                    bad=None
                else:
                    if score.is_mate():continue
                    bad=sf.analyse(b,chess.engine.Limit(nodes=50000),root_moves=[played],game=object())['score'].pov(b.turn)
                    if not bad.is_mate() or not -3<=bad.mate()<=-1:continue
                roots.append(dict(game=index,kind=kind,fen=b.fen(en_passant='fen'),key=position_key(b).hex(),
                    initial_fen=b.root().fen(en_passant='fen'),history=[m.uci() for m in b.move_stack],
                    reference_moves=list(dict.fromkeys(refs)),teacher_best=dict(cp=score.score(),mate=score.mate()),
                    teacher_played=dict(cp=bad.score(),mate=bad.mate()) if bad is not None else None))
                count[kind]+=1;break
            (out/'status.json').write_text(json.dumps(dict(scanned_games=index,counts=count)))
    if min(count.values())!=a.per_kind:raise ValueError(f'Source exhausted: {count}')
    text='\n'.join(r['fen'] for r in roots)+'\n';(out/'roots.fen').write_text(text)
    (out/'plan.json').write_text(json.dumps(dict(args=vars(a),roots=roots,counts=count,sha256=hashlib.sha256(text.encode()).hexdigest(),
        selection='One root per new game; teacher-only fixed mate<=3 or played move mated<=3 with a best move without a reported mate. No tested-player outputs used.',
        note='Stratified tactical challenge, not representative frequency or Elo. Finite-node labels; final common-union regrading may differ.'),indent=2))
    print(json.dumps(dict(counts=count,first_game=roots[0]['game'],last_game=roots[-1]['game'])),flush=True)


if __name__=='__main__':main()
