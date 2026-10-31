"""Find the first substantial teacher-assessed loss in each recorded student game."""
import argparse
import concurrent.futures
import json
from multiprocessing.util import Finalize
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import chess
import chess.engine
import chess.pgn

ENGINE=None


def initialize(path):
    global ENGINE
    ENGINE=chess.engine.SimpleEngine.popen_uci(path,timeout=60)
    ENGINE.configure({'Threads':1,'Hash':32});Finalize(None,ENGINE.quit,exitpriority=10)


def assess(job):
    b=chess.Board(job['initial_fen'])
    for move in job['history']:b.push_uci(move)
    played=b.parse_uci(job['move']);info=ENGINE.analyse(b,chess.engine.Limit(nodes=50000),game=object())
    moves=list(dict.fromkeys([played,info['pv'][0]]))
    infos=ENGINE.analyse(b,chess.engine.Limit(nodes=100000),root_moves=moves,multipv=len(moves),game=object())
    labels={}
    for i in infos:
        s=i['score'].pov(b.turn);w=s.wdl(model='sf16',ply=32)
        labels[i['pv'][0].uci()]=dict(cp=s.score(),mate=s.mate(),sort_score=s.score(mate_score=100000),
            value=(w.wins-w.losses)/1000,pv=[m.uci() for m in i['pv'][:10]])
    best=max(labels,key=lambda m:labels[m]['sort_score'])
    return dict(**job,labels=labels,best=best,drop=labels[best]['sort_score']-labels[job['move']]['sort_score'])


def main():
    p=argparse.ArgumentParser();p.add_argument('--pgn',required=True);p.add_argument('--stockfish',required=True)
    p.add_argument('--out',required=True);p.add_argument('--workers',type=int,default=6)
    a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=False);jobs=[];games=[]
    with open(a.pgn) as f:
        while (g:=chess.pgn.read_game(f)) is not None:
            index=len(games);games.append(dict(headers=dict(g.headers)))
            color=g.headers.get('White')=='NeuroChess';b=g.board();initial=b.fen(en_passant='fen');history=[]
            for ply,move in enumerate(g.mainline_moves()):
                if b.turn==color:jobs.append(dict(game=index,ply=ply,fen=b.fen(en_passant='fen'),initial_fen=initial,history=list(history),move=move.uci()))
                b.push(move);history.append(move.uci())
    rows=[]
    with concurrent.futures.ProcessPoolExecutor(a.workers,initializer=initialize,initargs=(a.stockfish,)) as pool:
        for row in pool.map(assess,jobs,chunksize=1):rows.append(row)
    critical=[]
    for i in range(len(games)):
        rs=[r for r in rows if r['game']==i];large=[r for r in rs if r['drop']>=100]
        if large:critical.append(large[0])
        elif rs:critical.append(max(rs,key=lambda r:r['drop']))
    report=dict(args=vars(a),games=games,positions=rows,critical=critical,
        note='Finite teacher search, 50k proposal +100k coherent candidate comparison; mate sort scores are not centipawns.')
    (out/'result.json').write_text(json.dumps(report,indent=2));print(json.dumps([dict(game=r['game'],ply=r['ply'],move=r['move'],best=r['best'],drop=r['drop']) for r in critical]),flush=True)


if __name__=='__main__':main()
