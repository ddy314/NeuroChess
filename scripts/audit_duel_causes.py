"""First teacher-assessed errors from completed duels, preserving game history."""
import argparse
import concurrent.futures
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import chess
import chess.pgn
from scripts.audit_search_games import initialize,assess


def inspect(job):
    path,contender,index=job
    with open(path) as f:game=chess.pgn.read_game(f)
    board=game.board();initial=board.fen(en_passant='fen');history=[];best=None;checked=0
    color=game.headers['White']==contender
    for ply,move in enumerate(game.mainline_moves()):
        if board.turn==color:
            row=assess(dict(game=index,source=path,ply=ply,fen=board.fen(en_passant='fen'),
                            initial_fen=initial,history=list(history),move=move.uci()))
            checked+=1
            if best is None or row['drop']>best['drop']:best=row
            if row['drop']>=100:break
        board.push(move);history.append(move.uci())
    return dict(critical=best,checked=checked)


def main():
    p=argparse.ArgumentParser()
    for key in ['duel','contender','stockfish','out']:p.add_argument('--'+key,required=True)
    p.add_argument('--games',type=int,default=16);p.add_argument('--workers',type=int,default=4)
    a=p.parse_args();paths=sorted(Path(a.duel).glob('*/*.pgn'))[:a.games];rows=[]
    with concurrent.futures.ProcessPoolExecutor(a.workers,initializer=initialize,initargs=(a.stockfish,)) as pool:
        for row in pool.map(inspect,[(str(f),a.contender,i) for i,f in enumerate(paths)]):
            rows.append(row);print(json.dumps(dict(game=len(rows)-1,checked=row['checked'],drop=row['critical']['drop'])),flush=True)
    Path(a.out).write_text(json.dumps(dict(args=vars(a),critical=[r['critical'] for r in rows],
        checked=sum(r['checked'] for r in rows),note='Development failures from old games; sequential first >=100 CP candidate-union error per game. Finite SF teacher, mate sort scores are not CP.'),indent=2))

if __name__=='__main__':main()
