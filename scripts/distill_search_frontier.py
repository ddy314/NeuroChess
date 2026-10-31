"""Generate grouped CPU-search frontier examples, then label with local Stockfish.

One sampled root per PGN game; augmentation train/val games are disjoint. Entire
games containing heldout roots are excluded, as are all exact heldout keys.
"""
import argparse
import concurrent.futures
import io
import json
from pathlib import Path
import random
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import chess
import chess.pgn
import zstandard
from neurochess.best_first import BestFirstPlayer
from neurochess.data import ShardWriter
from neurochess.distill import init_engine,annotate
from neurochess.encoding import position_key
from neurochess.prepare import FilteredBuilder
from neurochess.search_diagnostics import node_board,node_path,tree_children


def main():
    p=argparse.ArgumentParser();p.add_argument('--pgn',required=True);p.add_argument('--cheap',required=True)
    p.add_argument('--exclude',required=True);p.add_argument('--stockfish',required=True);p.add_argument('--out',required=True)
    p.add_argument('--roots',type=int,default=160);p.add_argument('--per-root',type=int,default=16)
    p.add_argument('--workers',type=int,default=6);p.add_argument('--nodes',type=int,default=50000)
    a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=False);rng=random.Random(6193171)
    excluded={position_key(chess.Board(s)) for s in Path(a.exclude).read_text().splitlines() if s.strip()}
    raw=open(a.pgn,'rb');stream=zstandard.ZstdDecompressor().stream_reader(raw) if a.pgn.endswith('.zst') else raw
    roots=[];scanned=0;skipped=0
    with io.TextIOWrapper(stream) as f:
        while len(roots)<a.roots:
            game=chess.pgn.read_game(f,Visitor=lambda:FilteredBuilder({'min_elo':2000}))
            if game is None:break
            scanned+=1
            if game.errors or len(list(game.mainline_moves()))<16:continue
            b=game.board();candidates=[];related=False
            for ply,move in enumerate(game.mainline_moves(),1):
                b.push(move);key=position_key(b)
                if key in excluded:related=True
                if 16<=ply<=100 and not b.is_game_over(claim_draw=True):candidates.append(b.copy())
            if related:skipped+=1;continue
            if candidates:
                roots.append(dict(game=scanned,fen=rng.choice(candidates).fen(en_passant='fen'),
                                  split='val' if len(roots)%5==0 else 'train'))
    (out/'roots.json').write_text(json.dumps(roots,indent=2));jobs=[];seen=set()
    player=BestFirstPlayer(a.cheap,expansions=128,max_depth=32)
    for root in roots:
        b=chess.Board(root['fen']);player.choose(b);trace=player.trace();nodes=trace['nodes'];ch=tree_children(trace)
        rank=sorted(ch[0],key=lambda i:nodes[i]['value']);selected=[0]+rank[:4]
        for child in rank[:8]:
            i=child
            while ch.get(i):i=min(ch[i],key=lambda k:nodes[k]['value'])
            selected.append(i)
        leaves=[n['id'] for n in nodes if not n['expanded'] and n['solved']==2 and n['depth']>=2]
        rng.shuffle(leaves);selected+=leaves[:a.per_root]
        count=0
        for i in dict.fromkeys(selected):
            board=node_board(b,trace,i);key=position_key(board)
            if key in seen or key in excluded or board.is_game_over(claim_draw=True):continue
            seen.add(key);jobs.append(dict(fen=board.fen(en_passant='fen'),key=key.hex(),game=root['game'],split=root['split'],depth=nodes[i]['depth'],path=node_path(trace,i)))
            count+=1
            if count>=a.per_root:break
    player.close();(out/'jobs.json').write_text(json.dumps(jobs,indent=2))
    writer=ShardWriter(out/'data');start=time.perf_counter()
    try:
        with concurrent.futures.ProcessPoolExecutor(a.workers,initializer=init_engine,
                initargs=(a.stockfish,1,32)) as pool:
            for offset in range(0,len(jobs),a.workers*8):
                chunk=jobs[offset:offset+a.workers*8]
                results=pool.map(annotate,[(j['fen'],a.nodes,1,100) for j in chunk],chunksize=1)
                for j,r in zip(chunk,results):
                    if r is not None:writer.add(j['split'],r[1])
                status=dict(completed=min(offset+len(chunk),len(jobs)),total=len(jobs),seconds=time.perf_counter()-start)
                (out/'status.json').write_text(json.dumps(status,indent=2))
        writer.close(dict(kind='student-search-frontier',args=vars(a),roots=len(roots),
            scanned_games=scanned,excluded_related_games=skipped,split='one root per game; train/val game groups disjoint',
            exact_keys_deduplicated=True,teacher_nodes=a.nodes,teacher_value='searched sf16 W-L proxy'))
    finally:raw.close()
    print(json.dumps(dict(counts=writer.counts,seconds=time.perf_counter()-start,excluded_related_games=skipped)),flush=True)


if __name__=='__main__':main()
