"""Expand local search boundaries and retain CP + sibling MultiPV ordering.

Freeze game assignments before search/annotation. All children of a parent
share one teacher MultiPV search, so their relative labels are coherent.
"""
import argparse
import concurrent.futures
import hashlib
import io
import json
from pathlib import Path
import random
import sqlite3
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import chess
import chess.engine
import chess.pgn
import numpy as np
import zstandard
from neurochess.best_first import BestFirstPlayer
from neurochess.distill import init_engine
from neurochess.encoding import encode, position_key
from neurochess.frontier import FRONTIER_DTYPE
from neurochess.prepare import FilteredBuilder
from neurochess.search_diagnostics import node_board, node_path, tree_children


def digest(path):
    with open(path,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def label(task):
    from neurochess.distill import _ENGINE
    job,nodes=task
    board=chess.Board(job['initial_fen'])
    for move in job['history']:board.push_uci(move)
    if board.is_game_over(claim_draw=True):return None
    infos=_ENGINE.analyse(board,chess.engine.Limit(nodes=nodes),multipv=min(4,board.legal_moves.count()),game=object())
    row=np.zeros((),dtype=FRONTIER_DTYPE)
    row['game']=job['game'];row['node']=job['node'];row['depth']=job['depth']
    root_turn=board.turn
    for slot,info in enumerate([infos[0]]+infos):
        b=board.copy()
        if slot:b.push(info['pv'][0])
        score=info['score'].pov(b.turn)
        row['tokens'][slot],row['meta'][slot]=encode(b)
        row['key'][slot]=np.frombuffer(position_key(b),dtype=np.uint8)
        row['valid'][slot]=True
        if score.is_mate():row['mate'][slot]=score.mate()
        else:row['cp'][slot]=score.score();row['cp_valid'][slot]=True
        w=score.wdl(model='sf16',ply=32);row['wdl'][slot]=(w.wins-w.losses)/1000
    return job['split'],row


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ['pgn','cheap','stockfish','key-db','out']:p.add_argument('--'+arg,required=True)
    p.add_argument('--after-game',type=int,default=39191);p.add_argument('--games',type=int,default=2000)
    p.add_argument('--eval-games',type=int,default=128);p.add_argument('--per-root',type=int,default=16)
    p.add_argument('--min-elo',type=int,default=1600);p.add_argument('--seed',type=int,default=431903)
    p.add_argument('--nodes',type=int,default=20000);p.add_argument('--workers',type=int,default=8)
    a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    if min(a.games,a.eval_games,a.per_root,a.nodes,a.workers)<1:p.error('Positive resources required')
    rng=random.Random(a.seed);roots=[];index=0;start=time.perf_counter()
    db=sqlite3.connect(f'file:{Path(a.key_db).resolve()}?mode=ro',uri=True)
    with open(a.pgn,'rb') as raw,io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(raw)) as stream:
        while len(roots)<a.games+a.eval_games:
            g=chess.pgn.read_game(stream,Visitor=lambda:FilteredBuilder({'min_elo':a.min_elo}))
            if g is None:break
            index+=1
            if index<=a.after_game or g.errors:continue
            b=g.board();choices=[]
            for ply,move in enumerate(g.mainline_moves(),1):
                b.push(move)
                if 16<=ply<=100 and not b.is_game_over(claim_draw=True):choices.append(b.copy())
            rng.shuffle(choices)
            for b in choices:
                if db.execute('select 1 from positions where key=?',(position_key(b),)).fetchone():continue
                roots.append(dict(game=index,fen=b.fen(en_passant='fen'),initial_fen=b.root().fen(en_passant='fen'),
                    history=[m.uci() for m in b.move_stack]));break
    if len(roots)!=a.games+a.eval_games:raise ValueError(f'Source exhausted: {len(roots)} games')
    db.close();rng.shuffle(roots)
    for i,r in enumerate(roots):r['split']='eval' if i<a.eval_games else 'val' if (i-a.eval_games)%5==0 else 'train'
    # Reserve every real trajectory position of evaluation games, not just roots.
    reserved=set()
    for r in roots:
        if r['split']!='eval':continue
        b=chess.Board(r['initial_fen'])
        for m in r['history']:b.push_uci(m);reserved.add(position_key(b))
    eval_roots=[r for r in roots if r['split']=='eval']
    (out/'eval.fen').write_text('\n'.join(r['fen'] for r in eval_roots)+'\n')
    (out/'roots.json').write_text(json.dumps(roots,indent=2))
    (out/'plan.json').write_text(json.dumps(dict(args=vars(a),roots_sha256=digest(out/'roots.json'),
        pgn_sha256=digest(a.pgn),cheap_sha256=digest(a.cheap),stockfish_sha256=digest(a.stockfish),
        split='whole source games; frozen before label/search; no pretrained initialization required',
        base_overlap='Evaluation root exact keys excluded from old base. Whole-source overlap of old frozen large model is unknown.'),indent=2))
    jobs=[];seen_parents=set();player=BestFirstPlayer(a.cheap,expansions=128,max_depth=32)
    for r in roots:
        if r['split']=='eval':continue
        b=chess.Board(r['initial_fen'])
        for move in r['history']:b.push_uci(move)
        player.choose(b);trace=player.trace();nodes=trace['nodes'];ch=tree_children(trace)
        rank=sorted(ch[0],key=lambda i:nodes[i]['value']);selected=[0]+rank[:4]
        for child in rank[:8]:
            i=child
            while ch.get(i):i=min(ch[i],key=lambda k:nodes[k]['value'])
            selected.append(i)
        leaves=[n['id'] for n in nodes if not n['expanded'] and n['solved']==2 and n['depth']>=2]
        rng.shuffle(leaves);selected+=leaves
        count=0
        for i in dict.fromkeys(selected):
            board=node_board(b,trace,i);key=position_key(board)
            if key in seen_parents or key in reserved or board.is_game_over(claim_draw=True):continue
            seen_parents.add(key);jobs.append(dict(game=r['game'],split=r['split'],node=i,depth=nodes[i]['depth'],
                initial_fen=r['initial_fen'],history=r['history']+node_path(trace,i)))
            count+=1
            if count==a.per_root:break
        if len(jobs)%512==0:(out/'status.json').write_text(json.dumps(dict(phase='searching',jobs=len(jobs),seconds=time.perf_counter()-start)))
    player.close();(out/'jobs.json').write_text(json.dumps(jobs))
    buffers={s:[] for s in ['train','val']};counts={s:0 for s in buffers};owners={};cross_dropped=0
    # Validation games first: duplicated effective boards cannot leak into training.
    jobs.sort(key=lambda j:(j['split']!='val',j['game'],j['node']))
    with concurrent.futures.ProcessPoolExecutor(a.workers,initializer=init_engine,initargs=(a.stockfish,1,32)) as pool:
        for offset in range(0,len(jobs),a.workers*16):
            chunk=jobs[offset:offset+a.workers*16]
            for result in pool.map(label,[(j,a.nodes) for j in chunk],chunksize=1):
                if result is None:continue
                split,row=result
                for i in range(5):
                    if not row['valid'][i]:continue
                    key=row['key'][i].tobytes()
                    if key in reserved or (key in owners and owners[key]!=split):
                        row['valid'][i]=False;cross_dropped+=1
                    else:owners[key]=split
                if row['valid'].any():buffers[split].append(row);counts[split]+=int(row['valid'].sum())
            (out/'status.json').write_text(json.dumps(dict(phase='labeling',completed=offset+len(chunk),total=len(jobs),states=counts,seconds=time.perf_counter()-start)))
    for split,rows in buffers.items():np.save(out/f'{split}.npy',np.array(rows,dtype=FRONTIER_DTYPE))
    manifest=dict(schema='rank-frontier-v1',args=vars(a),groups={s:len(v) for s,v in buffers.items()},states=counts,
        games={s:sum(r['split']==s for r in roots) for s in ['train','val','eval']},
        target='state-side raw centipawns (mates separate), sf16 ply32 W-L, same-search MultiPV siblings',
        cp_scale=600,cp_training_clip=3000,history='full played history + search path passed to Stockfish; cheap model masks repetition/clocks',
        split='whole source games; overlapping annotated train/val keys removed; heldout played-prefix keys excluded',
        cross_split_slots_removed=cross_dropped,seconds=time.perf_counter()-start,
        fingerprints={f:digest(out/f) for f in ['train.npy','val.npy','roots.json','jobs.json','eval.fen','plan.json']})
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest),flush=True)


if __name__=='__main__':main()
