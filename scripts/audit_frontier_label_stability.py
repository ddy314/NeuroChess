"""Reassess frozen student validation siblings with a larger teacher budget."""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import chess
import chess.engine
import numpy as np
from neurochess.encoding import position_key
from scripts.audit_search_games import initialize


def assess(task):
    from scripts.audit_search_games import ENGINE
    job,old,nodes=task;b=chess.Board(job['initial_fen'])
    for move in job['history']:b.push_uci(move)
    moves={}
    for m in b.legal_moves:
        c=b.copy();c.push(m);moves[position_key(c).hex()]=m
    originals={moves[x['key']].uci():x for x in old}
    infos=ENGINE.analyse(b,chess.engine.Limit(nodes=nodes),root_moves=[b.parse_uci(x) for x in originals],multipv=len(originals),game=object())
    labels={}
    for info in infos:
        s=info['score'].pov(b.turn);labels[info['pv'][0].uci()]=dict(cp=s.score(),mate=s.mate(),depth=info.get('depth'))
    pairs=flips=robust=robust_flips=0;ms=list(originals)
    for i,m in enumerate(ms):
        for other in ms[i+1:]:
            if originals[m]['cp'] is None or originals[other]['cp'] is None or labels[m]['cp'] is None or labels[other]['cp'] is None:continue
            oldgap=originals[m]['cp']-originals[other]['cp'];newgap=labels[m]['cp']-labels[other]['cp']
            if abs(oldgap)<30:continue
            pairs+=1;flips+=oldgap*newgap<0
            if abs(newgap)>=30:robust+=1;robust_flips+=oldgap*newgap<0
    errors=[abs(labels[m]['cp']-v['cp']) for m,v in originals.items() if labels[m]['cp'] is not None and v['cp'] is not None]
    all_cp=all(v['cp'] is not None for v in originals.values()) and all(v['cp'] is not None for v in labels.values())
    top_changed=None
    if all_cp:
        oldbest=max(originals,key=lambda m:originals[m]['cp']);best=max(labels,key=lambda m:labels[m]['cp']);top_changed=oldbest!=best
    return dict(game=job['game'],node=job['node'],fen=b.fen(en_passant='fen'),old=originals,new=labels,
                pairs=pairs,flips=flips,robust_pairs=robust,robust_flips=robust_flips,absolute_cp_changes=errors,top_changed=top_changed)


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--stockfish',required=True);p.add_argument('--out',required=True)
    p.add_argument('--groups',type=int,default=96);p.add_argument('--nodes',type=int,default=500000);p.add_argument('--workers',type=int,default=4);a=p.parse_args()
    data=Path(a.data);val=np.load(data/'val.npy',mmap_mode='r');jobs=json.loads((data/'jobs.json').read_text());lookup={(j['game'],j['node']):j for j in jobs};rng=np.random.default_rng(813903)
    indices=sorted(map(int,rng.choice(len(val),size=a.groups,replace=False)));tasks=[]
    for index in indices:
        row=val[index];old=[dict(key=row['key'][i].tobytes().hex(),cp=-float(row['cp'][i]) if row['cp_valid'][i] else None) for i in range(1,5) if row['valid'][i]]
        tasks.append((lookup[(int(row['game']),int(row['node']))],old,a.nodes))
    with concurrent.futures.ProcessPoolExecutor(a.workers,initializer=initialize,initargs=(a.stockfish,)) as pool:rows=list(pool.map(assess,tasks))
    errors=[e for r in rows for e in r['absolute_cp_changes']];pairs=sum(r['pairs'] for r in rows);flips=sum(r['flips'] for r in rows);robust=sum(r['robust_pairs'] for r in rows);rf=sum(r['robust_flips'] for r in rows);top=[r['top_changed'] for r in rows if r['top_changed'] is not None]
    result=dict(args=vars(a),validation_indices=indices,data_sha256=hashlib.sha256((data/'val.npy').read_bytes()).hexdigest(),
        summary=dict(groups=len(rows),cp_states=len(errors),mean_abs_cp_change=float(np.mean(errors)),median_abs_cp_change=float(np.median(errors)),
                     old_gap30_pairs=pairs,pair_flips=flips,robust_gap30_both_pairs=robust,robust_pair_flips=rf,cp_groups=len(top),best_child_changed=sum(top)),
        rows=rows,note='Random frozen validation groups; original20k full-root MultiPV vs500k restricted same siblings. Larger finite search is a stability probe, not ground truth. No new labels used for fitting or model selection.')
    Path(a.out).write_text(json.dumps(result,indent=2));print(json.dumps(result['summary'],indent=2))

if __name__=='__main__':main()
