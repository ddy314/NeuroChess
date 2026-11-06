"""Supplement only: complete, unbounded MultiPV iterations for unresolved references."""
import sys,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.lichess_failure_mechanisms import *

def complete_search(engine,board,moves,nodes):
    m=len(moves);engine.send(f'setoption name MultiPV value {m}\nucinewgame\nisready');list(engine.until('readyok'))
    engine.send('position fen '+board.fen(en_passant='fen'))
    engine.send(f'go nodes {nodes} searchmoves '+' '.join(moves));levels={};complete={}
    for line in engine.until('bestmove '):
        if not line.startswith('info ') or ' score ' not in line or ' pv ' not in line:continue
        head,pv=line.split(' pv ',1);tokens=head.split()
        if 'depth' not in tokens or 'multipv' not in tokens:continue
        d=int(tokens[tokens.index('depth')+1]);k=int(tokens[tokens.index('multipv')+1]);i=tokens.index('score');kind=tokens[i+1];value=int(tokens[i+2])
        row=dict(move=pv.split()[0],cp=value if kind=='cp' else None,mate=value if kind=='mate' else None,
            sort_score=value if kind=='cp' else chess.engine.Mate(value).score(mate_score=100000),depth=d,
            nodes=int(tokens[tokens.index('nodes')+1]),lowerbound='lowerbound' in tokens,upperbound='upperbound' in tokens,pv=pv.split())
        levels.setdefault(d,{})[k]=row
        r=levels[d]
        if set(r)==set(range(1,m+1)) and not any(x['lowerbound'] or x['upperbound'] for x in r.values()) and {x['move'] for x in r.values()}==set(moves):
            complete[d]=[dict(r[i]) for i in range(1,m+1)]
    if not complete:raise RuntimeError('No complete exact MultiPV iteration')
    return complete[max(complete)]

def main():
    if not (OUT/'matched-budget/search-results.json').exists():raise RuntimeError('Matched search incomplete')
    matched=read(OUT/'matched-budget/search-results.json');jobs=read(OUT/'jobs.json');results={}
    for index in FAILURES:
        ref=read(OUT/'teachers'/f'{index:03d}.json');moves=list(dict.fromkeys([r['move'] for r in ref['final']]+[r['move'] for r in matched if r['index']==index]))
        if ref['depth24_exact'] and set(moves)=={r['move'] for r in ref['final']}:
            results[str(index)]=ref;write(OUT/'coherent-teachers'/f'{index:03d}.json',ref);continue
        board=chess.Board(next(j for j in jobs if j['index']==index)['fen']);engine=FastTeacher(ENGINE,threads=1,hash_mb=64,timeout=360)
        try:
            for nodes in [80000000,160000000]:
                rows=complete_search(engine,board,moves,nodes)
                if min(r['depth'] for r in rows)>=24:break
            for r in rows:
                b=board.copy()
                for move in r['pv']:b.push_uci(move)
            result=dict(ref,final=rows,depth24_exact=min(r['depth'] for r in rows)>=24,
                supplemental_protocol='Latest full same-depth exact MultiPV iteration, same SF18/Threads1/Hash64, fixed80M then160M cap; original references preserved',supplemental_requested_nodes=nodes)
            write(OUT/'coherent-teachers'/f'{index:03d}.json',result);results[str(index)]=result
            print(json.dumps(dict(index=index,depth=min(r['depth'] for r in rows),requested_nodes=nodes)),flush=True)
        finally:engine.quit()
    write(OUT/'coherent-teachers/results.json',dict(completed=True,roots=results,training_resumed=False))

if __name__=='__main__':main()
