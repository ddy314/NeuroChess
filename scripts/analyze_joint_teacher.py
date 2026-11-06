"""Adjudicate every tested decision using the same fresh Stockfish candidate set."""
import concurrent.futures as futures,json,multiprocessing,sys,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import chess,numpy as np
from scripts import joint_diagnostic_teacher as teacher


def write(path,data):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2));tmp.replace(path)


def board_for(j):
    b=chess.Board(j['initial_fen'])
    for u in j['history']:b.push_uci(u)
    return b


def main(out,ready_only=False):
    from neurochess.multihead import fuse_heads
    roots=json.loads((out/'search-jobs.json').read_text())
    if ready_only:
        excluded={j['key'] for label in ['winning','neutral','losing'] for j in [x for x in roots if x['selection']==label][:8]}
        roots=[j for j in roots if j['key'] not in excluded];children=[]
    else:children=json.loads((out/'continuation-jobs.json').read_text())
    for j in children:j.setdefault('case_id',hashlib.sha256((j['initial_fen']+'|'+' '.join(j['history'])).encode()).hexdigest())
    jobs=roots+children
    decisions=[json.loads(l) for l in (out/'search-decisions.jsonl').read_text().splitlines()]
    rows=np.load(out/'new_test.npy');indices={bytes(k).hex():i for i,k in enumerate(rows['key'])};raw=[np.load(out/f'new_test-{label}-raw.npy',mmap_mode='r') for label in ['parent','final']]
    tasklist=[]
    for j in jobs:
        moves=[r['move'] for r in decisions if r['key']==j['key']]
        if j['key'] in indices:
            i=indices[j['key']];b=board_for(j);legal=list(b.legal_moves)
            for r in raw:
                a=r[i:i+1];f,_,_,support=fuse_heads(a[:,:256],a[:,256:512],a[:,512:],rows[i:i+1]['moves']>=0)
                moves.extend(legal[int(k)].uci() for k in np.flatnonzero(support[0]))
                moves.extend([legal[int(f.argmax())].uci(),legal[int(a[0,:256].argmax())].uci()])
        tasklist.append(dict(job=j,moves=list(dict.fromkeys(moves))))
    path=out/'deep-teacher.jsonl';done=set()
    if path.exists():done={json.loads(l)['job'].get('case_id',json.loads(l)['job']['key']) for l in path.read_text().splitlines()}
    tasklist=[t for t in tasklist if t['job'].get('case_id',t['job']['key']) not in done]
    with futures.ProcessPoolExecutor(8,mp_context=multiprocessing.get_context('spawn'),initializer=teacher.initialize,initargs=(str(ROOT/'output/targeted-scale-local-20261102/inputs/stockfish'),)) as pool:
        pending={pool.submit(teacher.adjudicate,t):t for t in tasklist}
        for i,f in enumerate(futures.as_completed(pending),1):
            result=f.result()
            with path.open('a') as stream:stream.write(json.dumps(result)+'\n')
            write(out/'teacher-status.json',dict(phase='adjudicating',completed=len(done)+i,total=len(jobs),updated_at=time.time()))
    write(out/'teacher-status.json',dict(phase='complete',completed=len(jobs),total=len(jobs),updated_at=time.time()))


if __name__=='__main__':main(Path(sys.argv[1]),ready_only='--ready-only' in sys.argv[2:])
