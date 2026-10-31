"""Resumable CPU teacher annotation with a single coherent candidate search."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import argparse,concurrent.futures,hashlib,json,multiprocessing,os,sqlite3,time,random
import chess,chess.engine
import numpy as np
from neurochess.data import record,ShardWriter,DTYPE
from neurochess.encoding import split_for,position_key
from neurochess.targets import assign_targets,TARGET_SPEC
from neurochess.audit import audit

ENGINE=None


def order_jobs(jobs):
    train=sorted((j for j in jobs if j['split']=='train'),key=lambda j:j['key'])
    val=sorted((j for j in jobs if j['split']=='val'),key=lambda j:j['key'])
    rng=random.Random(20261031);rng.shuffle(train);rng.shuffle(val);result=[]
    for i in range(0,len(train),100):
        result.extend(train[i:i+100])
        if i//100<len(val):result.append(val[i//100])
    return result


def initialize(engine):
    global ENGINE
    ENGINE=chess.engine.SimpleEngine.popen_uci(engine,timeout=90)
    ENGINE.configure({'Threads':1,'Hash':64,'Skill Level':20,'UCI_LimitStrength':False})


def annotate(task):
    job,nodes=task;b=chess.Board(job['fen'])
    if position_key(b).hex()!=job['key'] or split_for(b)!=job['split']:raise ValueError('Job identity/split mismatch')
    coarse=ENGINE.analyse(b,chess.engine.Limit(nodes=20000),multipv=min(4,b.legal_moves.count()),game=object())
    candidates=list(dict.fromkeys([x['pv'][0] for x in coarse]+[b.parse_uci(m) for m in job['student_candidates']]))
    # All labels in this row are from this one fixed-budget coherent search.
    infos=ENGINE.analyse(b,chess.engine.Limit(nodes=nodes),root_moves=candidates,multipv=len(candidates),game=object())
    row,legal=record(b);pvs=[]
    for info in infos:
        score=info['score'].white();pv=dict(line=info['pv'][0].uci())
        pv['mate' if score.is_mate() else 'cp']=score.mate() if score.is_mate() else score.score()
        pvs.append(pv)
    assign_targets(row,b,legal,pvs,TARGET_SPEC['temperature_cp'])
    return job['key'],job['split'],row.tobytes(),json.dumps(dict(candidates=len(candidates),depths=[i.get('depth') for i in infos],pvs=pvs))


def annotate_chunk(item):
    tasks,engine=item
    initialize(engine)
    try:return [annotate(task) for task in tasks]
    finally:ENGINE.quit()


def main():
    p=argparse.ArgumentParser();p.add_argument('--jobs',required=True);p.add_argument('--stockfish',required=True)
    p.add_argument('--work',required=True);p.add_argument('--out',required=True)
    p.add_argument('--workers',type=int,default=12);p.add_argument('--nodes',type=int,default=200000)
    p.add_argument('--limit',type=int);p.add_argument('--curriculum-order',action='store_true')
    p.add_argument('--start-index',type=int,default=0);p.add_argument('--end-index',type=int)
    a=p.parse_args();work=Path(a.work);work.mkdir(parents=True,exist_ok=True)
    output=Path(a.out)
    if output.exists():raise ValueError('Completed dataset already exists')
    jobs=[json.loads(l) for l in Path(a.jobs).read_text().splitlines()]
    if a.curriculum_order:jobs=order_jobs(jobs)
    if a.limit:jobs=jobs[:a.limit]
    scope=dict(start=a.start_index,end=len(jobs) if a.end_index is None else a.end_index,total=len(jobs))
    if not 0<=scope['start']<scope['end']<=len(jobs):raise ValueError('Invalid annotation scope')
    scope_file=work/'scope.json'
    if scope_file.exists() and json.loads(scope_file.read_text())!=scope:raise ValueError('Resume scope changed')
    scope_file.write_text(json.dumps(scope,indent=2))
    jobs=jobs[scope['start']:scope['end']]
    db=sqlite3.connect(work/'labels.sqlite');db.execute('PRAGMA journal_mode=WAL')
    db.execute('CREATE TABLE IF NOT EXISTS labels(key TEXT PRIMARY KEY,split TEXT,payload BLOB,metadata TEXT)')
    signature=dict(jobs_sha256=hashlib.file_digest(open(a.jobs,'rb'),'sha256').hexdigest(),nodes=a.nodes,
        engine_sha256=hashlib.file_digest(open(a.stockfish,'rb'),'sha256').hexdigest(),target_spec=TARGET_SPEC,limit=a.limit,curriculum_order=a.curriculum_order)
    spec=work/'spec.json'
    if spec.exists() and json.loads(spec.read_text())!=signature:raise ValueError('Resume specification changed')
    spec.write_text(json.dumps(signature,indent=2))
    done={r[0] for r in db.execute('SELECT key FROM labels')};pending=[j for j in jobs if j['key'] not in done]
    if not done.issubset({j['key'] for j in jobs}):raise ValueError('Existing labels fall outside selected scope')
    start=time.time();completed=len(done);last=0
    with concurrent.futures.ProcessPoolExecutor(a.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        for offset in range(0,len(pending),a.workers*32):
            tasks=[(j,a.nodes) for j in pending[offset:offset+a.workers*32]]
            chunks=[(tasks[i:i+32],a.stockfish) for i in range(0,len(tasks),32)]
            for results in pool.map(annotate_chunk,chunks,chunksize=1):
                for key,split,payload,metadata in results:
                    db.execute('INSERT INTO labels VALUES (?,?,?,?)',(key,split,payload,metadata));completed+=1
            db.commit()
            seconds=time.time()-start;rate=(completed-len(done))/max(1,seconds)
            info=dict(phase='annotating',completed=completed,total=len(jobs),seconds=seconds,positions_per_second=rate,
                      eta_hours=(len(jobs)-completed)/max(.001,rate)/3600)
            (work/'progress.json').write_text(json.dumps(info,indent=2))
            if seconds-last>=15:print(json.dumps(info),flush=True);last=seconds
    writer=ShardWriter(output.with_name(output.name+'.building'))
    for key,split,payload in db.execute('SELECT key,split,payload FROM labels ORDER BY key'):
        row=np.frombuffer(payload,dtype=DTYPE)[0];writer.add(split,row)
    writer.close(dict(kind='targeted-stockfish-distillation',**signature,teacher_coarse_nodes=20000,
        teacher_root_candidates=4,student_candidates=5,coherent_search=True,reannotation_of_existing_core=True))
    result=audit(writer.root);(writer.root/'audit.json').write_text(json.dumps(result,indent=2))
    writer.root.replace(output);db.close()
    (work/'progress.json').write_text(json.dumps(dict(phase='complete',counts=result['counts'],seconds=time.time()-start),indent=2))
    print(json.dumps(dict(phase='complete',counts=result['counts'],seconds=time.time()-start)),flush=True)


if __name__=='__main__':main()
