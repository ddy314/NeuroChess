"""Same-task teacher transport/CPU benchmark, with exact output parity checks."""
import argparse,concurrent.futures as futures,hashlib,json,multiprocessing,sqlite3,sys,time,threading
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import local_teacher_worker as teacher


def initialize(binary,backend):
    if backend=='legacy':teacher.initialize(binary)
    else:
        from neurochess.teacher_uci import FastTeacher
        teacher.ENGINE=FastTeacher(binary);threading._register_atexit(teacher.ENGINE.quit)


def main(args):
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    if args.tasks:tasks=json.loads(Path(args.tasks).read_text())
    else:
        con=sqlite3.connect(f'file:{Path(args.source).resolve()}?mode=ro',uri=True)
        samples=[json.loads(r[0]) for r in con.execute('SELECT metadata FROM results WHERE screened=1 ORDER BY key LIMIT ?', (args.samples,))];con.close()
        tasks=[(r['job'],r['prediction']) for r in samples]
    (out/'tasks.json').write_text(json.dumps(tasks))
    results=[]
    for backend in args.backends.split(','):
        for workers in map(int,args.workers.split(',')):
            started=time.monotonic()
            with futures.ProcessPoolExecutor(workers,mp_context=multiprocessing.get_context('spawn'),initializer=initialize,initargs=(args.binary,backend)) as pool:
                labels=list(pool.map(teacher.label,tasks,chunksize=1))
            elapsed=time.monotonic()-started
            semantic=[dict(key=m['job']['key'],teacher=m['teacher'],stable=m['stable'],decision=m['decision']) for m in labels]
            digest=hashlib.sha256(json.dumps(semantic,sort_keys=True).encode()).hexdigest()
            row=dict(backend=backend,workers=workers,n=len(tasks),seconds=elapsed,positions_per_second=len(tasks)/elapsed,output_sha256=digest,mistakes=sum(m['repair'] for m in labels),unstable=sum(m['stable'] is False for m in labels))
            results.append(row);print(json.dumps(row),flush=True)
            (out/f'{backend}-{workers}-labels.json').write_text(json.dumps(labels))
            (out/'benchmark.json').write_text(json.dumps(dict(protocol='Identical frozen tasks, SF18 Hash64 Threads1; independent ucinewgame for every10k/30k/100k/300k stage; includes worker start/stop',binary=args.binary,results=results,identical_outputs=len({r['output_sha256'] for r in results})==1),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',default='output/targeted-scale-local-20261102/annotation.sqlite3');p.add_argument('--tasks');p.add_argument('--binary',default='output/targeted-scale-local-20261102/inputs/stockfish');p.add_argument('--out',required=True);p.add_argument('--samples',type=int,default=256);p.add_argument('--workers',default='12');p.add_argument('--backends',default='legacy,fast');main(p.parse_args())
