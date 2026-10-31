"""Cloud CPU preparation: novel puzzle selection, teacher labels and verified delivery."""
import fcntl,hashlib,json,os,shutil,sqlite3,subprocess,sys,tarfile,threading,time,urllib.request
from pathlib import Path
ROOT=Path('/content/neurochess');WORK=Path('/content/fresh-v4');MOUNT=Path('/content/drive')
SAVE=MOUNT/'MyDrive/neurochess/fresh-v4-preparation';DELIVERY=MOUNT/'MyDrive/neurochess-data/fresh-v4-500k'


def sha(path):
    with open(path,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def copy(source,target):
    target=Path(target);target.parent.mkdir(parents=True,exist_ok=True)
    temp=target.with_name(target.name+'.tmp');shutil.copyfile(source,temp)
    if sha(source)!=sha(temp):raise OSError('Drive readback failed: '+str(target))
    temp.replace(target)


def download(url,path):
    path=Path(path)
    if path.exists():return
    temp=path.with_name(path.name+'.download')
    with urllib.request.urlopen(url,timeout=120) as response,temp.open('wb') as stream:
        meta=dict(url=url,headers={k:response.headers.get(k) for k in ['ETag','Last-Modified','Content-Length']})
        shutil.copyfileobj(response,stream,1024*1024)
    temp.replace(path);meta.update(sha256=sha(path),bytes=path.stat().st_size)
    path.with_suffix(path.suffix+'.source.json').write_text(json.dumps(meta,indent=2))


def main():
    if not os.path.ismount(MOUNT):raise RuntimeError('Command-line Drive mount required')
    lock=open('/tmp/neurochess-fresh-v4-prepare.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    os.chdir(ROOT);sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
    from fresh_hard_data import make_exclusions,collect,annotate,atomic_json,export
    WORK.mkdir(exist_ok=True);SAVE.mkdir(parents=True,exist_ok=True)
    plan=json.loads((ROOT/'configs/fresh-v4.json').read_text());copy(ROOT/'configs/fresh-v4.json',SAVE/'plan.json')
    copy(ROOT/'fresh-v4-code.tar.gz',SAVE/'code.tar.gz')
    state={'phase':'copying-old-data','updated_at':time.time()};stop=threading.Event();failure=None
    def persist():
        atomic_json(WORK/'pipeline-status.json',dict(state,updated_at=time.time()))
        for name in ['pipeline-status.json','selection-progress.json','selection-audit.json','jobs.jsonl','pipeline-error.json','opening-library.json','ready.json']:
            if (WORK/name).exists():copy(WORK/name,SAVE/name)
        label_work=WORK/'teacher'
        for name in ['progress.json','spec.json']:
            if (label_work/name).exists():copy(label_work/name,SAVE/name)
        if (label_work/'labels.sqlite').exists():
            snapshot=WORK/'labels-backup.sqlite'
            if snapshot.exists():snapshot.unlink()
            with sqlite3.connect(label_work/'labels.sqlite',timeout=60) as db,sqlite3.connect(snapshot) as target:db.backup(target)
            copy(snapshot,SAVE/'labels.sqlite')
    def replicate():
        last_db=0
        while not stop.wait(30):
            try:
                atomic_json(WORK/'pipeline-status.json',dict(state,updated_at=time.time()))
                for src in [WORK/'pipeline-status.json',WORK/'selection-progress.json',WORK/'selection-audit.json',WORK/'teacher/progress.json']:
                    if src.exists():copy(src,SAVE/src.name)
                if time.time()-last_db>300:persist();last_db=time.time()
            except Exception as error:print('Persistence retry',repr(error),flush=True)
    worker=threading.Thread(target=replicate,daemon=True);worker.start()
    try:
        cache=MOUNT/'MyDrive/neurochess-data/evals-v2-5m.tar.gz';core=ROOT/'data/evals-v2-5m';core.parent.mkdir(exist_ok=True)
        if not (core/'manifest.json').exists():
            local_cache=WORK/'core.tar.gz';copy(cache,local_cache)
            if sha(local_cache)!='a0b5dbef770fcf2b7b4bc31a709f4a2b8379421965aa591f69291a6abbc7c2ef':raise ValueError('Core cache SHA differs')
            core.mkdir(exist_ok=True)
            with tarfile.open(local_cache) as archive:archive.extractall(core,filter='data')
        replay=ROOT/'data/human-replay'
        if not (replay/'manifest.json').exists():shutil.copytree(MOUNT/'MyDrive/neurochess-data/elite-recent-2026q3',replay,dirs_exist_ok=True)
        # Validate rows/shard counts before claiming old-data novelty.
        import numpy as np
        for root in [core,replay]:
            actual={s:sum(len(np.load(f,mmap_mode='r')) for f in root.glob(s+'-*.npy')) for s in ['train','val']}
            if actual!=json.loads((root/'manifest.json').read_text())['counts']:raise ValueError('Incomplete old-data exclusions')
        state['phase']='downloading-puzzle-library';source=WORK/'lichess_db_puzzle.csv.zst'
        download(plan['sources']['puzzles'],source)
        state['phase']='indexing-old-core-and-human-keys';old=make_exclusions([core,replay],WORK/'old-keys.sqlite')
        # All old frozen reference/arena roots are excluded, including clocks-free identity.
        reserved=[]
        for file in (ROOT/'fresh-v4-assets').glob('*.fen'):reserved.extend(file.read_text().splitlines())
        state['phase']='selecting-512000-novel-positions';jobs=collect(source,plan,old,WORK,reserved);old.close()
        copy(jobs,SAVE/'jobs.jsonl');copy(WORK/'selection-audit.json',SAVE/'selection-audit.json')
        atomic_json(WORK/'jobs-ready.json',dict(jobs_sha256=sha(jobs),total=512000,partitions=[[0,256000],[256000,512000]]))
        copy(WORK/'jobs-ready.json',SAVE/'jobs-ready.json')
        # Keep curated opening sequences as a separate policy-only supplement.
        state['phase']='preparing-opening-book';book=[]
        from fresh_hard_data import group_split
        import csv,io,chess,chess.pgn
        for volume in 'abcde':
            target=WORK/f'opening-{volume}.tsv';download(f'https://raw.githubusercontent.com/lichess-org/chess-openings/master/{volume}.tsv',target)
            copy(target,SAVE/target.name)
            for row in csv.DictReader(io.StringIO(target.read_text()),delimiter='\t'):
                game=chess.pgn.read_game(io.StringIO(row['pgn']))
                if game is None or game.errors:raise ValueError('Invalid official opening line')
                book.append(dict(eco=row['eco'],name=row['name'],moves=[m.uci() for m in game.mainline_moves()]))
        atomic_json(WORK/'opening-library.json',dict(source=plan['sources']['openings'],license='CC0',lines=book,
            usage='Policy-only supplement, shared in both arms; named book continuations are not unique best moves.'))
        copy(WORK/'opening-library.json',SAVE/'opening-library.json')
        state['phase']='stockfish-annotation-primary';annotate(jobs,str(ROOT/'bin/stockfish'),WORK/'teacher',WORK/'primary-data',plan,0,256000)
        state['phase']='waiting-for-extra-cpu-delivery'
        helper=MOUNT/'MyDrive/neurochess/fresh-v4-teacher-extra';deadline=time.time()+4*3600
        while not (helper/'ready.json').exists():
            if time.time()>deadline:raise RuntimeError('Second teacher delivery timeout; primary labels preserved')
            time.sleep(30)
        delivery=json.loads((helper/'ready.json').read_text())
        if delivery['jobs_sha256']!=sha(jobs) or delivery['range']!=[256000,512000]:raise ValueError('Second teacher scope differs')
        other=WORK/'extra-labels.sqlite';copy(helper/'labels.sqlite',other)
        if sha(other)!=delivery['labels_sha256']:raise ValueError('Second teacher labels SHA differs')
        own_spec=json.loads((WORK/'teacher/spec.json').read_text());helper_spec=json.loads((helper/'spec.json').read_text())
        for field in ['jobs_sha256','engine_sha256','target_spec','nodes','coarse_nodes']:
            if own_spec[field]!=helper_spec[field]:raise ValueError('Parallel teacher label contract differs: '+field)
        all_jobs=[json.loads(line) for line in jobs.read_text().splitlines()]
        with sqlite3.connect(WORK/'teacher/labels.sqlite') as db,sqlite3.connect(other) as helper_db:
            own={r[0] for r in db.execute('SELECT key FROM labels')};extra={r[0] for r in helper_db.execute('SELECT key FROM labels')}
            primary_keys={j['key'] for j in all_jobs[:256000]};extra_keys={j['key'] for j in all_jobs[256000:]}
            if extra!=extra_keys or own not in (primary_keys,primary_keys|extra_keys):raise ValueError('Parallel coverage differs')
            if own==primary_keys:
                if own&extra:raise ValueError('Parallel labels overlap')
                db.execute('ATTACH DATABASE ? AS helper',(str(other),));db.execute('INSERT INTO labels SELECT * FROM helper.labels');db.commit();db.execute('DETACH DATABASE helper')
            spec=json.loads((WORK/'teacher/spec.json').read_text());spec.update(start=0,end=512000,parallel_workers=16,source_partitions=[[0,256000],[256000,512000]])
            if not (WORK/'data/annotation-audit.json').exists():export(db,WORK/'data',spec)
        proof=json.loads((WORK/'data/annotation-audit.json').read_text())
        if proof['trainval']['counts']!={'train':500000,'val':10000} or proof['test']['counts']!={'train':0,'val':2000}:raise ValueError('Final dataset counts differ')
        state['phase']='packing-and-verifying-delivery';archive=WORK/'fresh-v4-500k.tar.gz'
        with tarfile.open(archive,'w:gz',compresslevel=1) as tar:tar.add(WORK/'data',arcname='data');tar.add(jobs,arcname='jobs.jsonl');tar.add(WORK/'opening-library.json',arcname='opening-library.json');tar.add(WORK/'selection-audit.json',arcname='selection-audit.json')
        copy(archive,Path(str(DELIVERY)+'.tar.gz'))
        atomic_json(WORK/'ready.json',dict(complete=True,counts=proof,archive_sha256=sha(archive),plan=plan,created_at=time.time()))
        copy(WORK/'ready.json',Path(str(DELIVERY)+'-ready.json'));state['phase']='complete'
    except Exception as error:
        failure=error;state['phase']='failed';atomic_json(WORK/'pipeline-error.json',dict(error=repr(error),time=time.time()))
        raise
    finally:
        stop.set();worker.join();persist()
        # Data/code/debug state now survived a byte-for-byte readback. No idle CPU rent.
        from neurochess.autostop import request_unassign,find_runtime_address
        request_unassign(find_runtime_address())

if __name__=='__main__':main()
