"""Second cloud CPU, disjoint frozen half of the 512k jobs, verified auto-release."""
import fcntl,hashlib,json,os,shutil,sqlite3,sys,threading,time
from pathlib import Path
ROOT=Path('/content/neurochess');WORK=Path('/content/fresh-v4-extra');MOUNT=Path('/content/drive')
SAVE=MOUNT/'MyDrive/neurochess/fresh-v4-teacher-extra';PRIMARY=MOUNT/'MyDrive/neurochess/fresh-v4-preparation'


def main():
    if not os.path.ismount(MOUNT):raise RuntimeError('Command-line Drive mount required')
    os.chdir(ROOT);sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
    from fresh_v4_prepare_cloud import sha,copy
    from fresh_hard_data import annotate,atomic_json
    lock=open('/tmp/neurochess-fresh-v4-extra.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    WORK.mkdir(exist_ok=True);SAVE.mkdir(parents=True,exist_ok=True)
    ready=json.loads((PRIMARY/'jobs-ready.json').read_text());jobs=WORK/'jobs.jsonl';copy(PRIMARY/'jobs.jsonl',jobs)
    if sha(jobs)!=ready['jobs_sha256'] or ready['partitions']!=[[0,256000],[256000,512000]]:raise ValueError('Wrong frozen parallel scope')
    plan=json.loads((ROOT/'configs/fresh-v4.json').read_text());stop=threading.Event()
    def persist():
        path=WORK/'teacher/labels.sqlite'
        if path.exists():
            snap=WORK/'snapshot.sqlite'
            if snap.exists():snap.unlink()
            with sqlite3.connect(path,timeout=60) as source,sqlite3.connect(snap) as dest:source.backup(dest)
            copy(snap,SAVE/'labels.sqlite')
        for name in ['progress.json','spec.json']:
            if (WORK/'teacher'/name).exists():copy(WORK/'teacher'/name,SAVE/name)
        if (WORK/'pipeline-error.json').exists():copy(WORK/'pipeline-error.json',SAVE/'pipeline-error.json')
    def backup():
        while not stop.wait(120):
            try:persist()
            except OSError as error:print('Persistence retry',repr(error),flush=True)
    worker=threading.Thread(target=backup,daemon=True);worker.start()
    try:
        annotate(jobs,str(ROOT/'bin/stockfish'),WORK/'teacher',WORK/'data',plan,256000,512000)
        stop.set();worker.join();persist()
        record=dict(complete=True,jobs_sha256=sha(jobs),range=[256000,512000],labels_sha256=sha(SAVE/'labels.sqlite'),
            annotation_audit=json.loads((WORK/'data/annotation-audit.json').read_text()))
        atomic_json(WORK/'ready.json',record);copy(WORK/'ready.json',SAVE/'ready.json')
    except Exception as error:
        atomic_json(WORK/'pipeline-error.json',dict(error=repr(error),time=time.time()));raise
    finally:
        stop.set();worker.join()
        if not (WORK/'ready.json').exists():persist()
        from neurochess.autostop import request_unassign,find_runtime_address
        request_unassign(find_runtime_address())

if __name__=='__main__':main()
