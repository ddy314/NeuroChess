"""CPU teacher: annotate locally on VM, publish immutable audited waves to Drive."""
import fcntl,hashlib,json,os,shutil,sqlite3,subprocess,sys,tarfile,time
from pathlib import Path
EXTRA=os.environ.get('TEACHER_ROLE')=='extra'
ROOT=Path('/content/neurochess'); WORK=Path('/content/teacher-v3-extra' if EXTRA else '/content/teacher-v3')
DRIVE=Path('/content/drive/MyDrive'); SAVE=DRIVE/('neurochess/curriculum-v3-teacher-extra' if EXTRA else 'neurochess/curriculum-v3-teacher')
HELPER_SAVE=DRIVE/'neurochess/curriculum-v3-teacher-extra'

def sha(p):
    with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def copy(src,dst):
    dst.parent.mkdir(parents=True,exist_ok=True);tmp=dst.with_name(dst.name+'.tmp');shutil.copyfile(src,tmp)
    if sha(src)!=sha(tmp):raise OSError('Persistent readback failed')
    tmp.replace(dst)

def publish(db,jobs,n,train_per_wave=50000,val_per_wave=500):
    import numpy as np
    from neurochess.data import ShardWriter,DTYPE
    from neurochess.targets import TARGET_SPEC
    from neurochess.audit import audit
    marker=DRIVE/f'neurochess-data/hard-v3-wave-{n}-ready.json'
    if marker.exists():
        ready=json.loads(marker.read_text());archive=Path(str(marker).replace('-ready.json','.tar.gz'))
        if ready['counts']!={'train':train_per_wave*n,'val':val_per_wave*n} or sha(archive)!=ready['sha256']:raise ValueError('Existing wave verification failed')
        return True
    selected=jobs[:(train_per_wave+val_per_wave)*n];keys={j['key'] for j in selected}
    if not keys.issubset({r[0] for r in db.execute('SELECT key FROM labels')}):return False
    rows=list(db.execute('SELECT key,split,payload FROM labels'))
    dest=WORK/f'wave-{n}'
    if dest.exists():shutil.rmtree(dest)
    writer=ShardWriter(dest)
    for key,split,payload in sorted(rows):
        if key in keys:writer.add(split,np.frombuffer(payload,dtype=DTYPE)[0])
    writer.close(dict(kind='targeted-stockfish-distillation',target_spec=TARGET_SPEC,teacher_nodes=200000,
        teacher_coarse_nodes=20000,reannotation_of_existing_core=True,wave=n,source_jobs_sha256=sha(ROOT/'jobs.jsonl')))
    proof=audit(dest)
    if proof['counts']!={'train':train_per_wave*n,'val':val_per_wave*n} or proof['within_split_duplicates']:raise ValueError('Invalid wave')
    (dest/'audit.json').write_text(json.dumps(proof,indent=2))
    archive=WORK/f'hard-v3-wave-{n}.tar.gz'
    with tarfile.open(archive,'w:gz',compresslevel=1) as tar:
        for f in sorted(dest.iterdir()):tar.add(f,arcname=f.name)
    target=DRIVE/'neurochess-data'/archive.name;copy(archive,target)
    ready=dict(wave=n,sha256=sha(target),counts=proof['counts'],target_spec=TARGET_SPEC)
    temporary=WORK/'ready.json';temporary.write_text(json.dumps(ready,indent=2));copy(temporary,marker)
    print(json.dumps(dict(published=ready)),flush=True);return True

def merge_labels(db,other,expected_keys):
    with sqlite3.connect(other) as source:
        actual={r[0] for r in source.execute('SELECT key FROM labels')}
    own={r[0] for r in db.execute('SELECT key FROM labels')}
    if actual!=set(expected_keys):raise ValueError('Helper scope differs from the assigned jobs')
    if actual&own:raise ValueError('Helper and primary labels overlap')
    db.execute('ATTACH DATABASE ? AS helper',(str(other),))
    try:
        db.execute('INSERT INTO labels SELECT * FROM helper.labels');db.commit()
    finally:db.execute('DETACH DATABASE helper')


def main():
    if not os.path.ismount('/content/drive'):raise RuntimeError('Drive is not mounted')
    os.chdir(ROOT);sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
    from distill_hard_positions import order_jobs
    lock=open('/tmp/neurochess-teacher.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    WORK.mkdir(exist_ok=True);SAVE.mkdir(parents=True,exist_ok=True);work=WORK/'work';work.mkdir(exist_ok=True)
    for name in ['labels.sqlite','spec.json','scope.json']:
        if not (work/name).exists() and (SAVE/name).exists():copy(SAVE/name,work/name)
    copy(ROOT/'jobs.jsonl',SAVE/'jobs.jsonl');copy(ROOT/'teacher-kit.tar.gz',SAVE/'teacher-kit.tar.gz')
    copy(Path(__file__),SAVE/'teacher_colab.py')
    copy(ROOT/'scripts/distill_hard_positions.py',SAVE/'distill_hard_positions.py')
    jobs=order_jobs([json.loads(x) for x in (ROOT/'jobs.jsonl').read_text().splitlines()])
    if len(jobs)!=202000:raise ValueError('Expected 202000 jobs')
    start,end=(101000,202000) if EXTRA else (0,101000)
    plan=WORK/'parallel-plan.json';plan.write_text(json.dumps(dict(role='extra' if EXTRA else 'primary',start=start,end=end,
        total=202000,workers=8,teacher_nodes=200000,coarse_nodes=20000),indent=2));copy(plan,SAVE/plan.name)
    complete=WORK/('complete-extra' if EXTRA else 'complete-primary');child=None
    if not complete.exists():
        with (WORK/'console.log').open('a') as log:
            child=subprocess.Popen([sys.executable,'-u','scripts/distill_hard_positions.py','--jobs','jobs.jsonl','--stockfish','bin/stockfish',
                '--work',str(work),'--out',str(complete),'--workers','8','--nodes','200000','--curriculum-order',
                '--start-index',str(start),'--end-index',str(end)],stdout=log,stderr=subprocess.STDOUT)
    last_backup=0;published=set();finished_backed=False;wait_started=None;merged=False
    while True:
        finished=child is None or child.poll() is not None
        if finished and child is not None and child.returncode:raise RuntimeError('Teacher worker failed; preserving labels for recovery')
        if (work/'labels.sqlite').exists():
            with sqlite3.connect(work/'labels.sqlite') as db:
                if not EXTRA:
                    if finished and not merged:
                        marker=HELPER_SAVE/'ready.json'
                        if marker.exists():
                            ready=json.loads(marker.read_text());other=WORK/'helper-labels.sqlite'
                            if ready['range']!=[101000,202000] or ready['counts']!={'train':100000,'val':1000}:raise ValueError('Helper delivery scope differs')
                            if json.loads((HELPER_SAVE/'spec.json').read_text())!=json.loads((work/'spec.json').read_text()):raise ValueError('Teacher label specifications differ')
                            copy(HELPER_SAVE/'labels.sqlite',other)
                            if sha(other)!=ready['labels_sha256']:raise ValueError('Helper database hash differs')
                            own={r[0] for r in db.execute('SELECT key FROM labels')}
                            if own!={j['key'] for j in jobs}:merge_labels(db,other,{j['key'] for j in jobs[101000:]})
                            if {r[0] for r in db.execute('SELECT key FROM labels')}!={j['key'] for j in jobs}:raise ValueError('Combined teacher coverage differs')
                            merged=True;last_backup=0;copy(marker,SAVE/'helper-delivery.json')
                        else:
                            if wait_started is None:wait_started=time.time()
                            if time.time()-wait_started>4*3600:raise RuntimeError('Helper delivery timed out')
                    for n in range(1,5):
                        if n not in published and publish(db,jobs,n):published.add(n)
                if time.time()-last_backup>300 or (finished and not finished_backed):
                    snapshot=WORK/'backup.sqlite'
                    if snapshot.exists():snapshot.unlink()
                    with sqlite3.connect(snapshot) as target:db.backup(target)
                    copy(snapshot,SAVE/'labels.sqlite');last_backup=time.time();finished_backed=finished
                    for name in ['spec.json','scope.json','progress.json']:
                        if (work/name).exists():copy(work/name,SAVE/name)
                    copy(WORK/'console.log',SAVE/'console.log')
        if finished and (EXTRA or (merged and published=={1,2,3,4})):break
        time.sleep(30)
    proof=WORK/'completion.json';record=dict(complete=True,waves=sorted(published),labels_sha256=sha(SAVE/'labels.sqlite'))
    if EXTRA:
        record.update(range=[101000,202000],counts={'train':100000,'val':1000})
        if json.loads((complete/'audit.json').read_text())['counts']!=record['counts']:raise ValueError('Helper audit count differs')
    proof.write_text(json.dumps(record,indent=2));copy(proof,SAVE/proof.name)
    if EXTRA:copy(proof,SAVE/'ready.json')
    from neurochess.autostop import request_unassign,find_runtime_address
    request_unassign(find_runtime_address())

if __name__=='__main__':main()
