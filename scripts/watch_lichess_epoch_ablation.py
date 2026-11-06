"""Pause only after the original step-4000 checkpoint has passed readback."""
import fcntl,hashlib,json,os,signal,subprocess,sys,time,traceback
from pathlib import Path

work=Path('/content/lichess-joint-epoch1')
save=Path('/content/drive/MyDrive/neurochess/lichess-joint-epoch1-20261104-r2')
root=Path('/content/lichess-epoch-code')
out=Path('/content/lichess-experiment-4000')
dest=save/'experiment-4000'
out.mkdir(exist_ok=True);dest.mkdir(exist_ok=True)
lock=(out/'watch.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
def write(path,value):
    q=path.with_name(path.name+'.tmp');q.write_text(json.dumps(value,indent=2));q.replace(path)
def emit(phase,**extra):
    d=dict(phase=phase,updated_at=time.time(),target_step=4000,main_training_resumes_automatically=False,**extra)
    for p in [out/'watch-status.json',dest/'watch-status.json',work/'experiment-status.json']:write(p,d)
def read(path):
    try:return json.loads(path.read_text())
    except (OSError,ValueError):return {}
def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
try:
    proc=read(work/'training-process.json');pid=proc['pid'];stat=Path(f'/proc/{pid}/stat')
    identity=stat.read_text().rsplit(')',1)[1].split()[19]
    emit('armed',pid=pid,process_start_ticks=identity)
    while True:
        current=stat.read_text().rsplit(')',1)[1].split()
        if current[19]!=identity or current[0]=='Z':raise RuntimeError('Main training process disappeared before pause')
        proof=read(work/'checkpoint-verification.json')
        gate=read(work/'latest-gate.json')
        status=read(work/'status.json')
        if gate.get('step',0)>=4000 and not gate.get('passed'):raise RuntimeError('Step 4000 did not pass original gates')
        if proof.get('step',0)>4000:raise RuntimeError('Missed exact step-4000 committed checkpoint')
        if proof.get('step')==4000:
            if gate.get('step')!=4000 or not gate.get('passed') or not gate.get('search_passed'):
                raise RuntimeError('Step-4000 full metric/search certificate missing')
            os.kill(pid,signal.SIGSTOP)
            for _ in range(100):
                if stat.read_text().rsplit(')',1)[1].split()[0]=='T':break
                time.sleep(.02)
            else:raise RuntimeError('Main process did not stop')
            # Hard links preserve the exact committed inode. The paused trainer
            # cannot replace these files; the durable copies are verified below.
            for name in ['last.pt','best.pt']:
                target=out/('anchor-'+name)
                if not target.exists():os.link(work/name,target)
                if sha(target)!=proof['sha256'][name]:raise RuntimeError('Frozen checkpoint hash mismatch: '+name)
            certificate=dict(step=4000,pid=pid,process_start_ticks=identity,process_stopped=True,checkpoint_sha256=proof['sha256'],gate=gate,observed_status=read(work/'status.json'),paused_at=time.time(),default_promoted=False)
            for path in [out/'pause-certificate.json',dest/'pause-certificate.json']:write(path,certificate)
            status.update(phase='experiment-paused',step=4000,new_positions=4000*768,replay_positions=4000*256,epoch_fraction=4000*768/9800000,updated_at=time.time())
            write(work/'status.json',status);write(save/'status.json',status)
            emit('main-paused',pid=pid,checkpoint_step=4000)
            break
        if status.get('phase') in ['failed','guard-stopped','complete','final-evaluation']:
            raise RuntimeError('Main training stopped before experimental checkpoint: '+status['phase'])
        time.sleep(.1)
    while not (out/'deployment-ready.json').exists():time.sleep(1)
    ready=read(out/'deployment-ready.json')
    for name,want in ready['sha256'].items():
        if sha(root/name)!=want:raise RuntimeError('Experiment deployment mismatch: '+name)
    emit('diagnostics-starting',pid=pid)
    log=(out/'console.log').open('ab')
    result=subprocess.run([sys.executable,'-u','-m','scripts.lichess_epoch_ablation','--main-work',str(work),'--out',str(out),'--save',str(dest)],cwd=root,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'2','PYTHONUNBUFFERED':'1'})
    if result.returncode:raise RuntimeError('Experiment coordinator exited '+str(result.returncode))
except BaseException:
    emit('failed',traceback=traceback.format_exc())
    raise
