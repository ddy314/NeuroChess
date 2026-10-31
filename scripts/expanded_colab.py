"""Persist audited expanded data, check a short continuation, then train on L4.

Run in the existing mounted runtime. A new run directory preserves engine-v1.
"""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import threading
import time

ROOT = Path('/content/neurochess')
LOCAL = Path('/content/neurochess-runs/engine-v2-expanded')
MOUNT = Path('/content/drive')
PERSISTENT = MOUNT/'MyDrive/neurochess/engine-v2-expanded'
DATA = ROOT/'data/evals-v2-5m'
PARENT = Path('/content/neurochess-checkpoints/engine-v1-step-1500/last.pt')


def digest(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    if not os.path.ismount(MOUNT): raise RuntimeError('Existing Google Drive mount is required')
    os.chdir(ROOT); sys.path.insert(0,str(ROOT))
    from neurochess.persistence import RunMirror
    from neurochess.telemetry import write_status
    import torch
    if 'L4' not in torch.cuda.get_device_name(): raise RuntimeError('Expected the existing L4 runtime')
    lock = open('/tmp/neurochess-expanded-launcher.lock','w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    LOCAL.mkdir(parents=True,exist_ok=True); PERSISTENT.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((DATA/'manifest.json').read_text())
    if manifest['counts']['train'] != 5_000_000: raise RuntimeError('Expanded data target not reached')
    audit=json.loads((DATA/'audit.json').read_text())
    if audit['counts']!=manifest['counts'] or audit['within_split_duplicates'] or audit['exact_position_leakage']:
        raise RuntimeError('Expanded dataset audit did not pass')
    checksums=json.loads((DATA/'checksums.json').read_text())
    for name,expected in checksums.items():
        if digest(DATA/name)!=expected: raise RuntimeError(f'Uploaded data differs: {name}')
    parent_record=json.loads(PARENT.with_name('checkpoint-verification.json').read_text())
    if parent_record['checkpoint_step']!=1500 or digest(PARENT)!=parent_record['sha256']:
        raise RuntimeError('Parent checkpoint verification failed')
    config=json.loads((ROOT/'configs/phase2-expanded-l4.json').read_text())
    config.update(data=str(DATA),out=str(LOCAL),replay_data=str(ROOT/'data/human-replay'))
    (LOCAL/'run.json').write_text(json.dumps(config,indent=2))
    (LOCAL/'dataset.json').write_text((DATA/'manifest.json').read_text())
    (LOCAL/'validation-dataset.json').write_text((ROOT/'data/evals-v1/manifest.json').read_text())
    (LOCAL/'replay-dataset.json').write_text((ROOT/'data/human-replay/manifest.json').read_text())
    (LOCAL/'parent-checkpoint.json').write_text(json.dumps(parent_record,indent=2))
    write_status(LOCAL,'copying_data',step=0,total_steps=config['steps'],note='Caching verified expanded data to Drive')
    # Keep a compressed cache, rather than a second uncompressed 9+ GiB copy on Drive.
    cache=Path('/content/neurochess-data-cache');cache.mkdir(exist_ok=True)
    archive=cache/'evals-v2-5m.tar.gz'
    if not archive.exists():
        temporary=archive.with_suffix('.tmp')
        with tarfile.open(temporary,'w:gz',compresslevel=1) as tar:
            for file in sorted(DATA.iterdir()):
                if file.suffix in {'.npy','.json'} and file.name!='progress.json':tar.add(file,arcname=file.name)
        temporary.replace(archive)
    expected=digest(archive)
    destination=MOUNT/'MyDrive/neurochess-data'/archive.name
    if not destination.parent.is_dir(): raise RuntimeError('Original persistent data directory is missing')
    if not destination.exists() or digest(destination)!=expected:
        temporary=destination.with_suffix('.tmp')
        shutil.copyfile(archive,temporary)
        if digest(temporary)!=expected: raise RuntimeError('Drive data cache hash mismatch')
        temporary.replace(destination)
    cache_record=dict(path=str(destination),sha256=expected,bytes=archive.stat().st_size,
                      counts=manifest['counts'],verified_at=time.time())
    (LOCAL/'data-cache.json').write_text(json.dumps(cache_record,indent=2))
    print(json.dumps(dict(data_persisted=cache_record)),flush=True)
    code=LOCAL/'training-code.tar.gz'
    if not code.exists():
        with tarfile.open(code,'w:gz',compresslevel=1) as tar:
            for folder in ['neurochess','configs','scripts']:
                for file in sorted((ROOT/folder).rglob('*')):
                    if file.is_file() and '__pycache__' not in file.parts and file.suffix not in {'.pyc','.so'}:
                        tar.add(file,arcname=str(file.relative_to(ROOT)))
            tar.add(ROOT/'pyproject.toml',arcname='pyproject.toml')
            if (ROOT/'phase2-exclude.fen').exists():
                tar.add(ROOT/'phase2-exclude.fen',arcname='phase2-exclude.fen')
    mirror=RunMirror(LOCAL,PERSISTENT,MOUNT)
    sync_lock=threading.Lock();stop=threading.Event()
    def sync():
        with sync_lock:
            mirror.sync()
            (LOCAL/'persistence.json').write_text(json.dumps(dict(last_sync=time.time(),destination=str(PERSISTENT))))
    def replicate():
        while not stop.wait(30):
            try:sync()
            except OSError as error:print('Drive sync delayed:',error,flush=True)
    worker=threading.Thread(target=replicate,daemon=True);worker.start()
    def train(extra):
        args=[]
        for key,value in config.items():
            args.append('--'+key.replace('_','-'))
            if value is not True:args.append(str(value))
        with (LOCAL/'console.log').open('a') as log:
            return subprocess.call([sys.executable,'-u','-m','neurochess.train',*args,*extra],stdout=log,stderr=subprocess.STDOUT)
    try:
        if not (LOCAL/'last.pt').exists():
            code=train(['--continue-from',str(PARENT),'--stop-after','25'])
            if code:raise RuntimeError(f'Continuation pilot exited {code}')
            saved=torch.load(LOCAL/'last.pt',map_location='cpu',weights_only=False,mmap=True)
            assert saved['step']==25 and saved['origin_step']==1500
            updates={int(s['step']) for s in saved['optimizer']['state'].values()}
            if updates!={1525}:raise RuntimeError(f'Optimizer was not continued: {updates}')
            verification=dict(stage_step=25,cumulative_step=1525,optimizer_updates=sorted(updates),
                              data_hash=saved['data_hash'],checkpoint_sha256=digest(LOCAL/'last.pt'),
                              fixed_validation_positions=8457,verified_at=time.time())
            (LOCAL/'continuation-verification.json').write_text(json.dumps(verification,indent=2))
            del saved
            sync()
            print(json.dumps(dict(continuation_verified=verification)),flush=True)
        code=train(['--resume',str(LOCAL/'last.pt')])
        if code:raise RuntimeError(f'Expanded training exited {code}')
    finally:
        stop.set();worker.join()
        sync();sync()
    print('Expanded training completed and persisted',flush=True)
    if os.environ.get('NC_AUTO_STOP', '1') == '1':
        from neurochess.autostop import finalize_and_stop
        saved=torch.load(LOCAL/'last.pt',map_location='cpu',weights_only=False,mmap=True)
        final_hash=saved['data_hash'];del saved
        finalize_and_stop(LOCAL,PERSISTENT,MOUNT,config['steps'],final_hash)


if __name__=='__main__':main()
