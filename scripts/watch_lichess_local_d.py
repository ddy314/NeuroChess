"""Wait for authorized Drive downloads, verify numeric shards, then run once."""
import fcntl, hashlib, json, os, shutil, sys, time, traceback, zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/lichess-local-d-20261105'
SOURCE=ROOT/'data/lichess-depth24-knodes1000-clean-10m-20261104'
EVIDENCE=ROOT/'output/lichess-clean-10m-20261104/final-evidence'
PYTHON='/home/xia/code/ml_frontier/.venv/bin/python'

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def write(p,data):
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(data,indent=2));tmp.replace(p)

def main():
    OUT.mkdir(parents=True,exist_ok=True);SOURCE.mkdir(parents=True,exist_ok=True)
    handle=(OUT/'launcher.lock').open('w');fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (OUT/'started.json').exists():raise RuntimeError('Experiment already started; refusing a second launch')
    ready=json.loads((EVIDENCE/'ready.json').read_text())
    required={i['data']['name']:i['data'] for i in ready['shards']}
    hashes={p:sha(ROOT/p) for p in ['scripts/lichess_local_d.py','neurochess/value_kl_diagnostic.py','scripts/lichess_epoch_ablation.py','neurochess/epoch_ablation.py','neurochess/guarded_epoch.py']}
    seen={};verified=set();start=time.time()
    checkpoint_verified=sha('/home/xia/Downloads/last.pt')=='7e62cea5558ae39dc2cc4925de7768cec18f2fd40ca1784aa6c1131420948655'
    if not checkpoint_verified:raise ValueError('Full checkpoint hash mismatch')
    write(OUT/'launcher.json',dict(pid=os.getpid(),started_at=start,python=PYTHON,source_hashes=hashes,wait_limit_hours=6))
    for name in ['ready.json','completion.json']:shutil.copyfile(EVIDENCE/name,SOURCE/name)
    while time.time()-start<6*3600:
        for n,m in required.items():
            p=SOURCE/n
            if n not in verified and p.is_file() and sha(p)==m['sha256']:verified.add(n)
        for archive in sorted(Path('/home/xia/Downloads').glob('*.zip')):
            st=archive.stat()
            if st.st_mtime<start-3600 or seen.get(str(archive))==(st.st_size,st.st_mtime_ns):continue
            try:
                with zipfile.ZipFile(archive) as z:
                    matches=[i for i in z.infolist() if Path(i.filename).name in required]
                    for i in matches:
                        name=Path(i.filename).name
                        if name in verified:continue
                        meta=required[name]
                        if i.file_size!=meta['bytes']:raise ValueError('Downloaded shard size differs: '+name)
                        dest=SOURCE/name;tmp=dest.with_suffix(dest.suffix+'.tmp')
                        with z.open(i) as src,tmp.open('wb') as dst:shutil.copyfileobj(src,dst)
                        if sha(tmp)!=meta['sha256']:raise ValueError('Downloaded shard hash differs: '+name)
                        tmp.replace(dest);verified.add(name)
                seen[str(archive)]=(st.st_size,st.st_mtime_ns)
            except zipfile.BadZipFile:
                continue  # Browser may still be finalizing this download.
        state=dict(phase='waiting-for-data-download',pid=os.getpid(),updated_at=time.time(),
            numeric_shards_verified=len(verified),numeric_shards_required=len(required),diagnostic_started=False,
            python=PYTHON,main_step=4000,main_training_resumes_automatically=False,
            checkpoint_verified=checkpoint_verified)
        write(OUT/'status.json',state)
        if len(verified)==len(required):
            for p,want in hashes.items():
                if sha(ROOT/p)!=want:raise ValueError('Runner changed after launch preparation: '+p)
            write(OUT/'download-verification.json',dict(passed=True,shards=len(verified),finished_at=time.time()))
            os.chdir(ROOT)
            os.execv(PYTHON,[PYTHON,'-u',str(ROOT/'scripts/lichess_local_d.py'),'--data',str(SOURCE)])
        time.sleep(30)
    raise TimeoutError('Data download not complete within six hours; no diagnostic or training started')

if __name__=='__main__':
    try:main()
    except BaseException:
        error=dict(phase='download-preparation-failed',traceback=traceback.format_exc(),updated_at=time.time(),diagnostic_started=False)
        write(OUT/'launcher-failure.json',error);write(OUT/'status.json',error);raise
