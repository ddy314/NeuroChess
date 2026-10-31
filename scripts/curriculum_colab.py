"""Two-stage continued training, hard-data integration, 128 games, verified release."""
import fcntl,hashlib,json,os,shutil,subprocess,sys,tarfile,threading,time
from pathlib import Path

ROOT=Path('/content/neurochess');LOCAL=Path('/content/neurochess-runs/curriculum-v3')
MOUNT=Path('/content/drive');PERSISTENT=MOUNT/'MyDrive/neurochess/curriculum-v3'


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def copy_atomic(source,dest):
    dest=Path(dest);dest.parent.mkdir(parents=True,exist_ok=True)
    temp=dest.with_name(dest.name+'.tmp');shutil.copyfile(source,temp)
    if sha(temp)!=sha(source):raise OSError('Copy failed readback verification')
    temp.replace(dest)


def extract(archive,destination):
    destination.mkdir(parents=True,exist_ok=True)
    with tarfile.open(archive) as tar:tar.extractall(destination,filter='data')


def filtered_core(core,hard,target):
    import numpy as np
    from neurochess.data import ShardWriter
    from neurochess.audit import audit
    from neurochess.targets import TARGET_SPEC
    keys=set()
    for f in hard.glob('train-*.npy'):
        keys.update(r.tobytes() for r in np.load(f,mmap_mode='r')['key'])
    if not keys:raise ValueError('Expected 200000 unique training hard cases')
    blocked=hashlib.sha256(b''.join(sorted(keys))).hexdigest()
    if (target/'manifest.json').exists():
        if json.loads((target/'manifest.json').read_text())['provenance']['blocked_keys_sha256']!=blocked:
            raise ValueError('Existing filtered dataset differs')
        return
    building=target.with_name(target.name+'.building')
    if building.exists():shutil.rmtree(building)
    writer=ShardWriter(building)
    for split in ['train','val']:
        for f in sorted(core.glob(split+'-*.npy')):
            rows=np.load(f,mmap_mode='r')
            for row in rows:
                if split=='train' and row['key'].tobytes() in keys:continue
                writer.add(split,row)
    if writer.counts['train']!=5000000-len(keys):raise ValueError('Hard/core exclusion count differs')
    writer.close(dict(kind='core-with-reannotated-hard-cases-removed',target_spec=TARGET_SPEC,
        parent_manifest_sha256=sha(core/'manifest.json'),blocked_keys_sha256=blocked,excluded_training_keys=len(keys)))
    result=audit(building);(building/'audit.json').write_text(json.dumps(result,indent=2))
    building.replace(target)


def main():
    if not os.path.ismount(MOUNT):raise RuntimeError('Mount the existing training Drive account first')
    os.chdir(ROOT);sys.path.insert(0,str(ROOT))
    import torch
    from neurochess.persistence import RunMirror
    from neurochess.targets import TARGET_SPEC
    from neurochess.telemetry import write_status
    from neurochess.evaluation_check import verify_arena
    if 'L4' not in torch.cuda.get_device_name():raise RuntimeError('Expected L4')
    lock=open('/tmp/neurochess-curriculum-v3.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if not LOCAL.exists() and (PERSISTENT/'stage-a/last.pt').exists():shutil.copytree(PERSISTENT,LOCAL)
    LOCAL.mkdir(parents=True,exist_ok=True);PERSISTENT.mkdir(parents=True,exist_ok=True)
    plan=json.loads((ROOT/'configs/curriculum-v3.json').read_text());(LOCAL/'curriculum-plan.json').write_text(json.dumps(plan,indent=2))
    data=ROOT/'data';data.mkdir(exist_ok=True)
    core=data/'evals-v2-5m';cache=MOUNT/'MyDrive/neurochess-data/evals-v2-5m.tar.gz'
    cache_sha='a0b5dbef770fcf2b7b4bc31a709f4a2b8379421965aa591f69291a6abbc7c2ef'
    if sha(cache)!=cache_sha:raise ValueError('Original persistent core cache differs')
    if not (core/'manifest.json').exists():
        archive=Path('/content/evals-v2-5m.tar.gz');copy_atomic(cache,archive);extract(archive,core)
    checks=json.loads((core/'checksums.json').read_text())
    for name,expected in checks.items():
        if sha(core/name)!=expected:raise ValueError('Core upload/cache integrity failed: '+name)
    replay=data/'human-replay'
    if not (replay/'manifest.json').exists():shutil.copytree(MOUNT/'MyDrive/neurochess-data/elite-recent-2026q3',replay,dirs_exist_ok=True)
    import numpy as np
    replay_counts={split:sum(len(np.load(f,mmap_mode='r')) for f in replay.glob(split+'-*.npy')) for split in ['train','val']}
    if replay_counts!=json.loads((replay/'manifest.json').read_text())['counts']:raise ValueError('Human replay copy is incomplete')
    reference=data/'evals-v1'
    if not (reference/'manifest.json').exists():
        reference.mkdir(exist_ok=True)
        for f in sorted(core.glob('train-*.npy'))[:52]:os.link(f,reference/f.name)
        os.link(core/'val-000000.npy',reference/'val-000000.npy')
        copy_atomic(MOUNT/'MyDrive/neurochess/engine-v1/dataset.json',reference/'manifest.json')
    parent=ROOT/'parent-v2.pt';parent_info=json.loads((MOUNT/'MyDrive/neurochess/engine-v2-expanded/autostop-verification.json').read_text())
    if not parent.exists():copy_atomic(MOUNT/'MyDrive/neurochess/engine-v2-expanded/last.pt',parent)
    if sha(parent)!=parent_info['verified_files']['last.pt']:raise ValueError('Parent checkpoint differs')
    saved=torch.load(parent,map_location='cpu',weights_only=False,mmap=True)
    if saved['step']!=6000 or saved.get('origin_step')!=1500:raise ValueError('Wrong parent training stage')
    del saved
    (LOCAL/'parent-checkpoint.json').write_text(json.dumps(parent_info,indent=2))
    (LOCAL/'data-cache.json').write_text(json.dumps(dict(path=str(cache),sha256=cache_sha),indent=2))
    (LOCAL/'label-contract.json').write_text(json.dumps(TARGET_SPEC,indent=2))
    for name in ['eval-openings.fen','evaluation-freeze.json','training-exclusions.fen']:
        copy_atomic(ROOT/'curriculum-assets'/name,LOCAL/name)
    code=LOCAL/'training-code.tar.gz'
    if not code.exists():
        with tarfile.open(code.with_suffix('.tmp'),'w:gz',compresslevel=1) as tar:
            for folder in ['neurochess','configs','scripts','curriculum-assets','bin']:
                for file in sorted((ROOT/folder).rglob('*')):
                    if file.is_file() and '__pycache__' not in file.parts and file.suffix not in ['.pyc','.so']:
                        tar.add(file,arcname=str(file.relative_to(ROOT)))
            tar.add(ROOT/'pyproject.toml',arcname='pyproject.toml')
        code.with_suffix('.tmp').replace(code)
    mirror=RunMirror(LOCAL,PERSISTENT,MOUNT);sync_lock=threading.Lock();stop=threading.Event()
    def sync():
        with sync_lock:
            mirror.sync();(LOCAL/'persistence.json').write_text(json.dumps(dict(last_sync=time.time(),destination=str(PERSISTENT))))
    stage=['setup']
    def replicate():
        while not stop.wait(30):
            try:
                (LOCAL/'pipeline-status.json').write_text(json.dumps(dict(stage=stage[0],updated_at=time.time(),additional_steps=8000)))
                sync()
            except OSError as error:print('Drive sync delayed:',error,flush=True)
    worker=threading.Thread(target=replicate,daemon=True);worker.start()
    with (LOCAL/'dashboard.log').open('a') as log:
        subprocess.Popen([sys.executable,'-m','neurochess.dashboard','--root',str(LOCAL.parent),'--port','8765'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    def train(name,out,source,core_data,hard=None,wave=0):
        config=json.loads((ROOT/('configs/curriculum-'+name+'.json')).read_text())
        if wave:config.update(steps=1000,lr=[8e-6,6e-6,4e-6,3e-6][wave-1],warmup=50)
        config.update(data=str(core_data),out=str(out),replay_data=str(replay),validation_data=str(reference),arena_stockfish=str(ROOT/'bin/stockfish'))
        if hard:config['hard_data']=str(hard)
        out.mkdir(parents=True,exist_ok=True)
        def run(extra):
            args=[]
            for k,v in config.items():
                args.append('--'+k.replace('_','-'))
                if v is not True:args.append(str(v))
            with (out/'console.log').open('a') as log:
                subprocess.run([sys.executable,'-u','-m','neurochess.train',*args,*extra],stdout=log,stderr=subprocess.STDOUT,check=True)
        if not (out/'last.pt').exists():
            run(['--continue-from',str(source),'--stop-after','25'])
            saved=torch.load(out/'last.pt',map_location='cpu',weights_only=False,mmap=True)
            origin=saved['origin_step'];updates={int(s['step']) for s in saved['optimizer']['state'].values()}
            if updates!={origin+25}:raise ValueError('Optimizer continuation failed')
            (out/'continuation-verification.json').write_text(json.dumps(dict(step=25,origin_step=origin,optimizer_updates=sorted(updates)),indent=2))
            del saved;sync();print('Verified '+name+' continuation',flush=True)
        saved=torch.load(out/'last.pt',map_location='cpu',weights_only=False,mmap=True);finished=saved['step']>=config['steps'];del saved
        if not finished:run(['--resume',str(out/'last.pt')])
    try:
        stage[0]='stage-a';train('stage-a',LOCAL/'stage-a',parent,core)
        checkpoints=[LOCAL/'stage-a/best.pt'];source=LOCAL/'stage-a/last.pt'
        for wave in range(1,5):
            stage[0]=f'waiting-for-hard-wave-{wave}'
            shared=MOUNT/f'MyDrive/neurochess-data/hard-v3-wave-{wave}'
            marker=Path(str(shared)+'-ready.json');hard_archive=LOCAL/f'hard-wave-{wave}-data.tar.gz'
            deadline=time.time()+4*3600
            while not marker.exists():
                if time.time()>deadline:raise RuntimeError('Cloud teacher wave timeout; retaining checkpoint')
                time.sleep(30)
            ready=json.loads(marker.read_text())
            if not hard_archive.exists():copy_atomic(Path(str(shared)+'.tar.gz'),hard_archive)
            if sha(hard_archive)!=ready['sha256']:raise ValueError('Cloud teacher wave hash mismatch')
            copy_atomic(marker,LOCAL/f'hard-wave-{wave}-delivery.json')
            hard=data/f'hard-v3-wave-{wave}'
            if not (hard/'manifest.json').exists():extract(hard_archive,hard)
            hm=json.loads((hard/'manifest.json').read_text());ha=json.loads((hard/'audit.json').read_text())
            if hm['counts']!={'train':50000*wave,'val':500*wave} or ha['within_split_duplicates'] or ha['exact_position_leakage'] or hm['provenance']['target_spec']!=TARGET_SPEC:
                raise ValueError('Hard dataset count/label/audit differs')
            stage[0]=f'filtering-old-hard-labels-{wave}';filtered=data/f'core-filtered-v3-wave-{wave}'
            filtered_core(core,hard,filtered)
            stage[0]=f'hard-wave-{wave}';out=LOCAL/f'hard-wave-{wave}' if wave<4 else LOCAL
            train('stage-b',out,source,filtered,hard,wave=wave)
            source=out/'last.pt';checkpoints.append(out/'best.pt')
        choices=[]
        for file in checkpoints:
            saved=torch.load(file,map_location='cpu',weights_only=False,mmap=True)
            choices.append((saved['validation']['val_loss'],file,dict(step=saved['step'],origin_step=saved.get('origin_step',0))))
            del saved
        loss,file,selection=min(choices,key=lambda x:x[0]);copy_atomic(file,LOCAL/'selected.pt')
        (LOCAL/'selection.json').write_text(json.dumps(dict(source=str(file),val_loss=loss,checkpoint_sha256=sha(LOCAL/'selected.pt'),**selection),indent=2))
        stage[0]='sf1000-evaluation';evaluation=LOCAL/'evaluations/final'
        (LOCAL/'evaluation-required.json').write_text(json.dumps(dict(directory='evaluations/final',games_per_mode=64)))
        if not (evaluation/'result.json').exists():
            if evaluation.exists():
                # Keep the incomplete attempt as evidence; restart its fixed games.
                evaluation.rename(evaluation.with_name('interrupted-'+str(int(time.time()))))
            evaluation.parent.mkdir(exist_ok=True)
            with (LOCAL/'evaluation-console.log').open('a') as log:
                subprocess.run([sys.executable,'-u','scripts/multihead_arena.py','--checkpoint',str(LOCAL/'selected.pt'),
                    '--openings',str(LOCAL/'eval-openings.fen'),'--pairs','32','--batch-games','16','--nodes','1000',
                    '--simulations','512','--stockfish',str(ROOT/'bin/stockfish'),'--out',str(evaluation)],check=True,stdout=log,stderr=subprocess.STDOUT)
        result=verify_arena(evaluation,64,LOCAL/'selected.pt')
        (LOCAL/'evaluation-verification.json').write_text(json.dumps(result,indent=2))
        stage[0]='complete';(LOCAL/'pipeline-status.json').write_text(json.dumps(dict(stage='complete',additional_updates=8000,updated_at=time.time())))
    except Exception as error:
        (LOCAL/'pipeline-error.json').write_text(json.dumps(dict(stage=stage[0],error=repr(error),time=time.time()),indent=2));raise
    finally:
        stop.set();worker.join();sync();sync()
    saved=torch.load(LOCAL/'last.pt',map_location='cpu',weights_only=False,mmap=True);data_hash=saved['data_hash'];del saved
    from neurochess.autostop import finalize_and_stop
    print('Training and both 64-game SF1000 evaluations persisted; releasing runtime',flush=True)
    finalize_and_stop(LOCAL,PERSISTENT,MOUNT,1000,data_hash)


if __name__=='__main__':main()
