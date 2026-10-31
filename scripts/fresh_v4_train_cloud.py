"""Equal-budget A/B continuation from current selected weights, then frozen evaluation."""
import fcntl,hashlib,json,os,shutil,subprocess,sys,tarfile,threading,time
from pathlib import Path
ROOT=Path('/content/neurochess');MOUNT=Path('/content/drive');WORK=Path('/content/fresh-v4-train')
SAVE=MOUNT/'MyDrive/neurochess/fresh-v4-training'


def sha(path):
    with open(path,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def copy(source,target):
    target=Path(target);target.parent.mkdir(parents=True,exist_ok=True);temp=target.with_name(target.name+'.tmp');shutil.copyfile(source,temp)
    if sha(source)!=sha(temp):raise OSError('Persistent readback failed')
    temp.replace(target)


def prepare_book(library,old_core,fresh,out,reserved_fens):
    import chess,numpy as np
    from neurochess.data import record,ShardWriter
    from neurochess.encoding import position_key
    from neurochess.audit import audit
    if (out/'manifest.json').exists():return out
    blocked=set()
    for root in [old_core,fresh/'trainval',fresh/'test']:
        for file in root.glob('val-*.npy'):blocked.update(r.tobytes() for r in np.load(file,mmap_mode='r')['key'])
    blocked.update(position_key(chess.Board(f)) for f in reserved_fens if f.strip())
    positions={}
    for line in library['lines']:
        board=chess.Board()
        for uci in line['moves']:
            key=position_key(board);move=board.parse_uci(uci)
            if key not in blocked:
                if key not in positions:
                    row,legal=record(board);positions[key]=(row,legal,{})
                row,legal,counts=positions[key];idx=legal.index(move);counts[idx]=counts.get(idx,0)+1
            board.push(move)
    writer=ShardWriter(out,size=256)
    for key,(row,legal,counts) in sorted(positions.items()):
        total=sum(counts.values())
        for idx,count in counts.items():row['policy'][idx]=count/total
        writer.add('train',row)
    writer.close(dict(kind='curated-opening-policy-only',source=library['source'],license='CC0',
        supervision='Frequency of book continuations; no claim of unique optimality',shared_in_both_arms=True,
        excluded_all_reference_and_fresh_heldout_keys=True,counts_as_novel_500k=False))
    proof=audit(out)
    if proof['within_split_duplicates'] or proof['counts']['train']<512:raise ValueError('Invalid opening supplement')
    return out


def main():
    if not os.path.ismount(MOUNT):raise RuntimeError('Mount Drive with colab drivemount first')
    lock=open('/tmp/neurochess-fresh-v4-train.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    os.chdir(ROOT);sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
    import torch,numpy as np
    from fresh_hard_data import atomic_json
    from neurochess.persistence import RunMirror
    if 'L4' not in torch.cuda.get_device_name():raise ValueError('Expected cost-efficient L4')
    plan=json.loads((ROOT/'configs/fresh-v4.json').read_text())
    if not WORK.exists() and any((SAVE/arm/'last.pt').exists() for arm in ['control','fresh']):
        if json.loads((SAVE/'experiment-plan.json').read_text())!=plan:raise ValueError('Recovery experiment plan differs')
        shutil.copytree(SAVE,WORK)
        import zipfile
        for file in WORK.rglob('*.pt'):
            with zipfile.ZipFile(file) as archive:
                if archive.testzip() is not None:raise ValueError('Recovered checkpoint CRC differs')
    WORK.mkdir(exist_ok=True);SAVE.mkdir(parents=True,exist_ok=True)
    copy(ROOT/'configs/fresh-v4.json',WORK/'experiment-plan.json')
    marker=MOUNT/'MyDrive/neurochess-data/fresh-v4-500k-ready.json'
    if not marker.exists():raise RuntimeError('Audited data not ready; refuse idle GPU wait')
    ready=json.loads(marker.read_text());copy(marker,WORK/'data-delivery.json')
    archive=WORK/'data.tar.gz'
    if not archive.exists():copy(MOUNT/'MyDrive/neurochess-data/fresh-v4-500k.tar.gz',archive)
    if sha(archive)!=ready['archive_sha256']:raise ValueError('Fresh data delivery SHA differs')
    if not (WORK/'data/trainval/manifest.json').exists():
        with tarfile.open(archive) as tar:tar.extractall(WORK,filter='data')
    core=ROOT/'data/evals-v2-5m';core.parent.mkdir(exist_ok=True)
    if not (core/'manifest.json').exists():
        old_archive=WORK/'core.tar.gz';copy(MOUNT/'MyDrive/neurochess-data/evals-v2-5m.tar.gz',old_archive)
        if sha(old_archive)!='a0b5dbef770fcf2b7b4bc31a709f4a2b8379421965aa591f69291a6abbc7c2ef':raise ValueError('Old core SHA differs')
        core.mkdir(exist_ok=True)
        with tarfile.open(old_archive) as tar:tar.extractall(core,filter='data')
    replay=ROOT/'data/human-replay'
    if not (replay/'manifest.json').exists():shutil.copytree(MOUNT/'MyDrive/neurochess-data/elite-recent-2026q3',replay,dirs_exist_ok=True)
    parent=WORK/'parent-selected.pt'
    if not parent.exists():copy(MOUNT/'MyDrive/neurochess/curriculum-v3/selected.pt',parent)
    if sha(parent)!=plan['parent_sha256']:raise ValueError('Parent is not the current selected model')
    book=prepare_book(json.loads((WORK/'opening-library.json').read_text()),core,WORK/'data',WORK/'opening-replay',(ROOT/'fresh-v4-assets/eval-openings.fen').read_text().splitlines())
    copy(ROOT/'fresh-v4-code.tar.gz',WORK/'training-code.tar.gz')
    mirror=RunMirror(WORK,SAVE,MOUNT);sync_lock=threading.Lock();stop=threading.Event();phase=['setup']
    def sync():
        with sync_lock:mirror.sync()
    def replicate():
        while not stop.wait(60):
            try:atomic_json(WORK/'pipeline-status.json',dict(phase=phase[0],updated_at=time.time(),updates_per_arm=2000));sync()
            except OSError as error:print('Persistence retry',repr(error),flush=True)
    worker=threading.Thread(target=replicate,daemon=True);worker.start()
    with (WORK/'dashboard.log').open('a') as log:subprocess.Popen([sys.executable,'-m','neurochess.dashboard','--root',str(WORK),'--port','8765'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    try:
        # Same seed, reset optimizer, initial SHA, LR, batch, count and validation.
        for arm,hard in [('control',core),('fresh',WORK/'data/trainval')]:
            phase[0]='training-'+arm;out=WORK/arm;out.mkdir(exist_ok=True)
            args=[sys.executable,'-u','-m','neurochess.train','--data',str(core),'--hard-data',str(hard),
                '--replay-data',str(replay),'--book-data',str(book),'--book-every','10','--mixture-core-batches','5','--mixture-hard-batches','4',
                '--validation-data',str(WORK/'data/trainval'),'--out',str(out),'--steps','2000','--batch-size','64','--accum','16',
                '--lr','6e-6','--warmup','100','--seed',str(plan['seed']),'--workers','2','--cpu-threads','2',
                '--save-every','250','--eval-batches','1000','--q-weight','.5','--value-weight','1','--history-free','--compile']
            status=json.loads((out/'status.json').read_text()) if (out/'status.json').exists() else {}
            if status.get('phase')!='complete':
                if not (out/'last.pt').exists():
                    with (out/'console.log').open('a') as log:
                        subprocess.run(args+['--init',str(parent),'--stop-after','25'],check=True,stdout=log,stderr=subprocess.STDOUT)
                    pilot=torch.load(out/'last.pt',map_location='cpu',weights_only=False,mmap=True)
                    if pilot['step']!=25 or {int(s['step']) for s in pilot['optimizer']['state'].values()}!={25}:
                        raise ValueError('Arm pilot optimizer/checkpoint verification failed')
                    atomic_json(out/'pilot-verification.json',dict(step=25,optimizer_updates=25,parent_sha256=sha(parent),
                        checkpoint_sha256=sha(out/'last.pt')));del pilot;sync()
                with (out/'console.log').open('a') as log:
                    subprocess.run(args+['--resume',str(out/'last.pt')],check=True,stdout=log,stderr=subprocess.STDOUT)
            saved=torch.load(out/'last.pt',map_location='cpu',weights_only=False,mmap=True)
            if saved['step']!=2000:raise ValueError('Arm did not finish equal update budget')
            del saved;sync()
        # Access the frozen test labels only after both arms finish and choose best.pt.
        phase[0]='frozen-puzzle-evaluation'
        with (WORK/'puzzle-evaluation.log').open('a') as log:
            subprocess.run([sys.executable,'scripts/eval_fresh_puzzles.py','--jobs',str(WORK/'jobs.jsonl'),'--out',str(WORK/'puzzle-evaluation.json'),
                '--checkpoint',str(parent),'--checkpoint',str(WORK/'control/best.pt'),'--checkpoint',str(WORK/'fresh/best.pt')],check=True,stdout=log,stderr=subprocess.STDOUT)
        phase[0]='frozen-blunder-evaluation'
        with (WORK/'blunder-evaluation.log').open('a') as log:
            subprocess.run([sys.executable,'scripts/eval_fresh_blunders.py','--puzzle-result',str(WORK/'puzzle-evaluation.json'),
                '--jobs',str(WORK/'jobs.jsonl'),'--stockfish',str(ROOT/'bin/stockfish'),'--out',str(WORK/'blunder-evaluation.json')],
                check=True,stdout=log,stderr=subprocess.STDOUT)
        openings=ROOT/'fresh-v4-assets/eval-openings.fen'
        from neurochess.evaluation_check import verify_arena
        for name,checkpoint in [('parent',parent),('control',WORK/'control/best.pt'),('fresh',WORK/'fresh/best.pt')]:
            phase[0]='sf1000-'+name;out=WORK/'evaluations'/name;out.parent.mkdir(exist_ok=True)
            if not (out/'result.json').exists():
                if out.exists():out.rename(out.with_name(name+'-interrupted-'+str(int(time.time()))))
                with (WORK/f'evaluation-{name}.log').open('a') as log:
                    subprocess.run([sys.executable,'scripts/multihead_arena.py','--checkpoint',str(checkpoint),'--openings',str(openings),
                        '--stockfish',str(ROOT/'bin/stockfish'),'--out',str(out),'--pairs','32','--batch-games','16','--nodes','1000','--simulations','512'],check=True,stdout=log,stderr=subprocess.STDOUT)
            proof=verify_arena(out,64,checkpoint);atomic_json(WORK/f'evaluation-{name}-verification.json',proof);sync()
        phase[0]='complete';atomic_json(WORK/'pipeline-status.json',dict(phase='complete',updated_at=time.time(),default_promoted=False))
    except Exception as error:
        phase[0]='failed';atomic_json(WORK/'pipeline-error.json',dict(error=repr(error),time=time.time()));raise
    finally:
        stop.set();worker.join();sync();sync()
        # Re-read persistent checkpoints before releasing this paid GPU.
        proof={}
        for file in WORK.rglob('*.pt'):
            relative=file.relative_to(WORK)
            if sha(file)!=sha(SAVE/relative):raise OSError('Checkpoint persistence failed')
            proof[str(relative)]=sha(file)
        atomic_json(WORK/'persistent-checkpoint-verification.json',dict(phase=phase[0],sha256=proof,time=time.time()));sync()
        from neurochess.autostop import request_unassign,find_runtime_address
        request_unassign(find_runtime_address())

if __name__=='__main__':main()
