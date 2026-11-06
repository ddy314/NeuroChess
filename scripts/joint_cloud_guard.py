"""Prepare one audited joint run, persist its results, then release its VM."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time
import traceback
from neurochess.autostop import request_unassign, find_runtime_address
from neurochess.persistence import RunMirror


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def write(path,data):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2));tmp.replace(path)


def main():
    lock=Path('/tmp/joint-cloud.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    plan=json.loads(Path('/content/joint-plan.json').read_text());paths=plan['paths']
    work=Path(paths['work']);work.mkdir(exist_ok=True,parents=True)
    save=Path(paths['save']);save.mkdir(exist_ok=True,parents=True)
    assert os.path.ismount('/content/drive')
    mirror=RunMirror(work,save,'/content/drive');step=0;phase='preparing';success=False
    def status():
        write(work/'status.json',dict(phase=phase,step=step,total_steps=plan['next_training']['steps'],updated_at=time.time()));mirror.sync()
    try:
        write(work/'run.json',dict(plan['next_training'],batch_size=16,accum=20,steps=plan['next_training']['steps']))
        status()
        persistent=save.parent
        delivery=json.loads(Path('/content/joint-delivery.json').read_text())
        for name,want in delivery.items():
            source=Path('/content/joint-'+name);assert sha(source)==want,name
            shutil.copyfile(source,persistent/name);assert sha(persistent/name)==want,name
        shutil.copyfile('/content/joint-plan.json',work/'plan.json')
        shutil.copyfile('/content/joint-delivery.json',work/'delivery.json')
        with tarfile.open('/content/joint-data.tar.gz') as archive:
            archive.extractall('/content/joint-inputs',filter='data')
        for name,want in json.loads(Path(paths['ready']).read_text())['sha256'].items():
            assert sha(Path(paths['new_data'])/name)==want,name
        pilot=Path('/content/drive/MyDrive/neurochess/targeted-distill-20261102')
        shutil.copyfile(pilot/'last.pt',paths['parent']);assert sha(paths['parent'])==plan['parent_sha256']
        shutil.copyfile(pilot/'general-validation.npy',paths['general_validation'])
        cache=Path('/content/old-core.tar.gz')
        shutil.copyfile('/content/drive/MyDrive/neurochess-data/evals-v2-5m.tar.gz',cache)
        assert sha(cache)==plan['core_sha256']
        with tarfile.open(cache) as archive:archive.extractall(paths['old_data'],filter='data')
        for name,want in json.loads((Path(paths['old_data'])/'checksums.json').read_text()).items():
            assert sha(Path(paths['old_data'])/name)==want,name
        shutil.copyfile('/content/drive/MyDrive/neurochess/targeted-scale-20261102/migration-proof.json',work/'parent-migration-proof.json')
        kit='/content/drive/MyDrive/neurochess/curriculum-v3-teacher/teacher-kit.tar.gz'
        with tarfile.open(kit) as archive:
            members=[m for m in archive.getmembers() if m.name=='bin/stockfish'];assert len(members)==1
            archive.extractall('/content/neurochess',members=members,filter='data')
        Path('/content/neurochess/bin/stockfish').chmod(0o755)
        assert sha('/content/neurochess/bin/stockfish')=='6b087694916228c905a5e14db74cca8c7e5643602226af1fa5d42353c455b9f9'
        subprocess.run([sys.executable,'scripts/build_native_mcts.py'],check=True)
        write(work/'preparation-verification.json',dict(ready=True,parent_sha256=sha(paths['parent']),data_archive_sha256=delivery['data.tar.gz'],code_sha256=delivery['source.tar.gz'],new_positions=plan['authorized_training_positions'],verified_at=time.time()))
        phase='training';status()
        subprocess.run([sys.executable,'-u','-m','scripts.joint_distill_train','/content/joint-plan.json'],check=True,timeout=43200)
        step=plan['next_training']['steps'];phase='fixed-position-evaluation';status()
        try:
            subprocess.run([sys.executable,'-u','-m','scripts.joint_distill_evaluate','/content/joint-plan.json'],check=True,timeout=600)
        except Exception:
            write(work/'evaluation-failure.json',dict(traceback=traceback.format_exc(),training_completed=True))
        phase='verifying-persistence';status()
        mirror.sync()
        import torch
        state=torch.load(save/'last.pt',map_location='cpu',mmap=True,weights_only=False)
        assert state['joint_updates']==step and state['step']==3000+step and 'optimizer' in state
        assert state['value_training_mode']=='joint' and state['policy_q_frozen'] is False
        del state
        hashes={str(p.relative_to(work)):sha(p) for p in work.rglob('*') if p.is_file() and p.suffix!='.tmp' and p.name!='training.lock'}
        for relative,want in hashes.items():assert sha(save/relative)==want,relative
        write(work/'persistence-verification.json',dict(joint_updates=step,sha256=hashes,verified_at=time.time(),default_promoted=False,evaluation_complete=(work/'fixed-position-evaluation.json').exists() and json.loads((work/'fixed-position-evaluation.json').read_text())['completed_pairs']==48))
        phase='complete';status();success=True
    except BaseException:
        write(work/'guard-failure.json',dict(phase=phase,traceback=traceback.format_exc(),updated_at=time.time()))
        phase='failed';status()
        raise
    finally:
        # Keep the complete console and failure state before releasing a paid VM.
        shutil.copyfile('/content/joint-launcher.log',work/'console.log')
        mirror.sync()
        write(save/'shutdown-request.json',dict(success=success,requested_at=time.time(),reason='Completed and persisted' if success else 'Failure state persisted; avoid idle GPU'))
        request_unassign(find_runtime_address())


if __name__=='__main__':main()
