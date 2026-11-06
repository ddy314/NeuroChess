"""Switch a paused checkpoint to measured batching without losing updates."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback
from neurochess.persistence import RunMirror
from neurochess.autostop import request_unassign,find_runtime_address


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def write(path,data):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2));tmp.replace(path)


def main():
    lock=Path('/tmp/joint-resume-guard.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    plan_path=Path('/content/joint-plan.json');plan=json.loads(plan_path.read_text());paths=plan['paths']
    work=Path(paths['work']);save=Path(paths['save']);mirror=RunMirror(work,save,'/content/drive')
    deadline=time.time()+3600
    while not Path('/content/checkpoint-pause.json').exists():
        if time.time()>deadline:raise TimeoutError('Checkpoint boundary not reached; original trainer unchanged')
        time.sleep(2)
    pause=json.loads(Path('/content/checkpoint-pause.json').read_text())
    if 'failure' in pause:raise RuntimeError(pause['failure'])
    # The watcher stops the old trainer immediately after atomic checkpoint rename.
    import torch
    torch.set_num_threads(3)
    state=torch.load(work/'last.pt',map_location='cpu',mmap=True,weights_only=False)
    boundary=state['joint_updates'];assert boundary==7000 and state['step']==10000
    assert 'optimizer' in state and all(k in state for k in ['numpy_rng','torch_rng','cuda_rng'])
    assert {int(v['step']) for v in state['optimizer']['state'].values()}=={boundary}
    assert json.loads((work/'metrics.jsonl').read_text().splitlines()[-1])['step']==boundary
    del state
    write(work/'status.json',dict(phase='saving',step=boundary,total_steps=10000,updated_at=time.time(),note='Preserving complete7000-step checkpoint before batching/schedule continuation'))
    before=sha(work/'last.pt');backup=save.parent/'before-optimization-step7000.pt'
    shutil.copyfile(work/'last.pt',backup);assert sha(backup)==before
    local_backup=Path('/content/joint-before-optimization.pt')
    if not local_backup.exists():os.link(work/'last.pt',local_backup)
    write(work/'checkpoint-transition.json',dict(joint_step=boundary,checkpoint_sha256=before,persistent_backup=str(backup),weights_and_optimizer_preserved=True,updated_at=time.time()))
    shutil.copyfile(plan_path,work/'plan-before-optimization.json')
    plan['next_training'].update(microbatch=64,accumulation=5,remaining_schedule=dict(kind='cosine',start_step=boundary,end_step=10000,initial_lr=1e-6,min_lr=3e-7))
    write(plan_path,plan);write(work/'plan.json',plan)
    proof=json.loads(Path('/content/optimization-delivery.json').read_text())
    for relative,want in proof.items():
        assert sha(Path('/content/neurochess')/relative)==want,relative
    shutil.copyfile('/content/optimization-code.tar.gz',work/'optimization-code.tar.gz')
    shutil.copyfile('/content/optimization-delivery.json',work/'optimization-delivery.json')
    write(work/'optimization-plan.json',dict(start_step=boundary,total_steps=10000,effective_batch=320,old_microbatch=16,new_microbatch=64,accumulation=5,sampling_mixture_unchanged=True,original_per16_loss_normalization_preserved=True,schedule=plan['next_training']['remaining_schedule'],benchmark='batch-benchmark.json',optimizer_reset=False,source_checkpoint_sha256=before))
    mirror.sync()
    # Both originals are stopped; release their locks only after durable backup.
    for pid,needle in [(pause['guard'],b'scripts.joint_cloud_guard'),(pause['child'],b'scripts.joint_distill_train')]:
        assert needle in Path(f'/proc/{pid}/cmdline').read_bytes()
        os.kill(pid,signal.SIGKILL)
    for _ in range(100):
        if not Path(f"/proc/{pause['child']}").exists():break
        time.sleep(.1)
    success=False;step=boundary
    try:
        subprocess.run([sys.executable,'-u','-m','scripts.joint_distill_train',str(plan_path),'--resume'],check=True,timeout=18000)
        step=10000
        write(work/'status.json',dict(phase='fixed-position-evaluation',step=step,total_steps=10000,updated_at=time.time()));mirror.sync()
        try:
            subprocess.run([sys.executable,'-u','-m','scripts.joint_distill_evaluate',str(plan_path)],check=True,timeout=600)
        except Exception:write(work/'evaluation-failure.json',dict(traceback=traceback.format_exc(),training_completed=True))
        mirror.sync()
        state=torch.load(save/'last.pt',map_location='cpu',mmap=True,weights_only=False)
        assert state['joint_updates']==step and state['step']==13000 and 'optimizer' in state
        assert state['value_training_mode']=='joint' and not state['policy_q_frozen'];del state
        hashes={str(p.relative_to(work)):sha(p) for p in work.rglob('*') if p.is_file() and p.suffix!='.tmp' and p.name!='training.lock'}
        for relative,want in hashes.items():assert sha(save/relative)==want,relative
        write(work/'persistence-verification.json',dict(joint_updates=step,sha256=hashes,verified_at=time.time(),default_promoted=False,evaluation_complete=(work/'fixed-position-evaluation.json').exists() and json.loads((work/'fixed-position-evaluation.json').read_text())['completed_pairs']==48))
        write(work/'status.json',dict(phase='complete',step=step,total_steps=10000,updated_at=time.time()));mirror.sync();success=True
    except BaseException:
        write(work/'resume-guard-failure.json',dict(traceback=traceback.format_exc(),updated_at=time.time()))
        status=json.loads((work/'status.json').read_text());step=status.get('step',step)
        write(work/'status.json',dict(phase='failed',step=step,total_steps=10000,updated_at=time.time()));mirror.sync();raise
    finally:
        shutil.copyfile('/content/joint-resume-launcher.log',work/'continuation-console.log');mirror.sync()
        write(save/'shutdown-request.json',dict(success=success,requested_at=time.time(),reason='Continuation finished and persisted' if success else 'Continuation failure preserved'))
        request_unassign(find_runtime_address())


if __name__=='__main__':main()
