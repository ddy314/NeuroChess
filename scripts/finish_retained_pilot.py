"""Recover the pilot from its intact persistent optimizer checkpoint; keep VM."""
import json,shutil,time,sys,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch,numpy as np
from neurochess.model import ModelConfig
from neurochess.policy_value_pair import PolicyValuePair,checkpoint,policy_digest
from scripts.value_sustained_cloud import gpu_metrics,sha
from scripts.local_value_repair import write


def main():
 work=Path('/content/targeted-distill');save=Path('/content/drive/MyDrive/neurochess/targeted-distill-20261102');plan=json.loads((work/'plan.json').read_text())
 parent=save/'last.pt';s=torch.load(parent,map_location='cpu',mmap=True,weights_only=False);start=s['step']-1000;assert start==1500
 torch.set_num_threads(3);torch.set_float32_matmul_precision('high')
 model=PolicyValuePair(ModelConfig(**s['model_config']));model.load_state_dict(s['model']);model.set_value_training('backbone');model=model.to('cuda').eval()
 optimizer=torch.optim.AdamW(model.value_model.parameters(),lr=plan['lr'],weight_decay=.01);optimizer.load_state_dict(s['optimizer']);lineage=s['lineage'];del s
 core=Path('/content/targeted-core');files=sorted(core.glob('train-*.npy'));hard=np.load(work/'train.npy');rng=np.random.default_rng(plan['seed'])
 sizes={str(f):len(np.load(f,mmap_mode='r')) for f in files}
 for _ in range(start):
  for slot in range(5):
   n=len(hard) if slot==4 else sizes[str(rng.choice(files))]
   rng.integers(n,size=plan['microbatch'])
 # Preserve original logs, then remove only the uncheckpointed tail before replay.
 shutil.copyfile(work/'metrics.jsonl',work/'pre-recovery-metrics.jsonl')
 rows=[json.loads(l) for l in (work/'metrics.jsonl').read_text().splitlines()];(work/'metrics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows if r['step']<=start))
 old=work/'last.pt'
 if old.exists():old.rename(work/'interrupted-checkpoint.bin')
 before=policy_digest(model)
 for step in range(start+1,2001):
  optimizer.zero_grad(set_to_none=True);total=0.
  for slot in range(5):
   source=hard if slot==4 else np.load(rng.choice(files),mmap_mode='r');r=source[rng.integers(len(source),size=plan['microbatch'])].copy()
   t=torch.as_tensor(r['tokens'],device='cuda');m=torch.as_tensor(r['meta'],device='cuda');y=torch.as_tensor(r['value'],device='cuda')
   with torch.autocast('cuda',dtype=torch.bfloat16):v=model.value_only(t,m).float()
   loss=(-(y*v.log_softmax(-1)).sum(-1)).mean()/5;loss.backward();total+=loss.item()
  norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.);assert torch.isfinite(norm);optimizer.step()
  if step%50==0:
   with (work/'metrics.jsonl').open('a') as f:f.write(json.dumps(dict(step=step,loss=total,timestamp=time.time(),recovery=True))+'\n')
   status=dict(phase='training',step=step,total_steps=2000,updated_at=time.time(),note='从完整1500步检查点与优化器恢复，补跑500步；实例保留。')
   write(work/'status.json',status);shutil.copyfile(work/'status.json',save/'status.json')
 assert before==policy_digest(model)
 write(work/'validation-2000.json',gpu_metrics(model,np.load(work/'general-validation.npy')))
 temp=work/'last.tmp';torch.save(checkpoint(model,3000,lineage,optimizer=optimizer.state_dict(),default_promoted=False,recovered_from_step=2500),temp);temp.replace(work/'last.pt')
 restored=torch.load(work/'last.pt',map_location='cpu',mmap=True,weights_only=False);assert restored['step']==3000 and restored['optimizer'];del restored
 for name in ['last.pt','metrics.jsonl','validation-2000.json']:
  temp=save/(name+'.tmp');shutil.copyfile(work/name,temp);assert sha(temp)==sha(work/name);temp.replace(save/name)
 proof=dict(training_updates=2000,parent_step=2500,additional_recovered_updates=500,optimizer_restored=True,rng_sampling_replayed_from_seed=True,policy_Q_frozen=True,checkpoint_sha256=sha(work/'last.pt'),persistent_checkpoint_sha256=sha(save/'last.pt'),keep_instance=True)
 for p in [work/'pilot-training-verification.json',save/'pilot-training-verification.json']:write(p,proof)
 status=dict(phase='paused',step=2000,total_steps=2000,updated_at=time.time(),note='2000步试训已完整保存并校验；实例保留，准备新增精英数据与三头联合训练。')
 for p in [work/'status.json',save/'status.json']:write(p,status)
 print(json.dumps(proof),flush=True)

if __name__=='__main__':main()
