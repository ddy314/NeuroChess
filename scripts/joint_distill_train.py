"""Joint P/Q/WDL continuation on old replay, new teacher states and mistakes."""
import argparse,json,time,sys,copy,hashlib,fcntl
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch
from neurochess.model import ModelConfig
from neurochess.policy_value_pair import PolicyValuePair,checkpoint
from neurochess.train import losses,atomic_save
from neurochess.data import FIELDS
from neurochess.persistence import RunMirror
from scripts.local_value_repair import write
from scripts.joint_training_runtime import canonical_update, backward_update, remaining_lr, verify_restored_state


def sample_rows(arrays,ends,rng,size):
 indices=rng.integers(int(ends[-1]),size=size);result=np.empty(size,dtype=arrays[0].dtype)
 for i in np.unique(np.searchsorted(ends,indices,side='right')):
  lo=0 if i==0 else ends[i-1];mask=(indices>=lo)&(indices<ends[i]);result[mask]=arrays[i][indices[mask]-lo]
 return result


def train(plan_path,resume=False):
 plan=json.loads(Path(plan_path).read_text());cfg=plan['next_training'];paths=plan['paths'];work=Path(paths['work']);work.mkdir(exist_ok=True,parents=True)
 lock=(work/'training.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if (work/'last.pt').exists() and not resume:raise RuntimeError('Existing checkpoint: resume explicitly instead of overwriting')
 save=Path(paths['save']);save.mkdir(exist_ok=True,parents=True);mirror=RunMirror(work,save,'/content/drive')
 data=Path(paths['new_data']);ready=json.loads(Path(paths['ready']).read_text())
 if ready.get('requires_loss_mode'):
  raise RuntimeError('Masked dynamic curriculum requires its v2 sampler and available-target-normalized-v2 loss; do not silently train it with the legacy loop')
 assert ready['labelled']['train']==plan['authorized_training_positions'] and ready['verified_training_mistakes']>=10000
 assert ready['screened']>=100000 and min(ready['labelled']['val'],ready['labelled']['test'])>=3000
 for name,want in ready['sha256'].items():
  with (data/name).open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==want,name
 torch.set_num_threads(3);torch.manual_seed(20261103);torch.set_float32_matmul_precision('high');rng=np.random.default_rng(20261103)
 parent=Path(paths['parent']);s=torch.load(work/'last.pt' if resume else parent,map_location='cpu',mmap=True,weights_only=False)
 origin=s['step']-s.get('joint_updates',0) if resume else s['step'];assert origin==3000
 model=PolicyValuePair(ModelConfig(**s['model_config']));model.load_state_dict(s['model']);model.set_value_training('joint');model=model.cuda()
 opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=cfg['lr'],weight_decay=.01)
 resume_step=0
 if resume:
  resume_step=s['joint_updates'];opt.load_state_dict(s['optimizer'])
  rng.bit_generator.state=s['numpy_rng'];torch.set_rng_state(s['torch_rng']);torch.cuda.set_rng_state_all(s['cuda_rng'])
  optimizer_steps=sorted({int(v['step'].item()) for v in opt.state.values() if 'step' in v})
  assert optimizer_steps==[resume_step],optimizer_steps
  verified=verify_restored_state(model,opt,s)
  assert rng.bit_generator.state==s['numpy_rng'] and torch.equal(torch.get_rng_state(),s['torch_rng'])
  assert all(torch.equal(a,b) for a,b in zip(torch.cuda.get_rng_state_all(),s['cuda_rng']))
  write(work/'resume-verification.json',dict(resumed_joint_step=resume_step,cumulative_step=s['step'],optimizer_steps=optimizer_steps,optimizer_reset=False,numpy_rng_restored=True,torch_rng_restored=True,cuda_rng_restored=True,exact_restoration=verified,source=str(work/'last.pt'),verified_at=time.time()))
 del s
 pools=dict(old=sorted(Path(paths['old_data']).glob('train-*.npy')),new=sorted(data.glob('train-*.npy')),mistake=sorted((data/'mistakes').glob('train-*.npy')))
 assert all(pools.values());pools={k:[np.load(p,mmap_mode='r') for p in files] for k,files in pools.items()}
 pool_sizes={k:np.cumsum([len(a) for a in arrays]) for k,arrays in pools.items()}
 def sample(pool,size):
  return sample_rows(pools[pool],pool_sizes[pool],rng,size)
 validations=dict(new=np.concatenate([np.load(p) for p in data.glob('val-*.npy')]),general=np.load(paths['general_validation']))
 prior_metrics=[]
 if resume:
  prior_metrics=[json.loads(l) for l in (work/'metrics.jsonl').read_text().splitlines()]
  assert prior_metrics[-1]['step']==resume_step
 prior_seconds=prior_metrics[-1]['session_seconds'] if prior_metrics else 0.
 phase='training';step=resume_step;started=time.time()-prior_seconds;seen=resume_step*320;lineage=dict(parent=str(parent),teacher_ready=ready)
 microbatch=cfg.get('microbatch',16);accum=320//microbatch;resumed_started=time.time()
 def status():
  write(work/'status.json',dict(phase=phase,step=step,total_steps=cfg['steps'],updated_at=time.time()));mirror.sync()
 def batch(r):return {k:torch.as_tensor(np.array(r[k],copy=True),device='cuda') for k in FIELDS}
 def validate(name,records):
  totals=np.zeros(8);mae=[];truth=[];pred=[]
  model.eval()
  with torch.inference_mode():
   for start in range(0,len(records),32):
    r=records[start:start+32];b=batch(r)
    with torch.autocast('cuda',dtype=torch.bfloat16):outputs=model(b['tokens'],b['meta'],b['moves']);loss,parts=losses(outputs,b,.5,1.)
    n=len(r);qmask=b['qmask'];v=outputs[2].float().softmax(-1);pv=v[:,0]-v[:,2];tv=b['value'][:,0]-b['value'][:,2]
    totals+=np.array([loss.item()*n,parts['policy'].item()*n,parts['q'].item()*n,parts['value'].item()*n,(outputs[0].argmax(-1)==b['policy'].argmax(-1)).sum().item(),((outputs[1]-b['q'].float()).abs()*qmask).sum().item(),qmask.sum().item(),n]);pred.extend(pv.cpu().tolist());truth.extend(tv.cpu().tolist())
  err=abs(np.array(pred)-truth);truth=np.array(truth)
  return dict(step=step,split=name,val_loss=totals[0]/totals[7],val_policy_loss=totals[1]/totals[7],val_q_loss=totals[2]/totals[7],val_value_loss=totals[3]/totals[7],val_top1=totals[4]/totals[7],val_q_mae=totals[5]/max(1,totals[6]),q_labels=int(totals[6]),positions=int(totals[7]),value_mae=float(err.mean()),value_strata={k:dict(n=int(m.sum()),mae=float(err[m].mean()) if m.any() else None) for k,m in dict(winning=truth>.25,neutral=abs(truth)<=.25,losing=truth<-.25).items()})
 write(work/'run.json',dict(cfg,batch_size=microbatch,accum=accum,steps=cfg['steps'],origin_step=origin,validation_data=paths['new_data'],training_branches=1));write(work/'dataset.json',dict(counts=dict(train=ready['labelled']['train'],val=len(validations['new'])),provenance=dict(kind='40% old teacher replay +40% new elite +20% confirmed mistake oversampling')));write(work/'runtime.json',dict(device=torch.cuda.get_device_name(),precision='bfloat16',parameters=sum(p.numel() for p in model.parameters()),origin_step=origin,gpu_total_gib=torch.cuda.get_device_properties(0).total_memory/2**30))
 write(work/'lineage.json',dict(parent_checkpoint_sha256=plan['parent_sha256'],optimizer_reset=not resume,initial_optimizer_reset=True,resume_joint_step=resume_step,training_mode='joint',both_encoders_trainable=True,default_promoted=False))
 write(work/'replay-dataset.json',dict(counts=dict(train=int(pool_sizes['old'][-1]))))
 write(work/'validation-dataset.json',dict(counts=dict(val=len(validations['new']))))
 for name,r in ([] if resume else validations.items()):
  v=validate(name,r)
  with (work/('validation.jsonl' if name=='new' else 'general-validation.jsonl')).open('a') as f:f.write(json.dumps(v)+'\n')
 status()
 try:
  for step in range(resume_step+1,cfg['steps']+1):
   model.train();opt.zero_grad(set_to_none=True)
   lr=remaining_lr(step,cfg.get('remaining_schedule'),cfg['lr'])
   for group in opt.param_groups:group['lr']=lr
   rows=canonical_update(sample);total,labels=backward_update(model,rows,microbatch);positions=320
   norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
   if not torch.isfinite(norm):raise FloatingPointError('Nonfinite joint gradient')
   opt.step();seen+=positions
   if step%10==0:
    row=dict(step=step,cumulative_step=origin+step,loss=total.detach().cpu().tolist(),lr=lr,timestamp=time.time(),gradient_norm=float(norm),q_weight=.5,value_weight=1.,q_labels=int(labels),batch_positions=positions,microbatch=microbatch,accumulation=accum,session_positions=seen,session_seconds=time.time()-started,positions_per_second=seen/(time.time()-started),continuation_positions_per_second=(step-resume_step)*320/max(1,time.time()-resumed_started),gpu_allocated_gib=torch.cuda.memory_allocated()/2**30,gpu_reserved_gib=torch.cuda.memory_reserved()/2**30,gpu_peak_gib=torch.cuda.max_memory_allocated()/2**30)
    with (work/'metrics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    status()
   if step%1000==0:
    phase='validating';status()
    for name,r in validations.items():
     report=validate(name,r)
     with (work/('validation.jsonl' if name=='new' else 'general-validation.jsonl')).open('a') as f:f.write(json.dumps(report)+'\n')
    phase='saving';status();atomic_save(checkpoint(model,origin+step,lineage,optimizer=opt.state_dict(),joint_updates=step,numpy_rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),default_promoted=False),work/'last.pt');phase='training';status()
  phase='complete';status()
 except BaseException:
  import traceback
  write(work/'failure.json',dict(step=step,traceback=traceback.format_exc()));phase='failed';status();raise
 # User explicitly retains this instance; no automatic unassign.

if __name__=='__main__':train(sys.argv[1],resume='--resume' in sys.argv[2:])
