"""Local head-only repair with broad original-distribution trust region.

No cloud calls or complete-game matches. Selection uses only numeric validation;
the fixed chess gate is accessed later, after the candidate is frozen.
"""
import argparse,copy,json,hashlib,sys,importlib.util,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch
if importlib.util.find_spec('chess') is None:
 for site in (ROOT/'.venv/lib').glob('python*/site-packages'):
  if (site/'chess').is_dir():sys.path.append(str(site));break
from neurochess.train import load_model
from neurochess.data import DTYPE
from neurochess.encoding import position_key
import chess


def write(path,obj):
 p=Path(path);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2)+'\n');tmp.replace(p)


def metrics(head,x,target):
 ce=[];pred=[]
 with torch.inference_mode():
  for start in range(0,len(x),512):
   logits=head(x[start:start+512]);truth=target[start:start+512]
   ce.extend((-(truth*logits.log_softmax(-1)).sum(-1)).tolist());v=logits.softmax(-1);pred.extend((v[:,0]-v[:,2]).tolist())
 truth=(target[:,0]-target[:,2]).numpy();error=abs(np.array(pred)-truth)
 strata={}
 for k,mask in [('winning',truth>.25),('neutral',abs(truth)<=.25),('losing',truth<-.25)]:
  if mask.any():strata[k]=dict(n=int(mask.sum()),mae=float(error[mask].mean()))
 return dict(ce=float(np.mean(ce)),mae=float(error.mean()),by_stratum=strata)


def eligible(reference,candidate):
 # Stricter calibration-selection limits than the unchanged final chess gate.
 return (candidate['general']['ce']<=reference['general']['ce']*1.01 and
         candidate['general']['by_stratum']['losing']['mae']<=reference['general']['by_stratum']['losing']['mae']+.005 and
         all(candidate['balanced']['by_stratum'][k]['mae']<=reference['balanced']['by_stratum'][k]['mae']+.01 for k in ['winning','neutral','losing']))


def main():
 p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--core',required=True);p.add_argument('--old',required=True);p.add_argument('--out',required=True)
 a=p.parse_args();out=Path(a.out);out.mkdir(exist_ok=True,parents=True);torch.set_num_threads(4);torch.manual_seed(20261103);torch.set_float32_matmul_precision('high')
 torch.cuda.set_per_process_memory_fraction(.28)
 core=Path(a.core);data=Path(a.data)
 def read(folder,split):return np.concatenate([np.load(f) for f in sorted(folder.glob(split+'-*.npy'))])
 records={'repair':read(data/'train','train'),'balanced':read(data/'val','val'),'general':read(data/'general-val','val')}
 blocked={bytes(k).hex() for r in [records['balanced'],records['general']] for k in r['key']}
 gates=json.loads((data/'gate-roots.json').read_text())
 for jobs in gates.values():
  for j in jobs:
   b=chess.Board(j['initial_fen'])
   for u in j['history']:b.push_uci(u)
   blocked.add(position_key(b).hex());b.push_uci(j['solution'][0]);blocked.add(position_key(b).hex())
 sample=[]
 for f in sorted(core.glob('train-*.npy')):
  r=np.load(f,mmap_mode='r');ix=np.linspace(0,len(r)-1,min(128,len(r)),dtype=int)
  for row in r[ix]:
   if bytes(row['key']).hex() not in blocked and row['value_weight']>0:sample.append(row.copy())
 rng=np.random.default_rng(20261103);rng.shuffle(sample);records['anchor']=np.array(sample[:32768],dtype=DTYPE)
 plan=dict(old_sha256=hashlib.file_digest(open(a.old,'rb'),'sha256').hexdigest(),steps=1000,lr=3e-6,batch_per_source=256,
   loss='0.25*repair_CE + general_KL_to_frozen_old +0.25*general_CE',counts={k:len(v) for k,v in records.items()},
   policy_Q='frozen original fresh branch, not loaded during feature cache or CPU head training',selection='Numeric balanced/general validation only; fixed chess gate untouched until checkpoint selection',cloud_allowed=False)
 assert plan['old_sha256']=='e4ed8978db91eeef8360c4dea87faf7a4f826325bce45aa7efc024692d585ef7'
 write(out/'plan.json',plan)
 model,state=load_model(a.old,'cuda');model.eval();del state
 initial_head=copy.deepcopy(model.value).cpu();head=copy.deepcopy(initial_head)
 features={};targets={}
 for name,r in records.items():
  cache=out/(name+'-features.npz')
  if not cache.exists():
   chunks=[]
   with torch.inference_mode():
    for start in range(0,len(r),16):
     batch=r[start:start+16]
     with torch.autocast('cuda',dtype=torch.bfloat16):x=model.encode_squares(torch.as_tensor(batch['tokens'].copy(),device='cuda'),torch.as_tensor(batch['meta'].copy(),device='cuda')).mean(1)
     chunks.append(x.float().cpu().numpy().astype(np.float16))
     if start%1024==0:write(out/'status.json',dict(phase='feature-cache',dataset=name,completed=start,total=len(r),updated_at=time.time()))
   np.savez(cache,x=np.concatenate(chunks),value=r['value'].copy(),key=r['key'].copy())
  cached=np.load(cache);assert np.array_equal(cached['key'],r['key'])
  features[name]=torch.from_numpy(cached['x'].astype(np.float32));targets[name]=torch.from_numpy(cached['value'].copy())
 del model;torch.cuda.empty_cache()
 with torch.inference_mode():anchor=initial_head(features['anchor']).softmax(-1)
 anchor=anchor.clone()
 reference={k:metrics(initial_head,features[k],targets[k]) for k in ['balanced','general']};write(out/'initial-validation.json',reference)
 opt=torch.optim.AdamW(head.parameters(),lr=3e-6,weight_decay=.01);best=reference['balanced']['mae'];selected=None
 for step in range(1,1001):
  ri=torch.randint(len(features['repair']),(256,));gi=torch.randint(len(features['anchor']),(256,));opt.zero_grad(set_to_none=True)
  rl=head(features['repair'][ri]);gl=head(features['anchor'][gi]);repair_ce=-(targets['repair'][ri]*rl.log_softmax(-1)).sum(-1).mean();general_ce=-(targets['anchor'][gi]*gl.log_softmax(-1)).sum(-1).mean();kl=torch.nn.functional.kl_div(gl.log_softmax(-1),anchor[gi],reduction='batchmean')
  loss=.25*repair_ce+kl+.25*general_ce;loss.backward();norm=torch.nn.utils.clip_grad_norm_(head.parameters(),1.)
  if not torch.isfinite(norm):raise FloatingPointError('Nonfinite head gradient')
  opt.step()
  if step%50==0:
   candidate={k:metrics(head,features[k],targets[k]) for k in ['balanced','general']};ok=eligible(reference,candidate)
   row=dict(step=step,loss=float(loss.detach()),repair_ce=float(repair_ce.detach()),general_ce=float(general_ce.detach()),anchor_kl=float(kl.detach()),eligible=ok,validation=candidate)
   with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
   if ok and candidate['balanced']['mae']<best:
    best=candidate['balanced']['mae'];selected=dict(step=step,validation=candidate);torch.save(dict(value_head=head.state_dict(),step=step,plan=plan,validation=candidate),out/'selected-head.pt')
   write(out/'status.json',dict(phase='local-head-training',step=step,total_steps=1000,selected=selected,updated_at=time.time()));print(json.dumps(row),flush=True)
 write(out/'result.json',dict(complete=True,reference=reference,selected=selected,improvement=reference['balanced']['mae']-best,gate_passed=False,note='Candidate frozen by numeric validation; chess gate not yet run.'))

if __name__=='__main__':main()
