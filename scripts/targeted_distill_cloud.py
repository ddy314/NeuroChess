"""One cloud run: screen real continuations, persist audited mistakes, mix replay, train.

No default promotion. Fixed 2000-update window; small interim score fluctuations
are recorded, not automatic stopping events. Every source family has one split.
"""
import argparse,concurrent.futures,copy,fcntl,hashlib,json,multiprocessing,os,shutil,sqlite3,sys,tarfile,threading,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import chess,chess.engine,numpy as np,torch
from neurochess.data import record,DTYPE,ShardWriter
from neurochess.targets import assign_targets
from neurochess.encoding import encode,legal_ids,position_key
from neurochess.multihead import MultiheadPlayer
from neurochess.policy_value_pair import from_original_states,checkpoint,policy_digest
from scripts.targeted_mistakes import classify,stable_labels,needs_repair
from scripts.local_value_repair import write
from scripts.value_sustained_cloud import sha,gpu_metrics
from neurochess.persistence import RunMirror
from neurochess.autostop import request_unassign,find_runtime_address

ENGINE=None

def teacher_init(binary):
 global ENGINE
 ENGINE=chess.engine.SimpleEngine.popen_uci(binary,timeout=120)
 ENGINE.configure({'Threads':1,'Hash':64,'Skill Level':20,'UCI_LimitStrength':False})
 # python-chess owns a non-daemon asyncio thread. multiprocessing Finalize
 # runs after Python joins non-daemon threads, so quitting there deadlocks
 # pool shutdown. Close the engine before that thread join instead.
 threading._register_atexit(ENGINE.quit)


def label(task):
 job,pred,*extra=task;b=chess.Board(job['initial_fen'])
 for u in job['history']:b.push_uci(u)
 if position_key(b).hex()!=job['key']:raise ValueError('Node key/history mismatch')
 coarse=ENGINE.analyse(b,chess.engine.Limit(nodes=10000),multipv=min(4,b.legal_moves.count()),game=object())
 moves=list(dict.fromkeys([i['pv'][0].uci() for i in coarse]+pred['top_moves']))
 def analyze(nodes):
  infos=ENGINE.analyse(b,chess.engine.Limit(nodes=nodes),root_moves=[b.parse_uci(m) for m in moves],multipv=len(moves),game=object())
  pvs=[]
  for i in infos:
   s=i['score'].white();p=dict(line=' '.join(m.uci() for m in i['pv']),nodes=i.get('nodes'),depth=i.get('depth'))
   p['mate' if s.is_mate() else 'cp']=s.mate() if s.is_mate() else s.score();pvs.append(p)
  return pvs
 first=analyze(30000);decision=classify(first,pred['move'],pred['value'],b.turn);stable=None;deep=None
 if needs_repair(decision) or (extra and extra[0]):
  first=analyze(100000);deep=analyze(300000);stable=stable_labels(first,deep,b.turn)
  decision=classify(deep,pred['move'],pred['value'],b.turn)
 final=deep or first;r,legal=record(b);assign_targets(r,b,legal,final)
 usable=bool(needs_repair(decision) and stable)
 # Full labels retained for validation. Unstable rows never train Q/value/policy.
 return dict(job=job,prediction=pred,teacher=final,teacher_previous=first,stable=stable,
             decision=decision,repair=usable,teacher_move=decision['best']),r


def predict(model,jobs):
 rows=[]
 for j in jobs:
  b=chess.Board(j['initial_fen'])
  for u in j['history']:b.push_uci(u)
  rows.append(record(b)[0])
 a=np.array(rows,dtype=DTYPE);net=MultiheadPlayer(model,'cuda');raw=net.raw_outputs(a['tokens'].copy(),a['meta'].copy(),a['moves'].copy())
 from neurochess.multihead import fuse_heads
 fused,_,values,_=fuse_heads(raw[:,:256],raw[:,256:512],raw[:,512:],a['moves']>=0)
 result=[]
 for i,j in enumerate(jobs):
  b=chess.Board(j['initial_fen'])
  for u in j['history']:b.push_uci(u)
  legal=list(b.legal_moves);rank=np.argsort(-fused[i,:len(legal)])[:2]
  result.append(dict(move=legal[rank[0]].uci(),top_moves=[legal[k].uci() for k in rank],value=float(values[i]),wdl=raw[i,512:].tolist()))
 return result


def main():
 p=argparse.ArgumentParser();p.add_argument('--plan',required=True);a=p.parse_args();plan=json.loads(Path(a.plan).read_text())
 mount=Path('/content/drive');assert os.path.ismount(mount)
 lock=open('/tmp/targeted-distill.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 work=Path('/content/targeted-distill');work.mkdir(exist_ok=True);save=mount/'MyDrive/neurochess/targeted-distill-20261102';save.mkdir(parents=True,exist_ok=True)
 mirror=RunMirror(work,save,mount);phase='setup';step=0;model=None;optimizer=None
 def status(**kw):
  write(work/'status.json',dict(phase=phase,step=step,updated_at=time.time(),**kw));mirror.sync()
 try:
  write(work/'plan.json',plan);torch.set_num_threads(3);torch.manual_seed(plan['seed']);torch.set_float32_matmul_precision('high')
  parents=[]
  for key in ['policy','value']:
   f=Path(plan['parents'][key]['path']);assert sha(f)==plan['parents'][key]['sha256'];parents.append(torch.load(f,map_location='cpu',weights_only=False,mmap=True))
  model=from_original_states(*parents,'cuda').eval();del parents
  head=Path('/content/targeted-inputs/head-1000.pt');assert sha(head)==plan['head_sha256']
  model.value_model.value.load_state_dict(torch.load(head,map_location='cpu',weights_only=False)['value_head'])
  policy_sha=policy_digest(model);initial_value=copy.deepcopy(model.value_model.state_dict())
  core=Path('/content/targeted-core');core.mkdir(exist_ok=True)
  cache=mount/'MyDrive/neurochess-data/evals-v2-5m.tar.gz';assert sha(cache)==plan['core_sha256']
  with tarfile.open(cache) as t:t.extractall(core,filter='data')
  checks=json.loads((core/'checksums.json').read_text())
  for name,want in checks.items():assert sha(core/name)==want,name
  forbidden=set(json.loads(Path('/content/targeted-inputs/frozen.json').read_text())['keys'])
  # Full old core exclusion for every generated node, not just selected roots.
  for f in core.glob('*.npy'):
   for k in np.load(f,mmap_mode='r')['key']:forbidden.add(bytes(k).hex())
  roots=json.loads(Path('/content/targeted-inputs/roots.json').read_text());owners={};records=[];metadata=[];seen=set();phase='screening'
  # Validation/test first; any transposition into a different family is dropped.
  roots.sort(key=lambda j:({'test':0,'val':1,'train':2}[j['split']],j['group']))
  with concurrent.futures.ProcessPoolExecutor(max_workers=plan['workers'],mp_context=multiprocessing.get_context('spawn'),initializer=teacher_init,initargs=(str(ROOT/'bin/stockfish'),)) as pool:
   for offset in range(0,len(roots),32):
    chunk=roots[offset:offset+32];front=[]
    for root in chunk:
     for branch in ['teacher','student']:
      front.append(dict(root,branch=branch,depth=0))
    for depth in range(5):
     active=[]
     for j in front:
      b=chess.Board(j['initial_fen'])
      for u in j['history']:b.push_uci(u)
      if b.is_game_over(claim_draw=True):continue
      j=dict(j,key=position_key(b).hex());active.append(j)
     if not active:break
     predictions=predict(model,active)
     # Identical effective nodes from both branches share a teacher result.
     tasks={j['key']:(j,pred) for j,pred in zip(active,predictions)}
     unique=list(tasks)
     result_by_key=dict(zip(unique,pool.map(label,[tasks[k] for k in unique])))
     results=[result_by_key[j['key']] for j in active];following=[]
     for (meta,row),j in zip(results,active):
      meta=dict(meta,job=j)
      key=j['key'];owner=owners.get(key)
      if key not in forbidden and key not in seen and (owner is None or owner==j['group']):
       owners[key]=j['group'];seen.add(key);records.append(row);metadata.append(meta)
      u=meta['teacher_move'] if j['branch']=='teacher' else meta['prediction']['move']
      following.append(dict(j,history=j['history']+[u],depth=depth+1))
     front=following
    np.save(work/'screened.npy',np.array(records,dtype=DTYPE))
    (work/'screened.jsonl').write_text(''.join(json.dumps(m)+'\n' for m in metadata))
    status(roots_completed=min(offset+32,len(roots)),roots_total=len(roots),unique_states=len(records),verified_mistakes=sum(m['repair'] for m in metadata))
    if len(records)>=plan['screen_states']:break
  if len(records)<plan['screen_states']:raise ValueError(f'Insufficient real new states: {len(records)}; no repetition padding')
  phase='auditing';array=np.array(records,dtype=DTYPE)
  selected={s:[i for i,m in enumerate(metadata) if m['job']['split']==s and (m['repair'] if s=='train' else True)] for s in ['train','val','test']}
  groups={s:{metadata[i]['job']['group'] for i in ix} for s,ix in selected.items()}
  assert not groups['train']&(groups['val']|groups['test']) and not groups['val']&groups['test']
  for s,ix in selected.items():np.save(work/(s+'.npy'),array[ix])
  audit=dict(unique_states=len(array),counts={s:len(ix) for s,ix in selected.items()},groups={s:len(v) for s,v in groups.items()},old_core_exact_overlap=0,cross_split_key_overlap=0,
    teacher='30k screen; suspected errors independently rechecked at100k and300k same candidate union;30cp stability',history='Full legal source and continuation history',historical_source_game_overlap='Unknown for historical core',ready=len(selected['train'])>=128 and len(selected['test'])>=128)
  write(work/'data-audit.json',audit);status()
  if not audit['ready']:raise ValueError('Too few verified errors/test states for planned training; saved screening results')
  phase='training';hard=np.load(work/'train.npy');general=np.load(core/'val-000000.npy',mmap_mode='r')[:4096].copy()
  # Explicitly exclude all new val/test keys from replay (already excluded by old core audit).
  train_files=sorted(core.glob('train-*.npy'));rng=np.random.default_rng(plan['seed']);optimizer=torch.optim.AdamW(model.value_model.parameters(),lr=plan['lr'],weight_decay=.01)
  model.set_value_training('backbone');model.eval();ref_validation=gpu_metrics(model,general);write(work/'initial-validation.json',ref_validation)
  np.save(work/'general-validation.npy',general)
  # 80% broad replay and20% verified mistakes, exactly four +one microbatches.
  for step in range(1,plan['steps']+1):
   optimizer.zero_grad(set_to_none=True);loss_total=0.
   for slot in range(5):
    source=hard if slot==4 else np.load(rng.choice(train_files),mmap_mode='r')
    r=source[rng.integers(len(source),size=plan['microbatch'])].copy();t=torch.as_tensor(r['tokens'],device='cuda');m=torch.as_tensor(r['meta'],device='cuda');y=torch.as_tensor(r['value'],device='cuda')
    with torch.autocast('cuda',dtype=torch.bfloat16):logits=model.value_only(t,m).float()
    loss=(-(y*logits.log_softmax(-1)).sum(-1)).mean()/5
    loss.backward();loss_total+=loss.item()
   norm=torch.nn.utils.clip_grad_norm_([v for v in model.parameters() if v.requires_grad],1.)
   if not torch.isfinite(norm):raise FloatingPointError('Nonfinite gradient')
   optimizer.step()
   if step%50==0:
    with (work/'metrics.jsonl').open('a') as f:f.write(json.dumps(dict(step=step,loss=loss_total,timestamp=time.time()))+'\n')
    status(total_steps=plan['steps'])
   if step%500==0:
    validation=gpu_metrics(model,general);write(work/f'validation-{step}.json',validation)
    torch.save(checkpoint(model,1000+step,dict(plan=plan),optimizer=optimizer.state_dict(),default_promoted=False),work/'last.pt');status(total_steps=plan['steps'])
  assert policy_digest(model)==policy_sha
  # Keep the complete window; final fixed-budget paired repeated search is observational.
  phase='search-evaluation';from neurochess.native_mcts import NativeMCTSPlayer
  config=json.loads((ROOT/'configs/search-baseline-lock.json').read_text())['config'];test_jobs=[]
  for i in selected['test']:
   j=metadata[i]['job']
   if j['depth'] in [0,2,4]:test_jobs.append((i,j))
  rng.shuffle(test_jobs);test_jobs=test_jobs[:32];assert len(test_jobs)==32
  trained=copy.deepcopy(model.value_model.state_dict());rows=[]
  for repeat in range(3):
   for index,(i,j) in enumerate(test_jobs):
    pair=[]
    for mode in (['initial','trained'] if (repeat+index)%2==0 else ['trained','initial']):
     model.value_model.load_state_dict(initial_value if mode=='initial' else trained);b=chess.Board(j['initial_fen'])
     for u in j['history']:b.push_uci(u)
     player=NativeMCTSPlayer(MultiheadPlayer(model,'cuda'),**config)
     try:move=player.choose(b);stats=player.search_stats
     finally:player.close()
     pair.append(dict(repeat=repeat,mode=mode,key=j['key'],group=j['group'],depth=j['depth'],move=move.uci(),stats=stats))
    # Both choices are adjudicated in ONE candidate union at the same budget.
    teacher_init(str(ROOT/'bin/stockfish'))
    try:meta,_=label((j,dict(move=pair[0]['move'],top_moves=list(dict.fromkeys(r['move'] for r in pair)),value=metadata[i]['prediction']['value']),True))
    finally:ENGINE.quit()
    b=chess.Board(j['initial_fen'])
    for u in j['history']:b.push_uci(u)
    for row in pair:
     row.update(decision=classify(meta['teacher'],row['move'],metadata[i]['prediction']['value'],b.turn),teacher_stable=meta['stable'])
     rows.append(row)
    write(work/'search-evaluation.json',dict(rows=rows,protocol='3 counterbalanced repetitions;32 fresh nodes across depths0/2/4, locked2s config; teacher regret, no whole games; not automatic promotion'));status(completed_pairs=len(rows)//2,total_pairs=96)
  model.value_model.load_state_dict(trained)
  write(work/'completion.json',dict(completed=True,steps=step,training_branches=1,default_promoted=False,selection='Final model retained for review; no early small-sample gate or automatic promotion'))
  phase='complete';status()
 except BaseException:
  write(work/'failure.json',dict(phase=phase,step=step,traceback=traceback.format_exc()));print(traceback.format_exc(),flush=True)
  if model is not None and step:
   torch.save(checkpoint(model,1000+step,dict(plan=plan),optimizer=optimizer.state_dict() if optimizer else None,unvalidated=True),work/'failure.pt')
  phase='failed';status()
 finally:
  mirror.sync();hashes={str(f.relative_to(work)):sha(f) for f in work.rglob('*') if f.is_file() and f.suffix!='.tmp'}
  for rel,want in hashes.items():assert sha(save/rel)==want,rel
  write(work/'persistence-verification.json',dict(phase=phase,sha256=hashes));mirror.sync()
  assert sha(work/'persistence-verification.json')==sha(save/'persistence-verification.json')
  print('Persisted; releasing runtime',phase,flush=True);request_unassign(find_runtime_address())

if __name__=='__main__':main()
