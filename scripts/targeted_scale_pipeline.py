"""Retained-VM continuation: new elite data, verified mistakes, joint training."""
import concurrent.futures,hashlib,json,multiprocessing,sys,time,traceback,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import chess,chess.engine,numpy as np,torch
from neurochess.data import record,DTYPE,ShardWriter
from neurochess.targets import assign_targets
from neurochess.encoding import position_key
from neurochess.model import ModelConfig
from neurochess.policy_value_pair import PolicyValuePair
from scripts import targeted_distill_cloud as teacher
from scripts.local_value_repair import write
from scripts.value_sustained_cloud import sha


def general_label(job):
 b=chess.Board(job['initial_fen'])
 for u in job['history']:b.push_uci(u)
 assert position_key(b).hex()==job['key'],'History/position key mismatch'
 infos=teacher.ENGINE.analyse(b,chess.engine.Limit(nodes=100000),multipv=min(4,b.legal_moves.count()),game=object());pvs=[]
 for i in infos:
  score=i['score'].white();p=dict(line=' '.join(m.uci() for m in i['pv']),depth=i.get('depth'),nodes=i.get('nodes'))
  p['mate' if score.is_mate() else 'cp']=score.mate() if score.is_mate() else score.score();pvs.append(p)
 row,legal=record(b);assign_targets(row,b,legal,pvs)
 return dict(job=job,teacher=pvs,repair=False,stable=None,prediction=None),row


def label_task(task):
 job,pred=task
 return teacher.label((job,pred)) if pred is not None else general_label(job)


def main():
 work=Path('/content/targeted-scale');save=Path('/content/drive/MyDrive/neurochess/targeted-scale-20261102');work.mkdir(exist_ok=True);save.mkdir(exist_ok=True,parents=True)
 phase='waiting-for-candidates';progress={}
 def status(**kw):
  progress.update(kw);s=dict(phase=phase,updated_at=time.time(),**progress)
  for p in [work/'pipeline-status.json',save/'pipeline-status.json']:write(p,s)
  print(json.dumps(s),flush=True)
 try:
  while not (work/'candidate-ready.json').exists() or not Path('/content/targeted-distill/pilot-training-verification.json').exists():
   if (work/'failure.json').exists():
    failure=json.loads((work/'failure.json').read_text())
    if 'Sources exhausted before unique-position target' not in failure.get('traceback',''):raise RuntimeError('Candidate collection failed; inspect saved complete failure')
    # A finite source list can run out. Reuse all valid candidates and extend
    # the published source pool rather than repeating positions to meet quotas.
    for base in [work,save]:
     (base/'failure.json').rename(base/'prior-source-exhaustion.json')
    from scripts.novel_elite_collect_cloud import main as collect_more
    collect_more()
   status();time.sleep(30)
  # Extra raw candidates replace unstable labels without padding the500k target.
  plan=json.loads(Path('/content/scale-plan.json').read_text());plan['candidate_counts']['train']=650000
  write(Path('/content/scale-plan.json'),plan)
  phase='collecting-reserve-candidates';status()
  from scripts.novel_elite_collect_cloud import main as collect
  if not (work/'candidate-ready.json').exists() or json.loads((work/'candidate-ready.json').read_text())['candidate_counts']['train']<650000:collect()
  ready=json.loads((work/'candidate-ready.json').read_text());assert ready['candidate_counts']['train']>=650000
  phase='teacher-screening-and-labelling';status()
  parent=Path('/content/targeted-distill/last.pt');s=torch.load(parent,map_location='cpu',weights_only=False,mmap=True);assert s['step']==3000
  model=PolicyValuePair(ModelConfig(**s['model_config']));model.load_state_dict(s['model']);model=model.cuda().eval();del s
  torch.set_num_threads(3);torch.set_float32_matmul_precision('high')
  # Preflight all candidate keys so a duplicate cannot cross the split boundary.
  owners={};quarantined=set()
  for path in sorted(work.glob('candidates-*.jsonl')):
   with path.open() as stream:
    for line in stream:
     j=json.loads(line);key=j['key']
     if key in owners and owners[key]!=j['split']:quarantined.add(key)
     else:owners[key]=j['split']
  write(work/'duplicate-key-audit.json',dict(candidate_keys=len(owners),cross_split_keys_quarantined=len(quarantined)))
  # Recover targets from the complete saved teacher metadata, not by re-running SF.
  stamp=str(int(time.time()))
  for base in [work,save]:
   for folder in ['teacher-data','mistakes']:
    if (base/folder).exists():(base/folder).rename(base/(folder+'-before-recovery-'+stamp))
  for path in work.glob('test-*.npy'):path.rename(path.with_suffix('.npy.before-recovery-'+stamp))
  dataset=ShardWriter(work/'teacher-data');mistakes=ShardWriter(work/'mistakes');label_counts=dict(train=0,val=0,test=0);tested=0;wrong=0;discarded=0;seen=set();test_buffer=[];test_shards=0;groups={s:set() for s in label_counts}
  meta_path=work/'teacher-metadata.jsonl';buffer=[];processed=set()
  if meta_path.exists():
   with meta_path.open() as stream:
    for line in stream:
     try:meta=json.loads(line)
     except ValueError:continue
     j=meta['job'];key=j['key']
     if key in processed or key in quarantined:continue
     processed.add(key);tested+=int(meta.get('prediction') is not None)
     if not meta.get('accepted'):discarded+=int(meta.get('stable') is False);continue
     b=chess.Board(j['initial_fen'])
     for u in j['history']:b.push_uci(u)
     row,legal=record(b);assert position_key(b).hex()==key;assign_targets(row,b,legal,meta['teacher'])
     seen.add(bytes(row['key']));split=j['split'];label_counts[split]+=1;groups[split].add(j['group'])
     if split=='test':test_buffer.append(row)
     else:dataset.add(split,row)
     if split=='train' and meta.get('repair'):mistakes.add('train',row);wrong+=1
  status(labelled=label_counts.copy(),screened=tested,verified_training_mistakes=wrong,unstable_discarded=discarded,recovered_from_metadata=True)
  def flush_test():
   nonlocal test_shards
   if test_buffer:
    np.save(work/f'test-{test_shards:04d}.npy',np.array(test_buffer,dtype=DTYPE));test_buffer.clear();test_shards+=1
  def persist_shards():
   for writer in [dataset,mistakes]:
    for split in ['train','val']:writer.flush(split)
   flush_test()
   for f in list((work/'teacher-data').glob('*.npy'))+list((work/'mistakes').glob('*.npy'))+list(work.glob('test-*.npy')):
    dst=save/f.relative_to(work);dst.parent.mkdir(exist_ok=True,parents=True)
    if not dst.exists() or dst.stat().st_size!=f.stat().st_size:
     shutil.copyfile(f,dst);assert sha(f)==sha(dst)
   # Metadata snapshots grow; copy only once per completed candidate shard.
   shutil.copyfile(meta_path,save/meta_path.name)
  with concurrent.futures.ProcessPoolExecutor(6,mp_context=multiprocessing.get_context('spawn'),initializer=teacher.teacher_init,initargs=(str(ROOT/'bin/stockfish'),)) as pool:
   for path in sorted(work.glob('candidates-*.jsonl')):
    jobs=[json.loads(l) for l in path.read_text().splitlines()]
    for start in range(0,len(jobs),32):
     chunk=[]
     for j in jobs[start:start+32]:
      limit=500000 if j['split']=='train' else 3000
      if j['key'] in processed or j['key'] in quarantined:continue
      if label_counts[j['split']]<limit:chunk.append(j);processed.add(j['key'])
     if not chunk:continue
     mine=[j['split']=='train' and (tested<plan['screening_minimum'] or wrong<plan['verified_training_mistakes_minimum']) for j in chunk]
     predictions=teacher.predict(model,[j for j,m in zip(chunk,mine) if m]) if any(mine) else [];it=iter(predictions)
     tasks=[(j,next(it) if m else None) for j,m in zip(chunk,mine)]
     results=list(pool.map(label_task,tasks))
     with meta_path.open('a') as log:
      for (meta,row),j,mined in zip(results,chunk,mine):
       tested+=int(mined)
       if meta.get('stable') is False:discarded+=1;log.write(json.dumps(dict(meta,accepted=False))+'\n');continue
       key=bytes(row['key'])
       if key in seen:continue
       seen.add(key);split=j['split'];limit=500000 if split=='train' else 3000
       if label_counts[split]>=limit:continue
       groups[split].add(j['group']);label_counts[split]+=1;log.write(json.dumps(dict(meta,accepted=True))+'\n')
       if split=='test':test_buffer.append(row)
       else:dataset.add(split,row)
       if split=='train' and meta.get('repair'):
        mistakes.add('train',row);wrong+=1
     status(labelled=label_counts.copy(),screened=tested,verified_training_mistakes=wrong,unstable_discarded=discarded,label_goal=500000,screen_goal=100000,mistake_goal=10000)
    persist_shards()
    if label_counts['train']==500000 and min(label_counts['val'],label_counts['test'])>=3000 and tested>=100000 and wrong>=10000:break
  assert not groups['train']&(groups['val']|groups['test']) and not groups['val']&groups['test']
  if label_counts['train']<500000 or min(label_counts['val'],label_counts['test'])<3000 or tested<100000 or wrong<10000:raise RuntimeError('Not enough independently accepted labels/mistakes; enlarge source pool, never duplicate to meet counts')
  provenance=dict(kind='new-elite-independent-SF18',teacher_nodes=100000,screened=tested,verified_mistakes=wrong,unstable_discarded=discarded,game_group_isolation=True,source_license='CC-BY-SA-4.0',attribution='Lichess broadcast contributors; https://database.lichess.org/#broadcasts',old_key_exclusion=ready['excluded_keys'])
  dataset.close(provenance);mistakes.close(provenance);persist_shards()
  files=[p for folder in ['teacher-data','mistakes'] for p in (work/folder).glob('*') if p.is_file()]+list(work.glob('test-*.npy'))
  hashes={str(p.relative_to(work)):sha(p) for p in files}
  for rel,want in hashes.items():
   dst=save/rel
   if not dst.exists():shutil.copyfile(work/rel,dst)
   assert sha(dst)==want
  proof=dict(training_positions=label_counts['train'],validation_positions=label_counts['val'],test_positions=label_counts['test'],verified_training_mistakes=wrong,screened=tested,sha256=hashes,source_candidate_audit=sha(work/'candidate-ready.json'),parent_checkpoint_sha256=sha(parent))
  for p in [work/'teacher-ready.json',save/'teacher-ready.json']:write(p,proof)
  del model;torch.cuda.empty_cache();phase='joint-training';status()
  from scripts.joint_distill_train import train
  train('/content/scale-plan.json')
  phase='joint-training-complete-instance-retained';status()
 except BaseException:
  phase='failed-instance-retained'
  if 'dataset' in locals():
   for writer in [dataset,mistakes]:
    for split in ['train','val']:writer.flush(split)
   flush_test();persist_shards()
  for p in [work/'pipeline-failure.json',save/'pipeline-failure.json']:write(p,dict(traceback=traceback.format_exc(),progress=progress))
  status();raise
 # No shutdown: user explicitly retains this existing VM.

if __name__=='__main__':main()
