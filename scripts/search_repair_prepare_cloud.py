"""CPU preparation for one value repair branch; independently label real child states."""
import os,sys,json,hashlib,sqlite3,shutil,tarfile,threading,time,fcntl,traceback
from pathlib import Path
ROOT=Path('/content/neurochess');WORK=Path('/content/search-repair-prep');MOUNT=Path('/content/drive')
SAVE=MOUNT/'MyDrive/neurochess/search-repair-v1/preparation'

def sha(p):
 with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def copy(a,b):
 b=Path(b);b.parent.mkdir(parents=True,exist_ok=True);t=b.with_name(b.name+'.tmp');shutil.copyfile(a,t);assert sha(a)==sha(t);t.replace(b)

def main():
 if not os.path.ismount(MOUNT):raise RuntimeError('Mount Drive via CLI')
 WORK.mkdir(exist_ok=True);SAVE.mkdir(parents=True,exist_ok=True);lock=open('/tmp/search-repair-prep.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 os.chdir(ROOT);sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
 import chess,numpy as np
 from fresh_hard_data import atomic_json
 from search_state_curriculum import expand_jobs,annotate_jobs,value_bucket
 from neurochess.data import DTYPE,ShardWriter
 from neurochess.encoding import position_key
 state={'phase':'copying-inputs'};stop=threading.Event();sync_lock=threading.Lock()
 def sync():
  with sync_lock:
   atomic_json(WORK/'status.json',dict(state,updated_at=time.time()))
   for p in WORK.glob('*.json'):copy(p,SAVE/p.name)
   db=WORK/'teacher/labels.sqlite'
   if db.exists():
    with sqlite3.connect(db,timeout=60) as source,sqlite3.connect(WORK/'snapshot.sqlite') as dest:source.backup(dest)
    copy(WORK/'snapshot.sqlite',SAVE/'labels.sqlite')
   for n in ['progress.json','spec.json']:
    p=WORK/'teacher'/n
    if p.exists():copy(p,SAVE/('teacher-'+n))
 def backup():
  while not stop.wait(120):
   try:sync()
   except Exception as e:print('snapshot_retry',repr(e),flush=True)
 thread=threading.Thread(target=backup,daemon=True);thread.start()
 try:
  preparation=MOUNT/'MyDrive/neurochess/fresh-v4-preparation'
  for name in ['jobs.jsonl','quality-gate.json']:
   if not (WORK/name).exists():copy(preparation/name,WORK/name)
  if not (WORK/'seed.sqlite').exists():copy(preparation/'labels.sqlite',WORK/'seed.sqlite')
  core=WORK/'core'
  if not (core/'manifest.json').exists():
   if not (WORK/'core.tar.gz').exists():copy(MOUNT/'MyDrive/neurochess-data/evals-v2-5m.tar.gz',WORK/'core.tar.gz')
   assert sha(WORK/'core.tar.gz')=='a0b5dbef770fcf2b7b4bc31a709f4a2b8379421965aa591f69291a6abbc7c2ef'
   core.mkdir(exist_ok=True)
   with tarfile.open(WORK/'core.tar.gz') as tar:tar.extractall(core,filter='data')
  original=[json.loads(l) for l in (WORK/'jobs.jsonl').read_text().splitlines()]
  assert hashlib.sha256((WORK/'jobs.jsonl').read_bytes()).hexdigest()=='747abe500eb217d51a6428f6f850388a89391564fa1a6eef0d5943812b31a504'
  limits={'train':2000,'val':512};seeds=[]
  for split,count in limits.items():seeds.extend(sorted((j for j in original if j['split']==split),key=lambda j:hashlib.sha256(('repair-v1:'+j['key']).encode()).digest())[:count])
  test=[j for j in original if j['split']=='test'];frozen_keys={j['key'] for j in test};frozen_groups={j['group'] for j in test}
  for f in (ROOT/'fresh-v4-assets/eval-openings.fen').read_text().splitlines():
   if f.strip():frozen_keys.add(position_key(chess.Board(f)).hex())
  seed=sqlite3.connect(f'file:{WORK}/seed.sqlite?mode=ro',uri=True);candidates={}
  for j in seeds:
   metadata=json.loads(seed.execute('SELECT metadata FROM labels WHERE key=?',(j['key'],)).fetchone()[0]);candidates[j['key']]=[p['line'].split()[0] for p in metadata['pvs'] if p['line'].split()[0]!=j['solution'][0]][:2]
  jobs,audit=expand_jobs(seeds,candidates,frozen_keys,frozen_groups,descendant_limits=limits)
  audit.update(source_seed_counts=limits,frozen_test_games_excluded=len(frozen_groups),additional_novel_root_claim=False)
  atomic_json(WORK/'expansion-audit.json',audit)
  (WORK/'search-jobs.jsonl').write_text(''.join(json.dumps(j,sort_keys=True)+'\n' for j in jobs));copy(WORK/'search-jobs.jsonl',SAVE/'search-jobs.jsonl')
  state['phase']='annotating-real-child-states';sync()
  spec=annotate_jobs(jobs,str(ROOT/'bin/stockfish'),WORK/'teacher',WORK/'seed.sqlite',WORK/'quality-gate.json',workers=min(8,len(os.sched_getaffinity(0))))
  # Frozen source test is never relabelled or trained; generic replay is identified separately.
  state['phase']='balancing-and-auditing';sync();db=sqlite3.connect(WORK/'teacher/labels.sqlite');bykey={j['key']:j for j in jobs}
  pools={s:{k:[] for k in ['winning','neutral','losing']} for s in ['train','val']};used=set()
  for key,split,payload in db.execute('SELECT key,split,payload FROM labels'):
   row=np.frombuffer(payload,dtype=DTYPE)[0].copy();bucket=value_bucket(row)
   if bucket:pools[split][bucket].append((row,bykey[key]['role']))
   used.add(key)
  valrows=np.concatenate([np.load(p) for p in sorted(core.glob('val-*.npy'))]);old_val_keys={bytes(k).hex() for k in valrows['key']}
  protected=frozen_keys|{j['key'] for j in jobs if j['split']=='val'}|old_val_keys
  for split,files in [('train',sorted(core.glob('train-*.npy'))),('val',sorted(core.glob('val-*.npy')))]:
   for p in files:
    a=np.load(p,mmap_mode='r');indices=np.linspace(0,len(a)-1,min(256,len(a)),dtype=int)
    for row in a[indices]:
     key=bytes(row['key']).hex()
     if key in used or key in frozen_keys or (split=='train' and key in protected):continue
     bucket=value_bucket(row)
     if bucket:pools[split][bucket].append((row.copy(),'generic-replay'));used.add(key)
  output=WORK/'data';output.mkdir(exist_ok=True);proof={};train_keys=set()
  for split,quota in [('train',10000),('val',100)]:
   for bucket,items in pools[split].items():
    if split=='train':items[:]=[(r,role) for r,role in items if bytes(r['key']).hex() not in protected]
    if len(items)<quota:raise ValueError(f'Insufficient real {split}/{bucket}: {len(items)}<{quota}')
   if (output/split).exists():shutil.rmtree(output/split)
   writer=ShardWriter(output/split,size=8192);roles={};keys=set()
   for i in range(quota):
    for bucket in ['winning','neutral','losing']:
     row,role=pools[split][bucket][i];key=bytes(row['key']).hex();assert key not in keys;keys.add(key)
     row['qmask']=False # This stage repairs value only; policy/Q branch is frozen.
     writer.add('train' if split=='train' else 'val',row);roles[role]=roles.get(role,0)+1
   writer.close(dict(kind='balanced-value-repair',value_stratum_counts={k:quota for k in pools[split]},roles=roles,teacher_spec=spec,generic_replay_counted_as_novel=False))
   if split=='train':train_keys=keys
   else:assert not keys&train_keys
   proof[split]=dict(count=len(keys),per_value_stratum=quota,roles=roles)
  # Independent general validation, excluding balanced validation and training keys.
  balanced_val={bytes(k).hex() for p in (output/'val').glob('*.npy') for k in np.load(p,mmap_mode='r')['key']}
  writer=ShardWriter(output/'general-val',size=8192) if not (output/'general-val').exists() else None
  if writer:
   count=0
   for row in valrows:
    if bytes(row['key']).hex() in balanced_val|train_keys:continue
    writer.add('val',row);count+=1
    if count==4096:break
   assert count==4096;writer.close(dict(kind='untouched-general-validation',positions=count))
  # Fixed diagnostics plus32 deterministic untouched source test games for bounded gate.
  diagnostics=json.loads((ROOT/'repair-assets/diagnostic-roots.json').read_text());known={j['key'] for j in diagnostics}
  holdout=sorted((j for j in test if j['key'] not in known),key=lambda j:hashlib.sha256(('gate:'+j['key']).encode()).digest())[:32]
  atomic_json(output/'gate-roots.json',dict(diagnostics=diagnostics,holdout=holdout))
  data_audit=dict(complete=True,counts=proof,train_val_overlap=0,frozen_test_keys_excluded=True,new_source_family_split_isolated=True,
   generic_replay_limit='Old core source games are unknown; exact heldout keys excluded. No claim of historical source-game disjointness.',teacher_spec=spec)
  atomic_json(output/'data-audit.json',data_audit)
  archive=WORK/'data.tar.gz'
  with tarfile.open(archive,'w:gz',compresslevel=1) as tar:tar.add(output,arcname='data')
  copy(archive,SAVE/'data.tar.gz');ready=dict(complete=True,archive_sha256=sha(archive),audit=data_audit,plan='750-step value-head repair then gate;4000-step conditional single continuation')
  atomic_json(WORK/'ready.json',ready);copy(WORK/'ready.json',SAVE/'ready.json');state['phase']='complete';sync()
 except BaseException:
  state['phase']='failed';atomic_json(WORK/'failure.json',dict(traceback=traceback.format_exc(),time=time.time()));raise
 finally:
  stop.set();thread.join();sync()
  from neurochess.autostop import request_unassign,find_runtime_address
  request_unassign(find_runtime_address())

if __name__=='__main__':main()
