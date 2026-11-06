"""Collect genuinely new, game-grouped elite positions on the retained VM.

Candidate preparation only: existing PGN evaluations are provenance, never
silently substituted for independently adjudicated training labels.
"""
import collections,hashlib,io,json,os,shutil,sys,time,traceback
from pathlib import Path
from urllib.request import urlopen
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import chess,chess.pgn,numpy as np,zstandard
from neurochess.encoding import position_key
from neurochess.prepare import FilteredBuilder,rejection
from scripts.local_value_repair import write
from scripts.value_sustained_cloud import sha


def game_group(board,moves):
 return hashlib.sha256((board.fen()+'|'+' '.join(m.uci() for m in moves)).encode()).hexdigest()


def group_split(group):
 n=int(group[:12],16)%100
 return 'test' if n==0 else 'val' if n==1 else 'train'


def main():
 work=Path('/content/targeted-scale');work.mkdir(exist_ok=True)
 save=Path('/content/drive/MyDrive/neurochess/targeted-scale-20261102');save.mkdir(parents=True,exist_ok=True)
 assert Path('/content/KEEP_INSTANCE.json').exists(),'User-requested retention marker required'
 plan=json.loads(Path('/content/scale-plan.json').read_text());write(work/'plan.json',plan);shutil.copyfile(work/'plan.json',save/'plan.json')
 counts=collections.Counter();reasons=collections.Counter();sources=[];blocked=set();started=time.time();phase='indexing-prior-data'
 def status(**extra):
  s=dict(phase=phase,counts=dict(counts),goals=plan['candidate_counts'],blocked_keys=len(blocked),rejections=dict(reasons),sources=sources,updated_at=time.time(),elapsed_seconds=time.time()-started,**extra)
  for p in [work/'status.json',save/'status.json']:write(p,s)
  print(json.dumps({k:v for k,v in s.items() if k!='sources'}),flush=True)
 try:
  if (work/'excluded-keys.npy').exists() and (save/'excluded-keys.npy').exists():
   assert sha(work/'excluded-keys.npy')==sha(save/'excluded-keys.npy')
   blocked.update(bytes(k) for k in np.load(work/'excluded-keys.npy',mmap_mode='r'))
  else:
   for folder in [Path('/content/targeted-core'),Path('/content/drive/MyDrive/neurochess-data/elite-recent-2026q3')]:
    for i,p in enumerate(sorted(folder.glob('*.npy'))):
     if '/drive/' in str(p):
      cache=work/'prior-key-shard.npy';shutil.copyfile(p,cache);a=np.load(cache,mmap_mode='r')
     else:a=np.load(p,mmap_mode='r')
     blocked.update(bytes(k) for k in a['key']);del a
     if i%16==0:status(index_source=str(folder),index_shard=i)
   for rel in ['fresh-v4-preparation/jobs.jsonl','search-repair-v1/preparation/search-jobs.jsonl']:
    path=Path('/content/drive/MyDrive/neurochess')/rel
    if path.exists():
     with path.open() as f:
      for line in f:blocked.add(bytes.fromhex(json.loads(line)['key']))
     status(index_source=rel)
   pilot=Path('/content/targeted-distill/screened.npy')
   if pilot.exists():blocked.update(bytes(k) for k in np.load(pilot,mmap_mode='r')['key'])
   frozen=json.loads(Path('/content/targeted-inputs/frozen.json').read_text());blocked.update(bytes.fromhex(k) for k in frozen['keys'])
   for p in Path('/content/drive/MyDrive/neurochess/curriculum-v3').glob('*exclusions*.fen'):
    for line in p.read_text().splitlines():
     if line.strip():blocked.add(position_key(chess.Board(line.strip())))
   array=np.frombuffer(b''.join(sorted(blocked)),dtype=np.uint8).reshape(-1,16);np.save(work/'excluded-keys.npy',array);del array
   shutil.copyfile(work/'excluded-keys.npy',save/'excluded-keys.npy');assert sha(work/'excluded-keys.npy')==sha(save/'excluded-keys.npy')
  phase='collecting-new-elite';seen=set();groups=set();buffer=[];shard=0
  for path in sorted(work.glob('candidates-*.jsonl')):
   with path.open() as f:
    for line in f:
     j=json.loads(line);seen.add(bytes.fromhex(j['key']));groups.add(j['group']);counts[j['split']]+=1
   shard+=1
  for marker in [work/'candidate-ready.json',save/'candidate-ready.json']:
   marker.unlink(missing_ok=True)
  def flush():
   nonlocal shard
   if not buffer:return
   p=work/f'candidates-{shard:05d}.jsonl';p.write_text(''.join(json.dumps(j)+'\n' for j in buffer));shutil.copyfile(p,save/p.name)
   assert sha(p)==sha(save/p.name);buffer.clear();shard+=1
  for source in plan['sources']:
   status(source=source['name']);archive=work/(source['name']+'.pgn.zst')
   if not archive.exists() or sha(archive)!=source['sha256']:
    with urlopen(source['url'],timeout=90) as response,archive.open('wb') as dest:shutil.copyfileobj(response,dest)
   expected=source['sha256'];assert sha(archive)==expected,'Published source checksum differs'
   shutil.copyfile(archive,save/archive.name);assert sha(archive)==sha(save/archive.name)
   summary=dict(source,download_bytes=archive.stat().st_size,accepted_games=0,positions=0);sources.append(summary)
   with archive.open('rb') as raw,io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(raw)) as stream:
    while (g:=chess.pgn.read_game(stream,Visitor=lambda:FilteredBuilder(source))) is not None:
     why=rejection(g,source)
     if why:reasons[why]+=1;continue
     # Some freestyle broadcasts use Variant=Standard despite a Chess960 FEN.
     # Require the actual normal starting position for this full-game corpus.
     if g.board().chess960 or g.board().fen()!=chess.STARTING_FEN:
      reasons['nonstandard_initial_position']+=1;continue
     moves=list(g.mainline_moves());group=game_group(g.board(),moves);split=group_split(group)
     if group in groups:reasons['duplicate_game']+=1;continue
     if counts[split]>=plan['candidate_counts'][split]:continue
     groups.add(group);b=g.board();positions=[]
     for ply,node in enumerate(g.mainline(),1):
      b.push(node.move)
      if ply<14 or ply>160 or ply%3 or b.is_game_over(claim_draw=True):continue
      key=position_key(b)
      if key in blocked or key in seen:reasons['old_or_duplicate_key']+=1;continue
      score=node.eval();source_eval=None
      if score is not None:
       w=score.white();source_eval={'mate':w.mate()} if w.is_mate() else {'cp':w.score()}
      positions.append(dict(key=key.hex(),group=group,split=split,initial_fen=g.board().fen(),history=[m.uci() for m in b.move_stack],
        source=source['name'],source_url=source['url'],license=source['license'],source_game_headers=dict(g.headers),source_eval_white=source_eval,
        source_eval_role='Provenance only; engine/version/budget may be unknown',ply=ply))
     # Deterministically spread sampling across the whole game, including endings.
     if len(positions)>40:positions=[positions[i] for i in np.linspace(0,len(positions)-1,40,dtype=int)]
     for j in positions:
      if bytes.fromhex(j['key']) in seen:reasons['repeated_position_in_game']+=1;continue
      if counts[split]>=plan['candidate_counts'][split]:break
      seen.add(bytes.fromhex(j['key']));buffer.append(j);counts[split]+=1;summary['positions']+=1
      if len(buffer)>=2048:flush();status(source=source['name'])
     if positions:summary['accepted_games']+=1
     if all(counts[s]>=goal for s,goal in plan['candidate_counts'].items()):break
   flush();status(source=source['name'])
   if all(counts[s]>=goal for s,goal in plan['candidate_counts'].items()):break
  if not all(counts[s]>=n for s,n in plan['candidate_counts'].items()):raise RuntimeError('Sources exhausted before unique-position target; never repeat to pad counts')
  proof=dict(candidate_counts=dict(counts),source_games=len(groups),old_key_overlap=0,split='SHA of complete legal game trajectory; one family one split',
    excluded_keys=len(blocked),historical_source_game_isolation='Historical old-data source games unavailable; exact keys excluded, no claim of full old-game isolation',
    files={p.name:sha(p) for p in work.glob('candidates-*.jsonl')},teacher_labels_ready=False,novelty='Candidate states are new against indexed core/human/fresh-root/repair-state/pilot/frozen keys; not a count of newly labelled examples')
  for p in [work/'candidate-ready.json',save/'candidate-ready.json']:write(p,proof)
  phase='candidates-ready-awaiting-teacher';status()
 except BaseException:
  phase='collection-failed';trace=traceback.format_exc()
  for p in [work/'failure.json',save/'failure.json']:write(p,dict(traceback=trace))
  status();raise
 # Explicit user instruction: leave this VM allocated for follow-up experiments.

if __name__=='__main__':main()
