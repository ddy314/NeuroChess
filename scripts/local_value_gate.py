"""Matched local fixed-position gate for a selected value head; no training or cloud calls."""
import argparse,hashlib,json,sys,importlib.util,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch
if importlib.util.find_spec('chess') is None:
 for site in (ROOT/'.venv/lib').glob('python*/site-packages'):
  if (site/'chess').is_dir():sys.path.append(str(site));break
import chess
from neurochess.policy_value_pair import from_original_states,checkpoint,policy_digest
from neurochess.multihead import MultiheadPlayer
from neurochess.native_mcts import NativeMCTSPlayer
from neurochess.encoding import encode,legal_ids
from scripts.search_repair_run_cloud import gate_decision
from scripts.local_value_repair import write


def main():
 p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--old',required=True);p.add_argument('--new',required=True);p.add_argument('--head',required=True,action='append');p.add_argument('--out',required=True)
 p.add_argument('--fixed-simulations',type=int,help='Additional budget-matched diagnostic; never substitutes for the locked two-second gate.')
 a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2);torch.set_float32_matmul_precision('high');torch.cuda.set_per_process_memory_fraction(.28)
 states=[torch.load(path,map_location='cpu',weights_only=False,mmap=True) for path in [a.new,a.old]];model=from_original_states(*states,device='cuda').eval();del states
 base_head={k:v.detach().cpu().clone() for k,v in model.value_model.value.state_dict().items()};policy_sha=policy_digest(model)
 selections={}
 for path in a.head:
  selected=torch.load(path,map_location='cpu',weights_only=False);name='candidate' if len(a.head)==1 else 'step-'+str(selected['step'])
  if name in selections:raise ValueError('Duplicate checkpoint step')
  selections[name]=(selected,path)
 data=Path(a.data);roots=json.loads((data/'gate-roots.json').read_text());search_config=json.loads((ROOT/'configs/search-baseline-lock.json').read_text())['config'];net=MultiheadPlayer(model,'cuda');reports={k:{} for k in ['initial',*selections]}
 if a.fixed_simulations is not None:
  if a.fixed_simulations<=0:raise ValueError('Positive simulation budget required')
  search_config.update(simulations=a.fixed_simulations,time_limit=None)
 def set_head(mode):model.value_model.value.load_state_dict(base_head if mode=='initial' else selections[mode][0]['value_head'])
 def val(records):
  ces=[];pred=[]
  with torch.inference_mode():
   for start in range(0,len(records),32):
    r=records[start:start+32];target=torch.as_tensor(r['value'].copy(),device='cuda')
    with torch.autocast('cuda',dtype=torch.bfloat16):logits=model.value_only(torch.as_tensor(r['tokens'].copy(),device='cuda'),torch.as_tensor(r['meta'].copy(),device='cuda')).float()
    ces.extend((-(target*logits.log_softmax(-1)).sum(-1)).cpu().tolist());prob=logits.softmax(-1);pred.extend((prob[:,0]-prob[:,2]).cpu().tolist())
  truth=records['value'][:,0]-records['value'][:,2];err=abs(np.asarray(pred)-truth);by={}
  for k,mask in [('winning',truth>.25),('neutral',abs(truth)<=.25),('losing',truth<-.25)]:
   if mask.any():by[k]=dict(n=int(mask.sum()),mae=float(err[mask].mean()))
  return dict(n=len(records),ce=float(np.mean(ces)),mae=float(err.mean()),by_stratum=by)
 for mode in reports:
  set_head(mode)
  for key,folder in [('balanced','val'),('general','general-val')]:
   records=np.concatenate([np.load(p) for p in sorted((data/folder).glob('val-*.npy'))]);reports[mode][key]=val(records)
 # Warmup outside timed decisions.
 b=chess.Board(roots['diagnostics'][0]['initial_fen'])
 for u in roots['diagnostics'][0]['history']:b.push_uci(u)
 t,m=encode(b);_,ids=legal_ids(b);moves=np.full((64,256),-1,np.int16);moves[:,:len(ids)]=ids
 for size in [1,64]:net.raw_outputs(np.repeat(t[None],size,axis=0),np.repeat(m[None],size,axis=0),moves[:size])
 torch.cuda.synchronize();opponents={k:[] for k in reports}
 for group in ['diagnostics','holdout']:
  for mode in reports:reports[mode][group]=dict(correct=0,total=0,rows=[])
  for index,j in enumerate(roots[group]):
   names=list(reports);offset=index%len(names);order=names[offset:]+names[:offset]
   for mode in order:
    set_head(mode);b=chess.Board(j['initial_fen'])
    for u in j['history']:b.push_uci(u)
    player=NativeMCTSPlayer(net,**search_config)
    try:move=player.choose(b);stats=player.search_stats;elapsed=player.last_result['elapsed_seconds']
    finally:player.close()
    b.push(move);ok=move.uci()==j['solution'][0] or b.is_checkmate();report=reports[mode][group];report['correct']+=int(ok);report['total']+=1
    report['rows'].append(dict(key=j['key'],correct=ok,move=move.uci(),expected=j['solution'][0],elapsed=elapsed,stats=stats))
    if group=='diagnostics':
     b=chess.Board(j['initial_fen'])
     for u in j['history']+[j['solution'][0]]:b.push_uci(u)
     t,m=encode(b)
     with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):v=model.value_only(torch.as_tensor(t[None],device='cuda'),torch.as_tensor(m[None],device='cuda')).float().softmax(-1)
     opponents[mode].append(float((v[0,0]-v[0,2]).cpu()))
   write(out/'progress.json',dict(group=group,completed=index+1,total=len(roots[group]),reports=reports));print(group,index+1,flush=True)
 for mode in reports:reports[mode]['opponent_mean_value']=float(np.mean(opponents[mode]))
 decisions={name:gate_decision(reports['initial'],reports[name]) for name in selections}
 decision=next(iter(decisions.values())) if len(decisions)==1 else dict(passed=all(d['passed'] for d in decisions.values()),checkpoints=decisions)
 assert policy_digest(model)==policy_sha
 mode='Counterbalanced matched2s PUCT per root' if a.fixed_simulations is None else f'Additional fixed-{a.fixed_simulations} simulation probe; not a replacement for the two-second gate'
 write(out/'result.json',dict(reports=reports,decision=decision,decisions_by_checkpoint=decisions,policy_q_digest=policy_sha,search_config=search_config,mode=mode,selected_before_chess_gate=True,checkpoint_steps={name:s[0]['step'] for name,s in selections.items()},complete_games=0,device=torch.cuda.get_device_name()))
 if decision['passed'] and a.fixed_simulations is None:
  for name,(selected,path) in selections.items():
   set_head(name);lineage=dict(policy_q_parent_sha256='343d19699d86f3e0b8e2ff1a9cf3237a403fb25c4e0af98ad7d63a72d6c78247',value_parent_sha256='e4ed8978db91eeef8360c4dea87faf7a4f826325bce45aa7efc024692d585ef7',local_value_head_sha256=hashlib.file_digest(open(path,'rb'),'sha256').hexdigest())
   filename='passed-pair.pt' if len(selections)==1 else 'passed-pair-'+name+'.pt'
   torch.save(checkpoint(model,selected['step'],lineage,gate_passed=True,window_gate_passed=True),out/filename)
 print(json.dumps(decision),flush=True)

if __name__=='__main__':main()
