"""Paired held-out root search and full output-head interventions."""
import argparse,importlib.util,itertools,json,sys,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
if importlib.util.find_spec('chess') is None:sys.path.extend(str(p) for p in (ROOT/'.venv/lib').glob('python*/site-packages'))
import chess,numpy as np,torch
from scripts.analyze_joint_checkpoint import load,write,sha
from neurochess.multihead import MultiheadPlayer,fuse_heads
from neurochess.native_mcts import NativeMCTSPlayer
from neurochess.encoding import encode,legal_ids,position_key


def board_for(j):
    b=chess.Board(j['initial_fen'])
    for u in j['history']:b.push_uci(u)
    assert position_key(b).hex()==j['key'];return b


class Swap:
    device='cuda'
    def __init__(self,nets,sources):self.nets=nets;self.sources=sources
    def evaluate_search_heads(self,tokens,meta,moves):
        raw=[n.raw_outputs(tokens,meta,moves) for n in self.nets]
        p=raw[self.sources[0]][:,:256];q=raw[self.sources[1]][:,256:512];v=raw[self.sources[2]][:,512:]
        fused,prior,value,_=fuse_heads(p,q,v,moves>=0);return fused,prior,value


def main(out,old,new):
    torch.set_num_threads(2);torch.set_float32_matmul_precision('high');models=[load(p)[0] for p in [old,new]];nets=[MultiheadPlayer(m,'cuda') for m in models]
    lock=json.loads((ROOT/'configs/search-baseline-lock.json').read_text());config=lock['config'];jobs=json.loads((out/'search-jobs.json').read_text())
    # No source/config tuning: use the frozen maintained search implementation.
    source_hashes={}
    for name in ['neurochess/native_mcts.py','neurochess/native/mcts.cpp','neurochess/native/libmcts.so','neurochess/multihead.py']:
        source_hashes[name]=sha(ROOT/name);assert source_hashes[name]==lock['source_sha256'][name],name
    write(out/'search-protocol.json',dict(config=config,source_sha256=source_hashes,hardware=torch.cuda.get_device_name(),primary_roots=len(jobs),selection='Blinded32 per teacher-value stratum, unique game per root; balanced diagnostic, not natural population rate',
         interventions='24 roots, full2x2x2 output swaps including branch trunks, same512 simulations, both networks in EVERY cell; diagnostic only, not speed/floor qualification',
         primary_repeats='All roots once; preselected24 roots repeated twice more. Statistical unit is source game, never repeat.'))
    b=board_for(jobs[0]);t,m=encode(b);_,ids=legal_ids(b);ms=np.full((64,256),-1,np.int16);ms[:,:len(ids)]=ids
    for n in nets:
        for size in [1,64]:n.raw_outputs(np.repeat(t[None],size,0),np.repeat(m[None],size,0),ms[:size])
    selected=[j for label in ['winning','neutral','losing'] for j in [x for x in jobs if x['selection']==label][:8]]
    journal=out/'search-decisions.jsonl';existing=[]
    if journal.exists():existing=[json.loads(l) for l in journal.read_text().splitlines()]
    completed={(r['key'],r['mode'],r.get('repeat',0),r['sources']) for r in existing}
    def decide(j,net,cfg,mode,sources,repeat=0):
        key=(j['key'],mode,repeat,sources)
        if key in completed:return
        b=board_for(j);player=NativeMCTSPlayer(net,**cfg);start=time.time()
        try:move=player.choose(b);stats=player.search_stats;root=player.last_result
        finally:player.close()
        row=dict(key=j['key'],group=j['group'],selection=j['selection'],mode=mode,sources=sources,repeat=repeat,move=move.uci(),seconds=time.time()-start,stats=stats,root=root)
        with journal.open('a') as f:f.write(json.dumps(row)+'\n')
        completed.add(key)
    # Independent timed controls use one real network each, matching deployment.
    for repeat in range(3):
        for i,j in enumerate(jobs if repeat==0 else selected):
            for which in ([0,1] if (i+repeat)%2==0 else [1,0]):decide(j,nets[which],config,'locked2s','000' if which==0 else '111',repeat)
            write(out/'search-status.json',dict(phase='locked2s',repeat=repeat,completed=i+1,total=len(jobs if repeat==0 else selected),updated_at=time.time()))
    cfg=dict(config,time_limit=None,simulations=512);combos=list(itertools.product([0,1],repeat=3))
    for i,j in enumerate(selected):
        order=combos[i%8:]+combos[:i%8]
        for sources in order:decide(j,Swap(nets,sources),cfg,'head-swap-fixed512',''.join(map(str,sources)))
        write(out/'search-status.json',dict(phase='head-swaps',completed=i+1,total=len(selected),updated_at=time.time()))
    # Construct new exact-history future states, then label each state independently.
    metadata=json.loads((out/'metadata.json').read_text());children=[];seen=set()
    for j in selected:
        teacher=metadata[j['key']]['teacher'];b=board_for(j)
        from neurochess.targets import score_targets
        pv=max(teacher,key=lambda p:score_targets(p,b.turn)[0])['line'].split()
        for branch in ['teacher','parent','final']:
            b=board_for(j)
            for depth in range(1,5):
                if b.is_game_over(claim_draw=True):break
                if branch=='teacher':
                    if depth>len(pv):break
                    move=b.parse_uci(pv[depth-1])
                else:move=nets[0 if branch=='parent' else 1].choose(b)
                b.push(move)
                if depth not in [1,2,4] or b.is_game_over(claim_draw=True):continue
                key=position_key(b).hex();history=[m.uci() for m in b.move_stack]
                identity=hashlib.sha256((j['initial_fen']+'|'+' '.join(history)).encode()).hexdigest()
                if identity in seen:continue
                seen.add(identity);children.append(dict(j,key=key,case_id=identity,history=history,root_key=j['key'],branch=branch,depth=depth))
    write(out/'continuation-jobs.json',children)
    write(out/'search-status.json',dict(phase='complete',primary_roots=len(jobs),intervention_roots=len(selected),continuation_positions=len(children),updated_at=time.time()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--old',required=True);p.add_argument('--new',required=True);a=p.parse_args();main(Path(a.out),Path(a.old),Path(a.new))
