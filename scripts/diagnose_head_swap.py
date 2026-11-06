"""Local, bounded output-head interventions; never merges checkpoint weights.

Every intervention evaluates both original models, then swaps their OUTPUTS.
Thus each head includes its original trunk, and shared-trunk changes are included.
Not a playing-strength or search-speed benchmark.
"""
import argparse,hashlib,itertools,json,sys,time,importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
if importlib.util.find_spec('chess') is None:
    for site in (ROOT/'.venv/lib').glob('python*/site-packages'):
        if (site/'chess').is_dir():sys.path.append(str(site));break
import chess
from neurochess.train import load_model
from neurochess.encoding import encode,legal_ids,position_key
from neurochess.multihead import MultiheadPlayer,fuse_heads
from neurochess.native_mcts import NativeMCTSPlayer


def mix_outputs(old,new,sources):
    return np.concatenate([([old,new][sources[0]])[:,:256],([old,new][sources[1]])[:,256:512],([old,new][sources[2]])[:,512:]],axis=1)


class SwappedNetwork:
    device='cuda'
    def __init__(self,networks,sources):self.networks=networks;self.sources=sources
    def evaluate_search_heads(self,tokens,meta,moves):
        # Both forward passes in EVERY cell, including all-old/all-new controls.
        old,new=[n.raw_outputs(tokens,meta,moves) for n in self.networks]
        mixed=mix_outputs(old,new,self.sources)
        p,q,v,_=fuse_heads(mixed[:,:256],mixed[:,256:512],mixed[:,512:],moves>=0)
        return p,q,v


def board_for(job):
    b=chess.Board(job['initial_fen'])
    for u in job['history']:b.push_uci(u)
    assert position_key(b).hex()==job['key']
    return b


def main():
    p=argparse.ArgumentParser();p.add_argument('--old',required=True);p.add_argument('--new',required=True);p.add_argument('--roots',required=True);p.add_argument('--out',required=True)
    p.add_argument('--fixed-simulations',type=int,default=0,help='Additional same-simulation diagnostic instead of wall time')
    a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2);torch.set_float32_matmul_precision('high');torch.manual_seed(20261101)
    lock=json.loads((ROOT/'configs/search-baseline-lock.json').read_text());config=lock['config'].copy()
    if a.fixed_simulations:config.update(time_limit=None,simulations=a.fixed_simulations)
    roots=json.loads(Path(a.roots).read_text());networks=[];shas=[]
    for path in [a.old,a.new]:
        with open(path,'rb') as f:shas.append(hashlib.file_digest(f,'sha256').hexdigest())
        model,state=load_model(path,'cuda');del state;networks.append(MultiheadPlayer(model,'cuda'))
    assert shas==['e4ed8978db91eeef8360c4dea87faf7a4f826325bce45aa7efc024692d585ef7','343d19699d86f3e0b8e2ff1a9cf3237a403fb25c4e0af98ad7d63a72d6c78247']
    plan=dict(checkpoints=shas,config=config,device=torch.cuda.get_device_name(),root_sha256=hashlib.sha256(Path(a.roots).read_bytes()).hexdigest(),
        method='Full2x2x2 output intervention; both complete models run in every cell. Original trunks preserved. Reset tree and cache for each root/cell.',
        limitations=['Retrospectively selected diagnostic roots, not unbiased accuracy or Elo.','Timed intervention performs two forwards; no claim of latency or baseline playing-quality equivalence.'])
    (out/'plan.json').write_text(json.dumps(plan,indent=2));rows=[];calibration=[]
    # Warm up both networks at the same batch shapes before timed decisions.
    b=board_for(roots[0]);t,m=encode(b);_,ids=legal_ids(b);ms=np.full((64,256),-1,np.int16);ms[:,:len(ids)]=ids
    for n in networks:
        for size in [1,64]:n.raw_outputs(np.repeat(t[None],size,0),np.repeat(m[None],size,0),ms[:size])
    torch.cuda.synchronize()
    # Root + exact source continuation: observe value across alternating turns.
    for job in roots:
        b=board_for(job)
        for ply in range(min(3,len(job['solution'])+1)):
            t,m=encode(b);legal,ids=legal_ids(b)
            if not legal:break
            ms=np.full((1,256),-1,np.int16);ms[0,:len(ids)]=ids
            raw=[n.raw_outputs(t[None],m[None],ms) for n in networks]
            calibration.append(dict(key=job['key'],ply=ply,turn=b.turn,old_wdl=raw[0][0,512:].tolist(),new_wdl=raw[1][0,512:].tolist(),fen=b.fen()))
            if ply<len(job['solution']):b.push_uci(job['solution'][ply])
    (out/'value-probes.json').write_text(json.dumps(calibration,indent=2))
    combinations=list(itertools.product([0,1],repeat=3))
    for i,job in enumerate(roots):
        # Deterministic rotation prevents one combination always running first.
        order=combinations[i%8:]+combinations[:i%8]
        for sources in order:
            player=NativeMCTSPlayer(SwappedNetwork(networks,sources),**config);b=board_for(job)
            try:
                move=player.choose(b);b.push(move);ok=move.uci()==job['solution'][0] or b.is_checkmate()
                row=dict(key=job['key'],selection=job['selection'],sources=''.join(map(str,sources)),move=move.uci(),expected=job['solution'][0],correct=ok,decision=player.last_result,stats=player.search_stats)
                rows.append(row)
            finally:player.close()
        summary={''.join(map(str,c)):{kind:dict(correct=sum(x['correct'] for x in rows if x['sources']==''.join(map(str,c)) and x['selection']==kind),total=sum(x['sources']==''.join(map(str,c)) and x['selection']==kind for x in rows)) for kind in ['regression','improved','retained']} for c in combinations}
        record=dict(completed_roots=i+1,total_roots=len(roots),summary=summary,rows=rows)
        temp=out/'progress.tmp';temp.write_text(json.dumps(record,indent=2));temp.replace(out/'progress.json')
        print(json.dumps(dict(root=i+1,total=len(roots),summary=summary)),flush=True)
    effects={}
    for head,index in [('policy',0),('Q',1),('value',2)]:
        pairs=[]
        for row in rows:
            if row['sources'][index]!='0':continue
            counterpart=row['sources'][:index]+'1'+row['sources'][index+1:]
            other=next(x for x in rows if x['key']==row['key'] and x['sources']==counterpart)
            pairs.append(float(other['correct'])-float(row['correct']))
        effects[head]=dict(mean_correct_delta_new_minus_old=float(np.mean(pairs)),paired_interventions=len(pairs))
    record['conditional_head_effects']=effects
    (out/'result.json').write_text(json.dumps(record,indent=2)+'\n')

if __name__=='__main__':main()
