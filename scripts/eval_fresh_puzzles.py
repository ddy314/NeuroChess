"""Frozen puzzle first-move accuracy; exact mate-in-one alternatives accepted.

Test accessed only after both training arms and validation checkpoint selection.
Paired confidence intervals resample complete source-game groups.
"""
import argparse,hashlib,json,sys
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import chess
import numpy as np
import torch
from neurochess.train import load_model
from neurochess.multihead import MultiheadPlayer
from neurochess.native_mcts import NativeMCTSPlayer


def board_for(job):
    b=chess.Board(job['initial_fen'])
    for move in job['history']:b.push_uci(move)
    return b


def correct(job,move):
    b=board_for(job)
    if move not in b.legal_moves:raise ValueError('Illegal puzzle prediction')
    if move.uci()==job['solution'][0]:return True
    b.push(move)
    return b.is_checkmate()


def paired_groups(before,after,jobs,seed=20261101):
    groups=defaultdict(list)
    for a,b,j in zip(before,after,jobs):groups[j['group']].append(float(b)-float(a))
    keys=sorted(groups);sums=np.array([sum(groups[k]) for k in keys]);counts=np.array([len(groups[k]) for k in keys])
    rng=np.random.default_rng(seed);samples=[]
    for _ in range(10000):
        idx=rng.integers(len(keys),size=len(keys));samples.append(float(sums[idx].sum()/counts[idx].sum()))
    return dict(delta=float((np.asarray(after,dtype=float)-np.asarray(before,dtype=float)).mean()),
        source_game_bootstrap95=np.quantile(samples,[.025,.975]).tolist(),independent_groups=len(keys))


def main():
    p=argparse.ArgumentParser();p.add_argument('--jobs',required=True);p.add_argument('--checkpoint',action='append',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();torch.set_num_threads(4);torch.set_float32_matmul_precision('high')
    jobs=[json.loads(l) for l in Path(a.jobs).read_text().splitlines() if json.loads(l)['split']=='test']
    if len(jobs)!=2000:raise ValueError('Expected 2000 frozen test positions')
    # Frozen deterministic 256 roots; no model outputs used to pick them.
    search_jobs=sorted(jobs,key=lambda j:hashlib.sha256(('search-v4:'+j['key']).encode()).digest())[:256]
    report=dict(protocol='First-move success against source only-move solutions; any immediate checkmate accepted. No teacher used by student at inference.',
        jobs_sha256=hashlib.file_digest(open(a.jobs,'rb'),'sha256').hexdigest(),search_keys=[j['key'] for j in search_jobs],
        search_simulations=512,results=[],limitations=['First-move accuracy is not whole-puzzle solution accuracy or Elo.'])
    outcomes={}
    for checkpoint in a.checkpoint:
        model,state=load_model(checkpoint,'cuda');del state;network=MultiheadPlayer(model,'cuda');entry={'checkpoint':checkpoint,
            'sha256':hashlib.file_digest(open(checkpoint,'rb'),'sha256').hexdigest()};outcomes[checkpoint]={}
        for mode,subset in [('direct',jobs),('search',search_jobs)]:
            player=network if mode=='direct' else NativeMCTSPlayer(network,simulations=512,leaf_batch_size=64)
            predictions=[];flags=[];chunk=64 if mode=='direct' else 16
            for start in range(0,len(subset),chunk):
                group=subset[start:start+chunk];moves=player.choose_many([board_for(j) for j in group])
                predictions.extend(m.uci() for m in moves);flags.extend(correct(j,m) for j,m in zip(group,moves))
            if mode=='search':player.close()
            outcomes[checkpoint][mode]=flags
            entry[mode]=dict(correct=sum(flags),positions=len(flags),accuracy=float(np.mean(flags)),by_category={c:dict(correct=sum(ok for j,ok in zip(subset,flags) if j['category']==c),positions=sum(j['category']==c for j in subset)) for c in ['tactical','endgame','opening']},
                predictions=[dict(key=j['key'],group=j['group'],predicted=m,expected=j['solution'][0],correct=ok) for j,m,ok in zip(subset,predictions,flags)])
        report['results'].append(entry);del network,model;torch.cuda.empty_cache()
    report['paired_comparisons']={}
    for before,after in [(a.checkpoint[0],a.checkpoint[-1]),(a.checkpoint[-2],a.checkpoint[-1])]:
        report['paired_comparisons'][before+' -> '+after]={mode:paired_groups(outcomes[before][mode],outcomes[after][mode],subset) for mode,subset in [('direct',jobs),('search',search_jobs)]}
    Path(a.out).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({r['checkpoint']:{m:{k:v for k,v in r[m].items() if k!='predictions'} for m in ['direct','search']} for r in report['results']},indent=2))

if __name__=='__main__':main()
