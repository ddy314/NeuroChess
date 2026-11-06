"""Bounded search-node repair helpers, preserving original P/Q/WDL semantics."""
import math
import numpy as np
import torch
from torch.nn import functional as F
import chess,chess.engine
from .data import record
from .targets import assign_targets

def fused_tensors(outputs,moves):
    p,q,logv=outputs;wdl=logv.float().softmax(-1);p=p.float();q=q.float();legal=moves>=0
    p=p.masked_fill(~legal,-1e9);prob=p.softmax(-1)
    order=torch.argsort(p,dim=-1,descending=True,stable=True)
    support=torch.zeros_like(legal).scatter(1,order[:,:5],True)&legal&(p>=p.max(-1,keepdim=True).values-3)
    weights=prob*support;weights=weights/weights.sum(-1,keepdim=True).clamp_min(1e-20)
    confidence=(1+(wdl*wdl.clamp_min(1e-8).log()).sum(-1)/math.log(3)).clamp(0,1)
    logits=p+2*(.25+.75*confidence[:,None])*(q-(q*weights).sum(-1,keepdim=True))*support
    value=wdl[:,0]-wdl[:,2];seed=.5*q+.5*value[:,None]
    return logits,seed,value,support

def functional_anchor(outputs,reference,moves,mask,total):
    if not mask.any() or total==0:return outputs[0].sum()*0
    current=fused_tensors(outputs,moves);target=fused_tensors(reference,moves)
    kl=F.kl_div(current[0][mask].log_softmax(-1),target[0][mask].softmax(-1),reduction='sum')/total
    support=target[3][mask];seed=((current[1][mask]-target[1][mask]).square()*support).sum(-1)/support.sum(-1).clamp_min(1)
    value=(current[2][mask]-target[2][mask]).square().sum()/total
    return .25*kl+.125*seed.sum()/total+value

def ranking_pairs(cp):
    valid=np.isfinite(cp);pair=valid[:,:,None]&valid[:,None,:]&(cp[:,:,None]-cp[:,None,:]>=100)
    return tuple(np.asarray(x,dtype=np.int64) for x in np.nonzero(pair))

def pair_loss(policy,pairs,start,end,total):
    if total==0:return policy.sum()*0
    row,good,bad=pairs;mask=(row>=start)&(row<end)
    if not mask.any():return policy.sum()*0
    i=torch.as_tensor(row[mask]-start,device=policy.device);g=torch.as_tensor(good[mask],device=policy.device);b=torch.as_tensor(bad[mask],device=policy.device)
    return .05*F.softplus(-(policy[i,g].float()-policy[i,b].float())).sum()/total

def complete_analysis(engine,board,nodes,moves=None):
    """Highest complete exact single-depth iteration, including true unrestricted PV1."""
    m=1 if moves is None else len(moves)
    engine.send(f'setoption name MultiPV value {m}\nucinewgame\nisready');list(engine.until('readyok'))
    engine.send('position fen '+board.fen(en_passant='fen'))
    engine.send(f'go nodes {nodes}'+('' if moves is None else ' searchmoves '+' '.join(moves)))
    levels={};complete={}
    for line in engine.until('bestmove '):
        if not line.startswith('info ') or ' score ' not in line or ' pv ' not in line:continue
        head,pv=line.split(' pv ',1);tokens=head.split()
        if 'depth' not in tokens:continue
        d=int(tokens[tokens.index('depth')+1]);k=int(tokens[tokens.index('multipv')+1]) if 'multipv' in tokens else 1
        i=tokens.index('score');kind=tokens[i+1];v=int(tokens[i+2])
        if kind not in ['cp','mate']:raise ValueError('Invalid teacher score kind')
        row=dict(move=pv.split()[0],cp=v if kind=='cp' else None,mate=v if kind=='mate' else None,
            sort_score=v if kind=='cp' else chess.engine.Mate(v).score(mate_score=100000),depth=d,
            nodes=int(tokens[tokens.index('nodes')+1]),lowerbound='lowerbound' in tokens,upperbound='upperbound' in tokens,pv=pv.split())
        levels.setdefault(d,{})[k]=row;r=levels[d]
        if set(r)==set(range(1,m+1)) and not any(x['lowerbound'] or x['upperbound'] for x in r.values()):
            if moves is None or {x['move'] for x in r.values()}==set(moves):complete[d]=[dict(r[j]) for j in range(1,m+1)]
    if not complete:raise ValueError('No complete exact teacher iteration')
    rows=complete[max(complete)]
    for r in rows:
        b=board.copy(stack=False)
        for move in r['pv']:b.push_uci(move)
    return rows

def white_pvs(rows,turn):
    result=[]
    for r in sorted(rows,key=lambda x:-x['sort_score']):
        p=dict(line=' '.join(r['pv']));kind='cp' if r['cp'] is not None else 'mate'
        p[kind]=int(r[kind])*(1 if turn else -1);result.append(p)
    return result

def wl(row):
    s=chess.engine.Cp(row['cp']) if row['cp'] is not None else chess.engine.Mate(row['mate'])
    w=s.wdl(model='sf16',ply=32);return (w.wins-w.losses)/1000

def qualify_label(first,second,global_first,global_second):
    if len(first)<2 or {r['move'] for r in first}!={r['move'] for r in second}:return False,'candidate-set'
    rows=first+second+global_first+global_second
    if any(r['depth']<24 or r['nodes']<1000000 or r['lowerbound'] or r['upperbound'] for r in rows):return False,'depth-nodes-bounds'
    a={r['move']:r for r in first};b={r['move']:r for r in second}
    for m in a:
        if (a[m]['cp'] is None)!=(b[m]['cp'] is None):return False,'mate-kind-instability'
        if a[m]['cp'] is not None and abs(a[m]['cp']-b[m]['cp'])>100:return False,'cp-instability'
        if a[m]['mate'] is not None and np.sign(a[m]['mate'])!=np.sign(b[m]['mate']):return False,'mate-sign-instability'
        if abs(wl(a[m])-wl(b[m]))>.05+1e-7:return False,'WDL-instability'
    for f,g in [(first,global_first),(second,global_second)]:
        best=max(f,key=lambda x:x['sort_score']);ref=g[0]
        if ref['move'] not in a:return False,'global-move-missing'
        if abs(wl(best)-wl(ref))>.05+1e-7:return False,'global-WDL-disagreement'
        if (best['cp'] is None)!=(ref['cp'] is None):return False,'global-mate-kind'
        if best['cp'] is not None and abs(best['cp']-ref['cp'])>100:return False,'global-cp-disagreement'
    if abs(wl(global_first[0])-wl(global_second[0]))>.05+1e-7:return False,'global-instability'
    # Large CP order disagreements are not silently turned into policy labels.
    for m in a:
        for n in a:
            if a[m]['cp'] is not None and a[n]['cp'] is not None and b[m]['cp'] is not None and b[n]['cp'] is not None:
                if abs(a[m]['cp']-a[n]['cp'])>=100 and (a[m]['cp']-a[n]['cp'])*(b[m]['cp']-b[n]['cp'])<=0:return False,'large-gap-order'
    return True,'qualified'

def labeled_row(fen,rows):
    board=chess.Board(fen);row,legal=record(board);row['meta'][[5,6,7,10]]=0
    assign_targets(row,board,legal,white_pvs(rows,board.turn));cp=np.full(256,np.nan,np.float32)
    for r in rows:
        if r['cp'] is not None:cp[legal.index(board.parse_uci(r['move']))]=r['cp']
    return row,cp
