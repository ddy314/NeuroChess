"""Controlled continuations of one committed epoch checkpoint; no promotion."""
from collections import defaultdict
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from .data import FIELDS
from .guarded_epoch import ExactEpoch, balance_value_weights, learning_rate, replay_count


VARIANTS = {
    'A': dict(lr_multiplier=1., balance_strength=1.),
    'B': dict(lr_multiplier=1.5, balance_strength=1.),
    'C': dict(lr_multiplier=1., balance_strength=.5),
}


def variant_lr(step, total, peak, warmup, variant, offset):
    if variant not in VARIANTS or offset < 1:
        raise ValueError('Unknown variant or invalid continuation offset')
    multiplier = 1+(VARIANTS[variant]['lr_multiplier']-1)*min(offset/50, 1.)
    return learning_rate(step, total, peak, warmup)*multiplier


def value_weights(rows, strength):
    """Blend original confidence weights with full balancing, without relabeling."""
    if not 0 <= strength <= 1:
        raise ValueError('Balance strength outside [0, 1]')
    original = rows['value_weight'].copy()
    balance_value_weights(rows)
    rows['value_weight'] = (1-strength)*original+strength*rows['value_weight']
    return rows


def objective_terms(out, batch, nt, nq, nv, old_mask, teacher, nold):
    """The exact weighted terms used by guarded_epoch.joint_objective."""
    p,q,v=out
    target=batch['policy'].float()
    target=target/target.sum(-1,keepdim=True).clamp_min(1e-8)
    policy=-(target*F.log_softmax(p.float(),-1)).sum()/nt
    qloss=.5*((q.float()-batch['q'].float()).square()*batch['qmask']).sum()/max(1,nq)
    value=(-(batch['value'].float()*F.log_softmax(v.float(),-1)).sum(-1)*batch['value_weight']).sum()/max(1,nv)
    anchor=policy*0
    if teacher is not None and old_mask.any() and nold:
        tp,tq,tv=teacher
        pk=F.kl_div(F.log_softmax(p[old_mask].float(),-1),tp.float().softmax(-1),reduction='sum')/nold
        vk=F.kl_div(F.log_softmax(v[old_mask].float(),-1),tv.float().softmax(-1),reduction='sum')/nold
        legal=batch['moves'][old_mask]>=0
        qk=(((q[old_mask]-tq.float()).square()*legal).sum(-1)/legal.sum(-1).clamp_min(1)).sum()/nold
        anchor=.5*pk+.25*qk+4.*vk
    return dict(policy=policy,Q=qloss,value=value,retention=anchor)


def mixed_rows(stream, replay, plan, absolute_step, strength):
    nnew=plan['new_per_update'];nold=replay_count(nnew)
    rows=np.concatenate([stream.next(nnew),replay.sample(nold)])
    old=np.r_[np.zeros(nnew,bool),np.ones(nold,bool)]
    order=np.random.default_rng(np.random.SeedSequence([plan['seed'],absolute_step])).permutation(len(rows))
    rows=rows[order];old=old[order]
    digest=hashlib.sha256(rows['key'].tobytes()+old.tobytes()).hexdigest()
    return value_weights(rows,strength),old,nnew,nold,digest


def restore_random_states(checkpoint,replay):
    replay.rng.bit_generator.state=checkpoint['replay_rng']
    torch.set_rng_state(checkpoint['torch_rng'])
    if torch.cuda.is_available():torch.cuda.set_rng_state_all(checkpoint['cuda_rng'])


def batch(rows,device='cuda'):
    return {k:torch.as_tensor(np.array(rows[k],copy=True),device=device) for k in FIELDS}


def validate(model,rows,microbatch=64,device='cuda'):
    from .train import losses
    model.eval();tot=np.zeros(7);truth=[];pred=[]
    with torch.inference_mode():
        for i in range(0,len(rows),microbatch):
            r=rows[i:i+microbatch];b=batch(r,device)
            with torch.autocast(device,dtype=torch.bfloat16,enabled=device=='cuda'):
                out=model(b['tokens'],b['meta'],b['moves'])
            loss,parts=losses(out,b,.5,1.);n=len(r);qm=b['qmask'];v=out[2].softmax(-1)
            tot+=np.array([float(loss)*n,float(parts['policy'])*n,float(parts['q'])*n,float(parts['value'])*n,
                int((out[0].argmax(-1)==b['policy'].argmax(-1)).sum()),float(((out[1]-b['q'].float()).abs()*qm).sum()),int(qm.sum())])
            pred.extend((v[:,0]-v[:,2]).cpu().tolist());truth.extend((b['value'][:,0]-b['value'][:,2]).cpu().tolist())
    t=np.array(truth);err=abs(np.array(pred)-t)
    return dict(positions=len(rows),loss=tot[0]/len(rows),policy_loss=tot[1]/len(rows),q_loss=tot[2]/len(rows),
        value_loss=tot[3]/len(rows),top1=tot[4]/len(rows),q_mae=tot[5]/max(1,tot[6]),value_mae=float(err.mean()),
        value_strata={k:dict(n=int(m.sum()),mae=float(err[m].mean()) if m.any() else None)
            for k,m in dict(winning=t>.25,neutral=abs(t)<=.25,losing=t<-.25).items()})


def validate_all(model,validation,step):
    return {k:dict(step=step,split=k,**validate(model,rows)) for k,rows in validation.items()}


def search_screen(model,frozen,stockfish,lock,step,progress=None):
    """Identical parent comparison and thresholds to train_lichess_epoch."""
    import chess,chess.engine
    from .multihead import MultiheadPlayer
    from .native_mcts import NativeMCTSPlayer
    net=MultiheadPlayer(model,'cuda');model.eval();rows=[]
    with chess.engine.SimpleEngine.popen_uci(stockfish) as sf:
        sf.configure({'Threads':1,'Hash':64,'Skill Level':20,'UCI_LimitStrength':False})
        for i,root in enumerate(frozen):
            board=chess.Board(root['fen']);player=NativeMCTSPlayer(net,**lock['config']);tick=time.time()
            try:move=player.choose(board);stats=player.last_result
            finally:player.close()
            baseline_move=root['baseline_move']
            proposal=sf.analyse(board,chess.engine.Limit(nodes=100000),game=object())['pv'][0]
            moves=list(dict.fromkeys([proposal,board.parse_uci(baseline_move),move]))
            info=sf.analyse(board,chess.engine.Limit(nodes=1000000),root_moves=moves,multipv=len(moves),game=object())
            scores={x['pv'][0].uci():x['score'].pov(board.turn).score(mate_score=100000) for x in info};best=max(scores.values())
            rows.append(dict(index=i,fen=root['fen'],move=move.uci(),baseline_move=baseline_move,
                candidate_regret=best-scores[move.uci()],baseline_regret=best-scores[baseline_move],scores=scores,search=stats,seconds=time.time()-tick))
            if progress:progress(i+1,len(frozen))
    return dict(step=step,**screen_search_rows(rows),rows=rows,
        scope='Same 48 development roots, frozen original-parent choices, L4 and locked 2s PUCT; finite 1M-node candidate-union teachers; not an Elo test')


def screen_search_rows(rows):
    bad=[r['index'] for r in rows if r['candidate_regret']>=100 and r['candidate_regret']-r['baseline_regret']>=75]
    delta=float(np.mean([r['candidate_regret']-r['baseline_regret'] for r in rows]))
    return dict(passed=not bad and delta<=10.,introduced_blunders=bad,mean_regret_delta_cp=delta)


def gradient_geometry(vectors):
    """Full shared-trunk gradients, weighted as in the actual objective."""
    names=list(vectors)
    def dot(a,b):return float(torch.stack([(x*y).sum() for x,y in zip(a,b)]).sum())
    squares={k:max(0.,dot(v,v)) for k,v in vectors.items()}
    norms={k:math.sqrt(x) for k,x in squares.items()}
    dots={a+'|'+b:dot(vectors[a],vectors[b]) for i,a in enumerate(names) for b in names[i+1:]}
    cosine={k:v/max(norms[k.split('|')[0]]*norms[k.split('|')[1]],1e-30) for k,v in dots.items()}
    total_sq=max(0.,sum(squares.values())+2*sum(dots.values()))
    return dict(norms=norms,cosines=cosine,cancellation_ratio=math.sqrt(total_sq)/max(sum(norms.values()),1e-30))


def diagnose(model,teacher,checkpoint,files,replay,plan,progress,batches=8):
    restore_random_states(checkpoint,replay)
    stream=ExactEpoch(files,plan['seed'],checkpoint['new_positions'])
    named=[(n,p) for n,p in model.named_parameters() if n.startswith(('piece.','rank.','file.','meta.','blocks.','norm.'))]
    params=[p for _,p in named];records=[];model.train()
    for index in range(batches):
        rows,old,_,nold,input_hash=mixed_rows(stream,replay,plan,checkpoint['epoch_updates']+index,1.)
        nt=len(rows);nq=int(rows['qmask'].sum());nv=float(rows['value_weight'].sum())
        vectors={k:[torch.zeros_like(p) for p in params] for k in ['policy','Q','value','retention']}
        losses=defaultdict(float)
        for i in range(0,nt,plan['microbatch']):
            r=rows[i:i+plan['microbatch']];b=batch(r);mask=torch.as_tensor(old[i:i+len(r)],device='cuda')
            with torch.autocast('cuda',dtype=torch.bfloat16):
                with torch.no_grad():ref=teacher(b['tokens'][mask],b['meta'][mask],b['moves'][mask]) if mask.any() else None
                out=model(b['tokens'],b['meta'],b['moves']);terms=objective_terms(out,b,nt,nq,nv,mask,ref,nold)
            for j,(name,loss) in enumerate(terms.items()):
                grad=torch.autograd.grad(loss,params,retain_graph=j<3,allow_unused=True)
                for total,g in zip(vectors[name],grad):
                    if g is not None:total.add_(g)
                losses[name]+=float(loss.detach())
                del grad
            del out,terms,b,ref
        record=dict(batch=index,input_sha256=input_hash,losses=dict(losses),**gradient_geometry(vectors))
        records.append(record);del vectors;progress(index+1,batches)
    model.zero_grad(set_to_none=True)
    summary={}
    for kind in ['norms','cosines']:
        summary[kind]={k:dict(mean=float(np.mean([r[kind][k] for r in records])),min=min(r[kind][k] for r in records),max=max(r[kind][k] for r in records)) for k in records[0][kind]}
    summary['cancellation_ratio']=dict(mean=float(np.mean([r['cancellation_ratio'] for r in records])))
    return dict(batches=batches,positions_per_batch=1024,shared_parameter_count=sum(p.numel() for p in params),
        component_coefficients=dict(policy=1.,Q=.5,value=1.,retention_policy_KL=.5,retention_Q_MSE=.25,retention_value_KL=4.),
        records=records,summary=summary,optimizer_steps=0,scope='Raw weighted shared-trunk gradient geometry; descriptive, not proof of the cause or Adam update direction')


def paired_comparison(a,b,seed=4000,repeats=2000):
    """Paired bootstrap over development positions; descriptive after tuning."""
    a=np.asarray(a,dtype=np.float64);b=np.asarray(b,dtype=np.float64)
    if a.shape!=b.shape or a.ndim!=1:raise ValueError('Aligned one-dimensional pairs required')
    diff=b-a;rng=np.random.default_rng(seed)
    means=np.array([diff[rng.integers(len(diff),size=len(diff))].mean() for _ in range(repeats)])
    return dict(mean_delta=float(diff.mean()),bootstrap_95=np.quantile(means,[.025,.975]).tolist(),n=len(diff),
        scope='Position bootstrap, not game-clustered; fixed development data selected during training; not independent confirmation')


def position_metrics(model,rows,microbatch=64):
    model.eval();values=defaultdict(list)
    with torch.inference_mode():
        for i in range(0,len(rows),microbatch):
            b=batch(rows[i:i+microbatch])
            with torch.autocast('cuda',dtype=torch.bfloat16):out=model(b['tokens'],b['meta'],b['moves'])
            target=b['policy'].float();target=target/target.sum(-1,keepdim=True).clamp_min(1e-8)
            wdl=out[2].softmax(-1)
            measurements=dict(top1=(out[0].argmax(-1)==target.argmax(-1)).float(),
                policy_ce=-(target*F.log_softmax(out[0],-1)).sum(-1),
                q_abs_sum=((out[1]-b['q'].float()).abs()*b['qmask']).sum(-1),q_count=b['qmask'].sum(-1),
                value_error=((wdl[:,0]-wdl[:,2])-(b['value'][:,0]-b['value'][:,2])).abs())
            for key,value in measurements.items():values[key].append(value.cpu().numpy())
    return dict(keys=rows['key'].copy(),**{k:np.concatenate(v) for k,v in values.items()})


def paired_q_comparison(a_sum,b_sum,counts,seed=4000,repeats=2000):
    a=np.asarray(a_sum,dtype=np.float64);b=np.asarray(b_sum,dtype=np.float64);counts=np.asarray(counts)
    if a.shape!=b.shape or a.shape!=counts.shape or counts.sum()<=0:raise ValueError('Aligned annotated-Q counts required')
    diff=b-a;rng=np.random.default_rng(seed);samples=[]
    for _ in range(repeats):
        idx=rng.integers(len(diff),size=len(diff));denom=counts[idx].sum()
        if denom:samples.append(diff[idx].sum()/denom)
    return dict(mean_delta=float(diff.sum()/counts.sum()),bootstrap_95=np.quantile(samples,[.025,.975]).tolist(),
        n=len(diff),q_labels=int(counts.sum()),scope='Position-paired bootstrap of label-weighted Q MAE; development data, not independent confirmation')
