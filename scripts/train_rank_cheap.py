"""Controlled local-GPU distillation: WDL, auxiliary CP, and sibling ranking.

Only game-grouped new data is used, from scratch. All variants retain the same
WDL inference/export head, keeping existing large-model corrections compatible.
The CP head is auxiliary at training time; ranking also supervises WDL logits.
"""
import argparse
from dataclasses import asdict
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
if importlib.util.find_spec('chess') is None:
    sys.path.append(str(next((ROOT/'.venv/lib').glob('python*/site-packages'))))
import numpy as np
import torch
from torch.nn import functional as F
from neurochess.cheap import CheapConfig,CheapModel,save_cheap
from neurochess.frontier import FRONTIER_DTYPE,ranking_loss,ranking_metrics


def tensor_batch(rows,device):
    return {k:torch.from_numpy(np.array(rows[k],copy=True)).to(device) for k in
            ['tokens','meta','valid','cp','cp_valid','wdl']}


def loss_heads(model,b):
    shape=b['valid'].shape
    v,c=model.heads(b['tokens'].flatten(0,1),b['meta'].flatten(0,1))
    return v.reshape(shape),c.reshape(shape) if c is not None else None


@torch.inference_mode()
def validate(model,rows,batch=512):
    model.eval();device=next(model.parameters()).device
    sums=dict(states=0,wdl_squared=0.,cp_states=0,cp_abs=0.,cp_squared=0.,rank_loss=0.,
              pairs=0,pairs_correct=0,groups=0,cp_regret_sum=0.,blunders_100=0)
    for start in range(0,len(rows),batch):
        r=rows[start:start+batch];b=tensor_batch(r,device);v,c=loss_heads(model,b)
        valid=b['valid'];cp_valid=valid & b['cp_valid']
        sums['states']+=int(valid.sum());sums['wdl_squared']+=float(((v.tanh()-b['wdl'])[valid]**2).sum())
        if c is not None:
            error=(600*c-b['cp'])[cp_valid];sums['cp_states']+=len(error)
            sums['cp_abs']+=float(error.abs().sum());sums['cp_squared']+=float((error**2).sum())
        metrics=ranking_metrics(v.cpu().numpy(),r['cp'],r['valid'] & r['cp_valid'])
        sums['rank_loss']+=float(ranking_loss(v,b['cp'],cp_valid))*metrics['pairs']
        for key,value in metrics.items():sums[key]+=value
    return dict(states=sums['states'],wdl_mse=sums['wdl_squared']/sums['states'],
        cp_states=sums['cp_states'],cp_mae=sums['cp_abs']/sums['cp_states'] if sums['cp_states'] else None,
        cp_rmse=math.sqrt(sums['cp_squared']/sums['cp_states']) if sums['cp_states'] else None,
        rank_pairs=sums['pairs'],rank_accuracy=sums['pairs_correct']/max(1,sums['pairs']),
        rank_bce=sums['rank_loss']/max(1,sums['pairs']),candidate_groups=sums['groups'],
        mean_sibling_cp_regret=sums['cp_regret_sum']/max(1,sums['groups']),
        sibling_blunders_100=sums['blunders_100'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',required=True);p.add_argument('--out',required=True)
    p.add_argument('--steps',type=int,default=20000);p.add_argument('--batch-groups',type=int,default=256)
    p.add_argument('--width',type=int,default=128);p.add_argument('--hidden',type=int,default=32)
    p.add_argument('--cp-weight',type=float,default=.5);p.add_argument('--rank-weight',type=float,default=.2)
    p.add_argument('--lr',type=float,default=.0005);p.add_argument('--seed',type=int,default=431903)
    p.add_argument('--validate-every',type=int,default=1000);p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    a=p.parse_args()
    if min(a.steps,a.batch_groups,a.width,a.hidden,a.lr,a.validate_every)<=0 or min(a.cp_weight,a.rank_weight)<0:p.error('Invalid limits')
    if a.device=='cuda' and not torch.cuda.is_available():p.error('CUDA unavailable')
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False);data=Path(a.data)
    train=np.load(data/'train.npy',mmap_mode='r');val=np.load(data/'val.npy',mmap_mode='r')
    if train.dtype!=FRONTIER_DTYPE or val.dtype!=FRONTIER_DTYPE:raise ValueError('Wrong frontier schema')
    if not len(train) or not len(val):raise ValueError('Empty split')
    torch.set_num_threads(2);torch.manual_seed(a.seed);rng=np.random.default_rng(a.seed)
    model=CheapModel(CheapConfig(a.width,a.hidden,cp_head=True)).to(a.device)
    opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=.0001,fused=a.device=='cuda')
    provenance=dict(args=vars(a),config=asdict(model.config),torch=str(torch.__version__),
        gpu=torch.cuda.get_device_name() if a.device=='cuda' else None,parameters=sum(x.numel() for x in model.parameters()),
        manifest_sha256=hashlib.sha256((data/'manifest.json').read_bytes()).hexdigest(),
        dataset=json.loads((data/'manifest.json').read_text()),selection='validation WDL MSE + 0.2 * sibling rank BCE (same criterion for all variants)',
        cp_target='raw cp preserved; auxiliary Huber on clipped +/-3000 cp / 600; mate rows excluded only from CP/rank',
        inference='original WDL head only; native export unchanged; auxiliary CP head not needed at inference')
    (out/'run.json').write_text(json.dumps(provenance,indent=2))
    (out/'initial.json').write_text(json.dumps(validate(model,val),indent=2))
    best=float('inf');step=seen=epoch=0;start=time.perf_counter()
    with (out/'metrics.jsonl').open('w') as log:
        while step<a.steps:
            for at in range(0,len(train),a.batch_groups):
                if at==0:order=rng.permutation(len(train));epoch+=1
                b=tensor_batch(train[order[at:at+a.batch_groups]],a.device)
                lr=a.lr*(.1+.9*.5*(1+math.cos(math.pi*step/a.steps)))
                for group in opt.param_groups:group['lr']=lr
                model.train();opt.zero_grad(set_to_none=True);v,c=loss_heads(model,b);mask=b['valid']
                wdl=F.mse_loss(v[mask].tanh(),b['wdl'][mask])+.02*F.smooth_l1_loss(v[mask],torch.atanh(b['wdl'][mask].clamp(-.995,.995)))
                cp_mask=mask & b['cp_valid'];cp=F.smooth_l1_loss(c[cp_mask],b['cp'][cp_mask].clamp(-3000,3000)/600) if cp_mask.any() else c.sum()*0
                rank=ranking_loss(v,b['cp'],cp_mask)
                loss=wdl+a.cp_weight*cp+a.rank_weight*rank
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);opt.step()
                step+=1;seen+=int(mask.sum())
                if step%a.validate_every==0 or step==a.steps:
                    validation=validate(model,val);selection=validation['wdl_mse']+.2*validation['rank_bce']
                    row=dict(step=step,epoch=epoch,seen=seen,seconds=time.perf_counter()-start,lr=lr,
                        train=dict(loss=float(loss.detach()),wdl=float(wdl.detach()),cp=float(cp.detach()),rank=float(rank.detach())),
                        validation=validation,selection_score=selection,phase='complete' if step==a.steps else 'training')
                    save_cheap(model,out/'last.pt',step=step,validation=validation,provenance=provenance);model.export(out/'last.bin')
                    if selection<best:
                        best=selection;save_cheap(model,out/'best.pt',step=step,validation=validation,provenance=provenance);model.export(out/'best.bin')
                    (out/'status.json').write_text(json.dumps(row,indent=2));log.write(json.dumps(row)+'\n');log.flush();print(json.dumps(row),flush=True)
                if step>=a.steps:break


if __name__=='__main__':main()
