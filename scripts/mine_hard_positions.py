"""Rank a deterministic sample using current policy/Q/WDL errors."""
import importlib.util
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
if importlib.util.find_spec('chess') is None:
    for site in (ROOT/'.venv/lib').glob('python*/site-packages'):
        if (site/'chess').is_dir():sys.path.append(str(site));break
import argparse,hashlib,json,time
import chess
import numpy as np
import torch
from neurochess.encoding import position_key,legal_ids

from neurochess.train import load_model


def decode(row):
    b=chess.Board(None);b.turn=bool(row['meta'][9]);flip=0 if b.turn else 56
    for square,token in enumerate(row['tokens']):
        if token:b.set_piece_at(square^flip,chess.Piece((int(token)-1)%6+1,b.turn if token<=6 else not b.turn))
    for flag,square in zip(row['meta'][:4],[chess.H1 if b.turn else chess.H8,chess.A1 if b.turn else chess.A8,
                                                chess.H8 if b.turn else chess.H1,chess.A8 if b.turn else chess.A1]):
        if flag:b.castling_rights|=chess.BB_SQUARES[square]
    b.ep_square=(round(float(row['meta'][4])*63)^flip) if row['meta'][11] else None
    b.halfmove_clock=0;b.fullmove_number=1
    if not b.is_valid() or position_key(b)!=row['key'].tobytes():raise ValueError('Decoded position differs from original key')
    return b


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--checkpoint',required=True)
    p.add_argument('--exclude',required=True);p.add_argument('--out',required=True)
    p.add_argument('--sample',type=int,default=600000);p.add_argument('--select',type=int,default=200000)
    a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(3);torch.set_float32_matmul_precision('high')
    model,state=load_model(a.checkpoint,'cuda');model.eval();del state
    excluded={position_key(chess.Board(f)) for f in Path(a.exclude).read_text().splitlines() if f.strip()};rng=np.random.default_rng(20261031);started=time.time()
    train=sorted(Path(a.data).glob('train-*.npy'));all_files=train+sorted(Path(a.data).glob('val-*.npy'))[1:]
    candidates=[]
    with torch.inference_mode():
        for file_index,file in enumerate(all_files):
            rows=np.load(file,mmap_mode='r',allow_pickle=False)
            n=min(len(rows),int(np.ceil(a.sample/len(train))) if file_index<len(train) else 700)
            indices=rng.choice(len(rows),n,replace=False)
            for start in range(0,n,128):
                idx=indices[start:start+128];batch=rows[idx]
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    policy,q,wdl=model(torch.as_tensor(np.array(batch['tokens']),device='cuda'),
                        torch.as_tensor(np.array(batch['meta']),device='cuda'),torch.as_tensor(np.array(batch['moves']),device='cuda'))
                prob=policy.log_softmax(-1);target=torch.as_tensor(np.array(batch['policy']),device='cuda').float()
                ce=-(target*prob).sum(-1)
                mask=torch.as_tensor(np.array(batch['qmask']),device='cuda')
                qe=((q-torch.as_tensor(np.array(batch['q']),device='cuda')).abs()*mask).sum(-1)/mask.sum(-1).clamp_min(1)
                ve=-(torch.as_tensor(np.array(batch['value']),device='cuda')*wdl.log_softmax(-1)).sum(-1)
                mismatch=(policy.argmax(-1)!=target.argmax(-1)).float()
                scores=(ce+2*qe+ve+mismatch).cpu().numpy();top=policy.topk(5,dim=-1).indices.cpu().numpy()
                for j,i in enumerate(idx):
                    if batch[j]['key'].tobytes() in excluded:continue
                    candidates.append((float(scores[j]),file_index,int(i),int((batch[j]['tokens']!=0).sum()),bool(batch[j]['meta'][8]),
                                       batch[j]['moves'][top[j]].tolist(),file_index>=len(train)))
            progress=dict(files=file_index+1,total_files=len(all_files),candidates=len(candidates),seconds=time.time()-started)
            (out/'mining-progress.json').write_text(json.dumps(progress));print(json.dumps(progress),flush=True)
    training=[r for r in candidates if not r[-1]];selected=[];chosen=set()
    # Balanced selection across sparse endgames, checks, and remaining positions.
    for predicate,quota in [(lambda r:r[3]<=12,a.select//5),(lambda r:r[4],a.select//5),(lambda r:True,a.select)]:
        count=sum(bool(predicate(x)) for x in selected)
        for r in sorted((x for x in training if predicate(x)),reverse=True):
            key=(r[1],r[2])
            if key in chosen:continue
            chosen.add(key);selected.append(r);count+=1
            if count>=quota:break
        if len(selected)>=a.select:break
    selected=selected[:a.select]
    selected+=sorted((r for r in candidates if r[-1]),reverse=True)[:2000]
    loaded={};keys=set();counts={'train':0,'val':0}
    with (out/'jobs.jsonl.tmp').open('w') as output:
        for score,fi,index,pieces,check,top,is_val in selected:
            if fi not in loaded:loaded[fi]=np.load(all_files[fi],mmap_mode='r',allow_pickle=False)
            row=loaded[fi][index];b=decode(row);key=position_key(b).hex()
            if key in keys:raise ValueError('Duplicate hard position')
            keys.add(key);ms,ids=legal_ids(b);mapping={int(i):m.uci() for i,m in zip(ids,ms)}
            split='val' if is_val else 'train';counts[split]+=1
            output.write(json.dumps(dict(key=key,fen=b.fen(en_passant='fen'),student_candidates=[mapping[i] for i in top if i in mapping],
                score=score,split=split,pieces=pieces,in_check=check))+'\n')
    (out/'jobs.jsonl.tmp').replace(out/'jobs.jsonl')
    report=dict(counts=counts,pool=len(candidates),method='policy CE + 2*annotated Q MAE + WDL CE + policy top1 mismatch',
                note='Reannotation of existing distinct engine positions; not new unique positions relative to the core dataset',
                checkpoint_sha256=hashlib.file_digest(open(a.checkpoint,'rb'),'sha256').hexdigest(),seconds=time.time()-started)
    (out/'mining.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)


if __name__=='__main__':main()
