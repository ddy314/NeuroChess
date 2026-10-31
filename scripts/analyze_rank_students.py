"""Recheck selected students on grouped data and against native CPU inference."""
import argparse
import ctypes as C
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import chess
import numpy as np
import torch
from neurochess.best_first import BestFirstPlayer
from neurochess.cheap import load_cheap
from neurochess.encoding import encode,position_key
from neurochess.native_mcts import history
from scripts.train_rank_cheap import validate


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();root=Path(a.root)
    torch.set_num_threads(2);data=root/'data';train=np.load(data/'train.npy',mmap_mode='r');val=np.load(data/'val.npy',mmap_mode='r')
    rng=np.random.default_rng(431903);sample=train[rng.choice(len(train),min(2048,len(train)),replace=False)]
    roots=[r for r in json.loads((data/'roots.json').read_text()) if r['split']=='eval'][:100]
    paths={'old':'output/cheap-minimax-20261031/student-selected/best.pt',**{n:str(root/n/'best.pt') for n in ['wdl','cp','cp-rank']}}
    result={}
    for name,path in paths.items():
        model,state=load_cheap(path);player=BestFirstPlayer(Path(path).with_suffix('.bin'),expansions=1)
        pred_error=acc_error=0.
        for r in roots:
            b=chess.Board(r['initial_fen'])
            for m in r['history']:b.push_uci(m)
            t,m=encode(b)
            with torch.inference_mode():expected=model(torch.from_numpy(t[None]),torch.from_numpy(m[None])).item()
            actual,error=C.c_float(),C.c_float()
            player._check(player.lib.bf_probe(player.handle,*history(b),C.byref(actual),C.byref(error)))
            pred_error=max(pred_error,abs(actual.value-np.clip(expected,-.9999,.9999)));acc_error=max(acc_error,error.value)
        player.close();assert pred_error<1e-4 and acc_error<1e-4
        result[name]=dict(checkpoint=path,step=state.get('step'),parameters=sum(p.numel() for p in model.parameters()),
            binary_bytes=Path(path).with_suffix('.bin').stat().st_size,
            checkpoint_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            validation=validate(model,val),train_sample=validate(model,sample),
            native_parity=dict(positions=len(roots),max_prediction_error=float(pred_error),max_incremental_error=acc_error))
    keys={r['key'][i].tobytes() for rows in [train,val] for r in rows for i in range(5) if r['valid'][i]}
    critical=json.loads(Path('output/search-diagnosis-20261031/old-game-audit/result.json').read_text())['critical']
    result['known_failure_overlap']=[dict(game=c['game'],overlap=position_key(chess.Board(c['fen'])) in keys) for c in critical]
    (root/'student-analysis.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
