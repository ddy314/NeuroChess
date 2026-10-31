"""CPU-only profiling with fixed synthetic corrections; never a strength test."""
import argparse
import ctypes as C
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import chess
import numpy as np
from neurochess.best_first import BestFirstPlayer
from neurochess.native_mcts import history


class FixedNetwork:
    device='cpu'
    def evaluate_encoded(self,tokens,meta,ids):
        return np.zeros(ids.shape,np.float32),np.zeros(len(tokens),np.float32)


def main():
    p=argparse.ArgumentParser();p.add_argument('--weights',required=True);p.add_argument('--fens',required=True)
    p.add_argument('--out',required=True);p.add_argument('--positions',type=int,default=16);a=p.parse_args()
    fens=Path(a.fens).read_text().splitlines()[:a.positions];rows=[]
    for repeats in range(3):
        for index,fen in enumerate(fens):
            player=BestFirstPlayer(a.weights,FixedNetwork(),expansions=512,neural_budget=32,
                root_policy_weight=.1,align_root_objective=True,local_exploration=True,
                mate_nodes=16384,tactical_nodes=16384)
            start=time.perf_counter();move=player.choose(chess.Board(fen));elapsed=time.perf_counter()-start
            rows.append(dict(repeat=repeats,index=index,seconds=elapsed,move=move.uci(),stats=player.search_stats,result=player.last_result))
            player.close()
    player=BestFirstPlayer(a.weights,expansions=1);hist=history(chess.Board(fens[0]));v,e=C.c_float(),C.c_float()
    start=time.perf_counter()
    for _ in range(20000):player._check(player.lib.bf_probe(player.handle,*hist,C.byref(v),C.byref(e)))
    probe=(time.perf_counter()-start)/20000;player.close()
    out=dict(rows=rows,median_cpu_seconds=float(np.median([r['seconds'] for r in rows])),probe_seconds=probe,
        note='Fixed zero corrections; CPU profiling, not GPU latency or strength evidence.')
    Path(a.out).write_text(json.dumps(out,indent=2));print(json.dumps({k:v for k,v in out.items() if k!='rows'}))


if __name__=='__main__':main()
