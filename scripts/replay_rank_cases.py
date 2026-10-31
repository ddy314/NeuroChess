"""Replay the previously identified game failures with the fixed variant suite."""
import argparse
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import shutil

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
if importlib.util.find_spec('chess') is None:sys.path.append(str(next((ROOT/'.venv/lib').glob('python*/site-packages'))))
import chess
import chess.engine
import torch
from neurochess.best_first import BestFirstPlayer
import neurochess.best_first as bf_module
import neurochess.native_mcts as mcts_module
from neurochess.multihead import MultiheadPlayer
from neurochess.native_mcts import NativeMCTSPlayer
from neurochess.train import load_model


def main():
    p=argparse.ArgumentParser()
    for arg in ['audit','suite','checkpoint','stockfish','out']:p.add_argument('--'+arg,required=True)
    p.add_argument('--warm-history',action='store_true',help='Search earlier same-color turns on the recorded trajectory before grading the critical move')
    a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    bf_native=bf_module.load_library();mcts_native=mcts_module.load_library()
    bf_module.load_library=lambda:bf_native;mcts_module.load_library=lambda:mcts_native
    fingerprints={}
    for source in ['neurochess/native/best_first.cpp','neurochess/native/mcts.cpp',
                   'neurochess/native/libbestfirst.so','neurochess/native/libmcts.so',
                   'neurochess/best_first.py','neurochess/native_mcts.py',a.suite,a.audit,a.checkpoint]:
        with open(source,'rb') as f:fingerprints[source]=hashlib.file_digest(f,'sha256').hexdigest()
    for name in ['libbestfirst.so','libmcts.so']:shutil.copy2(ROOT/'neurochess/native'/name,out/name)
    torch.set_num_threads(2);torch.set_float32_matmul_precision('high')
    model,state=load_model(a.checkpoint,'cuda');del state;network=MultiheadPlayer(model,'cuda')
    suite=json.loads(Path(a.suite).read_text());cases=json.loads(Path(a.audit).read_text())['critical'];report=[]
    with chess.engine.SimpleEngine.popen_uci(a.stockfish) as teacher:
        teacher.configure({'Threads':1,'Hash':64})
        for job in cases:
            board=chess.Board(job['initial_fen'])
            for move in job['history']:board.push_uci(move)
            rows=[]
            for v in suite:
                kind=v['kind'];name=v['name']
                if kind=='direct':player=network
                elif kind=='puct':player=NativeMCTSPlayer(network,**v['options'])
                else:player=BestFirstPlayer(v['cheap'],network,**v['options'])
                start=time.perf_counter();before=network.search_stats['forward_positions'];warm_choices=0
                if a.warm_history:
                    replay=chess.Board(job['initial_fen'])
                    for text in job['history']:
                        if replay.turn==board.turn:
                            player.choose(replay);warm_choices+=1
                        replay.push_uci(text)
                prefinal=network.search_stats['forward_positions']
                move=player.choose(board);row=dict(method=name,move=move.uci(),stats=dict(player.search_stats),
                    seconds=time.perf_counter()-start,warm_choices=warm_choices,
                    final_large_positions=network.search_stats['forward_positions']-prefinal,
                    total_large_positions=network.search_stats['forward_positions']-before)
                if kind=='best-first':
                    row['root']=player.last_result
                    with gzip.open(out/f"game-{job['game']}-{name}.json.gz",'wt') as f:json.dump(player.trace(),f)
                rows.append(row)
                if hasattr(player,'close'):player.close()
            moves=list(dict.fromkeys([chess.Move.from_uci(r['move']) for r in rows]+[board.parse_uci(job['move']),board.parse_uci(job['best'])]))
            infos=teacher.analyse(board,chess.engine.Limit(nodes=500000),root_moves=moves,multipv=len(moves),game=object())
            labels={}
            for info in infos:
                score=info['score'].pov(board.turn)
                labels[info['pv'][0].uci()]=dict(cp=score.score(),mate=score.mate(),pv=[m.uci() for m in info['pv'][:10]])
            best=max(s['cp'] for s in labels.values()) if all(s['cp'] is not None for s in labels.values()) else None
            for r in rows:r.update(teacher=labels[r['move']],cp_regret=best-labels[r['move']]['cp'] if best is not None else None)
            report.append(dict(game=job['game'],ply=job['ply'],fen=job['fen'],methods=rows,teacher_candidates=labels))
            print(json.dumps(dict(game=job['game'],regrets={r['method']:r['cp_regret'] for r in rows})),flush=True)
    (out/'result.json').write_text(json.dumps(dict(args=vars(a),fingerprints=fingerprints,cases=report,note='Known diagnostic cases, not independent final evaluation. Full histories preserved; fresh shared candidate-union teacher grades.'),indent=2))


if __name__=='__main__':main()
