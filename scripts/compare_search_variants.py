"""Source-fingerprinted, counterbalanced comparison on a frozen position suite."""
import argparse
import hashlib
import importlib.util
import json
import shutil
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
if importlib.util.find_spec('chess') is None:
    sys.path.append(str(next((ROOT/'.venv/lib').glob('python*/site-packages'))))
import chess
import chess.engine
import numpy as np
import torch
from neurochess.best_first import BestFirstPlayer
import neurochess.best_first as best_first_module
from neurochess.encoding import encode,legal_ids
from neurochess.multihead import MultiheadPlayer
from neurochess.native_mcts import NativeMCTSPlayer
from neurochess.train import load_model


def digest(path):
    with open(path,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--suite',required=True);p.add_argument('--openings',required=True)
    p.add_argument('--checkpoint',required=True);p.add_argument('--stockfish',required=True);p.add_argument('--out',required=True)
    p.add_argument('--teacher-nodes',type=int,default=500000);p.add_argument('--limit',type=int)
    p.add_argument('--candidate-plan',help='Teacher-only reference moves from a frozen tactical plan; never supplied to players')
    a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    # Validate once, then pin this loaded binary for the complete experiment.
    # Subsequent builds cannot change the implementation between positions.
    native=best_first_module.load_library()
    best_first_module.load_library=lambda:native
    shutil.copy2(ROOT/'neurochess/native/libbestfirst.so',out/'libbestfirst.frozen.so')
    suite=json.loads(Path(a.suite).read_text());fens=[x for x in Path(a.openings).read_text().splitlines() if x.strip()]
    references={r['fen']:r.get('reference_moves',[]) for r in json.loads(Path(a.candidate_plan).read_text())['roots']} if a.candidate_plan else {}
    if a.limit:fens=fens[:a.limit]
    torch.set_num_threads(2);torch.set_float32_matmul_precision('high')
    model,state=load_model(a.checkpoint,'cuda');del state;network=MultiheadPlayer(model,'cuda')
    b=chess.Board(fens[0]);t,m=encode(b);ids=np.full((64,256),-1,np.int16);legal=legal_ids(b)[1];ids[:,:len(legal)]=legal
    for batch in [1,4,16,64]:
        for _ in range(2):network.evaluate_search_heads(np.tile(t,(batch,1)),np.tile(m,(batch,1)),ids[:batch])
    report=dict(args=vars(a),suite=suite,checkpoint_sha256=digest(a.checkpoint),
        openings_sha256=digest(a.openings),student_hashes={v['cheap']:digest(v['cheap']) for v in suite if v.get('cheap')},
        native_library_sha256=digest(out/'libbestfirst.frozen.so'),
        source_hashes={str(p):digest(p) for p in [Path('neurochess/native/best_first.cpp'),Path('neurochess/best_first.py'),
            Path('neurochess/cheap.py'),Path('neurochess/native/mcts.cpp'),Path(__file__)]},positions=[],
        note='Frozen exact-position holdout; not guaranteed game/source-independent from original base training. Candidate-union regret, not Elo.')
    if a.candidate_plan:report['candidate_plan_sha256']=digest(a.candidate_plan)
    with chess.engine.SimpleEngine.popen_uci(a.stockfish,timeout=60) as teacher:
        teacher.configure({'Threads':1,'Hash':64});report['teacher']=dict(id=teacher.id,binary_sha256=digest(a.stockfish))
        for index,fen in enumerate(fens):
            b=chess.Board(fen);rows=[];order=suite[index%len(suite):]+suite[:index%len(suite)]
            for variant in order:
                name=variant['name'];kind=variant['kind']
                if kind=='direct':player=network
                elif kind=='puct':player=NativeMCTSPlayer(network,**variant['options'])
                else:player=BestFirstPlayer(variant['cheap'],network,**variant['options'])
                before=network.search_stats['forward_positions'];start=time.perf_counter();move=player.choose(b);seconds=time.perf_counter()-start
                row=dict(method=name,move=move.uci(),seconds=seconds,large_positions=network.search_stats['forward_positions']-before,
                    stats=dict(player.search_stats))
                if kind=='best-first':row['root']=player.last_result
                rows.append(row)
                if hasattr(player,'close'):player.close()
            info=teacher.analyse(b,chess.engine.Limit(nodes=100000),game=object())
            moves=list(dict.fromkeys([chess.Move.from_uci(r['move']) for r in rows]+[info['pv'][0]]))
            moves=list(dict.fromkeys(moves+[b.parse_uci(m) for m in references.get(fen,[])]))
            infos=teacher.analyse(b,chess.engine.Limit(nodes=a.teacher_nodes),root_moves=moves,multipv=len(moves),game=object())
            labels={}
            for i in infos:
                s=i['score'].pov(b.turn);w=s.wdl(model='sf16',ply=32)
                labels[i['pv'][0].uci()]=dict(cp=s.score(),mate=s.mate(),value=(w.wins-w.losses)/1000,
                    depth=i.get('depth'),pv=[m.uci() for m in i['pv'][:8]])
            all_cp=all(x['cp'] is not None for x in labels.values())
            cp_best=max(x['cp'] for x in labels.values()) if all_cp else None
            value_best=max(x['value'] for x in labels.values())
            for row in rows:
                row['teacher']=labels[row['move']];row['value_regret']=value_best-row['teacher']['value']
                row['cp_regret']=cp_best-row['teacher']['cp'] if all_cp else None
            report['positions'].append(dict(index=index,fen=fen,methods=rows,teacher_candidates=labels))
            (out/'progress.json').write_text(json.dumps(report,indent=2))
            print(json.dumps(dict(index=index,regrets={r['method']:r['cp_regret'] for r in rows})),flush=True)
    summary={}
    for v in suite:
        rows=[next(r for r in e['methods'] if r['method']==v['name']) for e in report['positions']]
        cp=[r['cp_regret'] for r in rows if r['cp_regret'] is not None]
        summary[v['name']]=dict(positions=len(rows),cp_positions=len(cp),mean_cp_regret=float(np.mean(cp)) if cp else None,
            median_cp_regret=float(np.median(cp)) if cp else None,cp_blunders_100=sum(c>=100 for c in cp),
            mean_value_regret=float(np.mean([r['value_regret'] for r in rows])),
            median_seconds=float(np.median([r['seconds'] for r in rows])),
            mean_large_positions=float(np.mean([r['large_positions'] for r in rows])))
    report['summary']=summary;(out/'result.json').write_text(json.dumps(report,indent=2));print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
