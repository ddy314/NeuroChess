"""Fixed-position, cold-search comparison: minimax query budgets vs native PUCT.

Includes local Stockfish search over the union of selected moves. This measures
latency/query use and candidate-set regret, not playing strength or absolute Elo.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
if importlib.util.find_spec('chess') is None:
    for site in (ROOT/'.venv/lib').glob('python*/site-packages'):
        if (site/'chess').is_dir():
            sys.path.append(str(site)); break
import chess
import chess.engine
import torch
from neurochess.best_first import BestFirstPlayer
from neurochess.multihead import MultiheadPlayer
from neurochess.native_mcts import NativeMCTSPlayer
from neurochess.train import load_model


def digest(path):
    with open(path,'rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cheap', required=True); p.add_argument('--checkpoint', required=True)
    p.add_argument('--openings', required=True); p.add_argument('--stockfish', required=True)
    p.add_argument('--out', required=True); p.add_argument('--limit',type=int,default=16)
    p.add_argument('--expansions',type=int,default=512)
    p.add_argument('--budgets',default='0,8,16,32')
    p.add_argument('--simulations',type=int,default=512)
    p.add_argument('--teacher-nodes',type=int,default=50000)
    p.add_argument('--quiescence-depth',type=int,default=0)
    p.add_argument('--device',default='cuda'); p.add_argument('--threads',type=int,default=4)
    a = p.parse_args(); out = Path(a.out); out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(a.threads); torch.set_float32_matmul_precision('high')
    budgets = [int(s) for s in a.budgets.split(',')]
    fens = [s.strip() for s in Path(a.openings).read_text().splitlines() if s.strip()][:a.limit]
    if not fens or len(set(fens))!=len(fens):
        raise ValueError('Distinct nonempty positions required')
    model, state = load_model(a.checkpoint,a.device); del state
    network = MultiheadPlayer(model,a.device)
    from neurochess.encoding import encode, legal_ids
    import numpy as np
    for batch in [1,4,64]:
        b = chess.Board(fens[0]); t,m = encode(b); ids = np.full((batch,256),-1,np.int16)
        legal = legal_ids(b)[1]; ids[:,:len(legal)]=legal
        for _ in range(2):
            network.evaluate_search_heads(np.tile(t,(batch,1)),np.tile(m,(batch,1)),ids)
    report = dict(args=vars(a), checkpoint_sha256=digest(a.checkpoint), cheap_sha256=digest(a.cheap),
        openings_sha256=digest(a.openings), device=torch.cuda.get_device_name() if a.device=='cuda' else a.device,
        torch=torch.__version__, positions=[],
        notes=['Cold tree and empty output cache for each position/method; warmed network; model load excluded.',
               'Other local GPU jobs may contend; counts are exact, timing is environment-dependent.',
               'Teacher regret is within one coherently searched candidate union, not global optimum or Elo.',
               'These development positions are not a final independent strength test.'])
    report['sources'] = {f:digest(ROOT/f) for f in [
        'neurochess/native/best_first.cpp','neurochess/native/chess_rules.hpp',
        'neurochess/best_first.py','neurochess/cheap.py','neurochess/native/mcts.cpp',
        'neurochess/multihead.py','scripts/benchmark_best_first.py']}
    with chess.engine.SimpleEngine.popen_uci(a.stockfish,timeout=60) as teacher:
        teacher.configure({'Threads':1,'Hash':64,'Skill Level':20,'UCI_LimitStrength':False})
        report['teacher'] = dict(id=teacher.id, binary_sha256=digest(a.stockfish), nodes=a.teacher_nodes)
        for index,fen in enumerate(fens):
            b = chess.Board(fen)
            if not b.is_valid() or b.is_game_over(claim_draw=True):
                raise ValueError(f'Invalid evaluation root: {fen}')
            methods = [('direct',None)] + [(f'best-first-{n}',n) for n in budgets] + [('puct',-1)]
            rows=[]
            for name,budget in methods:
                if name=='direct': player=network
                elif name=='puct': player=NativeMCTSPlayer(network,simulations=a.simulations,leaf_batch_size=64)
                else: player=BestFirstPlayer(a.cheap,network,expansions=a.expansions,neural_budget=budget,
                                            correction_batch=4,quiescence_depth=a.quiescence_depth)
                before = network.search_stats['forward_positions']; start = time.perf_counter()
                move = player.choose(b); elapsed = time.perf_counter()-start
                row = dict(method=name,move=move.uci(),seconds=elapsed,
                    large_model_positions=network.search_stats['forward_positions']-before,
                    stats=dict(player.search_stats))
                if isinstance(player,BestFirstPlayer): row['root']=player.last_result
                rows.append(row)
                if hasattr(player,'close'): player.close()
            initial = teacher.analyse(b,chess.engine.Limit(nodes=a.teacher_nodes),game=object())
            candidates = list(dict.fromkeys([initial['pv'][0]]+[chess.Move.from_uci(r['move']) for r in rows]))
            infos = teacher.analyse(b,chess.engine.Limit(nodes=a.teacher_nodes),root_moves=candidates,
                                    multipv=len(candidates),game=object())
            labels = {}
            for info in infos:
                s = info['score'].pov(b.turn)
                labels[info['pv'][0].uci()] = dict(cp=s.score(),mate=s.mate(),depth=info.get('depth'),
                    value=(s.wdl(model='sf16',ply=32).wins-s.wdl(model='sf16',ply=32).losses)/1000)
            best = max(labels.values(),key=lambda r:r['value'])['value']
            cp_best = max((x['cp'] for x in labels.values() if x['cp'] is not None),default=None)
            for row in rows:
                label=labels[row['move']]; row['teacher']=label
                row['candidate_value_regret']=best-label['value']
                row['candidate_cp_regret']=cp_best-label['cp'] if cp_best is not None and label['cp'] is not None else None
            entry=dict(index=index,fen=fen,methods=rows,teacher_candidates=labels)
            report['positions'].append(entry)
            (out/'progress.json').write_text(json.dumps(report,indent=2))
            print(json.dumps(dict(index=index,methods=[{k:r[k] for k in [
                'method','move','seconds','large_model_positions','candidate_value_regret']} for r in rows])),flush=True)
    summary={}
    for name in [r['method'] for r in report['positions'][0]['methods']]:
        rows=[next(r for r in e['methods'] if r['method']==name) for e in report['positions']]
        latency=sorted(r['seconds'] for r in rows)
        cp=[r['candidate_cp_regret'] for r in rows if r['candidate_cp_regret'] is not None]
        summary[name]=dict(positions=len(rows),median_seconds=statistics.median(latency),
            p95_seconds=float(np.quantile(latency,.95)),
            mean_large_model_positions=statistics.mean(r['large_model_positions'] for r in rows),
            mean_value_regret=statistics.mean(r['candidate_value_regret'] for r in rows),
            mean_cp_regret=statistics.mean(cp) if cp else None,
            cp_regret_positions=len(cp))
    report['summary']=summary
    (out/'result.json').write_text(json.dumps(report,indent=2)); print(json.dumps(summary,indent=2),flush=True)


if __name__ == '__main__': main()
