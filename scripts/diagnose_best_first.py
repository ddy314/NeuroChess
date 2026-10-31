"""Freeze failing trees, compare evaluator/backup/scheduler interventions."""
import argparse
import gzip
import importlib.util
import json
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
from neurochess.cheap import load_cheap
from neurochess.encoding import encode,legal_ids
from neurochess.multihead import MultiheadPlayer
from neurochess.native_mcts import NativeMCTSPlayer
from neurochess.search_diagnostics import (OracleCorrections,teacher_label,tree_children,
    node_board,node_path,fixed_tree_backup)
from neurochess.train import load_model


def main():
    p=argparse.ArgumentParser();p.add_argument('--cheap',required=True);p.add_argument('--checkpoint',required=True)
    p.add_argument('--openings',required=True);p.add_argument('--stockfish',required=True);p.add_argument('--out',required=True)
    p.add_argument('--limit',type=int,default=8);p.add_argument('--oracle',action='store_true')
    a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);torch.set_float32_matmul_precision('high')
    model,state=load_model(a.checkpoint,'cuda');del state;network=MultiheadPlayer(model,'cuda')
    student,_=load_cheap(Path(a.cheap).with_suffix('.pt'));rng=np.random.default_rng(3171)
    variants=[('baseline',{}),('no-internal-anchors',dict(anchor_weight=0,q_anchor_weight=0)),
        ('decay-8',dict(anchor_decay=8)),('frontier',dict(correction_scheduler='frontier')),
        ('frontier-pure',dict(correction_scheduler='frontier',anchor_weight=0,q_anchor_weight=0)),
        ('decay-frontier',dict(correction_scheduler='frontier',anchor_decay=8)),
        ('policy-regularized',dict(root_policy_weight=.1)),
        ('decay-frontier-policy',dict(correction_scheduler='frontier',anchor_decay=8,root_policy_weight=.1))]
    fens=Path(a.openings).read_text().splitlines()[:a.limit];results=[];error_samples=[]
    with chess.engine.SimpleEngine.popen_uci(a.stockfish,timeout=60) as teacher:
        teacher.configure({'Threads':1,'Hash':64})
        oracle=OracleCorrections(network,teacher,50000)
        for index,fen in enumerate(fens):
            board=chess.Board(fen);rows=[];baseline=None
            for name,opts in variants+([('oracle-WDL',{})] if a.oracle else []):
                player=BestFirstPlayer(a.cheap,oracle if name=='oracle-WDL' else network,
                    expansions=512,neural_budget=32,cache_size=0,**opts)
                start=time.perf_counter();move=player.choose(board);trace=player.trace()
                row=dict(method=name,move=move.uci(),seconds=time.perf_counter()-start,
                    stats=player.search_stats,options=opts)
                if name=='baseline':
                    baseline=trace;ch=tree_children(trace)
                    with gzip.open(out/f'tree-{index:03d}.json.gz','wt') as f:json.dump(trace,f)
                    fixed=fixed_tree_backup(trace);row['fixed_tree_pure_move']=fixed['move']
                    row['fixed_tree_pure_leaders']=fixed['leaders'][:8]
                    # Label roots, every root child, and reproducible deep frontier samples.
                    candidates=[(0,'root')]+[(i,'root_child') for i in ch[0]]
                    leaves=[n['id'] for n in trace['nodes'] if not n['expanded'] and n['depth']>=4 and n['solved']==2]
                    if leaves:
                        candidates += [(int(i),'deep_frontier') for i in rng.choice(leaves,min(8,len(leaves)),replace=False)]
                    # The actual selected minimax line is more relevant than uniform leaves.
                    i=next(i for i in ch[0] if trace['nodes'][i]['move']==move.uci())
                    while ch.get(i):
                        i=min(ch[i],key=lambda x:trace['nodes'][x]['value'])
                    candidates.append((i,'selected_frontier'))
                    boards=[node_board(board,trace,i) for i,_ in candidates]
                    encoded=[encode(b) for b in boards];ids=np.full((len(boards),256),-1,np.int16)
                    for j,b in enumerate(boards):
                        legal=legal_ids(b)[1];ids[j,:len(legal)]=legal
                    tokens=np.stack([e[0] for e in encoded]);meta=np.stack([e[1] for e in encoded])
                    _,_,big=network.evaluate_search_heads(tokens,meta,ids)
                    with torch.inference_mode():small=student(torch.from_numpy(tokens),torch.from_numpy(meta)).numpy()
                    for j,((node,kind),b) in enumerate(zip(candidates,boards)):
                        label=teacher_label(teacher,b,50000)
                        error_samples.append(dict(case=index,node=node,kind=kind,path=node_path(trace,node),
                            fen=b.fen(),cheap=float(small[j]),large=float(big[j]),search=trace['nodes'][node]['value'],teacher=label))
                if name=='oracle-WDL':row['teacher_queries_total']=oracle.calls
                rows.append(row);player.close()
            puct=NativeMCTSPlayer(network,simulations=512,leaf_batch_size=64)
            move=puct.choose(board);rows.append(dict(method='puct',move=move.uci(),stats=puct.search_stats));puct.close()
            first=teacher_label(teacher,board,100000)
            moves=list(dict.fromkeys([board.parse_uci(r['move']) for r in rows]+[board.parse_uci(first['pv'][0])]))
            infos=teacher.analyse(board,chess.engine.Limit(nodes=200000),root_moves=moves,multipv=len(moves),game=object())
            labels={}
            for info in infos:
                score=info['score'].pov(board.turn);wdl=score.wdl(model='sf16',ply=32)
                labels[info['pv'][0].uci()]=dict(cp=score.score(),mate=score.mate(),value=(wdl.wins-wdl.losses)/1000,
                    pv=[m.uci() for m in info['pv'][:8]])
            best=max(x['value'] for x in labels.values());cp_best=max(x['cp'] for x in labels.values() if x['cp'] is not None)
            for r in rows:
                r['teacher']=labels[r['move']];r['value_regret']=best-r['teacher']['value']
                r['cp_regret']=cp_best-r['teacher']['cp'] if r['teacher']['cp'] is not None else None
            results.append(dict(index=index,fen=fen,methods=rows,teacher_candidates=labels))
            report=dict(args=vars(a),positions=results,error_samples=error_samples,
                note='Development interventions; concurrent capacity training makes these timings unsuitable for latency claims.')
            (out/'progress.json').write_text(json.dumps(report,indent=2))
            print(json.dumps(dict(index=index,methods=[dict(method=r['method'],move=r['move'],regret=r['cp_regret']) for r in rows])),flush=True)
    summary={}
    for name in [r['method'] for r in results[0]['methods']]:
        rs=[next(r for r in e['methods'] if r['method']==name) for e in results]
        summary[name]=dict(mean_cp_regret=float(np.mean([r['cp_regret'] for r in rs if r['cp_regret'] is not None])),
            mean_value_regret=float(np.mean([r['value_regret'] for r in rs])))
    report['summary']=summary
    report['error_strata']={}
    for kind in sorted({r['kind'] for r in error_samples}):
        rs=[r for r in error_samples if r['kind']==kind]
        report['error_strata'][kind]=dict(positions=len(rs),**{f'{name}_mse':float(np.mean([
            (r[name]-r['teacher']['value'])**2 for r in rs])) for name in ['cheap','large','search']})
    (out/'result.json').write_text(json.dumps(report,indent=2));print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
