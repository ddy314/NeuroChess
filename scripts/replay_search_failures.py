"""Replay failures selected from recorded games, retaining their legal histories."""
import argparse
import gzip
import importlib.util
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
if importlib.util.find_spec('chess') is None:sys.path.append(str(next((ROOT/'.venv/lib').glob('python*/site-packages'))))
import chess
import chess.engine
import numpy as np
import torch
from neurochess.best_first import BestFirstPlayer
from neurochess.cheap import load_cheap
from neurochess.encoding import encode,legal_ids
from neurochess.multihead import MultiheadPlayer
from neurochess.native_mcts import NativeMCTSPlayer
from neurochess.search_diagnostics import OracleCorrections,tree_children
from neurochess.train import load_model


def main():
    p=argparse.ArgumentParser();p.add_argument('--audit',required=True);p.add_argument('--cheap',required=True)
    p.add_argument('--checkpoint',required=True);p.add_argument('--stockfish',required=True);p.add_argument('--out',required=True)
    p.add_argument('--verify-final',action='store_true')
    a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);torch.set_float32_matmul_precision('high');model,state=load_model(a.checkpoint,'cuda');del state
    network=MultiheadPlayer(model,'cuda');student,_=load_cheap(Path(a.cheap).with_suffix('.pt'));results=[]
    with chess.engine.SimpleEngine.popen_uci(a.stockfish,timeout=60) as teacher:
        teacher.configure({'Threads':1,'Hash':64});oracle=OracleCorrections(network,teacher,50000)
        for job in json.loads(Path(a.audit).read_text())['critical']:
            board=chess.Board(job['initial_fen'])
            for m in job['history']:board.push_uci(m)
            rows=[];baseline=None
            variants=[('baseline',{}),('guarded',dict(root_policy_weight=.1)),
                      ('aligned',dict(root_policy_weight=.1,align_root_objective=True)),('oracle-WDL',{})]
            if a.verify_final:
                variants += [('aligned-checked',dict(root_policy_weight=.1,align_root_objective=True,verify_final=True)),
                             ('oracle-WDL-checked',dict(verify_final=True))]
            for name,opts in variants:
                player=BestFirstPlayer(a.cheap,oracle if name.startswith('oracle-WDL') else network,expansions=512,
                    neural_budget=32,cache_size=0,**opts)
                move=player.choose(board);trace=player.trace()
                with gzip.open(out/f"game-{job['game']}-{name}.json.gz",'wt') as f:json.dump(trace,f)
                rows.append(dict(method=name,move=move.uci(),root=player.last_result))
                if name=='baseline':baseline=trace
                player.close()
            for sims,batch in [(31,4),(512,64)]:
                player=NativeMCTSPlayer(network,simulations=sims,leaf_batch_size=batch)
                rows.append(dict(method=f'puct-{sims}',move=player.choose(board).uci()));player.close()
            moves=list(dict.fromkeys([board.parse_uci(r['move']) for r in rows]+
                [board.parse_uci(job['move']),board.parse_uci(job['best'])]))
            infos=teacher.analyse(board,chess.engine.Limit(nodes=500000),root_moves=moves,multipv=len(moves),game=object())
            labels={}
            for info in infos:
                score=info['score'].pov(board.turn);w=score.wdl(model='sf16',ply=32)
                labels[info['pv'][0].uci()]=dict(cp=score.score(),mate=score.mate(),value=(w.wins-w.losses)/1000,
                    pv=[m.uci() for m in info['pv'][:10]])
            children=tree_children(baseline);nodes=baseline['nodes'];ids=[];boards=[]
            for uci in [job['move'],job['best']]:
                idx=next(i for i in children[0] if nodes[i]['move']==uci);ids.append(idx)
                b=board.copy();b.push_uci(uci);boards.append(b)
            encoded=[encode(b) for b in boards];action_ids=np.full((2,256),-1,np.int16)
            for i,b in enumerate(boards):
                ms=legal_ids(b)[1];action_ids[i,:len(ms)]=ms
            t=np.stack([e[0] for e in encoded]);meta=np.stack([e[1] for e in encoded])
            _,_,large=network.evaluate_search_heads(t,meta,action_ids)
            with torch.inference_mode():small=student(torch.from_numpy(t),torch.from_numpy(meta)).numpy()
            comparisons=[]
            for j,uci in enumerate([job['move'],job['best']]):
                node=nodes[ids[j]];coverage=[];current=0
                for move in labels[uci]['pv']:
                    match=[i for i in children.get(current,[]) if nodes[i]['move']==move]
                    if not match:coverage.append(dict(move=move,present=False));break
                    current=match[0];n=nodes[current]
                    coverage.append(dict(move=move,present=True,visits=n['visits'],expanded=n['expanded'],
                        corrected=n['corrected'],value=n['value']))
                comparisons.append(dict(move=uci,cheap_root_value=-float(small[j]),large_root_value=-float(large[j]),
                    search_root_value=-node['value'],teacher=labels[uci],root_node=node,teacher_pv_coverage=coverage))
            cp_best=max(x['cp'] for x in labels.values() if x['cp'] is not None)
            for row in rows:
                row['teacher']=labels[row['move']];row['cp_regret']=cp_best-row['teacher']['cp'] if row['teacher']['cp'] is not None else None
            result=dict(game=job['game'],ply=job['ply'],fen=job['fen'],original_move=job['move'],
                baseline_reproduced=rows[0]['move']==job['move'],methods=rows,comparisons=comparisons)
            results.append(result);print(json.dumps(dict(game=job['game'],reproduced=result['baseline_reproduced'],
                methods={r['method']:(r['move'],r['cp_regret']) for r in rows})),flush=True)
    (out/'result.json').write_text(json.dumps(dict(args=vars(a),cases=results,
        note='Cold replays with full histories; teacher oracle is diagnostic only; timings not used.'),indent=2))


if __name__=='__main__':main()
