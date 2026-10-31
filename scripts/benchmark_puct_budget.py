"""Single-game PUCT latency/quality budget sweep with frozen model/source hashes."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
if importlib.util.find_spec('chess') is None:sys.path.append(str(next((ROOT/'.venv/lib').glob('python*/site-packages'))))
import chess
import numpy as np
import torch
import neurochess.native_mcts as native
from neurochess.multihead import MultiheadPlayer
from neurochess.train import load_model


def digest(path):
    with open(path,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser()
    for k in ['checkpoint','openings','suite','out']:p.add_argument('--'+k,required=True)
    p.add_argument('--limit',type=int,default=16);p.add_argument('--repeats',type=int,default=1);a=p.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False);suite=json.loads(Path(a.suite).read_text())
    lib=native.load_library();native.load_library=lambda:lib;shutil.copy2(ROOT/'neurochess/native/libmcts.so',out/'libmcts.frozen.so')
    fens=[s for s in Path(a.openings).read_text().splitlines() if s.strip()][:a.limit]
    torch.set_num_threads(2);torch.set_float32_matmul_precision('high')
    model,state=load_model(a.checkpoint,'cuda');del state
    networks={'eager':MultiheadPlayer(model,'cuda')}
    if any(v.get('inference')=='cuda-graph' for v in suite):
        from neurochess.cuda_inference import CudaGraphMultiheadPlayer
        networks['cuda-graph']=CudaGraphMultiheadPlayer(model,'cuda',batch_sizes=sorted({v['options']['leaf_batch_size'] for v in suite if v.get('inference')=='cuda-graph'}))
    b=chess.Board(fens[0]);from neurochess.encoding import encode,legal_ids
    t,m=encode(b);ids=np.full((256,256),-1,np.int16);legal=legal_ids(b)[1];ids[:,:len(legal)]=legal
    start=time.perf_counter()
    for name,net in networks.items():
        for batch in sorted({1,4,16}|{v['options']['leaf_batch_size'] for v in suite}):
            for _ in range(2):net.evaluate_search_heads(np.tile(t,(batch,1)),np.tile(m,(batch,1)),ids[:batch])
    warmup=time.perf_counter()-start
    files=['neurochess/native/mcts.cpp','neurochess/native_mcts.py','neurochess/multihead.py','neurochess/model.py',__file__]
    if 'cuda-graph' in networks:files.append('neurochess/cuda_inference.py')
    report=dict(args=vars(a),suite=suite,checkpoint_sha256=digest(a.checkpoint),openings_sha256=digest(a.openings),
        source_sha256={f:digest(f) for f in files},native_sha256=digest(out/'libmcts.frozen.so'),torch=str(torch.__version__),gpu=torch.cuda.get_device_name(),warmup_seconds=warmup,rows=[],
        note='One game slot, fresh tree each root; no cross-game batching. Wall time includes search, transfers, network and fusion; model load/warmup excluded. Batch changes alter search trajectories. Finite sampled latency, not a real-time guarantee.')
    for repeat in range(a.repeats):
        for index,fen in enumerate(fens):
            order=suite[index%len(suite):]+suite[:index%len(suite)]
            for variant in order:
                net=networks[variant.get('inference','eager')];player=native.NativeMCTSPlayer(net,**variant['options']);before=net.search_stats['forward_positions'];tic=time.perf_counter()
                move=player.choose(chess.Board(fen));seconds=time.perf_counter()-tic
                result=player.last_result if hasattr(player,'last_result') else player.decision_samples[-1]
                report['rows'].append(dict(method=variant['name'],index=index,repeat=repeat,fen=fen,move=move.uci(),seconds=seconds,large_positions=net.search_stats['forward_positions']-before,stats=dict(player.search_stats),root=result))
                player.close()
            (out/'progress.json').write_text(json.dumps(report,indent=2));print(json.dumps(dict(repeat=repeat,index=index,seconds={x['method']:round(x['seconds'],4) for x in report['rows'][-len(suite):]})),flush=True)
    summary={}
    for v in suite:
        rows=[r for r in report['rows'] if r['method']==v['name']];ts=[r['seconds'] for r in rows]
        summary[v['name']]=dict(samples=len(rows),median_seconds=float(np.median(ts)),p95_seconds=float(np.quantile(ts,.95)),max_seconds=max(ts),over_2s=sum(t>2 for t in ts),mean_inputs=float(np.mean([r['large_positions'] for r in rows])),
            mean_native_seconds=float(np.mean([r['stats']['native_seconds'] for r in rows])),mean_inference_seconds=float(np.mean([r['stats']['inference_seconds'] for r in rows])))
    report['summary']=summary;(out/'result.json').write_text(json.dumps(report,indent=2));print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
