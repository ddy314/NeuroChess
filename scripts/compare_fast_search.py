"""Frozen2s baseline versus candidate inference/search, graded by shared teacher."""
import argparse,gc,hashlib,importlib.util,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
if importlib.util.find_spec('chess') is None:sys.path.append(str(next((ROOT/'.venv/lib').glob('python*/site-packages'))))
import chess,chess.engine,numpy as np,torch
from neurochess.train import load_model
from neurochess.multihead import MultiheadPlayer
from neurochess.native_mcts import NativeMCTSPlayer
from neurochess.fast_inference import resident_linears,TritonChessModel


def main():
    p=argparse.ArgumentParser()
    for arg in ['suite','openings','out']:p.add_argument('--'+arg,required=True)
    p.add_argument('--limit',type=int);a=p.parse_args();root=Path('output/puct-10x-20261031');manifest=json.loads((root/'frozen-baseline/manifest.json').read_text());out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    for file,sha in manifest['source_sha256'].items():
        if file.endswith(('.cpp','.hpp','.py','.so')):assert hashlib.sha256(Path(file).read_bytes()).hexdigest()==sha,(file,'locked baseline changed')
    checkpoint=manifest['checkpoint'];assert hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()==manifest['checkpoint_sha256']
    suite=json.loads(Path(a.suite).read_text());fens=Path(a.openings).read_text().splitlines();fens=fens[:a.limit] if a.limit else fens
    torch.set_num_threads(2);torch.set_float32_matmul_precision('high');nets={};warm={}
    from neurochess.encoding import encode,legal_ids
    b=chess.Board(fens[0]);t,m=encode(b);ids=np.full((64,256),-1,np.int16);li=legal_ids(b)[1];ids[:,:len(li)]=li
    for backend in sorted({v.get('backend','eager') for v in suite}):
        setup=time.perf_counter();model,state=load_model(checkpoint,'cuda');del state
        if backend in ['tensorrt','tensorrt-int8','tensorrt-int8graph']:
            sys.path.insert(0,str(ROOT/'output/puct-10x-20261031/trt-deps'))
            from neurochess.tensorrt_inference import TensorRTModel
            model=TensorRTModel(ROOT/'output/puct-10x-20261031'/('tensorrt-int8' if backend.startswith('tensorrt-int8') else 'tensorrt')/'model.plan',model.config)
        if backend in ['int8-ffn','int8-all','int8-graph']:
            from neurochess.int8_inference import quantize_linears
            model=quantize_linears(model,'all' if backend in ['int8-all','int8-graph'] else 'ffn')
        if backend in ['int8-fused','int8-fusedgraph']:
            from neurochess.int8_inference import Int8FusedChessModel
            model=Int8FusedChessModel(model)
        if backend=='resident':model=resident_linears(model)
        if backend=='triton':model=TritonChessModel(model)
        if backend=='compile':model=torch.compile(resident_linears(model),fullgraph=True,dynamic=True)
        net=MultiheadPlayer(model,'cuda');
        if backend=='tensorrt-int8graph':
            from neurochess.tensorrt_graph import TensorRTGraphMultiheadPlayer
            net=TensorRTGraphMultiheadPlayer(model,'cuda',batch_sizes=range(2,17))
        if backend=='int8-fusedgraph':
            from neurochess.fused_heads import FusedGraphMultiheadPlayer
            net=FusedGraphMultiheadPlayer(model,'cuda',batch_sizes=list(range(2,17)))
        if backend=='int8-graph':
            from neurochess.cuda_inference import CudaGraphMultiheadPlayer
            net=CudaGraphMultiheadPlayer(model,'cuda',batch_sizes=list(range(2,17)))
        tic=time.perf_counter()
        for n in range(1,1+max(v['options']['leaf_batch_size'] for v in suite if v.get('backend','eager')==backend)):net.evaluate_search_heads(np.tile(t,(n,1)),np.tile(m,(n,1)),ids[:n])
        warm[backend]=time.perf_counter()-setup;nets[backend]=net
    sources=[*Path('neurochess').glob('*inference.py'),Path('neurochess/fused_heads.py'),Path('neurochess/tensorrt_graph.py'),Path('neurochess/pipeline_mcts.py'),Path('neurochess/native/mcts_pipeline.cpp'),Path('neurochess/native/libmcts_pipeline.so'),Path(__file__)]
    report=dict(args=vars(a),suite=suite,checkpoint_sha256=manifest['checkpoint_sha256'],source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},engine_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in root.glob('tensorrt*/model.plan')},warmup_seconds=warm,positions=[],note='Frozen final2s baseline; GPU jobs serialized. Candidate reductions are not presumed quality-preserving. Finite teacher grades are not game outcomes.')
    with chess.engine.SimpleEngine.popen_uci('local/stockfish/stockfish/stockfish-ubuntu-x86-64-avx2',timeout=120) as sf:
        sf.configure({'Threads':1,'Hash':64})
        for index,fen in enumerate(fens):
            b=chess.Board(fen);rows=[]
            for variant in suite[index%len(suite):]+suite[:index%len(suite)]:
                net=nets[variant.get('backend','eager')]
                if variant.get('engine')=='pipeline':
                    from neurochess.pipeline_mcts import PipelineMCTSPlayer
                    player=PipelineMCTSPlayer(net,**variant['options'])
                else:player=NativeMCTSPlayer(net,**variant['options'])
                before=net.search_stats['forward_positions'];tic=time.perf_counter();move=player.choose(b);elapsed=time.perf_counter()-tic
                rows.append(dict(method=variant['name'],move=move.uci(),seconds=elapsed,large_positions=net.search_stats['forward_positions']-before,root=player.last_result,stats=dict(player.search_stats)));player.close()
            proposal=sf.analyse(b,chess.engine.Limit(nodes=100000),game=object())['pv'][0];moves=list(dict.fromkeys([b.parse_uci(x['move']) for x in rows]+[proposal]))
            labels={}
            for info in sf.analyse(b,chess.engine.Limit(nodes=1000000),root_moves=moves,multipv=len(moves),game=object()):
                s=info['score'].pov(b.turn);w=s.wdl(model='sf16',ply=32);labels[info['pv'][0].uci()]=dict(cp=s.score(),mate=s.mate(),value=(w.wins-w.losses)/1000)
            best=max(s['cp'] for s in labels.values()) if all(s['cp'] is not None for s in labels.values()) else None
            for row in rows:row.update(teacher=labels[row['move']],cp_regret=best-labels[row['move']]['cp'] if best is not None else None)
            report['positions'].append(dict(index=index,fen=fen,methods=rows,teacher_candidates=labels));(out/'progress.json').write_text(json.dumps(report,indent=2));print(json.dumps(dict(index=index,regrets={x['method']:x['cp_regret'] for x in rows})),flush=True)
    summary={}
    for variant in suite:
        name=variant['name'];rows=[next(x for x in p['methods'] if x['method']==name) for p in report['positions']];cp=[x['cp_regret'] for x in rows if x['cp_regret'] is not None]
        summary[name]=dict(samples=len(rows),cp_samples=len(cp),mean_cp_regret=float(np.mean(cp)) if cp else None,blunders_100=sum(x>=100 for x in cp),median_seconds=float(np.median([x['seconds'] for x in rows])),p95_seconds=float(np.quantile([x['seconds'] for x in rows],.95)),median_simulations=float(np.median([x['root']['new_simulations'] for x in rows])))
    report['summary']=summary;(out/'result.json').write_text(json.dumps(report,indent=2));print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
