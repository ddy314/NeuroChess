"""Frozen-weight backend microbenchmarks, including numeric/decision drift."""
import argparse,gc,importlib.util,json,time,hashlib
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
if importlib.util.find_spec('chess') is None:sys.path.append(str(next((ROOT/'.venv/lib').glob('python*/site-packages'))))
import chess,numpy as np,torch
from neurochess.train import load_model
from neurochess.encoding import encode,legal_ids
from neurochess.multihead import MultiheadPlayer
from neurochess.fast_inference import resident_linears,TritonChessModel


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--out',required=True)
    p.add_argument('--backends',default='eager,resident,triton,compile');p.add_argument('--batches',default='1,16,32,64');a=p.parse_args()
    torch.set_num_threads(2);torch.set_float32_matmul_precision('high');out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    fens=Path('output/puct-2s-20261031/final-roots/roots.fen').read_text().splitlines();batches=list(map(int,a.batches.split(',')));inputs={}
    for n in batches:
        boards=[chess.Board(fens[i%len(fens)]) for i in range(n)];pairs=[encode(b) for b in boards];ids=np.full((n,256),-1,np.int16)
        for i,b in enumerate(boards):li=legal_ids(b)[1];ids[i,:len(li)]=li
        inputs[n]=(np.stack([x[0] for x in pairs]),np.stack([x[1] for x in pairs]),ids)
    report=dict(args=vars(a),checkpoint_sha256=hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest(),torch=str(torch.__version__),gpu=torch.cuda.get_device_name(),backends={},note='Model load/compile/warmup excluded from steady-state but measured separately. Different reduction kernels can change floats; matching sample moves does not prove universal strength parity.')
    refs={}
    for backend in a.backends.split(','):
        model,state=load_model(a.checkpoint,'cuda');del state
        if backend in ['tensorrt','tensorrt-int8']:
            sys.path.insert(0,str(ROOT/'output/puct-10x-20261031/trt-deps'))
            from neurochess.tensorrt_inference import TensorRTModel
            model=TensorRTModel(ROOT/'output/puct-10x-20261031'/('tensorrt-int8' if backend=='tensorrt-int8' else 'tensorrt')/'model.plan',model.config)
        if backend in ['int8-ffn','int8-all','int8-graph']:
            from neurochess.int8_inference import quantize_linears
            model=quantize_linears(model,'all' if backend in ['int8-all','int8-graph'] else 'ffn')
        if backend in ['int8-fused','int8-fusedgraph']:
            from neurochess.int8_inference import Int8FusedChessModel
            model=Int8FusedChessModel(model)
        if backend=='resident':model=resident_linears(model)
        elif backend=='triton':model=TritonChessModel(model)
        elif backend=='compile':model=torch.compile(resident_linears(model),dynamic=True,fullgraph=True,mode='default')
        net=MultiheadPlayer(model,'cuda');
        if backend=='int8-fusedgraph':
            from neurochess.fused_heads import FusedGraphMultiheadPlayer
            net=FusedGraphMultiheadPlayer(model,'cuda',batch_sizes=list(range(2,9))+[16])
        if backend=='int8-graph':
            from neurochess.cuda_inference import CudaGraphMultiheadPlayer
            net=CudaGraphMultiheadPlayer(model,'cuda',batch_sizes=[2,3,4,5,6,7,8,16])
        rows=[];start=time.perf_counter()
        try:
            for n,args in inputs.items():
                warm=time.perf_counter()
                for _ in range(3):net.raw_outputs(*args)
                warm=time.perf_counter()-warm;samples=[]
                for _ in range(12):
                    tic=time.perf_counter();actual=net.raw_outputs(*args);samples.append(time.perf_counter()-tic)
                if backend=='eager':refs[n]=actual.copy()
                legal=args[2]>=0;diff=np.abs(actual-refs[n]);valid=np.concatenate([legal,legal,np.ones((n,3),bool)],axis=1)
                p1,q1,v1=net.evaluate_search_heads(*args)
                from neurochess.multihead import fuse_heads
                p0,_,_,_=fuse_heads(refs[n][:,:256],refs[n][:,256:512],refs[n][:,512:],legal)
                row=dict(batch=n,warmup_seconds=warm,median_seconds=float(np.median(samples)),p95_seconds=float(np.quantile(samples,.95)),max_abs_error=float(diff[valid].max()),mean_abs_error=float(diff[valid].mean()),top1_agreement=int((p1.argmax(1)==p0.argmax(1)).sum()),positions=n)
                rows.append(row);print(json.dumps(dict(backend=backend,**row)),flush=True)
            report['backends'][backend]=dict(rows=rows,total_seconds=time.perf_counter()-start,peak_memory=int(torch.cuda.max_memory_allocated()))
        except Exception as e:
            report['backends'][backend]=dict(error=repr(e),rows=rows);print(repr(e),flush=True)
        (out/'progress.json').write_text(json.dumps(report,indent=2));del net,model;gc.collect();torch.cuda.empty_cache()
    (out/'result.json').write_text(json.dumps(report,indent=2))

if __name__=='__main__':main()
