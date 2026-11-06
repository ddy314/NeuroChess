"""Export the frozen mixed-precision model and build a candidate TensorRT plan."""
import argparse,gc,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'output/puct-10x-20261031/trt-deps'))
sys.path.append(str(next((ROOT/'.venv/lib').glob('python*/site-packages'))))
import torch,tensorrt as trt
from neurochess.train import load_model
from neurochess.fast_inference import resident_linears


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=False);tic=time.perf_counter()
    torch.set_num_threads(2);m,state=load_model('output/puct-10x-20261031/frozen-baseline/model.pt','cuda');del state;m=resident_linears(m)
    tokens=torch.zeros((8,64),dtype=torch.int32,device='cuda');meta=torch.zeros((8,12),device='cuda');moves=torch.zeros((8,256),dtype=torch.int32,device='cuda')
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        torch.onnx.export(m,(tokens,meta,moves),str(out/'model.onnx'),input_names=['tokens','meta','moves'],output_names=['policy','q','wdl_logits'],dynamic_axes={n:{0:'batch'} for n in ['tokens','meta','moves','policy','q','wdl_logits']},opset_version=18,dynamo=False)
    del m;gc.collect();torch.cuda.empty_cache();export=time.perf_counter()-tic;print('exported',export,flush=True)
    logger=trt.Logger(trt.Logger.WARNING);builder=trt.Builder(logger);network=builder.create_network(0);parser=trt.OnnxParser(network,logger)
    if not parser.parse_from_file(str(out/'model.onnx')):raise RuntimeError('\n'.join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    config=builder.create_builder_config();config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE,1<<30)
    if hasattr(trt.BuilderFlag,'TF32'):config.clear_flag(trt.BuilderFlag.TF32)
    profile=builder.create_optimization_profile()
    for name,tail in [('tokens',(64,)),('meta',(12,)),('moves',(256,))]:profile.set_shape(name,(1,*tail),(8,*tail),(16,*tail))
    config.add_optimization_profile(profile);plan=builder.build_serialized_network(network,config)
    if plan is None:raise RuntimeError('TensorRT build failed')
    (out/'model.plan').write_bytes(bytes(plan));(out/'build.json').write_text(json.dumps(dict(tensorrt=trt.__version__,export_seconds=export,total_seconds=time.perf_counter()-tic,precision='preserve exported mixed FP32/BF16 graph; no TF32'),indent=2))
    print('built',time.perf_counter()-tic,flush=True)

if __name__=='__main__':main()
