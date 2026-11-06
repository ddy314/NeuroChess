"""Calibrate on old TRAIN inputs, inject SmoothQuant-style Q/DQ, build candidate."""
import gc,json,re,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'output/puct-10x-20261031/trt-deps'))
import numpy as np,torch,onnx,ml_dtypes,tensorrt as trt
from onnx import helper,numpy_helper
sys.path.append(str(next((ROOT/'.venv/lib').glob('python*/site-packages'))))
from neurochess.train import load_model
from neurochess.fast_inference import resident_linears
r=ROOT/'output/puct-10x-20261031';out=r/'tensorrt-int8';out.mkdir(exist_ok=False);tic=time.perf_counter();torch.set_num_threads(2)
model,state=load_model(r/'frozen-baseline/model.pt','cuda');del state;model=resident_linears(model);amax={};hooks=[]
for name,module in model.named_modules():
 if name.startswith('blocks.') and isinstance(module,torch.nn.Linear):
  def hook(layer,args,name=name):
   x=args[0].detach().to(torch.bfloat16).float().reshape(-1,args[0].shape[-1]);v=x.abs().amax(0)
   amax[name]=torch.maximum(amax[name],v) if name in amax else v
  hooks.append(module.register_forward_pre_hook(hook))
shard=next((ROOT/'data/evals-v2-5m').glob('train-*.npy'));data=np.load(shard,mmap_mode='r');rng=np.random.default_rng(103110);indices=rng.choice(len(data),size=1024,replace=False)
with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
 for start in range(0,len(indices),64):
  rows=data[indices[start:start+64]];model(torch.as_tensor(np.array(rows['tokens']),device='cuda'),torch.as_tensor(np.array(rows['meta']),device='cuda'),torch.as_tensor(np.array(rows['moves']),device='cuda'))
for h in hooks:h.remove()
amax={k:v.cpu().numpy() for k,v in amax.items()};del model;gc.collect();torch.cuda.empty_cache();print('calibrated',time.perf_counter()-tic,flush=True)
m=onnx.load(r/'tensorrt/model.onnx');m.opset_import[0].version=21;weights={w.name:w for w in m.graph.initializer};nodes=[];changed=[]
for node in m.graph.node:
 match=re.fullmatch(r'/blocks\.(\d+)/(qkv|out|ff/ff\.[02])/MatMul',node.name)
 if match is None or node.input[1] not in weights:nodes.append(node);continue
 layer=f"blocks.{match[1]}."+match[2].replace('ff/','');w=weights[node.input[1]];values=numpy_helper.to_array(w).astype(np.float32)
 aw=amax[layer];ww=np.abs(values).max(1);smooth=np.sqrt(np.maximum(aw,1e-6)/np.maximum(ww,1e-6))
 scaled=values*smooth[:,None];w.CopyFrom(numpy_helper.from_array(scaled.astype(ml_dtypes.bfloat16),w.name))
 base=node.name.replace('/','_');sname=base+'_smooth';asname=base+'_ascale';wsname=base+'_wscale';az=base+'_az';wz=base+'_wz'
 arrays={sname:smooth.astype(ml_dtypes.bfloat16),asname:np.array(max(float(np.max(aw/smooth)),1e-6)/127,dtype=ml_dtypes.bfloat16),wsname:(np.abs(scaled).max(0).clip(1e-6)/127).astype(ml_dtypes.bfloat16),az:np.array(0,np.int8),wz:np.zeros(values.shape[1],np.int8)}
 for name,array in arrays.items():m.graph.initializer.append(numpy_helper.from_array(array,name))
 div=base+'_div';aq=base+'_aq';adq=base+'_adq';wq=base+'_wq';wdq=base+'_wdq'
 nodes.extend([helper.make_node('Div',[node.input[0],sname],[div]),helper.make_node('QuantizeLinear',[div,asname,az],[aq]),helper.make_node('DequantizeLinear',[aq,asname,az],[adq]),helper.make_node('QuantizeLinear',[node.input[1],wsname,wz],[wq],axis=1),helper.make_node('DequantizeLinear',[wq,wsname,wz],[wdq],axis=1)])
 node.input[0]=adq;node.input[1]=wdq;nodes.append(node);changed.append(layer)
del m.graph.node[:];m.graph.node.extend(nodes);onnx.checker.check_model(m);onnx.save(m,out/'model.onnx');del m;print('injected',len(changed),flush=True)
logger=trt.Logger(trt.Logger.WARNING);builder=trt.Builder(logger);network=builder.create_network(0);parser=trt.OnnxParser(network,logger)
if not parser.parse_from_file(str(out/'model.onnx')):raise RuntimeError('\n'.join(str(parser.get_error(i)) for i in range(parser.num_errors)))
config=builder.create_builder_config();config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE,1<<30)
if hasattr(trt.BuilderFlag,'TF32'):config.clear_flag(trt.BuilderFlag.TF32)
profile=builder.create_optimization_profile()
for name,tail in [('tokens',(64,)),('meta',(12,)),('moves',(256,))]:profile.set_shape(name,(1,*tail),(8,*tail),(16,*tail))
config.add_optimization_profile(profile);plan=builder.build_serialized_network(network,config)
if plan is None:raise RuntimeError('TensorRT INT8 build failed')
(out/'model.plan').write_bytes(bytes(plan));(out/'build.json').write_text(json.dumps(dict(changed_layers=changed,calibration_shard=str(shard),calibration_indices=indices.tolist(),calibration='TRAIN inputs only, no heldout labels',alpha=.5,total_seconds=time.perf_counter()-tic),indent=2));print('built',time.perf_counter()-tic,flush=True)
