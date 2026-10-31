"""Short full-training throughput probe using real shards and production losses.

Random model initialization deliberately avoids transferring a 620 MB checkpoint;
this measures execution and numerical stability, never chess strength.
"""
import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys
import threading
import time
import torch
from torch.utils.data import DataLoader
from neurochess.data import Batches, ReplayBatches
from neurochess.model import ChessModel, ModelConfig
from neurochess.train import losses


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',required=True);p.add_argument('--batch-size',type=int,required=True)
    p.add_argument('--accum',type=int,required=True);p.add_argument('--workers',type=int,default=2)
    p.add_argument('--precision',choices=['bf16','fp16'],default='bf16')
    p.add_argument('--compile',action='store_true');p.add_argument('--steps',type=int,default=5)
    p.add_argument('--warmup',type=int,default=2);p.add_argument('--out',required=True)
    p.add_argument('--replay',action='store_true')
    a=p.parse_args();result=dict(vars(a),kind='real-shards-full-optimizer-step',initialization='random seed 42')
    torch.manual_seed(42);torch.set_num_threads(2);torch.set_float32_matmul_precision('high')
    result.update(gpu=torch.cuda.get_device_name(),torch=torch.__version__,cuda=torch.version.cuda,
        gpu_gib=torch.cuda.get_device_properties(0).total_memory/2**30)
    samples=[];stop=threading.Event()
    def monitor():
        while not stop.is_set():
            try:
                row=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,power.draw,clocks.sm,temperature.gpu','--format=csv,noheader,nounits'],text=True)
                samples.append([float(x.strip()) for x in row.split(',')])
            except Exception:pass
            stop.wait(1)
    thread=threading.Thread(target=monitor,daemon=True)
    try:
        model=ChessModel(ModelConfig(history_free=True)).cuda()
        result['parameters']=sum(x.numel() for x in model.parameters())
        runner=torch.compile(model) if a.compile else model
        optimizer=torch.optim.AdamW(model.parameters(),lr=5e-5,weight_decay=.1,fused=True)
        scaler=torch.amp.GradScaler('cuda',enabled=a.precision=='fp16')
        dtype=torch.bfloat16 if a.precision=='bf16' else torch.float16
        epoch=0
        def stream(ep):
            ds=Batches(a.data,'train',a.batch_size,42,ep)
            if a.replay:ds=ReplayBatches(ds,a.data,8)
            return iter(DataLoader(ds,batch_size=None,num_workers=a.workers,pin_memory=True))
        batches=stream(epoch);durations=[];positions=[];loss_values=[];norms=[]
        warmstart=time.perf_counter()
        for step in range(a.warmup+a.steps):
            if step==a.warmup:
                torch.cuda.synchronize();result['warmup_seconds']=time.perf_counter()-warmstart
                torch.cuda.reset_peak_memory_stats();thread.start()
            torch.cuda.synchronize();start=time.perf_counter()
            optimizer.zero_grad(set_to_none=True);seen=0
            for _ in range(a.accum):
                try:batch=next(batches)
                except StopIteration:
                    epoch+=1;batches=stream(epoch);batch=next(batches)
                batch={k:v.cuda(non_blocking=True) for k,v in batch.items()}
                with torch.autocast('cuda',dtype=dtype):
                    loss,_=losses(runner(batch['tokens'],batch['meta'],batch['moves']),batch,.5,1.)
                scaler.scale(loss/a.accum).backward();seen+=len(batch['tokens'])
            scaler.unscale_(optimizer)
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            if not torch.isfinite(norm):raise FloatingPointError('Nonfinite gradient')
            scaler.step(optimizer);scaler.update();torch.cuda.synchronize()
            elapsed=time.perf_counter()-start
            if step>=a.warmup:
                durations.append(elapsed);positions.append(seen);loss_values.append(float(loss));norms.append(float(norm))
        result.update(status='ok',seconds=sum(durations),step_seconds=durations,positions=positions,
            positions_per_second=sum(positions)/sum(durations),median_step_seconds=statistics.median(durations),
            min_step_seconds=min(durations),max_step_seconds=max(durations),loss=loss_values,gradient_norm=norms,
            peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
            peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
    except torch.cuda.OutOfMemoryError:
        result.update(status='oom',peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
            peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
    except Exception as error:
        result.update(status='error',error=repr(error))
    finally:
        stop.set()
        if thread.is_alive():thread.join(timeout=3)
        if samples:
            result['hardware_samples']=samples
            result['hardware_mean']=dict(zip(['gpu_utilization_pct','power_watts','sm_clock_mhz','temperature_c'],[statistics.mean(x[i] for x in samples) for i in range(4)]))
        Path(a.out).parent.mkdir(parents=True,exist_ok=True)
        Path(a.out).write_text(json.dumps(result,indent=2))
        print(json.dumps(result),flush=True)
    if result['status']=='error':sys.exit(1)

if __name__=='__main__':main()
