"""Refine the measured winner; bound expensive compilation to 180 seconds."""
import json,os,subprocess,sys,time
from pathlib import Path
root=Path('/content/neurochess');os.chdir(root)
out=root/'reports/l4'
results=json.loads((out/'sweep.json').read_text())
def run(batch,workers=2,precision='bf16',compiled=False,steps=5,tag='',replay=False):
    name=f'{precision}-b{batch}-a{1024//batch}-w{workers}'+('-compile' if compiled else '')+tag
    args=[sys.executable,'scripts/benchmark_training.py','--data','data/bench','--batch-size',str(batch),
          '--accum',str(1024//batch),'--workers',str(workers),'--precision',precision,'--steps',str(steps),'--out',str(out/(name+'.json'))]
    if compiled:args+=['--compile']
    if replay:args+=['--replay']
    start=time.monotonic()
    try:
        with (out/(name+'.log')).open('w') as log:
            code=subprocess.run(args,stdout=log,stderr=subprocess.STDOUT,timeout=180 if compiled else 90).returncode
        r=json.loads((out/(name+'.json')).read_text()) if (out/(name+'.json')).exists() else dict(status='failed',code=code)
    except subprocess.TimeoutExpired:r=dict(status='timeout',compile=compiled)
    r=dict(name=name,elapsed=time.monotonic()-start,**r);results.append(r)
    (out/'all.json').write_text(json.dumps(results,indent=2))
    print(json.dumps({k:v for k,v in r.items() if k not in ['hardware_samples','loss','gradient_norm']}),flush=True)
    return r
for batch in [16,32]:run(batch)
valid=[r for r in results if r.get('status')=='ok']
best=max(valid,key=lambda r:r['positions_per_second']);batch=best['batch_size']
r=run(batch,workers=0,tag='-confirm',steps=10)
workers=0 if r.get('status')=='ok' and r['positions_per_second']>best['positions_per_second']*1.03 else 2
run(batch,workers=workers,precision='fp16')
run(batch,workers=workers,compiled=True)
run(batch,workers=workers,steps=10,tag='-replay',replay=True)
print('REFINE_COMPLETE',flush=True)
