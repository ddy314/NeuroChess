"""Bounded initial L4 sweep; each candidate owns a fresh CUDA process."""
import json,os,subprocess,sys,time
from pathlib import Path
root=Path('/content/neurochess');os.chdir(root)
out=root/'reports/l4';out.mkdir(parents=True,exist_ok=True)
results=[]
for batch in [64,128,256,512,1024]:
    name=f'bf16-b{batch}-a{1024//batch}-w2'
    args=[sys.executable,'scripts/benchmark_training.py','--data','data/bench',
          '--batch-size',str(batch),'--accum',str(1024//batch),'--workers','2','--out',str(out/(name+'.json'))]
    start=time.monotonic()
    try:
        with (out/(name+'.log')).open('w') as log:
            code=subprocess.run(args,stdout=log,stderr=subprocess.STDOUT,timeout=120).returncode
        r=json.loads((out/(name+'.json')).read_text()) if (out/(name+'.json')).exists() else dict(status='failed',code=code)
    except subprocess.TimeoutExpired:r=dict(status='timeout')
    results.append(dict(name=name,elapsed=time.monotonic()-start,**r))
    (out/'sweep.json').write_text(json.dumps(results,indent=2))
    print(json.dumps({k:v for k,v in results[-1].items() if k not in ['hardware_samples','loss','gradient_norm']}),flush=True)
    if r.get('status')=='oom':break
print('SWEEP_COMPLETE',flush=True)
