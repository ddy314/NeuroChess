import json,os,subprocess,sys,time
from pathlib import Path
root=Path('/content/neurochess');os.chdir(root);out=root/'reports/l4'
results=json.loads((out/'all.json').read_text())
print(subprocess.check_output(['nvidia-smi','--query-gpu=name,temperature.gpu,power.draw,power.limit,clocks.sm,clocks_throttle_reasons.active','--format=csv,noheader'],text=True),flush=True)
for batch,compiled in [(128,False),(64,False),(64,True),(128,True)]:
    name=f'bf16-b{batch}-a{1024//batch}-w2-hot-replay'+('-compile' if compiled else '')
    args=[sys.executable,'scripts/benchmark_training.py','--data','data/bench','--batch-size',str(batch),'--accum',str(1024//batch),
          '--workers','2','--steps','8','--warmup','3','--replay','--out',str(out/(name+'.json'))]
    if compiled:args+=['--compile']
    start=time.monotonic()
    try:
        with (out/(name+'.log')).open('w') as log:
            code=subprocess.run(args,stdout=log,stderr=subprocess.STDOUT,timeout=120).returncode
        r=json.loads((out/(name+'.json')).read_text()) if (out/(name+'.json')).exists() else dict(status='failed',code=code)
    except subprocess.TimeoutExpired:r=dict(status='timeout',compile=compiled)
    r=dict(name=name,elapsed=time.monotonic()-start,**r);results.append(r)
    (out/'all.json').write_text(json.dumps(results,indent=2))
    print(json.dumps({k:v for k,v in r.items() if k not in ['hardware_samples','loss','gradient_norm']}),flush=True)
print('FINAL_COMPARISON_COMPLETE',flush=True)
