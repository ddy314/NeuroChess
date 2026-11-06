"""Freeze unseen local roots; compare uniform versus source-prioritized mining."""
import argparse,json,sqlite3,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import importlib.util
if importlib.util.find_spec('chess') is None:sys.path.extend(str(p) for p in (ROOT/'.venv/lib').glob('python*/site-packages'))
import numpy as np
import chess
from scripts.dynamic_student import Student
from neurochess.targets import score_targets


def main(args):
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    db=sqlite3.connect(f'file:{Path(args.source).resolve()}?mode=ro',uri=True)
    query="SELECT job FROM jobs WHERE split='train' ORDER BY key LIMIT ?" if args.source_kind=='dynamic-reservoir' else "SELECT j.job FROM jobs j LEFT JOIN results r ON j.key=r.key WHERE r.key IS NULL AND j.quarantine=0 AND j.split='train' AND NOT EXISTS(SELECT 1 FROM exclusions e WHERE e.key=j.key) ORDER BY j.key LIMIT ?"
    jobs=[json.loads(r[0]) for r in db.execute(query,(args.roots,))];db.close()
    model=Student(args.model,'99931bbb7f68a812e226364c9636a3725599a54dcc3affb0cc206a48318c4d3a');pred=[];started=time.monotonic()
    if args.value_first and not args.independent_encoding:
        from neurochess.dynamic_curriculum import encode_jobs_grouped
        tokens,meta=encode_jobs_grouped(jobs)
        for i in range(0,len(jobs),args.batch):pred.extend(model.values_encoded(tokens[i:i+args.batch],meta[i:i+args.batch]))
    else:
        for i in range(0,len(jobs),args.batch):pred.extend((model.values if args.value_first else model.jobs)(jobs[i:i+args.batch]))
    elapsed=time.monotonic()-started
    values=[]
    for j,p in zip(jobs,pred):
        source=j.get('source_eval_white');turn=(len(j['history'])%2==0)
        score=abs(score_targets(source,turn)[1]-p['value']) if source else -1.
        values.append(score)
    rng=np.random.default_rng(20261103);uniform=[];groups=set()
    for i in rng.permutation(len(jobs)):
        if jobs[i]['group'] in groups:continue
        uniform.append(int(i));groups.add(jobs[i]['group'])
        if len(uniform)==args.sample:break
    priority=[]
    for i in sorted(range(len(jobs)),key=lambda i:-values[i]):
        if jobs[i]['group'] in groups or values[i]<0:continue
        priority.append(i);groups.add(jobs[i]['group'])
        if len(priority)==args.sample:break
    assert len(uniform)==len(priority)==args.sample
    selected_seconds=0.
    if args.value_first:
        selected=uniform+priority;selected_started=time.monotonic()
        for i in range(0,len(selected),args.batch):
            chunk=selected[i:i+args.batch];full=model.jobs([jobs[k] for k in chunk])
            for k,p in zip(chunk,full):
                if abs(p['value']-pred[k]['value'])>1e-6:raise ValueError('Value-first path differs from full student')
                pred[k]=p
        selected_seconds=time.monotonic()-selected_started
    for label,indices in [('uniform',uniform),('priority',priority)]:
        (out/f'{label}-tasks.json').write_text(json.dumps([(jobs[i],pred[i]) for i in indices]))
    uniform_set=set(uniform)
    (out/'root-predictions.jsonl').write_text(''.join(json.dumps(dict(job=j,prediction=None if p.get('value_only') else p,value_prediction=p if p.get('value_only') else None,replay_anchor=i in uniform_set,source_priority=10. if i in uniform_set else s))+'\n' for i,(j,p,s) in enumerate(zip(jobs,pred,values))))
    (out/'preparation.json').write_text(json.dumps(dict(roots=len(jobs),unique_games=len({j['group'] for j in jobs}),batch=args.batch,seconds=elapsed,selected_full_prediction_seconds=selected_seconds,value_first=args.value_first,end_to_end_screened_roots_per_second=len(jobs)/(elapsed+selected_seconds),model_sha256=model.sha256,protocol='New never-labelled train roots from audited local reservoir; one distinct source game per teacher-pilot root; source eval only prioritizes, never supplies accepted targets; includes full-history encoding and GPU inference; value-first, if selected, equals full-model V and computes full P/Q/V only for submitted candidates',uniform_proxy_mean=float(np.mean([values[i] for i in uniform])),priority_proxy_mean=float(np.mean([values[i] for i in priority]))),indent=2))
    print((out/'preparation.json').read_text(),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',default='output/targeted-scale-local-20261102/annotation.sqlite3');p.add_argument('--source-kind',choices=['unused-legacy','dynamic-reservoir'],default='unused-legacy');p.add_argument('--model',default='output/joint-deep-analysis-20261103/final-weights.pt');p.add_argument('--out',required=True);p.add_argument('--roots',type=int,default=8192);p.add_argument('--sample',type=int,default=256);p.add_argument('--batch',type=int,default=128);p.add_argument('--value-first',action='store_true');p.add_argument('--independent-encoding',action='store_true');main(p.parse_args())
