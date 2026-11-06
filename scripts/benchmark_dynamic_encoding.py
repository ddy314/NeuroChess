"""Frozen-cohort end-to-end V screening, exact history and output parity."""
import argparse,importlib.util,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
if importlib.util.find_spec('chess') is None:sys.path.extend(str(p) for p in (ROOT/'.venv/lib').glob('python*/site-packages'))
import numpy as np
from neurochess.dynamic_curriculum import encode_jobs_grouped
from scripts.dynamic_student import Student


def main(a):
    jobs=[json.loads(s)['job'] for s in Path(a.inputs).read_text().splitlines()][:a.limit]
    if a.whole_games:
        jobs=[json.loads(s)['job'] for s in Path(a.inputs).read_text().splitlines()]
        jobs.sort(key=lambda j:(j['group'],len(j['history'])));jobs=jobs[:a.limit]
    model=Student(a.model,'99931bbb7f68a812e226364c9636a3725599a54dcc3affb0cc206a48318c4d3a')
    model.values(jobs[:a.batch]);rows=[];reference=None;out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
    for mode in (['full'] if a.include_full else [])+['independent','grouped']:
        grouped=mode=='grouped'
        started=time.monotonic();pred=[];encoding_seconds=None
        if grouped:
            tokens,meta=encode_jobs_grouped(jobs);encoding_seconds=time.monotonic()-started
            for i in range(0,len(jobs),a.batch):pred.extend(model.values_encoded(tokens[i:i+a.batch],meta[i:i+a.batch]))
        elif mode=='full':
            for i in range(0,len(jobs),a.batch):pred.extend(model.jobs(jobs[i:i+a.batch]))
        else:
            for i in range(0,len(jobs),a.batch):pred.extend(model.values(jobs[i:i+a.batch]))
        elapsed=time.monotonic()-started;values=np.array([r['wdl'] for r in pred])
        if reference is None:reference=values
        error=float(np.max(abs(reference-values)));assert error<=1e-6
        selected_seconds=0.
        if grouped and a.include_full:
            t=time.monotonic()
            for i in range(0,min(512,len(jobs)),a.batch):model.jobs(jobs[i:i+a.batch])
            selected_seconds=time.monotonic()-t
        rows.append(dict(mode=mode,grouped=grouped,positions=len(jobs),source_games=len({j['group'] for j in jobs}),seconds=elapsed,selected_full_PQV_seconds=selected_seconds,end_to_end_seconds=elapsed+selected_seconds,encoding_seconds=encoding_seconds,positions_per_second=len(jobs)/(elapsed+selected_seconds),max_wdl_error=error))
        out.write_text(json.dumps(dict(model_sha256=model.sha256,batch=a.batch,results=rows,teacher_calls=0,novel_teacher_labels=0),indent=2));print(json.dumps(rows[-1]),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--inputs',required=True);p.add_argument('--out',required=True);p.add_argument('--model',default='output/joint-deep-analysis-20261103/final-weights.pt');p.add_argument('--batch',type=int,default=256);p.add_argument('--limit',type=int,default=65536);p.add_argument('--whole-games',action='store_true');p.add_argument('--include-full',action='store_true');main(p.parse_args())
