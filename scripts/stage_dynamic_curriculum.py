"""Stage model-blind ordinary replay separately from the mining reservoir.

This is candidate preparation, not a teacher-ready dataset or a training run.
The full teacher validates student errors; coarse source scores never become
labels. Ordinary replay is selected before consulting model mistakes.
"""
import argparse,collections,hashlib,json,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.run_dynamic_curriculum import write


def select_anchors(jobs,n,max_per_game=12,seed=20261103):
    if n<0 or max_per_game<1:raise ValueError('Invalid replay target/quota')
    if n==0:return set()
    counts=collections.Counter();selected=set()
    for i in np.random.default_rng(seed).permutation(len(jobs)):
        if counts[jobs[i]['group']]>=max_per_game:continue
        counts[jobs[i]['group']]+=1;selected.add(int(i))
        if len(selected)==n:return selected
    raise RuntimeError(f'Insufficient independent ordinary replay under game quota: {len(selected)} < {n}')


def stage(a):
    source=Path(a.source);manifest=json.loads((source/'candidate-manifest.json').read_text());audit=json.loads((source/'full-audit.json').read_text())
    if not manifest['candidate_target_reached'] or not audit['passed'] or audit['counts']!=manifest['candidate_counts']:raise ValueError('Full candidate audit/counts required')
    file=source/'train.jsonl'
    with file.open('rb') as f:actual=hashlib.file_digest(f,'sha256').hexdigest()
    if actual!=manifest['files']['train.jsonl']:raise ValueError('Train candidates changed after hash publication')
    jobs=[json.loads(s) for s in file.read_text().splitlines()]
    anchors=select_anchors(jobs,a.replay_roots,a.max_per_game,a.seed);order=np.random.default_rng(a.seed+1).permutation(len(jobs))
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    with (out/'train-inputs.jsonl').open('w') as f:
        for i in order:f.write(json.dumps(dict(job=jobs[i],prediction=None,replay_anchor=int(i) in anchors,source_priority=10. if int(i) in anchors else 0.))+'\n')
    with (out/'train-inputs.jsonl').open('rb') as f:checksum=hashlib.file_digest(f,'sha256').hexdigest()
    files={'train-inputs.jsonl':checksum}
    for split in ['val','test']:
        dest=out/(split+'-inputs.jsonl');shutil.copyfile(source/(split+'.jsonl'),dest)
        with dest.open('rb') as f:files[dest.name]=hashlib.file_digest(f,'sha256').hexdigest()
        if files[dest.name]!=manifest['files'][split+'.jsonl']:raise ValueError('Heldout candidate copy changed')
    write(out/'plan.json',dict(candidate_roots=len(jobs),model_blind_replay_anchors=len(anchors),source_sha256=actual,input_sha256=checksum,sha256=files,seed=a.seed,max_source_roots_per_game=a.max_per_game,target_verified_policy_value_mistakes=100000,target_is_not_an_actual_label_count=True,teacher_labels_ready=False,training_started=False,automatic_million_expansion=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--out',required=True);p.add_argument('--replay-roots',type=int,default=160000);p.add_argument('--max-per-game',type=int,default=12);p.add_argument('--seed',type=int,default=20261103);stage(p.parse_args())
