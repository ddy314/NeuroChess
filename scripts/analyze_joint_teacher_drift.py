"""Audit deeper-reference sensitivity and exact continuation/train overlap."""
import json,sys
from pathlib import Path
import numpy as np
from scripts.summarize_joint_search import board_for
from scripts.analyze_joint_checkpoint import paired_interval,write,ROOT
from neurochess.targets import score_targets


def main(out):
    refs=[json.loads(l) for l in (out/'deep-teacher.jsonl').read_text().splitlines()]
    roots=[m for m in refs if 'depth' not in m['job']]
    metadata=json.loads((out/'metadata.json').read_text());a=np.load(out/'new_test.npy')
    index={bytes(k).hex():i for i,k in enumerate(a['key'])};raw=[np.load(out/f'new_test-{l}-raw.npy') for l in ['parent','final']]
    details=[]
    for m in roots:
        j=m['job'];b=board_for(j);v=max(score_targets(p,b.turn)[1] for p in m['teacher']);i=index[j['key']]
        old=float(a[i]['value'][0]-a[i]['value'][2]);pred=[float(r[i,512]-r[i,514]) for r in raw]
        best=max(metadata[j['key']]['teacher'],key=lambda p:score_targets(p,b.turn)[0])['line'].split()[0]
        matched=next((p for p in m['teacher'] if p['line'].split()[0]==best),None)
        best_cp=max(score_targets(p,b.turn)[0] for p in m['teacher'])
        gap=best_cp-score_targets(matched,b.turn)[0] if matched else None
        details.append(dict(key=j['key'],group=j['group'],stable=m['stable'],refined=m['refined'],old_target=old,deep_target=v,target_shift=abs(v-old),old_best_covered=matched is not None,old_best_regret_cp=gap,parent_error=abs(pred[0]-v),final_error=abs(pred[1]-v)))
    report={'root_details':details,'value_vs_deep_reference':{}}
    for stable in [False,True]:
        rows=[r for r in details if not stable or r['stable']]
        report['value_vs_deep_reference']['stable_only' if stable else 'all']=dict(n=len(rows),parent_mae=float(np.mean([r['parent_error'] for r in rows])),final_mae=float(np.mean([r['final_error'] for r in rows])),mean_target_shift=float(np.mean([r['target_shift'] for r in rows])),target_shift_ge025=sum(r['target_shift']>=.25 for r in rows),**paired_interval([r['final_error']-r['parent_error'] for r in rows],[r['group'] for r in rows]))
    children=[r['job'] for r in refs if 'depth' in r['job']];keys=np.array([bytes.fromhex(j['key']) for j in children],dtype='V16');overlap={}
    paths={'old_train':sorted((ROOT/'data/evals-v2-5m').glob('train-*.npy')),'new_train':sorted((ROOT/'output/targeted-scale-local-20261102/dataset').glob('train-*.npy'))}
    for label,files in paths.items():
        found=set()
        for p in files:
            arr=np.load(p,mmap_mode='r');v=arr['key'].copy().view('V16').reshape(-1)
            found.update(bytes(k).hex() for k in keys[np.isin(keys,v)])
        overlap[label]=dict(count=sum(j['key'] in found for j in children),keys=sorted(found))
    report['continuation_exact_train_overlap']=overlap
    report['continuation_dedup_limit']='Original generator retained one instance per canonical position; exact history preserved and labelled independently for every retained instance. This is a rollout probe, not a dump of actual MCTS leaves.'
    write(out/'teacher-drift-and-overlap.json',report)


if __name__=='__main__':main(Path(sys.argv[1]))
