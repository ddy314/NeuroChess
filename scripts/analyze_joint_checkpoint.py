"""Frozen, paired post-training diagnostics. Does not train or promote weights."""
import argparse,hashlib,importlib.util,json,sqlite3,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
if importlib.util.find_spec('chess') is None:
    sys.path.extend(str(p) for p in (ROOT/'.venv/lib').glob('python*/site-packages'))
import numpy as np
import chess
from neurochess.data import DTYPE


def write(path,data):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2));tmp.replace(path)


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def sample(files,n,rng):
    arrays=[np.load(p,mmap_mode='r') for p in files];ends=np.cumsum([len(a) for a in arrays])
    indices=rng.choice(int(ends[-1]),n,replace=False);result=np.empty(n,dtype=DTYPE)
    for i,a in enumerate(arrays):
        lo=0 if i==0 else ends[i-1];mask=(indices>=lo)&(indices<ends[i]);result[mask]=a[indices[mask]-lo]
    return result


def prepare(out):
    data=ROOT/'output/targeted-scale-local-20261102/dataset';rng=np.random.default_rng(20261103)
    sets=dict(new_val=np.load(data/'val-000000.npy'),new_test=np.load(data/'test-000000.npy'),
              new_train=sample(sorted(data.glob('train-*.npy')),3000,rng),
              mistakes_train=sample(sorted((data/'mistakes').glob('train-*.npy')),1000,rng),
              old_val=np.load(ROOT/'data/evals-v2-5m/val-000000.npy',mmap_mode='r')[:4096].copy(),
              old_train=sample(sorted((ROOT/'data/evals-v2-5m').glob('train-*.npy')),2048,rng))
    db=sqlite3.connect('file:'+str(ROOT/'output/targeted-scale-local-20261102/annotation.sqlite3')+'?mode=ro',uri=True)
    metadata={};keys=sorted({bytes(k).hex() for name,a in sets.items() if name.startswith(('new','mistakes')) for k in a['key']})
    for start in range(0,len(keys),400):
        chunk=keys[start:start+400]
        for key,meta in db.execute('SELECT key,metadata FROM results WHERE key IN ('+','.join('?'*len(chunk))+')',chunk):metadata[key]=json.loads(meta)
    assert set(metadata)==set(keys)
    for name,a in sets.items():np.save(out/(name+'.npy'),a)
    write(out/'metadata.json',metadata)
    audit={}
    for name,a in sets.items():
        legal=(a['moves']>=0).sum(1);annot=a['qmask'].sum(1);target=a['value'][:,0]-a['value'][:,2]
        audit[name]=dict(n=len(a),unique_keys=len({bytes(k) for k in a['key']}),
          legal_moves_mean=float(legal.mean()),annotated_moves_mean=float(annot.mean()),q_label_coverage=float(annot.sum()/legal.sum()),
          target_value_strata={k:int(m.sum()) for k,m in dict(winning=target>.25,neutral=abs(target)<=.25,losing=target<-.25).items()},
          saturated_q_target_fraction=float(((abs(a['q'])>.99)&a['qmask']).sum()/annot.sum()))
        if name.startswith(('new','mistakes')):
            groups=[metadata[bytes(k).hex()]['job']['group'] for k in a['key']]
            audit[name]['source_games']=len(set(groups))
    # Primary held-out search roots are selected before any model prediction.
    test=sets['new_test'];order=rng.permutation(len(test));selected=[];seen=set()
    for label in ['winning','neutral','losing']:
        for index in order:
            key=bytes(test[index]['key']).hex();m=metadata[key];j=m['job'];v=float(test[index]['value'][0]-test[index]['value'][2])
            bucket='winning' if v>.25 else 'losing' if v<-.25 else 'neutral'
            if bucket!=label or j['group'] in seen:continue
            seen.add(j['group']);selected.append(dict(j,selection=label));
            if sum(x['selection']==label for x in selected)==32:break
    write(out/'search-jobs.json',selected)
    write(out/'data-audit.json',dict(datasets=audit,search_roots=len(selected),search_unique_games=len(seen),
        search_selection='Up to32 distinct games per teacher-value stratum, RNG20261103; selected before model outputs',
        old_source_game_ids='Unavailable; old-set uncertainty cannot be clustered by original game',
        expected_exposure=dict(old_general=1280000,new_general=1280000,mistake_oversampling=640000,
                              new_general_equivalent_passes=1280000/472576,mistake_extra_equivalent_passes=640000/15761)))
    db.close()


def load(path):
    import torch
    from neurochess.model import ModelConfig
    from neurochess.policy_value_pair import PolicyValuePair
    s=torch.load(path,map_location='cpu',mmap=True,weights_only=False)
    m=PolicyValuePair(ModelConfig(**s['model_config']));m.load_state_dict(s['model']);m.set_value_training(s.get('value_training_mode','head'))
    step=s['step'];del s
    return m.cuda().eval(),step


def paired_interval(delta,groups,weights=None):
    delta=np.asarray(delta);_,inverse=np.unique(groups,return_inverse=True);n=int(inverse.max()+1)
    weights=np.ones(len(delta)) if weights is None else np.asarray(weights)
    sums=np.bincount(inverse,weights=delta*weights);sizes=np.bincount(inverse,weights=weights)
    rng=np.random.default_rng(20261103);draws=rng.integers(n,size=(2000,n))
    bootstrap=sums[draws].sum(1)/sizes[draws].sum(1)
    return dict(delta_final_minus_parent=float(np.average(delta,weights=weights)),ci95=np.quantile(bootstrap,[.025,.975]).tolist(),clusters=n)


def row_metrics(raw,a):
    from neurochess.multihead import fuse_heads
    logits=raw[:,:256];q=raw[:,256:512];wdl=raw[:,512:]
    lp=logits-logits.max(1,keepdims=True);logp=lp-np.log(np.exp(lp).sum(1,keepdims=True))
    targetp=a['policy'].astype(np.float32);targetp/=np.maximum(targetp.sum(1,keepdims=True),1e-8)
    fused,_,value,support=fuse_heads(logits,q,wdl,a['moves']>=0)
    target=a['value'][:,0]-a['value'][:,2];mask=a['qmask'];counts=mask.sum(1)
    policy_choice=logits.argmax(1);fused_choice=fused.argmax(1);index=np.arange(len(a))
    metrics=dict(policy_ce=-(targetp*logp).sum(1),policy_top1=(policy_choice==targetp.argmax(1)).astype(float),
        q_mse=(((q-a['q'])**2)*mask).sum(1)/counts,q_mae=(abs(q-a['q'])*mask).sum(1)/counts,
        value_ce=-(a['value']*np.log(np.maximum(wdl,1e-30))).sum(1),value_mae=abs(value-target),
        value_bias=value-target,teacher_wdl_brier=((wdl-a['value'])**2).sum(1),
        fused_top1=(fused_choice==targetp.argmax(1)).astype(float),
        policy_choice_labelled=mask[index,policy_choice].astype(float),fused_choice_labelled=mask[index,fused_choice].astype(float),
        q_support_label_coverage=(mask&support).sum(1)/np.maximum(support.sum(1),1),
        q_max_vs_value_gap=abs(np.where(mask,q,-np.inf).max(1)-value))
    return metrics,value,target


def raw_analysis(out,old,new):
    import torch
    from neurochess.multihead import MultiheadPlayer
    torch.set_num_threads(2);torch.set_float32_matmul_precision('high')
    metadata=json.loads((out/'metadata.json').read_text());models=[];steps=[]
    for path in [old,new]:
        model,step=load(path);models.append(model);steps.append(step)
    report=dict(checkpoints=dict(parent=dict(path=str(old),sha256=sha(old),step=steps[0]),final=dict(path=str(new),sha256=sha(new),step=steps[1])),datasets={})
    for name in ['old_val','new_val','new_test','old_train','new_train','mistakes_train']:
        a=np.load(out/(name+'.npy'));outputs=[];by_model=[]
        for label,model in zip(['parent','final'],models):
            net=MultiheadPlayer(model,'cuda');parts=[]
            for start in range(0,len(a),32):
                r=a[start:start+32];parts.append(net.raw_outputs(r['tokens'].copy(),r['meta'].copy(),r['moves'].copy()))
            raw=np.concatenate(parts);np.save(out/f'{name}-{label}-raw.npy',raw);outputs.append(raw)
            metrics,value,target=row_metrics(raw,a);by_model.append(metrics)
        groups=[metadata[bytes(k).hex()]['job']['group'] for k in a['key']] if name.startswith(('new','mistakes')) else [bytes(k).hex() for k in a['key']]
        result=dict(n=len(a),uncertainty_unit='source game' if name.startswith(('new','mistakes')) else 'position; historical game IDs unavailable',metrics={})
        for key in by_model[0]:
            weights=a['qmask'].sum(1) if key=='q_mae' else np.ones(len(a))
            result['metrics'][key]=dict(parent=float(np.average(by_model[0][key],weights=weights)),final=float(np.average(by_model[1][key],weights=weights)),**paired_interval(by_model[1][key]-by_model[0][key],groups,weights))
        result['value_strata']={}
        for stratum,mask in dict(winning=target>.25,neutral=abs(target)<=.25,losing=target<-.25).items():
            result['value_strata'][stratum]=dict(n=int(mask.sum()),mae={l:float(d['value_mae'][mask].mean()) for l,d in zip(['parent','final'],by_model)},bias={l:float(d['value_bias'][mask].mean()) for l,d in zip(['parent','final'],by_model)},**paired_interval((by_model[1]['value_mae']-by_model[0]['value_mae'])[mask],np.asarray(groups)[mask]))
        report['datasets'][name]=result;write(out/'raw-analysis.json',report);print('raw',name,flush=True)
    # Relative tensor movement, not a claim that a small movement implies low skill.
    states=[m.state_dict() for m in models];movement={}
    for name in states[0]:
        a=states[0][name].detach().float();b=states[1][name].detach().float()
        if a.is_floating_point():movement[name]=dict(relative_l2=float((b-a).norm()/a.norm().clamp_min(1e-12)),parent_norm=float(a.norm()),delta_norm=float((b-a).norm()))
    write(out/'weight-movement.json',movement)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','raw']);p.add_argument('--out',required=True);p.add_argument('--old');p.add_argument('--new');a=p.parse_args();out=Path(a.out);out.mkdir(exist_ok=True,parents=True)
    if a.phase=='prepare':prepare(out)
    else:raw_analysis(out,Path(a.old),Path(a.new))
