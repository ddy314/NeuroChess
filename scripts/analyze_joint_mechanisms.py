"""Inspect sampler exposure, selected-error generalization, and representation drift."""
import argparse,json,importlib.util,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
if importlib.util.find_spec('chess') is None:sys.path.extend(str(p) for p in (ROOT/'.venv/lib').glob('python*/site-packages'))
import numpy as np
from scripts.analyze_joint_checkpoint import row_metrics,paired_interval,write,load


def cpu(out):
    metadata=json.loads((out/'metadata.json').read_text());results={}
    for name in ['new_test','new_train','mistakes_train','old_val']:
        a=np.load(out/(name+'.npy'));raw=[np.load(out/f'{name}-{l}-raw.npy') for l in ['parent','final']]
        metric=[row_metrics(r,a)[0] for r in raw];target=a['value'][:,0]-a['value'][:,2]
        groups=[metadata[bytes(k).hex()]['job']['group'] for k in a['key']] if name!='old_val' else [bytes(k).hex() for k in a['key']]
        # Equal initial-value-error criterion, rather than comparing an error-only
        # training sample to a mostly-easy held-out population.
        mask=metric[0]['value_mae']>=.25
        result=dict(initial_value_error_subset=dict(n=int(mask.sum()),parent_mae=float(metric[0]['value_mae'][mask].mean()),final_mae=float(metric[1]['value_mae'][mask].mean()),still_error_fraction=float((metric[1]['value_mae'][mask]>=.25).mean()),**paired_interval((metric[1]['value_mae']-metric[0]['value_mae'])[mask],np.asarray(groups)[mask])))
        for label,r in zip(['parent','final'],raw):
            value=r[:,512]-r[:,514];slope,intercept=np.polyfit(target,value,1)
            result[label+'_value_mapping']=dict(slope=float(slope),intercept=float(intercept),predicted_abs_mean=float(abs(value).mean()),target_abs_mean=float(abs(target).mean()))
        result['entropy_floor']=dict(policy=float(-(a['policy'].astype(float)*np.log(np.maximum(a['policy'].astype(float),1e-30))).sum(1).mean()),teacher_wdl=float(-(a['value']*np.log(np.maximum(a['value'],1e-30))).sum(1).mean()))
        results[name]=result
    # Exact reconstruction already matched the final checkpoint RNG state.
    arrays=[np.load(p,mmap_mode='r') for p in sorted((ROOT/'output/targeted-scale-local-20261102/dataset').glob('train-*.npy'))]
    keys=np.concatenate([a['key'].copy().view('V16').reshape(-1) for a in arrays]);sort=np.argsort(keys);sorted_keys=keys[sort]
    a=np.load(out/'new_train.npy');samplekeys=a['key'].copy().view('V16').reshape(-1);indices=sort[np.searchsorted(sorted_keys,samplekeys)];assert np.array_equal(keys[indices],samplekeys)
    exposure=np.load(out/'new-exposure-counts.npy')[indices]
    raw=[np.load(out/f'new_train-{l}-raw.npy') for l in ['parent','final']];metric=[row_metrics(r,a)[0] for r in raw]
    results['new_train_seen_vs_unseen']={}
    for label,mask in dict(unseen=exposure==0,seen=exposure>0).items():
        results['new_train_seen_vs_unseen'][label]=dict(n=int(mask.sum()),value_mae=[float(m['value_mae'][mask].mean()) for m in metric],policy_top1=[float(m['policy_top1'][mask].mean()) for m in metric],q_mae=[float(m['q_mae'][mask].mean()) for m in metric])
    write(out/'mechanisms-cpu.json',results)


def linear_cka(a,b):
    a=a-a.mean(0);b=b-b.mean(0);ga=a@a.T;gb=b@b.T
    return float((ga*gb).sum()/np.sqrt((ga*ga).sum()*(gb*gb).sum()))


def gpu(out,old,new):
    import torch
    from neurochess.train import losses
    torch.set_num_threads(2);torch.set_float32_matmul_precision('high');a=np.load(out/'new_test.npy')
    rng=np.random.default_rng(20261103);indices=rng.choice(len(a),256,replace=False);r=a[indices]
    features={};gradients={}
    for label,path in [('parent',old),('final',new)]:
        model,_=load(path);features[label]={}
        for branch in ['policy_model','value_model']:
            network=getattr(model,branch);saved={i:[] for i in [0,7,15]};handles=[]
            for i in saved:
                def hook(module,args,output,i=i):saved[i].append(output.float().mean(1).cpu().numpy())
                handles.append(network.blocks[i].register_forward_hook(hook))
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                for start in range(0,len(r),16):network.encode_squares(torch.as_tensor(r[start:start+16]['tokens'].copy(),device='cuda'),torch.as_tensor(r[start:start+16]['meta'].copy(),device='cuda'))
            for h in handles:h.remove()
            features[label][branch]={str(i):np.concatenate(parts) for i,parts in saved.items()}
        # Value has a separate trunk: measure P/Q conflict only where parameters
        # really are shared, rather than inventing policy/value gradient conflict.
        model.set_value_training('joint');network=model.policy_model
        params=[network.blocks[i].qkv.weight for i in [0,7,15]];cosines=[]
        for start in range(0,64,8):
            batch=r[start:start+8];b={k:torch.as_tensor(batch[k].copy(),device='cuda') for k in ['tokens','meta','moves','policy','q','qmask','value','value_weight']}
            with torch.autocast('cuda',dtype=torch.bfloat16):
                p,q,_=network(b['tokens'],b['meta'],b['moves']);target=b['policy'].float();target/=target.sum(-1,keepdim=True)
                pl=-(target*p.log_softmax(-1)).sum(-1).mean();ql=(((q-b['q'].float())**2)*b['qmask']).sum()/b['qmask'].sum()
            gp=torch.autograd.grad(pl,params,retain_graph=True);gq=torch.autograd.grad(ql,params)
            cosines.append([float(torch.nn.functional.cosine_similarity(x.flatten(),y.flatten(),dim=0)) for x,y in zip(gp,gq)])
        gradients[label]=dict(layers=[0,7,15],cosines=cosines,mean=np.mean(cosines,0).tolist(),negative_fraction=(np.asarray(cosines)<0).mean(0).tolist())
        del model,network;torch.cuda.empty_cache()
    scores={}
    for branch in ['policy_model','value_model']:
        scores[branch]={i:dict(linear_cka=linear_cka(features['parent'][branch][i],features['final'][branch][i]),coordinate_cosine=float(np.mean(np.sum(features['parent'][branch][i]*features['final'][branch][i],axis=1)/(np.linalg.norm(features['parent'][branch][i],axis=1)*np.linalg.norm(features['final'][branch][i],axis=1))))) for i in ['0','7','15']}
    write(out/'representation-and-gradient.json',dict(n_feature_positions=256,source='Fixed random new_test subset; eight batches of8 for gradient diagnostic',features=scores,policy_Q_gradient=gradients,policy_value_shared_trunk=False,interpretation='Output-swap tests are functional interventions; gradient cosine/CKA describe association and do not establish training causality'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['cpu','gpu']);p.add_argument('--out',required=True);p.add_argument('--old');p.add_argument('--new');a=p.parse_args()
    if a.phase=='cpu':cpu(Path(a.out))
    else:gpu(Path(a.out),Path(a.old),Path(a.new))
