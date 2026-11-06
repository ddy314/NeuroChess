"""Run the authorized step-4000 gradient diagnosis and three 500-step branches."""
import argparse,copy,fcntl,gc,hashlib,json,os,shutil,sys,time,traceback
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch

from neurochess.data import DTYPE
from neurochess.encoding import position_key
from neurochess.epoch_ablation import (VARIANTS,batch,diagnose,mixed_rows,restore_random_states,search_screen,validate_all,variant_lr,
    position_metrics,paired_comparison,paired_q_comparison)
from neurochess.guarded_epoch import ExactEpoch,GUARD,guard_metrics,joint_objective
from neurochess.pawn_data import contains,sha,write_json
from neurochess.train import load_model,atomic_save
from scripts.train_lichess_epoch import choose_rows
from scripts.train_pawn_cloud import ReplayRows


class Experiment:
    def __init__(self,work,out,save):
        self.work=Path(work);self.out=Path(out);self.save=Path(save)
        self.out.mkdir(exist_ok=True);self.save.mkdir(exist_ok=True)
        self.last_sync={};self.last_status=0.;self.started=time.time()
        self.plan=json.loads((self.work/'plan.json').read_text())
        self.certificate=json.loads((self.out/'pause-certificate.json').read_text())
        self.original_baseline=self.certificate['gate']['baseline']
        if json.loads((self.work/'guard-spec.json').read_text())!=GUARD:raise ValueError('Original guard specification differs')
        self.results={}

    def sync(self):
        if not os.path.ismount('/content/drive') or not self.save.is_dir():raise OSError('Persistent Drive mount lost')
        paths=sorted(p for p in self.out.rglob('*') if p.is_file() and not p.name.endswith(('.tmp','.log','.lock')) and p.name not in ['watch.lock'])
        for p in paths:
            rel=p.relative_to(self.out);signature=(p.stat().st_ino,p.stat().st_size,p.stat().st_mtime_ns)
            if self.last_sync.get(str(rel))==signature:continue
            dest=self.save/rel;dest.parent.mkdir(exist_ok=True,parents=True);temp=dest.with_name(dest.name+'.tmp')
            shutil.copyfile(p,temp)
            if sha(p)!=sha(temp):raise OSError('Drive readback differs: '+str(rel))
            temp.replace(dest);self.last_sync[str(rel)]=signature

    def status(self,phase,force=False,**extra):
        now=time.time()
        if not force and now-self.last_status<10:return
        d=dict(phase=phase,updated_at=now,main_step=4000,main_paused=True,main_training_resumes_automatically=False,
            branch_steps=500,branches=self.results,seconds=now-self.started,**extra)
        for path in [self.out/'status.json',self.save/'status.json',self.work/'experiment-status.json']:write_json(path,d)
        labels={'preparing-experiment':'准备对照','gradient-diagnosis':'梯度诊断','branch-training':'训练对照',
            'branch-validation':'验证对照','branch-search':'搜索检查','awaiting-report-copy':'结果已保存，正在回传',
            'complete':'对照已完成，等待用户决定','failed':'对照出现异常'}
        note=f"主训练已停在第 4000 步；{labels.get(phase,phase)}"
        if extra.get('variant'):note+=f" {extra['variant']}，{extra.get('offset',0)}/500 步"
        m=dict(phase='paused',step=4000,
            new_positions=3072000,target_new_positions=9800000,epoch_fraction=3072000/9800000,replay_positions=1024000,
            updated_at=now,default_promoted=False,note=note,experiment_variant=extra.get('variant'))
        write_json(self.work/'status.json',m);write_json(Path(self.plan['save'])/'status.json',m)
        self.last_status=now

    def prepare(self):
        import chess
        if self.certificate['step']!=4000 or not self.certificate['process_stopped']:raise ValueError('Exact paused main checkpoint required')
        p=Path('/proc')/str(self.certificate['pid'])/'stat'
        if not p.exists() or p.read_text().rsplit(')',1)[1].split()[0]!='T':raise RuntimeError('Main process is not paused')
        self.status('preparing-experiment',force=True)
        self.checkpoint=torch.load(self.out/'anchor-last.pt',map_location='cpu',weights_only=False,mmap=True)
        s=self.checkpoint
        if s['epoch_updates']!=4000 or s['new_positions']!=3072000:raise ValueError('Wrong branch start cursor')
        if s['plan_sha256']!=hashlib.sha256(json.dumps(self.plan,sort_keys=True).encode()).hexdigest():raise ValueError('Original plan changed')
        for name,want in self.certificate['checkpoint_sha256'].items():
            if sha(self.out/('anchor-'+name))!=want:raise ValueError('Anchor altered: '+name)
        self.lock=json.loads(Path('configs/search-baseline-lock.json').read_text())
        for name,want in self.lock['source_sha256'].items():
            if not name.endswith('.so') and sha(Path(name))!=want:raise ValueError('Frozen search source changed: '+name)
        if sha(self.work/'parent.pt')!=self.plan['parent_sha256']:raise ValueError('Frozen parent changed')
        source=Path(self.plan['data'])
        if sha(source/'ready.json')!=self.plan['ready_sha256']:raise ValueError('Dataset certificate changed')
        ready=json.loads((source/'ready.json').read_text());self.files={k:[] for k in ['train','val','test']}
        for item in ready['shards']:
            local=self.work/'data'/item['data']['name'].removesuffix('.zst')
            if not local.is_file():raise ValueError('Verified local data missing')
            self.files[item['split']].append(local)
        self.files={k:sorted(v) for k,v in self.files.items()}
        verified=json.loads((self.work/'data-verification.json').read_text())
        if not verified['passed'] or verified['exact_overlap']!=0:raise ValueError('Main data isolation did not pass')
        self.frozen=copy.deepcopy(s['frozen']);reserved=[]
        # Test labels remain sealed; only their position keys are exclusions.
        for k in ['val','test']:reserved.extend(np.load(p,mmap_mode='r')['key'].copy().view('V16').reshape(-1) for p in self.files[k])
        general=np.load(self.work/'general-validation.npy',allow_pickle=False)
        if sha(self.work/'general-validation.npy')!=self.plan['general_validation_sha256']:raise ValueError('General monitor changed')
        old_val=sorted((self.work/'old-core').glob('val-*.npy'))
        reserved.extend(np.load(p,mmap_mode='r')['key'].copy().view('V16').reshape(-1) for p in old_val)
        reserved.append(general['key'].copy().view('V16').reshape(-1))
        reserved.append(np.array([position_key(chess.Board(x['fen'])) for x in self.frozen],dtype='V16'))
        self.excluded=np.unique(np.concatenate(reserved))
        self.replay=ReplayRows(self.work/'old-core',self.excluded,self.plan['seed']+3)
        if self.replay.total!=verified['old_replay']:raise ValueError('Replay population changed')
        self.validation=dict(new=choose_rows(self.files['val'],8192,self.plan['seed']),old_core=choose_rows(old_val,8192,self.plan['seed']+1),old_general=general)
        self.teacher,state=load_model(self.work/'parent.pt','cuda');del state
        self.teacher.eval();self.teacher.requires_grad_(False)
        model=self.new_model()
        self.start_validation=validate_all(model,self.validation,4000)
        if not guard_metrics(self.original_baseline,self.start_validation)['passed']:raise RuntimeError('Revalidated anchor fails original gates')
        write_json(self.out/'start-validation.json',self.start_validation)
        self.status('gradient-diagnosis',force=True)
        result=diagnose(model,self.teacher,s,self.files['train'],self.replay,self.plan,
            lambda n,total:self.status('gradient-diagnosis',batches_done=n,batches_total=total))
        write_json(self.out/'gradient-diagnosis.json',result)
        del model;gc.collect();torch.cuda.empty_cache();self.sync()

    def new_model(self):
        from neurochess.model import ChessModel,ModelConfig
        model=ChessModel(ModelConfig(**self.checkpoint['model_config'])).to('cuda')
        model.load_state_dict(self.checkpoint['model']);model.requires_grad_(True)
        return model

    def payload(self,model,optimizer,variant,offset,input_hash,last_good_offset=0,best_loss=None):
        s=self.checkpoint;step=4000+offset
        experiment_plan=dict(original_plan_sha256=s['plan_sha256'],variant=variant,settings=VARIANTS[variant],
            start_update=4000,target_updates=500,lr_ramp_steps=50)
        return dict(model=model.state_dict(),model_config=asdict(model.config),optimizer=optimizer.state_dict(),
            step=s['origin_step']+step,origin_step=s['origin_step'],epoch_updates=step,new_positions=s['new_positions']+offset*768,
            replay_positions=s['replay_positions']+offset*256,replay_rng=self.replay.rng.bit_generator.state,
            torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),baseline=self.original_baseline,
            frozen=self.frozen,original_plan_sha256=s['plan_sha256'],
            plan_sha256=hashlib.sha256(json.dumps(experiment_plan,sort_keys=True).encode()).hexdigest(),experiment_plan=experiment_plan,
            best_loss=best_loss,last_good_step=4000+last_good_offset,lineage=s['lineage'],
            experiment=dict(variant=variant,**VARIANTS[variant],start_update=4000,updates=offset,lr_ramp_steps=50,
                source_sha256=self.certificate['checkpoint_sha256']['last.pt'],input_sequence_sha256=input_hash,
                original_guard=GUARD,requires_experiment_aware_resume=True),default_promoted=False)

    def branch(self,name):
        folder=self.out/name;folder.mkdir(exist_ok=True)
        if (folder/'result.json').exists():
            result=json.loads((folder/'result.json').read_text())
            if result['source_sha256']!=self.certificate['checkpoint_sha256']['last.pt']:raise ValueError('Old branch has different parent')
            return result
        if (folder/'started.json').exists():raise RuntimeError('Incomplete branch exists; refusing silent replay: '+name)
        write_json(folder/'started.json',dict(variant=name,started_at=time.time(),**VARIANTS[name]))
        model=self.new_model();optimizer=torch.optim.AdamW(model.parameters(),lr=self.plan['lr'],weight_decay=.01,fused=True)
        optimizer.load_state_dict(copy.deepcopy(self.checkpoint['optimizer']))
        restore_random_states(self.checkpoint,self.replay)
        stream=ExactEpoch(self.files['train'],self.plan['seed'],self.checkpoint['new_positions'])
        input_digest=hashlib.sha256();input_prefix={};gates=[];offset=0;completed=0;last_good=0
        total_steps=12761;started=time.time();last_checkpoint=None;search=None;best_new_loss=self.start_validation['new']['loss']
        result=dict(variant=name,settings=VARIANTS[name],source_sha256=self.certificate['checkpoint_sha256']['last.pt'],target_updates=500,
            start_update=4000,initial_validation=self.start_validation,completed_updates=0,status='running',gates=gates)
        try:
            for offset in range(1,501):
                before=4000+offset-1
                rows,old,_,nold,input_hash=mixed_rows(stream,self.replay,self.plan,before,VARIANTS[name]['balance_strength'])
                input_digest.update(bytes.fromhex(input_hash));nt=len(rows);nq=int(rows['qmask'].sum());nv=float(rows['value_weight'].sum())
                optimizer.zero_grad(set_to_none=True);model.train();lr=variant_lr(before+1,total_steps,self.plan['lr'],self.plan['warmup'],name,offset)
                for group in optimizer.param_groups:group['lr']=lr
                sums=np.zeros(4)
                for i in range(0,nt,self.plan['microbatch']):
                    r=rows[i:i+self.plan['microbatch']];b=batch(r);mask=torch.as_tensor(old[i:i+len(r)],device='cuda')
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        with torch.no_grad():reference=self.teacher(b['tokens'][mask],b['meta'][mask],b['moves'][mask]) if mask.any() else None
                        output=model(b['tokens'],b['meta'],b['moves'])
                        loss,parts=joint_objective(output,b,nt,nq,nv,mask,reference,nold)
                    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite objective')
                    loss.backward();sums+=parts.cpu().numpy()
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                if not torch.isfinite(norm):raise FloatingPointError('Nonfinite gradient')
                optimizer.step()
                completed=offset
                if offset%10==0:
                    row=dict(offset=offset,step=before+1,loss=sums.tolist(),lr=lr,gradient_norm=float(norm),timestamp=time.time())
                    with (folder/'metrics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
                    self.status('branch-training',variant=name,offset=offset)
                if offset in [100,250,500]:
                    input_prefix[str(offset)]=input_digest.hexdigest();self.status('branch-validation',force=True,variant=name,offset=offset)
                    current=validate_all(model,self.validation,before+1);gate=guard_metrics(self.original_baseline,current)
                    gate.update(offset=offset,step=before+1,current=current,baseline_reference='Original frozen parent; thresholds unchanged')
                    if gate['passed'] and offset in [100,500]:
                        self.status('branch-search',force=True,variant=name,offset=offset)
                        search=search_screen(model,self.frozen,self.plan['stockfish'],self.lock,before+1,
                            lambda n,total:self.status('branch-search',variant=name,offset=offset,search_done=n,search_total=total))
                        write_json(folder/f'search-{offset:03d}.json',search)
                        gate.update(search_passed=search['passed']);gate['passed']=bool(gate['passed'] and search['passed'])
                    gates.append(gate);write_json(folder/f'gate-{offset:03d}.json',gate)
                    payload=self.payload(model,optimizer,name,offset,input_digest.hexdigest(),last_good,best_new_loss)
                    if not gate['passed']:
                        atomic_save(payload,folder/'rejected.pt');result['status']='guard-stopped';break
                    last_good=offset;payload['last_good_step']=4000+offset
                    if offset in [100,500] and current['new']['loss']<best_new_loss:
                        best_new_loss=current['new']['loss']
                        payload['best_loss']=best_new_loss
                        atomic_save({k:payload[k] for k in ['model','model_config','step','origin_step','epoch_updates','new_positions','lineage','experiment','default_promoted']},folder/'best.pt')
                    atomic_save(payload,folder/'last.pt');last_checkpoint=offset
                    self.sync()
            else:result['status']='complete'
            result.update(completed_updates=offset,last_good_offset=last_good,last_checkpoint_offset=last_checkpoint,
                end_update=4000+offset,additional_new_positions=offset*768,additional_replay_positions=offset*256,
                input_sequence_sha256=input_digest.hexdigest(),input_prefix_sha256=input_prefix,
                final_validation=gates[-1]['current'] if gates else None,all_gates_passed=all(x['passed'] for x in gates),
                final_search={k:v for k,v in search.items() if k!='rows'} if search else None,seconds=time.time()-started,
                parameter_changes=self.parameter_changes(model),original_main_checkpoint_modified=False)
            if result['status']=='complete':
                for split,rows in self.validation.items():np.savez_compressed(folder/f'positions-{split}.npz',**position_metrics(model,rows))
        except BaseException:
            result.update(status='failed',completed_updates=completed,traceback=traceback.format_exc(),seconds=time.time()-started)
            try:
                emergency=self.payload(model,optimizer,name,completed,input_digest.hexdigest(),last_good,best_new_loss)
                emergency.update(unvalidated_candidate=True,automatic_resume_eligible=False,rng_may_include_failed_batch=True)
                atomic_save(emergency,folder/'interrupted.pt')
            except BaseException:result['emergency_failure']=traceback.format_exc()
        finally:
            del model,optimizer;gc.collect();torch.cuda.empty_cache()
        write_json(folder/'result.json',result);self.results[name]={k:result[k] for k in ['variant','status','completed_updates','all_gates_passed'] if k in result}
        self.sync();return result

    def parameter_changes(self,model):
        groups={k:[0.,0.] for k in ['shared','policy_Q_projection','policy','Q','value']}
        with torch.no_grad():
            for name,p in model.named_parameters():
                original=self.checkpoint['model'][name].to(p.device)
                if name=='action.1.weight' or name=='action.1.bias':
                    for i,k in enumerate(['policy','Q']):
                        groups[k][0]+=float((p[i]-original[i]).square().sum());groups[k][1]+=float(original[i].square().sum())
                else:
                    key='value' if name.startswith('value.') else 'policy_Q_projection' if name.startswith(('source.','destination.','promotion.')) else 'shared'
                    groups[key][0]+=float((p-original).square().sum());groups[key][1]+=float(original.square().sum())
        return {k:dict(delta_l2=v[0]**.5,relative_delta_l2=(v[0]/max(v[1],1e-30))**.5) for k,v in groups.items()}

    def run(self):
        self.prepare();results={name:self.branch(name) for name in VARIANTS}
        shared={}
        for offset in ['100','250','500']:
            hashes={r.get('input_prefix_sha256',{}).get(offset) for r in results.values() if r.get('input_prefix_sha256',{}).get(offset)}
            shared[offset]=len(hashes)<=1
        if not all(shared.values()):raise RuntimeError('A/B/C consumed different data sequences')
        for name,want in self.certificate['checkpoint_sha256'].items():
            if sha(self.work/name)!=want:raise RuntimeError('Paused main checkpoint changed')
        paired={}
        if results['A']['status']=='complete':
            for name in ['B','C']:
                if results[name]['status']!='complete':continue
                paired[name]={}
                for split in self.validation:
                    with np.load(self.out/'A'/f'positions-{split}.npz') as a,np.load(self.out/name/f'positions-{split}.npz') as b:
                        if not np.array_equal(a['keys'],b['keys']) or not np.array_equal(a['q_count'],b['q_count']):raise ValueError('Paired evaluation populations differ')
                        paired[name][split]={metric:paired_comparison(a[metric],b[metric]) for metric in ['top1','policy_ce','value_error']}
                        paired[name][split]['Q_MAE']=paired_q_comparison(a['q_abs_sum'],b['q_abs_sum'],a['q_count'])
                        paired[name][split]['top1_counts']=dict(improved=int(((a['top1']==0)&(b['top1']==1)).sum()),regressed=int(((a['top1']==1)&(b['top1']==0)).sum()))
        report=dict(main_checkpoint_step=4000,main_new_positions=3072000,original_guard=GUARD,branches=results,paired_vs_A=paired,
            matched_data_sequences=shared,original_main_checkpoint_sha256=self.certificate['checkpoint_sha256'],
            gradient_diagnosis='gradient-diagnosis.json',finished_at=time.time(),main_training_resumed=False,
            selected_variant=None,automatic_promotion=False,test_labels_evaluated=False,completed_epoch=False,
            scope='500-step development comparisons; all three heads trained; original frozen-parent metric/search thresholds; guard failures stop individual branches early; no automatic selection or extra epoch')
        write_json(self.out/'results.json',report)
        text=['# Step-4000 controlled comparison','', 'Main training remains paused at update 4000. No variant selected.','']
        for name,r in results.items():
            text.append(f"- {name}: {r['status']}, {r['completed_updates']}/500 updates, gates passed: {r.get('all_gates_passed',False)}")
        text+=['','Results are development measurements; full heldout test labels were not used.']
        (self.out/'REPORT.md').write_text('\n'.join(text)+'\n');self.sync()
        files={str(p.relative_to(self.out)):sha(p) for p in self.out.rglob('*') if p.is_file() and not p.name.endswith(('.log','.tmp','.lock')) and p.name not in ['status.json','watch-status.json']}
        for name,want in files.items():
            if sha(self.save/name)!=want:raise OSError('Final persistent file differs: '+name)
        completion=dict(completed=True,all_three_completed_500=all(r['status']=='complete' for r in results.values()),
            files_sha256=files,report_sha256=sha(self.out/'results.json'),finished_at=time.time(),no_auto_continue=True)
        write_json(self.out/'completion.json',completion);self.sync();self.status('awaiting-report-copy',force=True)
        # Give the authenticated local collector time to fetch a report before
        # releasing. Drive remains the durable fallback if the PC is offline.
        deadline=time.time()+180
        while time.time()<deadline:
            ack=self.out/'local-report-ack.json'
            if ack.exists() and json.loads(ack.read_text()).get('report_sha256')==completion['report_sha256']:break
            time.sleep(2)
        self.status('complete',force=True)
        for name in ['results.json','completion.json','gradient-diagnosis.json']:
            if sha(self.out/name)!=sha(self.save/name):raise OSError('Pre-release result verification failed')
        from neurochess.autostop import find_runtime_address,request_unassign
        write_json(self.save/'shutdown-request.json',dict(reason='Authorized experiment completed and persisted; main training paused; awaiting user decision',requested_at=time.time()))
        request_unassign(find_runtime_address())


def main():
    p=argparse.ArgumentParser();p.add_argument('--main-work',required=True);p.add_argument('--out',required=True);p.add_argument('--save',required=True);a=p.parse_args()
    torch.set_num_threads(2);torch.set_float32_matmul_precision('high')
    if not torch.cuda.is_available() or 'L4' not in torch.cuda.get_device_name():raise RuntimeError('Authorized same L4 required')
    lock=(Path(a.out)/'experiment.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    experiment=Experiment(a.main_work,a.out,a.save)
    try:experiment.run()
    except BaseException:
        error=dict(traceback=traceback.format_exc(),updated_at=time.time(),main_resumed=False)
        write_json(Path(a.out)/'failure.json',error)
        try:experiment.sync();experiment.status('failed',force=True,error=error['traceback'])
        except BaseException:print(traceback.format_exc(),flush=True)
        raise


if __name__=='__main__':main()
