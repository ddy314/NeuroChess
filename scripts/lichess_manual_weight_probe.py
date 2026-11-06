"""Bounded, reversible weight interpolation candidates. No optimizer/training.

Each candidate is stored separately, never replacing any original checkpoint.
The five failures are regression checks, not independent selection evidence.
"""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.lichess_failure_mechanisms import *
from neurochess.epoch_ablation import validate
from neurochess.guarded_epoch import guard_metrics
from neurochess.model import ChessModel,ModelConfig
from neurochess.multihead import MultiheadPlayer
BASE=ROOT/'output/lichess-joint-epoch1-20261104'
DEST=OUT/'manual-weight-probes'

def interpolate_state(parent,current,alpha):
    if not 0<=alpha<=1 or parent.keys()!=current.keys():raise ValueError('Invalid interpolation or mismatched state')
    result={}
    for k,a in parent.items():
        b=current[k]
        if a.shape!=b.shape or a.dtype!=b.dtype:raise ValueError('Mismatched tensor '+k)
        if not a.is_floating_point():
            if not torch.equal(a,b):raise ValueError('Different nonfloat state')
            result[k]=a.clone();continue
        x=torch.lerp(a,b,alpha)
        if not torch.isfinite(x).all():raise ValueError('Nonfinite edited tensor')
        result[k]=x
    return result

def choose(files,size,seed):
    arrays=[np.load(p,mmap_mode='r') for p in sorted(files)];ends=np.cumsum([len(a) for a in arrays])
    picks=np.sort(np.random.default_rng(seed).choice(int(ends[-1]),size,replace=False))
    return np.concatenate([a[picks[(picks>=lo)&(picks<hi)]-lo] for a,lo,hi in zip(arrays,np.r_[0,ends[:-1]],ends)])

def main():
    deadline=time.time()+7200
    while time.time()<deadline:
        for name in ['failure.json','matched-budget-failure.json']:
            if (OUT/name).exists():raise RuntimeError('Upstream diagnostic failed, refusing edits')
        if (OUT/'completion.json').exists() and (OUT/'matched-budget/status.json').exists():
            if read(OUT/'matched-budget/status.json')['phase']=='search-complete' and not Path('/proc/'+str(read(OUT/'status.json')['pid'])).exists():break
        time.sleep(5)
    else:raise TimeoutError('Waiting for diagnostic GPU work')
    DEST.mkdir(exist_ok=True);lock,cert=verify_inputs()
    torch.set_num_threads(2);torch.set_float32_matmul_precision('high')
    # Fixed three-point bounded family. No per-case edits or gradient updates.
    plan=dict(formula='theta(alpha)=lerp(frozen_parent,step4000,alpha)',alphas=[.25,.5,.75],
        source_checkpoint_sha256=cert['checkpoint_sha256'],scope='Weights-only prototypes; no promotion or resume. No full512 candidate qualification.',
        candidate_selection='Report all three, reject any original metric gate or any five-root regression failure. Fresh4096 validation is diagnostic, not final test.')
    write(DEST/'plan.json',plan)
    original_plan=read(BASE/'plan-r2.json');files=list((ROOT/'data/lichess-clean-10m').glob('val-*.npy'))
    monitor=choose(files,8192,original_plan['seed']);old=choose(list((ROOT/'data/evals-v2-5m').glob('val-*.npy')),8192,original_plan['seed']+1)
    general_path=Path('/home/xia/Downloads/general-validation.npy')
    if sha(general_path)!=original_plan['general_validation_sha256']:raise ValueError('General validation hash mismatch')
    general=np.load(general_path)
    prior=np.load(ROOT/'output/lichess-next-step-review-20261105/weights-new_disjoint-parent_trunk__parent_heads.npz')['keys']
    excluded=np.unique(np.concatenate([a.copy().view('V16').ravel() for a in [monitor['key'],prior,np.load(AUDIT/'cohort.npy')['key']]]))
    allval=np.concatenate([np.load(p,mmap_mode='r') for p in sorted(files)]);allval=allval[~np.isin(allval['key'].copy().view('V16').ravel(),excluded)]
    fresh=allval[np.random.default_rng(202611057).choice(len(allval),4096,replace=False)];np.save(DEST/'fresh-validation.npy',fresh)
    write(DEST/'fresh-validation-selection.json',dict(n=4096,keys_sha256=hashlib.sha256(fresh['key'].tobytes()).hexdigest(),
        disjoint_from_monitor_diagnostic_audit=True,selection_before_any_candidate_evaluation=True,test_labels_used=False))
    validation=dict(new=monitor,old_core=old,old_general=general)
    oldstate=torch.load(PARENT,map_location='cpu',weights_only=False,mmap=True)
    newstate=torch.load(CANDIDATE,map_location='cpu',weights_only=False,mmap=True)
    model=ChessModel(ModelConfig(**oldstate['model_config']));model.load_state_dict(oldstate['model']);model.eval().requires_grad_(False).cuda()
    local_base={k:validate(model,a,microbatch=32) for k,a in validation.items()};original_base=cert['gate']['baseline']
    write(DEST/'frozen-local-validation.json',local_base)
    jobs=read(OUT/'jobs.json');results=[]
    for alpha in plan['alphas']:
        slug=f'alpha-{alpha:.2f}';path=DEST/(slug+'.pt')
        edited=interpolate_state(oldstate['model'],newstate['model'],alpha)
        payload=dict(model=edited,model_config=newstate['model_config'],manual_edit=dict(alpha=alpha,formula=plan['formula'],
            source_epoch_updates=4000,source_checkpoint_sha256=cert['checkpoint_sha256']),weights_only=True,default_promoted=False,requires_explicit_training_plan=True)
        tmp=path.with_suffix('.tmp');torch.save(payload,tmp);tmp.replace(path)
        model.load_state_dict(edited);del edited,payload
        network=MultiheadPlayer(model,'cuda');t,m,ids,_=pack(chess.Board(jobs[0]['fen']))
        for n in [1,64]:network.raw_outputs(np.repeat(t,n,0),np.repeat(m,n,0),np.repeat(ids,n,0))
        searches=[]
        for j in jobs:
            player=NativeMCTSPlayer(network,**lock['config'])
            try:move=player.choose(chess.Board(j['fen']));root=player.last_result
            finally:player.close()
            ref=read(OUT/'teachers'/f"{j['index']:03d}.json")['final'];scores={a['move']:a['sort_score'] for a in ref}
            teacher_extra=None
            if move.uci() not in scores:
                engine=FastTeacher(ENGINE,threads=1,hash_mb=64,timeout=300)
                try:teacher_extra=scored_infos(engine,chess.Board(j['fen']),10000000,list(scores)+[move.uci()])
                finally:engine.quit()
                scores={a['move']:a['sort_score'] for a in teacher_extra}
            regret=max(scores.values())-scores[move.uci()]
            parent_move=j['search']['parent']['move'];delta=scores[parent_move]-scores[move.uci()]
            searches.append(dict(index=j['index'],group=j['diagnostic_group'],move=move.uci(),regret=regret,parent_delta=delta,
                introduced_blunder=regret>=100 and delta>=75,root=root,extra_teacher=teacher_extra))
        current={k:validate(model,a,microbatch=32) for k,a in validation.items()};fresh_metrics=validate(model,fresh,microbatch=32)
        r=dict(alpha=alpha,path=str(path),sha256=sha(path),metric_gate_original=guard_metrics(original_base,current),
            metric_gate_local_parent=guard_metrics(local_base,current),validation=current,fresh_validation=fresh_metrics,searches=searches,
            five_failure_regressions=[s['index'] for s in searches if s['group']=='failure' and s['introduced_blunder']],
            retained_control_regressions=[s['index'] for s in searches if s['group']=='retained_control' and s['introduced_blunder']],
            original_search_gate_scope='Ten selected diagnostic roots only, full512 qualification not run',optimizer_steps=0,default_promoted=False)
        write(DEST/(slug+'.json'),r);results.append(r);write(DEST/'status.json',dict(phase='probing',completed=len(results),total=3,training_resumed=False))
    verify_inputs();write(DEST/'results.json',dict(completed=True,candidates=results,automatic_selection=False,training_resumed=False,optimizer_steps=0))
    write(DEST/'completion.json',dict(completed=True,report_sha256=sha(DEST/'results.json'),files_sha256={str(p.relative_to(DEST)):sha(p) for p in DEST.rglob('*') if p.is_file() and p.suffix not in ['.tmp','.log'] and p.name not in ['completion.json','status.json']},training_resumed=False))
    write(DEST/'status.json',dict(phase='complete',completed=3,training_resumed=False))

if __name__=='__main__':
    try:main()
    except BaseException:
        write(OUT/'manual-probe-failure.json',dict(traceback=traceback.format_exc(),training_resumed=False));raise
