"""Local, conditional D experiment from the immutable update-4000 checkpoint.

Uses only existing environments; no installation, cloud allocation or promotion.
"""
import torch  # Resolve GPU torch before appending already installed dependencies.
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.append('/home/xia/code/llmlearn/.venv/lib/python3.13/site-packages')
import argparse, copy, fcntl, gc, hashlib, json, os, shutil, signal, time, traceback
from collections import defaultdict
from dataclasses import asdict
import numpy as np
import chess
from neurochess.value_kl_diagnostic import NAMES, RULE, split_terms, gram, geometry, adam_counterfactual, decision
from neurochess.guarded_epoch import ExactEpoch, GUARD, guard_metrics, learning_rate
from neurochess.epoch_ablation import (batch, mixed_rows, restore_random_states, validate, position_metrics,
    paired_comparison, paired_q_comparison, screen_search_rows)
from neurochess.encoding import position_key
from neurochess.pawn_data import sha, write_json, load_published, contains
from neurochess.train import load_model
from scripts.train_lichess_epoch import choose_rows
from scripts.train_pawn_cloud import ReplayRows
import scripts.lichess_epoch_ablation as original

BASE = ROOT/'output/lichess-joint-epoch1-20261104'
PARENT = ROOT/'output/puct-10x-20261031/frozen-baseline/model.pt'
STOCKFISH = ROOT/'local/stockfish/stockfish/stockfish-ubuntu-x86-64-avx2'


def local_validate(model, validation, step):
    return {k:dict(step=step, split=k, **validate(model, rows, microbatch=32)) for k,rows in validation.items()}


def d_objective(out, b, nt, nq, nv, mask, reference, nold):
    t=split_terms(out,b,nt,nq,nv,mask,reference,nold,value_kl=2.)
    parts=[t['new_policy']+t['old_policy'], 2*(t['new_Q']+t['old_Q']),
           t['new_value']+t['old_value'],sum(t[n] for n in NAMES[6:])]
    return sum(t.values()),torch.stack([p.detach() for p in parts])


class LocalExperiment(original.Experiment):
    def __init__(self, args):
        self.args=args; self.out=Path(args.out).resolve(); self.out.mkdir(parents=True,exist_ok=True)
        self.started=time.time(); self.last_status=0.;self.results={};self.last_sync={}
        self.plan=json.loads((BASE/'plan-r2.json').read_text())
        self.original_plan=copy.deepcopy(self.plan)
        self.certificate=json.loads((BASE/'auto4000/results/pause-certificate.json').read_text())
        self.original_baseline=self.certificate['gate']['baseline']
        self.plan['microbatch']=32; self.plan['stockfish']=str(STOCKFISH)
        self.experiment_plan=dict(variant='D',diagnostic_batches=16,diagnostic_rule=RULE,
            value_kl_weight=2.,value_kl_schedule='constant 2 from first D update',
            effective_batch=1024,new_per_batch=768,replay_per_batch=256,
            diagnostic_and_training_microbatch=32,validation_microbatch=32,
            lr_schedule='Original cosine schedule, no multiplier',balance_strength=1.,
            target_updates=500,metric_gates=[100,250,500],search_gates=[100,500],original_guard=GUARD,
            original_checkpoint_sha256=self.certificate['checkpoint_sha256'],
            hardware='Local existing GPU; original frozen weights/configuration unchanged',
            limitations=['Different GPU, torch and microbatch from L4 A/B/C; not a strictly same-runtime paired training trial',
                '16 batches and development gates cannot prove causality, Elo, or generalization'],
            original_test_labels_used=False,main_training_resumes_automatically=False,automatic_promotion=False)
        write_json(self.out/'experiment-plan.json',self.experiment_plan)

    def status(self,phase,force=False,**extra):
        if not force and time.time()-self.last_status<10:return
        self.last_status=time.time()
        d=dict(phase=phase,pid=os.getpid(),updated_at=time.time(),seconds=time.time()-self.started,
            main_step=4000,main_paused=True,main_training_resumes_automatically=False,default_promoted=False,
            branches=self.results,**extra)
        write_json(self.out/'status.json',d);print(json.dumps(d),flush=True)

    def sync(self):
        # Atomic writes on local durable storage; cloud mount/allocation forbidden.
        pass

    def prepare(self):
        self.status('verifying-inputs',force=True)
        s_path=Path(self.args.checkpoint)
        if sha(s_path)!=self.certificate['checkpoint_sha256']['last.pt']:raise ValueError('Wrong full checkpoint SHA')
        if sha(Path(self.args.best))!=self.certificate['checkpoint_sha256']['best.pt']:raise ValueError('Wrong best checkpoint SHA')
        if sha(PARENT)!=self.plan['parent_sha256']:raise ValueError('Frozen parent changed')
        self.lock=json.loads((ROOT/'configs/search-baseline-lock.json').read_text())
        for name,want in self.lock['source_sha256'].items():
            if sha(ROOT/name)!=want:raise ValueError('Frozen search source changed: '+name)
        s=torch.load(s_path,map_location='cpu',weights_only=False,mmap=True);self.checkpoint=s
        if s['epoch_updates']!=4000 or s['new_positions']!=3072000 or s['replay_positions']!=1024000:raise ValueError('Wrong checkpoint cursor')
        if s['plan_sha256']!=hashlib.sha256(json.dumps(self.original_plan,sort_keys=True).encode()).hexdigest():raise ValueError('Original plan differs')
        for k in ['optimizer','torch_rng','cuda_rng','replay_rng','frozen']:
            if k not in s:raise ValueError('Full resume state missing: '+k)
        self.frozen=copy.deepcopy(s['frozen'])
        source=Path(self.args.data)
        if sha(source/'ready.json')!=self.plan['ready_sha256'] or sha(source/'completion.json')!=self.plan['audit_sha256']:raise ValueError('Dataset certificate differs')
        ready=json.loads((source/'ready.json').read_text());audit=json.loads((source/'completion.json').read_text())
        if not ready['complete'] or not audit['passed'] or ready['counts']!=dict(train=9800000,val=100000,test=100000):raise ValueError('Incomplete dataset audit')
        cache=Path(self.args.cache);cache.mkdir(parents=True,exist_ok=True);self.files={s:[] for s in ['train','val','test']}
        for i,item in enumerate(ready['shards']):
            a=load_published(item['data'],source,cache);del a
            self.files[item['split']].append(cache/item['data']['name'].removesuffix('.zst'))
            self.status('verifying-data',prepared_shards=i+1,total_shards=len(ready['shards']))
        self.files={k:sorted(v) for k,v in self.files.items()}
        oldroot=ROOT/'data/evals-v2-5m';checksums=json.loads((oldroot/'checksums.json').read_text())
        actual={p.name for p in oldroot.glob('*.npy')}
        expected={n for n in checksums if n.endswith('.npy')}
        if actual!=expected:raise ValueError('Old replay file set differs')
        for i,(name,want) in enumerate(checksums.items()):
            if sha(oldroot/name)!=want:raise ValueError('Old replay hash differs: '+name)
            self.status('verifying-replay',files_done=i+1,total_files=len(checksums))
        general_path=Path(self.args.general)
        if sha(general_path)!=self.plan['general_validation_sha256']:raise ValueError('General validation hash differs')
        general=np.load(general_path,allow_pickle=False);old_val=sorted(oldroot.glob('val-*.npy'))
        reserved=[]
        # Only test position keys are read, never test targets for diagnosis/gates.
        for k in ['val','test']:reserved.extend(np.load(p,mmap_mode='r')['key'].copy().view('V16').reshape(-1) for p in self.files[k])
        reserved.extend(np.load(p,mmap_mode='r')['key'].copy().view('V16').reshape(-1) for p in old_val)
        reserved.append(general['key'].copy().view('V16').reshape(-1))
        reserved.append(np.array([position_key(chess.Board(x['fen'])) for x in self.frozen],dtype='V16'))
        self.excluded=np.unique(np.concatenate(reserved));del reserved
        self.replay=ReplayRows(oldroot,self.excluded,self.plan['seed']+3)
        old_verified=json.loads((BASE/'progress-r2-20261104/data-verification.json').read_text())
        if not old_verified['passed'] or self.replay.total!=old_verified['old_replay']:raise ValueError('Replay population differs from original')
        total=0
        for p in self.files['train']:
            rows=np.load(p,mmap_mode='r');total+=len(rows)
            if contains(self.excluded,rows['key'].copy().view('V16').reshape(-1)).any():raise ValueError('Heldout overlap')
        if total!=9800000:raise ValueError('Training population differs')
        write_json(self.out/'data-verification.json',dict(passed=True,new_train=total,old_replay=self.replay.total,exact_overlap=0,
            ready_sha256=sha(source/'ready.json'),old_checksums_sha256=sha(oldroot/'checksums.json'),test_labels_used=False))
        self.validation=dict(new=choose_rows(self.files['val'],8192,self.plan['seed']),
            old_core=choose_rows(old_val,8192,self.plan['seed']+1),old_general=general)
        self.teacher,ts=load_model(PARENT,'cuda');del ts
        self.teacher.eval();self.teacher.requires_grad_(False)
        model=self.new_model();self.status('anchor-validation',force=True)
        self.start_validation=local_validate(model,self.validation,4000)
        gate=guard_metrics(self.original_baseline,self.start_validation)
        write_json(self.out/'start-validation.json',dict(metrics=self.start_validation,gate=gate))
        if not gate['passed']:raise RuntimeError('Local anchor fails unchanged original gates: '+str(gate['failed']))
        self.diagnostic=self.diagnose_split(model)
        del model;gc.collect();torch.cuda.empty_cache()

    def diagnose_split(self,model):
        self.status('gradient-diagnosis',force=True,batches_done=0,batches_total=16)
        restore_random_states(self.checkpoint,self.replay)
        stream=ExactEpoch(self.files['train'],self.plan['seed'],self.checkpoint['new_positions'])
        named=list(model.named_parameters());params=[p for _,p in named]
        shared=[i for i,(n,_) in enumerate(named) if n.startswith(('piece.','rank.','file.','meta.','blocks.','norm.'))]
        records=[];model.train();lr=learning_rate(4001,12761,self.plan['lr'],self.plan['warmup'])
        for index in range(16):
            tick=time.time()
            rows,old,_,nold,ih=mixed_rows(stream,self.replay,self.plan,4000+index,1.)
            nt=len(rows);nq=int(rows['qmask'].sum());nv=float(rows['value_weight'].sum())
            vectors=[[torch.zeros_like(p,device='cpu') for p in params] for _ in NAMES];losses=defaultdict(float)
            for i in range(0,nt,self.plan['microbatch']):
                r=rows[i:i+self.plan['microbatch']];b=batch(r);mask=torch.as_tensor(old[i:i+len(r)],device='cuda')
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    with torch.no_grad():ref=self.teacher(b['tokens'][mask],b['meta'][mask],b['moves'][mask]) if mask.any() else None
                    out=model(b['tokens'],b['meta'],b['moves']);terms=split_terms(out,b,nt,nq,nv,mask,ref,nold)
                for j,n in enumerate(NAMES):
                    if not torch.isfinite(terms[n]):raise FloatingPointError('Nonfinite diagnostic loss')
                    grad=torch.autograd.grad(terms[n],params,retain_graph=j<8,allow_unused=True)
                    for dest,g in zip(vectors[j],grad):
                        if g is not None:dest.add_(g.detach().cpu())
                    losses[n]+=float(terms[n].detach());del grad
                del out,terms,b,ref
                self.status('gradient-diagnosis',batches_done=index,batches_total=16,microbatches_done=i//self.plan['microbatch']+1,microbatches_per_batch=nt//self.plan['microbatch'])
            matrix=gram(vectors);shared_matrix=gram([[v[i] for i in shared] for v in vectors])
            if not torch.isfinite(matrix).all() or not torch.isfinite(shared_matrix).all():raise FloatingPointError('Nonfinite gradient geometry')
            adam=adam_counterfactual(params,vectors,self.checkpoint['optimizer'],lr)
            record=dict(batch=index+1,input_sha256=ih,losses=dict(losses),gram_all=matrix.tolist(),gram_shared=shared_matrix.tolist(),
                shared=geometry(shared_matrix),norms={n:float(matrix[i,i].clamp_min(0).sqrt()) for i,n in enumerate(NAMES)},
                **geometry(matrix),adam=adam,seconds=time.time()-tick)
            records.append(record);del vectors
            write_json(self.out/'gradient-diagnosis-progress.json',dict(records=records,batches_done=len(records),optimizer_steps=0))
            self.status('gradient-diagnosis',force=True,batches_done=index+1,batches_total=16,batch_seconds=time.time()-tick)
        model.zero_grad(set_to_none=True)
        result=dict(records=records,batches=16,positions_per_batch=1024,names=NAMES,optimizer_steps=0,
            shared_parameter_count=sum(params[i].numel() for i in shared),all_parameter_count=sum(p.numel() for p in params),
            decision=decision(records),adam_scope='Hypothetical next update at fixed anchor and original Adam moments; not a 16-step optimization trajectory',
            test_labels_used=False)
        write_json(self.out/'gradient-diagnosis.json',result);return result

    def payload(self,model,optimizer,variant,offset,input_hash,last_good_offset=0,best_loss=None):
        s=self.checkpoint;step=4000+offset
        return dict(model=model.state_dict(),model_config=asdict(model.config),optimizer=optimizer.state_dict(),
            step=s['origin_step']+step,origin_step=s['origin_step'],epoch_updates=step,new_positions=s['new_positions']+offset*768,
            replay_positions=s['replay_positions']+offset*256,replay_rng=self.replay.rng.bit_generator.state,
            torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),baseline=self.original_baseline,frozen=self.frozen,
            original_plan_sha256=s['plan_sha256'],plan_sha256=hashlib.sha256(json.dumps(self.experiment_plan,sort_keys=True).encode()).hexdigest(),
            experiment_plan=self.experiment_plan,best_loss=best_loss,last_good_step=4000+last_good_offset,lineage=s['lineage'],
            experiment=dict(variant='D',value_kl_weight=2.,start_update=4000,updates=offset,
                source_sha256=self.certificate['checkpoint_sha256']['last.pt'],input_sequence_sha256=input_hash,
                original_guard=GUARD,requires_experiment_aware_resume=True),default_promoted=False)

    def baseline_search(self):
        from neurochess.multihead import MultiheadPlayer
        from neurochess.native_mcts import NativeMCTSPlayer
        self.status('local-frozen-parent-search',force=True);net=MultiheadPlayer(self.teacher,'cuda');roots=copy.deepcopy(self.frozen)
        for i,r in enumerate(roots):
            player=NativeMCTSPlayer(net,**self.lock['config'])
            try:move=player.choose(chess.Board(r['fen']))
            finally:player.close()
            r['original_L4_baseline_move']=r['baseline_move'];r['baseline_move']=move.uci()
            self.status('local-frozen-parent-search',search_done=i+1,search_total=len(roots))
        self.local_frozen=roots
        write_json(self.out/'local-frozen-parent-search.json',dict(roots=roots,checkpoint_sha256=self.plan['parent_sha256'],config=self.lock['config'],hardware=torch.cuda.get_device_name()))

    def local_search(self,model,frozen,stockfish,lock,step,progress=None):
        import chess.engine
        from neurochess.multihead import MultiheadPlayer
        from neurochess.native_mcts import NativeMCTSPlayer
        net=MultiheadPlayer(model,'cuda');model.eval();rows=[]
        with chess.engine.SimpleEngine.popen_uci(stockfish) as sf:
            sf.configure({'Threads':1,'Hash':64,'Skill Level':20,'UCI_LimitStrength':False})
            for i,r in enumerate(self.local_frozen):
                board=chess.Board(r['fen']);player=NativeMCTSPlayer(net,**lock['config']);tick=time.time()
                try:move=player.choose(board);stats=player.last_result
                finally:player.close()
                proposal=sf.analyse(board,chess.engine.Limit(nodes=100000),game=object())['pv'][0]
                moves=list(dict.fromkeys([proposal,board.parse_uci(r['baseline_move']),board.parse_uci(r['original_L4_baseline_move']),move]))
                info=sf.analyse(board,chess.engine.Limit(nodes=1000000),root_moves=moves,multipv=len(moves),game=object())
                scores={x['pv'][0].uci():x['score'].pov(board.turn).score(mate_score=100000) for x in info};best=max(scores.values())
                rows.append(dict(index=i,fen=r['fen'],move=move.uci(),baseline_move=r['baseline_move'],original_L4_baseline_move=r['original_L4_baseline_move'],
                    candidate_regret=best-scores[move.uci()],baseline_regret=best-scores[r['baseline_move']],
                    original_L4_baseline_regret=best-scores[r['original_L4_baseline_move']],scores=scores,search=stats,seconds=time.time()-tick))
                if progress:progress(i+1,len(self.local_frozen))
        local=screen_search_rows(rows);cloud=screen_search_rows([dict(r,baseline_regret=r['original_L4_baseline_regret']) for r in rows])
        return dict(step=step,original_L4_gate=cloud,local_gate=local,rows=rows,
            introduced_blunders=local['introduced_blunders'],mean_regret_delta_cp=local['mean_regret_delta_cp'],
            scope='Same frozen parent weights/configuration/48 roots/thresholds, local same-GPU parent choices plus original L4 choices; no Elo claim',
            **{'passed':bool(local['passed'] and cloud['passed'])})

    def run(self):
        self.prepare();self.results={}
        if self.diagnostic['decision']['run_D']:
            self.baseline_search()
            # Reuse the audited branch control flow, changing only D's objective/runtime adapters.
            original.VARIANTS['D']=dict(lr_multiplier=1.,balance_strength=1.,value_kl_weight=2.)
            original.variant_lr=lambda step,total,peak,warmup,name,offset:learning_rate(step,total,peak,warmup)
            original.joint_objective=d_objective;original.validate_all=local_validate;original.search_screen=self.local_search
            result=self.branch('D');self.results={'D':result}
        else:
            self.results={'D':dict(status='not-triggered',completed_updates=0,reason=self.diagnostic['decision'])}
        paired={}
        if self.results['D']['status']=='complete':
            for split in self.validation:
                with np.load(BASE/'auto4000/results/A'/f'positions-{split}.npz') as a,np.load(self.out/'D'/f'positions-{split}.npz') as d:
                    if not np.array_equal(a['keys'],d['keys']) or not np.array_equal(a['q_count'],d['q_count']):raise ValueError('Paired populations differ')
                    paired[split]={m:paired_comparison(a[m],d[m]) for m in ['top1','policy_ce','value_error']}
                    paired[split]['Q_MAE']=paired_q_comparison(a['q_abs_sum'],d['q_abs_sum'],a['q_count'])
            aresult=json.loads((BASE/'auto4000/results/A/result.json').read_text())
            if self.results['D']['input_sequence_sha256']!=aresult['input_sequence_sha256']:raise ValueError('D data sequence differs from A')
        for path,name in [(self.args.checkpoint,'last.pt'),(self.args.best,'best.pt')]:
            if sha(Path(path))!=self.certificate['checkpoint_sha256'][name]:raise RuntimeError('Original checkpoint modified')
        report=dict(branches=self.results,diagnostic_decision=self.diagnostic['decision'],paired_vs_A=paired,
            experiment_plan=self.experiment_plan,main_step=4000,main_training_resumed=False,automatic_promotion=False,
            test_labels_evaluated=False,finished_at=time.time(),hardware=torch.cuda.get_device_name())
        write_json(self.out/'results.json',report)
        manifest={str(p.relative_to(self.out)):sha(p) for p in self.out.rglob('*') if p.is_file() and p.suffix not in ['.log','.lock'] and p.name not in ['status.json','completion.json']}
        write_json(self.out/'completion.json',dict(completed=True,report_sha256=sha(self.out/'results.json'),files_sha256=manifest,no_auto_continue=True))
        self.status('failed' if self.results['D']['status']=='failed' else 'complete',force=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',default='/home/xia/Downloads/last.pt');p.add_argument('--best',default='/home/xia/Downloads/best.pt')
    p.add_argument('--general',default='/home/xia/Downloads/general-validation.npy');p.add_argument('--data',required=True)
    p.add_argument('--cache',default=str(ROOT/'data/lichess-clean-10m'));p.add_argument('--out',default=str(ROOT/'output/lichess-local-d-20261105'))
    a=p.parse_args();os.chdir(ROOT);torch.set_num_threads(2);torch.set_float32_matmul_precision('high')
    if not torch.cuda.is_available():raise RuntimeError('Existing local CUDA environment required')
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    lock=(out/'experiment.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (out/'started.json').exists():raise RuntimeError('Experiment already started; no automatic rerun')
    write_json(out/'started.json',dict(pid=os.getpid(),time=time.time(),python=sys.executable,torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name()))
    for folder in ['neurochess','scripts']:
        for src in (ROOT/folder).glob('*.py'):
            dest=out/'source'/folder/src.name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dest)
    experiment=LocalExperiment(a)
    def timeout(*unused):raise TimeoutError('18 hour local experiment budget exhausted')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(18*3600)
    try:experiment.run()
    except BaseException:
        error=traceback.format_exc();write_json(out/'failure.json',dict(traceback=error,time=time.time(),main_resumed=False));experiment.status('failed',force=True,error=error);raise
    finally:signal.alarm(0)

if __name__=='__main__':main()
