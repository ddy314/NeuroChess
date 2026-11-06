"""Local search-node preparation then exactly at most100 joint repair updates."""
import torch
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
sys.path.append('/home/xia/code/llmlearn/.venv/lib/python3.13/site-packages')
import argparse,concurrent.futures,fcntl,gc,hashlib,json,os,time,traceback,shutil,threading
import numpy as np
import chess
from neurochess.data import DTYPE,record
from neurochess.encoding import encode,legal_ids,position_key
from neurochess.guarded_epoch import ExactEpoch,GUARD
from neurochess.multihead import fuse_heads
from neurochess.search_diagnostics import decode_board
from neurochess.native_mcts import NativeMCTSPlayer
from neurochess.teacher_uci import FastTeacher
from neurochess.repair_pilot import complete_analysis,qualify_label,labeled_row,wl
from scripts.lichess_search_audit import verify_inputs,load_net,PARENT,CANDIDATE,ENGINE,sha,read,write

OUT=ROOT/'output/lichess-local-repair-pilot-20261105'
BASE=ROOT/'output/lichess-joint-epoch1-20261104'
DATA=ROOT/'data/lichess-clean-10m'
SEED=202611058
SPEC=dict(schema=1,steps=100,effective_batch=256,microbatch=32,repair_per_step=32,clean_per_step=160,old_per_step=64,
    root_candidates=80,training_root_groups=64,nodes_per_group=4,accepted_nodes=256,candidates_per_root=16,
    skip_original_new_rows=3456000,teacher_workers=4,teacher_min_depth=24,teacher_min_nodes=1000000,
    teacher_budgets=[1000000,5000000,20000000,80000000],teacher_confirmation_cap=160000000,
    labels_wall_budget_seconds=8*3600,lr=3e-7,warmup_steps=10,old_anchor_coefficients=[.5,.25,4.],
    functional_anchor_coefficients=[.25,.125,1.],policy_large_cp_rank_weight=.05,policy_large_cp_gap=100,
    validation_steps=[0,25,50,75,100],search_steps=[0,50,100],original_guard=GUARD,
    regression_only=[147,296,302,386,420],test_labels_used=False,automatic_expansion=False,automatic_promotion=False)

def status(phase,**kw):
    write(OUT/'status.json',dict(phase=phase,pid=os.getpid(),updated_at=time.time(),training_resumed=False,original_main_updates=4000,**kw))
    print(json.dumps(dict(phase=phase,**kw)),flush=True)

def key_array(rows):return rows['key'].copy().view('V16').ravel()

def exclusions():
    path=OUT/'exclusions.npy'
    if path.exists():return exclude_previous_presentations(np.load(path))
    chunks=[]
    for pattern,folder in [('val-*.npy',DATA),('test-*.npy',DATA),('val-*.npy',ROOT/'data/evals-v2-5m')]:
        for p in sorted(folder.glob(pattern)):chunks.append(key_array(np.load(p,mmap_mode='r')))
    general=Path('/home/xia/Downloads/general-validation.npy');chunks.append(key_array(np.load(general)))
    frozen=torch.load(CANDIDATE,map_location='cpu',weights_only=False,mmap=True)['frozen']
    reserved=[position_key(chess.Board(j['fen'])) for j in frozen]
    # Quarantine all five roots, every legal descendant up to2 plies, teacher PVs and diagnostic visited nodes.
    for index in SPEC['regression_only']:
        j=read(ROOT/f'output/lichess-search-audit-512-20261105/search/{index:03d}.json');root=chess.Board(j['fen']);frontier=[root]
        for depth in range(3):
            nxt=[]
            for b in frontier:
                reserved.append(position_key(b))
                if depth<2:
                    for move in b.legal_moves:
                        c=b.copy(stack=False);c.push(move);nxt.append(c)
            frontier=nxt
        teacher=read(ROOT/f'output/lichess-five-failure-mechanisms-20261105/coherent-teachers/{index:03d}.json')
        for line in teacher['final']:
            b=root.copy()
            for move in line['pv']:
                b.push_uci(move);reserved.append(position_key(b))
    review=read(ROOT/'output/lichess-five-failure-mechanisms-20261105/review/results.json')
    reserved.extend(position_key(chess.Board(j['fen'])) for j in review['visited_node_probes'])
    chunks.append(np.array(reserved,dtype='V16'))
    np.save(OUT/'replay-exclusions.npy',np.unique(np.concatenate(chunks)))
    for p in sorted((ROOT/'data/evals-v2-5m').glob('train-*.npy')):chunks.append(key_array(np.load(p,mmap_mode='r')))
    result=np.unique(np.concatenate(chunks));np.save(path,result)
    write(OUT/'exclusions.json',dict(total_keys=len(result),reserved_regression_related_keys=len(set(reserved)),
        scope='All new validation/test keys, known old corpus keys, original frozen48, five regressions2-ply/PV/diagnostic-node closure. No original source-game IDs; no game-independent claim.'))
    return exclude_previous_presentations(result)

def exclude_previous_presentations(blocked):
    """Fresh node targets also exclude the committed main/A/B/C input prefix."""
    proof=OUT/'previous-presentations-excluded.json'
    if proof.exists():
        p=read(proof)
        if sha(OUT/'exclusions.npy')!=p['exclusions_sha256']:raise ValueError('Presentation exclusion proof differs')
        return blocked
    stream=ExactEpoch(sorted(DATA.glob('train-*.npy')),20261104);keys=[]
    remaining=SPEC['skip_original_new_rows']
    while remaining:
        n=min(8192,remaining);keys.append(key_array(stream.next(n)));remaining-=n
    previous=np.unique(np.concatenate(keys));np.save(OUT/'previous-presentations.npy',previous)
    blocked=np.unique(np.concatenate([blocked,previous]));np.save(OUT/'exclusions.npy',blocked)
    description=read(OUT/'exclusions.json');description.update(total_keys=len(blocked),previous_new_prefix_rows=stream.consumed)
    write(OUT/'exclusions.json',description)
    write(proof,dict(original_main_and_ABC_prefix_rows=stream.consumed,unique_previous_positions=len(previous),
        previous_keys_sha256=sha(OUT/'previous-presentations.npy'),exclusions_sha256=sha(OUT/'exclusions.npy'),
        scope='Exact same deterministic original-new stream; excludes 4000 committed main updates and the shared next500 A/B/C updates.'))
    return blocked

def select_inputs(blocked):
    if (OUT/'clean.npy').exists():
        roots,clean=np.load(OUT/'roots.npy'),np.load(OUT/'clean.npy')
        keys=np.r_[key_array(roots),key_array(clean)]
        if len(np.unique(keys))!=len(keys) or np.isin(keys,blocked).any():raise ValueError('Existing root/clean input isolation failed')
        return roots,clean
    stream=ExactEpoch(sorted(DATA.glob('train-*.npy')),20261104,SPEC['skip_original_new_rows']);selected=[]
    need=SPEC['root_candidates']+SPEC['steps']*SPEC['clean_per_step']
    while len(selected)<need:
        rows=stream.next(512);valid=~np.isin(key_array(rows),blocked);selected.extend(rows[valid].copy())
    rows=np.array(selected[:need],dtype=DTYPE);roots=rows[:80];clean=rows[80:]
    assert len(np.unique(key_array(rows)))==len(rows)
    np.save(OUT/'roots.npy',roots);np.save(OUT/'clean.npy',clean)
    write(OUT/'input-selection.json',dict(root_count=len(roots),clean_count=len(clean),unused_original_cursor_start=SPEC['skip_original_new_rows'],
        original_cursor_examined_to=stream.consumed,exact_exclusion_overlap=0,rows_sha256=sha(OUT/'clean.npy'),roots_sha256=sha(OUT/'roots.npy')))
    return roots,clean

class Recorder:
    device='cuda'
    def __init__(self,net,seed):self.net=net;self.model=net.model;self.rng=np.random.default_rng(seed);self.rows=[];self.count=0
    def evaluate_search_heads(self,tokens,meta,moves):
        raw=self.net.raw_outputs(tokens,meta,moves)
        for t,m,ids in zip(tokens,meta,moves):
            if m[6] or m[7] or m[5]*150>=80:continue
            self.count+=1;x=(t.copy(),m.copy(),ids.copy())
            if len(self.rows)<128:self.rows.append(x)
            else:
                j=int(self.rng.integers(self.count))
                if j<128:self.rows[j]=x
        return fuse_heads(raw[:,:256],raw[:,256:512],raw[:,512:],moves>=0)[:3]

def observations(rows):
    result=[]
    for t,m,ids in rows:
        b=decode_board(t,m);b.halfmove_clock=0;b.fullmove_number=1
        if b.is_game_over() or b.legal_moves.count()<2:continue
        row,_=record(b);row['meta'][[5,6,7,10]]=0
        # Native and python-chess enumerate the same actions in different orders.
        # Fresh labels and fresh predictions below both use the Python row order.
        native_ids=ids[ids>=0];python_ids=row['moves'][row['moves']>=0]
        effective=np.array(m,copy=True);effective[[5,6,7,10]]=0
        if not np.array_equal(row['tokens'],t) or not np.allclose(row['meta'],effective,atol=1e-7) or len(np.unique(native_ids))!=len(native_ids) or not np.array_equal(np.sort(python_ids),np.sort(native_ids)):
            raise ValueError('Captured node input/legal set mismatch')
        # Preserve observed raw en-passant features, even when no EP capture is legal.
        result.append((b.fen(en_passant='fen'),row))
    return result

def collect(roots,clean,blocked,lock):
    if (OUT/'node-jobs.json').exists():return read(OUT/'node-jobs.json')
    nets=[load_net(PARENT),load_net(CANDIDATE)];seen=set(map(bytes,blocked));seen.update(map(bytes,key_array(clean)));jobs=[]
    for group,row in enumerate(roots):
        path=OUT/'collected'/f'{group:03d}.json'
        if path.exists():items=read(path)
        else:
            board=decode_board(row['tokens'],row['meta']);pool=[]
            for n,net in enumerate(nets):
                recorder=Recorder(net,SEED+group*2+n);player=NativeMCTSPlayer(recorder,**lock['config'])
                try:player.choose(board)
                finally:player.close()
                pool.extend(observations(recorder.rows))
            unique={bytes(r['key']):(fen,r) for fen,r in pool};pool=list(unique.values());items=[]
            if pool:
                rr=np.array([r for _,r in pool],dtype=DTYPE);raw=[]
                for net in nets:
                    raw.append(np.concatenate([net.raw_outputs(a['tokens'],a['meta'],a['moves']) for a in [rr[i:i+32] for i in range(0,len(rr),32)]]))
                scores=[]
                for i,(fen,r) in enumerate(pool):
                    direct=[]
                    for output in raw:
                        p,q,v=output[i,:256],output[i,256:512],output[i,512:]
                        fused,prior,value,support=fuse_heads(p[None],q[None],v[None],r['moves'][None]>=0)
                        b=chess.Board(fen);legal=list(b.legal_moves);choices=[int(p.argmax()),int(fused[0].argmax()),int(np.where(support[0],q,-1e9).argmax())]
                        direct.append(dict(moves=list(dict.fromkeys(legal[k].uci() for k in choices)),value=float(value[0]),fused=legal[choices[1]].uci()))
                    score=abs(direct[0]['value']-direct[1]['value'])+int(direct[0]['fused']!=direct[1]['fused'])
                    scores.append((score,i,direct))
                priority=sorted(scores,key=lambda z:(-z[0],z[1]));random=list(np.random.default_rng(SEED+group).permutation(len(pool)))
                order=[]
                for k in range(len(pool)):
                    for i in [priority[k][1],int(random[k])]:
                        if i not in order:order.append(i)
                byindex={i:(score,direct) for score,i,direct in scores}
                for i in order:
                    fen,r=pool[i];score,d=byindex[i];items.append(dict(group=group,source_root_key=bytes(row['key']).hex(),key=bytes(r['key']).hex(),fen=fen,
                        disagreement=score,predictions=d,history='Canonical history-free target; observed repetitions and near50-move states excluded'))
            write(path,items)
        chosen=[]
        for item in items:
            key=bytes.fromhex(item['key'])
            if key in seen:continue
            seen.add(key);chosen.append(item)
            if len(chosen)==SPEC['candidates_per_root']:break
        jobs.extend(chosen);status('collecting-search-nodes',root_done=group+1,root_total=len(roots),candidate_nodes=len(jobs))
    del nets;gc.collect();torch.cuda.empty_cache();write(OUT/'node-jobs.json',jobs);return jobs

class ActualSearchUnion:
    """Serialize GPU controls; CPU teachers retain their independent workers."""
    def __init__(self,lock):
        self.config=lock['config'];self.nets=dict(parent=load_net(PARENT),step4000=load_net(CANDIDATE));self.mutex=threading.Lock()
    def choices(self,job,deadline):
        path=OUT/'node-search-choices'/f"{job['group']:03d}-{job['key']}.json"
        with self.mutex:
            if path.exists():return read(path)
            if time.time()>deadline:raise TimeoutError('Deep label8-hour limit before search union')
            board=chess.Board(job['fen']);result={}
            for name,net in self.nets.items():
                player=NativeMCTSPlayer(net,**self.config)
                try:
                    move=player.choose(board);result[name]=dict(move=move.uci(),stats=player.last_result)
                finally:player.close()
            write(path,result);return result

def label_one(job,deadline,controls):
    path=OUT/'teacher-attempts'/f"{job['group']:03d}-{job['key']}.json"
    if path.exists():return read(path)
    engine=FastTeacher(ENGINE,threads=1,hash_mb=64,timeout=900);board=chess.Board(job['fen']);evidence={};start=time.time()
    def call(nodes,moves=None):
        if time.time()>deadline:raise TimeoutError('Deep label8-hour limit')
        return complete_analysis(engine,board,nodes,moves)
    def deep(moves=None):
        for nodes in SPEC['teacher_budgets']:
            rows=call(nodes,moves)
            if all(r['depth']>=24 and r['nodes']>=1000000 for r in rows):return nodes,rows
        raise ValueError('depth24-budget-cap')
    try:
        choices=controls.choices(job,deadline);evidence['actual_search_choices']=choices
        gnodes,g1=deep();g2=call(min(4*gnodes,160000000));evidence.update(global_first=g1,global_second=g2)
        candidates=list(dict.fromkeys([g1[0]['move'],g2[0]['move']]+[m for p in job['predictions'] for m in p['moves']]+[r['move'] for r in choices.values()]))
        if len(candidates)<2:
            candidates.extend(m.uci() for m in board.legal_moves if m.uci() not in candidates)
            candidates=candidates[:2]
        cnodes,first=deep(candidates);second=call(min(cnodes*4,160000000),candidates)
        evidence.update(first=first,second=second,candidates=candidates)
        ok,reason=qualify_label(first,second,g1,g2)
        scores={r['move']:r['sort_score'] for r in second};best=max(scores.values());p=job['predictions'][0]
        trusted=ok and best-scores[p['fused']]<=25 and abs(p['value']-wl(max(second,key=lambda r:r['sort_score'])))<=.05
        result=dict(job=job,accepted=bool(ok),reason=reason,anchor_trusted=bool(trusted),evidence=evidence,seconds=time.time()-start)
    except ValueError as e:result=dict(job=job,accepted=False,reason=str(e),evidence=evidence,seconds=time.time()-start)
    finally:engine.quit()
    write(path,result);return result

def label_groups(jobs,deadline,controls):
    bygroup={g:[j for j in jobs if j['group']==g] for g in range(80)}
    def worker(group):
        accepted=[]
        for j in bygroup[group]:
            r=label_one(j,deadline,controls)
            if r['accepted']:accepted.append(r)
            if len(accepted)==4:break
        write(OUT/'labeled-groups'/f'{group:03d}.json',dict(group=group,accepted=accepted,complete=len(accepted)==4))
        return group,accepted
    accepted_groups={}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for start,end in [(0,64),(64,80)]:
            futures={pool.submit(worker,g):g for g in range(start,end)}
            while futures:
                done,_=concurrent.futures.wait(futures,timeout=30,return_when=concurrent.futures.FIRST_COMPLETED)
                for f in done:
                    group,rows=f.result();futures.pop(f)
                    if len(rows)==4:accepted_groups[group]=rows
                counts=len(list((OUT/'teacher-attempts').glob('*.json')))
                status('deep-labeling',groups_qualified=len(accepted_groups),target_groups=64,attempts=counts,deadline=deadline)
            if len(accepted_groups)>=64:break
    if len(accepted_groups)<64:raise RuntimeError('Insufficient qualified training groups; refusing training')
    selected=sorted(accepted_groups)[:64];return [r for g in selected for r in accepted_groups[g]]

def prepare(plan):
    if (OUT/'data-ready.json').exists():
        ready=read(OUT/'data-ready.json')
        for n,h in ready['files_sha256'].items():
            if sha(OUT/n)!=h:raise ValueError('Prepared data changed: '+n)
        return
    status('verifying-local-data');lock,_=verify_inputs();ready=read(ROOT/'data/lichess-depth24-knodes1000-clean-10m-20261104/ready.json')
    original=read(BASE/'plan-r2.json')
    if sha(ROOT/'data/lichess-depth24-knodes1000-clean-10m-20261104/ready.json')!=original['ready_sha256']:raise ValueError('Data certificate changed')
    for i,item in enumerate(ready['shards']):
        p=DATA/item['data']['name'].removesuffix('.zst')
        if sha(p)!=item['data']['npy_sha256']:raise ValueError('Cached shard hash differs: '+str(p))
        if i%100==0:status('verifying-local-data',done=i,total=len(ready['shards']))
    checks=read(ROOT/'data/evals-v2-5m/checksums.json')
    for name,h in checks.items():
        if sha(ROOT/'data/evals-v2-5m'/name)!=h:raise ValueError('Old corpus changed: '+name)
    if sha(Path('/home/xia/Downloads/general-validation.npy'))!=original['general_validation_sha256']:raise ValueError('General validation changed')
    blocked=exclusions();roots,clean=select_inputs(blocked);jobs=collect(roots,clean,blocked,lock)
    if not (OUT/'label-deadline.json').exists():write(OUT/'label-deadline.json',dict(deadline=time.time()+SPEC['labels_wall_budget_seconds']))
    controls=ActualSearchUnion(lock)
    labeled=label_groups(jobs,read(OUT/'label-deadline.json')['deadline'],controls)
    del controls;gc.collect();torch.cuda.empty_cache()
    rows=[];cps=[];trusted=[]
    for r in labeled:
        row,cp=labeled_row(r['job']['fen'],r['evidence']['second']);rows.append(row);cps.append(cp);trusted.append(r['anchor_trusted'])
    rows=np.array(rows,dtype=DTYPE);keys=key_array(rows)
    if len(np.unique(keys))!=256 or np.isin(keys,blocked).any() or np.isin(keys,key_array(clean)).any():raise ValueError('Repair data isolation failed')
    np.save(OUT/'repair.npy',rows);np.save(OUT/'repair-cp.npy',np.array(cps));np.save(OUT/'repair-trusted.npy',np.array(trusted))
    write(OUT/'accepted-labels.json',labeled)
    files=['clean.npy','roots.npy','repair.npy','repair-cp.npy','repair-trusted.npy','accepted-labels.json','exclusions.npy','replay-exclusions.npy','previous-presentations.npy','previous-presentations-excluded.json','node-jobs.json']
    write(OUT/'data-ready.json',dict(passed=True,unique_repair_nodes=256,root_groups=64,clean_unique=16000,exact_heldout_old_or_regression_overlap=0,
        node_targets_all_three_heads=True,original_game_independence='Unavailable in Lichess eval source; exact canonical keys and root groups verified',
        files_sha256={n:sha(OUT/n) for n in files},plan_sha256=sha(OUT/'plan.json')))
    status('data-ready',unique_repair_nodes=256,root_groups=64)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare-only',action='store_true');args=parser.parse_args()
    OUT.mkdir(exist_ok=True,parents=True);lockfile=(OUT/'pipeline.lock').open('w');fcntl.flock(lockfile,fcntl.LOCK_EX|fcntl.LOCK_NB)
    torch.set_num_threads(2);torch.set_float32_matmul_precision('high');torch.manual_seed(SEED)
    try:
        lock,cert=verify_inputs();plan=dict(spec=SPEC,seed=SEED,parent_sha256=lock['checkpoint_sha256'],collector_checkpoint_sha256=cert['checkpoint_sha256']['last.pt'],
            reset_optimizer_from_frozen_parent=True,single_encoder=True,initial_model='Frozen parent, not step4000 or rejected edited candidates',
            source_contract='History-free canonical training nodes; repetition/near50 observed states excluded; no test labels; five validation regressions and closure reserved')
        if (OUT/'plan.json').exists() and read(OUT/'plan.json')!=plan:raise ValueError('Immutable repair plan changed')
        write(OUT/'plan.json',plan);prepare(plan)
        if not args.prepare_only:
            from scripts.train_lichess_repair_pilot import train
            train(plan)
    except BaseException:
        write(OUT/'failure.json',dict(traceback=traceback.format_exc(),at=time.time(),training_resumed=False));status('failed');raise

if __name__=='__main__':main()
