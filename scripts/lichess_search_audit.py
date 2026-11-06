"""Read-only 384-random/128-hard local audit of the immutable step4000 model.

No training, source mutation, baseline promotion, cloud access or test labels.
Resume only already frozen jobs; each completed root is atomically persisted.
"""
import torch  # Existing CUDA environment; no dependencies are installed.
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.append('/home/xia/code/llmlearn/.venv/lib/python3.13/site-packages')
import argparse,concurrent.futures,fcntl,hashlib,json,math,os,signal,threading,time,traceback
import numpy as np
import chess,chess.engine
from neurochess.encoding import encode,legal_ids,position_key
from neurochess.search_diagnostics import decode_board
from neurochess.model import ChessModel,ModelConfig
from neurochess.multihead import MultiheadPlayer,fuse_heads
from neurochess.native_mcts import NativeMCTSPlayer
from neurochess.teacher_uci import FastTeacher
from neurochess.epoch_ablation import screen_search_rows

BASE=ROOT/'output/lichess-joint-epoch1-20261104'
OUT=ROOT/'output/lichess-search-audit-512-20261105'
PARENT=ROOT/'output/puct-10x-20261031/frozen-baseline/model.pt'
CANDIDATE=Path('/home/xia/Downloads/last.pt')
ENGINE=ROOT/'local/stockfish/stockfish/stockfish-ubuntu-x86-64-avx2'
RULE=dict(random=384,hard=128,pool=4096,seed=202611055,
    discovery_nodes=1000000,comparison_nodes=1000000,confirmation_nodes=5000000,
    original_new_blunder_cp=100,original_worsening_cp=75,original_mean_delta_cp=10,
    confirmation_disagreement_cp=25,teacher_workers=2,teacher_threads=1,teacher_hash_mb=64,
    hard_score='4*fused_unannotated +2*parent_student_fused_disagree +1*student_fused_supportedQ_disagree + normalized_policy_entropy',
    random_sampling='Uniform without replacement from fresh validation positions excluding prior 8192 monitor, 2048 diagnostic and frozen48; no game IDs available',
    teacher_quality='Original fixed1M-node screen, stronger unrestricted1M discovery; confirmation5M. Actual depth/bounds reported. Not training labels.',
    confirmations='Any search >=100 surrogate_cp regret; abs search delta>=25 when choices differ; discovery/union best differs by>=25',
    search_order='Alternate parent/student order by frozen root index; one GPU, no concurrent model searches',
    test_labels_used=False,training_resumed=False,automatic_promotion=False)

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)

def read(p):return json.loads(Path(p).read_text())

def reconstruct(row):
    b=decode_board(row['tokens'],row['meta'])
    if not b.is_valid() or b.is_game_over() or b.legal_moves.count()<2:raise ValueError('Invalid audit root')
    tokens,meta=encode(b);meta[[5,6,7,10]]=0;moves,ids=legal_ids(b)
    if not np.array_equal(tokens,row['tokens']) or not np.allclose(meta,row['meta'],atol=1e-7):raise ValueError('Encoded board roundtrip failed')
    if position_key(b)!=bytes(row['key']) or not np.array_equal(ids,row['moves'][:len(ids)]):raise ValueError('Key/legal-order roundtrip failed')
    return b,moves

def root_outputs(net,rows):
    pieces=[]
    for i in range(0,len(rows),32):
        a=rows[i:i+32];pieces.append(net.raw_outputs(a['tokens'],a['meta'],a['moves']))
    return np.concatenate(pieces)

def direct(row,raw):
    board,moves=reconstruct(row);legal=row['moves']>=0
    p,q,wdl=raw[:256],raw[256:512],raw[512:]
    fused,prior,v,support=fuse_heads(p[None],q[None],wdl[None],legal[None])
    selected={'policy':int(p.argmax()),'Q_supported':int(np.where(support[0],q,-1e9).argmax()),'fused':int(fused[0].argmax())}
    top=list(np.argsort(-fused[0])[:min(3,len(moves))])
    return dict(moves={k:moves[i].uci() for k,i in selected.items()},
        annotated={k:bool(row['qmask'][i]) for k,i in selected.items()},
        top3=[moves[i].uci() for i in top],support=[moves[i].uci() for i in np.flatnonzero(support[0])],
        value=float(v[0]),predicted_Q={moves[i].uci():float(q[i]) for i in np.flatnonzero(legal)},
        predicted_policy={moves[i].uci():float(p[i]) for i in np.flatnonzero(legal)})

def difficulty(row,parent,student,raw):
    p=raw[:256];legal=row['moves']>=0;z=p[legal].astype(float);z-=z.max();probs=np.exp(z);probs/=probs.sum()
    entropy=float(-(probs*np.log(np.maximum(probs,1e-30))).sum()/max(math.log(len(z)),1e-9))
    return 4*int(not student['annotated']['fused'])+2*int(parent['moves']['fused']!=student['moves']['fused'])+int(student['moves']['fused']!=student['moves']['Q_supported'])+entropy

def verify_inputs():
    lock=read(ROOT/'configs/search-baseline-lock.json');certificate=read(BASE/'auto4000/results/pause-certificate.json')
    for n,h in lock['source_sha256'].items():
        if sha(ROOT/n)!=h:raise ValueError('Frozen search source differs: '+n)
    if sha(PARENT)!=lock['checkpoint_sha256'] or sha(CANDIDATE)!=certificate['checkpoint_sha256']['last.pt']:raise ValueError('Checkpoint SHA differs')
    return lock,certificate

def load_net(p):
    s=torch.load(p,map_location='cpu',weights_only=False,mmap=True)
    if not s['model_config'].get('history_free'):raise ValueError('Expected original history-free model')
    model=ChessModel(ModelConfig(**s['model_config']));model.load_state_dict(s['model'])
    if p==CANDIDATE and (s.get('epoch_updates')!=4000 or s.get('new_positions')!=3072000):raise ValueError('Wrong candidate cursor')
    model.eval().requires_grad_(False).cuda();return MultiheadPlayer(model,'cuda')

def prepare(out,nets,lock):
    if (out/'cohort.json').exists():
        cohort=read(out/'cohort.json');manifest=read(out/'cohort-manifest.json')
        if sha(out/'cohort.json')!=manifest['cohort_sha256'] or sha(out/'cohort.npy')!=manifest['rows_sha256']:raise ValueError('Frozen cohort changed')
        return cohort
    allrows=np.concatenate([np.load(p,mmap_mode='r') for p in sorted((ROOT/'data/lichess-clean-10m').glob('val-*.npy'))])
    if len(allrows)!=100000:raise ValueError('Wrong validation population')
    excluded=[]
    for p in ['weights-new_monitor-parent_trunk__parent_heads.npz','weights-new_disjoint-parent_trunk__parent_heads.npz']:
        excluded.append(np.load(ROOT/'output/lichess-next-step-review-20261105'/p)['keys'].copy().view('V16').ravel())
    frozen=read(BASE/'auto4000/results/pause-certificate.json')
    # The certificate's search journal carries the original48 roots.
    s=torch.load(CANDIDATE,map_location='cpu',weights_only=False,mmap=True)
    excluded.append(np.array([position_key(chess.Board(r['fen'])) for r in s['frozen']],dtype='V16'));del s
    excluded=np.unique(np.concatenate(excluded));keys=allrows['key'].copy().view('V16').ravel()
    available=np.flatnonzero(~np.isin(keys,excluded));rng=np.random.default_rng(RULE['seed']);order=rng.permutation(available)
    random_rows=allrows[order[:384]];pool=allrows[order[384:384+RULE['pool']]]
    raw={n:root_outputs(net,pool) for n,net in nets.items()}
    scored=[]
    for i,row in enumerate(pool):
        p=direct(row,raw['parent'][i]);c=direct(row,raw['candidate'][i]);scored.append((difficulty(row,p,c,raw['candidate'][i]),i,p,c))
    hard=sorted(scored,key=lambda x:(-x[0],x[1]))[:128]
    selected=np.concatenate([random_rows,pool[[x[1] for x in hard]]]);np.save(out/'cohort.npy',selected)
    outputs={n:root_outputs(net,selected) for n,net in nets.items()};jobs=[]
    for i,row in enumerate(selected):
        b,moves=reconstruct(row);value=float(row['value'][0]-row['value'][2])
        jobs.append(dict(index=i,key=bytes(row['key']).hex(),fen=b.fen(),cohort='random' if i<384 else 'hard',
            hard_score=None if i<384 else hard[i-384][0],selection_stratum='winning' if value>.25 else 'losing' if value<-.25 else 'neutral',
            source_annotation={m.uci():dict(Q=float(row['q'][j]),policy=float(row['policy'][j])) for j,m in enumerate(moves) if row['qmask'][j]},
            direct={n:direct(row,outputs[n][i]) for n in nets}))
    if len({j['key'] for j in jobs})!=512:raise ValueError('Duplicate selected roots')
    write(out/'cohort.json',jobs)
    write(out/'cohort-manifest.json',dict(cohort_sha256=sha(out/'cohort.json'),rows_sha256=sha(out/'cohort.npy'),
        selected=512,random=384,hard=128,pool=4096,eligible=len(available),excluded_keys=len(excluded),exact_overlap=0,
        scope='Validation-only, position-disjoint from prior monitored/diagnostic sets; no original game identities. Selected before timed search or fresh teacher scores.'))
    return jobs

def scored_infos(engine,b,nodes,moves=None):
    infos=engine.analyse(b,chess.engine.Limit(nodes=nodes),multipv=1 if moves is None else len(moves),
        root_moves=None if moves is None else [b.parse_uci(m) for m in moves])
    rows=[]
    for x in infos:
        if not x['pv']:raise ValueError('Empty teacher PV')
        c=b.copy(stack=False)
        for m in x['pv']:
            if m not in c.legal_moves:raise ValueError('Illegal teacher PV')
            c.push(m)
        s=x['score'].pov(b.turn)
        rows.append(dict(move=x['pv'][0].uci(),cp=s.score(),mate=s.mate(),sort_score=s.score(mate_score=100000),
            depth=x['depth'],nodes=x['nodes'],lowerbound=x.get('lowerbound',False),upperbound=x.get('upperbound',False),
            pv=[m.uci() for m in x['pv']]))
    if moves is not None and set(moves)!={x['move'] for x in rows}:raise ValueError('Teacher omitted requested candidates')
    return rows

def assessment(job,teacher):
    scores={x['move']:x['sort_score'] for x in teacher};best=max(scores.values())
    regrets={}
    for n in ['parent','candidate']:
        regrets[n]={k:best-scores[m] for k,m in {**job['direct'][n]['moves'],'search':job['search'][n]['move']}.items()}
    p,c=regrets['parent']['search'],regrets['candidate']['search']
    return dict(regrets=regrets,candidate_regret=c,baseline_regret=p,delta=c-p,
        introduced_blunder=c>=100 and c-p>=75,
        all_cp=all(x['cp'] is not None for x in teacher),
        minimum_depth=min(x['depth'] for x in teacher),
        all_exact=not any(x['lowerbound'] or x['upperbound'] for x in teacher))

def needs_confirmation(job,a,discovery,teacher):
    same=job['search']['parent']['move']==job['search']['candidate']['move']
    best=max(teacher,key=lambda x:x['sort_score'])
    found=next(x for x in teacher if x['move']==discovery[0]['move'])
    return bool(a['candidate_regret']>=100 or a['baseline_regret']>=100 or
        (not same and abs(a['delta'])>=25) or best['sort_score']-found['sort_score']>=25 or not a['all_exact'])

def teacher_job(out,job):
    path=out/'teachers'/f"{job['index']:03d}.json"
    if path.exists():
        x=read(path)
        if x['key']!=job['key']:raise ValueError('Teacher resume key mismatch')
        return x
    start=time.time();b=chess.Board(job['fen']);engine=FastTeacher(ENGINE,hash_mb=64,threads=1,timeout=240)
    try:
        discovery=scored_infos(engine,b,1000000)
        moves=[discovery[0]['move']]
        for n in ['parent','candidate']:
            moves.extend(job['direct'][n]['top3']);moves.extend(job['direct'][n]['moves'].values());moves.append(job['search'][n]['move'])
        moves=list(dict.fromkeys(moves))
        first=scored_infos(engine,b,1000000,moves);initial=assessment(job,first)
        confirm=needs_confirmation(job,initial,discovery,first)
        deep_discovery=deep=None
        if confirm:
            deep_discovery=scored_infos(engine,b,5000000)
            moves=list(dict.fromkeys(moves+[deep_discovery[0]['move']]))
            deep=scored_infos(engine,b,5000000,moves)
        result=dict(index=job['index'],key=job['key'],cohort=job['cohort'],fen=job['fen'],
            discovery=discovery,initial_teacher=first,initial=initial,confirmation_required=confirm,
            confirmation_discovery=deep_discovery,confirmation_teacher=deep,
            final=assessment(job,deep or first),seconds=time.time()-start,
            training_labels_ready=False,teacher_identity=engine.identity)
        write(path,result);return result
    finally:engine.quit()

def interval(values):
    a=np.asarray(values,dtype=float);rng=np.random.default_rng(RULE['seed'])
    draws=np.array([a[rng.integers(len(a),size=len(a))].mean() for _ in range(4000)])
    return dict(mean=float(a.mean()),bootstrap95=np.quantile(draws,[.025,.975]).tolist(),n=len(a),scope='Position-paired exploratory bootstrap, no game clustering or independent test confirmation')

def teacher_wl(item):
    score=chess.engine.Cp(item['cp']) if item['cp'] is not None else chess.engine.Mate(item['mate'])
    wdl=score.wdl(model='sf16',ply=32)
    return (wdl.wins-wdl.losses)/1000

def rank_accuracy(prediction,teacher):
    values=[]
    for i,a in enumerate(teacher):
        for b in teacher[i+1:]:
            if a['sort_score']==b['sort_score']:continue
            d=(prediction[a['move']]-prediction[b['move']])*(a['sort_score']-b['sort_score'])
            values.append(1. if d>0 else .5 if d==0 else 0.)
    return float(np.mean(values)) if values else None

def summarize(out,jobs):
    teachers=[read(out/'teachers'/f"{j['index']:03d}.json") for j in jobs]
    report=dict(completed=True,cohorts={},all_original_gates_passed=True,
        candidate_updates=4000,training_resumed=False,automatic_promotion=False,test_labels_used=False,
        limitation='FEN-only histories; development sample, hard cohort selected on model outputs; finite teachers; not Elo or universal quality proof')
    for cohort in ['random','hard']:
        pairs=[(j,t) for j,t in zip(jobs,teachers) if j['cohort']==cohort];ts=[t for _,t in pairs]
        rows=[dict(index=t['index'],candidate_regret=t['initial']['candidate_regret'],baseline_regret=t['initial']['baseline_regret']) for t in ts]
        gate=screen_search_rows(rows);report['all_original_gates_passed'] &= gate['passed']
        summary=dict(n=len(ts),original_1M_gate=gate,final_search_delta=interval([t['final']['delta'] for t in ts]),
            introduced_blunders_initial=[t['index'] for t in ts if t['initial']['introduced_blunder']],
            introduced_blunders_confirmed=[t['index'] for t in ts if t['final']['introduced_blunder']],
            same_search_moves=sum(j['search']['parent']['move']==j['search']['candidate']['move'] for j,_ in pairs),
            confirmed_roots=sum(t['confirmation_required'] for t in ts),
            depth24_roots=sum(t['final']['minimum_depth']>=24 for t in ts),
            all_cp_roots=sum(t['final']['all_cp'] for t in ts),all_exact_roots=sum(t['final']['all_exact'] for t in ts),
            models={})
        cp=[t['final']['delta'] for t in ts if t['final']['all_cp']]
        summary['cp_only_search_delta']=interval(cp) if cp else None
        for model in ['parent','candidate']:
            summary['models'][model]=dict(modes={m:dict(mean_regret=float(np.mean([t['final']['regrets'][model][m] for t in ts])),
                errors_ge100=sum(t['final']['regrets'][model][m]>=100 for t in ts)) for m in ['policy','Q_supported','fused','search']},
                search_seconds_mean=float(np.mean([j['search'][model]['seconds'] for j,_ in pairs])),
                search_seconds_p95=float(np.quantile([j['search'][model]['seconds'] for j,_ in pairs],.95)),
                fused_unannotated=sum(not j['direct'][model]['annotated']['fused'] for j,_ in pairs))
            ranked={k:[] for k in ['policy','Q']};q_errors=[];value_errors=[];by_stratum={}
            for j,t in pairs:
                reference=t['confirmation_teacher'] or t['initial_teacher']
                discovery=t['confirmation_discovery'] or t['discovery']
                d=j['direct'][model]
                for k in ranked:
                    v=rank_accuracy(d['predicted_policy' if k=='policy' else 'predicted_Q'],reference)
                    if v is not None:ranked[k].append(v)
                q_errors.extend(abs(d['predicted_Q'][x['move']]-teacher_wl(x)) for x in reference)
                error=abs(d['value']-teacher_wl(discovery[0]));value_errors.append(error)
                by_stratum.setdefault(j['selection_stratum'],[]).append(error)
            summary['models'][model].update(fresh_teacher_Q_MAE=float(np.mean(q_errors)),fresh_teacher_Q_labels=len(q_errors),
                fresh_unrestricted_Value_MAE=float(np.mean(value_errors)),
                fresh_Value_source_strata={k:dict(n=len(v),mae=float(np.mean(v))) for k,v in by_stratum.items()},
                candidate_union_pairwise_rank_accuracy={k:dict(mean=float(np.mean(v)) if v else None,n=len(v)) for k,v in ranked.items()})
        report['cohorts'][cohort]=summary
    write(out/'results.json',report)
    return report

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',default=str(OUT));p.add_argument('--pilot',type=int,default=0);args=p.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True);(out/'search').mkdir(exist_ok=True);(out/'teachers').mkdir(exist_ok=True)
    handle=(out/'audit.lock').open('w');fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
    started=time.time();signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('6-hour audit limit')));signal.alarm(6*3600)
    def status(phase,**extra):
        write(out/'status.json',dict(phase=phase,pid=os.getpid(),updated_at=time.time(),seconds=time.time()-started,
            training_resumed=False,total=512,**extra));print(json.dumps(dict(phase=phase,**extra)),flush=True)
    try:
        torch.set_num_threads(2);torch.set_float32_matmul_precision('high')
        lock,certificate=verify_inputs()
        if not torch.cuda.is_available() or '4060' not in torch.cuda.get_device_name():raise RuntimeError('Existing local4060 required')
        plan=dict(rule=RULE,search=lock,checkpoint_sha256=certificate['checkpoint_sha256'],candidate=str(CANDIDATE),parent=str(PARENT),
            stockfish_sha256=sha(ENGINE),hardware=torch.cuda.get_device_name(),torch=torch.__version__,precision='Existing MultiheadPlayer BF16 inference, high matmul',
            source_sha256={__file__:sha(__file__)},history='FEN-only, unknown history not fabricated; retained clocks are dataset defaults')
        if (out/'plan.json').exists():
            if read(out/'plan.json')!=plan:raise ValueError('Immutable audit plan changed')
        else:write(out/'plan.json',plan)
        status('loading-models');nets={n:load_net(v) for n,v in [('parent',PARENT),('candidate',CANDIDATE)]}
        # Warm both networks on identical inputs before timed searches.
        b=chess.Board();tokens,meta=encode(b);_,ids=legal_ids(b);mm=np.full((64,256),-1,np.int16);mm[:,:len(ids)]=ids
        for net in nets.values():
            for n in [1,64]:net.raw_outputs(np.repeat(tokens[None],n,0),np.repeat(meta[None],n,0),mm[:n])
        status('selecting-cohort');jobs=prepare(out,nets,lock);status('cohort-frozen',cohort_sha256=sha(out/'cohort.json'))
        count=min(args.pilot,len(jobs)) if args.pilot else len(jobs)
        # Alternate early cohorts so a bounded pilot also exercises difficult roots.
        order=[v for pair in zip(jobs[:128],jobs[384:]) for v in pair]+jobs[128:384]
        futures=[]
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            for i,job in enumerate(order[:count]):
                search_path=out/'search'/f"{job['index']:03d}.json"
                if search_path.exists():
                    saved=read(search_path)
                    if saved['key']!=job['key']:raise ValueError('Search resume key mismatch')
                    job=saved
                else:
                    job=dict(job,search={})
                    for name in (['parent','candidate'] if job['index']%2==0 else ['candidate','parent']):
                        player=NativeMCTSPlayer(nets[name],**lock['config']);tick=time.perf_counter()
                        try:move=player.choose(chess.Board(job['fen']));stats=player.last_result;timing=player.timing
                        finally:player.close()
                        job['search'][name]=dict(move=move.uci(),seconds=time.perf_counter()-tick,stats=stats,timing=timing)
                    write(search_path,job)
                futures.append(executor.submit(teacher_job,out,job))
                for f in futures:
                    if f.done():f.result()
                status('auditing',search_done=i+1,search_target=count,teacher_done=sum(f.done() for f in futures),teacher_target=count)
            status('teacher-finalization',search_done=count,teacher_done=sum(f.done() for f in futures))
            for f in concurrent.futures.as_completed(futures):
                f.result();status('teacher-finalization',search_done=count,teacher_done=sum(f.done() for f in futures))
        verify_inputs()
        if args.pilot:
            write(out/'pilot.json',dict(passed=True,roots=count,seconds=time.time()-started,searches=2*count,
                teacher_confirmations=sum(read(out/'teachers'/f"{j['index']:03d}.json")['confirmation_required'] for j in order[:count])))
            status('pilot-complete',search_done=count,teacher_done=count);return
        loaded=[read(out/'search'/f"{j['index']:03d}.json") for j in jobs]
        report=summarize(out,loaded)
        completion=dict(completed=True,report_sha256=sha(out/'results.json'),files_sha256={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file() and p.suffix not in ['.log','.lock','.tmp'] and p.name not in ['status.json','completion.json']},
            finished_at=time.time(),main_training_resumed=False,optimizer_steps=0)
        write(out/'completion.json',completion);status('complete',search_done=512,teacher_done=512,original_gates_passed=report['all_original_gates_passed'])
    except BaseException:
        error=traceback.format_exc();write(out/'failure.json',dict(traceback=error,at=time.time(),main_training_resumed=False));status('failed',error=error);raise

if __name__=='__main__':main()
