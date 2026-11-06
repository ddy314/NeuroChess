"""Local asynchronous GPU screening + CPU teacher + legal student successors.

No optimizer, cloud VM, authentication or automatic training. SQLite commits
completed work in bounded batches. Resume does not repeat completed teacher
work; teacher evidence may be reused across student versions only when the
exact history, protocol and required candidate set are covered.
"""
import argparse,collections,concurrent.futures as futures,fcntl,hashlib,importlib.util,json,multiprocessing,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
if importlib.util.find_spec('chess') is None:sys.path.extend(str(p) for p in (ROOT/'.venv/lib').glob('python*/site-packages'))
from neurochess.dynamic_curriculum import Store,board_for,state_id,state_job,digest,teacher_spec_id,DYNAMIC_TARGET_SPEC
from scripts import dynamic_teacher_worker as teacher


def write(path,data):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2));tmp.replace(path)


def run(args):
    from scripts.dynamic_student import Student
    from neurochess.targets import score_targets
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True);lock=(out/'pipeline.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    cfg=json.loads(Path(args.config).read_text());model_path=Path(args.model or cfg['model']);model=Student(model_path,args.model_sha or cfg['model_sha256'])
    binary=Path(args.binary or cfg['teacher_binary']).resolve();spec=dict(cfg['teacher'],target_spec=DYNAMIC_TARGET_SPEC,engine_sha256=hashlib.file_digest(binary.open('rb'),'sha256').hexdigest(),score_pov='white',context='initial FEN plus complete legal history')
    store=Store(out/'curriculum.sqlite3');store.register_model(model.sha256,model.step,model_path)
    initial_proof=store.counts(model.sha256,teacher_spec_id(spec))
    predict_queue=collections.deque();teacher_queue=collections.deque();queued=set();expanded=set();student_context={};student_advanced=set();teacher_pending={};prediction_pending=None;completed=0;cache_hits=0;errors=[];initial_roots=0;started=time.monotonic();last_status=started;followups=0
    # Keep known training/frozen canonical keys out of this new-data snapshot.
    if args.exclude_keys:
        import numpy as np
        keys=np.load(args.exclude_keys,mmap_mode='r')
        if keys.dtype!=np.dtype('V16') or keys.ndim!=1:raise ValueError('Expected sortedV16 prior key index')
        def excluded(key):
            value=np.array(bytes.fromhex(key),dtype='V16');i=np.searchsorted(keys,value)
            return i<len(keys) and keys[i]==value
    else:excluded=lambda key:False
    if args.frozen_groups:
        forbidden=set(json.loads(Path(args.frozen_groups).read_text()))
    else:forbidden=set()
    def usable(case):return not store.db.execute('SELECT quarantine FROM states WHERE case_id=?',(case,)).fetchone()[0]
    def enqueue(job,pred=None):
        nonlocal cache_hits
        if job['group'] in forbidden:return None
        if excluded(job['key']):store.exclude([job['key']],'known prior/frozen canonical key')
        case=store.add(job)
        if job.get('branch')=='student':student_context.setdefault(case,{})[(job['root_id'],job['depth'])]=job
        if not usable(case):
            if store.job(case).get('claimable_draw'):return case
            if job.get('branch')=='student' and case not in queued and not store.job(case).get('terminal'):
                queued.add(case);predict_queue.append(case)
            return case
        if case in queued:
            existing=store.db.execute('SELECT prediction FROM predictions WHERE case_id=? AND model_sha=?',(case,model.sha256)).fetchone()
            if existing:advance_student(case,json.loads(existing[0]))
            evidence=store.db.execute('SELECT t.metadata FROM decisions d JOIN teachers t ON t.label_id=d.label_id WHERE d.case_id=? AND d.model_sha=? AND t.spec_sha=?',(case,model.sha256,teacher_spec_id(spec))).fetchone()
            if evidence:advance_adversary(case,json.loads(evidence[0]))
            return case
        queued.add(case)
        if store.job(case).get('terminal'):
            store.exact_terminal(case,model.sha256);return case
        existing=store.db.execute('SELECT prediction FROM predictions WHERE case_id=? AND model_sha=?',(case,model.sha256)).fetchone()
        if existing:pred=json.loads(existing[0])
        if pred and pred.get('model_sha256')==model.sha256:
            store.predict(case,pred);after_prediction(case,pred)
        else:predict_queue.append(case)
        return case
    def advance_student(case,pred):
        # The root player's turns alternate with independent SF replies. Do
        # not let the same student play a weak opponent and hide refutations.
        for origin,j in list(student_context.get(case,{}).items()):
            token=(case,*origin)
            if token in student_advanced or j['depth']%2 or j['depth']>=max(cfg['successor_plies']):continue
            student_advanced.add(token)
            b=board_for(j)
            if b.is_game_over(claim_draw=True):continue
            b.push_uci(pred['move']);child=state_job(j,b,'student',j['depth']+1);child_id=enqueue(child)
            if child_id:store.link(case,pred['move'],child_id,'student',model.sha256,child['root_id'],child['depth'])
    def advance_adversary(case,meta):
        for origin,j in list(student_context.get(case,{}).items()):
            token=(case,*origin)
            if token in student_advanced or j['depth']%2==0 or j['depth']>=max(cfg['successor_plies']):continue
            student_advanced.add(token);b=board_for(j)
            if b.is_game_over(claim_draw=True):continue
            references=meta.get('value_audit',[])
            move=(references[-1]['teacher'][0] if references else max(meta['teacher'],key=lambda p:score_targets(p,b.turn)[0]))['line'].split()[0]
            b.push_uci(move);child=state_job(j,b,'student',j['depth']+1);child_id=enqueue(child)
            if child_id:store.link(case,move,child_id,'teacher_reply_to_student',model.sha256,child['root_id'],child['depth'])
    def after_prediction(case,pred):
        nonlocal cache_hits
        j=store.job(case)
        advance_student(case,pred)
        reply_needed=any(ctx['depth']%2 and ctx['depth']<max(cfg['successor_plies']) for ctx in student_context.get(case,{}).values())
        if not usable(case) and not reply_needed:return
        if j.get('role','root')=='descendant' and j['depth'] not in cfg['successor_plies'] and not reply_needed:return
        cached=store.cached_label(case,pred,spec)
        if cached:
            identity,meta=cached;store.refresh(case,model.sha256,identity);cache_hits+=1
            advance_adversary(case,meta)
            if j.get('role','root')=='root':expand(case,pred,meta)
        else:
            anchor=store.db.execute("SELECT 1 FROM root_pools WHERE case_id=? AND pool='general'",(case,)).fetchone()
            teacher_queue.append(dict(job=j,prediction=pred,spec=spec,cached_value_audit=store.cached_value_audit(case,spec),needs_best_reply=bool(reply_needed),force_confirm=j.get('role')=='descendant' or bool(reply_needed) or bool(anchor)))
    def expand(case,pred,meta):
        nonlocal followups
        active=store.db.execute('SELECT active FROM decisions WHERE case_id=? AND model_sha=?',(case,model.sha256)).fetchone()
        if case in expanded or not active or not active[0] or followups>=args.expand_roots:return
        expanded.add(case);followups+=1;j=store.job(case);b=board_for(j)
        first=b.parse_uci(pred['move']);b.push(first);child=state_job(j,b,'student',1);child_id=enqueue(child)
        if child_id:store.link(case,first.uci(),child_id,'student',model.sha256,child['root_id'],child['depth'])
        b=board_for(j);references=meta.get('value_audit',[])
        pv=(references[-1]['teacher'][0] if references else max(meta['teacher'],key=lambda p:score_targets(p,b.turn)[0]))['line'].split();parent_id=case
        for depth,u in enumerate(pv[:max(cfg['successor_plies'])],1):
            if b.is_game_over(claim_draw=True):break
            b.push_uci(u);child=state_job(j,b,'teacher',depth);child_id=enqueue(child)
            if child_id:store.link(parent_id,u,child_id,'teacher',model.sha256,child['root_id'],child['depth']);parent_id=child_id
            else:break
    def status(phase):
        store.db.commit();proof=store.counts(model.sha256,teacher_spec_id(spec))
        new_roots=proof['verified_unique_mistake_roots']-initial_proof['verified_unique_mistake_roots']
        new_primary=proof['verified_policy_value_mistake_roots']-initial_proof['verified_policy_value_mistake_roots']
        unresolved=store.db.execute("SELECT COUNT(*) FROM decisions d JOIN teachers t ON d.label_id=t.label_id WHERE d.model_sha=? AND t.stable=1 AND json_extract(t.metadata,'$.supervision.value')=0",(model.sha256,)).fetchone()[0]
        write(out/'status.json',dict(phase=phase,model_sha256=model.sha256,model_step=model.step,teacher_spec=spec,
            roots_requested=initial_roots,expanded_roots=followups,completed_teacher_states=completed,cached_teacher_hits=cache_hits,
            pending_predictions=len(predict_queue),pending_teachers=len(teacher_queue)+len(teacher_pending),
            elapsed_seconds=time.monotonic()-started,new_root_mistakes=new_roots,new_policy_value_mistakes=new_primary,unsettled_value_targets=unresolved,
            usable_root_mistakes_per_second=new_roots/max(1,time.monotonic()-started),new_policy_value_mistakes_per_second=new_primary/max(1,time.monotonic()-started),
            errors=errors[-10:],training_started=False,new_cloud_instances=0,**proof))
    try:
        incoming=[]
        if args.refresh_store:
            incoming=[(0.,json.loads(j),None) for j, in store.db.execute("SELECT job FROM states WHERE quarantine=0 ORDER BY CASE role WHEN 'root' THEN 0 ELSE 1 END,case_id")]
        else:
            if not args.inputs:raise ValueError('Provide --inputs or --refresh-store')
            with Path(args.inputs).open() as stream:
                for line in stream:
                    r=json.loads(line);j=r.get('job',r);j=dict(j,role='root',depth=0)
                    if r.get('replay_anchor'):j['replay_anchor']=True
                    incoming.append((float(r.get('source_priority',0)),j,r.get('prediction')))
        incoming.sort(key=lambda r:-r[0]);root_groups=collections.Counter()
        with futures.ProcessPoolExecutor(args.workers,mp_context=multiprocessing.get_context('spawn'),initializer=teacher.initialize,initargs=(str(binary),)) as cpu, futures.ThreadPoolExecutor(1) as gpu:
            source=iter(incoming);source_exhausted=False
            status('local-dynamic-mining')
            while not source_exhausted or predict_queue or teacher_queue or teacher_pending or prediction_pending:
                # Do not reconstruct/store half a million histories before the
                # first GPU/SF task can start. Bound producer work as well as
                # worker futures, including adversarial child expansions.
                while not source_exhausted and len(predict_queue)+len(teacher_queue)+len(teacher_pending)<2*args.batch+4*args.workers:
                    if args.roots is not None and initial_roots>=args.roots:source_exhausted=True;break
                    try:_,j,pred=next(source)
                    except StopIteration:source_exhausted=True;break
                    if not args.refresh_store and root_groups[j['group']]>=args.max_roots_per_game:continue
                    root_groups[j['group']]+=1;enqueue(j,pred);initial_roots+=1
                if prediction_pending is None and predict_queue:
                    ids=[predict_queue.popleft() for _ in range(min(args.batch,len(predict_queue)))];jobs=[store.job(case) for case in ids]
                    prediction_pending=(ids,gpu.submit(model.jobs,jobs))
                if prediction_pending is not None and prediction_pending[1].done():
                    ids,f=prediction_pending;predictions=f.result();prediction_pending=None
                    for case,pred in zip(ids,predictions):store.predict(case,pred);after_prediction(case,pred)
                while teacher_queue and len(teacher_pending)<4*args.workers:
                    task=teacher_queue.popleft();case=state_id(task['job'])
                    existing=store.db.execute('SELECT 1 FROM decisions d JOIN teachers t ON d.label_id=t.label_id WHERE d.case_id=? AND d.model_sha=? AND t.spec_sha=?',(case,model.sha256,teacher_spec_id(spec))).fetchone()
                    if existing:continue
                    teacher_pending[cpu.submit(teacher.label,task)]=case
                if teacher_pending:
                    done,_=futures.wait(teacher_pending,timeout=.05,return_when=futures.FIRST_COMPLETED)
                    for f in done:
                        case=teacher_pending.pop(f);meta=f.result();identity=store.save_label(case,spec,meta);completed+=1
                        advance_adversary(case,meta)
                        if meta['stable']:
                            store.refresh(case,model.sha256,identity)
                            if meta['job'].get('role','root')=='root':expand(case,meta['prediction'],meta)
                        else:
                            # Persist screened-correct/unstable results too, avoiding SF reruns on resume.
                            store.db.execute('INSERT OR REPLACE INTO decisions VALUES (?,?,?,?,?,?)',(case,model.sha256,identity,0,.05,json.dumps(meta['decision'])))
                        if completed%32==0:store.db.commit()
                elif prediction_pending is not None:
                    futures.wait([prediction_pending[1]],timeout=.05)
                if time.monotonic()-last_status>=2:status('local-dynamic-mining');last_status=time.monotonic()
            status('local-pilot-complete-training-deferred')
    except BaseException:
        import traceback
        errors.append(traceback.format_exc());status('failed');raise
    finally:store.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/dynamic-curriculum-v1.json');p.add_argument('--inputs');p.add_argument('--refresh-store',action='store_true');p.add_argument('--out',required=True);p.add_argument('--roots',type=int);p.add_argument('--expand-roots',type=int,default=24);p.add_argument('--max-roots-per-game',type=int,default=4);p.add_argument('--workers',type=int,default=16);p.add_argument('--batch',type=int,default=128);p.add_argument('--model');p.add_argument('--model-sha');p.add_argument('--binary');p.add_argument('--exclude-keys');p.add_argument('--frozen-groups');run(p.parse_args())
