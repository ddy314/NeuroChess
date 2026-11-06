"""Adjudicate source/teacher conflicts before publishing the fresh curriculum.

A finite-node teacher is not proof that the curated puzzle solution is inferior.
Conflicts get a fresh coherent larger candidate-union search. If still disputed,
retain the source's policy supervision and mask Q/WDL, preserving all unique rows.
Original labels and all reviewed labels remain in the same durable SQLite DB.
"""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib,json,multiprocessing
from pathlib import Path
import sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import chess,chess.engine
import numpy as np
from neurochess.data import DTYPE,record
from neurochess.targets import assign_targets,TARGET_SPEC


def conflict(payload,metadata,threshold_cp=30):
    row=np.frombuffer(payload,dtype=DTYPE)[0]
    sign=1 if row['meta'][9] else -1
    scores={p['line']:{k:p[k]*sign for k in ['cp','mate'] if k in p} for p in metadata['pvs']}
    source=metadata['puzzle_solution'][0]
    if source not in scores:raise ValueError('Missing original puzzle solution in teacher candidate union')
    chosen=scores[source];finite=[s['cp'] for s in scores.values() if 'cp' in s]
    gap=max(finite)-chosen['cp'] if 'cp' in chosen and finite else None
    flags=dict(cp_gap=gap,cp_gap_gt_threshold=gap is not None and gap>threshold_cp,
        winning_mate_conflict=any(s.get('mate',0)>0 for s in scores.values()) and chosen.get('mate',0)<=0,
        avoidable_mate_loss=any('cp' in s or s.get('mate',0)>0 for s in scores.values()) and chosen.get('mate',0)<0)
    flags['disputed']=any(flags[k] for k in ['cp_gap_gt_threshold','winning_mate_conflict','avoidable_mate_loss'])
    return flags


def review_chunk(item):
    tasks,engine,nodes,coarse_nodes,threshold=item;results=[]
    with chess.engine.SimpleEngine.popen_uci(engine,timeout=120) as sf:
        sf.configure({'Threads':1,'Hash':64,'Skill Level':20,'UCI_LimitStrength':False})
        for job,original_payload,raw_metadata in tasks:
            old=json.loads(raw_metadata);before=conflict(original_payload,old,threshold)
            if not before['disputed']:raise ValueError('Only preselected disputed labels may be reviewed')
            b=chess.Board(job['initial_fen'])
            for move in job['history']:b.push_uci(move)
            from neurochess.encoding import position_key
            if position_key(b).hex()!=job['key'] or old['puzzle_solution']!=job['solution']:raise ValueError('Source identity differs')
            coarse=sf.analyse(b,chess.engine.Limit(nodes=coarse_nodes),multipv=min(4,b.legal_moves.count()),game=object())
            candidates=list(dict.fromkeys([i['pv'][0] for i in coarse]+[b.parse_uci(p['line']) for p in old['pvs']]+[b.parse_uci(job['solution'][0])]))
            infos=sf.analyse(b,chess.engine.Limit(nodes=nodes),root_moves=candidates,multipv=len(candidates),game=object())
            pvs=[]
            for info in infos:
                score=info['score'].white();pv={'line':info['pv'][0].uci()}
                pv['mate' if score.is_mate() else 'cp']=score.mate() if score.is_mate() else score.score();pvs.append(pv)
            row,legal=record(b);assign_targets(row,b,legal,pvs)
            metadata=dict(pvs=pvs,depths=[i.get('depth') for i in infos],puzzle_solution=job['solution'],
                category=job['category'],source_group=job['group'],teacher_nodes=nodes,teacher_coarse_nodes=coarse_nodes)
            after=conflict(row.tobytes(),metadata,threshold)
            fallback=after['disputed']
            if fallback:
                # Source-only move supervision, just like other policy-only replay.
                # Do not invent coherent Q/WDL labels when the two teachers disagree.
                row['policy']=0;row['policy'][legal.index(b.parse_uci(job['solution'][0]))]=1
                row['qmask']=False;row['value_weight']=0
            metadata['quality_review']=dict(schema=1,teacher_nodes=nodes,coarse_nodes=coarse_nodes,
                before=before,after=after,source_policy_only=fallback)
            results.append((job['key'],row.tobytes(),json.dumps(metadata)))
    return results


def row_digest(db):
    h=hashlib.sha256()
    for key,split,payload,metadata in db.execute('SELECT key,split,payload,metadata FROM labels ORDER BY key'):
        for value in [key.encode(),split.encode(),payload,metadata.encode()]:
            h.update(len(value).to_bytes(8,'little'));h.update(value)
    return h.hexdigest()


def quality_gate(db,jobs,engine,out,workers=8,nodes=1000000,coarse_nodes=100000,threshold_cp=30):
    from fresh_hard_data import atomic_json,digest
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    spec=dict(schema=1,teacher_nodes=nodes,coarse_nodes=coarse_nodes,threshold_cp=threshold_cp,
        engine_sha256=digest(engine),target_spec=TARGET_SPEC,unresolved='source-policy-only; candidate-Q and value masks zero')
    path=out/'quality-spec.json'
    if path.exists() and json.loads(path.read_text())!=spec:raise ValueError('Quality adjudication specification differs')
    atomic_json(path,spec)
    db.execute('CREATE TABLE IF NOT EXISTS quality_reviews(key TEXT PRIMARY KEY,original_payload BLOB NOT NULL,original_metadata TEXT NOT NULL,reviewed_payload BLOB NOT NULL,reviewed_metadata TEXT NOT NULL)')
    db.commit()
    # Detect source mismatches on every row; repaired rows remain auditable.
    candidates={};counts=Counter();started=time.time()
    for key,split,payload,raw in db.execute('SELECT key,split,payload,metadata FROM labels'):
        counts[split]+=1;metadata=json.loads(raw)
        if metadata.get('quality_review'):continue
        if conflict(payload,metadata,threshold_cp)['disputed']:candidates[key]=(payload,raw)
    selected_jobs={j['key']:j for j in jobs if j['key'] in candidates}
    if set(selected_jobs)!=set(candidates):raise ValueError('Disputed source jobs are missing')
    tasks=[(selected_jobs[key],*candidates[key]) for key in sorted(candidates)]
    if tasks:
        chunks=[(tasks[i:i+8],engine,nodes,coarse_nodes,threshold_cp) for i in range(0,len(tasks),8)]
        with ProcessPoolExecutor(workers,mp_context=multiprocessing.get_context('spawn')) as pool:
            for results in pool.map(review_chunk,chunks,chunksize=1):
                for key,payload,raw in results:
                    original,metadata=candidates[key]
                    db.execute('INSERT INTO quality_reviews VALUES (?,?,?,?,?)',(key,original,metadata,payload,raw))
                    db.execute('UPDATE labels SET payload=?,metadata=? WHERE key=?',(payload,raw,key))
                db.commit()
                atomic_json(out/'quality-progress.json',dict(phase='adjudicating',reviewed=db.execute('SELECT count(*) FROM quality_reviews').fetchone()[0],
                    pending_selected=len(tasks),seconds=time.time()-started,updated_at=time.time()))
    reviewed=resolved=policy_only=0
    for key,original,old,new,raw in db.execute('SELECT * FROM quality_reviews'):
        metadata=json.loads(raw);review=metadata['quality_review'];row=np.frombuffer(new,dtype=DTYPE)[0]
        current=db.execute('SELECT payload,metadata FROM labels WHERE key=?',(key,)).fetchone()
        if current!=(new,raw):raise ValueError('Quality review and training row diverged')
        reviewed+=1
        if review['source_policy_only']:
            if row['qmask'].any() or row['value_weight']!=0 or np.count_nonzero(row['policy'])!=1:raise ValueError('Unresolved source dispute must be policy-only')
            policy_only+=1
        else:
            if conflict(new,metadata,threshold_cp)['disputed']:raise ValueError('Unresolved teacher conflict escaped masking')
            resolved+=1
    # Full final scan, not only previously flagged keys. A newly appended row
    # cannot accidentally bypass the gate during a resumed preparation.
    unsupported=0
    for payload,raw in db.execute('SELECT payload,metadata FROM labels'):
        metadata=json.loads(raw)
        if conflict(payload,metadata,threshold_cp)['disputed'] and not metadata.get('quality_review',{}).get('source_policy_only'):unsupported+=1
    if unsupported:raise ValueError('Source conflicts without adjudication remain')
    proof=dict(complete=True,counts=dict(counts),rows=sum(counts.values()),reviewed=reviewed,resolved_by_deeper_teacher=resolved,
        unresolved_source_policy_only=policy_only,unsupported_source_conflicts=0,labels_sha256=row_digest(db),spec=spec,
        original_labels_preserved_in='SQLite quality_reviews table',seconds=time.time()-started,
        limitation='Deeper finite-budget evaluation is not exact chess truth; unresolved cases preserve curated source policy without Q/WDL targets.')
    atomic_json(out/'quality-gate.json',proof)
    return proof
