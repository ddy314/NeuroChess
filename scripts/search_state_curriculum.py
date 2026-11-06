"""Build grouped legal search-state jobs and balanced supervision from verified labels.

Planning/export only. No cloud allocation, teacher process or training is launched.
Descendants are derived states, never additional 'novel puzzle roots'. Each derived
state needs an independent side-to-move teacher label, not a sign-flipped root label.
"""
import argparse,collections,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import chess
import numpy as np
from neurochess.encoding import position_key
from neurochess.data import DTYPE,ShardWriter
from neurochess.audit import audit


def expand_jobs(roots,candidates=None,excluded_keys=(),excluded_groups=(),line_plies=4,descendant_limits=None):
    candidates=candidates or {};excluded_keys=set(excluded_keys);excluded_groups=set(excluded_groups)
    raw=[];group_splits=collections.defaultdict(set)
    for job in roots:group_splits[job['group']].add(job['split'])
    if any(len(s)>1 for s in group_splits.values()):raise ValueError('Source game appears in multiple splits')
    expanded=set()
    for split in ['train','val','test']:
        family=sorted((j for j in roots if j['split']==split),key=lambda j:hashlib.sha256(('expand:'+j['key']).encode()).digest())
        limit=(descendant_limits or {}).get(split,len(family))
        expanded.update(j['key'] for j in family[:limit])
    for job in roots:
        if job['group'] in excluded_groups:continue
        b=chess.Board(job['initial_fen'])
        for move in job['history']:b.push_uci(move)
        if position_key(b).hex()!=job['key']:raise ValueError('Root identity mismatch')
        def add(board,history,role,solution):
            key=position_key(board).hex()
            if board.is_game_over(claim_draw=True) or key in excluded_keys:return
            raw.append(dict(key=key,group=job['group'],split=job['split'],category=job.get('category','unknown'),
                initial_fen=job['initial_fen'],history=history,fen=board.fen(en_passant='fen'),solution=solution,
                root_key=job['key'],role=role,requires_independent_teacher_label=True,
                counts_as_additional_novel_puzzle_root=False,source_puzzle_id=job.get('id')))
        add(b,list(job['history']),'solver-root',job['solution'])
        if job['key'] not in expanded:continue
        branch=b.copy();history=list(job['history'])
        for ply,move in enumerate(job['solution'][:line_plies],1):
            branch.push_uci(move);history.append(move)
            # Source suffix is provenance only, not a value/Q target for this node.
            add(branch,list(history),'solution-opponent' if ply%2 else 'solution-solver',[])
        for move in sorted(set(candidates.get(job['key'],[]))):
            if move==job['solution'][0]:continue
            branch=b.copy();branch.push_uci(move)
            add(branch,list(job['history'])+[move],'candidate-child',[])
    # Canonical collisions across families/splits are quarantined, not assigned
    # according to whichever parent happens to be processed first.
    owners=collections.defaultdict(set)
    for j in raw:owners[j['key']].add((j['group'],j['split']))
    output={};quarantined={k for k,v in owners.items() if len(v)>1}
    for j in sorted(raw,key=lambda j:(j['key'],j['root_key'],j['role'])):
        if j['key'] not in quarantined:
            # Prefer the curated root policy when a same-family child equals it.
            if j['key'] not in output or j['role']=='solver-root':output[j['key']]=j
    return list(output.values()),dict(quarantined_keys=len(quarantined),jobs=len(output),roles=dict(collections.Counter(j['role'] for j in output.values())))


def value_bucket(row,threshold=.25):
    if row['value_weight']<=0:return None
    expected=float(row['value'][0]-row['value'][2])
    return 'winning' if expected>threshold else 'losing' if expected<-threshold else 'neutral'


def balanced_rows(records,quota):
    """Equal teacher-value strata interleaved within shards; preserve root policy.

    Surplus solver roots retain policy only. Surplus derived states are omitted.
    No repeated or flipped records are generated to satisfy a quota.
    """
    if quota<1:raise ValueError('Positive per-state quota required')
    by_key={}
    for job,row in records:
        if row.dtype!=DTYPE or bytes(row['key']).hex()!=job['key']:raise ValueError('Label/job identity mismatch')
        if job['key'] in by_key:raise ValueError('Duplicate canonical label')
        if not np.isfinite(row['value']).all() or (row['value']<0).any() or (row['value_weight']>0 and not np.isclose(row['value'].sum(),1,atol=.002)):
            raise ValueError('Invalid WDL target')
        legal=row['moves']>=0
        if not np.isfinite(row['policy']).all() or (row['policy']<0).any() or not np.isclose(row['policy'].sum(),1,atol=.002) or row['policy'][~legal].any():
            raise ValueError('Invalid policy target')
        if row['qmask'][~legal].any() or not np.isfinite(row['q'][row['qmask']]).all() or (abs(row['q'][row['qmask']])>1).any():
            raise ValueError('Invalid candidate Q target')
        by_key[job['key']]=(job,row)
    buckets={k:[] for k in ['winning','neutral','losing']}
    for job,row in by_key.values():
        bucket=value_bucket(row)
        if bucket:buckets[bucket].append((job,row))
    for name,items in buckets.items():
        items.sort(key=lambda x:hashlib.sha256(('balance:'+x[0]['key']).encode()).digest())
        if len(items)<quota:raise ValueError(f'Insufficient {name} labels: {len(items)} < {quota}; annotate real states, never duplicate to fill')
    selected=[buckets[k][:quota] for k in ['winning','neutral','losing']]
    chosen={j['key'] for bucket in selected for j,_ in bucket}
    policy_only=[]
    for job,row in sorted(by_key.values(),key=lambda x:x[0]['key']):
        if job['role']=='solver-root' and job['key'] not in chosen:
            row=row.copy();row['qmask']=False;row['value_weight']=0;policy_only.append(row)
    output=[];cursor=0
    for i in range(quota):
        for bucket in selected:output.append(bucket[i][1])
        # Spread policy-only roots across all value strata, including per-worker shards.
        end=(i+1)*len(policy_only)//quota
        output.extend(policy_only[cursor:end]);cursor=end
    return output,dict(value_counts={k:quota for k in buckets},policy_only_solver_roots=len(policy_only),unique_rows=len(output))


def export_balanced(jobs,db,out,quotas,teacher_spec):
    import sqlite3
    jobs_by_key={j['key']:j for j in jobs}
    if len(jobs_by_key)!=len(jobs):raise ValueError('Duplicate jobs')
    group_splits=collections.defaultdict(set)
    for j in jobs:group_splits[j['group']].add(j['split'])
    if any(len(s)>1 for s in group_splits.values()):raise ValueError('Source family crosses splits')
    from neurochess.targets import TARGET_SPEC
    if teacher_spec.get('target_spec')!=TARGET_SPEC:raise ValueError('Teacher target convention differs')
    if teacher_spec.get('jobs_sha256')!=hashlib.sha256((''.join(json.dumps(j,sort_keys=True)+'\n' for j in jobs)).encode()).hexdigest():raise ValueError('Teacher jobs SHA differs')
    con=sqlite3.connect(f'file:{Path(db).resolve()}?mode=ro',uri=True);records={s:[] for s in ['train','val','test']};seen=set()
    for key,split,payload in con.execute('SELECT key,split,payload FROM labels'):
        if key not in jobs_by_key or split!=jobs_by_key[key]['split']:raise ValueError('Label outside frozen job scope')
        if len(payload)!=DTYPE.itemsize:raise ValueError('Invalid label size')
        seen.add(key);records[split].append((jobs_by_key[key],np.frombuffer(payload,dtype=DTYPE)[0].copy()))
    if seen!=set(jobs_by_key):raise ValueError('Incomplete teacher labels')
    # Validate every split before creating any output artifact.
    prepared={};proof={}
    for split in records:
        if not records[split] and quotas.get(split,0)==0:continue
        prepared[split],proof[split]=balanced_rows(records[split],quotas[split])
    out=Path(out);out.mkdir(exist_ok=False,parents=True)
    for split in prepared:
        writer=ShardWriter(out/split,size=8192)
        for row in prepared[split]:writer.add('train' if split=='train' else 'val',row)
        writer.close(dict(kind='grouped-balanced-search-states',**proof[split],teacher_spec=teacher_spec,
            novel_puzzle_count_claim=False,solver_policy_retained=True,split=split))
        audit(writer.root)
    (out/'ready.json').write_text(json.dumps(dict(complete=True,counts=proof,teacher_spec=teacher_spec),indent=2)+'\n')


def annotate_jobs(jobs,engine,work,seed_db,seed_gate,workers=8,nodes=100000,coarse_nodes=10000):
    """Explicit teacher command; curated roots must reuse adjudicated source labels."""
    import sqlite3,multiprocessing
    from concurrent.futures import ProcessPoolExecutor
    sys.path.insert(0,str(ROOT/'scripts'))
    from fresh_hard_data import annotate_chunk,atomic_json,digest
    from fresh_teacher_quality import row_digest
    from neurochess.targets import TARGET_SPEC
    gate=json.loads(Path(seed_gate).read_text())
    seed=sqlite3.connect(f'file:{Path(seed_db).resolve()}?mode=ro',uri=True)
    if not gate.get('complete') or gate.get('unsupported_source_conflicts')!=0 or row_digest(seed)!=gate['labels_sha256']:
        raise ValueError('Source-root quality gate/digest missing or changed')
    if gate['spec']['engine_sha256']!=digest(engine):raise ValueError('Teacher engine differs from adjudicated source')
    work=Path(work);work.mkdir(parents=True,exist_ok=True)
    text=''.join(json.dumps(j,sort_keys=True)+'\n' for j in jobs)
    spec=dict(schema=1,jobs_sha256=hashlib.sha256(text.encode()).hexdigest(),target_spec=TARGET_SPEC,
        engine_sha256=digest(engine),nodes=nodes,coarse_nodes=coarse_nodes,seed_labels_sha256=gate['labels_sha256'],
        source_root_labels='Reuse adjudicated payloads including policy-only unresolved roots; never overwrite',derived_labels='Independent side-to-move coherent MultiPV; no inherited or sign-flipped root labels')
    if (work/'spec.json').exists() and json.loads((work/'spec.json').read_text())!=spec:raise ValueError('Resume teacher specification changed')
    atomic_json(work/'spec.json',spec)
    db=sqlite3.connect(work/'labels.sqlite');db.execute('CREATE TABLE IF NOT EXISTS labels(key TEXT PRIMARY KEY,split TEXT,payload BLOB,metadata TEXT)')
    expected={j['key']:j for j in jobs};done={r[0] for r in db.execute('SELECT key FROM labels')}
    if not done<=set(expected):raise ValueError('Resume labels outside frozen scope')
    for job in jobs:
        if job['role']!='solver-root' or job['key'] in done:continue
        row=seed.execute('SELECT key,split,payload,metadata FROM labels WHERE key=?',(job['key'],)).fetchone()
        if row is None or row[1]!=job['split']:raise ValueError('Missing/mismatched adjudicated source root')
        db.execute('INSERT INTO labels VALUES (?,?,?,?)',row);done.add(job['key'])
    db.commit();pending=[j for j in jobs if j['key'] not in done]
    with ProcessPoolExecutor(workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        for offset in range(0,len(pending),workers*32):
            tasks=[(j,nodes,coarse_nodes) for j in pending[offset:offset+workers*32]]
            chunks=[(tasks[i:i+32],engine) for i in range(0,len(tasks),32)]
            for results in pool.map(annotate_chunk,chunks):
                for row in results:db.execute('INSERT INTO labels VALUES (?,?,?,?)',row);done.add(row[0])
            db.commit();atomic_json(work/'progress.json',dict(completed=len(done),total=len(jobs),ready_for_training=False))
    return spec


def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    build=sub.add_parser('jobs');build.add_argument('--roots',required=True);build.add_argument('--out',required=True)
    build.add_argument('--candidates');build.add_argument('--exclude-keys',required=True);build.add_argument('--exclude-groups',required=True)
    build.add_argument('--train-families',type=int,default=25000);build.add_argument('--val-families',type=int,default=2000)
    teacher=sub.add_parser('annotate');teacher.add_argument('--jobs',required=True);teacher.add_argument('--engine',required=True);teacher.add_argument('--work',required=True)
    teacher.add_argument('--seed-db',required=True);teacher.add_argument('--seed-quality-gate',required=True);teacher.add_argument('--workers',type=int,default=8)
    export=sub.add_parser('export');export.add_argument('--jobs',required=True);export.add_argument('--db',required=True);export.add_argument('--teacher-spec',required=True);export.add_argument('--out',required=True)
    export.add_argument('--quota',type=int,required=True);export.add_argument('--val-quota',type=int,required=True);export.add_argument('--test-quota',type=int,default=0)
    a=p.parse_args()
    if a.command=='jobs':
        roots=[json.loads(l) for l in Path(a.roots).read_text().splitlines()]
        jobs,proof=expand_jobs(roots,json.loads(Path(a.candidates).read_text()) if a.candidates else {},Path(a.exclude_keys).read_text().splitlines(),Path(a.exclude_groups).read_text().splitlines(),descendant_limits=dict(train=a.train_families,val=a.val_families,test=0))
        out=Path(a.out);out.mkdir(parents=True,exist_ok=False);text=''.join(json.dumps(j,sort_keys=True)+'\n' for j in jobs)
        (out/'jobs.jsonl').write_text(text);proof.update(jobs_sha256=hashlib.sha256(text.encode()).hexdigest(),ready_for_training=False,requires_teacher_annotation=True)
        (out/'plan.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof))
    else:
        jobs=[json.loads(l) for l in Path(a.jobs).read_text().splitlines()]
        if a.command=='annotate':annotate_jobs(jobs,a.engine,a.work,a.seed_db,a.seed_quality_gate,a.workers)
        else:export_balanced(jobs,a.db,a.out,dict(train=a.quota,val=a.val_quota,test=a.test_quota),json.loads(Path(a.teacher_spec).read_text()))

if __name__=='__main__':main()
