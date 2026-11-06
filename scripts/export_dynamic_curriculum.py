"""Export audited masked targets plus aligned history/provenance sidecars."""
import argparse,hashlib,itertools,json,sqlite3,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from neurochess.data import DTYPE
from neurochess.dynamic_curriculum import Store,DYNAMIC_TARGET_SPEC


def sha(path):return hashlib.file_digest(path.open('rb'),'sha256').hexdigest()


def export(db_path,out,model_sha,min_roots=100000,quality_proof=None,novelty_proof=None):
    out=Path(out);out.mkdir(parents=True,exist_ok=False);store=Store(db_path);store.db.commit()
    proof=store.counts(model_sha);buffers={s:[] for s in ['train','val','test']};sidecars={s:[] for s in buffers};counts={s:0 for s in buffers};shards={s:0 for s in buffers};files={};ambiguous=[];coverage={'policy':0,'Q_actions':0,'value':0,'terminal_value_only':0};strata={'winning':0,'neutral':0,'losing':0};groups={s:set() for s in buffers};split_coverage={s:dict(policy=0,Q=0,value=0) for s in buffers}
    unguarded=dict(policy=0,value=0)
    def flush(split):
        if not buffers[split]:return
        name=f'{split}-{shards[split]:06d}';array=out/(name+'.npy');meta=out/(name+'.jsonl')
        np.save(array,np.array(buffers[split],dtype=DTYPE));meta.write_text(''.join(json.dumps(m)+'\n' for m in sidecars[split]));files[array.name]=sha(array);files[meta.name]=sha(meta)
        buffers[split].clear();sidecars[split].clear();shards[split]+=1
    query='''SELECT s.input_key,s.case_id,s.position_key,s.split,s.role,s.job,d.active,d.priority,t.payload,t.metadata,t.label_id
       FROM states s JOIN decisions d ON d.case_id=s.case_id JOIN teachers t ON t.label_id=d.label_id
       WHERE d.model_sha=? AND s.quarantine=0 AND t.stable=1 AND t.payload IS NOT NULL ORDER BY s.input_key,s.case_id'''
    for key,items in itertools.groupby(store.db.execute(query,(model_sha,)),key=lambda r:r[0]):
        candidates=list(items);parsed=[np.frombuffer(r[8],dtype=DTYPE)[0].copy() for r in candidates];conflict=False
        for a in parsed:
            for b in parsed:
                if a['value_weight']>0 and b['value_weight']>0 and np.max(abs(a['value']-b['value']))>.05:conflict=True
                common=a['qmask']&b['qmask']
                if common.any() and np.max(abs(a['q'][common].astype(float)-b['q'][common].astype(float)))>.05:conflict=True
                if a['policy'].sum()>0 and b['policy'].sum()>0 and abs(a['policy'].astype(float)-b['policy'].astype(float)).sum()>.1:conflict=True
        if conflict:ambiguous.append(key);continue
        # One row per actual model input; no history aliases used as quantity padding.
        best=max(range(len(candidates)),key=lambda i:(int(parsed[i]['value_weight']>0)+int(parsed[i]['policy'].sum()>0),int(parsed[i]['qmask'].sum())))
        r=candidates[best];a=parsed[best];job=json.loads(r[5]);meta=json.loads(r[9]);legal=a['moves']>=0
        if not np.isfinite(a['policy']).all() or not np.isfinite(a['value']).all() or not np.isfinite(a['value_weight']) or a['value_weight']<0:raise ValueError('Nonfinite/invalid target weight')
        if a['policy'][~legal].any() or a['qmask'][~legal].any():raise ValueError('Illegal target action')
        if a['policy'].sum()>0 and not np.isclose(a['policy'].sum(),1.,atol=.002):raise ValueError('Unnormalized policy')
        if a['value_weight']>0 and not np.isclose(a['value'].sum(),1.,atol=.002):raise ValueError('Unnormalized value')
        if not np.isfinite(a['q'][a['qmask']]).all() or (abs(a['q'][a['qmask']])>1).any():raise ValueError('Invalid Q')
        split=r[3];groups[split].add(job['group']);buffers[split].append(a);counts[split]+=1
        coverage['policy']+=int(a['policy'].sum()>0);coverage['Q_actions']+=int(a['qmask'].sum());coverage['value']+=int(a['value_weight']>0);coverage['terminal_value_only']+=int(meta.get('terminal',False))
        guard=(meta.get('supervision') or {}).get('unrestricted_guard',{})
        unguarded['policy']+=int(a['policy'].sum()>0 and not guard.get('verified'))
        unguarded['value']+=int(a['value_weight']>0 and not meta.get('terminal') and not (guard.get('verified') and guard.get('global_wdl_stable') and guard.get('coherent_global_agreement')))
        split_coverage[split]['policy']+=int(a['policy'].sum()>0);split_coverage[split]['Q']+=int(a['qmask'].any());split_coverage[split]['value']+=int(a['value_weight']>0)
        if split=='train' and a['value_weight']>0:
            v=float(a['value'][0]-a['value'][2]);strata['winning' if v>.25 else 'losing' if v<-.25 else 'neutral']+=1
        anchor=store.db.execute("SELECT 1 FROM root_pools WHERE case_id=? AND pool='general'",(r[1],)).fetchone()
        visits=[dict(root_id=v[0],branch=v[1],depth=v[2],model_sha256=v[3]) for v in store.db.execute('SELECT root_id,branch,depth,model_sha FROM rollout_visits WHERE case_id=?',(r[1],))]
        sidecars[split].append(dict(case_id=r[1],position_key=r[2],input_key=key,split=split,role=r[4],rollout_visits=visits,replay_anchor=bool(anchor),active_mistake=bool(r[6]),priority=r[7],job=job,label_id=r[10],supervision=meta.get('supervision'),teacher_pending_heads=meta.get('pending_heads',[]),teacher_pvs=meta.get('teacher',[]),teacher_evidence=meta.get('attempts',[]),unrestricted_value_evidence=meta.get('value_audit',[]),teacher_attempts=[s['budget'] for s in meta.get('attempts',[])],model_sha256=model_sha))
        if len(buffers[split])>=8192:flush(split)
    for s in buffers:flush(s)
    if any(groups[a]&groups[b] for a,b in [('train','val'),('train','test'),('val','test')]):raise ValueError('Source game split leak')
    quality=json.loads(Path(quality_proof).read_text()) if quality_proof else {}
    novelty=json.loads(Path(novelty_proof).read_text()) if novelty_proof else {}
    keys=sorted(r[0] for r in store.db.execute("SELECT DISTINCT s.position_key FROM states s JOIN decisions d ON s.case_id=d.case_id WHERE s.role='root' AND s.quarantine=0 AND s.split='train' AND d.model_sha=?",(model_sha,)))
    cohort_sha=hashlib.sha256(('\n'.join(keys)+'\n').encode()).hexdigest()
    quality_passed=quality.get('passed') is True and quality.get('model_sha256')==model_sha and quality.get('root_cohort_sha256')==cohort_sha and quality.get('audited_unique_source_games',0)>=256
    novelty_certified=novelty.get('passed') is True and novelty.get('root_cohort_sha256')==cohort_sha and novelty.get('old_key_overlap')==0 and novelty.get('source_game_cross_split_overlap')==0
    heldout_qualified=all(split_coverage[s]['policy']>=512 and split_coverage[s]['Q']>=3000 and split_coverage[s]['value']>=3000 and len(groups[s])>=64 for s in ['val','test'])
    ready=proof['verified_policy_value_mistake_roots']>=min_roots and proof['qualified_general_replay_roots']>=120000 and proof['qualified_frontier_positions']>=40000 and heldout_qualified and quality_passed and novelty_certified and not ambiguous and not any(unguarded.values())
    manifest=dict(schema=2,target_spec=DYNAMIC_TARGET_SPEC,storage_dtype='legacy fixed-width tensors plus mandatory v2 sidecars',row_bytes=DTYPE.itemsize,counts=counts,source_games={k:len(v) for k,v in groups.items()},target_coverage=coverage,train_value_strata=strata,model_sha256=model_sha,
        requires_loss_mode='available-target-normalized-v2',policy_zero_means_missing_target=True,missing_Q_requires_qmask_false=True,missing_value_requires_weight_zero=True,
        descendants_are_not_additional_novel_roots=True,canonical_exposure_cap=8,unguarded_target_positions=unguarded,qualified_positions_by_split_and_head=split_coverage,heldout_heads_sufficient=heldout_qualified,ambiguous_input_keys=ambiguous,quality_passed=quality_passed,novelty_certified=novelty_certified,root_cohort_sha256=cohort_sha,teacher_labels_ready=ready,sha256=files,**proof)
    for name,expected in files.items():assert sha(out/name)==expected
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    if ready:(out/'teacher-ready.json').write_text(json.dumps(manifest,indent=2))
    store.close();return manifest


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--db',required=True);p.add_argument('--out',required=True);p.add_argument('--model-sha',required=True);p.add_argument('--min-roots',type=int,default=100000);a=p.parse_args()
    print(json.dumps(export(a.db,a.out,a.model_sha,a.min_roots),indent=2))
