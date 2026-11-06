import chess,numpy as np,pytest
from scripts.search_state_curriculum import expand_jobs,balanced_rows
from neurochess.encoding import position_key
from neurochess.data import record


def job(split='train',group='abcdefgh'):
 b=chess.Board();b.push_uci('e2e4')
 return dict(initial_fen=chess.STARTING_FEN,history=['e2e4'],key=position_key(b).hex(),group=group,split=split,solution=['e7e5','g1f3','b8c6'],category='tactical')


def test_children_preserve_real_history_split_and_require_own_labels():
 root=job();jobs,proof=expand_jobs([root],{root['key']:['d7d5']})
 assert len(jobs)==5 and proof['quarantined_keys']==0
 for j in jobs:
  b=chess.Board(j['initial_fen'])
  for m in j['history']:b.push_uci(m)
  assert position_key(b).hex()==j['key'] and j['group']==root['group'] and j['split']=='train'
  assert j['requires_independent_teacher_label'] and not j['counts_as_additional_novel_puzzle_root']
  if j['role']!='solver-root':assert j['solution']==[]
 assert {j['role'] for j in jobs}>={'solution-opponent','solution-solver','candidate-child'}


def test_same_position_cross_split_is_quarantined_and_frozen_games_excluded():
 jobs,proof=expand_jobs([job(),job('val','ijklmnop')])
 assert jobs==[] and proof['quarantined_keys']>0
 jobs,_=expand_jobs([job()],excluded_groups=['abcdefgh']);assert jobs==[]
 root=job();jobs,_=expand_jobs([root],excluded_keys=[root['key']]);assert all(j['key']!=root['key'] for j in jobs)
 with pytest.raises(ValueError):expand_jobs([job(),job('val')])


def rows():
 result=[];b=chess.Board()
 for move,value in [('e2e4',[1,0,0]),('d2d4',[0,1,0]),('c2c4',[0,0,1]),('g1f3',[1,0,0])]:
  c=b.copy();c.push_uci(move);r,_=record(c);r['policy'][0]=1;r['qmask'][0]=True;r['value']=value;r['value_weight']=1
  result.append((dict(key=position_key(c).hex(),role='solver-root'),r))
 return result


def test_balanced_value_targets_retain_surplus_puzzle_policy_without_duplicates():
 out,proof=balanced_rows(rows(),1)
 assert len(out)==4 and proof['value_counts']==dict(winning=1,neutral=1,losing=1)
 assert proof['policy_only_solver_roots']==1 and len({bytes(r['key']) for r in out})==4
 masked=[r for r in out if r['value_weight']==0]
 assert len(masked)==1 and not masked[0]['qmask'].any() and masked[0]['policy'].sum()==1
 supervised=[r for r in out if r['value_weight']>0]
 assert [int(r['value'].argmax()) for r in supervised]==[0,1,2]


def test_shortage_and_duplicate_labels_cannot_be_padded_into_ready_data():
 with pytest.raises(ValueError,match='Insufficient'):balanced_rows(rows()[:2],1)
 with pytest.raises(ValueError,match='Duplicate'):balanced_rows(rows()+rows(),1)
 with pytest.raises(ValueError,match='Positive'):balanced_rows(rows(),0)


def test_corrupt_or_unsupervised_targets_do_not_become_value_labels():
 records=rows();records[0][1]['value']=[float('nan'),0,0]
 with pytest.raises(ValueError,match='WDL'):balanced_rows(records,1)
 records=rows();records[1][1]['value_weight']=0
 with pytest.raises(ValueError,match='Insufficient neutral'):balanced_rows(records,1)


def test_balanced_export_requires_complete_scope_and_has_no_split_leakage(tmp_path):
 import json,hashlib,sqlite3
 from scripts.search_state_curriculum import export_balanced
 from neurochess.targets import TARGET_SPEC
 alljobs=[];con=sqlite3.connect(tmp_path/'labels.sqlite');con.execute('CREATE TABLE labels(key TEXT,split TEXT,payload BLOB)')
 for split,setup in [('train','e2e4'),('val','d2d4'),('test','c2c4')]:
  b=chess.Board();b.push_uci(setup)
  for move,value in [('e7e5',[1,0,0]),('d7d5',[0,1,0]),('c7c5',[0,0,1])]:
   c=b.copy();c.push_uci(move);r,_=record(c);r['policy'][0]=1;r['value']=value;r['value_weight']=1
   j=dict(key=position_key(c).hex(),group=split,split=split,role='candidate-child');alljobs.append(j)
   con.execute('INSERT INTO labels VALUES (?,?,?)',(j['key'],split,r.tobytes()))
 con.commit()
 spec=dict(target_spec=TARGET_SPEC,jobs_sha256=hashlib.sha256((''.join(json.dumps(j,sort_keys=True)+'\n' for j in alljobs)).encode()).hexdigest())
 export_balanced(alljobs,tmp_path/'labels.sqlite',tmp_path/'data',dict(train=1,val=1,test=1),spec)
 ready=json.loads((tmp_path/'data/ready.json').read_text());assert ready['complete']
 for split in ['train','val','test']:assert ready['counts'][split]['unique_rows']==3
 con.execute('DELETE FROM labels WHERE key=?',(alljobs[0]['key'],));con.commit()
 with pytest.raises(ValueError,match='Incomplete'):export_balanced(alljobs,tmp_path/'labels.sqlite',tmp_path/'incomplete',dict(train=1,val=1,test=1),spec)
 assert not (tmp_path/'incomplete').exists()


def test_teacher_reuses_original_adjudicated_policy_only_root(tmp_path):
 import json,hashlib,sqlite3,sys
 from scripts.search_state_curriculum import annotate_jobs
 sys.path.insert(0,str(__import__('pathlib').Path(__file__).resolve().parents[1]/'scripts'))
 from fresh_teacher_quality import row_digest
 root=job();root.update(role='solver-root')
 b=chess.Board(root['initial_fen'])
 for m in root['history']:b.push_uci(m)
 row,_=record(b);row['policy'][0]=1
 # This deliberately represents an unresolved root: no Q/value labels.
 db=sqlite3.connect(tmp_path/'seed.sqlite');db.execute('CREATE TABLE labels(key TEXT PRIMARY KEY,split TEXT,payload BLOB,metadata TEXT)')
 db.execute('INSERT INTO labels VALUES (?,?,?,?)',(root['key'],'train',row.tobytes(),'{}'));db.commit()
 engine=tmp_path/'engine';engine.write_bytes(b'not executed')
 gate=tmp_path/'gate.json';gate.write_text(json.dumps(dict(complete=True,unsupported_source_conflicts=0,labels_sha256=row_digest(db),spec=dict(engine_sha256=hashlib.sha256(engine.read_bytes()).hexdigest()))))
 annotate_jobs([root],engine,tmp_path/'work',tmp_path/'seed.sqlite',gate,workers=1)
 result=sqlite3.connect(tmp_path/'work/labels.sqlite').execute('SELECT payload FROM labels').fetchone()[0]
 assert result==row.tobytes()
 db.execute('UPDATE labels SET metadata=?',('{"changed":true}',));db.commit()
 with pytest.raises(ValueError,match='digest'):annotate_jobs([root],engine,tmp_path/'other',tmp_path/'seed.sqlite',gate,workers=1)


def test_family_budget_preserves_root_policy_without_expanding_every_puzzle():
 root=job();jobs,proof=expand_jobs([root],descendant_limits=dict(train=0))
 assert len(jobs)==1 and jobs[0]['role']=='solver-root'


def test_old_two_arm_launcher_cannot_accidentally_restart():
 import subprocess,sys
 from pathlib import Path
 p=Path(__file__).resolve().parents[1]/'scripts/fresh_v4_train_cloud.py'
 result=subprocess.run([sys.executable,str(p)],capture_output=True,text=True,timeout=10)
 assert result.returncode!=0 and 'two-arm fresh-v4 launcher is retired' in result.stderr
