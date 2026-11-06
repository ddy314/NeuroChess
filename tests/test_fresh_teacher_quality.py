import importlib.util,json,sqlite3,sys
from pathlib import Path
import chess
import numpy as np
import pytest
ROOT=Path(__file__).parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import fresh_teacher_quality as q
from neurochess.data import record,DTYPE
from neurochess.targets import assign_targets


def fixture(setup='e2e4',source='e7e5',alternative='c7c5',source_white=-100,alternative_white=-200):
 b=chess.Board();b.push_uci(setup);row,legal=record(b)
 pvs=[dict(line=source,cp=source_white),dict(line=alternative,cp=alternative_white)]
 assign_targets(row,b,legal,pvs)
 metadata=dict(pvs=pvs,puzzle_solution=[source],category='tactical',source_group='abcdefgh')
 job=dict(key=row['key'].tobytes().hex(),initial_fen=chess.STARTING_FEN,history=[setup],solution=[source],category='tactical',group='abcdefgh',split='train')
 return job,row.tobytes(),metadata


def test_conflict_uses_root_perspective_and_keeps_mates_separate():
 _,payload,meta=fixture()
 assert q.conflict(payload,meta)['cp_gap']==100
 assert q.conflict(payload,meta)['disputed']
 meta['pvs'][0]={'line':'e7e5','mate':-3}
 assert not q.conflict(payload,meta)['disputed']
 meta['pvs'][0]={'line':'e7e5','mate':3}
 assert q.conflict(payload,meta)['avoidable_mate_loss']
 assert q.conflict(payload,meta)['cp_gap'] is None


class Score:
 def __init__(self,value):self.value=value
 def white(self):return self
 def is_mate(self):return False
 def score(self):return self.value


class Engine:
 def __enter__(self):return self
 def __exit__(self,*a):pass
 def configure(self,*a):pass
 def analyse(self,*a,**kw):
  return [dict(pv=[chess.Move.from_uci(move)],score=Score(score),depth=25) for move,score in [('c7c5',-200),('e7e5',-100)]]


class InlinePool:
 def __init__(self,*a,**kw):pass
 def __enter__(self):return self
 def __exit__(self,*a):pass
 def map(self,fn,tasks,**kw):return map(fn,tasks)


def test_unresolved_conflict_masks_q_value_and_preserves_source_policy(monkeypatch):
 monkeypatch.setattr(chess.engine.SimpleEngine,'popen_uci',lambda *a,**kw:Engine())
 job,payload,meta=fixture()
 key,new,raw=q.review_chunk(([(job,payload,json.dumps(meta))],'fixture',1000000,100000,30))[0]
 row=np.frombuffer(new,dtype=DTYPE)[0];b=chess.Board();b.push_uci('e2e4');_,legal=record(b)
 assert key==job['key'] and json.loads(raw)['quality_review']['source_policy_only']
 assert not row['qmask'].any() and row['value_weight']==0
 assert np.count_nonzero(row['policy'])==1 and row['policy'][legal.index(chess.Move.from_uci('e7e5'))]==1


def test_gate_preserves_originals_is_resumable_and_scans_every_row(tmp_path,monkeypatch):
 monkeypatch.setattr(chess.engine.SimpleEngine,'popen_uci',lambda *a,**kw:Engine())
 monkeypatch.setattr(q,'ProcessPoolExecutor',InlinePool)
 db=sqlite3.connect(tmp_path/'labels.sqlite');db.execute('CREATE TABLE labels(key TEXT PRIMARY KEY,split TEXT,payload BLOB,metadata TEXT)')
 bad,payload,meta=fixture();db.execute('INSERT INTO labels VALUES (?,?,?,?)',(bad['key'],'train',payload,json.dumps(meta)))
 good,good_payload,good_meta=fixture('d2d4','d7d5','g8f6',-200,-100)
 db.execute('INSERT INTO labels VALUES (?,?,?,?)',(good['key'],'val',good_payload,json.dumps(good_meta)));db.commit()
 # The process is mocked above; only its persistent identity needs a real file.
 engine=tmp_path/'mock-stockfish';engine.write_bytes(b'mocked teacher identity')
 proof=q.quality_gate(db,[bad,good],str(engine),tmp_path/'quality',workers=1)
 assert proof['rows']==2 and proof['reviewed']==1 and proof['unresolved_source_policy_only']==1
 assert proof['unsupported_source_conflicts']==0
 assert db.execute('SELECT original_payload FROM quality_reviews').fetchone()[0]==payload
 assert db.execute('SELECT payload FROM labels WHERE key=?',(good['key'],)).fetchone()[0]==good_payload
 again=q.quality_gate(db,[bad,good],str(engine),tmp_path/'quality',workers=1)
 assert again['labels_sha256']==proof['labels_sha256'] and again['reviewed']==1
 with pytest.raises(ValueError,match='specification differs'):
  q.quality_gate(db,[bad,good],str(engine),tmp_path/'quality',workers=1,nodes=2000000)
