import json
import subprocess
import sys
import chess
import chess.engine
from neurochess.encoding import position_key
from scripts import local_teacher_worker as worker
from scripts.local_targeted_annotation import build_index, connect, counts, save_result, pending_jobs, audit_standard_sources
import pytest


def job(split='train', family='a'):
    b=chess.Board()
    return dict(initial_fen=b.fen(), history=[], key=position_key(b).hex(), split=split, group=family)


def test_worker_has_no_torch_import_and_matches_cloud_teacher(monkeypatch):
    from scripts import targeted_distill_cloud as cloud
    class Engine:
        def __init__(self): self.budgets=[]
        def analyse(self, board, limit, multipv, game, root_moves=None):
            self.budgets.append(limit.nodes)
            moves=root_moves or [chess.Move.from_uci('e2e4'),chess.Move.from_uci('d2d4')]
            return [dict(pv=[m],score=chess.engine.PovScore(chess.engine.Cp(400 if m.uci()=='e2e4' else 0),board.turn),depth=15,nodes=limit.nodes) for m in moves]
    a,b=Engine(),Engine()
    monkeypatch.setattr(worker,'ENGINE',a);monkeypatch.setattr(cloud,'ENGINE',b)
    task=(job(),dict(move='d2d4',top_moves=['d2d4'],value=0))
    assert worker.label(task)==cloud.label(task)[0]
    assert a.budgets==b.budgets==[10000,30000,100000,300000]
    subprocess.run([sys.executable,'-c',"import scripts.local_teacher_worker; import sys; assert 'torch' not in sys.modules"],check=True)


def test_resume_dedup_quarantine_and_unstable_exclusion(tmp_path):
    inputs=tmp_path/'inputs';inputs.mkdir()
    first=job();duplicate=job()
    # Another position transposes across train/test and must be quarantined.
    board=chess.Board();board.push_uci('e2e4')
    cross=dict(first,history=['e2e4'],key=position_key(board).hex())
    rows=[first,duplicate,cross,dict(cross,split='test',group='test-game')]
    (inputs/'candidates-0.jsonl').write_text(''.join(json.dumps(j)+'\n' for j in rows))
    db=connect(tmp_path/'db.sqlite');build_index(db,inputs);build_index(db,inputs)
    assert db.execute('SELECT COUNT(*) FROM jobs').fetchone()[0]==2
    limits=dict(train=10,val=10,test=10);progress=counts(db)
    meta=dict(job=first,teacher=[dict(line='e2e4',cp=10)],prediction=dict(move='e2e4'),stable=True,repair=True)
    assert save_result(db,meta,limits,progress)
    assert not save_result(db,meta,limits,progress)
    assert not save_result(db,dict(meta,job=cross),limits,progress)
    db.commit();db.close();db=connect(tmp_path/'db.sqlite')
    assert counts(db)==progress
    assert counts(db)['verified_training_mistakes']==1
    board.push_uci('e7e5');unstable=dict(first,history=['e2e4','e7e5'],key=position_key(board).hex())
    db.execute('INSERT INTO jobs(key,split,family,job) VALUES (?,?,?,?)',(unstable['key'],'train','a',json.dumps(unstable)))
    save_result(db,dict(meta,job=unstable,stable=False,repair=False),limits,progress)
    assert progress['labelled']['train']==1 and progress['unstable_discarded']==1
    assert db.execute('SELECT row FROM results WHERE key=?',(unstable['key'],)).fetchone()[0] is None
    db.close()


def test_pending_job_pages_do_not_keep_wal_reader_open(tmp_path):
    db=connect(tmp_path/'db.sqlite')
    for i in range(600):
        j=dict(job(),key=str(i))
        db.execute('INSERT INTO jobs(key,split,family,job) VALUES (?,?,?,?)',(str(i),'train','a',json.dumps(j)))
    db.commit()
    iterator=iter(pending_jobs(db));seen=[]
    for encoded,_ in iterator:
        j=json.loads(encoded);seen.append(j['key'])
        db.execute('UPDATE jobs SET prediction=? WHERE key=?',('{}',j['key']));db.commit()
        if len(seen)==1:
            # A live SELECT cursor would report database-table-locked here.
            assert db.execute('PRAGMA wal_checkpoint(TRUNCATE)').fetchone()[0]==0
    assert len(seen)==len(set(seen))==600
    db.close()


def test_mislabeled_chess960_is_quarantined_before_teacher_and_revokes_prior_rows(tmp_path):
    bad=dict(job(family='freestyle'),initial_fen='rbbqnknr/pppppppp/8/8/8/8/PPPPPPPP/RBBQNKNR w KQkq - 0 1',
             source_game_headers={'Variant':'Standard'})
    with pytest.raises(worker.CandidateError,match='nonstandard_initial_position'):
        worker.board_for(bad)
    db=connect(tmp_path/'db.sqlite')
    db.execute('INSERT INTO jobs(key,split,family,job) VALUES (?,?,?,?)',('bad','train','freestyle',json.dumps(bad)))
    db.execute('INSERT INTO results VALUES (?,?,?,?,?,?,?,?,?)',('bad','train',1,1,1,0,json.dumps(dict(job=bad,accepted=True)),b'prior',0))
    db.commit()
    audit=audit_standard_sources(db,tmp_path)
    assert audit['quarantined_families']==1 and audit['excluded_positions']==1
    assert not list(pending_jobs(db))
    assert counts(db)['labelled']['train']==counts(db)['screened']==0
    row=db.execute('SELECT accepted,row,metadata FROM results').fetchone()
    assert row[0]==0 and row[1] is None and json.loads(row[2])['accepted'] is False
    db.close()
