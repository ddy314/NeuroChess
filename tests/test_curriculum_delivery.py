import importlib.util,json,sys
from pathlib import Path
from types import SimpleNamespace
import chess
import pytest
from neurochess.evaluation_check import verify_arena


def test_arena_schedules_and_verifies_64_games_per_mode(tmp_path,monkeypatch):
    spec=importlib.util.spec_from_file_location('arena_runner',Path('scripts/multihead_arena.py'))
    runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
    fens=[]
    for i in range(32):
        b=chess.Board('8/8/8/8/8/8/K7/7k w - - 0 1');b.fullmove_number=i+1;fens.append(b.fen())
    openings=tmp_path/'openings.fen';openings.write_text('\n'.join(fens)+'\n')
    checkpoint=tmp_path/'model.pt';checkpoint.write_bytes(b'model')
    class Player:
        def __init__(self,*a,**kw):self.config={};self.search_stats={}
        def close(self):pass
    class Engine:
        id={'name':'Stockfish 18'}
        def configure(self,*args):pass
        def quit(self):pass
    monkeypatch.setattr(runner,'load_model',lambda *a:(SimpleNamespace(config=SimpleNamespace(history_free=True)),{'step':1000}))
    monkeypatch.setattr(runner,'MultiheadPlayer',Player);monkeypatch.setattr(runner,'NativeMCTSPlayer',Player)
    monkeypatch.setattr(runner.chess.engine.SimpleEngine,'popen_uci',lambda *a,**k:Engine())
    monkeypatch.setattr(runner.torch.cuda,'get_device_name',lambda:'fixture')
    out=tmp_path/'arena'
    monkeypatch.setattr(sys,'argv',['arena','--checkpoint',str(checkpoint),'--openings',str(openings),'--stockfish','fixture','--out',str(out),'--pairs','32'])
    runner.main();proof=verify_arena(out,64,checkpoint)
    assert proof['games']==128 and proof['results']['direct']['draws']==64 and proof['results']['search']['draws']==64
    rows=(out/'games.jsonl').read_text().splitlines();row=json.loads(rows[0]);row['opening']=100;rows[0]=json.dumps(row)
    (out/'games.jsonl').write_text('\n'.join(rows)+'\n')
    with pytest.raises(ValueError,match='coverage'):verify_arena(out,64,checkpoint)


def test_teacher_publishes_only_complete_audited_wave(tmp_path,monkeypatch):
    import sqlite3
    import numpy as np
    from neurochess.data import DTYPE
    spec=importlib.util.spec_from_file_location('teacher',Path('scripts/teacher_colab.py'))
    teacher=importlib.util.module_from_spec(spec);spec.loader.exec_module(teacher)
    work=tmp_path/'work';work.mkdir();root=tmp_path/'root';root.mkdir()
    (root/'jobs.jsonl').write_text('fixture')
    monkeypatch.setattr(teacher,'WORK',work);monkeypatch.setattr(teacher,'ROOT',root);monkeypatch.setattr(teacher,'DRIVE',tmp_path/'drive')
    jobs=[dict(key=f'{i:032x}',split='train' if i<2 else 'val') for i in range(3)]
    db=sqlite3.connect(':memory:');db.execute('CREATE TABLE labels(key TEXT PRIMARY KEY,split TEXT,payload BLOB)')
    for job in jobs:
        row=np.zeros((),dtype=DTYPE);row['key']=np.frombuffer(bytes.fromhex(job['key']),dtype='u1')
        row['moves']=-1;row['moves'][0]=1;row['policy'][0]=1
        db.execute('INSERT INTO labels VALUES(?,?,?)',(job['key'],job['split'],row.tobytes()))
        if job!=jobs[-1]:assert not teacher.publish(db,jobs,1,train_per_wave=2,val_per_wave=1)
    assert teacher.publish(db,jobs,1,train_per_wave=2,val_per_wave=1)
    ready=json.loads((tmp_path/'drive/neurochess-data/hard-v3-wave-1-ready.json').read_text())
    assert ready['counts']=={'train':2,'val':1}
    assert ready['sha256']==teacher.sha(tmp_path/'drive/neurochess-data/hard-v3-wave-1.tar.gz')


@pytest.mark.parametrize('case',['success','overlap','wrong_scope'])
def test_parallel_teacher_merge_requires_disjoint_exact_scope(tmp_path,case):
    import sqlite3
    spec=importlib.util.spec_from_file_location('parallel_teacher',Path('scripts/teacher_colab.py'))
    teacher=importlib.util.module_from_spec(spec);spec.loader.exec_module(teacher)
    schema='CREATE TABLE labels(key TEXT PRIMARY KEY,split TEXT,payload BLOB,metadata TEXT)'
    main=sqlite3.connect(':memory:');main.execute(schema);main.execute('INSERT INTO labels VALUES(?,?,?,?)',('a','train',b'a','{}'));main.commit()
    helper=tmp_path/'helper.sqlite'
    with sqlite3.connect(helper) as db:
        db.execute(schema);db.execute('INSERT INTO labels VALUES(?,?,?,?)',('a' if case=='overlap' else 'b','train',b'b','{}'))
    expected={'a'} if case=='overlap' else {'c'} if case=='wrong_scope' else {'b'}
    if case=='success':
        teacher.merge_labels(main,helper,expected)
        assert main.execute('SELECT key FROM labels ORDER BY key').fetchall()==[('a',),('b',)]
    else:
        with pytest.raises(ValueError):teacher.merge_labels(main,helper,expected)
        assert main.execute('SELECT COUNT(*) FROM labels').fetchone()[0]==1
