import json
from pathlib import Path
import chess
import chess.engine
import pytest
from neurochess.teacher_uci import FastTeacher
from neurochess.dynamic_curriculum import Store,DYNAMIC_TARGET_SPEC
from scripts import dynamic_teacher_worker as worker
from scripts.export_dynamic_curriculum import export
from tests.test_dynamic_curriculum import job,labelled,SHA


def test_transport_matches_python_chess_on_full_history_and_special_positions():
    binary=Path('output/targeted-scale-local-20261102/inputs/stockfish')
    if not binary.exists():pytest.skip('Local SF18 binary unavailable')
    slow=chess.engine.SimpleEngine.popen_uci(str(binary));slow.configure({'Threads':1,'Hash':64,'Skill Level':20,'UCI_LimitStrength':False})
    fast=FastTeacher(binary)
    boards=[chess.Board(),chess.Board('7k/P7/8/8/8/8/7p/4K3 w - - 0 1')]
    b=chess.Board()
    for m in ['e2e4','a7a6','e4e5','d7d5']:b.push_uci(m)
    boards.append(b)
    try:
        for b in boards:
            limit=chess.engine.Limit(nodes=5000);n=min(4,b.legal_moves.count())
            a=slow.analyse(b,limit,multipv=n,game=object());c=fast.analyse(b,limit,n)
            assert [(x['score'].white(),x['pv'],x['depth'],x['nodes']) for x in a]==[(x['score'].white(),x['pv'],x['depth'],x['nodes']) for x in c]
        assert fast.stats['searches']==3
    finally:slow.quit();fast.quit()
    assert fast.process.poll() is not None


def test_successor_is_independently_labelled_and_partial_policy_is_explicit(monkeypatch):
    class Engine:
        def __init__(self):self.budgets=[]
        def analyse(self,board,limit,multipv,root_moves=None):
            self.budgets.append(limit.nodes);ms=root_moves or [chess.Move.from_uci('e2e4'),chess.Move.from_uci('d2d4')]
            return [dict(pv=[m],score=chess.engine.PovScore(chess.engine.Cp((600 if limit.nodes<300000 else 900) if m.uci()=='e2e4' else 0),board.turn),depth=15,nodes=limit.nodes) for m in ms]
    engine=Engine();monkeypatch.setattr(worker,'ENGINE',engine)
    spec=dict(seed_nodes=10000,screen_nodes=30000,confirm_nodes=[100000,300000,1000000],minimum_confirm_nodes=300000,stability_cp=30,wdl_stability=.02,easy_audit_fraction=0,target_spec=DYNAMIC_TARGET_SPEC)
    result=worker.label(dict(job=job(),prediction=dict(model_sha256=SHA,move='d2d4',top_moves=['d2d4'],value=0),spec=spec,force_confirm=True))
    assert engine.budgets==[10000,30000,100000,300000]
    assert result['stable'] and result['repair'] and not result['supervision']['policy']
    assert all(s['independent'] for s in result['attempts'])


def test_snapshot_never_becomes_ready_from_a_small_pilot_or_without_proofs(tmp_path):
    db=tmp_path/'store.sqlite';store=Store(db);labelled(store);store.close()
    out=tmp_path/'snapshot';m=export(db,out,SHA,min_roots=1)
    assert m['counts']==dict(train=1,val=0,test=0)
    assert m['target_coverage']['policy']==0 and m['target_coverage']['value']==1
    assert m['requires_loss_mode']=='available-target-normalized-v2'
    assert not m['teacher_labels_ready'] and not (out/'teacher-ready.json').exists()
    metadata=json.loads((out/'train-000000.jsonl').read_text().splitlines()[0])
    assert metadata['job']['history']==[] and metadata['supervision']['policy'] is False
    with pytest.raises(FileExistsError):export(db,out,SHA,min_roots=1)
