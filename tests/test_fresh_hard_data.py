import importlib.util
from pathlib import Path
import json,sqlite3
import chess
import numpy as np
import pytest
spec=importlib.util.spec_from_file_location('fresh_hard_data',Path(__file__).parents[1]/'scripts/fresh_hard_data.py')
f=importlib.util.module_from_spec(spec);spec.loader.exec_module(f)


def puzzle():
 return dict(PuzzleId='fixture',FEN=chess.STARTING_FEN,Moves='e2e4 e7e5 g1f3 b8c6',Rating='1900',Themes='opening advantage',GameUrl='https://lichess.org/abcdefgh/black#1',OpeningTags='Kings_Pawn_Game',Popularity='95',NbPlays='1000')


def test_solver_root_follows_setup_and_complete_solution_legal():
 r=puzzle();r['GameUrl']='https://lichess.org/abcdefgh#1';j=f.puzzle_job(r,37)
 b=chess.Board();b.push_uci('e2e4')
 assert j['fen']==b.fen(en_passant='fen') and j['solution'][0]=='e7e5'
 assert j['key']==f.position_key(b).hex()
 r['Moves']='e2e4 e7e4'
 with pytest.raises(ValueError):f.puzzle_job(r,37)


def test_source_game_group_is_independent_of_ply_and_player_token():
 assert f.group_id('https://lichess.org/abcdefgh1234#25')=='abcdefgh'
 assert f.group_id('https://lichess.org/abcdefgh/black#25')=='abcdefgh'
 assert f.group_split(f.group_id('https://lichess.org/abcdefgh#2'),37)==f.group_split(f.group_id('https://lichess.org/abcdefgh#99'),37)


def test_novelty_excludes_train_val_and_replay(tmp_path):
 roots=[]
 for name,move in [('core','e2e4'),('human','d2d4')]:
  root=tmp_path/name;root.mkdir();roots.append(root);b=chess.Board();b.push_uci(move);r,_=f.record(b)
  np.save(root/'val-000000.npy',np.array([r],dtype=f.DTYPE))
 db=f.make_exclusions(roots,tmp_path/'old.sqlite')
 assert db.execute('SELECT count(*) FROM old_keys').fetchone()[0]==2
 b=chess.Board();b.push_uci('d2d4')
 assert db.execute('SELECT 1 FROM old_keys WHERE key=?',(f.position_key(b),)).fetchone()


def test_coherent_labels_keep_root_perspective_and_annotated_mask(tmp_path):
 engine=str(Path(__file__).parents[1]/'local/stockfish/stockfish/stockfish-ubuntu-x86-64-avx2')
 f.initialize(engine)
 try:
  r=puzzle();r['GameUrl']='https://lichess.org/abcdefgh#1';j=f.puzzle_job(r,37)
  key,split,payload,meta=f.label((j,1000,500));row=np.frombuffer(payload,dtype=f.DTYPE)[0]
  assert key==j['key'] and split==j['split']
  assert row['qmask'].sum()>=2 and not row['qmask'][row['moves']<0].any()
  assert np.isclose(row['policy'].sum(),1,atol=.002)
  assert np.isclose(row['value'].sum(),1)
 finally:f.ENGINE.quit()


def test_opening_book_combines_continuations_and_excludes_heldout(tmp_path):
 spec=importlib.util.spec_from_file_location('fresh_train',Path(__file__).parents[1]/'scripts/fresh_v4_train_cloud.py')
 m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 # Exercise actual aggregation with enough unique generated legal book positions.
 import random
 rng=random.Random(12);lines=[]
 for i in range(60):
  b=chess.Board();moves=[]
  for _ in range(16):
   if b.is_game_over():break
   mv=rng.choice(list(b.legal_moves));moves.append(mv.uci());b.push(mv)
  lines.append(dict(moves=moves))
 core=tmp_path/'core';core.mkdir();fresh=tmp_path/'fresh';fresh.mkdir()
 for name in ['trainval','test']:(fresh/name).mkdir()
 out=m.prepare_book(dict(lines=lines,source='fixture'),core,fresh,tmp_path/'book',[chess.STARTING_FEN])
 rows=np.concatenate([np.load(p) for p in out.glob('train-*.npy')])
 assert not any(r['key'].tobytes()==f.position_key(chess.Board()) for r in rows)
 assert not rows['qmask'].any() and not rows['value_weight'].any()


def test_parallel_annotation_finishes_and_exports_all_three_splits(tmp_path):
 import subprocess,sys
 jobs=[]
 for i,moves in enumerate(['e2e4 e7e5','d2d4 d7d5','g1f3 d7d5']):
  r=puzzle();r['Moves']=moves;r['GameUrl']=f'https://lichess.org/abcde{i:03d}#1'
  j=f.puzzle_job(r,37);j['split']=['train','val','test'][i];jobs.append(j)
 path=tmp_path/'jobs.jsonl';path.write_text(''.join(json.dumps(j)+'\n' for j in jobs))
 plan=tmp_path/'plan.json';plan.write_text(json.dumps(dict(teacher_nodes=1000,teacher_coarse_nodes=500,teacher_workers=2)))
 engine=str(Path(__file__).parents[1]/'local/stockfish/stockfish/stockfish-ubuntu-x86-64-avx2')
 result=subprocess.run([sys.executable,str(Path(f.__file__)),'--plan',str(plan),'--jobs',str(path),'--engine',engine,
  '--work',str(tmp_path/'work'),'--out',str(tmp_path/'out')],capture_output=True,text=True,timeout=30)
 assert result.returncode==0,result.stderr
 audit=json.loads((tmp_path/'out/annotation-audit.json').read_text())
 assert audit['trainval']['counts']==dict(train=1,val=1)
 assert audit['test']['counts']==dict(train=0,val=1)
 # Completed calls are idempotent; preserve the frozen artifact.
 result=subprocess.run([sys.executable,str(Path(f.__file__)),'--plan',str(plan),'--jobs',str(path),'--engine',engine,
  '--work',str(tmp_path/'work'),'--out',str(tmp_path/'out')],capture_output=True,text=True,timeout=30)
 assert result.returncode==0,result.stderr


def test_puzzle_accuracy_accepts_alternative_immediate_mates():
 import sys
 scripts=str(Path(__file__).parents[1]/'scripts')
 if scripts not in sys.path:sys.path.insert(0,scripts)
 from eval_fresh_puzzles import correct
 j=dict(initial_fen='7k/5Q2/6K1/8/8/8/8/8 w - - 0 1',history=[],solution=['f7g7'])
 assert correct(j,chess.Move.from_uci('f7f8'))


def test_blunder_teacher_compares_every_method_in_one_candidate_union():
 import sys
 scripts=str(Path(__file__).parents[1]/'scripts')
 if scripts not in sys.path:sys.path.insert(0,scripts)
 from eval_fresh_blunders import inspect_chunk
 r=puzzle();r['GameUrl']='https://lichess.org/abcdefgh#1';job=f.puzzle_job(r,37)
 engine=str(Path(__file__).parents[1]/'local/stockfish/stockfish/stockfish-ubuntu-x86-64-avx2')
 rows=inspect_chunk(([(job,{'control':'e7e5','fresh':'c7c5'})],engine,1000))
 assert len(rows)==1 and set(rows[0]['methods'])=={'control','fresh'}
 assert {'e7e5','c7c5'}<=set(rows[0]['teacher_scores'])
 assert all(('cp' in s)!=('mate' in s) for s in rows[0]['teacher_scores'].values())
