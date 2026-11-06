import argparse,hashlib,io,json,sqlite3
import chess,chess.pgn,numpy as np,zstandard
from scripts.collect_dynamic_elite import collect,sha
from scripts.novel_elite_collect_cloud import game_group,group_split
from neurochess.dynamic_curriculum import board_for
from scripts.stage_dynamic_curriculum import select_anchors
import pytest


def test_ordinary_replay_selection_is_model_blind_capped_and_reproducible():
    jobs=[dict(group=str(i//5),model_error=1.) for i in range(100)]
    indices=select_anchors(jobs,40,max_per_game=2)
    assert len(indices)==40 and indices==select_anchors([dict(j,model_error=0.) for j in jobs],40,max_per_game=2)
    assert all(sum(jobs[i]['group']==g for i in indices)<=2 for g in {j['group'] for j in jobs})
    assert select_anchors(jobs,0)==set()
    with pytest.raises(RuntimeError,match='Insufficient'):select_anchors(jobs,41,max_per_game=2)


def test_new_source_history_origin_hash_and_resume(tmp_path):
    game=chess.pgn.read_game(io.StringIO('''[WhiteElo "2500"]
[BlackElo "2500"]
[TimeControl "600+5"]
[Result "1-0"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7
6. Re1 b5 7. Bb3 d6 8. c3 O-O 9. h3 Nb8 10. d4 Nbd7
11. c4 c6 12. Nc3 Qc7 13. a3 Bb7 14. Be3 Rfe8 15. Rc1 Bf8 1-0
'''))
    assert group_split(game_group(game.board(),list(game.mainline_moves())))=='train'
    archive=tmp_path/'fixture.pgn.zst';archive.write_bytes(zstandard.ZstdCompressor().compress(str(game).encode()))
    sources=tmp_path/'sources.json';sources.write_text(json.dumps([dict(name='fixture',url=archive.as_uri(),sha256=sha(archive),min_elo=2400,min_base_seconds=600,license='test')]))
    index=tmp_path/'keys.npy';np.save(index,np.array([b'0'*16],dtype='V16'))
    cert=tmp_path/'audit.json';cert.write_text(json.dumps(dict(complete_original_7590978_key_index=True,index_sha256=sha(index))))
    prior=tmp_path/'prior.sqlite';db=sqlite3.connect(prior);db.execute('CREATE TABLE groups(family TEXT)');db.close()
    args=argparse.Namespace(sources=str(sources),exclude_keys=str(index),exclusion_audit=str(cert),prior_db=str(prior),out=str(tmp_path/'out'),train=3,heldout=0)
    collect(args);rows=[json.loads(s) for s in (tmp_path/'out/train.jsonl').read_text().splitlines()]
    assert len(rows)==3
    for j in rows:
        assert j['initial_fen']==chess.STARTING_FEN and len(j['history'])>=14
        board_for(j)
    before=sha(tmp_path/'out/train.jsonl');collect(args)
    assert sha(tmp_path/'out/train.jsonl')==before
    manifest=json.loads((tmp_path/'out/candidate-manifest.json').read_text())
    assert manifest['candidate_target_reached'] and not manifest['teacher_labels_ready']
    assert manifest['source_game_cross_split_overlap']==0
