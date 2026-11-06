import sqlite3
import chess
import numpy as np
import pytest
from neurochess.data import DTYPE
from neurochess.pawn_data import (convert_games, keys_from_games, contains,
    exact_filter, publish_array, load_published)


def fixture():
    board=chess.Board();vocabulary={};raw=[];search=[]
    moves=['e2e4','e7e5']
    for uci in moves:
        entries=[];chosen=None
        for move in board.legal_moves:
            key=str(len(vocabulary));vocabulary[key]=move.uci()
            entries.append(dict(move_idx=int(key),score_cp=-100))
            if move.uci()==uci:chosen=int(key)
        search.append([dict(move_idx=chosen,score_cp=100)])
        raw.append(entries);board.push_uci(uci)
    game=dict(uci=moves,game_length=2,cp_evals=search,nnue_evals=raw,global_game_index=17,nodes=1024)
    return game,vocabulary


def test_mover_pov_and_separate_dense_labels():
    game,vocab=fixture();rows,raw,groups,plies,counts=convert_games(([game],vocab))
    assert (rows['q'][rows['qmask']]>0).all()  # Both white and black, same mover score.
    assert (raw['q'][raw['qmask']]<0).all()
    assert rows['qmask'].sum()==2 and raw['qmask'].sum()==40
    assert np.array_equal(groups,[17,17]) and np.array_equal(plies,[0,1])
    assert not rows['meta'][:,[5,6,7,10]].any()
    assert np.array_equal(rows['key'].copy().view('V16').reshape(-1),keys_from_games([game]))


def test_exact_exclusions_duplicates_and_target_cap():
    game,vocab=fixture();rows=convert_games(([game],vocab))[0]
    keys=rows['key'].copy().view('V16').reshape(-1)
    db=sqlite3.connect(':memory:');db.execute('CREATE TABLE seen(key BLOB PRIMARY KEY) WITHOUT ROWID')
    rows=np.concatenate([rows,rows])
    kept,counts=exact_filter(rows,np.sort(keys[:1]),np.array([],dtype='V16'),db,10)
    assert kept.tolist()==[1] and counts==dict(prior_overlap=2,heldout_overlap=0,duplicates=1)
    db.rollback()
    kept,counts=exact_filter(rows,np.array([],dtype='V16'),np.sort(keys[1:]),db,1)
    assert kept.tolist()==[0] and counts['heldout_overlap']==2
    assert contains(np.array([],dtype='V16'),keys).tolist()==[False,False]


def test_delivery_roundtrip_and_corruption(tmp_path):
    game,vocab=fixture();rows=convert_games(([game],vocab))[0]
    meta=publish_array(rows,tmp_path/'local',tmp_path/'persistent','train-000000.npy')
    result=load_published(meta,tmp_path/'persistent',tmp_path/'cache')
    assert result.dtype==DTYPE and np.array_equal(result,rows)
    del result
    (tmp_path/'cache'/'train-000000.npy').unlink()
    (tmp_path/'persistent'/meta['name']).write_bytes(b'corrupt')
    with pytest.raises(OSError,match='checksum'):
        load_published(meta,tmp_path/'persistent',tmp_path/'cache')


def test_reject_malformed_annotation():
    game,vocab=fixture();game['nnue_evals'][1].pop()
    with pytest.raises(ValueError,match='Incomplete'):
        convert_games(([game],vocab))


def test_stream_carries_partial_batches_across_shards(tmp_path):
    from neurochess.pawn_data import write_json
    from scripts.train_pawn_cloud import ShardStream
    game,vocab=fixture();rows=convert_games(([game],vocab))[0]
    source=tmp_path/'source';local=tmp_path/'local'
    a=publish_array(rows,local,source,'train-000000.npy')
    b=publish_array(rows,local,source,'train-000001.npy')
    write_json(source/'ready.json',dict(shards=[dict(data=a),dict(data=b)],complete=True,heldout_isolation_complete=True))
    stream=ShardStream(source,tmp_path/'consumer-cache')
    assert len(stream.next(3))==3 and len(stream.next(1))==1 and stream.consumed==4
    with pytest.raises(EOFError):stream.next(1)


def test_validation_gate_numpy_scalars_are_saved(tmp_path):
    import json
    from neurochess.pawn_data import write_json
    path=tmp_path/'gate.json'
    write_json(path,dict(passed=True,criteria=dict(loss=np.float64(2.3)<=2.4),
                        updates=np.int64(100),value=np.float32(.25)))
    assert json.loads(path.read_text())==dict(passed=True,criteria=dict(loss=True),updates=100,value=.25)


def test_old_replay_excludes_new_heldout_keys(tmp_path):
    from scripts.train_pawn_cloud import ReplayRows
    game,vocab=fixture();rows=convert_games(([game],vocab))[0]
    np.save(tmp_path/'train-000000.npy',rows,allow_pickle=False)
    excluded=rows['key'][:1].copy().view('V16').reshape(-1)
    replay=ReplayRows(tmp_path,excluded)
    sampled=replay.sample(20)
    assert replay.total==1 and not contains(excluded,sampled['key'].copy().view('V16').reshape(-1)).any()
