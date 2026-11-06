import copy
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import random
import struct

import chess
import numpy as np
import pytest
import zstandard as zstd

from neurochess.lichess_clean import (Rejected, strict_convert, split_key,
                                     audit_rows, convert_batch)
from neurochess.encoding import move_id
from scripts import prepare_lichess_cloud as cloud


def entry(board=None):
    board=board or chess.Board()
    moves=list(board.legal_moves)[:2];sign=1 if board.turn else -1
    return dict(fen=' '.join(board.fen().split()[:4]),evals=[dict(depth=24,knodes=1000,
        pvs=[dict(cp=50*sign,line=moves[0].uci()),dict(cp=-20*sign,line=moves[1].uci())])])


def test_thresholds_one_coherent_search_and_history_mask():
    e=entry();row,proof=strict_convert(e)
    assert row['qmask'].sum()==2 and not row['meta'][[5,6,7,10]].any()
    audit_rows(np.array([row],dtype=row.dtype))
    bad=copy.deepcopy(e);bad['evals']=[dict(bad['evals'][0],depth=23),dict(bad['evals'][0],knodes=999)]
    with pytest.raises(Rejected,match='below_depth'):strict_convert(bad)
    bad=copy.deepcopy(e);bad['evals'][0]['pvs']=bad['evals'][0]['pvs'][:1]
    with pytest.raises(Rejected,match='multipv'):strict_convert(bad)


def test_black_signs_and_uci960_castling():
    b=chess.Board();b.push_uci('e2e4');e=entry(b);row,_=strict_convert(e)
    ix=row['moves'].tolist().index(move_id(chess.Move.from_uci(e['evals'][0]['pvs'][0]['line']),b.turn))
    assert row['q'][ix]>0 and row['value'][0]>row['value'][2]
    e=entry(chess.Board('r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1'))
    e['evals'][0]['pvs']=[dict(cp=50,line='e1h1 e8a8'),dict(cp=-20,line='e1a1 e8h8')]
    row,_=strict_convert(e);assert row['qmask'].sum()==2


def test_full_pv_illegality_duplicate_roots_and_invalid_scores():
    e=entry();e['evals'][0]['pvs'][0]['line']='g1h3 e2e4'
    with pytest.raises(Rejected,match='illegal_pv'):strict_convert(e)
    e=entry();e['evals'][0]['pvs'][1]['line']=e['evals'][0]['pvs'][0]['line']
    with pytest.raises(Rejected,match='duplicate_pv'):strict_convert(e)
    e=entry();e['evals'][0]['pvs'][0]['mate']=3
    with pytest.raises(Rejected,match='malformed_score'):strict_convert(e)
    e=entry();e['evals'][0]['pvs'][0]['line']='0000'
    with pytest.raises(Rejected,match='null_move'):strict_convert(e)
    e=entry();e['evals'][0]['pvs'][0]['cp']=True
    with pytest.raises(Rejected,match='malformed_score'):strict_convert(e)


def test_global_reference_contradiction_and_near_tie():
    e=entry();e['evals'].append(dict(depth=30,knodes=5000,pvs=[dict(cp=500,line='e2e4')]))
    with pytest.raises(Rejected,match='global_'):strict_convert(e)
    e['evals'][-1]['pvs'][0]['cp']=55
    row,proof=strict_convert(e)
    assert row['qmask'].sum()==2 and proof['global_cp_gap']==5
    e['evals'][-1]['pvs'][0]=dict(mate=3,line='e2e4')
    with pytest.raises(Rejected):strict_convert(e)


def test_pzstd_skippable_metadata_and_partial_json_lines():
    raw=b''.join(json.dumps(dict(i=i)).encode()+b'\n' for i in range(30))
    cuts=[0,37,113,198,len(raw)];parts=[];starts=[]
    for lo,hi in zip(cuts,cuts[1:]):
        block=zstd.ZstdCompressor().compress(raw[lo:hi])
        parts.append(struct.pack('<III',0x184D2A50,4,len(block)));starts.append(sum(map(len,parts)))
        parts.append(block)
    blob=b''.join(parts)
    source=dict(url='fixture',etag='v1',bytes=len(blob))
    def fetch(url,etag,start,end):return blob[start:end+1]
    lines=list(cloud.source_lines(source,0,len(blob),fetch=fetch))
    assert [json.loads(line) for _,line in lines]==[dict(i=i) for i in range(30)]
    cropped=list(cloud.source_lines(source,1,starts[2],fetch=fetch))
    assert cropped and all(isinstance(json.loads(line)['i'],int) for _,line in cropped)
    # The final record crossing a stop is completed, without swallowing all
    # remaining rows in the next unselected frame.
    assert len(cropped)<30


def test_exact_quotas_and_hash_split():
    for total in [1,100000,9800000]:
        assert sum(cloud.quota(total,i,128) for i in range(128))==total
    assert split_key(bytes(16))=='val'
    assert split_key((1).to_bytes(16,'little'))=='test'
    assert split_key((2).to_bytes(16,'little'))=='train'


@pytest.mark.parametrize('empty_first', [False, True])
def test_committed_shards_resume_after_pair_publication_failure(tmp_path,monkeypatch,empty_first):
    work=tmp_path/'work';save=tmp_path/'save';save.mkdir();(save/'source.tar.gz').write_bytes(b'fixture source')
    monkeypatch.setattr(cloud.os.path,'ismount',lambda p:True)
    empty=np.array([],dtype='V16')
    def exclusion(plan,work,save,status):
        np.save(save/'excluded-keys.npy',empty);np.save(work/'excluded-keys.npy',empty)
        cloud.write_json(save/'exclusion-audit.json',dict(keys=0))
        return empty
    monkeypatch.setattr(cloud,'build_exclusions',exclusion)
    class Response:
        headers={'ETag':'v1','Content-Length':'100000000'}
        def __enter__(self):return self
        def __exit__(self,*args):pass
    monkeypatch.setattr(cloud,'urlopen',lambda *a,**kw:Response())
    monkeypatch.setattr(cloud,'ProcessPoolExecutor',lambda max_workers,**kw:ThreadPoolExecutor(max_workers,initializer=kw.get('initializer'),initargs=kw.get('initargs',())))
    rng=random.Random(17);b=chess.Board();lines=[]
    for _ in range(3000):
        if b.is_game_over() or b.legal_moves.count()<2:b=chess.Board()
        lines.append(json.dumps(entry(b)).encode());b.push(rng.choice(list(b.legal_moves)))
    def fixture_lines(source,start,stop):
        return iter([]) if empty_first and stop<=50_000_000 else iter((12,l) for l in lines)
    monkeypatch.setattr(cloud,'source_lines',fixture_lines)
    plan=dict(work=str(work),save=str(save),source=dict(url='https://example.invalid/fixture',etag='v1',bytes=100000000),
              counts=dict(train=20,val=2,test=2),strata=2 if empty_first else 1,seed=7,workers=2,shard_rows=4,release_when_done=False)
    original=cloud.publish_evidence;calls=0
    def crash(*args):
        nonlocal calls
        calls+=1
        if calls==3:raise OSError('simulated interruption before ready commit')
        return original(*args)
    monkeypatch.setattr(cloud,'publish_evidence',crash)
    with pytest.raises(OSError,match='simulated'):cloud.run(plan)
    interrupted=json.loads((save/'ready.json').read_text())
    assert sum(interrupted['counts'].values())>0 and not interrupted['complete']
    immutable={x['data']['name']:cloud.sha(save/x['data']['name']) for x in interrupted['shards']}
    monkeypatch.setattr(cloud,'publish_evidence',original);cloud.run(plan)
    ready=json.loads((save/'ready.json').read_text())
    assert ready['complete'] and ready['counts']==plan['counts']
    assert all(cloud.sha(save/name)==digest for name,digest in immutable.items())
    audit=json.loads((save/'audit.json').read_text())
    assert audit['unique_positions']==24 and audit['duplicate_keys']==audit['cross_split_position_overlap']==0


def test_known_keys_rejected_before_replaying_pvs():
    from neurochess.encoding import position_key
    keys=np.array([position_key(chess.Board())],dtype='V16')
    e=entry();e['evals'][0]['pvs'][0]['line']='g1h3 e2e4'
    with pytest.raises(Rejected,match='prior_or_heldout_overlap'):
        strict_convert(e,keys)
