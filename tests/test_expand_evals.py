import json
import os
from pathlib import Path
import random
import subprocess
import sys

import chess
import zstandard

from neurochess.audit import audit
from neurochess.data import ShardWriter
from neurochess.encoding import position_key, split_for
from neurochess.import_evals import convert


def test_expansion_preserves_old_shards_and_reindexes_after_interruption(tmp_path):
    rng = random.Random(17)
    board = chess.Board(); entries = []; seen = set()
    while len(entries) < 250:
        if board.is_game_over(claim_draw=True): board = chess.Board()
        moves = list(board.legal_moves)
        if len(moves) >= 2 and position_key(board) not in seen:
            seen.add(position_key(board))
            entries.append(dict(fen=' '.join(board.fen().split()[:4]), evals=[dict(
                depth=24,knodes=100,pvs=[dict(cp=20,line=moves[0].uci()),dict(cp=-10,line=moves[1].uci())])]))
        board.push(rng.choice(moves))
    base = tmp_path/'base'; writer = ShardWriter(base,size=8)
    for entry in entries:
        b,row,_,_=convert(entry)[0]; split=split_for(b)
        if writer.counts[split] < (8 if split=='train' else 1): writer.add(split,row)
    writer.close({'fixture':True})
    assert writer.counts == {'train':8,'val':1}
    source=tmp_path/'source.zst'
    source.write_bytes(zstandard.ZstdCompressor().compress(''.join(json.dumps(x)+'\n' for x in entries).encode()))
    excluded=tmp_path/'exclude.fen';excluded.write_text('')
    out=tmp_path/'out'
    command=[sys.executable,'scripts/expand_evals.py','--base',str(base),'--out',str(out),
             '--workers','2','--source-command-json',json.dumps(['cat',str(source)]),
             '--source-url','https://example.invalid/source','--source-etag','fixture',
             '--source-bytes',str(source.stat().st_size),'--exclude-fens',str(excluded)]
    env={**os.environ,'OMP_NUM_THREADS':'2'}
    subprocess.run(command+['--train-positions','20'],check=True,capture_output=True,env=env)
    assert audit(out)['counts']['train']==20
    old={p.name:p.read_bytes() for p in out.glob('*.npy')}
    # A crash may leave closed shards without a published final manifest.
    (out/'manifest.json').unlink()
    subprocess.run(command+['--train-positions','35','--resume-existing'],check=True,capture_output=True,env=env)
    assert audit(out)['counts']['train']==35
    for name, content in old.items(): assert (out/name).read_bytes()==content
    manifest=json.loads((out/'manifest.json').read_text())
    assert manifest['provenance']['accepted']==sum(manifest['counts'].values())
    assert manifest['provenance']['expansion']['base_counts']=={'train':8,'val':1}
