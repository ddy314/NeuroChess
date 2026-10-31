import json
import os
from pathlib import Path
import random
import subprocess
import sys
import chess
import pytest
from neurochess.data import ShardWriter, record
from neurochess.encoding import split_for
from neurochess.job import run_job


def dataset(path):
    rng = random.Random(52); board = chess.Board()
    writer = ShardWriter(path, size=128)
    while writer.counts['train'] < 32 or writer.counts['val'] < 2:
        if board.is_game_over():
            board = chess.Board()
        row, moves = record(board)
        row['policy'][0] = 1; row['value'][1] = 1; row['value_weight'] = 1
        writer.add(split_for(board), row)
        board.push(rng.choice(moves))
    writer.close({'kind':'fixture'})


def test_cache_restore_preparation_only(tmp_path):
    dataset(tmp_path/'data')
    cache = tmp_path/'cache'
    spec = dict(args=['--data',str(tmp_path/'data'),'--out',str(tmp_path/'prepare')],
                data_cache=str(cache), prepare_only=True, arena=False)
    run_job(spec)
    assert (cache/'manifest.json').exists()
    spec['args']=['--data',str(tmp_path/'restored'),'--out',str(tmp_path/'restore-run')]
    run_job(spec)
    assert json.loads((tmp_path/'restore-run/status.json').read_text())['phase']=='prepared'
    assert (tmp_path/'restored/manifest.json').read_bytes()==(cache/'manifest.json').read_bytes()


def test_periodic_arena_and_resume(tmp_path):
    engine=Path('local/stockfish/stockfish/stockfish-ubuntu-x86-64-avx2').resolve()
    if not engine.exists():
        pytest.skip('local reference engine not available')
    dataset(tmp_path/'data')
    openings=tmp_path/'openings.fen';openings.write_text(chess.STARTING_FEN+'\n')
    run=tmp_path/'run'
    command=[sys.executable,'-m','neurochess.train','--data',str(tmp_path/'data'),'--out',str(run),
             '--config','configs/tiny.json','--steps','2','--batch-size','8','--accum','1','--workers','0',
             '--log-every','1','--arena-every','1','--arena-stockfish',str(engine),
             '--arena-openings',str(openings),'--arena-pairs','1','--arena-nodes','100',
             '--arena-max-seconds','0.001','--warmup','1']
    env={**os.environ,'OMP_NUM_THREADS':'2'}
    subprocess.run(command+['--stop-after','1'],check=True,env=env,capture_output=True)
    assert json.loads((run/'status.json').read_text())['phase']=='paused'
    subprocess.run(command+['--resume',str(run/'last.pt')],check=True,env=env,capture_output=True)
    assert json.loads((run/'status.json').read_text())['phase']=='complete'
    rows=[json.loads(s) for s in (run/'arena.jsonl').read_text().splitlines()]
    assert [r['step'] for r in rows]==[1,2]
    for row in rows:
        assert row['completed']+row['truncated']+row['unplayed']==2
        assert row['completed']==row['draws']==0
        assert (run/row['pgn']).exists()
