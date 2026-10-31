"""CPU-parallel Stockfish MultiPV annotation. GPU is not needed for this stage."""
import argparse
from multiprocessing.util import Finalize
import concurrent.futures
import hashlib
import json
from itertools import islice
import os
import chess
import chess.engine
import numpy as np
from .data import ShardWriter, record
from .encoding import split_for
from .targets import assign_targets, TARGET_SPEC

_ENGINE = None


def init_engine(path, threads, hash_mb):
    global _ENGINE
    _ENGINE = chess.engine.SimpleEngine.popen_uci(path)
    _ENGINE.configure({'Threads': threads, 'Hash': hash_mb})
    Finalize(None, _ENGINE.quit, exitpriority=10)


def annotate(task):
    fen, nodes, multipv, temperature = task
    board = chess.Board(fen)
    if not board.is_valid() or board.is_game_over():
        return None
    row, legal = record(board)
    # FEN cannot reconstruct repetition history. Keep these features explicitly zero.
    infos = _ENGINE.analyse(board, chess.engine.Limit(nodes=nodes), multipv=min(multipv, len(legal)), game=object())
    pvs=[]
    for info in infos:
        score=info['score'].white()
        entry=dict(line=info['pv'][0].uci())
        entry['mate' if score.is_mate() else 'cp']=score.mate() if score.is_mate() else score.score()
        pvs.append(entry)
    assign_targets(row,board,legal,pvs,temperature)
    return split_for(board), row


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--fens', required=True, help='One standard FEN per line')
    p.add_argument('--stockfish', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) // 2))
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--hash-mb', type=int, default=64)
    p.add_argument('--nodes', type=int, default=10000)
    p.add_argument('--multipv', type=int, default=8)
    p.add_argument('--temperature-cp', dest='temperature', type=float, default=100)
    a = p.parse_args()
    if min(a.nodes, a.workers, a.threads, a.multipv, a.hash_mb) < 1 or a.temperature <= 0:
        p.error('positive resource limits and temperature required')
    writer = ShardWriter(a.out)
    with chess.engine.SimpleEngine.popen_uci(a.stockfish) as engine:
        engine_id = engine.id
    digest = hashlib.sha256()
    with open(a.stockfish, 'rb') as binary:
        for chunk in iter(lambda: binary.read(1024 * 1024), b''):
            digest.update(chunk)
    with open(a.fens) as f, concurrent.futures.ProcessPoolExecutor(max_workers=a.workers,
            initializer=init_engine, initargs=(a.stockfish, a.threads, a.hash_mb)) as pool:
        # Bounded submissions: Executor.map eagerly consumes generators on older Python.
        while True:
            lines = list(islice(f, a.workers * 16))
            if not lines:
                break
            tasks = [(line.strip(), a.nodes, a.multipv, a.temperature) for line in lines if line.strip()]
            for result in pool.map(annotate, tasks, chunksize=4):
                if result is not None:
                    writer.add(*result)
            print(json.dumps(writer.counts), flush=True)
    writer.close(dict(kind='stockfish-multipv', engine=engine_id, binary_sha256=digest.hexdigest(),
        args=vars(a), target_spec=TARGET_SPEC, wdl_conversion='python-chess sf16, fixed ply32', policy='softmax over annotated candidates; others zero',
        history='FEN only, repetition unavailable', split='position-hash-1-percent'))


if __name__ == '__main__':
    main()
