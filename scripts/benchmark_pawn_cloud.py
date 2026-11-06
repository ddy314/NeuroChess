"""Bounded PAWN CPU conversion benchmark; never starts model training."""
import concurrent.futures
import functools
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import platform
import shutil
import sqlite3
import time

import chess
import numpy as np

from neurochess.data import DTYPE, record
from neurochess.targets import assign_targets, score_targets


@functools.lru_cache(maxsize=65536)
def raw_q(cp):
    return score_targets({'cp': int(cp)}, True)[1]


def convert_job(job):
    games, vocabulary, mode = job
    rows = []
    counts = dict(positions=0, invalid=0, q_labels=0)
    for game in games:
        if len(game['uci']) != len(game['cp_evals']) or len(game['uci']) != len(game['nnue_evals']):
            raise ValueError('Per-ply list length mismatch')
        board = chess.Board()
        for uci, search, raw in zip(game['uci'], game['cp_evals'], game['nnue_evals']):
            counts['positions'] += 1
            # The current model masks history. Keep clocks but do not repeatedly
            # scan full game history to compute features which it will discard.
            row, legal = record(board)
            pvs = [dict(line=vocabulary[str(e['move_idx'])],
                        cp=int(e['score_cp']) * (1 if board.turn else -1)) for e in search]
            if not pvs:
                counts['invalid'] += 1
            else:
                assign_targets(row, board, legal, pvs)
                row['meta'][[5, 6, 7, 10]] = 0
                if mode == 'raw_nnue_q':
                    row['q'][:] = 0
                    row['qmask'][:] = False
                    lookup = {m.uci(): i for i, m in enumerate(legal)}
                    for e in raw:
                        i = lookup[vocabulary[str(e['move_idx'])]]
                        if row['qmask'][i]:
                            raise ValueError('Duplicate raw NNUE move')
                        row['q'][i] = raw_q(e['score_cp'])
                        row['qmask'][i] = True
                    if int(row['qmask'].sum()) != len(legal):
                        raise ValueError('Raw NNUE does not cover every legal move')
                if not np.isfinite(row['value']).all() or abs(float(row['policy'].sum()) - 1) > .002:
                    raise ValueError('Invalid targets')
                counts['q_labels'] += int(row['qmask'].sum())
                rows.append(row)
            board.push_uci(uci)
            board.clear_stack()
    return np.array(rows, dtype=DTYPE), counts


def run_conversion(games, vocabulary, prior, mode, workers, output):
    root = output / f'{mode}-{workers}workers'
    root.mkdir()
    db = sqlite3.connect(root / 'dedup.sqlite')
    db.execute('PRAGMA journal_mode=WAL')
    db.execute('CREATE TABLE seen (key BLOB PRIMARY KEY) WITHOUT ROWID')
    prior_keys = prior.view('V16').reshape(-1)
    totals = dict(positions=0, invalid=0, q_labels=0, prior_overlap=0, duplicates=0, retained=0)
    jobs = [(games[i:i+16], vocabulary, mode) for i in range(0, len(games), 16)]
    started = time.perf_counter()
    if workers == 1:
        executor = None
        results = map(convert_job, jobs)
    else:
        executor = concurrent.futures.ProcessPoolExecutor(max_workers=workers,
                    mp_context=multiprocessing.get_context('spawn'))
        results = executor.map(convert_job, jobs)
    for shard, (rows, counts) in enumerate(results):
        for k, value in counts.items():
            totals[k] += value
        keys = rows['key'].copy().view('V16').reshape(-1)
        ix = np.searchsorted(prior_keys, keys)
        overlap = (ix < len(prior_keys)) & (prior_keys[np.minimum(ix, len(prior_keys)-1)] == keys)
        totals['prior_overlap'] += int(overlap.sum())
        keep = []
        for j in np.flatnonzero(~overlap):
            if db.execute('INSERT OR IGNORE INTO seen VALUES (?)', (bytes(keys[j]),)).rowcount:
                keep.append(j)
            else:
                totals['duplicates'] += 1
        db.commit()
        np.save(root / f'train-{shard:06d}.npy', rows[keep])
        totals['retained'] += len(keep)
    if executor:
        executor.shutdown()
    db.close()
    seconds = time.perf_counter() - started
    npy_bytes = sum(p.stat().st_size for p in root.glob('*.npy'))
    report = dict(mode=mode, workers=workers, games=len(games), seconds=seconds,
                  scanned_positions_per_second=totals['positions']/seconds,
                  retained_positions_per_second=totals['retained']/seconds,
                  npy_bytes=npy_bytes, sqlite_bytes=(root/'dedup.sqlite').stat().st_size,
                  **totals)
    print('CONVERSION', json.dumps(report), flush=True)
    return report


def main():
    from huggingface_hub import HfFileSystem
    import pyarrow.parquet as pq
    import requests
    source = json.loads(Path('/content/pawn-source.json').read_text())
    output = Path('/content/pawn-speed')
    output.mkdir(exist_ok=False)
    vocab_response = requests.get('https://raw.githubusercontent.com/thomas-schweich/PAWN/main/searchless_chess_vocabulary.json', timeout=60)
    vocab_response.raise_for_status()
    vocabulary = vocab_response.json()['action_to_move']
    prior = np.load('/content/pawn-prior-keys.npy', mmap_mode='r', allow_pickle=False)
    if prior.dtype != np.dtype('V16') or not np.array_equal(prior, np.sort(prior)):
        raise ValueError('Expected sorted exact prior-key index')
    fs = HfFileSystem()
    games = []
    reads = []
    columns = ['uci','game_length','cp_evals','nnue_evals','global_game_index']
    # Bounded sample across three separate source shards, at most384 games.
    for index in [0, len(source['files'])//2, len(source['files'])-1]:
        path = source['files'][index]
        remote = f"datasets/thomas-schweich/pawn-stockfish-100m@{source['revision']}/{path}"
        started = time.perf_counter()
        transferred = [0]
        with fs.open(remote, 'rb', block_size=1024*1024) as stream:
            fetch = stream._fetch_range
            def counted(start, end):
                data = fetch(start, end)
                transferred[0] += len(data)
                return data
            stream._fetch_range = counted
            stream.cache.fetcher = counted
            parquet = pq.ParquetFile(stream)
            batch = next(parquet.iter_batches(batch_size=128, row_groups=[0], columns=columns))
            batch_games = batch.to_pylist()
            seconds = time.perf_counter()-started
            info = dict(path=path, games=len(batch_games), positions=sum(g['game_length'] for g in batch_games),
                        seconds=seconds, range_payload_bytes=transferred[0], file_bytes=stream.size,
                        row_group_games=parquet.metadata.row_group(0).num_rows)
            reads.append(info)
            games.extend(batch_games)
            print('READ',json.dumps(info),flush=True)
    reports = []
    for mode, workers in [('search_q',1),('search_q',min(8,os.cpu_count())),('raw_nnue_q',min(8,os.cpu_count()))]:
        reports.append(run_conversion(games,vocabulary,prior,mode,workers,output))
    report = dict(dataset='thomas-schweich/pawn-stockfish-100m', tier='nodes_1024', split='train',
                  source_revision=source['revision'], vocabulary_sha256=hashlib.sha256(vocab_response.content).hexdigest(),
                  hardware=dict(cpu_count=os.cpu_count(),cpu_model=platform.processor(),
                                cpuinfo=Path('/proc/cpuinfo').read_text().split('model name')[1].split('\n')[0],
                                meminfo=Path('/proc/meminfo').read_text().splitlines()[:3],
                                disk=shutil.disk_usage('/content')._asdict()),
                  prior_keys=len(prior),prior_sha256=hashlib.file_digest(open('/content/pawn-prior-keys.npy','rb'),'sha256').hexdigest(),
                  dtype_bytes=DTYPE.itemsize, reads=reads, conversions=reports,
                  scope='Speed pilot only. No model training. Train split only. Exact exclusion against known prior positions; not a complete future heldout-isolation audit.',
                  target_semantics=dict(policy='search top5 softmax T100cp',value='best search action sf16/ply32 WDL proxy',
                     search_q='search action W-L proxy; unlabelled actions masked',
                     raw_nnue_q='all legal raw NNUE W-L proxy; separate experimental supervision, not merged with search Q',
                     mate='int16 score_cp interpreted as clipped centipawn proxy; no invented mate distances'),
                  limitations=['384-game prefix sample across3 shards, not a whole-corpus throughput guarantee',
                               '8.23M-key prior index; SQLite index in this pilot is much smaller than a150M-row index',
                               'Network/row-group amplification is measured separately; serial reads and conversion are not overlapped'])
    (output/'summary.json').write_text(json.dumps(report,indent=2))
    print('SUMMARY',json.dumps(report),flush=True)


if __name__ == '__main__':
    main()
