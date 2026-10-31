"""Keep audited v1 shards and add distinct quality-filtered engine positions."""
import argparse
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor
import hashlib
import io
import itertools
import json
import multiprocessing
import os
from pathlib import Path
import sqlite3
import subprocess
import time

import numpy as np
import zstandard
from neurochess.data import DTYPE, ShardWriter
from neurochess.import_evals import convert, exclusion_keys


def convert_lines(lines):
    records, depths, pvs, reasons = [], [], [], Counter()
    for line in lines:
        try:
            result, reason = convert(json.loads(line))
            if result is None:
                reasons[reason] += 1
            else:
                _, row, depth, count = result
                records.append(row); depths.append(depth); pvs.append(count)
        except (ValueError, KeyError, TypeError, IndexError):
            reasons['malformed_or_illegal'] += 1
    return np.array(records, dtype=DTYPE), depths, pvs, reasons


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--train-positions', type=int, default=5_000_000)
    p.add_argument('--workers', type=int, default=6)
    p.add_argument('--source-command-json', required=True)
    p.add_argument('--source-url', required=True)
    p.add_argument('--source-etag', required=True)
    p.add_argument('--source-bytes', type=int, required=True)
    p.add_argument('--exclude-fens', nargs='+', required=True)
    p.add_argument('--resume-existing', action='store_true', help='Reindex closed shards after an interrupted import')
    a = p.parse_args()
    base = Path(a.base); root = Path(a.out)
    old_manifest = json.loads((base / 'manifest.json').read_text())
    assert a.train_positions > old_manifest['counts']['train']
    if a.resume_existing:
        if (root/'manifest.json').exists():raise ValueError('Output already has a complete manifest')
        writer = ShardWriter.__new__(ShardWriter)
        writer.root=root;writer.size=16384
        writer.buffers={s:[] for s in ['train','val']}
        writer.counts={s:0 for s in ['train','val']}
        writer.shards={s:0 for s in ['train','val']}
    else:
        writer = ShardWriter(root)
    db = sqlite3.connect(root / 'keys.sqlite')
    db.execute('PRAGMA journal_mode=WAL'); db.execute('PRAGMA synchronous=NORMAL')
    db.execute('DROP TABLE IF EXISTS positions')
    db.execute('CREATE TABLE positions (key BLOB PRIMARY KEY, split TEXT NOT NULL) WITHOUT ROWID')
    for split in ['train', 'val']:
        files = sorted((root if a.resume_existing else base).glob(f'{split}-*.npy'))
        for file in files:
            rows = np.load(file, mmap_mode='r', allow_pickle=False)
            if not a.resume_existing:os.link(file, root / file.name)
            db.executemany('INSERT INTO positions VALUES (?,?)', ((r.tobytes(), split) for r in rows['key']))
            writer.counts[split] += len(rows)
        writer.shards[split] = len(files)
    assert all(writer.counts[s]>=old_manifest['counts'][s] for s in ['train','val'])
    retained_counts = dict(writer.counts)
    db.commit()
    excluded = exclusion_keys(a.exclude_fens)
    reasons, depths, pvs, outcomes = Counter(), Counter(), Counter(), Counter()
    digest = hashlib.sha256(); scanned = 0
    added = sum(writer.counts.values())-sum(old_manifest['counts'].values())
    started = time.time(); last_log = 0
    command = json.loads(a.source_command_json)
    if not isinstance(command, list) or not all(isinstance(v, str) for v in command):
        raise ValueError('Source command must be an argv array')
    process = subprocess.Popen(command, stdout=subprocess.PIPE)
    pool = ProcessPoolExecutor(a.workers, mp_context=multiprocessing.get_context('fork'))
    stream = io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(
        process.stdout, read_across_frames=True), encoding='utf-8')
    pending = deque()
    def submit():
        lines = list(itertools.islice(stream, 512))
        if lines:
            pending.append((lines, pool.submit(convert_lines, lines)))
            return True
        return False
    try:
        for _ in range(a.workers * 2):
            if not submit(): break
        while pending and writer.counts['train'] < a.train_positions:
            lines, future = pending.popleft()
            scanned += len(lines)
            for line in lines: digest.update(line.encode())
            rows, row_depths, row_pvs, rejected = future.result(); reasons.update(rejected)
            for row, depth, count in zip(rows, row_depths, row_pvs):
                key = row['key'].tobytes()
                if key in excluded:
                    reasons['evaluation_opening'] += 1; continue
                split = 'val' if int.from_bytes(key[:8], 'little') % 100 == 0 else 'train'
                if db.execute('INSERT OR IGNORE INTO positions VALUES (?,?)', (key, split)).rowcount == 0:
                    reasons['duplicate_or_base_position'] += 1; continue
                writer.add(split, row); added += 1; depths[depth] += 1; pvs[count] += 1
                outcomes['win' if row['value'][0] > .75 else 'loss' if row['value'][2] > .75 else 'balanced'] += 1
                if writer.counts['train'] >= a.train_positions: break
            if time.time() - last_log > 15 or writer.counts['train'] >= a.train_positions:
                db.commit()
                info = dict(phase='expanding', counts=writer.counts, target_train=a.train_positions,
                            added=added, scanned=scanned, seconds=time.time()-started, rejected=dict(reasons))
                (root / 'progress.json').write_text(json.dumps(info, indent=2))
                print(json.dumps(info), flush=True); last_log = time.time()
            if writer.counts['train'] < a.train_positions: submit()
        if writer.counts['train'] < a.train_positions:
            raise RuntimeError(f'Source stopped early at {writer.counts}; output is incomplete')
        provenance = dict(kind='lichess-evals-expanded', base_provenance=old_manifest['provenance'],
            accepted=sum(writer.counts.values()), target_reached=True,
            min_depth=20, min_knodes=50, min_pvs=2, temperature_cp=100,
            split='position-hash-1-percent; compatible with phase1 and v1',
            expansion=dict(base_manifest_sha256=hashlib.sha256((base/'manifest.json').read_bytes()).hexdigest(),
            base_counts=old_manifest['counts'], source=a.source_url, source_etag=a.source_etag,
            source_bytes=a.source_bytes, transport='version-pinned checked byte ranges; retries; all zstd frames',
            added_unique_positions=added, retained_counts=retained_counts,
            scanned_prefix_lines=scanned, processed_batches_sha256=digest.hexdigest(),
            rejection_counts=dict(reasons), newly_converted_depth_histogram=dict(depths), newly_converted_pv_histogram=dict(pvs),
            newly_converted_outcomes=dict(outcomes), requested_train_positions=a.train_positions,
            target_reached=True, original_validation_preserved=True,
            sampling='extended filtered prefix, not a uniform full-database sample'))
        writer.close(provenance)
        (root / 'progress.json').write_text(json.dumps(dict(phase='prepared',counts=writer.counts,added=added,seconds=time.time()-started),indent=2))
        print(json.dumps(dict(phase='prepared',counts=writer.counts,added=added,seconds=time.time()-started)),flush=True)
    finally:
        stream.close()
        process.terminate()
        try: process.wait(timeout=15)
        except subprocess.TimeoutExpired: process.kill(); process.wait()
        pool.shutdown(wait=True, cancel_futures=True)
        db.commit(); db.close()


if __name__ == '__main__':
    main()
