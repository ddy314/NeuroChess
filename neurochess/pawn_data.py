"""PAWN game replay, exact exclusions and atomic, checksummed shard delivery."""
from functools import lru_cache
import hashlib
import io
import json
from pathlib import Path
import shutil

import chess
import numpy as np
import zstandard as zstd

from .data import DTYPE, MAX_MOVES, record
from .encoding import position_key
from .targets import TARGET_SPEC, assign_targets, score_targets

SOURCE = 'thomas-schweich/pawn-stockfish-100m'
REVISION = '7836c4977e5cb878c0d50d559d9662bdf0004598'
VOCABULARY_SHA256 = '17487dbe9e336c2be8ff1c51678b4f909747fa4d03fad3b6a51c78d1a263707f'
RAW_DTYPE = np.dtype([('q', '<f2', (MAX_MOVES,)), ('qmask', '?', (MAX_MOVES,))])


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    def scalar(obj):
        if isinstance(obj, np.generic):
            return obj.item()
        raise TypeError(f'Unsupported JSON value: {type(obj).__name__}')
    temporary.write_text(json.dumps(value, indent=2, default=scalar))
    temporary.replace(path)


def contains(sorted_keys, keys):
    if not len(sorted_keys):
        return np.zeros(len(keys), dtype=bool)
    ix = np.searchsorted(sorted_keys, keys)
    return (ix < len(sorted_keys)) & (sorted_keys[np.minimum(ix, len(sorted_keys)-1)] == keys)


def keys_from_games(games):
    keys = []
    for game in games:
        board = chess.Board()
        for uci in game['uci']:
            keys.append(position_key(board))
            board.push_uci(uci)
            board.clear_stack()
    return np.array(keys, dtype='V16')


@lru_cache(maxsize=65536)
def raw_q(cp):
    return score_targets({'cp': int(cp)}, True)[1]


def convert_games(job):
    games, vocabulary = job
    rows, raw_rows, groups, plies = [], [], [], []
    scanned = skipped = 0
    for game in games:
        n = len(game['uci'])
        if n != game['game_length'] or n != len(game['cp_evals']) or n != len(game['nnue_evals']):
            raise ValueError('PAWN per-ply array length mismatch')
        if game.get('nodes', 1024) != 1024:
            raise ValueError('Expected nodes_1024 tier')
        board = chess.Board()
        for ply, (uci, search, raw) in enumerate(zip(game['uci'], game['cp_evals'], game['nnue_evals'])):
            scanned += 1
            row, legal = record(board)
            if search:
                # PAWN scores are already mover-POV. The existing target helper
                # accepts white-POV, so adapt exactly once (including black).
                pvs = [dict(line=vocabulary[str(e['move_idx'])],
                            cp=int(e['score_cp']) * (1 if board.turn else -1)) for e in search]
                assign_targets(row, board, legal, pvs)
                row['meta'][[5, 6, 7, 10]] = 0
                dense = np.zeros((), dtype=RAW_DTYPE)
                lookup = {m.uci(): i for i, m in enumerate(legal)}
                for e in raw:
                    i = lookup[vocabulary[str(e['move_idx'])]]
                    if dense['qmask'][i]:
                        raise ValueError('Duplicate NNUE action')
                    dense['q'][i] = raw_q(e['score_cp'])
                    dense['qmask'][i] = True
                if int(dense['qmask'].sum()) != len(legal):
                    raise ValueError('Incomplete all-legal NNUE annotation')
                if not np.isfinite(row['value']).all() or abs(float(row['policy'].sum())-1) > .002:
                    raise ValueError('Invalid normalized target')
                rows.append(row); raw_rows.append(dense)
                groups.append(int(game['global_game_index'])); plies.append(ply)
            else:
                skipped += 1
            board.push_uci(uci)
            board.clear_stack()
    return (np.array(rows, dtype=DTYPE), np.array(raw_rows, dtype=RAW_DTYPE),
            np.array(groups, dtype=np.uint64), np.array(plies, dtype=np.uint16),
            dict(scanned=scanned, unlabelled=skipped))


def exact_filter(rows, prior, heldout, db, maximum):
    """Stable first occurrence; reserve keys in the caller's current transaction."""
    keys = rows['key'].copy().view('V16').reshape(-1)
    old = contains(prior, keys)
    reserved = contains(heldout, keys)
    kept = []
    duplicates = 0
    for i in np.flatnonzero(~(old | reserved)):
        if len(kept) >= maximum:
            break
        if db.execute('INSERT OR IGNORE INTO seen VALUES (?)', (bytes(keys[i]),)).rowcount:
            kept.append(i)
        else:
            duplicates += 1
    return np.asarray(kept, dtype=np.int64), dict(prior_overlap=int(old.sum()),
                    heldout_overlap=int(reserved.sum()), duplicates=duplicates)


def publish_array(array, local, destination, name):
    """Publish only a complete, read-back-verified compressed NPY file."""
    local, destination = Path(local), Path(destination)
    local.mkdir(parents=True, exist_ok=True); destination.mkdir(parents=True, exist_ok=True)
    npy = local / name
    np.save(npy, array, allow_pickle=False)
    compressed = npy.with_suffix(npy.suffix+'.zst')
    with npy.open('rb') as src, compressed.open('wb') as dst:
        zstd.ZstdCompressor(level=3).copy_stream(src, dst)
    digest = sha(compressed)
    target = destination / compressed.name
    temporary = target.with_name(target.name+'.tmp')
    shutil.copyfile(compressed, temporary)
    if sha(temporary) != digest:
        raise OSError('Persistent shard readback failed')
    temporary.replace(target)
    metadata = dict(name=target.name, rows=len(array), bytes=target.stat().st_size,
                    sha256=digest, npy_sha256=sha(npy), npy_bytes=npy.stat().st_size)
    npy.unlink(); compressed.unlink()
    return metadata


def load_published(metadata, source, cache):
    source, cache = Path(source), Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / metadata['name'].removesuffix('.zst')
    if target.exists() and sha(target) == metadata['npy_sha256']:
        return np.load(target, mmap_mode='r', allow_pickle=False)
    compressed = cache / metadata['name']
    shutil.copyfile(source / metadata['name'], compressed)
    if sha(compressed) != metadata['sha256']:
        raise OSError('Consumer shard checksum mismatch')
    temporary = target.with_name(target.name+'.tmp')
    with compressed.open('rb') as src, temporary.open('wb') as dst:
        zstd.ZstdDecompressor().copy_stream(src, dst)
    if sha(temporary) != metadata['npy_sha256']:
        raise OSError('Decompressed shard checksum mismatch')
    temporary.replace(target); compressed.unlink()
    array = np.load(target, mmap_mode='r', allow_pickle=False)
    if len(array) != metadata['rows'] or array.dtype != DTYPE:
        raise ValueError('Consumer shard schema/count mismatch')
    return array


PROVENANCE = dict(source=SOURCE, revision=REVISION, tier='nodes_1024', license='CC-BY-4.0',
    attribution='Thomas Schweich; PAWN; https://github.com/thomas-schweich/PAWN',
    target_spec=TARGET_SPEC, score_perspective='mover -> white -> existing mover target helper',
    q='search top5 only in training rows; full-legal raw NNUE retained in aligned separate sidecar',
    raw_q='raw NNUE W-L proxy, never silently substituted for search Q',
    mate='int16 normalized score_cp treated as a clipped centipawn proxy; no fabricated mate distance',
    history='history-free features; clocks/repetition explicitly masked',
    split='official whole-game splits AND exact exclusion of every official val/test position plus known prior keys')
