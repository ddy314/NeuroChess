"""Versioned fixed-width mmap shards; no chess parsing in the GPU training loop."""
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info
from .encoding import MAX_MOVES, encode, legal_ids, position_key

DTYPE = np.dtype([
    ('tokens', 'u1', (64,)), ('meta', '<f4', (12,)),
    ('moves', '<i2', (MAX_MOVES,)), ('policy', '<f2', (MAX_MOVES,)),
    ('q', '<f2', (MAX_MOVES,)), ('qmask', '?', (MAX_MOVES,)),
    ('value', '<f4', (3,)), ('value_weight', '<f4'), ('key', 'u1', (16,))])
FIELDS = ['tokens', 'meta', 'moves', 'policy', 'q', 'qmask', 'value', 'value_weight']


def record(board):
    r = np.zeros((), dtype=DTYPE)
    r['tokens'], r['meta'] = encode(board)
    moves, ids = legal_ids(board)
    if not moves:
        raise ValueError('Terminal positions have no policy target')
    r['moves'] = -1
    r['moves'][:len(ids)] = ids
    r['key'] = np.frombuffer(position_key(board), dtype=np.uint8)
    return r, moves


class ShardWriter:
    def __init__(self, root, size=16384):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=False)
        self.size = size
        self.buffers = {s: [] for s in ('train', 'val')}
        self.counts = {s: 0 for s in self.buffers}
        self.shards = {s: 0 for s in self.buffers}

    def add(self, split, row):
        self.buffers[split].append(row)
        self.counts[split] += 1
        if len(self.buffers[split]) >= self.size:
            self.flush(split)

    def flush(self, split):
        if self.buffers[split]:
            np.save(self.root / f'{split}-{self.shards[split]:06d}.npy', np.array(self.buffers[split], dtype=DTYPE))
            self.shards[split] += 1
            self.buffers[split].clear()

    def close(self, provenance):
        for s in self.buffers:
            self.flush(s)
        (self.root / 'manifest.json').write_text(json.dumps(dict(schema=1, counts=self.counts,
            row_bytes=DTYPE.itemsize, provenance=provenance), indent=2))


class Batches(IterableDataset):
    def __init__(self, root, split, batch_size, seed=0, epoch=0):
        self.files = sorted(Path(root).glob(f'{split}-*.npy'))
        if not self.files:
            raise ValueError(f'No {split} shards in {root}')
        self.batch_size, self.seed, self.epoch = batch_size, seed, epoch
        self.shuffle = split == 'train'

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        files = list(self.files)
        if self.shuffle:
            rng.shuffle(files)
        worker = get_worker_info()
        if worker:
            files = files[worker.id::worker.num_workers]
        for file in files:
            rows = np.load(file, mmap_mode='r', allow_pickle=False)
            if rows.dtype != DTYPE:
                raise ValueError(f'Incompatible schema: {file}')
            order = rng.permutation(len(rows)) if self.shuffle else np.arange(len(rows))
            for start in range(0, len(rows), self.batch_size):
                batch = rows[order[start:start + self.batch_size]]
                yield {k: torch.from_numpy(np.array(batch[k], copy=True)) for k in FIELDS}


class ReplayBatches(IterableDataset):
    """Seven primary microbatches then one human replay microbatch by default.

    Deterministic per worker and epoch; the ordinary resume cursor still applies.
    Human replay supervises policy only: its game-result value labels differ from
    the engine's score-proxy labels. Primary validation never includes replay.
    """
    def __init__(self, primary, replay_root, every=8):
        if every < 2:
            raise ValueError('replay interval must be >=2')
        self.primary, self.replay_root, self.every = primary, replay_root, every

    def __iter__(self):
        generation = 0
        def replay_iter():
            return iter(Batches(self.replay_root,'train',self.primary.batch_size,
                self.primary.seed + 100000, self.primary.epoch + generation))
        replay = replay_iter()
        for i, batch in enumerate(self.primary):
            yield batch
            if (i+1) % (self.every-1) == 0:
                try:
                    extra = next(replay)
                except StopIteration:
                    generation += 1; replay = replay_iter()
                    try: extra = next(replay)
                    except StopIteration: continue  # Worker has no replay shard.
                extra['qmask'].zero_(); extra['value_weight'].zero_()
                yield extra


class MixtureBatches(IterableDataset):
    """Deterministic 7 core / 2 hard / 1 policy-only human microbatches."""
    def __init__(self,primary,hard_root,replay_root,core_batches=7,hard_batches=2,book_root=None,book_every=10):
        if min(core_batches,hard_batches,book_every)<1:raise ValueError('Positive mixture counts required')
        self.primary,self.hard_root,self.replay_root=primary,hard_root,replay_root
        self.core_batches,self.hard_batches=core_batches,hard_batches
        self.book_root,self.book_every=book_root,book_every

    def __iter__(self):
        def cycle(root,offset):
            generation=0
            while True:
                count=0
                for batch in Batches(root,'train',self.primary.batch_size,self.primary.seed+offset,self.primary.epoch+generation):
                    count+=1;yield batch
                if not count:raise ValueError('A mixture worker has no secondary shards')
                generation+=1
        hard=cycle(self.hard_root,200000);human=cycle(self.replay_root,100000)
        book=cycle(self.book_root,300000) if self.book_root else None
        replay_count=0
        for i,batch in enumerate(self.primary):
            yield batch
            if (i+1)%self.core_batches==0:
                for _ in range(self.hard_batches):yield next(hard)
                replay_count+=1
                replay=next(book if book is not None and replay_count%self.book_every==0 else human)
                replay['qmask'].zero_();replay['value_weight'].zero_();yield replay
