"""Check labels, legal padding, and exact-position train/validation leakage."""
import argparse
import json
from pathlib import Path
import sqlite3
import tempfile
import time
from .telemetry import write_status
import numpy as np
from .data import DTYPE


def audit(root, status_dir=None):
    root = Path(root)
    counts = {}
    duplicates = 0
    last_update = 0.
    with tempfile.TemporaryDirectory() as tmp:
        db = sqlite3.connect(Path(tmp) / 'keys.db')
        db.execute('CREATE TABLE positions (key BLOB PRIMARY KEY, split TEXT NOT NULL)')
        for split in ['train', 'val']:
            count = 0
            for path in sorted(root.glob(f'{split}-*.npy')):
                rows = np.load(path, mmap_mode='r', allow_pickle=False)
                if rows.dtype != DTYPE:
                    raise ValueError(f'Wrong schema: {path}')
                for start in range(0, len(rows), 4096):
                    b = rows[start:start + 4096]
                    legal = b['moves'] >= 0
                    assert np.all(legal.sum(-1) > 0)
                    assert np.all((b['moves'] >= -1) & (b['moves'] < 20480))
                    assert np.all(np.isfinite(b['meta'])) and np.all(np.isfinite(b['policy']))
                    assert np.all(b['policy'] >= 0) and np.allclose(b['policy'].sum(-1), 1, atol=.002)
                    assert np.all(b['policy'][~legal] == 0) and not b['qmask'][~legal].any()
                    assert np.all(np.isfinite(b['q'])) and np.all(np.abs(b['q']) <= 1)
                    assert np.all(np.isfinite(b['value'])) and np.all(b['value'] >= 0)
                    assert np.allclose(b['value'].sum(-1)[b['value_weight'] > 0], 1, atol=.002)
                    for r in b:
                        ids = r['moves'][r['moves'] >= 0]
                        assert len(ids) == len(set(ids.tolist()))
                        key = r['key'].tobytes()
                        previous = db.execute('SELECT split FROM positions WHERE key=?', (key,)).fetchone()
                        if previous:
                            if previous[0] != split:
                                raise ValueError('Exact-position leakage across splits')
                            duplicates += 1
                        else:
                            db.execute('INSERT INTO positions VALUES (?,?)', (key, split))
                    count += len(b)
                    if status_dir and time.time() - last_update >= 2:
                        checked = sum(counts.values()) + count
                        write_status(status_dir, 'auditing', audited_positions=checked, note=f'已检查 {checked} 个局面')
                        last_update = time.time()
                db.commit()
            counts[split] = count
        if counts != json.loads((root / 'manifest.json').read_text())['counts']:
            raise ValueError('Manifest counts differ from shards')
    return dict(counts=counts, within_split_duplicates=duplicates, exact_position_leakage=False)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--data', required=True)
    p.add_argument('--status-dir')
    a = p.parse_args(); print(json.dumps(audit(a.data, a.status_dir)))


if __name__ == '__main__':
    main()
