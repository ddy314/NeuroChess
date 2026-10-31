import json
import os
import subprocess
import sys

import chess
import torch

from neurochess.data import ShardWriter
from neurochess.import_evals import convert


def test_new_data_stage_retains_optimizer_and_resumes_exactly(tmp_path):
    def dataset(name, train_count, val_count):
        root = tmp_path / name
        writer = ShardWriter(root, size=8)
        for split, count in [('train', train_count), ('val', val_count)]:
            for i in range(count):
                entry = dict(fen=' '.join(chess.STARTING_FEN.split()[:4]), evals=[dict(
                    depth=24, knodes=100, pvs=[dict(cp=i*10, line='e2e4'), dict(cp=-10, line='d2d4')])])
                writer.add(split, convert(entry)[0][1])
        writer.close({'fixture': name})
        return root

    old = dataset('old', 16, 3)
    new = dataset('new', 32, 7)
    env = {**os.environ, 'OMP_NUM_THREADS': '2'}
    common = [sys.executable, '-m', 'neurochess.train', '--device', 'cpu',
              '--config', 'configs/tiny.json', '--cpu-threads', '2', '--batch-size', '4',
              '--accum', '2', '--workers', '0', '--save-every', '1',
              '--eval-batches', '10', '--history-free', '--q-warmup', '0']
    parent = tmp_path / 'parent'
    subprocess.run(common + ['--data', str(old), '--out', str(parent), '--steps', '2'],
                   check=True, env=env, capture_output=True)
    full, resumed = tmp_path / 'full', tmp_path / 'resumed'
    stage = common + ['--data', str(new), '--validation-data', str(old), '--steps', '3']
    for out, limit in [(full, '2'), (resumed, '1')]:
        subprocess.run(stage + ['--out', str(out), '--continue-from', str(parent/'last.pt'),
                               '--stop-after', limit], check=True, env=env, capture_output=True)
    first = torch.load(resumed/'last.pt', weights_only=False)
    assert first['step'] == 1 and first['origin_step'] == 2
    assert first['lineage']['optimizer_retained']
    # Adam's moments must continue at update 3, rather than restart at update 1.
    assert {int(s['step']) for s in first['optimizer']['state'].values()} == {3}
    validation = [json.loads(line) for line in (resumed/'validation.jsonl').read_text().splitlines()]
    assert validation[-1]['positions'] == 3  # The fixed reference, not new's 7 rows.
    subprocess.run(stage + ['--out', str(resumed), '--resume', str(resumed/'last.pt'),
                            '--stop-after', '2'], check=True, env=env, capture_output=True)
    a = torch.load(full/'last.pt', weights_only=False)
    b = torch.load(resumed/'last.pt', weights_only=False)
    assert b['origin_step'] == 2 and b['step'] == 2
    assert {int(s['step']) for s in b['optimizer']['state'].values()} == {4}
    for name in a['model']:
        torch.testing.assert_close(a['model'][name], b['model'][name], rtol=0, atol=0)
    manifest = old / 'manifest.json'
    content = json.loads(manifest.read_text()); content['provenance']['changed'] = True
    manifest.write_text(json.dumps(content))
    failed = subprocess.run(stage + ['--out', str(resumed), '--resume', str(resumed/'last.pt')],
                            env=env, capture_output=True, text=True)
    assert failed.returncode != 0 and 'manifest differs' in failed.stderr
