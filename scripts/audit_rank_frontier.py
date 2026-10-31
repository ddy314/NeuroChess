"""Check game separation, exact-key exclusion, history and sibling perspectives."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import chess
import numpy as np
from neurochess.encoding import encode,position_key
from neurochess.frontier import FRONTIER_DTYPE,ranking_metrics


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);a=p.parse_args();root=Path(a.data)
    manifest=json.loads((root/'manifest.json').read_text());roots=json.loads((root/'roots.json').read_text())
    jobs=json.loads((root/'jobs.json').read_text());lookup={(j['game'],j['node']):j for j in jobs}
    game_sets={s:{r['game'] for r in roots if r['split']==s} for s in ['train','val','eval']}
    for a,b in [('train','val'),('train','eval'),('val','eval')]:assert not game_sets[a]&game_sets[b]
    assert sum(map(len,game_sets.values()))==len(roots)
    keys={};counts={};samples=0
    for split in ['train','val']:
        rows=np.load(root/f'{split}.npy',mmap_mode='r');assert rows.dtype==FRONTIER_DTYPE
        assert set(rows['game']).issubset(game_sets[split]);assert np.isfinite(rows['cp']).all()
        assert np.isfinite(rows['wdl']).all() and (abs(rows['wdl'])<=1).all()
        assert np.array_equal(rows['cp'][:,0],-rows['cp'][:,1])
        assert np.array_equal(rows['wdl'][:,0],-rows['wdl'][:,1])
        keys[split]={r['key'][i].tobytes() for r in rows for i in range(5) if r['valid'][i]}
        for row in rows[np.linspace(0,len(rows)-1,min(200,len(rows)),dtype=int)]:
            job=lookup[(int(row['game']),int(row['node']))];b=chess.Board(job['initial_fen'])
            for m in job['history']:b.push_uci(m)
            t,m=encode(b);np.testing.assert_array_equal(t,row['tokens'][0]);np.testing.assert_array_equal(m,row['meta'][0])
            assert row['key'][0].tobytes()==position_key(b)
            children={}
            for move in b.legal_moves:
                child=b.copy();child.push(move);children[position_key(child)]=child
            for i in range(1,5):
                if not row['valid'][i]:continue
                c=children[row['key'][i].tobytes()];t,m=encode(c)
                np.testing.assert_array_equal(t,row['tokens'][i]);np.testing.assert_array_equal(m,row['meta'][i])
            samples+=1
        metrics=ranking_metrics(np.zeros(rows['valid'].shape),rows['cp'],rows['valid'] & rows['cp_valid'])
        counts[split]=dict(groups=len(rows),states=int(rows['valid'].sum()),unique_keys=len(keys[split]),
            nonmate_states=int((rows['valid'] & rows['cp_valid']).sum()),rank_pairs=metrics['pairs'])
    assert not keys['train']&keys['val']
    for r in roots:
        if r['split']=='eval':assert position_key(chess.Board(r['fen'])) not in keys['train']|keys['val']
    for name,digest in manifest['fingerprints'].items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest
    result=dict(passed=True,game_groups={k:len(v) for k,v in game_sets.items()},counts=counts,
        cross_split_key_overlap=0,trajectory_samples=samples,raw_cp_preserved=True,
        note='Checks new small-model data; original frozen large-model source/game overlap remains unknown.')
    (root/'audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
