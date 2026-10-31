"""Verify exact-key exclusions and source-game separation for causal search tests."""
import argparse
import json
from pathlib import Path
import sqlite3
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import chess
import numpy as np
from neurochess.encoding import position_key


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();r=Path(a.root)
    data=Path('output/rank-frontier-20261031/data');small=set();train_games=set()
    for split in ['train','val']:
        rows=np.load(data/f'{split}.npy',mmap_mode='r');train_games.update(map(int,rows['game']))
        for row in rows:
            for i in range(len(row['valid'])):
                if row['valid'][i]:small.add(row['key'][i].tobytes())
    db=sqlite3.connect('file:'+str(Path('data/evals-v2-5m/keys.sqlite').resolve())+'?mode=ro',uri=True)
    sets={};summary={}
    old=json.loads(Path('output/cpu-speed-20261031/heldout-roots/plan.json').read_text())['roots']
    old_games={x['game'] for x in old};old_keys={position_key(chess.Board(x['fen'])) for x in old}
    for name,folder in [('openings','openings'),('heldout','heldout-roots')]:
        rows=json.loads((r/folder/'plan.json').read_text())['roots'];games={x['game'] for x in rows};keys={position_key(chess.Board(x['fen'])) for x in rows}
        assert len(keys)==len(games)==len(rows)
        assert not games&train_games and not games&old_games
        assert not keys&small and not keys&old_keys
        assert all(db.execute('select 1 from positions where key=?',(k,)).fetchone() is None for k in keys)
        sets[name]=(games,keys);summary[name]=dict(roots=len(rows),first_game=min(games),last_game=max(games))
    assert not sets['openings'][0]&sets['heldout'][0]
    assert not sets['openings'][1]&sets['heldout'][1]
    result=dict(passed=True,groups=summary,small_training_game_count=len(train_games),small_training_max_game=max(train_games),
        exact_key_overlap=0,source_game_overlap=0,note='Disjoint from the new student train/val and old development roots. Original large-model full source-game overlap unknown; only exact base-key exclusion verified.')
    (r/'holdout-audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':main()
