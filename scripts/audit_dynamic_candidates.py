"""Audit a consistent local candidate snapshot, including every legal history."""
import argparse,hashlib,itertools,json,sqlite3,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import chess,numpy as np
from neurochess.encoding import position_key
from scripts.run_dynamic_curriculum import write


def audit(args):
    started=time.monotonic();db=sqlite3.connect(f'file:{Path(args.db).resolve()}?mode=ro',uri=True);db.execute('BEGIN')
    index=np.load(args.exclude_keys,mmap_mode='r');counts={s:0 for s in ['train','val','test']};groups=0;keys=[];digest=hashlib.sha256()
    for group,items in itertools.groupby(db.execute('SELECT group_id,job FROM jobs ORDER BY group_id,key'),key=lambda r:r[0]):
        jobs=[json.loads(r[1]) for r in items];splits={j['split'] for j in jobs}
        if len(splits)!=1:raise ValueError('Source game crosses splits')
        longest=max(jobs,key=lambda j:len(j['history']));history=longest['history'];initial=longest['initial_fen'];wanted={len(j['history']):j['key'] for j in jobs}
        for j in jobs:
            if j['initial_fen']!=initial or j['history']!=history[:len(j['history'])]:raise ValueError('Source game has incompatible histories/origins')
        b=chess.Board(initial)
        for ply,move in enumerate(history,1):
            if move=='0000':raise ValueError('Null move in purported legal source game')
            b.push_uci(move)
            if ply in wanted and position_key(b).hex()!=wanted[ply]:raise ValueError('Canonical key does not match complete legal history')
        groups+=1
        for j in jobs:
            counts[j['split']]+=1;keys.append(bytes.fromhex(j['key']));digest.update((j['key']+'|'+group+'|'+j['split']+'\n').encode())
    a=np.array(keys,dtype='V16');where=np.searchsorted(index,a);valid=where<len(index);overlap=int(np.sum(index[where[valid]]==a[valid]));unique=len(np.unique(a))
    if overlap or unique!=len(keys):raise ValueError('Historical or duplicate canonical key')
    result=dict(passed=True,rows=sum(counts.values()),counts=counts,source_games=groups,all_histories_legal=True,all_keys_match_histories=True,old_key_overlap=overlap,duplicate_keys=len(keys)-unique,source_game_cross_split_overlap=0,cohort_sha256=digest.hexdigest(),database_snapshot=True,seconds=time.monotonic()-started,teacher_labels_ready=False)
    write(Path(args.out),result);db.close();print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--db',required=True);p.add_argument('--exclude-keys',required=True);p.add_argument('--out',required=True);audit(p.parse_args())
