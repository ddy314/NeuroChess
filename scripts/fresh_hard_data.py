"""Cloud-first novel puzzle curriculum, grouped by source game, with coherent SF labels.

Only canonical keys absent from ALL previous core/human positions are counted as
new. Puzzle FEN is before the setup move; the solver root follows the first move.
Source solutions are retained separately, never mislabeled as WDL/Q supervision.
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import io
import json
import multiprocessing
from pathlib import Path
import sqlite3
import sys
import time
import urllib.request
from urllib.parse import urlsplit

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import chess
import chess.engine
import chess.pgn
import numpy as np
import zstandard
from neurochess.encoding import position_key
from neurochess.data import record,ShardWriter,DTYPE
from neurochess.targets import assign_targets,TARGET_SPEC
from neurochess.audit import audit

ENGINE=None


def close_engine():
    if ENGINE is not None:
        try:ENGINE.quit()
        except chess.engine.EngineTerminatedError:pass


def digest(path):
    with open(path,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)


def group_id(url):
    # Lichess GameUrl has a ply fragment and sometimes a player token suffix.
    return urlsplit(url).path.strip('/').split('/')[0][:8]


def group_split(group,seed):
    n=int.from_bytes(hashlib.blake2b(f'{seed}:{group}'.encode(),digest_size=8).digest(),'little')%1000
    return 'test' if n<4 else 'val' if n<24 else 'train'


def puzzle_job(row,seed):
    board=chess.Board(row['FEN']);initial=board.fen(en_passant='fen')
    if not board.is_valid():raise ValueError('Invalid puzzle board')
    moves=row['Moves'].split()
    if len(moves)<2:raise ValueError('Missing setup/solution moves')
    board.push_uci(moves[0]);solver=board.copy()
    if solver.is_game_over(claim_draw=True):raise ValueError('Terminal puzzle root')
    for move in moves[1:]:board.push_uci(move)
    themes=set(row['Themes'].split())
    # Mutually exclusive, deterministic quotas. Endgame has priority.
    endgame=any(t.lower().endswith('endgame') for t in themes) or len(solver.piece_map())<=10
    category='endgame' if endgame else 'opening' if row.get('OpeningTags','').strip() or 'opening' in themes else 'tactical'
    group=group_id(row['GameUrl'])
    if len(group)!=8:raise ValueError('Expected a source game ID')
    return dict(id=row['PuzzleId'],group=group,split=group_split(group,seed),category=category,
                rating=int(row['Rating']),themes=sorted(themes),initial_fen=initial,history=[moves[0]],
                fen=solver.fen(en_passant='fen'),key=position_key(solver).hex(),solution=moves[1:])


def make_exclusions(roots,path):
    db=sqlite3.connect(path)
    db.execute('PRAGMA journal_mode=WAL');db.execute('CREATE TABLE IF NOT EXISTS old_keys(key BLOB PRIMARY KEY) WITHOUT ROWID')
    complete=db.execute("SELECT name FROM sqlite_master WHERE name='complete'").fetchone()
    if complete:return db
    counts={}
    for root in roots:
        count=0
        for file in sorted(Path(root).glob('*.npy')):
            rows=np.load(file,mmap_mode='r',allow_pickle=False)
            for start in range(0,len(rows),16384):
                keys=[(r.tobytes(),) for r in rows['key'][start:start+16384]]
                db.executemany('INSERT OR IGNORE INTO old_keys VALUES (?)',keys);count+=len(keys)
            db.commit()
        counts[str(root)]=count
    if not counts or not min(counts.values()):raise ValueError('Every old-data exclusion source must contain shards')
    db.execute('CREATE TABLE complete(counts TEXT)');db.execute('INSERT INTO complete VALUES (?)',(json.dumps(counts),));db.commit()
    return db


def quotas(plan):
    totals={'train':plan['training_positions'],'val':plan['validation_positions'],'test':plan['test_positions']}
    return {(split,c):round(total*f) for split,total in totals.items() for c,f in plan['category_fraction'].items()}


def collect(source,plan,old_db,out,excluded_fens=()):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    final=out/'jobs.jsonl'
    if final.exists():return final
    db=sqlite3.connect(out/'selection.sqlite');db.execute('PRAGMA journal_mode=WAL')
    db.execute('CREATE TABLE IF NOT EXISTS jobs(key TEXT PRIMARY KEY, split TEXT, category TEXT, payload TEXT)')
    reserved={position_key(chess.Board(fen)).hex() for fen in excluded_fens if fen.strip()}
    target=quotas(plan);counts=Counter({(s,c):n for s,c,n in db.execute('SELECT split,category,count(*) FROM jobs GROUP BY split,category')})
    previous=json.loads((out/'selection-progress.json').read_text()) if (out/'selection-progress.json').exists() else {}
    # The progress cursor is written only after the matching SQLite commit.
    skip_scanned=previous.get('scanned',0)
    rejections=Counter(previous.get('rejections',{}));start=time.time();last=0;scanned=0
    raw=open(source,'rb')
    with raw,zstandard.ZstdDecompressor().stream_reader(raw) as decoder,io.TextIOWrapper(decoder) as stream:
        for row in csv.DictReader(stream):
            scanned+=1
            if scanned<=skip_scanned:continue
            if int(row['Rating'])<plan['minimum_rating'] or int(row['Popularity'])<plan['minimum_popularity'] or int(row['NbPlays'])<plan['minimum_plays']:
                rejections['quality']+=1;continue
            try:job=puzzle_job(row,plan['seed'])
            except (ValueError,KeyError):rejections['invalid']+=1;continue
            bucket=(job['split'],job['category'])
            if counts[bucket]>=target[bucket]:continue
            if job['key'] in reserved:rejections['reserved_evaluation']+=1;continue
            if old_db.execute('SELECT 1 FROM old_keys WHERE key=?',(bytes.fromhex(job['key']),)).fetchone():
                rejections['old_position']+=1;continue
            if db.execute('INSERT OR IGNORE INTO jobs VALUES (?,?,?,?)',(job['key'],job['split'],job['category'],json.dumps(job))).rowcount:
                counts[bucket]+=1
            else:rejections['duplicate']+=1
            if time.time()-last>15:
                db.commit();atomic_json(out/'selection-progress.json',dict(phase='selecting',scanned=scanned,selected=sum(counts.values()),
                    target=sum(target.values()),counts={f'{s}/{c}':n for (s,c),n in counts.items()},rejections=rejections,seconds=time.time()-start));last=time.time()
            if all(counts[k]>=n for k,n in target.items()):break
    db.commit()
    if any(counts[k]!=n for k,n in target.items()):raise RuntimeError(f'Not enough novel source positions for frozen quotas: {counts} vs {target}')
    with final.with_suffix('.tmp').open('w') as f:
        for (payload,) in db.execute('SELECT payload FROM jobs ORDER BY key'):f.write(payload+'\n')
    final.with_suffix('.tmp').replace(final)
    atomic_json(out/'selection-audit.json',dict(scanned=scanned,counts={f'{s}/{c}':n for (s,c),n in counts.items()},
        rejections=rejections,old_overlap=0,duplicates=0,source_game_split='blake2b(seed:game_id), before selection',
        source_sha256=digest(source),jobs_sha256=digest(final),novelty='canonical board/castling/legal-EP keys absent from all supplied old-data shards',
        limits=['Canonical novelty is verified; old source-game overlap is not known.','Puzzle rating is human puzzle difficulty, not measured current-model error.']))
    return final


def initialize(engine,cleanup=True):
    global ENGINE
    ENGINE=chess.engine.SimpleEngine.popen_uci(engine,timeout=120)
    ENGINE.configure({'Threads':1,'Hash':64,'Skill Level':20,'UCI_LimitStrength':False})
    import multiprocessing.util
    if cleanup:multiprocessing.util.Finalize(None,close_engine,exitpriority=10)


def annotate_chunk(item):
    tasks,engine=item
    initialize(engine,cleanup=False)
    try:return [label(task) for task in tasks]
    finally:close_engine()


def label(task):
    job,nodes,coarse_nodes=task
    board=chess.Board(job['initial_fen'])
    for move in job['history']:board.push_uci(move)
    if position_key(board).hex()!=job['key']:raise ValueError('Puzzle identity differs')
    coarse=ENGINE.analyse(board,chess.engine.Limit(nodes=coarse_nodes),multipv=min(4,board.legal_moves.count()),game=object())
    candidates=list(dict.fromkeys([x['pv'][0] for x in coarse]+[board.parse_uci(job['solution'][0])]))
    infos=ENGINE.analyse(board,chess.engine.Limit(nodes=nodes),root_moves=candidates,multipv=len(candidates),game=object())
    row,legal=record(board);pvs=[]
    for info in infos:
        score=info['score'].white();pv={'line':info['pv'][0].uci()}
        pv['mate' if score.is_mate() else 'cp']=score.mate() if score.is_mate() else score.score();pvs.append(pv)
    assign_targets(row,board,legal,pvs)
    return job['key'],job['split'],row.tobytes(),json.dumps(dict(pvs=pvs,depths=[i.get('depth') for i in infos],
        puzzle_solution=job['solution'],category=job['category'],source_group=job['group']))


def annotate(jobs_path,engine,work,out,plan,start_index=0,end_index=None):
    work=Path(work);work.mkdir(parents=True,exist_ok=True);out=Path(out)
    if (out/'annotation-audit.json').exists():return
    jobs=[json.loads(l) for l in Path(jobs_path).read_text().splitlines()];jobs=jobs[start_index:end_index]
    spec=dict(jobs_sha256=digest(jobs_path),engine_sha256=digest(engine),target_spec=TARGET_SPEC,nodes=plan['teacher_nodes'],
        coarse_nodes=plan['teacher_coarse_nodes'],start=start_index,end=end_index)
    if (work/'spec.json').exists() and json.loads((work/'spec.json').read_text())!=spec:raise ValueError('Resume specification changed')
    atomic_json(work/'spec.json',spec)
    db=sqlite3.connect(work/'labels.sqlite');db.execute('PRAGMA journal_mode=WAL')
    db.execute('CREATE TABLE IF NOT EXISTS labels(key TEXT PRIMARY KEY,split TEXT,payload BLOB,metadata TEXT)')
    done={r[0] for r in db.execute('SELECT key FROM labels')};expected={j['key'] for j in jobs}
    if not done<=expected:raise ValueError('Resume contains labels outside this partition')
    pending=[j for j in jobs if j['key'] not in done];started=time.time();initial=len(done);completed=initial
    with ProcessPoolExecutor(plan['teacher_workers'],mp_context=multiprocessing.get_context('spawn')) as pool:
        for offset in range(0,len(pending),plan['teacher_workers']*32):
            tasks=[(j,plan['teacher_nodes'],plan['teacher_coarse_nodes']) for j in pending[offset:offset+plan['teacher_workers']*32]]
            chunks=[(tasks[i:i+32],engine) for i in range(0,len(tasks),32)]
            for results in pool.map(annotate_chunk,chunks,chunksize=1):
                for result in results:db.execute('INSERT INTO labels VALUES (?,?,?,?)',result);completed+=1
            db.commit();seconds=time.time()-started;rate=(completed-initial)/max(1,seconds)
            atomic_json(work/'progress.json',dict(phase='annotating',completed=completed,total=len(jobs),seconds=seconds,
                positions_per_second=rate,eta_hours=(len(jobs)-completed)/max(.001,rate)/3600,updated_at=time.time()))
            print(json.dumps(dict(completed=completed,total=len(jobs),positions_per_second=rate)),flush=True)
    export(db,out,spec)


def export(db,out,provenance):
    out=Path(out);building=out.with_name(out.name+'.building')
    if building.exists():
        import shutil
        shutil.rmtree(building)
    writers={s:ShardWriter(building/s,size=8192) for s in ['trainval','test']}
    for key,split,payload in db.execute('SELECT key,split,payload FROM labels ORDER BY key'):
        writers['test' if split=='test' else 'trainval'].add('val' if split=='test' else split,np.frombuffer(payload,dtype=DTYPE)[0])
    proofs={}
    for name,writer in writers.items():
        writer.close(dict(kind='novel-rated-puzzle-stockfish-distillation',reannotation_of_existing_core=False,**provenance))
        proofs[name]=audit(writer.root)
        if proofs[name]['within_split_duplicates']:raise ValueError('Unexpected duplicates')
    atomic_json(building/'annotation-audit.json',proofs)
    if out.exists():raise ValueError('Refuse to overwrite an existing completed dataset')
    building.replace(out)


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--jobs',required=True)
    p.add_argument('--engine',required=True);p.add_argument('--work',required=True);p.add_argument('--out',required=True)
    p.add_argument('--start-index',type=int,default=0);p.add_argument('--end-index',type=int)
    a=p.parse_args();annotate(a.jobs,a.engine,a.work,a.out,json.loads(Path(a.plan).read_text()),a.start_index,a.end_index)

if __name__=='__main__':main()
