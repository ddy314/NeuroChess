"""Local, resumable novel-game reservoir. Source evaluations only prioritize.

No teacher labels, optimizer, cloud VM or automatic training are produced here.
Every archive is pinned to the publisher checksum; the SQLite primary key and
immutable mmap index exclude duplicates and historical positions respectively.
"""
import argparse,collections,fcntl,hashlib,io,json,shutil,sqlite3,sys,time,traceback
from pathlib import Path
from urllib.request import urlopen
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import chess,chess.pgn,numpy as np,zstandard
from neurochess.encoding import position_key
from neurochess.prepare import FilteredBuilder,rejection
from scripts.novel_elite_collect_cloud import game_group,group_split
from scripts.run_dynamic_curriculum import write


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def collect(args):
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    lock=(out/'collection.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    sources=json.loads(Path(args.sources).read_text());index=np.load(args.exclude_keys,mmap_mode='r')
    if index.dtype!=np.dtype('V16') or index.ndim!=1:raise ValueError('Sorted V16 exclusion index required')
    if np.any(np.sort(index)!=index) or np.any(index[1:]==index[:-1]):raise ValueError('Exclusion index must be sorted and unique')
    cert=json.loads(Path(args.exclusion_audit).read_text())
    if not cert['complete_original_7590978_key_index'] or cert['index_sha256']!=sha(args.exclude_keys):raise ValueError('Incomplete or changed historical index')
    prior=sqlite3.connect(f'file:{Path(args.prior_db).resolve()}?mode=ro',uri=True)
    blocked_groups={r[0] for r in prior.execute('SELECT family FROM groups')};prior.close()
    db=sqlite3.connect(out/'candidates.sqlite3');db.execute('PRAGMA journal_mode=WAL');db.execute('PRAGMA synchronous=FULL')
    db.executescript('''CREATE TABLE IF NOT EXISTS jobs(key TEXT PRIMARY KEY,group_id TEXT NOT NULL,split TEXT NOT NULL,job TEXT NOT NULL);
      CREATE TABLE IF NOT EXISTS groups(group_id TEXT PRIMARY KEY,split TEXT NOT NULL);
      CREATE TABLE IF NOT EXISTS sources(name TEXT PRIMARY KEY,sha256 TEXT NOT NULL,summary TEXT NOT NULL);
      CREATE TABLE IF NOT EXISTS settings(key TEXT PRIMARY KEY,value TEXT NOT NULL);''')
    signature=json.dumps(dict(sources=sources,index=cert['index_sha256'],min_ply=14,max_ply=160,stride=3,max_per_game=40),sort_keys=True)
    old=db.execute("SELECT value FROM settings WHERE key='signature'").fetchone()
    if old and old[0]!=signature:
        previous=json.loads(old[0]);current=json.loads(signature);old_sources=previous.pop('sources');new_sources=current.pop('sources')
        if previous!=current or new_sources[:len(old_sources)]!=old_sources:raise ValueError('Existing source/index/sampling configuration changed; use a new output directory')
        db.execute("UPDATE settings SET value=? WHERE key='signature'",(signature,))
    db.execute("INSERT OR IGNORE INTO settings VALUES ('signature',?)",(signature,));db.commit()
    counts=collections.Counter(dict(db.execute('SELECT split,COUNT(*) FROM jobs GROUP BY split')))
    goals=dict(train=args.train,val=args.heldout,test=args.heldout);reasons=collections.Counter();started=time.monotonic();last=started
    def status(phase,**extra):
        db.commit();write(out/'status.json',dict(phase=phase,counts=dict(counts),goals=goals,unique_source_games=db.execute('SELECT COUNT(*) FROM groups').fetchone()[0],historical_source_groups_excluded=len(blocked_groups),historical_keys_excluded=len(index),exclusion_sha256=cert['index_sha256'],seconds=time.monotonic()-started,rejections=dict(reasons),teacher_labels_ready=False,training_started=False,**extra))
    def enough():return all(counts[s]>=n for s,n in goals.items())
    try:
        for source in sources:
            if enough():break
            done=db.execute('SELECT sha256 FROM sources WHERE name=?',(source['name'],)).fetchone()
            if done:
                if done[0]!=source['sha256']:raise ValueError('Previously processed source checksum changed')
                continue
            archive=out/(source['name']+'.pgn.zst');status('download',source=source['name'])
            if not archive.exists() or sha(archive)!=source['sha256']:
                partial=archive.with_suffix('.partial')
                with urlopen(source['url'],timeout=90) as response,partial.open('wb') as f:shutil.copyfileobj(response,f)
                if sha(partial)!=source['sha256']:raise ValueError('Archive differs from published checksum')
                partial.replace(archive)
            summary=dict(source,accepted_games=0,positions=0);status('collecting',source=source['name'])
            with archive.open('rb') as raw,io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(raw)) as stream:
                while (g:=chess.pgn.read_game(stream,Visitor=lambda:FilteredBuilder(source))) is not None:
                    why=rejection(g,source)
                    if why:reasons[why]+=1;continue
                    initial=g.board()
                    if initial.chess960 or initial.fen()!=chess.STARTING_FEN:reasons['nonstandard_start']+=1;continue
                    if any(not m for m in g.mainline_moves()):reasons['null_move']+=1;continue
                    group=game_group(initial,list(g.mainline_moves()));split=group_split(group)
                    if group in blocked_groups:reasons['historical_source_game']+=1;continue
                    if db.execute('SELECT 1 FROM groups WHERE group_id=?',(group,)).fetchone():reasons['duplicate_source_game']+=1;continue
                    if counts[split]>=goals[split]:continue
                    b=initial.copy();positions=[]
                    for ply,node in enumerate(g.mainline(),1):
                        b.push(node.move)
                        if ply<14 or ply>160 or ply%3 or b.is_game_over(claim_draw=True):continue
                        key=position_key(b);value=np.array(key,dtype='V16');where=int(np.searchsorted(index,value))
                        if where<len(index) and index[where]==value:reasons['historical_key']+=1;continue
                        score=node.eval();evaluation=None
                        if score is not None:
                            s=score.white();evaluation={'mate':s.mate()} if s.is_mate() else {'cp':s.score()}
                        positions.append(dict(key=key.hex(),group=group,split=split,initial_fen=initial.fen(),history=[m.uci() for m in b.move_stack],source=source['name'],source_url=source['url'],source_sha256=source['sha256'],license=source['license'],source_game_headers=dict(g.headers),source_eval_white=evaluation,source_eval_role='Queue priority/provenance only; never an accepted training label',ply=ply))
                    if len(positions)>40:positions=[positions[i] for i in np.linspace(0,len(positions)-1,40,dtype=int)]
                    db.execute('INSERT INTO groups VALUES (?,?)',(group,split));summary['accepted_games']+=1
                    for j in positions:
                        if counts[split]>=goals[split]:break
                        cursor=db.execute('INSERT OR IGNORE INTO jobs VALUES (?,?,?,?)',(j['key'],group,split,json.dumps(j)))
                        if cursor.rowcount:counts[split]+=1;summary['positions']+=1
                        else:reasons['duplicate_key']+=1
                    if time.monotonic()-last>=3:status('collecting',source=source['name']);last=time.monotonic()
                    if enough():break
            db.execute('INSERT INTO sources VALUES (?,?,?)',(source['name'],source['sha256'],json.dumps(summary)));status('source-complete',source=source['name'])
        files={}
        for split in goals:
            target=out/(split+'.jsonl');partial=target.with_suffix('.tmp')
            with partial.open('w') as f:
                for encoded, in db.execute('SELECT job FROM jobs WHERE split=? ORDER BY key',(split,)):f.write(encoded+'\n')
            partial.replace(target);files[target.name]=sha(target)
        manifest=dict(candidate_counts=dict(counts),goals=goals,candidate_target_reached=enough(),old_key_overlap=0,source_game_cross_split_overlap=0,known_historical_source_game_overlap=0,historical_games_with_unknown_provenance='Exact keys excluded; complete historical game isolation is unprovable',exclusion_sha256=cert['index_sha256'],files=files,teacher_labels_ready=False)
        write(out/'candidate-manifest.json',manifest);status('candidates-ready-awaiting-teacher' if enough() else 'sources-exhausted-no-padding')
    except BaseException:
        status('failed',traceback=traceback.format_exc());raise
    finally:db.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--sources',required=True);p.add_argument('--out',required=True);p.add_argument('--exclude-keys',required=True);p.add_argument('--exclusion-audit',required=True);p.add_argument('--prior-db',default='output/targeted-scale-local-20261102/annotation.sqlite3');p.add_argument('--train',type=int,default=500000);p.add_argument('--heldout',type=int,default=10000);collect(p.parse_args())
