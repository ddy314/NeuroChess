"""Resumable local annotation only. Never starts training or a cloud runtime.

SQLite is the canonical per-position checkpoint. All worker imports remain CPU
only; CUDA is loaded solely by the coordinator. A bounded asynchronous queue
keeps engines busy while GPU predictions are batched and cached.
"""
import argparse
import concurrent.futures as futures
import fcntl
import hashlib
import importlib.util
import json
import multiprocessing
import os
from pathlib import Path
import sqlite3
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if importlib.util.find_spec('chess') is None:
    sys.path.extend(str(p) for p in (ROOT / '.venv/lib').glob('python*/site-packages'))
from scripts import local_teacher_worker as teacher


def write(path, data):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    tmp.replace(path)


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def connect(path):
    db = sqlite3.connect(path)
    db.execute('PRAGMA journal_mode=WAL')
    db.execute('PRAGMA synchronous=FULL')
    db.executescript('''
        CREATE TABLE IF NOT EXISTS info (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS jobs (
          key TEXT PRIMARY KEY, split TEXT NOT NULL, family TEXT NOT NULL,
          job TEXT NOT NULL, quarantine INTEGER NOT NULL DEFAULT 0, prediction TEXT);
        CREATE TABLE IF NOT EXISTS groups (family TEXT PRIMARY KEY, split TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS results (
          key TEXT PRIMARY KEY REFERENCES jobs(key), split TEXT NOT NULL,
          accepted INTEGER NOT NULL, screened INTEGER NOT NULL, repair INTEGER NOT NULL,
          unstable INTEGER NOT NULL, metadata TEXT NOT NULL, row BLOB, completed REAL NOT NULL);
        CREATE INDEX IF NOT EXISTS result_split ON results(split, accepted);
        CREATE INDEX IF NOT EXISTS job_family ON jobs(family);
        CREATE TABLE IF NOT EXISTS exclusions (key TEXT PRIMARY KEY, reason TEXT NOT NULL, excluded_at REAL NOT NULL);
    ''')
    return db


def quarantine_family(db, family, reason):
    keys = [r[0] for r in db.execute('SELECT key FROM jobs WHERE family=?', (family,))]
    db.executemany('INSERT OR IGNORE INTO exclusions VALUES (?,?,?)',
                   [(key, reason, time.time()) for key in keys])
    db.execute('UPDATE jobs SET quarantine=1 WHERE family=?', (family,))
    # Preserve original teacher evidence while revoking it as training data.
    db.execute("UPDATE results SET accepted=0,repair=0,row=NULL,metadata=json_set(metadata,"
               "'$.accepted',json('false'),'$.quarantine_reason',?) WHERE key IN "
               '(SELECT key FROM jobs WHERE family=?)', (reason, family))
    return len(keys)


def audit_standard_sources(db, work):
    import chess
    families = db.execute("SELECT DISTINCT family FROM jobs WHERE "
                          "json_extract(job,'$.initial_fen')!=? OR "
                          "COALESCE(json_extract(job,'$.source_game_headers.Variant'),'Standard')!='Standard'",
                          (chess.STARTING_FEN,)).fetchall()
    for (family,) in families:
        quarantine_family(db, family, 'nonstandard_initial_position_or_variant')
    db.commit()
    proof = dict(checked_candidates=db.execute('SELECT COUNT(*) FROM jobs').fetchone()[0],
                 quarantined_families=len(families),
                 excluded_positions=db.execute('SELECT COUNT(*) FROM exclusions').fetchone()[0],
                 accepted_nonstandard_positions=db.execute('SELECT COUNT(*) FROM results r JOIN jobs j ON r.key=j.key '
                                                          'WHERE r.accepted=1 AND j.quarantine=1').fetchone()[0],
                 updated_at=time.time())
    assert proof['accepted_nonstandard_positions']==0
    write(work/'standard-source-audit.json', proof)
    return proof


def build_index(db, inputs):
    if db.execute("SELECT value FROM info WHERE key='indexed'").fetchone():
        return
    # Idempotent rebuild after interruption; quarantine all cross-split keys.
    total = 0
    for path in sorted(inputs.glob('candidates-*.jsonl')):
        with path.open() as stream:
            for line in stream:
                j = json.loads(line)
                db.execute('INSERT INTO jobs(key,split,family,job) VALUES (?,?,?,?) '
                           'ON CONFLICT(key) DO UPDATE SET quarantine=MAX(jobs.quarantine,jobs.split!=excluded.split)',
                           (j['key'], j['split'], j['group'], line.strip()))
                owner = db.execute('SELECT split FROM groups WHERE family=?', (j['group'],)).fetchone()
                if owner and owner[0] != j['split']:
                    raise ValueError('Source game crosses splits')
                db.execute('INSERT OR IGNORE INTO groups VALUES (?,?)', (j['group'], j['split']))
                total += 1
                if total % 10000 == 0:
                    db.commit()
        db.commit()
    db.execute("INSERT INTO info VALUES ('indexed',?)", (json.dumps(dict(raw_rows=total)),))
    db.commit()


def save_result(db, meta, limits, progress=None):
    from neurochess.data import record
    from neurochess.targets import assign_targets
    j = meta['job']
    owner = db.execute('SELECT split,quarantine FROM jobs WHERE key=?', (j['key'],)).fetchone()
    if owner is None or owner[1] or owner[0] != j['split']:
        return False
    if db.execute('SELECT 1 FROM results WHERE key=?', (j['key'],)).fetchone():
        return False
    accepted = meta.get('accepted', meta.get('stable') is not False)
    # Metadata recovery and asynchronous in-flight completions share this cap.
    count = progress['labelled'][j['split']] if progress is not None else db.execute(
        'SELECT COUNT(*) FROM results WHERE split=? AND accepted=1', (j['split'],)).fetchone()[0]
    accepted = bool(accepted and count < limits[j['split']])
    row_bytes = None
    if accepted:
        board = teacher.board_for(j)
        row, legal = record(board)
        assign_targets(row, board, legal, meta['teacher'])
        row_bytes = row.tobytes()
    meta = dict(meta, accepted=accepted)
    db.execute('INSERT INTO results VALUES (?,?,?,?,?,?,?,?,?)', (
        j['key'], j['split'], int(accepted), int(meta.get('prediction') is not None),
        int(bool(meta.get('repair')) and accepted), int(meta.get('stable') is False),
        json.dumps(meta), row_bytes, time.time()))
    if meta.get('prediction') is not None:
        db.execute('UPDATE jobs SET prediction=? WHERE key=?', (json.dumps(meta['prediction']), j['key']))
    if progress is not None:
        progress['labelled'][j['split']] += int(accepted)
        progress['screened'] += int(meta.get('prediction') is not None)
        progress['verified_training_mistakes'] += int(j['split']=='train' and accepted and bool(meta.get('repair')))
        progress['unstable_discarded'] += int(meta.get('stable') is False)
        progress['completed'] += 1
    return True


def counts(db):
    labelled = dict(train=0, val=0, test=0)
    for split, number in db.execute('SELECT split,COUNT(*) FROM results WHERE accepted=1 GROUP BY split'):
        labelled[split] = number
    screened, wrong, unstable, completed = db.execute(
        "SELECT COALESCE(SUM(screened),0),COALESCE(SUM(repair*(split='train')),0),"
        'COALESCE(SUM(unstable),0),COUNT(*) FROM results WHERE key IN '
        '(SELECT key FROM jobs WHERE quarantine=0)').fetchone()
    return dict(labelled=labelled, screened=screened, verified_training_mistakes=wrong,
                unstable_discarded=unstable, completed=completed)


def pending_jobs(db):
    # Finish each read statement before commits. A single 660k-row cursor keeps
    # a WAL reader alive for hours and prevents checkpointing completed work.
    last_id = 0
    while True:
        page = db.execute('SELECT j.rowid,j.job,j.prediction FROM jobs j '
                          'LEFT JOIN results r ON r.key=j.key '
                          'WHERE j.rowid>? AND j.quarantine=0 AND r.key IS NULL '
                          'ORDER BY j.rowid LIMIT 256', (last_id,)).fetchall()
        if not page:
            return
        last_id = page[-1][0]
        for _, job, prediction in page:
            yield job, prediction


def benchmark(db, binary, output, choices):
    # Same fixed mix of previously screened positions for every parallelism.
    samples = [json.loads(r[0]) for r in db.execute(
        'SELECT metadata FROM results WHERE screened=1 ORDER BY key LIMIT 256')]
    if len(samples) < 24:
        raise ValueError('Too few recovered screening samples for benchmark')
    tasks = [(m['job'], m['prediction']) for m in samples]
    reports = []
    for workers in choices:
        start = time.monotonic()
        with futures.ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context('spawn'),
                                          initializer=teacher.initialize, initargs=(binary,)) as pool:
            results = list(pool.map(teacher.label, tasks))
        elapsed = time.monotonic() - start
        semantic = [dict(key=m['job']['key'], teacher=m['teacher'], stable=m['stable'],
                         decision=m['decision']) for m in results]
        digest = hashlib.sha256(json.dumps(semantic, sort_keys=True).encode()).hexdigest()
        reports.append(dict(workers=workers, jobs=len(tasks), seconds=elapsed,
                            positions_per_second=len(tasks)/elapsed, output_sha256=digest,
                            unstable=sum(m['stable'] is False for m in results),
                            mistakes=sum(m['repair'] for m in results)))
        write(output, dict(protocol='Same256 full screening tasks, unchanged nodes, includes pool startup/shutdown', results=reports))
        print('benchmark', json.dumps(reports[-1]), flush=True)
    # Prefer fewer workers when within 5% of the best measured throughput.
    best = max(r['positions_per_second'] for r in reports)
    selected = min(r['workers'] for r in reports if r['positions_per_second'] >= .95*best)
    write(output, dict(protocol='Same256 full screening tasks;10k+30k and100k/300k refinement; no budget reductions',
                       results=reports, selected_workers=selected,
                       identical_outputs=len({r['output_sha256'] for r in reports}) == 1))
    return selected


def predictor(model_path):
    import torch
    import numpy as np
    from neurochess.model import ModelConfig
    from neurochess.policy_value_pair import PolicyValuePair
    from neurochess.multihead import MultiheadPlayer, fuse_heads
    from neurochess.data import record, DTYPE
    torch.set_num_threads(2)
    state = torch.load(model_path, map_location='cpu', weights_only=False, mmap=True)
    assert state['step'] == 3000 and state['architecture'] == 'policy_value_pair_v1'
    model = PolicyValuePair(ModelConfig(**state['model_config']))
    model.load_state_dict(state['model'])
    model.cuda().eval()
    del state
    net = MultiheadPlayer(model, 'cuda')

    def predict(jobs):
        boards = [teacher.board_for(j) for j in jobs]
        rows = np.array([record(b)[0] for b in boards], dtype=DTYPE)
        raw = net.raw_outputs(rows['tokens'].copy(), rows['meta'].copy(), rows['moves'].copy())
        fused, _, values, _ = fuse_heads(raw[:, :256], raw[:, 256:512], raw[:, 512:], rows['moves'] >= 0)
        result = []
        for i, board in enumerate(boards):
            legal = list(board.legal_moves)
            rank = np.argsort(-fused[i, :len(legal)])[:2]
            result.append(dict(move=legal[rank[0]].uci(), top_moves=[legal[k].uci() for k in rank],
                               value=float(values[i]), wdl=raw[i, 512:].tolist()))
        return result
    return predict


def export_ready(db, work, proof):
    import numpy as np
    from neurochess.data import DTYPE
    from neurochess.targets import TARGET_SPEC
    folder = work / 'dataset'
    folder.mkdir(exist_ok=True)
    files = {}
    for family, query in [('', 'accepted=1'), ('mistakes/', "accepted=1 AND repair=1 AND split='train'")]:
        target = folder / family
        target.mkdir(exist_ok=True)
        for split in ['train', 'val', 'test']:
            cursor = db.execute(f'SELECT row FROM results WHERE {query} AND split=? ORDER BY key', (split,))
            index = 0
            while batch := cursor.fetchmany(8192):
                path = target / f'{split}-{index:06d}.npy'
                with path.with_suffix('.tmp').open('wb') as stream:
                    np.save(stream, np.frombuffer(b''.join(r[0] for r in batch), dtype=DTYPE))
                path.with_suffix('.tmp').replace(path)
                files[str(path.relative_to(folder))] = sha(path)
                index += 1
        write(target / 'manifest.json', dict(schema=1, row_bytes=DTYPE.itemsize, target_spec=TARGET_SPEC,
              counts=proof['labelled'] if not family else dict(train=proof['verified_training_mistakes'], val=0),
              provenance=dict(kind='new-elite-independent-SF18', source_license='CC-BY-SA-4.0',
              attribution='Lichess broadcast contributors; https://database.lichess.org/#broadcasts',
              source_audit='../inputs/candidate-ready.json', historical_source_game_overlap='Unknown for old data')))
        files[str((target/'manifest.json').relative_to(folder))] = sha(target/'manifest.json')
    for relative, expected in files.items():
        assert sha(folder / relative) == expected
    write(work / 'teacher-ready.json', dict(proof, sha256=files, model_sha256=sha(work/'local-annotation-model.pt'),
          source_candidate_audit_sha256=sha(work/'inputs/candidate-ready.json'), training_started=False))


def run(args):
    work = Path(args.work).resolve()
    lock = (work/'annotation.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    inputs = work / 'inputs'
    db = connect(work/'annotation.sqlite3')
    plan = json.loads((inputs/'plan.json').read_text())
    limits = dict(train=500000, val=3000, test=3000)
    screen_goal = plan['screening_minimum']
    wrong_goal = plan['verified_training_mistakes_minimum']
    binary = str(inputs/'stockfish')
    Path(binary).chmod(0o755)
    phase = 'indexing-local-candidates'
    started = time.time()
    initial_completed = 0
    workers = args.workers
    progress = counts(db)
    def status():
        write(work/'pipeline-status.json', dict(phase=phase, updated_at=time.time(), started_at=started,
              pid=os.getpid(), workers=workers, label_goal=500000, screen_goal=screen_goal, mistake_goal=wrong_goal,
              local_completed=progress['completed']-initial_completed,
              local_positions_per_second=(progress['completed']-initial_completed)/max(1,time.time()-started),
              cloud_released=(work/'cloud-release-verification.json').exists(), training_started=False, **progress))
        return progress
    try:
        status()
        build_index(db, inputs)
        phase='auditing-standard-source-games'; status()
        audit_standard_sources(db, work)
        progress=counts(db)
        if not db.execute("SELECT 1 FROM info WHERE key='recovered'").fetchone():
            with (inputs/'teacher-metadata.jsonl').open() as stream:
                for line in stream:
                    try:
                        meta = json.loads(line)
                    except json.JSONDecodeError:
                        continue  # Interrupted last line is not a completed result.
                    save_result(db, meta, limits, progress)
            db.execute("INSERT INTO info VALUES ('recovered','true')")
            db.commit()
        initial_completed = counts(db)['completed']
        audit = dict(unique_keys=db.execute('SELECT COUNT(*) FROM jobs').fetchone()[0],
                     cross_split_quarantined=db.execute('SELECT COUNT(*) FROM jobs WHERE quarantine=1').fetchone()[0],
                     recovered=counts(db), source_game_cross_split_overlap=0)
        write(work/'local-data-audit.json', audit)
        if args.benchmark:
            phase = 'benchmarking-local-teacher'; status()
            workers = benchmark(db, binary, work/'benchmark.json', [4, 8, 12, 16])
        elif workers == 0:
            workers = json.loads((work/'benchmark.json').read_text())['selected_workers']
        if args.benchmark_only:
            phase = 'benchmark-complete'; status(); return
        model_hash = sha(work/'local-annotation-model.pt')
        expected = json.loads((work/'migration-proof.json').read_text())['/content/local-annotation-model.pt']['sha256']
        assert model_hash == expected, 'Transferred checkpoint hash mismatch'
        previous = db.execute("SELECT value FROM info WHERE key='model_sha256'").fetchone()
        if previous and previous[0] != model_hash:
            raise ValueError('Prediction cache belongs to a different model')
        db.execute("INSERT OR IGNORE INTO info VALUES ('model_sha256',?)", (model_hash,))
        db.commit()
        # Once screen/mistake quotas are met, remaining general labels need no GPU.
        predict = None
        phase = 'local-screening-and-labelling'
        started = time.time()
        initial_completed = counts(db)['completed']
        status()
        cursor = iter(pending_jobs(db))
        pending = {}
        exhausted = False
        last_status = time.monotonic()
        progress = counts(db)
        with futures.ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context('spawn'),
                                          initializer=teacher.initialize, initargs=(binary,)) as pool:
            while pending or not exhausted:
                # Keep enough tasks queued to overlap batched GPU and teacher work;
                # consume fast completions without waiting for the slowest position.
                if len(pending) < 2*workers and not exhausted:
                    tasks = []
                    while len(tasks) < args.batch:
                        item = next(cursor, None)
                        if item is None:
                            exhausted = True; break
                        job = json.loads(item[0])
                        if progress['labelled'][job['split']] >= limits[job['split']]:
                            continue
                        mine = job['split']=='train' and (progress['screened'] < screen_goal or progress['verified_training_mistakes'] < wrong_goal)
                        tasks.append([job, json.loads(item[1]) if mine and item[1] else None, mine])
                    missing = [t for t in tasks if t[2] and t[1] is None]
                    if missing:
                        if predict is None:
                            predict = predictor(work/'local-annotation-model.pt')
                        for task, pred in zip(missing, predict([t[0] for t in missing])):
                            task[1] = pred
                            db.execute('UPDATE jobs SET prediction=? WHERE key=?', (json.dumps(pred), task[0]['key']))
                        db.commit()
                    for job, pred, mine in tasks:
                        pending[pool.submit(teacher.label, (job, pred if mine else None))] = job['key']
                if pending:
                    done, _ = futures.wait(pending, timeout=2, return_when=futures.FIRST_COMPLETED)
                    for future in done:
                        try:
                            meta = future.result()
                        except teacher.CandidateError as error:
                            family=db.execute('SELECT family FROM jobs WHERE key=?', (pending[future],)).fetchone()[0]
                            quarantine_family(db, family, str(error)); db.commit()
                            progress=counts(db)
                            del pending[future]
                            continue
                        save_result(db, meta, limits, progress)
                        db.commit()  # Every completed teacher job survives interruption.
                        del pending[future]
                if time.monotonic()-last_status > 5:
                    status(); last_status=time.monotonic()
                if all(progress['labelled'][s] == limits[s] for s in limits) and not pending:
                    break
        progress = counts(db)
        if progress['labelled'] != limits or progress['screened'] < screen_goal or progress['verified_training_mistakes'] < wrong_goal:
            raise RuntimeError('Candidate pool exhausted before accepted-label/mistake quotas; more genuinely new sources required')
        phase='exporting-and-auditing'; status()
        export_ready(db, work, progress)
        phase='annotation-complete-training-deferred'; status()
    except BaseException:
        db.commit()
        write(work/'failure.json', dict(phase=phase, traceback=traceback.format_exc(), updated_at=time.time()))
        phase='failed-local-annotation'; status()
        raise
    finally:
        db.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--work', required=True)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--benchmark-only', action='store_true')
    run(parser.parse_args())
