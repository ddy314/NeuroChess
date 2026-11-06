"""Prepare exactly 10M fresh, strict Lichess roots on one CPU Colab VM.

Resumption reindexes only atomically committed, readback-verified shard pairs.
Sampling redistributes exhausted-stratum quotas and uses a seeded cyclic start;
it covers the file, but is not a uniform random sample of all database rows.
"""
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor
import fcntl
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import random
import shutil
import sqlite3
import sys
import time
import traceback
from urllib.request import Request, urlopen

import numpy as np
import zstandard as zstd

from neurochess.data import DTYPE
from neurochess.lichess_clean import CLEANING_SPEC, convert_batch, audit_rows, split_key, strict_convert, init_clean_worker
from neurochess.pawn_data import contains, sha, write_json, publish_array, load_published
from scripts.stream_evals import fetch_range

MAGIC = b'\x28\xb5\x2f\xfd'
CHUNK = 16 * 1024**2
SPLITS = ('train', 'val', 'test')


def source_lines(source, start, stop, fetch=fetch_range):
    """Yield complete JSON lines after an arbitrary offset in concatenated pzstd.

    Preserve partial lines across data frames and skip pzstd metadata frames.
    Finish a line crossing the stop boundary using the following data frame.
    """
    size = source['bytes']; offset = start; buffer = b''; next_byte = start
    def extend():
        nonlocal buffer, next_byte
        if next_byte >= size:
            return False
        end = min(size - 1, next_byte + CHUNK - 1)
        buffer += fetch(source['url'], source['etag'], next_byte, end)
        next_byte = end + 1
        return True
    extend()
    p = buffer.find(MAGIC)
    while p < 0:
        if next_byte >= stop or not extend():
            return
        p = buffer.find(MAGIC)
    buffer = buffer[p:]; offset += p
    first = True; carry = b''
    while buffer or extend():
        while len(buffer) < 8 and extend():
            pass
        tag = int.from_bytes(buffer[:4], 'little')
        if 0x184D2A50 <= tag <= 0x184D2A5F:
            length = 8 + int.from_bytes(buffer[4:8], 'little')
            while len(buffer) < length:
                if not extend():raise OSError('Truncated skippable frame')
            buffer = buffer[length:]; offset += length
            continue
        if buffer[:4] != MAGIC:
            raise ValueError(f'Invalid zstd frame at {offset}')
        if first and offset >= stop:
            return
        frame_offset = offset; obj = zstd.ZstdDecompressor().decompressobj(); parts = []
        while True:
            part = buffer; parts.append(obj.decompress(part))
            used = len(part) - len(obj.unused_data) if obj.eof else len(part)
            offset += used; buffer = obj.unused_data if obj.eof else b''
            if obj.eof:break
            if not extend():raise OSError('Truncated data frame')
        data = b''.join(parts)
        if first:
            # start=0 can locate the first frame immediately after its metadata;
            # every other arbitrary start lacks the previous frame's suffix.
            if start != 0:
                data = data[data.index(b'\n') + 1:]
            first = False
        body = carry + data
        if frame_offset >= stop:
            if carry:
                yield frame_offset, body[:body.index(b'\n')]
            return
        lines = body.split(b'\n'); carry = lines.pop()
        for line in lines:
            if line:
                yield frame_offset, line
        if next_byte >= size and not buffer:
            if carry:raise ValueError('Unterminated final JSON line')
            return


def batches(lines, size=1024):
    batch = []; frame = None
    for offset, line in lines:
        # Every row retains its own source frame. A batch stays in one frame.
        if batch and (offset != frame or len(batch) == size):
            yield frame, batch; batch = []
        frame = offset; batch.append(line)
    if batch:yield frame, batch


def quota(total, index, strata):
    return total // strata + (index < total % strata)


def publish_evidence(items, work, save, name):
    raw = b''.join(json.dumps(x, separators=(',', ':')).encode() + b'\n' for x in items)
    p = work / name; p.write_bytes(zstd.ZstdCompressor(level=3).compress(raw))
    digest = sha(p); temp = save / (name + '.tmp'); shutil.copyfile(p, temp)
    if sha(temp) != digest:raise OSError('Evidence persistence readback failed')
    target = save / name; temp.replace(target); p.unlink()
    return dict(name=name, rows=len(items), bytes=target.stat().st_size, sha256=digest,
                uncompressed_sha256=hashlib.sha256(raw).hexdigest())


def build_exclusions(plan, work, save, status):
    out = save / 'excluded-keys.npy'; proof = save / 'exclusion-audit.json'
    if out.exists() and proof.exists():
        audit = json.loads(proof.read_text())
        if sha(out) != audit['sha256']:raise ValueError('Persistent exclusions corrupted')
        local = work / out.name; shutil.copyfile(out, local)
        return np.load(local, mmap_mode='r', allow_pickle=False)
    base = Path(plan['local_prior_keys'])
    if sha(base) != plan['local_prior_sha256']:raise ValueError('Local prior keys changed')
    arrays = [np.load(base, allow_pickle=False)]; sources = []
    old = Path(plan['pawn_save'])
    complete = json.loads((old / 'ready.json').read_text())
    if not complete.get('complete') or complete['retained'] != 20_000_000:
        raise ValueError('Previous PAWN dataset is not complete')
    isolation = json.loads((old / 'heldout-isolation.json').read_text())
    for split in ['val', 'test']:
        p = old / f'{split}-all-keys.npy'
        if sha(p) != isolation['validation_key_sha256' if split == 'val' else 'test_key_sha256']:
            raise ValueError('PAWN full heldout keys changed')
        a = np.load(p, allow_pickle=False); arrays.append(a)
        sources.append(dict(path=str(p),keys=len(a),sha256=sha(p)))
    for i, item in enumerate(complete['shards']):
        rows = load_published(item['data'], old, work / 'old-cache')
        arrays.append(rows['key'].copy().view('V16').reshape(-1))
        sources.append(dict(path=str(old / item['data']['name']), keys=len(rows), sha256=item['data']['sha256']))
        del rows
        for p in (work / 'old-cache').glob('*'):p.unlink()
        if i % 25 == 0:status(phase='building-exclusions', indexed_old_shards=i+1, total_old_shards=len(complete['shards']))
    # Include the fixed labelled heldout samples as well (covered by full keys,
    # but explicitly verify the lineage and files used by the preceding run).
    for split in ['val','test']:
        meta=json.loads((old/f'{split}-sample.json').read_text())
        rows=load_published(meta,old,work/'old-cache');arrays.append(rows['key'].copy().view('V16').reshape(-1))
        del rows
        for p in (work/'old-cache').glob('*'):p.unlink()
    keys = np.unique(np.concatenate(arrays)); local = work / out.name
    np.save(local, keys, allow_pickle=False); del arrays, keys
    shutil.copyfile(local, out)
    digest = sha(local)
    if sha(out) != digest:raise OSError('Exclusion index persistence failed')
    result = dict(keys=len(np.load(local,mmap_mode='r')), sha256=digest,
                  local_prior_sha256=plan['local_prior_sha256'], pawn_full_heldout_keys=isolation['all_official_val_test_positions'],
                  pawn_train_keys=complete['retained'], sources=sources,
                  isolation='Exact canonical 128-bit keys; historical game independence unavailable')
    write_json(proof,result)
    shutil.copyfile(plan['local_prior_audit'],save/'local-exclusion-audit.json')
    return np.load(local,mmap_mode='r',allow_pickle=False)


def run(plan):
    work, save = Path(plan['work']), Path(plan['save'])
    if not os.path.ismount('/content/drive'):
        raise OSError('Persistent Drive mount required')
    work.mkdir(parents=True,exist_ok=True);save.mkdir(parents=True,exist_ok=True)
    lock=(work/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    started=time.time();phase='setup';ready=None;stats=Counter();last_status=0;db=None
    plan_digest=hashlib.sha256(json.dumps(plan,sort_keys=True).encode()).hexdigest()
    def status(**fields):
        row=dict(phase=phase,updated_at=time.time(),seconds=time.time()-started,
                 counts=ready['counts'] if ready else {s:0 for s in SPLITS},
                 target=plan['counts'],stats=dict(stats),**{k:v for k,v in fields.items() if k!='phase'})
        row['phase']=fields.get('phase',phase)
        write_json(work/'status.json',row);write_json(save/'status.json',row);print(json.dumps(row),flush=True)
    def flush(split, stratum):
        nonlocal buffers,evidence
        if not buffers[split]:return
        rows=np.array(buffers[split],dtype=DTYPE);audit_rows(rows)
        number=sum(x['split']==split for x in ready['shards'])
        meta=publish_array(rows,work,save,f'{split}-{number:06d}.npy')
        aux=publish_evidence(evidence[split],work,save,f'{split}-{number:06d}.jsonl.zst')
        profile=dict(depths=dict(Counter(str(x['evaluation']['depth']) for x in evidence[split])),
                     pvs=dict(Counter(str(len(x['evaluation']['pvs'])) for x in evidence[split])),
                     pieces=dict(Counter(str(int(np.count_nonzero(r['tokens']))) for r in rows)),
                     turn=dict(Counter('white' if r['meta'][9] else 'black' for r in rows)),
                     value=dict(Counter('win' if r['value'][0]>.75 else 'loss' if r['value'][2]>.75 else 'balanced' for r in rows)))
        ready['shards'].append(dict(split=split,stratum=stratum,data=meta,evidence=aux,profile=profile))
        ready['counts'][split]+=len(rows);ready['strata_counts'][str(stratum)][split]+=len(rows)
        ready['stats']=dict(stats);write_json(save/'ready.json',ready);db.commit()
        buffers[split]=[];evidence[split]=[]
    try:
        if (save/'plan.json').exists() and json.loads((save/'plan.json').read_text())!=plan:
            raise ValueError('Existing persistent output belongs to another plan')
        write_json(save/'plan.json',plan);write_json(save/'cleaning-spec.json',CLEANING_SPEC)
        # Source version was fixed before launch and is checked on every range.
        with urlopen(Request(plan['source']['url'],method='HEAD'),timeout=30) as response:
            if response.headers.get('ETag')!=plan['source']['etag'] or int(response.headers['Content-Length'])!=plan['source']['bytes']:
                raise ValueError('Pinned source no longer available')
        phase='building-exclusions';status()
        excluded=build_exclusions(plan,work,save,status)
        path=save/'ready.json'
        ready=json.loads(path.read_text()) if path.exists() else dict(schema=1,complete=False,plan_sha256=plan_digest,
             counts={s:0 for s in SPLITS},strata_counts={str(i):{s:0 for s in SPLITS} for i in range(plan['strata'])},shards=[],stats={})
        if ready['plan_sha256']!=plan_digest:raise ValueError('Resume plan mismatch')
        ready.setdefault('strata_goals', {});ready.setdefault('strata_completed', [])
        stats.update(ready['stats'])
        phase='reindexing-committed-shards';status()
        db=sqlite3.connect(work/'keys.sqlite');db.execute('PRAGMA journal_mode=WAL');db.execute('PRAGMA cache_size=-262144')
        db.execute('DROP TABLE IF EXISTS positions');db.execute('CREATE TABLE positions(key BLOB PRIMARY KEY,split TEXT NOT NULL) WITHOUT ROWID')
        for item in ready['shards']:
            rows=load_published(item['data'],save,work/'resume-cache')
            db.executemany('INSERT INTO positions VALUES (?,?)',((bytes(k),item['split']) for k in rows['key'].copy().view('V16').reshape(-1)))
            del rows
            for p in (work/'resume-cache').glob('*'):p.unlink()
        db.commit();buffers={s:[] for s in SPLITS};evidence={s:[] for s in SPLITS}
        rng=random.Random(plan['seed']);size=plan['source']['bytes'];strata=plan['strata']
        intervals=[]
        for i in range(strata):
            lo=size*i//strata;hi=size*(i+1)//strata
            intervals.append((lo,hi,rng.randint(lo,max(lo,hi-CHUNK))))
        write_json(save/'sampling-plan.json',dict(strata=intervals,seed=plan['seed'],
            sampling='accepted split quotas per compressed-byte stratum, deficits redistributed after exclusions, cyclic seeded start; not globally uniform'))
        phase='cleaning';status(exclusion_keys=len(excluded))
        with ProcessPoolExecutor(max_workers=plan['workers'],mp_context=multiprocessing.get_context('spawn'),
                                 initializer=init_clean_worker,initargs=(str(work/'excluded-keys.npy'),)) as pool:
            for index,(lo,hi,begin) in enumerate(intervals):
                if index in ready['strata_completed']:continue
                local=ready['strata_counts'][str(index)]
                if str(index) not in ready['strata_goals']:
                    ready['strata_goals'][str(index)]={s:local[s]+quota(plan['counts'][s]-ready['counts'][s],0,strata-index) for s in SPLITS}
                    write_json(save/'ready.json',ready)
                goals=ready['strata_goals'][str(index)]
                def done():return all(local[s]+len(buffers[s])>=goals[s] for s in SPLITS)
                if done():continue
                def inputs():
                    yield from batches(source_lines(plan['source'],begin,hi))
                    if begin>lo:yield from batches(source_lines(plan['source'],lo,begin))
                generator=iter(inputs());pending=deque()
                def submit():
                    try:payload=next(generator)
                    except StopIteration:return False
                    pending.append(pool.submit(convert_batch,payload));return True
                for _ in range(plan['workers']*2):
                    if not submit():break
                while pending and not done():
                    rows,items,counts=pending.popleft().result();stats.update(counts)
                    keys=rows['key'].copy().view('V16').reshape(-1);blocked=contains(excluded,keys)
                    stats['prior_or_heldout_overlap']+=int(blocked.sum())
                    for j in np.flatnonzero(~blocked):
                        key=bytes(keys[j]);split=split_key(key)
                        if local[split]+len(buffers[split])>=goals[split]:
                            stats['split_quota_full']+=1;continue
                        if not db.execute('INSERT OR IGNORE INTO positions VALUES (?,?)',(key,split)).rowcount:
                            stats['duplicate_position']+=1;continue
                        buffers[split].append(rows[j]);evidence[split].append(items[j])
                        if len(buffers[split])>=plan['shard_rows']:flush(split,index)
                        if done():break
                    if time.time()-last_status>=20:
                        status(stratum=index,strata_total=strata,pending_unpublished={s:len(buffers[s]) for s in SPLITS},
                               retained_per_second=sum(ready['counts'].values())/max(1,time.time()-started));last_status=time.time()
                    if not done():submit()
                for future in pending:future.cancel()
                # Running tasks must finish before this stratum's source generator
                # and its bounded input queue are discarded.
                for future in pending:
                    if not future.cancelled():future.result()
                generator.close()
                exhausted=not done()
                for split in SPLITS:flush(split,index)
                ready['strata_completed'].append(index);ready['stats']=dict(stats);write_json(save/'ready.json',ready)
                status(stratum=index,strata_total=strata,stratum_completed=True,source_stratum_exhausted=exhausted)
        if ready['counts']!=plan['counts']:raise ValueError('Exact dataset size not reached')
        phase='final-audit';status()
        all_keys=[];checked=Counter();recomputed=0
        for number,item in enumerate(ready['shards']):
            rows=load_published(item['data'],save,work/'audit-cache');audit_rows(rows)
            keys=rows['key'].copy().view('V16').reshape(-1)
            if contains(excluded,keys).any():raise ValueError('Historical/heldout leakage')
            if any(split_key(bytes(k))!=item['split'] for k in keys):raise ValueError('Non-deterministic split')
            all_keys.append(keys);checked[item['split']]+=len(rows)
            side=save/item['evidence']['name']
            if sha(side)!=item['evidence']['sha256']:raise OSError('Evidence final SHA mismatch')
            raw=zstd.ZstdDecompressor().decompress(side.read_bytes())
            if hashlib.sha256(raw).hexdigest()!=item['evidence']['uncompressed_sha256']:raise ValueError('Evidence decompression mismatch')
            lines=raw.splitlines()
            if len(lines)!=len(rows):raise ValueError('Evidence and training rows misaligned')
            # All PVs were checked before publication. Independently reconstruct
            # a deterministic sample per shard from retained original evidence.
            for j in range(0,len(rows),1024):
                info=json.loads(lines[j]);r,_=strict_convert(dict(fen=info['fen'],evals=[info['evaluation'],info['reference_pv1']]))
                if r.tobytes()!=rows[j].tobytes():raise ValueError('Evidence/target reconstruction mismatch')
                recomputed+=1
            del rows
            for p in (work/'audit-cache').glob('*'):p.unlink()
            if number%10==0:status(audited_shards=number+1,total_shards=len(ready['shards']))
        keys=np.concatenate(all_keys)
        if len(np.unique(keys))!=sum(plan['counts'].values()):raise ValueError('Duplicate or cross-split exact position')
        if dict(checked)!=plan['counts'] or db.execute('SELECT count(*) FROM positions').fetchone()[0]!=len(keys):
            raise ValueError('Manifest, SQLite and final shard counts differ')
        # Read back every persistent file before publishing complete=true.
        hashes={}
        for name in ['plan.json','cleaning-spec.json','sampling-plan.json','exclusion-audit.json','excluded-keys.npy','source.tar.gz']:
            hashes[name]=sha(save/name)
        audit=dict(passed=True,counts=dict(checked),unique_positions=len(keys),historical_or_heldout_overlap=0,
                   duplicate_keys=0,cross_split_position_overlap=0,all_training_rows_numeric_audit=True,
                   all_selected_pvs_legally_replayed_before_publication=True,reconstructed_rows=recomputed,
                   verified_persistent_files=hashes,game_independence='unavailable from evaluation cache',
                   default_promoted=False,training_started=False,finished_at=time.time(),seconds=time.time()-started)
        write_json(save/'audit.json',audit)
        ready['complete']=True;ready['stats']=dict(stats);write_json(save/'ready.json',ready)
        write_json(save/'manifest.json',dict(schema=1,counts=plan['counts'],compressed_shards=ready['shards'],
                  provenance=dict(source=plan['source'],license='CC0-1.0',cleaning=CLEANING_SPEC,
                  sampling='per-stratum accepted quotas with exhausted-stratum redistribution; not uniform over whole corpus',history_free=True)))
        write_json(save/'completion.json',audit);phase='complete';status()
    except BaseException:
        phase='failed';status(error=traceback.format_exc())
        write_json(save/'failure.json',dict(traceback=traceback.format_exc(),updated_at=time.time(),resume='Rerun the identical plan and pinned source; committed shards are reindexed'))
        raise
    finally:
        if db is not None:
            db.rollback();db.close()
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
        if (work/'console.log').exists():shutil.copyfile(work/'console.log',save/'console.log')
        if plan.get('release_when_done',True):
            from neurochess.autostop import request_unassign,find_runtime_address
            write_json(save/'shutdown-request.json',dict(phase=phase,requested_at=time.time()))
            request_unassign(find_runtime_address())


if __name__=='__main__':
    run(json.loads(Path(sys.argv[1]).read_text()))
