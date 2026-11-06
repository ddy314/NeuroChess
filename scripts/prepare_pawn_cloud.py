"""Convert exactly20M fresh PAWN roots, with complete official heldout exclusion."""
import argparse
import concurrent.futures
import fcntl
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import sqlite3
import sys
import time
import traceback

import numpy as np
import pyarrow.parquet as pq
import requests
from huggingface_hub import HfFileSystem

from neurochess.data import DTYPE
from neurochess.pawn_data import (SOURCE, REVISION, VOCABULARY_SHA256, RAW_DTYPE,
    PROVENANCE, sha, write_json, keys_from_games, convert_games, contains,
    exact_filter, publish_array, load_published)


def batches(fs, path, columns, size=1024):
    remote = f'datasets/{SOURCE}@{REVISION}/{path}'
    # One open iterator consumes whole row groups instead of repeatedly sampling
    # the first128 rows and paying their read amplification.
    with fs.open(remote, 'rb', block_size=8*1024*1024) as stream:
        parquet = pq.ParquetFile(stream)
        for i, batch in enumerate(parquet.iter_batches(batch_size=size, columns=columns)):
            yield i, batch.to_pylist()


def main(plan_path):
    plan = json.loads(Path(plan_path).read_text())
    work, save = Path(plan['work']), Path(plan['save'])
    if not os.path.ismount('/content/drive'):
        raise OSError('Persistent Drive mount required before conversion')
    work.mkdir(parents=True, exist_ok=True); save.mkdir(parents=True, exist_ok=True)
    lock = (work/'run.lock').open('w'); fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    write_json(save/'plan.json', plan)
    phase = 'setup'; started = time.time(); stats = {}; ready = None
    def status(**extra):
        row = dict(phase=phase, updated_at=time.time(), elapsed_seconds=time.time()-started,
                   target_train=plan['positions'], retained=ready['retained'] if ready else 0,
                   **stats, **extra)
        write_json(work/'status.json', row); write_json(save/'status.json', row)
        print(json.dumps(row), flush=True)
    try:
        if sha(plan['prior_keys']) != plan['prior_sha256']:
            raise ValueError('Prior exclusion index checksum mismatch')
        prior = np.load(plan['prior_keys'], mmap_mode='r', allow_pickle=False)
        if prior.dtype != np.dtype('V16') or not np.array_equal(prior, np.sort(prior)):
            raise ValueError('Sorted exact prior index required')
        prior_save = save/'known-prior-keys.npy'
        if not prior_save.exists(): shutil.copyfile(plan['prior_keys'], prior_save)
        if sha(prior_save) != plan['prior_sha256']: raise OSError('Prior index persistence failed')
        response = requests.get('https://raw.githubusercontent.com/thomas-schweich/PAWN/main/searchless_chess_vocabulary.json',timeout=60)
        response.raise_for_status()
        if __import__('hashlib').sha256(response.content).hexdigest() != VOCABULARY_SHA256:
            raise ValueError('Action vocabulary drift')
        vocabulary = response.json()['action_to_move']
        (save/'vocabulary.json').write_bytes(response.content)
        response = requests.get(f'https://huggingface.co/api/datasets/{SOURCE}/revision/{REVISION}',timeout=90)
        response.raise_for_status(); repo=response.json()
        all_files=[x['rfilename'] for x in repo['siblings'] if x['rfilename'].endswith('.parquet')]
        files={split:sorted(p for p in all_files if p.startswith(f'{directory}/nodes_1024/'))
               for split,directory in [('train','train'),('val','val'),('test','test')]}
        if not all(files.values()): raise ValueError('Missing official split')
        rng=np.random.default_rng(plan['seed']); files['train']=[files['train'][i] for i in rng.permutation(len(files['train']))]
        write_json(save/'source.json',dict(revision=REVISION,files=files,vocabulary_sha256=VOCABULARY_SHA256,provenance=PROVENANCE))
        fs=HfFileSystem()
        with concurrent.futures.ProcessPoolExecutor(max_workers=plan['workers'],
             mp_context=multiprocessing.get_context('spawn')) as pool:
            # Reserve EVERY position from all official validation/test games,
            # not only the small labelled evaluation sample.
            phase='building-full-heldout-index'; status()
            split_keys={}
            for split in ['val','test']:
                final=save/f'{split}-all-keys.npy'
                if not final.exists():
                    chunks=[]; games_seen=0
                    for source in files[split]:
                        for _, games in batches(fs,source,['uci','global_game_index'],size=2048):
                            jobs=[games[i:i+64] for i in range(0,len(games),64)]
                            chunks.extend(pool.map(keys_from_games,jobs));games_seen+=len(games)
                            status(index_split=split,index_games=games_seen,index_positions=sum(map(len,chunks)))
                    unique=np.unique(np.concatenate(chunks));np.save(final,unique,allow_pickle=False)
                    write_json(save/f'{split}-key-audit.json',dict(games=games_seen,unique_positions=len(unique),sha256=sha(final)))
                    if games_seen != 50000: raise ValueError('Official heldout game count differs')
                split_keys[split]=np.load(final,mmap_mode='r',allow_pickle=False)
            heldout=np.union1d(split_keys['val'],split_keys['test'])
            write_json(save/'heldout-isolation.json',dict(complete=True,all_official_val_test_positions=len(heldout),
                prior_keys=len(prior),prior_sha256=plan['prior_sha256'],
                validation_key_sha256=sha(save/'val-all-keys.npy'),test_key_sha256=sha(save/'test-all-keys.npy'),
                scope='All100000 official nodes_1024 heldout games, plus known historical exact position keys. Old source-game independence remains unknown.'))
            columns=['uci','game_length','cp_evals','nnue_evals','global_game_index','nodes']
            # A modest fixed labelled subset, still isolated against the ENTIRE
            # opposite heldout split and all known prior training/evaluation.
            phase='preparing-fixed-evaluation';status()
            for split in ['test','val']:
                meta=save/f'{split}-sample.json'
                if meta.exists():continue
                selected=[];seen=set();opposite=split_keys['val' if split=='test' else 'test']
                for source in files[split]:
                    for _,games in batches(fs,source,columns,size=128):
                        jobs=[(games[i:i+16],vocabulary) for i in range(0,len(games),16)]
                        for rows,raw,groups,plies,_ in pool.map(convert_games,jobs):
                            keys=rows['key'].copy().view('V16').reshape(-1)
                            candidates=np.flatnonzero(~contains(prior,keys)&~contains(opposite,keys)&(plies%16==8))
                            for i in candidates:
                                key=bytes(keys[i])
                                if key not in seen and len(selected)<plan['eval_positions']:
                                    selected.append(rows[i]);seen.add(key)
                        if len(selected)>=plan['eval_positions']:break
                    if len(selected)>=plan['eval_positions']:break
                if len(selected)!=plan['eval_positions']:raise ValueError('Insufficient independent heldout samples')
                array=np.array(selected,dtype=DTYPE)
                report=publish_array(array,work,save,f'{split}-sample.npy');write_json(meta,report)
            ready_path=save/'ready.json'
            ready=json.loads(ready_path.read_text()) if ready_path.exists() else dict(schema=1,retained=0,shards=[],cursor=[0,0],complete=False,
                provenance=PROVENANCE,heldout_isolation_complete=True,target_train=plan['positions'],stats={})
            if ready['target_train']!=plan['positions'] or ready['provenance']!=PROVENANCE:
                raise ValueError('Resume plan/data definition mismatch')
            stats=ready.get('stats',{})
            # Reconstruct exact dedup state from committed shards, so a crash
            # between file publication and SQLite commit cannot lose records.
            db=sqlite3.connect(work/'dedup.sqlite')
            db.execute('PRAGMA journal_mode=WAL');db.execute('PRAGMA cache_size=-262144')
            db.execute('DROP TABLE IF EXISTS seen');db.execute('CREATE TABLE seen(key BLOB PRIMARY KEY) WITHOUT ROWID')
            for item in ready['shards']:
                rows=load_published(item['data'],save,work/'resume-cache')
                db.executemany('INSERT INTO seen VALUES (?)',((bytes(k),) for k in rows['key'].copy().view('V16').reshape(-1)))
                del rows
                for p in (work/'resume-cache').glob('*'):p.unlink()
            db.commit()
            phase='converting';status()
            for file_index,source in enumerate(files['train']):
                if ready['retained']>=plan['positions']:break
                if file_index<ready['cursor'][0]:continue
                for batch_index,games in batches(fs,source,columns,size=1024):
                    if file_index==ready['cursor'][0] and batch_index<ready['cursor'][1]:continue
                    if ready['retained']>=plan['positions']:break
                    begin=time.time();outputs=list(pool.map(convert_games,[(games[i:i+64],vocabulary) for i in range(0,len(games),64)]))
                    rows=np.concatenate([r[0] for r in outputs]);raw=np.concatenate([r[1] for r in outputs])
                    groups=np.concatenate([r[2] for r in outputs]);plies=np.concatenate([r[3] for r in outputs])
                    kept,counts=exact_filter(rows,prior,heldout,db,plan['positions']-ready['retained'])
                    for key in ['scanned','unlabelled']:stats[key]=stats.get(key,0)+sum(r[4][key] for r in outputs)
                    for key,value in counts.items():stats[key]=stats.get(key,0)+value
                    if len(kept):
                        number=len(ready['shards']);data=publish_array(rows[kept],work,save,f'train-{number:06d}.npy')
                        aux_dtype=np.dtype([('raw',RAW_DTYPE),('game','<u8'),('ply','<u2')]);aux=np.empty(len(kept),dtype=aux_dtype)
                        aux['raw']=raw[kept];aux['game']=groups[kept];aux['ply']=plies[kept]
                        extra=publish_array(aux,work,save,f'aux-{number:06d}.npy')
                        ready['shards'].append(dict(data=data,aux=extra,source=source,batch=batch_index,
                                                   game_min=int(groups.min()),game_max=int(groups.max())))
                        ready['retained']+=len(kept)
                    ready['cursor']=[file_index,batch_index+1];ready['stats']=stats
                    write_json(ready_path,ready);db.commit()
                    status(last_batch_retained=len(kept),last_batch_seconds=time.time()-begin,
                           last_processing_retained_per_second=len(kept)/max(.001,time.time()-begin),shards=len(ready['shards']))
                    del rows,raw,groups,plies,outputs
                ready['cursor']=[file_index+1,0];write_json(ready_path,ready)
            if ready['retained']!=plan['positions']:raise ValueError('Source exhausted before target')
            phase='final-audit';status()
            count=db.execute('SELECT count(*) FROM seen').fetchone()[0]
            if count!=plan['positions'] or sum(s['data']['rows'] for s in ready['shards'])!=count:
                raise ValueError('Final exact count/dedup audit failed')
            for item in ready['shards']:
                for field in ['data','aux']:
                    if sha(save/item[field]['name'])!=item[field]['sha256']:raise OSError('Final persistent SHA mismatch')
            ready['complete']=True;write_json(ready_path,ready);db.close()
            write_json(save/'completion.json',dict(train=count,unique_train=count,heldout_position_overlap=0,prior_position_overlap=0,
                compressed_bytes=sum(s[k]['bytes'] for s in ready['shards'] for k in ['data','aux']),
                validation=plan['eval_positions'],test=plan['eval_positions'],completed_at=time.time(),
                elapsed_seconds=time.time()-started,default_promoted=False))
            phase='complete';status()
    except BaseException:
        phase='failed';write_json(save/'failure.json',dict(traceback=traceback.format_exc(),updated_at=time.time()));status();raise
    finally:
        log=work/'console.log'
        if log.exists():shutil.copyfile(log,save/'console.log')
        if plan.get('release_when_done',True):
            from neurochess.autostop import request_unassign,find_runtime_address
            write_json(save/'shutdown-request.json',dict(phase=phase,requested_at=time.time()))
            request_unassign(find_runtime_address())


if __name__=='__main__':
    main(sys.argv[1])
