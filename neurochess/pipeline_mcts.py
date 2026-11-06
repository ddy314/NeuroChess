"""Experimental LC0-style immediate cache backup and miss-only GPU batches.

Uses an independent native library. Rule states and proofs remain in their
own nodes. Stop/cancel restores every outstanding virtual reservation.
"""
from collections import OrderedDict
import ctypes as C
import json,time
from pathlib import Path
import chess,numpy as np
from .native_mcts import NativeMCTSPlayer,history


def pipeline_library(reference):
    path=Path(__file__).with_name('native')/'libmcts_pipeline.so'
    if not path.exists() or path.with_name('mcts_pipeline.cpp').stat().st_mtime>path.stat().st_mtime:raise RuntimeError('Build scripts/build_native_pipeline.py first')
    lib=C.CDLL(str(path))
    for name,fn in vars(reference).items():
        if name.startswith('nc_'):
            target=getattr(lib,name);target.argtypes=fn.argtypes;target.restype=fn.restype
    ptr=C.c_void_p
    lib.nc_append.argtypes=[ptr,C.c_int,ptr,ptr,ptr,ptr,ptr]
    lib.nc_submit_some.argtypes=[ptr,ptr,ptr,ptr,ptr,C.c_int]
    lib.nc_cancel_pending.argtypes=[ptr];lib.nc_root_ready.argtypes=[ptr]
    return lib


class PipelineMCTSPlayer(NativeMCTSPlayer):
    def __init__(self,network,**options):
        original=NativeMCTSPlayer(network,**options);state=original.__dict__.copy();lib=pipeline_library(original.lib);original.close()
        self.__dict__.update(state);self.lib=lib;self.handle=lib.nc_create(self._config)
        if not self.handle:raise ValueError(lib.nc_error().decode())
        self._check(lib.nc_batch_safety(self.handle,int(self.pending_loss_guard)))
        self.network=network;self.cache=OrderedDict();self.jobids=np.empty(self._config.batch,np.int32)
        self.cache_stats=dict(requested_positions=0,forward_positions=0,cache_hits=0,batch_duplicates=0)
        self.physical_batches=0

    @property
    def config(self):return dict(super().config,out_of_order_cache=True,gpu_batch='unique misses only; hits backed up immediately')

    def clear_cache(self):
        self.cache.clear()
        if self.handle:self._check(self.lib.nc_forget(self.handle))

    def _submit(self,entries):
        jobids=np.ascontiguousarray([x[0] for x in entries],np.int32)
        p=np.ascontiguousarray([x[1][0] for x in entries],np.float32);q=np.ascontiguousarray([x[1][1] for x in entries],np.float32);v=np.ascontiguousarray([x[1][2] for x in entries],np.float32)
        tic=time.perf_counter();self._check(self.lib.nc_submit_some(self.handle,jobids.ctypes.data,p.ctypes.data,q.ctypes.data,v.ctypes.data,len(entries)))
        self.timing['native_seconds']+=time.perf_counter()-tic

    def choose_many(self,boards):
        try:
            return self._choose_many(boards)
        finally:
            # Network/encoding failures must not strand virtual losses and
            # prevent subsequent reuse or clean shutdown.
            if self.handle:self._check(self.lib.nc_cancel_pending(self.handle))

    def _choose_many(self,boards):
        if len(boards)!=1 or not self.handle:raise ValueError('Pipeline supports one open game slot')
        b=boards[0];start=time.perf_counter();tic=start
        if not self.tree_reuse:self._check(self.lib.nc_forget(self.handle))
        self._check(self.lib.nc_resize(self.handle,1));self._check(self.lib.nc_start(self.handle,0,*history(b)))
        self._check(self.lib.nc_time_progress(self.handle,-1 if self.time_limit is None else 0));self.timing['bridge_seconds']+=time.perf_counter()-tic
        pool={};longest=0.;stopped=False;complete=False
        history_free=getattr(getattr(getattr(self.network,'model',None),'config',None),'history_free',False)
        while not complete:
            if self.time_limit is not None and self.lib.nc_root_ready(self.handle) and time.perf_counter()-start>=self.time_limit-max(self.time_margin,1.15*longest):
                stopped=True;self.time_stops+=1;break
            round_start=time.perf_counter();added=0
            while len(pool)<self._config.batch:
                tic=time.perf_counter()
                if self.time_limit is not None:self._check(self.lib.nc_time_progress(self.handle,min(1.,(tic-start)/self.time_limit)))
                n=self._check(self.lib.nc_append(self.handle,max(1,self._config.batch-len(pool)),self.tokens.ctypes.data,self.meta.ctypes.data,self.ids.ctypes.data,self.counts.ctypes.data,self.jobids.ctypes.data))
                self.timing['native_seconds']+=time.perf_counter()-tic
                if not n:break
                added+=n;hits=[]
                for i in range(n):
                    meta=self.meta[i].copy()
                    if history_free:meta[[5,6,7,10]]=0
                    key=self.tokens[i].tobytes()+meta.tobytes()+self.ids[i].tobytes();job=int(self.jobids[i]);self.cache_stats['requested_positions']+=1
                    if self.inference_cache_size and key in self.cache:
                        hits.append((job,self.cache[key]));self.cache.move_to_end(key);self.cache_stats['cache_hits']+=1
                    elif self.inference_cache_size and key in pool:
                        pool[key]['jobs'].append(job);self.cache_stats['batch_duplicates']+=1
                    else:
                        if not self.inference_cache_size:key=(job,key)
                        pool[key]=dict(jobs=[job],tokens=self.tokens[i].copy(),meta=self.meta[i].copy(),ids=self.ids[i].copy())
                if hits:self._submit(hits)
                if added>=self._config.batch*8:break
                if self.time_limit is not None and self.lib.nc_root_ready(self.handle) and time.perf_counter()-start>=self.time_limit-max(self.time_margin,1.15*longest):break
            if not pool:
                if not added:complete=True
                longest=max(longest,time.perf_counter()-round_start)
                continue
            if self.time_limit is not None and self.lib.nc_root_ready(self.handle) and time.perf_counter()-start>=self.time_limit-max(self.time_margin,1.15*longest):
                stopped=True;self.time_stops+=1;break
            t=np.stack([x['tokens'] for x in pool.values()]);m=np.stack([x['meta'] for x in pool.values()]);ids=np.stack([x['ids'] for x in pool.values()]);tic=time.perf_counter()
            if hasattr(self.network,'evaluate_search_heads'):p,q,v=self.network.evaluate_search_heads(t,m,ids)
            else:p,v=self.network.evaluate_encoded(t,m,ids);q=np.full_like(p,np.nan)
            if np.shape(p)!=(len(pool),256) or np.shape(q)!=(len(pool),256) or np.shape(v)!=(len(pool),):
                raise ValueError('Expected policy/Q (batch,256) and values (batch,)')
            self.timing['inference_seconds']+=time.perf_counter()-tic;self.cache_stats['forward_positions']+=len(pool)
            self.physical_batches+=1
            entries=[]
            for i,(key,item) in enumerate(pool.items()):
                value=(p[i].copy(),q[i].copy(),float(v[i]))
                if self.inference_cache_size:self.cache[key]=value
                entries.extend((job,value) for job in item['jobs'])
            while len(self.cache)>self.inference_cache_size:self.cache.popitem(last=False)
            self._submit(entries);pool.clear();longest=max(longest,time.perf_counter()-round_start)
        self._check(self.lib.nc_cancel_pending(self.handle))
        result=json.loads(self.lib.nc_result(self.handle));self.search_stats=result['stats']|self.timing|dict(time_stops=self.time_stops,physical_inference_batches=self.physical_batches,inference_cache=dict(self.cache_stats))
        root=result['roots'][0];self.last_result=dict(root,elapsed_seconds=time.perf_counter()-start,time_stopped=stopped,time_limit=self.time_limit)
        if len(self.decision_samples)<8:self.decision_samples.append(dict(fen=b.fen(),**root))
        move=chess.Move.from_uci(root['move'])
        if move not in b.legal_moves:raise RuntimeError('Illegal pipeline result')
        return [move]
