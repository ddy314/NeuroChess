"""Exact effective-input memoization for frozen search networks; no tree merging."""
from collections import OrderedDict
import numpy as np


class CachedSearchNetwork:
    def __init__(self,network,capacity=32768):
        if capacity<1:raise ValueError('Positive inference cache capacity required')
        self.network=network;self.model=getattr(network,'model',None);self.device=getattr(network,'device','cpu')
        self.capacity=capacity;self.cache=OrderedDict()
        self.search_stats=dict(requested_positions=0,forward_positions=0,cache_hits=0,batch_duplicates=0)

    def clear_cache(self):self.cache.clear()

    def evaluate_search_heads(self,tokens,meta,ids):
        n=len(tokens);p=np.empty((n,256),np.float32);q=np.empty_like(p);v=np.empty(n,np.float32)
        effective=np.array(meta,dtype=np.float32,copy=True)
        if getattr(getattr(self.model,'config',None),'history_free',False):effective[:,[5,6,7,10]]=0
        keys=[tokens[i].tobytes()+effective[i].tobytes()+ids[i].tobytes() for i in range(n)]
        missing={};hits=0
        for i,key in enumerate(keys):
            if key in self.cache:
                p[i],q[i],v[i]=self.cache[key];self.cache.move_to_end(key);hits+=1
            else:missing.setdefault(key,[]).append(i)
        if missing:
            representatives=[indexes[0] for indexes in missing.values()]
            inputs=tokens[representatives],meta[representatives],ids[representatives]
            if hasattr(self.network,'evaluate_search_heads'):mp,mq,mv=self.network.evaluate_search_heads(*inputs)
            else:
                mp,mv=self.network.evaluate_encoded(*inputs);mq=np.full_like(mp,np.nan)
            for j,(key,indexes) in enumerate(missing.items()):
                entry=(np.array(mp[j],np.float32,copy=True),np.array(mq[j],np.float32,copy=True),float(mv[j]))
                self.cache[key]=entry
                for i in indexes:p[i],q[i],v[i]=entry
            while len(self.cache)>self.capacity:self.cache.popitem(last=False)
        self.search_stats['requested_positions']+=n;self.search_stats['forward_positions']+=len(missing)
        self.search_stats['cache_hits']+=hits;self.search_stats['batch_duplicates']+=n-hits-len(missing)
        return p,q,v
