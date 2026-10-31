from types import SimpleNamespace
import chess
import numpy as np
from neurochess.inference_cache import CachedSearchNetwork
from neurochess.native_mcts import NativeMCTSPlayer


class Network:
    device='cpu'
    def __init__(self):
        self.model=SimpleNamespace(config=SimpleNamespace(history_free=True));self.positions=0
    def evaluate_search_heads(self,t,m,ids):
        self.positions+=len(t)
        return np.zeros(ids.shape,np.float32),np.full(ids.shape,np.nan,np.float32),t[:,0].astype(np.float32)/100


def inputs(n):
    return np.zeros((n,64),np.uint8),np.zeros((n,12),np.float32),np.zeros((n,256),np.int16)


def test_cache_deduplicates_effective_inputs_and_keeps_returned_arrays_independent():
    base=Network();net=CachedSearchNetwork(base,2);t,m,ids=inputs(3)
    m[:,5]=[0,1,2]
    p,q,v=net.evaluate_search_heads(t,m,ids);assert base.positions==1
    p[:]=100;v[:]=1
    p,q,v=net.evaluate_search_heads(t,m,ids)
    assert base.positions==1 and (p==0).all() and (v==0).all()
    ids[0,0]=5;net.evaluate_search_heads(t,m,ids);assert base.positions==2
    m[0,8]=1;net.evaluate_search_heads(t,m,ids);assert base.positions==3
    assert len(net.cache)==2
    net.clear_cache();net.evaluate_search_heads(t,m,ids);assert base.positions==5


def test_history_sensitive_models_do_not_mask_metadata():
    base=Network();base.model.config.history_free=False;net=CachedSearchNetwork(base)
    t,m,ids=inputs(2);m[1,6]=1;net.evaluate_search_heads(t,m,ids);assert base.positions==2


def test_player_cache_clear_discards_retained_tree_and_forces_new_evaluations():
    base=Network();p=NativeMCTSPlayer(base,simulations=31,leaf_batch_size=4,inference_cache_size=1024)
    b=chess.Board();p.choose(b);before=base.positions;p.choose(b)
    assert p.search_stats['tree_reuses']==1
    p.clear_cache();p.choose(b)
    assert p.search_stats['tree_reuses']==1 and base.positions>before
    assert p.search_stats['inference_cache']['cache_hits']>0
