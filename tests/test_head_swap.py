import numpy as np
from scripts.diagnose_head_swap import SwappedNetwork


def test_value_intervention_uses_original_wdl_and_both_forward_passes():
    class Network:
        def __init__(self,wdl):self.wdl=wdl;self.calls=0
        def raw_outputs(self,tokens,meta,moves):
            self.calls+=1
            output=np.zeros((len(tokens),515),np.float32)
            output[:,512:]=self.wdl
            return output
    old=Network([0,0,1]);new=Network([1,0,0])
    moves=np.full((1,256),-1,np.int16);moves[0,:2]=[1,2]
    args=(np.zeros((1,64),np.uint8),np.zeros((1,12),np.float32),moves)
    _,q,value=SwappedNetwork([old,new],(1,1,0)).evaluate_search_heads(*args)
    assert value.tolist()==[-1] and np.allclose(q[0,:2],-.5)
    _,q,value=SwappedNetwork([old,new],(1,1,1)).evaluate_search_heads(*args)
    assert value.tolist()==[1] and np.allclose(q[0,:2],.5)
    assert old.calls==new.calls==2
    assert np.isnan(q[0,2:]).all()
