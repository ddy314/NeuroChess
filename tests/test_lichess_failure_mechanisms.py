import unittest
from types import SimpleNamespace
import numpy as np
import torch
from scripts.lichess_failure_mechanisms import Mixed,effective_key
from neurochess.multihead import fuse_heads

class FakeNet:
    def __init__(self,wdl,q):
        self.model=SimpleNamespace(value=torch.nn.Sequential(torch.nn.Linear(1,1)),config=SimpleNamespace(history_free=True))
        self.wdl=wdl;self.q=q;self.calls=0
    def raw_outputs(self,tokens,meta,moves):
        self.calls+=1;r=np.zeros((len(tokens),515),np.float32);r[:,:256]=-1e9
        r[:,:2]=[1,0];r[:,256:258]=self.q;r[:,512:]=self.wdl
        return r

class MixTests(unittest.TestCase):
    def setUp(self):
        self.old=FakeNet([.1,.2,.7],[.2,-.4]);self.new=FakeNet([.6,.3,.1],[.3,-.2]);self.nets=[self.old,self.new]
        self.t=np.zeros((1,64),np.uint8);self.m=np.zeros((1,12),np.float32);self.ids=np.full((1,256),-1,np.int16);self.ids[0,:2]=[1,2]
    def evaluate(self,route,source=(1,1,1)):
        net=Mixed(self.nets,source,route)
        try:return net.evaluate_search_heads(self.t,self.m,self.ids)
        finally:net.close()
    def test_leaf_value_route_preserves_candidate_policy_and_Q_seed(self):
        base=self.evaluate('normal');other=self.evaluate('old_leaf_FPU')
        np.testing.assert_array_equal(base[0],other[0]);np.testing.assert_allclose(base[1],other[1],equal_nan=True)
        self.assertAlmostEqual(other[2][0],-.6,places=6)
        self.assertEqual(self.old.calls,self.new.calls)
    def test_Q_seed_value_route_changes_neither_leaf_nor_fused_policy(self):
        base=self.evaluate('normal');other=self.evaluate('old_value_Q_seed')
        np.testing.assert_array_equal(base[0],other[0]);np.testing.assert_array_equal(base[2],other[2])
        np.testing.assert_allclose(other[1][0,:2],[-.15,-.4],atol=1e-6)
        self.assertTrue(np.isnan(other[1][0,2:]).all())
    def test_Value_output_swap_preserves_both_complete_model_calls(self):
        p,q,v=self.evaluate('normal',(1,1,0));self.assertAlmostEqual(v[0],-.6,places=6)
        self.assertEqual((self.old.calls,self.new.calls),(1,1))
    def test_history_mask_matches_frozen_inference_cache(self):
        a=self.m[0].copy();b=a.copy();b[[5,6,7,10]]=1
        self.assertEqual(effective_key(self.t[0],a,self.ids[0]),effective_key(self.t[0],b,self.ids[0]))

if __name__=='__main__':unittest.main()
