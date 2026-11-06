import unittest
import chess
import numpy as np
import torch
from neurochess.multihead import fuse_heads
from neurochess.repair_pilot import fused_tensors,functional_anchor,ranking_pairs,pair_loss,qualify_label,labeled_row,complete_analysis

class RepairTests(unittest.TestCase):
    def test_differentiable_interface_matches_locked_inference(self):
        rng=np.random.default_rng(20);p=torch.tensor(rng.normal(size=(3,256)),dtype=torch.float32);q=p.tanh();v=torch.tensor(rng.normal(size=(3,3)),dtype=torch.float32)
        moves=torch.full((3,256),-1);moves[:,:8]=torch.arange(8)
        got=fused_tensors((p,q,v),moves);expected=fuse_heads(p.numpy(),q.numpy(),v.softmax(-1).numpy(),moves.numpy()>=0)
        np.testing.assert_allclose(got[0].numpy(),expected[0],atol=2e-6)
        np.testing.assert_allclose(got[1].numpy()[expected[3]],expected[1][expected[3]],atol=2e-6)
        np.testing.assert_allclose(got[2].numpy(),expected[2],atol=1e-6)
        np.testing.assert_array_equal(got[3].numpy(),expected[3])
    def test_anchor_zero_and_normalized_over_microbatches(self):
        torch.manual_seed(3);out=tuple(torch.randn(4,n,requires_grad=True) for n in [256,256,3]);ref=tuple(a.detach()+.1*torch.randn_like(a) for a in out)
        moves=torch.full((4,256),-1);moves[:,:5]=torch.arange(5);mask=torch.tensor([True,False,True,True]);n=int(mask.sum())
        zero=functional_anchor(out,tuple(a.detach() for a in out),moves,mask,n);self.assertAlmostEqual(float(zero.detach()),0,places=6)
        full=functional_anchor(out,ref,moves,mask,n)
        split=sum(functional_anchor(tuple(a[i:i+2] for a in out),tuple(a[i:i+2] for a in ref),moves[i:i+2],mask[i:i+2],n) for i in [0,2])
        torch.testing.assert_close(full,split);full.backward();self.assertTrue(all(a.grad is not None and torch.isfinite(a.grad).all() for a in out))
    def test_policy_rank_uses_large_cp_and_skips_mate_or_unknown(self):
        cp=np.array([[200.,0.,np.nan],[0.,10.,np.nan]],np.float32);pairs=ranking_pairs(cp)
        self.assertEqual([x.tolist() for x in pairs],[[0],[0],[1]])
        p=torch.zeros(2,3,requires_grad=True);loss=pair_loss(p,pairs,0,2,1);loss.backward();self.assertLess(float(p.grad[0,0]),0);self.assertGreater(float(p.grad[0,1]),0)
    def test_black_score_targets_keep_side_to_move_perspective(self):
        fen=chess.STARTING_FEN.replace(' w ',' b ');rows=[dict(move='e7e5',cp=100,mate=None,sort_score=100,pv=['e7e5']),dict(move='d7d5',cp=-100,mate=None,sort_score=-100,pv=['d7d5'])]
        r,cp=labeled_row(fen,rows);self.assertGreater(float(r['value'][0]-r['value'][2]),0);self.assertTrue((cp[np.isfinite(cp)]==[100,-100]).all())
    def test_qualification_refuses_missing_global_or_unqualified_depth(self):
        def row(m,cp):return dict(move=m,cp=cp,mate=None,sort_score=cp,depth=24,nodes=5000000,lowerbound=False,upperbound=False,pv=[m])
        a=[row('e2e4',100),row('d2d4',0)];self.assertTrue(qualify_label(a,a,[a[0]],[a[0]])[0])
        self.assertFalse(qualify_label(a,a,[row('g1f3',200)],[a[0]])[0])
        b=[dict(a[0],depth=23),a[1]];self.assertFalse(qualify_label(a,b,[a[0]],[a[0]])[0])
    def test_complete_iteration_ignores_incomplete_deeper_bound(self):
        class Fake:
            def send(self,s):pass
            def until(self,prefix):
                if prefix=='readyok':return iter([])
                return iter(['info depth 24 multipv 1 score cp 50 nodes 2000000 pv e2e4','info depth 24 multipv 2 score cp 40 nodes 2100000 pv d2d4','info depth 25 multipv 1 score cp 80 lowerbound nodes 2200000 pv e2e4'])
        r=complete_analysis(Fake(),chess.Board(),3000000,['e2e4','d2d4']);self.assertEqual([x['depth'] for x in r],[24,24])

if __name__=='__main__':unittest.main()
