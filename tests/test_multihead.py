import chess
import numpy as np
from neurochess.encoding import move_id
from neurochess.multihead import fuse_heads
from neurochess.native_mcts import NativeMCTSPlayer


def test_q_only_affects_policy_supported_legal_candidates():
    p=np.array([[2.,1.9,1.8,1.7,1.6,-5.,-1e9]],np.float32)
    q=np.array([[-.3,.7,0,0,0,1.,1.]],np.float32)
    fused,prior,value,support=fuse_heads(p,q,np.array([[.7,.2,.1]]),np.array([[1,1,1,1,1,1,0]],bool))
    assert fused.argmax(1)[0]==1
    assert not support[0,5:].any() and np.isnan(prior[0,5:]).all()
    assert fused[0,5]==p[0,5] and value[0]>.59
    q[0,5:]=-1
    other=fuse_heads(p,q,np.array([[.7,.2,.1]]),np.array([[1,1,1,1,1,1,0]],bool))
    np.testing.assert_array_equal(fused,other[0])


def test_value_confidence_controls_q_reranking():
    p=np.array([[2.,1.7]],np.float32);q=np.array([[0.,.4]],np.float32);mask=np.ones((1,2),bool)
    certain=fuse_heads(p,q,np.array([[1.,0,0]]),mask)[0]
    uncertain=fuse_heads(p,q,np.array([[1/3]*3]),mask)[0]
    assert certain.argmax()==1 and uncertain.argmax()==0


def test_native_search_consumes_neural_q_in_parent_perspective():
    favorite=move_id(chess.Move.from_uci('e2e4'),True)
    class Heads:
        device='cpu'
        def evaluate_search_heads(self,tokens,meta,ids):
            logits=np.zeros(ids.shape,np.float32)
            q=np.where(ids==favorite,.9,np.nan).astype(np.float32)
            return logits,q,np.zeros(len(tokens),np.float32)
    player=NativeMCTSPlayer(Heads(),simulations=1,leaf_batch_size=1,initial_width=256)
    assert player.choose(chess.Board())==chess.Move.from_uci('e2e4')
    player.close()
