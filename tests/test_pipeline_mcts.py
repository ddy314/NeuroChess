import chess,numpy as np
import pytest
from neurochess.native_mcts import NativeMCTSPlayer
from neurochess.pipeline_mcts import PipelineMCTSPlayer


class Net:
    device='cpu'
    def evaluate_encoded(self,t,m,ids):return np.zeros(ids.shape,np.float32),np.zeros(len(t),np.float32)


def test_no_cache_pipeline_matches_full_batch_reference_and_budget():
    b=chess.Board();options=dict(simulations=128,leaf_batch_size=8,pending_loss_guard=True)
    a=NativeMCTSPlayer(Net(),**options);p=PipelineMCTSPlayer(Net(),**options)
    assert a.choose(b)==p.choose(b)
    assert a.last_result['leaders']==p.last_result['leaders']
    assert p.search_stats['completed_simulations']==128


def test_pipeline_cached_partial_backups_and_cancel_leave_reusable_tree():
    b=chess.Board();p=PipelineMCTSPlayer(Net(),simulations=4096,leaf_batch_size=8,inference_cache_size=1024,time_limit=.015,time_margin=.002,pending_loss_guard=True)
    for _ in range(3):
        assert p.choose(b) in b.legal_moves
        assert p.search_stats['inference_cache']['requested_positions']>=p.search_stats['inference_cache']['forward_positions']
    assert p.search_stats['tree_reuses']==2
    p.clear_cache();p.choose(b);assert p.search_stats['tree_reuses']==2


def test_failed_network_call_cancels_reservations_and_can_resume():
    class Failing(Net):
        fail=True
        def evaluate_encoded(self,t,m,ids):
            if self.fail:raise RuntimeError('temporary inference failure')
            return super().evaluate_encoded(t,m,ids)
    net=Failing();p=PipelineMCTSPlayer(net,simulations=64,leaf_batch_size=8)
    with pytest.raises(RuntimeError,match='temporary inference failure'):p.choose(chess.Board())
    net.fail=False
    assert p.choose(chess.Board()) in chess.Board().legal_moves
    assert p.search_stats['completed_simulations']==64
    p.close()
