import chess
import numpy as np
import pytest
import torch
from neurochess.encoding import encode,legal_ids
from neurochess.model import ChessModel,ModelConfig
from neurochess.multihead import MultiheadPlayer
from neurochess.cuda_inference import CudaGraphMultiheadPlayer


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_graph_outputs_match_eager_across_changing_data_and_shape_order():
    torch.manual_seed(42)
    model=ChessModel(ModelConfig(width=32,layers=1,heads=4,head_width=16)).cuda().eval()
    eager=MultiheadPlayer(model,'cuda');graph=CudaGraphMultiheadPlayer(model,'cuda',batch_sizes=[4,8])
    saved=None
    for batch,ply in [(8,0),(1,1),(4,2),(3,3),(8,4),(1,5)]:
        b=chess.Board()
        for m in ['e2e4','e7e5','g1f3','b8c6','f1b5'][:ply]:b.push_uci(m)
        t,m=encode(b);ids=np.full((batch,256),-1,np.int16);legal=legal_ids(b)[1];ids[:,:len(legal)]=legal
        inputs=np.tile(t,(batch,1)),np.tile(m,(batch,1)),ids
        before=eager.raw_outputs(*inputs);actual=graph.raw_outputs(*inputs)
        np.testing.assert_allclose(actual,before,atol=1e-5,rtol=1e-5)
        if saved is not None:np.testing.assert_array_equal(saved[0],saved[1])
        saved=(actual,actual.copy())
    assert set(graph.graphs)=={1,4,8}
