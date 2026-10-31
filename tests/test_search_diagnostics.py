import random

import chess
import numpy as np

from neurochess.encoding import encode
from neurochess.search_diagnostics import decode_board,fixed_tree_backup


def test_diagnostic_decoder_preserves_raw_ep_and_effective_inputs():
    b=chess.Board();rng=random.Random(43171)
    for i in range(120):
        if b.is_game_over(claim_draw=True):b=chess.Board()
        b.push(rng.choice(list(b.legal_moves)))
        tokens,meta=encode(b);decoded=decode_board(tokens,meta)
        assert decoded.fen(en_passant='fen')==b.fen(en_passant='fen')
        actual_t,actual_m=encode(decoded)
        np.testing.assert_array_equal(actual_t,tokens)
        np.testing.assert_array_equal(actual_m[[0,1,2,3,4,5,8,9,10,11]],meta[[0,1,2,3,4,5,8,9,10,11]])


def test_fixed_topology_isolates_anchor_masking_of_accurate_frontier():
    # Trap has an accurate losing frontier; a wrong parent anchor can still
    # make it preferable to a safe alternative without any encoding/network bug.
    nodes=[dict(id=0,parent=-1,move='',value=.3525,cheap=0,anchor=0,anchor_weight=0,expanded=True,corrected=False,solved=2),
           dict(id=1,parent=0,move='trap',value=-.3525,cheap=-.8,anchor=-.8,anchor_weight=.75,expanded=True,corrected=True,solved=2),
           dict(id=2,parent=0,move='safe',value=0,cheap=0,anchor=0,anchor_weight=0,expanded=False,corrected=False,solved=2),
           dict(id=3,parent=1,move='refute',value=-.99,cheap=-.99,anchor=0,anchor_weight=0,expanded=False,corrected=False,solved=2)]
    trace=dict(nodes=nodes)
    assert fixed_tree_backup(trace,'anchored')['move']=='trap'
    assert fixed_tree_backup(trace,'pure')['move']=='safe'
