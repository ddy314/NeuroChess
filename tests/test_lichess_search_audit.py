import unittest
import chess
import numpy as np
from neurochess.data import record
from scripts.lichess_search_audit import reconstruct,assessment,needs_confirmation,scored_infos
from neurochess.epoch_ablation import screen_search_rows


class AuditTests(unittest.TestCase):
    def test_history_free_board_roundtrip_colors_castling_ep(self):
        for fen in [chess.STARTING_FEN,
            'r3k2r/ppp2ppp/2n5/3pp3/3PP3/2N5/PPP2PPP/R3K2R b KQkq - 4 8',
            'rnbqkbnr/pppp1ppp/8/8/3Pp3/8/PPP1PPPP/RNBQKBNR b KQkq d3 0 2']:
            b=chess.Board(fen);r,_=record(b);r['meta'][[5,6,7,10]]=0
            decoded,_=reconstruct(r)
            self.assertEqual(' '.join(b.fen().split()[:4]),' '.join(decoded.fen().split()[:4]))

    def test_key_corruption_rejected(self):
        r,_=record(chess.Board());r['meta'][[5,6,7,10]]=0;r['key'][0]^=1
        with self.assertRaises(ValueError):reconstruct(r)

    def test_original_gate_and_confirmation_remain_strict(self):
        job=dict(direct={n:dict(moves=dict(policy='e2e4',Q_supported='d2d4',fused='e2e4')) for n in ['parent','candidate']},
            search=dict(parent=dict(move='e2e4'),candidate=dict(move='d2d4')))
        teacher=[dict(move=m,sort_score=s,cp=s,depth=24,nodes=1000000,lowerbound=False,upperbound=False) for m,s in [('e2e4',0),('d2d4',-100)]]
        a=assessment(job,teacher);self.assertTrue(a['introduced_blunder'])
        self.assertTrue(needs_confirmation(job,a,teacher[:1],teacher))
        gate=screen_search_rows([dict(index=0,candidate_regret=a['candidate_regret'],baseline_regret=a['baseline_regret'])])
        self.assertFalse(gate['passed'])

    def test_mate_scores_explicitly_not_called_cp(self):
        job=dict(direct={n:dict(moves=dict(policy='e2e4',Q_supported='e2e4',fused='e2e4')) for n in ['parent','candidate']},
            search={n:dict(move='e2e4') for n in ['parent','candidate']})
        a=assessment(job,[dict(move='e2e4',sort_score=99999,cp=None,depth=24,lowerbound=False,upperbound=False)])
        self.assertFalse(a['all_cp']);self.assertEqual(a['delta'],0)

    def test_teacher_illegal_or_incomplete_pv_rejected(self):
        class Engine:
            def analyse(self,*a,**kw):return [dict(pv=[chess.Move.from_uci('e2e5')],score=chess.engine.PovScore(chess.engine.Cp(0),True),depth=24,nodes=1000000)]
        with self.assertRaises(ValueError):scored_infos(Engine(),chess.Board(),1000000,['e2e4'])

if __name__=='__main__':unittest.main()
