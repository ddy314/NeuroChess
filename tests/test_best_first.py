import ctypes as C
import random

import chess
import numpy as np
import pytest
import torch

from neurochess.best_first import BestFirstPlayer
from neurochess.cheap import CheapConfig, CheapModel, load_cheap, save_cheap
from neurochess.encoding import encode
from neurochess.native_mcts import history


@pytest.fixture
def weights(tmp_path):
    torch.manual_seed(731)
    model = CheapModel(CheapConfig(width=24, hidden=12)).eval()
    # Exercise the learned component as well as the material skip.
    torch.nn.init.normal_(model.output.weight, std=.1)
    path = tmp_path/'cheap.bin'; model.export(path)
    return model, path


class Network:
    device = 'cpu'

    def __init__(self):
        self.positions = 0
        self.batches = []

    def evaluate_encoded(self, tokens, meta, ids):
        self.positions += len(tokens); self.batches.append(len(tokens))
        return np.zeros(ids.shape, np.float32), np.zeros(len(tokens), np.float32)


def probe(player, board):
    value, error = C.c_float(), C.c_float()
    player._check(player.lib.bf_probe(player.handle, *history(board), C.byref(value), C.byref(error)))
    return value.value, error.value


def test_native_incremental_matches_torch_and_rebuilds(weights):
    model, path = weights
    player = BestFirstPlayer(path, expansions=1)
    rng = random.Random(71)
    boards = []
    for _ in range(4):
        b = chess.Board()
        for ply in range(100):
            if b.is_game_over(claim_draw=True):
                break
            b.push(rng.choice(list(b.legal_moves)))
            if ply % 4 == 0:
                boards.append(b.copy())
    for fen, moves in [
        ('r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1', ['e1g1', 'e8c8']),
        ('7k/P7/8/8/8/8/8/7K w - - 0 1', ['a7a8n']),
        ('7k/8/8/3pP3/8/8/8/7K w - d6 0 1', ['e5d6']),
    ]:
        b = chess.Board(fen)
        for move in moves:
            b.push_uci(move); boards.append(b.copy())
    for board in boards:
        tokens, meta = encode(board)
        with torch.inference_mode():
            expected = model(torch.from_numpy(tokens[None]), torch.from_numpy(meta[None])).item()
        actual, error = probe(player, board)
        assert error < 1e-5, board.fen()
        assert actual == pytest.approx(np.clip(expected, -.9999, .9999), abs=2e-5), board.fen()
    player.close()


@pytest.mark.parametrize('budget', [0, 1, 2, 9, 32])
def test_independent_expansion_and_large_model_budgets(weights, budget):
    network = Network()
    player = BestFirstPlayer(weights[1], network, expansions=20,
                             neural_budget=budget, correction_batch=4)
    board = chess.Board(); before = board.fen(), list(board.move_stack)
    assert player.choose(board) in board.legal_moves
    stats = player.last_result['stats']
    assert stats['expanded_nodes'] == 20
    assert stats['cheap_evaluations'] > 100
    assert stats['correction_positions'] <= budget
    assert network.positions <= budget
    assert network.positions > 0 or budget == 0
    assert (board.fen(), board.move_stack) == before
    assert player.last_result['proven_value'] is None
    player.close()


def test_corrections_can_change_the_root_decision(weights):
    class PreferE4(Network):
        def evaluate_encoded(self, tokens, meta, ids):
            p, v = super().evaluate_encoded(tokens, meta, ids)
            for i, t in enumerate(tokens):
                # At black-to-move child, White's e4 pawn is opponent token 7
                # at e5 in the rank-flipped frame.
                if meta[i, 9] == 0:
                    v[i] = -.8 if t[chess.E4 ^ 56] == 7 else .2
            return p, v
    network = PreferE4()
    player = BestFirstPlayer(weights[1], network, expansions=1, neural_budget=21, correction_batch=4)
    assert player.choose(chess.Board()).uci() == 'e2e4'
    assert network.positions == 21


@pytest.mark.parametrize('mirror', [False, True])
def test_exact_mate_overrides_cheap_and_neural_estimates(weights, mirror):
    b = chess.Board('7k/8/5KQ1/8/8/8/8/8 w - - 0 1')
    if mirror:
        b = b.mirror()
    player = BestFirstPlayer(weights[1], Network(), expansions=10, neural_budget=8)
    move = player.choose(b)
    b.push(move)
    assert b.is_checkmate()
    assert player.last_result['proven_value'] == 1


def test_depth_and_node_caps_do_not_create_false_proofs(weights):
    for options in [dict(max_depth=1), dict(max_nodes=257)]:
        player = BestFirstPlayer(weights[1], expansions=100, **options)
        assert player.choose(chess.Board()) in chess.Board().legal_moves
        assert player.last_result['proven_value'] is None
        assert player.last_result['stats']['expanded_nodes'] < 100
        assert player.last_result['stats']['nodes'] <= options.get('max_nodes', 65536)


def test_fools_mate_refutation_survives_optimistic_q(weights):
    from neurochess.encoding import move_id
    b = chess.Board(); b.push_uci('f2f3'); b.push_uci('e7e5')
    bad = move_id(chess.Move.from_uci('g2g4'), b.turn)
    class Optimistic(Network):
        def evaluate_search_heads(self, tokens, meta, ids):
            p, v = self.evaluate_encoded(tokens, meta, ids)
            p[ids == bad] = 20
            q = np.full(ids.shape, np.nan, np.float32); q[ids == bad] = 1
            return p, q, v
    player = BestFirstPlayer(weights[1], Optimistic(), expansions=50, neural_budget=1)
    assert player.choose(b).uci() != 'g2g4'


def test_output_cache_reuses_inference_without_merging_rule_states(weights):
    network = Network()
    player = BestFirstPlayer(weights[1], network, expansions=10, neural_budget=9)
    b = chess.Board(); first = player.choose(b); count = network.positions
    assert player.choose(b) == first
    assert network.positions == count
    assert player.search_stats['cache_hits'] > 0
    player.clear_cache(); player.choose(b)
    assert network.positions > count


def test_draw_claim_and_terminal_roots_rejected(weights):
    player = BestFirstPlayer(weights[1], expansions=10)
    b = chess.Board()
    for uci in ['g1f3','g8f6','f3g1','f6g8','g1f3','g8f6','f3g1']:
        b.push_uci(uci)
    assert b.can_claim_threefold_repetition()
    with pytest.raises(RuntimeError, match='Terminal root'):
        player.choose(b)
    with pytest.raises(RuntimeError, match='Terminal root'):
        player.choose(chess.Board('7k/6Q1/5K2/8/8/8/8/8 b - - 0 1'))


def test_native_rejects_corrupt_weights_and_configuration(tmp_path, weights):
    bad = tmp_path/'bad.bin'; bad.write_bytes(b'NCHEAP01')
    with pytest.raises(ValueError, match='weight'):
        BestFirstPlayer(bad)
    with pytest.raises(ValueError, match='configuration'):
        BestFirstPlayer(weights[1], max_nodes=1)
    with pytest.raises(ValueError, match='configuration'):
        BestFirstPlayer(weights[1], Network(), neural_budget=-1)


def test_student_checkpoint_round_trip(tmp_path, weights):
    model, _ = weights
    path = tmp_path/'student.pt'
    save_cheap(model, path, step=12, provenance={'torch':torch.__version__})
    loaded, state = load_cheap(path)
    assert state['step'] == 12
    assert loaded.config == model.config
    for key, value in model.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[key], value)


def test_large_accumulator_export_matches_cpu_native(tmp_path):
    torch.manual_seed(61)
    model=CheapModel(CheapConfig(512,128)).eval()
    torch.nn.init.normal_(model.output.weight,std=.1)
    path=tmp_path/'medium.bin';model.export(path);player=BestFirstPlayer(path,expansions=1)
    b=chess.Board();b.push_uci('e2e4');b.push_uci('e7e5')
    tokens,meta=encode(b)
    with torch.inference_mode():v=model(torch.from_numpy(tokens[None]),torch.from_numpy(meta[None])).item()
    actual,error=probe(player,b)
    assert actual==pytest.approx(np.clip(v,-.9999,.9999),abs=1e-4)
    assert error<1e-4


@pytest.mark.parametrize('scheduler',['hierarchical','frontier'])
def test_diagnostic_options_preserve_query_caps_and_tree_trace(weights,scheduler):
    network=Network();player=BestFirstPlayer(weights[1],network,expansions=20,neural_budget=9,
        anchor_weight=0,q_anchor_weight=0,anchor_decay=8,correction_scheduler=scheduler,root_policy_weight=.1)
    b=chess.Board();assert player.choose(b) in b.legal_moves
    trace=player.trace();assert len(trace['corrections'])<=9
    assert player.config['correction_scheduler']==scheduler
    assert player.config['root_policy_weight']==.1
    assert len(trace['nodes'])==player.last_result['stats']['nodes']
    assert all(0<=e['id']<len(trace['nodes']) for e in trace['corrections'])
    assert network.positions<=9
    player.close()
    with pytest.raises(ValueError,match='closed'):player.trace()


def test_exact_win_overrides_arbitrarily_strong_policy_penalty(weights):
    from neurochess.encoding import move_id
    board=chess.Board('7k/8/5KQ1/8/8/8/8/8 w - - 0 1');mates=[]
    for move in board.legal_moves:
        b=board.copy();b.push(move)
        if b.is_checkmate():mates.append(move_id(move,board.turn))
    class HideMates(Network):
        def evaluate_encoded(self,tokens,meta,ids):
            p,v=super().evaluate_encoded(tokens,meta,ids);p[np.isin(ids,mates)] = -100
            return p,v
    player=BestFirstPlayer(weights[1],HideMates(),expansions=20,neural_budget=8,root_policy_weight=1000)
    board.push(player.choose(board));assert board.is_checkmate()


@pytest.mark.parametrize('options',[dict(anchor_weight=-.1),dict(q_anchor_weight=1.1),
                                    dict(anchor_decay=-1),dict(root_policy_weight=float('nan'))])
def test_invalid_search_options_rejected(weights,options):
    with pytest.raises(ValueError,match='options'):
        BestFirstPlayer(weights[1],**options)


def test_aligned_objective_guides_root_expansion_and_calibration(weights):
    from neurochess.encoding import move_id
    favorite=move_id(chess.Move.from_uci('e2e4'),chess.WHITE)
    class FavorE4(Network):
        def evaluate_encoded(self,tokens,meta,ids):
            p,v=super().evaluate_encoded(tokens,meta,ids);p[ids==favorite]=10
            return p,v
    player=BestFirstPlayer(weights[1],FavorE4(),expansions=128,neural_budget=9,
        root_policy_weight=1,align_root_objective=True)
    assert player.choose(chess.Board()).uci()=='e2e4'
    trace=player.trace();node=next(n for n in trace['nodes'] if n['parent']==0 and n['move']=='e2e4')
    assert node['visits']>=100
    assert trace['corrections'][1]['id']==node['id']
    assert player.config['align_root_objective']
    assert player.search_stats['large_model_positions']<=9


@pytest.mark.parametrize('budget',[2,9,32])
def test_final_verification_respects_budget_after_rank_changes(weights,budget):
    class RefuteQueried(Network):
        def evaluate_encoded(self,tokens,meta,ids):
            p,v=super().evaluate_encoded(tokens,meta,ids)
            v[meta[:,9]==0]=.9
            return p,v
    n=RefuteQueried();player=BestFirstPlayer(weights[1],n,expansions=32,neural_budget=budget,
        verify_final=True,verification_reserve=8)
    b=chess.Board();assert player.choose(b) in b.legal_moves
    assert n.positions<=budget
    assert player.last_result['final_candidate_checked']
    trace=player.trace();chosen=next(x for x in trace['nodes'] if x['parent']==0 and x['move']==player.last_result['move'])
    assert chosen['corrected'] or chosen['solved']!=2


def test_policy_fallback_does_not_choose_proven_loss(weights):
    from neurochess.encoding import move_id
    b=chess.Board();b.push_uci('f2f3');b.push_uci('e7e5');bad=move_id(chess.Move.from_uci('g2g4'),b.turn)
    class BadPolicy(Network):
        def evaluate_encoded(self,tokens,meta,ids):
            p,v=super().evaluate_encoded(tokens,meta,ids);p[ids==bad]=100
            return p,v
    player=BestFirstPlayer(weights[1],BadPolicy(),expansions=50,neural_budget=1,
        root_policy_weight=1000,align_root_objective=True,verify_final=True)
    assert player.choose(b).uci()!='g2g4'


def test_local_exploration_reaches_high_prior_alternative_inside_favored_branch(weights):
    from neurochess.encoding import move_id
    e4=move_id(chess.Move.from_uci('e2e4'),chess.WHITE)
    c5=move_id(chess.Move.from_uci('c7c5'),chess.BLACK)
    class Prior(Network):
        def evaluate_encoded(self,tokens,meta,ids):
            p,v=super().evaluate_encoded(tokens,meta,ids)
            p[ids==e4]=12;p[ids==c5]=11
            return p,v
    n=Prior();player=BestFirstPlayer(weights[1],n,expansions=256,neural_budget=32,
        root_policy_weight=10,align_root_objective=True,local_exploration=True)
    player.choose(chess.Board());trace=player.trace()
    e=next(x for x in trace['nodes'] if x['parent']==0 and x['move']=='e2e4')
    c=next(x for x in trace['nodes'] if x['parent']==e['id'] and x['move']=='c7c5')
    assert c['visits']>0
    assert n.positions<=32
    assert player.config['local_exploration'] is True


def test_independent_cpu_proof_rejects_known_mate_in_three_despite_neural_prior(weights):
    from neurochess.encoding import move_id
    b=chess.Board('rq2r1k1/3Q1p1p/1p3p2/p7/6R1/4PN2/PPP3PP/2K1R3 b - - 1 18')
    bad=move_id(b.parse_uci('g8h8'),b.turn)
    class Bad(Network):
        def evaluate_encoded(self,tokens,meta,ids):
            p,v=super().evaluate_encoded(tokens,meta,ids);p[ids==bad]=100
            return p,v
    n=Bad();player=BestFirstPlayer(weights[1],n,expansions=32,neural_budget=9,
        root_policy_weight=10,align_root_objective=True,mate_nodes=16384,mate_depth=7)
    assert player.choose(b).uci()=='g8f8'
    node=next(x for x in player.trace()['nodes'] if x['parent']==0 and x['move']=='g8h8')
    assert node['solved']==1
    assert player.last_result['stats']['cpu_mate_proofs']>=1
    assert player.last_result['stats']['mate_nodes']<=16384
    assert n.positions<=9


@pytest.mark.parametrize('budget',[1,32,256])
def test_independent_cpu_budgets_never_create_approximate_proofs(weights,budget):
    n=Network();player=BestFirstPlayer(weights[1],n,expansions=4,neural_budget=2,
        mate_nodes=budget,tactical_nodes=budget,correction_clip=.15)
    b=chess.Board();assert player.choose(b) in b.legal_moves
    stats=player.last_result['stats']
    assert stats['mate_nodes']<=budget and stats['tactical_nodes']<=budget
    assert player.last_result['proven_value'] is None
    assert n.positions<=2


def test_clipped_correction_limits_direct_and_candidate_q_changes(weights):
    class Wrong(Network):
        def evaluate_search_heads(self,tokens,meta,ids):
            p,v=self.evaluate_encoded(tokens,meta,ids)
            return p,np.ones(ids.shape,np.float32),np.full(len(tokens),-.99,np.float32)
    player=BestFirstPlayer(weights[1],Wrong(),expansions=24,neural_budget=9,correction_clip=.15)
    player.choose(chess.Board());trace=player.trace()
    for n in trace['nodes']:
        if n['solved']==2 and n['anchor_weight']>0:
            assert abs(n['applied_correction'])<=.15*n['effective_anchor_weight']+2e-6
    assert any(n['anchor_clipped'] for n in trace['nodes'])


@pytest.mark.parametrize('opts',[dict(correction_clip=-.1),dict(mate_nodes=-1),dict(mate_depth=10),dict(tactical_nodes=-1),dict(root_value_clip=-.1),dict(cpu_correction_clip=-.1)])
def test_invalid_safety_limits(weights,opts):
    with pytest.raises(ValueError,match='safety'):BestFirstPlayer(weights[1],**opts)


def test_cpu_conditioned_clip_only_limits_nodes_with_completed_cpu_evidence(weights):
    class Wrong(Network):
        def evaluate_encoded(self,tokens,meta,ids):
            p,v=super().evaluate_encoded(tokens,meta,ids)
            return p,np.full(len(tokens),-.99,np.float32)
    player=BestFirstPlayer(weights[1],Wrong(),expansions=24,neural_budget=9,
        tactical_nodes=8192,cpu_correction_clip=.15)
    player.choose(chess.Board());trace=player.trace()
    gated=[n for n in trace['nodes'] if n['cpu_ready'] and n['anchor_weight']>0 and n['solved']==2]
    assert gated
    assert all(abs(n['applied_correction'])<=.15*n['effective_anchor_weight']+2e-6 for n in gated)
    assert not trace['nodes'][0]['cpu_ready']
    assert not trace['nodes'][0]['anchor_clipped']


@pytest.mark.parametrize('mode', ['legacy','visit-mean','evidence-minimax','neural-mean'])
def test_reuse_preserves_only_exact_history_and_keeps_new_budget(weights, mode):
    p=BestFirstPlayer(weights[1],Network(),expansions=50,neural_budget=9,
                      tree_reuse=True,backup_mode=mode,max_depth=4)
    board=chess.Board();move=p.choose(board);trace=p.trace()['nodes']
    child=next(n for n in trace if n['parent']==0 and n['move']==move.uci())
    replies=[n for n in trace if n['parent']==child['id']]
    assert replies
    board.push(move);board.push_uci(replies[0]['move'])
    assert p.choose(board) in board.legal_moves
    assert p.last_result['stats']['tree_reuses']==1
    assert p.last_result['stats']['retained_nodes']>0
    assert p.last_result['stats']['correction_positions']<=9
    # A reconstructed FEN is a different repetition history.
    assert p.choose(chess.Board(board.fen(en_passant='fen'))) in board.legal_moves
    assert p.last_result['stats']['tree_reuses']==0
    p.clear_cache();p.choose(chess.Board(board.fen(en_passant='fen')))
    assert p.last_result['stats']['tree_reuses']==0


def test_neural_mean_counts_distinct_observations_and_alternates_sign(weights):
    class Value(Network):
        def evaluate_encoded(self,tokens,meta,ids):
            p,v=super().evaluate_encoded(tokens,meta,ids);v[:]=.4
            return p,v
    player=BestFirstPlayer(weights[1],Value(),expansions=35,neural_budget=13,
                           backup_mode='neural-mean',tree_reuse=True)
    board=chess.Board();player.choose(board);nodes=player.trace()['nodes']
    for node in nodes:
        descendants=[]
        for leaf in nodes:
            if not leaf['corrected']:continue
            x=leaf
            while x['id']!=node['id'] and x['parent']>=0:x=nodes[x['parent']]
            if x['id']==node['id']:descendants.append(leaf)
        assert node['neural_count']==len(descendants)
        expected=sum(n['anchor']*(-1)**(n['depth']-node['depth']) for n in descendants)
        assert node['neural_sum']==pytest.approx(expected,abs=1e-6)
        if descendants and node['solved']==2:
            assert node['value']==pytest.approx(expected/len(descendants),abs=1e-6)
    player.choose(board)
    assert player.trace()['nodes'][0]['neural_count']>=13
    assert player.trace()['nodes'][0]['neural_count']==sum(n['corrected'] for n in player.trace()['nodes'])


@pytest.mark.parametrize('mode',['evidence-minimax','neural-mean'])
def test_uncalibrated_siblings_cannot_win_final_ranking(weights,mode):
    player=BestFirstPlayer(weights[1],Network(),expansions=12,neural_budget=2,
                           correction_batch=1,backup_mode=mode)
    chosen=player.choose(chess.Board());nodes=player.trace()['nodes']
    node=next(n for n in nodes if n['parent']==0 and n['move']==chosen.uci())
    assert node['evidence'] or node['solved']!=2
    assert player.last_result['stats']['correction_positions']==2


@pytest.mark.parametrize('mode',['material','neutral'])
def test_student_ablation_removes_learned_component(weights,mode):
    p=BestFirstPlayer(weights[1],cheap_evaluator=mode,expansions=2)
    b=chess.Board('7k/8/8/8/8/8/P7/7K w - - 0 1')
    actual,_=probe(p,b)
    assert actual==pytest.approx(np.tanh(100/600) if mode=='material' else 0,abs=1e-6)


def test_export_cp_head_uses_same_incremental_features(tmp_path):
    model=CheapModel(CheapConfig(width=24,hidden=12,cp_head=True)).eval()
    torch.nn.init.normal_(model.cp_output.weight,std=.1)
    path=tmp_path/'cp.bin';model.export(path,head='cp');p=BestFirstPlayer(path,expansions=1)
    b=chess.Board();b.push_uci('e2e4');t,m=encode(b)
    with torch.inference_mode():expected=model.heads(torch.from_numpy(t[None]),torch.from_numpy(m[None]))[1].tanh().item()
    actual,error=probe(p,b);assert error<1e-5;assert actual==pytest.approx(expected,abs=2e-5)
    with pytest.raises(ValueError):model.export(tmp_path/'bad.bin',head='unknown')


def test_root_coverage_calibrates_alternatives_within_original_cap(weights):
    net=Network();p=BestFirstPlayer(weights[1],net,expansions=25,neural_budget=9,
                                   correction_batch=4,root_coverage=8)
    p.choose(chess.Board());nodes=p.trace()['nodes'];children=[n for n in nodes if n['parent']==0]
    top=sorted(children,key=lambda n:n['prior'],reverse=True)[:8]
    assert all(n['corrected'] for n in top)
    assert net.positions==9
    assert p.last_result['stats']['expanded_nodes']==25


def test_legacy_backup_matches_negamax_perspective_and_anchor_formula(weights):
    p=BestFirstPlayer(weights[1],Network(),expansions=30,neural_budget=13)
    p.choose(chess.Board());nodes=p.trace()['nodes']
    for n in nodes:
        if not n['expanded'] or n['solved']!=2:continue
        children=[x for x in nodes if x['parent']==n['id']]
        best=max(-x['value'] for x in children);w=n['effective_anchor_weight']
        assert n['value']==pytest.approx((1-w)*best+w*n['anchor'],abs=2e-6)


def test_reused_expanded_children_keep_backup_after_cpu_reference_refresh(weights):
    p=BestFirstPlayer(weights[1],Network(),expansions=30,neural_budget=9,
                      tree_reuse=True,tactical_nodes=8192,max_depth=5)
    b=chess.Board();p.choose(b)
    # Inspect immediately after reroot/CPU refresh, before another expansion
    # can accidentally repair the stale child value.
    p._check(p.lib.bf_start(p.handle,*history(b)))
    nodes=p.trace()['nodes'];checked=0
    for n in nodes:
        if not n['expanded'] or n['solved']!=2:continue
        children=[x for x in nodes if x['parent']==n['id']]
        best=max(-x['value'] for x in children)
        base=.5*best+.5*n['cpu_value'] if n['cpu_ready'] else best
        weight=n['effective_anchor_weight']
        assert n['value']==pytest.approx((1-weight)*base+weight*n['anchor'],abs=2e-6)
        checked+=n['cpu_ready'] and n['parent']==0
    assert checked>0


@pytest.mark.parametrize('mirror',[False,True])
def test_reused_proven_root_is_playable_until_board_actually_terminal(weights,mirror):
    b=chess.Board('7k/8/5KQ1/8/8/8/8/8 w - - 0 1')
    if mirror:b=b.mirror()
    net=Network();p=BestFirstPlayer(weights[1],net,expansions=10,neural_budget=8,tree_reuse=True)
    first=p.choose(b);assert p.last_result['proven_value']==1;positions=net.positions
    second=p.choose(b)
    assert p.last_result['stats']['tree_reuses']==1
    assert p.last_result['proven_value']==1 and net.positions==positions
    b.push(second);assert b.is_checkmate()
    with pytest.raises(RuntimeError,match='Terminal root'):p.choose(b)
