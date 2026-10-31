import random
import chess
import numpy as np
import torch
from neurochess.encoding import encode, legal_ids, move_id, position_key, split_for
from neurochess.data import record, ShardWriter, Batches
from neurochess.model import ChessModel, ModelConfig
from neurochess.train import losses, atomic_save, load_model


def test_legal_encoding_random_play():
    rng = random.Random(42)
    for _ in range(8):
        board = chess.Board()
        for _ in range(200):
            moves, ids = legal_ids(board)
            assert len(ids) == len(set(ids.tolist()))
            assert all(0 <= i < 20480 for i in ids)
            if not moves:
                break
            board.push(rng.choice(moves))


def test_special_moves_and_perspective():
    for fen in ['4k3/P7/8/8/8/8/8/4K3 w - - 0 1',
                'r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1',
                '4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1']:
        b = chess.Board(fen)
        moves, ids = legal_ids(b)
        mirrored = b.mirror()
        assert sorted(ids) == sorted(legal_ids(mirrored)[1])
        assert np.array_equal(encode(b)[0], encode(mirrored)[0])
        if b.ep_square:
            assert any(b.is_en_passant(m) for m in moves)
        if 'P7' in fen:
            assert len([m for m in moves if m.promotion]) == 4
        if 'KQkq' in fen:
            assert len([m for m in moves if b.is_castling(m)]) == 2


def test_history_and_split():
    b = chess.Board()
    initial = position_key(b)
    for move in ['g1f3', 'g8f6', 'f3g1', 'f6g8'] * 2:
        b.push_uci(move)
    assert encode(b)[1][6:8].tolist() == [1., 1.]
    assert position_key(b) == initial
    assert split_for(b) == split_for(chess.Board())


def test_shard_and_train_roundtrip(tmp_path):
    torch.set_num_threads(2)
    b = chess.Board()
    row, _ = record(b)
    row['policy'][0] = 1; row['value'][1] = 1; row['value_weight'] = 1
    writer = ShardWriter(tmp_path / 'data', size=2)
    for s in ['train', 'val']:
        for _ in range(3):
            writer.add(s, row)
    writer.close({'test': True})
    batches = list(Batches(tmp_path / 'data', 'train', 2))
    assert sum(len(b['tokens']) for b in batches) == 3
    model = ChessModel(ModelConfig(64, 2, 4, 32))
    opt = torch.optim.AdamW(model.parameters(), lr=.01)
    batch = batches[0]
    loss_before = None
    for _ in range(15):
        opt.zero_grad()
        outputs = model(batch['tokens'], batch['meta'], batch['moves'])
        loss, _ = losses(outputs, batch)
        if loss_before is None:
            loss_before = loss.item()
        assert torch.isfinite(loss)
        loss.backward(); opt.step()
    assert loss.item() < loss_before * .3
    assert (outputs[0][batch['moves'] < 0] < -1e8).all()
    atomic_save({'model': model.state_dict(), 'model_config': vars(model.config)}, tmp_path / 'ckpt.pt')
    loaded, _ = load_model(tmp_path / 'ckpt.pt', 'cpu')
    torch.testing.assert_close(model(batch['tokens'], batch['meta'], batch['moves'])[0], loaded(batch['tokens'], batch['meta'], batch['moves'])[0])


def test_q_mask_has_no_gradient_for_unlabelled_moves():
    row, _ = record(chess.Board()); row['policy'][0] = 1
    batch = {k: torch.tensor(np.array(row[k])[None]) for k in ['policy', 'q', 'qmask', 'value', 'value_weight']}
    q = torch.zeros(1, 256, requires_grad=True)
    batch['qmask'][0, 0] = True; batch['q'][0, 0] = 1
    loss, _ = losses((torch.zeros(1, 256), q, torch.zeros(1, 3)), batch)
    loss.backward()
    assert q.grad[0, 0] != 0 and (q.grad[0, 1:] == 0).all()


def test_audit_rejects_split_leakage(tmp_path):
    from neurochess.audit import audit
    import pytest
    row, _ = record(chess.Board()); row['policy'][0] = 1
    writer = ShardWriter(tmp_path / 'bad')
    writer.add('train', row); writer.add('val', row); writer.close({})
    with pytest.raises(ValueError, match='leakage'):
        audit(tmp_path / 'bad')


def test_checkmate_and_oneply_terminal(tmp_path):
    from neurochess.play import Player
    model = ChessModel(ModelConfig(64, 2, 4, 32))
    checkpoint = tmp_path / 'model.pt'
    atomic_save({'model': model.state_dict(), 'model_config': vars(model.config)}, checkpoint)
    board = chess.Board('7k/5Q2/6K1/8/8/8/8/8 w - - 0 1')
    move = Player(checkpoint, mode='oneply').choose(board)
    board.push(move)
    assert board.is_checkmate()
