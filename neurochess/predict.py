"""Raw network evaluation without search (useful for inspection and testing).

    python -m neurochess.predict --model neurochess-13750.pt --fen "<FEN>"
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext

import chess
import torch

from .encoding import encode_inputs
from .model import load_model


@torch.inference_mode()
def evaluate(model, board: chess.Board) -> dict:
    """Evaluate one position.

    Returns ``{'wdl': (win, draw, loss), 'moves': [(move, prior, q), ...]}``
    from the side to move's perspective, sorted by prior.
    """
    if board.is_game_over(claim_draw=False):
        raise ValueError('Position has no legal moves')
    tokens, meta, moves, legal = encode_inputs(board)
    device = next(model.parameters()).device
    ctx = torch.autocast('cuda', dtype=torch.bfloat16) if device.type == 'cuda' else nullcontext()
    with ctx:
        policy, q, wdl = model(torch.as_tensor(tokens[None], device=device),
                               torch.as_tensor(meta[None], device=device),
                               torch.as_tensor(moves[None], device=device))
    n = len(legal)
    prior = policy[0, :n].float().softmax(-1).cpu().tolist()
    values = q[0, :n].float().cpu().tolist()
    ranked = sorted(zip(legal, prior, values), key=lambda item: -item[1])
    return dict(wdl=tuple(wdl[0].float().softmax(-1).cpu().tolist()), moves=ranked)


def main():
    parser = argparse.ArgumentParser(description='Evaluate a position with the NeuroChess network (no search).')
    parser.add_argument('--model', required=True, help='path to the .pt checkpoint')
    parser.add_argument('--fen', default=chess.STARTING_FEN)
    parser.add_argument('--top', type=int, default=8)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    board = chess.Board(args.fen)
    result = evaluate(load_model(args.model, args.device), board)
    win, draw, loss = result['wdl']
    print(f'WDL (side to move): {win:.3f} / {draw:.3f} / {loss:.3f}')
    print(f'{"move":<8}{"prior":>8}{"Q":>8}')
    for move, prior, q in result['moves'][:args.top]:
        print(f'{board.san(move):<8}{prior:>8.3f}{q:>+8.3f}')


if __name__ == '__main__':
    main()
