"""Game-grouped MultiPV supervision for the small value evaluator.

Slot zero is the parent, slots 1..4 are children. Every state label is from
that state's side to move; sibling ordering is consequently negated.
"""
import numpy as np
import torch
from torch.nn import functional as F

SLOTS = 5
FRONTIER_DTYPE = np.dtype([
    ('tokens', 'u1', (SLOTS, 64)), ('meta', '<f4', (SLOTS, 12)),
    ('key', 'u1', (SLOTS, 16)), ('valid', '?', (SLOTS,)),
    ('cp', '<f4', (SLOTS,)), ('cp_valid', '?', (SLOTS,)),
    ('mate', '<i2', (SLOTS,)), ('wdl', '<f4', (SLOTS,)),
    ('game', '<i4'), ('node', '<i4'), ('depth', '<i2'),
])


def ranking_loss(logits, cp, valid, min_gap=30., temperature=100.):
    """Order sibling states in the parent's frame; ignore near-tied CP labels.

    Soft targets reflect teacher CP gaps; predicted logits are unsquashed so
    the loss can distinguish siblings even when their WDL targets saturate.
    """
    scores = -logits[:, 1:]
    teacher = -cp[:, 1:]
    i, j = torch.triu_indices(4, 4, 1, device=logits.device)
    gap = teacher[:, i]-teacher[:, j]
    mask = valid[:, 1:][:, i] & valid[:, 1:][:, j] & (gap.abs() >= min_gap)
    target = torch.sigmoid(gap/temperature)
    loss = F.binary_cross_entropy_with_logits(scores[:, i]-scores[:, j], target, reduction='none')
    return (loss*mask).sum()/mask.sum().clamp_min(1)


def ranking_metrics(logits, cp, valid, min_gap=30.):
    scores, teacher = -logits[:, 1:], -cp[:, 1:]
    i, j = np.triu_indices(4, 1)
    gaps = teacher[:, i]-teacher[:, j]
    mask = valid[:, 1:][:, i] & valid[:, 1:][:, j] & (np.abs(gaps) >= min_gap)
    correct = ((scores[:, i]-scores[:, j])*gaps > 0) & mask
    counts = mask.sum(axis=1)
    # Teacher best among the available labeled children, never among all legal moves.
    candidate = valid[:, 1:]
    has_candidates = candidate.any(axis=1)
    chosen = np.argmax(np.where(candidate, scores, -np.inf), axis=1)
    best = np.max(np.where(candidate, teacher, -np.inf), axis=1)
    regret = best-teacher[np.arange(len(chosen)), chosen]
    regret = regret[has_candidates]
    return dict(pairs=int(counts.sum()), pairs_correct=int(correct.sum()),
                groups=int(has_candidates.sum()), cp_regret_sum=float(regret.sum()),
                blunders_100=int((regret >= 100).sum()))
