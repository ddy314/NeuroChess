"""NeuroChess network: a square-token Transformer with policy, action-value and WDL heads.

The architecture and parameter names match the released checkpoint exactly, so
``load_model`` performs a strict ``state_dict`` load.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class ModelConfig:
    width: int = 896
    layers: int = 16
    heads: int = 14
    head_width: int = 256
    history_free: bool = False


@dataclass
class QHeadConfig:
    width: int = 128
    bins: int = 128
    sigma_bins: float = .75


class Block(nn.Module):
    """Pre-norm Transformer block over the 64 board squares."""

    def __init__(self, c: ModelConfig):
        super().__init__()
        self.heads = c.heads
        self.norm1 = nn.LayerNorm(c.width)
        self.qkv = nn.Linear(c.width, c.width * 3, bias=False)
        self.out = nn.Linear(c.width, c.width, bias=False)
        self.norm2 = nn.LayerNorm(c.width)
        self.ff = nn.Sequential(nn.Linear(c.width, 4 * c.width), nn.GELU(approximate='tanh'),
                                nn.Linear(4 * c.width, c.width))

    def forward(self, x):
        b, s, d = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(b, s, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4).unbind(0)
        x = x + self.out(F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, s, d))
        return x + self.ff(self.norm2(x))


class ActionValueHead(nn.Module):
    """Per-move queries attend to the board encoding and predict a W-L histogram."""

    def __init__(self, board_width: int, config: QHeadConfig | None = None):
        super().__init__()
        self.config = config or QHeadConfig()
        d = self.config.width
        self.source = nn.Linear(board_width, d)
        self.destination = nn.Linear(board_width, d)
        self.promotion = nn.Embedding(5, d)
        self.global_context = nn.Linear(board_width, d)
        self.key = nn.Linear(board_width, d, bias=False)
        self.value = nn.Linear(board_width, d, bias=False)
        self.query_norm = nn.LayerNorm(d)
        self.refine = nn.Sequential(nn.LayerNorm(2 * d), nn.Linear(2 * d, 2 * d),
                                    nn.GELU(), nn.Linear(2 * d, d), nn.GELU())
        self.histogram = nn.Linear(d, self.config.bins)
        self.rank = nn.Linear(d, 1)    # auxiliary training head
        self.scalar = nn.Linear(d, 1)  # auxiliary training head
        edges = torch.linspace(-1., 1., self.config.bins + 1)
        self.register_buffer('centers', (edges[:-1] + edges[1:]) / 2)

    def forward(self, squares, moves):
        ids = moves.long().clamp_min(0)
        src, dst, promo = ids // 320, (ids // 5) % 64, ids % 5
        d = self.config.width
        a = self.source(squares).gather(1, src[..., None].expand(-1, -1, d))
        b = self.destination(squares).gather(1, dst[..., None].expand(-1, -1, d))
        query = self.query_norm(a + b + a * b / math.sqrt(d) + self.promotion(promo)
                                + self.global_context(squares.mean(1))[:, None])
        context = F.scaled_dot_product_attention(
            query[:, None], self.key(squares)[:, None], self.value(squares)[:, None])[:, 0]
        hidden = self.refine(torch.cat((query, context), -1))
        q = (self.histogram(hidden).float().softmax(-1) * self.centers).sum(-1)
        return q.masked_fill(moves < 0, 0)


class NeuroChessModel(nn.Module):
    """Dense square-token encoder; the move heads score every legal move in one pass.

    Inputs (side-to-move frame, see ``neurochess.encoding``):
        tokens: (B, 64) int  piece id per square
        meta:   (B, 12) float castling / en-passant / clocks / check flags
        moves:  (B, 256) int legal move ids, padded with -1
    Outputs:
        policy logits (B, 256), action values Q in [-1, 1] (B, 256), WDL logits (B, 3)
    """
    architecture = 'action_value_histogram_v2'

    def __init__(self, config: ModelConfig | None = None, q_config: QHeadConfig | None = None):
        super().__init__()
        self.config = c = config or ModelConfig()
        assert c.width % c.heads == 0
        self.piece = nn.Embedding(13, c.width)
        self.rank = nn.Embedding(8, c.width)
        self.file = nn.Embedding(8, c.width)
        self.meta = nn.Linear(12, c.width)
        self.blocks = nn.ModuleList([Block(c) for _ in range(c.layers)])
        self.norm = nn.LayerNorm(c.width)
        self.source = nn.Linear(c.width, c.head_width)
        self.destination = nn.Linear(c.width, c.head_width)
        self.promotion = nn.Embedding(5, c.head_width)
        self.action = nn.Sequential(nn.GELU(), nn.Linear(c.head_width, 2))
        self.value = nn.Sequential(nn.Linear(c.width, c.head_width), nn.GELU(), nn.Linear(c.head_width, 3))
        self.q_head = ActionValueHead(c.width, q_config)
        # History-dependent features (halfmove clock, repetitions, move number) are masked out.
        self.register_buffer('history_mask', torch.tensor([1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1], dtype=torch.float32),
                             persistent=False)
        self.register_buffer('squares', torch.arange(64), persistent=False)

    def encode_squares(self, tokens, meta):
        if self.config.history_free:
            meta = meta * self.history_mask
        x = self.piece(tokens.long()) + self.rank(self.squares // 8) + self.file(self.squares % 8)
        x = x + self.meta(meta.float()).unsqueeze(1)
        for block in self.blocks:
            x = block(x)
        return self.norm(x)

    def forward(self, tokens, meta, moves):
        x = self.encode_squares(tokens, meta)
        ids = moves.long().clamp_min(0)
        src, dst, promo = ids // 320, (ids // 5) % 64, ids % 5
        d = self.config.head_width
        a = self.source(x).gather(1, src[..., None].expand(-1, -1, d))
        b = self.destination(x).gather(1, dst[..., None].expand(-1, -1, d))
        policy = self.action(a + b + a * b / math.sqrt(d) + self.promotion(promo))[..., 0]
        policy = policy.float().masked_fill(moves < 0, -1e9)
        return policy, self.q_head(x, moves), self.value(x.mean(1)).float()

    def checkpoint(self) -> dict:
        return dict(architecture=self.architecture, model_config=asdict(self.config),
                    q_config=asdict(self.q_head.config), model=self.state_dict())


def load_model(path, device='cpu') -> NeuroChessModel:
    """Load a released ``.pt`` checkpoint in eval mode."""
    state = torch.load(path, map_location='cpu', weights_only=False)
    if state.get('architecture') != NeuroChessModel.architecture:
        raise ValueError(f'{path} is not a NeuroChess {NeuroChessModel.architecture} checkpoint')
    model = NeuroChessModel(ModelConfig(**state['model_config']), QHeadConfig(**state['q_config']))
    model.load_state_dict(state['model'], strict=True)
    return model.to(device).eval()
