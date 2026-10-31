from dataclasses import dataclass
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


class Block(nn.Module):
    def __init__(self, c):
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


class ChessModel(nn.Module):
    """Dense square-token encoder; legal-action head evaluates all moves in one pass.

    Inspired by square-token chess transformers; NOT a reproduction of GAB Chessformer.
    Plain SDPA intentionally keeps fused attention eligible without a dense bias mask.
    """
    def __init__(self, config=ModelConfig()):
        super().__init__()
        self.config = config
        c = config
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
        self.register_buffer('history_mask', torch.tensor([1,1,1,1,1,0,0,0,1,1,0,1], dtype=torch.float32), persistent=False)
        self.register_buffer('squares', torch.arange(64), persistent=False)

    def encode_squares(self, tokens, meta):
        if self.config.history_free:
            meta = meta * self.history_mask
        x = self.piece(tokens.long()) + self.rank(self.squares // 8) + self.file(self.squares % 8)
        x = x + self.meta(meta.float()).unsqueeze(1)
        for block in self.blocks:
            x = block(x)
        return self.norm(x)

    def value_only(self, tokens, meta):
        return self.value(self.encode_squares(tokens, meta).mean(1)).float()

    def forward(self, tokens, meta, moves):
        x = self.encode_squares(tokens, meta)
        ids = moves.long().clamp_min(0)
        src, dst, promo = ids // 320, (ids // 5) % 64, ids % 5
        dim = self.config.head_width
        a = self.source(x).gather(1, src.unsqueeze(-1).expand(-1, -1, dim))
        b = self.destination(x).gather(1, dst.unsqueeze(-1).expand(-1, -1, dim))
        logits, q = self.action(a + b + a * b / dim**0.5 + self.promotion(promo)).unbind(-1)
        return logits.float().masked_fill(moves < 0, -1e9), q.float().tanh(), self.value(x.mean(1)).float()
