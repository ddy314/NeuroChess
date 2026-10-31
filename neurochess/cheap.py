"""Small CPU value evaluator; both perspectives share an additive accumulator.

This is an NNUE-style float prototype, not Stockfish's architecture or weights.
The accumulator supports exact piece-delta updates in the native search. All
The inference head predicts side-to-move W-L proxies from engine search, not
game probabilities. An optional training-only CP head shares its representation.
"""
from dataclasses import asdict, dataclass
from pathlib import Path
import struct

import numpy as np
import torch
from torch import nn


@dataclass
class CheapConfig:
    width: int = 128
    hidden: int = 32
    cp_head: bool = False


class CheapModel(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or CheapConfig()
        c = self.config
        if not 1 <= c.width <= 1024 or not 1 <= c.hidden <= 512:
            raise ValueError('Unsupported cheap evaluator dimensions')
        self.accumulator = nn.Linear(768, c.width)
        self.hidden = nn.Linear(2*c.width + 12, c.hidden)
        self.output = nn.Linear(c.hidden, 1)
        nn.init.normal_(self.accumulator.weight, std=.025)
        nn.init.constant_(self.accumulator.bias, .5)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        if c.cp_head:
            self.cp_output = nn.Linear(c.hidden, 1)
            nn.init.zeros_(self.cp_output.weight)
            nn.init.zeros_(self.cp_output.bias)
        self.register_buffer('meta_mask', torch.tensor(
            [1,1,1,1,1,0,0,0,1,1,0,1], dtype=torch.float32), persistent=False)
        self.register_buffer('material', torch.tensor(
            [100,320,330,500,900,0,-100,-320,-330,-500,-900,0],
            dtype=torch.float32).repeat_interleave(64)/600, persistent=False)

    @staticmethod
    def features(tokens):
        """Canonical own/opponent piece-square planes; empty squares contribute zero."""
        t = tokens.long()
        squares = torch.arange(64, device=t.device).expand_as(t)
        indices = (t-1).clamp_min(0)*64 + squares
        own = torch.zeros((len(t), 768), device=t.device)
        own.scatter_(1, indices, (t != 0).float())
        # Change side to move: flip ranks and exchange piece colors.
        opposite_t = t[:, torch.arange(64, device=t.device)^56]
        opposite_t = torch.where(opposite_t == 0, 0, (opposite_t+5)%12+1)
        indices = (opposite_t-1).clamp_min(0)*64 + squares
        opposite = torch.zeros_like(own)
        opposite.scatter_(1, indices, (opposite_t != 0).float())
        return own, opposite

    def hidden_features(self, tokens, meta):
        own, opposite = self.features(tokens)
        a = self.accumulator(own).clamp(0, 1)
        b = self.accumulator(opposite).clamp(0, 1)
        x = torch.cat((a, b, meta.float()*self.meta_mask), dim=1)
        return self.hidden(x).relu(), own@self.material

    def heads(self, tokens, meta):
        h, material = self.hidden_features(tokens, meta)
        value = self.output(h).squeeze(1) + material
        cp = self.cp_output(h).squeeze(1) + material if self.config.cp_head else None
        return value, cp

    def logits(self, tokens, meta):
        h, material = self.hidden_features(tokens, meta)
        return self.output(h).squeeze(1) + material

    def forward(self, tokens, meta):
        return self.logits(tokens, meta).tanh()

    def export(self, path, head='value'):
        """Versioned, little-endian float weights consumed by best_first.cpp."""
        if head not in ('value','cp') or (head=='cp' and not self.config.cp_head):
            raise ValueError('Requested cheap head is unavailable')
        output=self.output if head=='value' else self.cp_output
        arrays = [self.accumulator.bias, self.accumulator.weight.T.contiguous(),
                  self.hidden.weight, self.hidden.bias, output.weight, output.bias]
        path = Path(path)
        temporary = path.with_suffix(path.suffix+'.tmp')
        with temporary.open('wb') as f:
            f.write(b'NCHEAP01')
            f.write(struct.pack('<II', self.config.width, self.config.hidden))
            for a in arrays:
                f.write(a.detach().cpu().numpy().astype('<f4').tobytes())
        temporary.replace(path)


def load_cheap(path):
    # Older local prototype checkpoints serialized torch.__version__ as this
    # string subclass. Keep weights-only loading and allow only that known type.
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        state = torch.load(path, map_location='cpu', weights_only=True)
    model = CheapModel(CheapConfig(**state['config']))
    model.load_state_dict(state['model'])
    return model.eval(), state


def save_cheap(model, path, **metadata):
    from .train import atomic_save
    atomic_save(dict(format='neurochess-cheap-v1', config=asdict(model.config),
                     model=model.state_dict(), **metadata), path)
