"""Synthetic model throughput only; not a chess strength or data-pipeline benchmark."""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import time
import torch
from .model import ChessModel, ModelConfig


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/base.json')
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--steps', type=int, default=20)
    p.add_argument('--warmup', type=int, default=5)
    p.add_argument('--compile', action='store_true')
    p.add_argument('--out')
    a = p.parse_args()
    cuda = torch.cuda.is_available()
    device = 'cuda' if cuda else 'cpu'
    torch.set_float32_matmul_precision('high')
    model = ChessModel(ModelConfig(**json.loads(Path(a.config).read_text()))).to(device)
    runner = torch.compile(model) if a.compile else model
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=cuda)
    dtype = torch.bfloat16 if cuda and (torch.cuda.get_device_capability()[0] >= 8 and torch.cuda.is_bf16_supported()) else torch.float16
    scaler = torch.amp.GradScaler('cuda', enabled=cuda and dtype == torch.float16)
    tokens = torch.randint(13, (a.batch_size, 64), device=device)
    meta = torch.rand(a.batch_size, 12, device=device)
    moves = torch.randint(20480, (a.batch_size, 256), device=device)
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    for step in range(a.warmup + a.steps):
        if step == a.warmup:
            if cuda:
                torch.cuda.synchronize()
            start = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=dtype) if cuda else nullcontext():
            policy, q, value = runner(tokens, meta, moves)
            loss = policy.square().mean() + q.square().mean() + value.square().mean()
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
    if cuda:
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    result = dict(kind='synthetic-full-optimizer-step', torch=torch.__version__, gpu=torch.cuda.get_device_name() if cuda else None,
        parameters=sum(p.numel() for p in model.parameters()), batch_size=a.batch_size, steps=a.steps,
        precision=str(dtype) if cuda else 'fp32', compiled=a.compile, seconds=elapsed,
        positions_per_second=a.steps * a.batch_size / elapsed,
        peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30 if cuda else None,
        peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30 if cuda else None)
    print(json.dumps(result), flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
